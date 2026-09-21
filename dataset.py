"""Multi-song DEAM dataset labeled with the official valence/arousal annotations.

Moods come from DEAM's human annotations (see annotations.py) when an
annotations directory is available: two labels, happy vs sad, split on
valence alone, with clips in the valence dead band dropped outright rather
than given an ambiguous label. The old audio-feature heuristic (median split
on a spectral proxy) is kept only as a fallback for running without the
annotation CSVs.

Because the full dataset is ~1800 songs, this module also:
  * clips each song from 15 s onward, matching the region the dynamic
    annotations actually cover (the first 15 s of DEAM is unannotated)
  * cuts each song into `clips_per_song` consecutive short clips, each
    labeled with the valence/arousal of its own window, and shuffles the
    result so no mood is clustered anywhere in the tensor
  * extracts melody (top-k CQT) during loading so raw waveforms don't
    have to be kept in memory
  * caches mels + melodies + labels to disk, so re-runs skip the
    ~30 min mp3 decode + CQT pass
"""

import glob
import os

import numpy as np
import torch
import librosa
from tqdm import tqdm

from pipeline import bigvgan_mel_spectrogram, FixedMelNormalizer
from annotations import (load_annotation_windows, mood_from_va,
                         song_id_from_filename, VALENCE_DEAD_BAND,
                         MOOD_HAPPY, MOOD_SAD)
from augment import augment_waveform, plan_augmentation, variants_for_clip


def _heuristic_song_features(wav: np.ndarray, sr: int) -> float:
    """Spectral centroid — the fallback's stand-in for valence. Tempo and RMS
    used to be collected here too; they only fed the arousal axis, which no
    longer takes part in labeling."""
    return float(librosa.feature.spectral_centroid(y=wav, sr=sr).mean())


def _heuristic_labels(features: list) -> list:
    """Fallback: happy/sad per clip from a median split on a valence proxy.

    No dead band here — the spectral-centroid proxy has no calibrated zero
    to put one around, so this path just splits at the median. It is a
    degraded escape hatch for running without the annotation CSVs; the real
    labels come from `mood_from_va`.
    """
    cents = np.array(features)
    return [MOOD_HAPPY if c > np.median(cents) else MOOD_SAD for c in cents]


def _load_span(path: str, sr: int, start_seconds: float,
               span_seconds: float):
    """Load one song's [start, start+span) region as mono audio.

    Returns:
        wav: (span_samples,) zero-padded to the full span
        real_samples: how many samples are actual audio (rest is padding)
    """
    wav, _ = librosa.load(path, sr=sr, mono=True,
                          offset=start_seconds, duration=span_seconds)
    if len(wav) == 0:  # song shorter than the offset: take from the start
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=span_seconds)
    max_samples = int(span_seconds * sr)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    real_samples = len(wav)
    if len(wav) < max_samples:
        wav = np.pad(wav, (0, max_samples - len(wav)))
    return wav, real_samples


def _cache_path(cache_dir: str, n_songs, clip_start: float, clip_len: float,
                clips_per_song: int, labeled: bool, aug_tag: str = "") -> str:
    tag = "all" if n_songs is None else str(n_songs)
    # The dead band is part of the cache key: it changes which clips exist,
    # so a cache built under a different band must not be reused.
    src = f"annot-v{VALENCE_DEAD_BAND:g}" if labeled else "heur"
    return os.path.join(
        cache_dir,
        f"deam_{tag}songs_{clip_start:g}s+{clip_len:g}sx{clips_per_song}"
        f"_{src}{aug_tag}.npz")


def build_dataset(audio_dir: str, n_songs, bigvgan_model,
                  clip_seconds: float = 5, annotations_dir: str = None,
                  clip_start_seconds: float = 15, melody_extractor=None,
                  cache_dir: str = None, clips_per_song: int = 1,
                  shuffle_seed: int = 0, augment_moods=(),
                  augment_target=None, augment_ratio=None,
                  max_aug_per_clip: int = 12,
                  aug_max_semitones: float = 2.0, aug_max_shift_frac: float = 0.2,
                  aug_snr_db_range=(20.0, 35.0), augment_seed: int = 0,
                  clap_embedder=None, clap_batch: int = 64):
    """
    Load DEAM clips with annotation-derived mood labels.

    Each song contributes up to `clips_per_song` consecutive clips of
    `clip_seconds`, each labeled from the valence averaged over that clip's
    own window (dynamic annotations are per-second, so one song can yield
    both a "happy" and a "sad" clip, and clips from the same song can be
    dropped independently for landing in the dead band). The finished
    dataset is shuffled with a fixed seed so clips from the same song —
    and runs of the same mood — never sit next to each other.

    Args:
        n_songs: number of songs to use, or None for the entire dataset.
                 Subsets are spread evenly across the collection.
        annotations_dir: root of the DEAM annotations download. If None or
                 the files are missing, falls back to heuristic labels.
        clip_start_seconds: where the first clip starts. Default 15 s,
                 the start of the annotated region.
        melody_extractor: optional MelodyExtractor; when given, melodies
                 are computed here and returned instead of raw waveforms.
        cache_dir: optional directory for caching the processed dataset.
        clips_per_song: consecutive clips cut from each song. Clips that
                 fall mostly past the end of a short song are dropped.
        augment_moods: moods to balance up via waveform augmentation (see
                 augment.py). Empty disables augmentation. Only applies with
                 real annotations, not the heuristic fallback.
        augment_target: target clip count per augmented mood (None = match
                 the largest mood). max_aug_per_clip caps variants per clip.
        clap_embedder: optional callable (wavs, sr) -> (B, clap_dim) tensor
                 of L2-normed CLAP *audio* embeddings, e.g.
                 ClapTextEncoder.encode_audio. When given, one embedding is
                 computed per clip (augmented clips included, since the
                 augmentation changes what the clip sounds like) and cached
                 alongside the mels. Used to condition diffusion on audio
                 embeddings instead of the 5 mood-label text embeddings.

    Returns:
        mel_batch: (N, 1, n_mels, T) globally-normalized mel spectrograms (CPU)
        melodies: (N, top_k, T_cqt) long tensor, or None if no extractor
        mood_texts: list of N mood label strings
        names: list of N "filename#clip" strings ("...#k~augj" for augmented)
        clap_audio: (N, clap_dim) float tensor, or None if no clap_embedder
    """
    sr = bigvgan_model.h.sampling_rate
    windows = [(clip_start_seconds + k * clip_seconds,
                clip_start_seconds + (k + 1) * clip_seconds)
               for k in range(clips_per_song)]

    va = None
    if annotations_dir:
        try:
            va = load_annotation_windows(annotations_dir, windows)
        except FileNotFoundError as e:
            print(f"[DATA] WARNING: {e}\n"
                  f"[DATA] Falling back to heuristic mood labels — "
                  f"conditioning will be much weaker.")

    # Augmentation only makes sense with real per-window mood labels.
    augment_moods = tuple(augment_moods) if va is not None else ()
    aug_tag = ""
    if augment_moods:
        if augment_ratio is not None:
            t = f"r{augment_ratio:g}"
        else:
            t = "max" if augment_target is None else str(augment_target)
        aug_tag = (f"_aug-{'+'.join(m.split()[0] for m in augment_moods)}"
                   f"-t{t}-c{max_aug_per_clip}-sd{augment_seed}")

    # CLAP embeddings change what the cache contains, so they get their own
    # tag — otherwise a run that wants them would silently load a cache
    # built without them.
    if clap_embedder is not None:
        aug_tag += "_clap"

    cache_file = None
    if cache_dir:
        cache_file = _cache_path(cache_dir, n_songs, clip_start_seconds,
                                 clip_seconds, clips_per_song,
                                 labeled=va is not None, aug_tag=aug_tag)
        if os.path.exists(cache_file):
            print(f"[DATA] Loading cached dataset: {cache_file}")
            data = np.load(cache_file, allow_pickle=False)
            mel_batch = torch.from_numpy(data["mels"].astype(np.float32))
            melodies = (torch.from_numpy(data["melodies"].astype(np.int64))
                        if "melodies" in data else None)
            mood_texts = [str(s) for s in data["moods"]]
            names = [str(s) for s in data["names"]]
            clap_audio = (torch.from_numpy(data["clap_audio"].astype(np.float32))
                          if "clap_audio" in data else None)
            _print_distribution(mood_texts, len(names))
            return mel_batch, melodies, mood_texts, names, clap_audio

    files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    if not files:
        raise FileNotFoundError(f"No MP3 files found in {audio_dir}")

    if va is not None:
        annotated = [f for f in files if song_id_from_filename(f) in va]
        skipped = len(files) - len(annotated)
        if skipped:
            print(f"[DATA] Skipping {skipped} songs without annotations")
        files = annotated

    if n_songs is not None and n_songs < len(files):
        idxs = np.unique(np.linspace(0, len(files) - 1, n_songs).astype(int))
        files = [files[i] for i in idxs]

    normalizer = FixedMelNormalizer()
    seg_samples = int(clip_seconds * sr)
    span_seconds = clips_per_song * clip_seconds

    # Decide how many augmented variants each clip of a rare mood needs.
    # Counts come straight from the annotations (no audio needed), assuming
    # all windows are kept — short-song drops make the realized totals a
    # hair under target, which is fine.
    aug_plan = {}
    aug_rng = None
    if augment_moods:
        from collections import Counter
        pre_counts = Counter(
            m for m in (mood_from_va(*va[song_id_from_filename(p)][k])
                        for p in files for k in range(clips_per_song))
            if m is not None)
        aug_plan, aug_target = plan_augmentation(
            pre_counts, augment_moods, augment_target, max_aug_per_clip,
            ratio=augment_ratio)
        aug_rng = np.random.default_rng(augment_seed)
        if aug_target is None:
            print(f"[AUG] Equal-ratio mode: every augmented mood gets the same "
                  f"~{augment_ratio:g} variants/clip, so artifacts carry no "
                  f"mood information. Class balance is left to the trainer's "
                  f"weighted sampler.")
        else:
            print(f"[AUG] Balancing to ~{aug_target} clips/mood:")
        for m in augment_moods:
            print(f"[AUG]   {m}: {pre_counts.get(m, 0)} real x "
                  f"~{aug_plan[m]:.2f} variants/clip")

    # CLAP embeddings are batched: holding every waveform would cost ~10 GB,
    # so they are flushed through the embedder a chunk at a time.
    clap_chunks, clap_buf = [], []

    def _flush_clap():
        if clap_buf:
            clap_chunks.append(clap_embedder(list(clap_buf), sr))
            clap_buf.clear()

    def _extract(wav_arr):
        """Mel-normalize + melody-extract one waveform, appending to lists."""
        wt = torch.FloatTensor(wav_arr).unsqueeze(0)
        mel = bigvgan_mel_spectrogram(wt, bigvgan_model)  # (1, M, T)
        mels.append(normalizer.normalize(mel))
        if melody_extractor is not None:
            melodies.append(torch.from_numpy(melody_extractor.extract(wav_arr)))
        if clap_embedder is not None:
            clap_buf.append(wav_arr)
            if len(clap_buf) >= clap_batch:
                _flush_clap()

    mels, melodies, features, mood_texts, names = [], [], [], [], []
    n_ambiguous = 0
    for path in tqdm(files, desc="Loading songs"):
        span, real_samples = _load_span(path, sr, clip_start_seconds,
                                        span_seconds)
        base = os.path.basename(path)
        for k in range(clips_per_song):
            # Keep a clip only if at least half of it is real audio
            # (always keep the first so short songs still contribute)
            if k > 0 and real_samples - k * seg_samples < seg_samples // 2:
                break
            wav = span[k * seg_samples:(k + 1) * seg_samples]

            mood = None
            if va is not None:
                valence, _ = va[song_id_from_filename(path)][k]
                mood = mood_from_va(valence)
                # Inside the valence dead band: too ambiguous to label, so
                # the clip never enters the dataset at all.
                if mood is None:
                    n_ambiguous += 1
                    continue

            _extract(wav)
            if mood is not None:
                mood_texts.append(mood)
            else:
                features.append(_heuristic_song_features(wav, sr))
            names.append(f"{base}#{k}")

            # Manufacture extra clips for under-represented moods.
            # Stochastic rounding of the fractional variant count keeps the
            # per-mood total on target instead of over/undershooting.
            n_aug = (variants_for_clip(aug_plan[mood], aug_rng)
                     if mood in aug_plan else 0)
            for j in range(n_aug):
                aug = augment_waveform(
                    wav, sr, aug_rng, max_semitones=aug_max_semitones,
                    max_shift_frac=aug_max_shift_frac,
                    snr_db_range=aug_snr_db_range)
                _extract(aug)
                mood_texts.append(mood)
                names.append(f"{base}#{k}~aug{j}")

    if va is None:
        mood_texts = _heuristic_labels(features)
    elif n_ambiguous:
        print(f"[DATA] Dropped {n_ambiguous} clips inside the valence dead "
              f"band (|v| < {VALENCE_DEAD_BAND:g})")

    _flush_clap()
    clap_audio = torch.cat(clap_chunks, dim=0) if clap_chunks else None
    if clap_audio is not None:
        print(f"[DATA] CLAP audio embeddings: {tuple(clap_audio.shape)}")

    # Shuffle so consecutive clips never share a song or a mood run;
    # fixed seed keeps the order (and the cache) reproducible.
    order = np.random.default_rng(shuffle_seed).permutation(len(names))
    mels = [mels[i] for i in order]
    if melodies:
        melodies = [melodies[i] for i in order]
    mood_texts = [mood_texts[i] for i in order]
    names = [names[i] for i in order]
    if clap_audio is not None:
        clap_audio = clap_audio[torch.from_numpy(order)]

    mel_batch = torch.stack(mels, dim=0)  # (N, 1, M, T)
    melody_batch = torch.stack(melodies, dim=0) if melodies else None

    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        arrays = {
            "mels": mel_batch.numpy().astype(np.float16),  # [0,1], fp16 is fine
            "moods": np.array(mood_texts),
            "names": np.array(names),
        }
        if melody_batch is not None:
            arrays["melodies"] = melody_batch.numpy().astype(np.int16)
        if clap_audio is not None:
            arrays["clap_audio"] = clap_audio.numpy().astype(np.float32)
        np.savez(cache_file, **arrays)
        print(f"[DATA] Cached dataset to {cache_file}")

    _print_distribution(mood_texts, len(names))
    return mel_batch, melody_batch, mood_texts, names, clap_audio


def _print_distribution(mood_texts, n):
    from collections import Counter
    print(f"[DATA] Loaded {n} clips, mood distribution: "
          f"{dict(Counter(mood_texts))}")
