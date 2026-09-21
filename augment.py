"""Waveform augmentation for balancing under-represented moods.

DEAM's valence ratings sit on a positive shift, so even after the dead band
(see annotations.VALENCE_DEAD_BAND) the happy/sad split is roughly 70/30.
Weighted sampling equalizes how often each mood is *seen*, but it can only
repeat the same sad clips. This module manufactures new, plausible sad clips
by perturbing the real ones so the model sees genuine variety instead of the
same handful over and over.

Three mood-preserving perturbations (small enough not to push a clip across
the valence split), composed per variant:

  * pitch shift   — +/- a few semitones (librosa phase-vocoder)
  * time shift    — translate the waveform in time, zero-filling the gap
  * gaussian noise — additive white noise at a randomized SNR

`plan_augmentation` decides how many variants each clip of a target mood
needs to reach parity with the majority; `build_dataset` calls
`augment_waveform` to realize them. Every op preserves clip length so the
augmented audio drops straight into the existing mel/melody extraction.
"""

import numpy as np
import librosa


def pitch_shift(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones (length-preserving)."""
    if n_steps == 0:
        return wav
    return librosa.effects.pitch_shift(y=wav, sr=sr, n_steps=n_steps)


def time_shift(wav: np.ndarray, shift_samples: int) -> np.ndarray:
    """Translate the waveform by shift_samples, zero-filling the vacated end.

    Positive shift delays the audio (silence prepended); negative advances
    it (silence appended). Unlike np.roll this does not wrap content around,
    which would splice the end of a clip onto its start.
    """
    if shift_samples == 0:
        return wav
    out = np.zeros_like(wav)
    n = len(wav)
    if shift_samples > 0:
        out[shift_samples:] = wav[:n - shift_samples]
    else:
        out[:n + shift_samples] = wav[-shift_samples:]
    return out


def add_gaussian_noise(wav: np.ndarray, snr_db: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Add white gaussian noise at a target signal-to-noise ratio (dB)."""
    sig_power = float(np.mean(wav ** 2))
    if sig_power <= 0:
        return wav
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape)
    return (wav + noise).astype(wav.dtype)


def augment_waveform(wav: np.ndarray, sr: int, rng: np.random.Generator,
                     max_semitones: float = 2.0,
                     max_shift_frac: float = 0.2,
                     snr_db_range=(20.0, 35.0)) -> np.ndarray:
    """One augmented variant: pitch shift + time shift + gaussian noise.

    Parameters are drawn randomly per call so repeated calls on the same
    clip give distinct variants. Output has the same length and dtype as
    the input and is clipped to [-1, 1].
    """
    # 1. pitch: random nonzero shift in [-max, +max] semitones
    n_steps = rng.uniform(-max_semitones, max_semitones)
    out = pitch_shift(wav, sr, n_steps)

    # 2. time: random translation up to max_shift_frac of the clip
    max_shift = int(max_shift_frac * len(wav))
    if max_shift > 0:
        out = time_shift(out, int(rng.integers(-max_shift, max_shift + 1)))

    # 3. noise: additive white noise at a random SNR
    snr_db = rng.uniform(*snr_db_range)
    out = add_gaussian_noise(out, snr_db, rng)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def plan_augmentation(counts: dict, moods_to_aug, target=None,
                      cap=None, ratio=None) -> tuple:
    """Decide how many augmented variants each clip of a mood needs.

    Two modes:

    target mode (ratio=None)
        Boost each listed mood UP TO `target` clips. Use this only when the
        trainer samples uniformly. It necessarily augments a rare mood harder
        than a common one, so the augmentation artifacts (pitch-shift ringing,
        added noise) end up correlated with the rare mood's label — a
        shortcut feature the model can learn instead of the actual mood.
        Measured on this corpus, the artifacts alone move CLAP +0.069 toward
        the sad caption and the valence probe -0.061.

    ratio mode (ratio=r)
        Give EVERY listed mood the same r variants per real clip, so the
        fraction of augmented clips is identical across moods and the
        artifacts carry no mood information. Class balance is left to the
        trainer's weighted sampler (train.train_diffusion already draws each
        batch with inverse-frequency weights), which is the right division of
        labour: the sampler equalizes counts, augmentation only adds variety.

    Returns a *fractional* variants-per-clip figure. Because a clip can
    only get a whole number of variants, the caller applies stochastic
    rounding (see `variants_for_clip`) so the realized total lands on the
    target instead of over/undershooting by a whole multiplier.

    Args:
        counts: {mood: n_clips} in the (un-augmented) dataset.
        moods_to_aug: iterable of moods to boost.
        target: desired per-mood clip count after augmentation. None means
                match the most common mood (full balance).
        cap: optional ceiling on variants-per-clip, to bound how hard a
             small pool of source songs is reused.

    Returns:
        (plan, target) where plan = {mood: mean_variants_per_existing_clip}.
        In ratio mode the second element is None — there is no single target
        count, since each mood keeps its own size.
    """
    if ratio is not None:
        r = float(ratio)
        if cap is not None:
            r = min(r, float(cap))
        r = max(0.0, r)
        return {mood: r for mood in moods_to_aug}, None

    if target is None:
        target = max(counts.values()) if counts else 0
    plan = {}
    for mood in moods_to_aug:
        cur = counts.get(mood, 0)
        if cur <= 0 or cur >= target:
            plan[mood] = 0.0
            continue
        extra = target / cur - 1.0           # avg extra copies per real clip
        if cap is not None:
            extra = min(extra, float(cap))
        plan[mood] = max(0.0, extra)
    return plan, target


def variants_for_clip(mean_variants: float,
                      rng: np.random.Generator) -> int:
    """Stochastically round a fractional variants-per-clip to an integer.

    E.g. mean_variants=2.4 yields 2 for ~60% of clips and 3 for ~40%, so
    the mood's clips average 2.4 extra variants and the total hits target.
    """
    base = int(mean_variants)
    if rng.random() < mean_variants - base:
        base += 1
    return base


def _demo():
    """CLI preview: print the balancing plan for the real DEAM annotations,
    and optionally render example augmented clips for auditioning."""
    import argparse
    import os
    from collections import Counter
    from annotations import load_annotation_windows, mood_from_va

    ap = argparse.ArgumentParser(description=__doc__)
    data_root = os.environ.get(
        "DATA_ROOT", "/orange/ufdatastudios/asherwheatle/DEAM_audio")
    ap.add_argument("--annotations_dir", type=str,
                    default=os.path.join(data_root, "DEAM_Annotations"))
    ap.add_argument("--clips_per_song", type=int, default=6)
    ap.add_argument("--clip_seconds", type=float, default=5)
    ap.add_argument("--clip_start_seconds", type=float, default=15)
    ap.add_argument("--mood", type=str, default="sad and melancholic",
                    help="mood to balance (repeatable via comma)")
    ap.add_argument("--target", type=int, default=None,
                    help="target clip count (default: match largest mood)")
    ap.add_argument("--cap", type=int, default=15,
                    help="max augmented variants per real clip")
    ap.add_argument("--demo_audio", type=str, default=None,
                    help="a .wav/.mp3 to render 3 example augmentations from")
    ap.add_argument("--demo_out", type=str, default="output/aug_demo")
    args = ap.parse_args()

    windows = [(args.clip_start_seconds + k * args.clip_seconds,
                args.clip_start_seconds + (k + 1) * args.clip_seconds)
               for k in range(args.clips_per_song)]
    va = load_annotation_windows(args.annotations_dir, windows)

    counts = Counter()
    dropped = 0
    for pairs in va.values():
        for v, a in pairs:
            mood = mood_from_va(v)
            if mood is None:      # inside the valence dead band
                dropped += 1
                continue
            counts[mood] += 1

    moods = [m.strip() for m in args.mood.split(",")]
    plan, target = plan_augmentation(counts, moods, args.target, args.cap)

    print(f"\nCurrent per-clip counts ({sum(counts.values())} clips, "
          f"{dropped} dropped in the valence dead band):")
    for m, c in counts.most_common():
        print(f"  {c:6d}  {m}")
    print(f"\nBalancing target: {target} clips/mood")
    for m in moods:
        cur = counts.get(m, 0)
        extra = plan[m]
        after = round(cur * (1 + extra))
        capped = " (CAP-limited)" if extra >= args.cap else ""
        print(f"  {m}: {cur} real x ~{extra:.2f} variants/clip "
              f"-> +{round(cur * extra)} augmented = ~{after} total{capped}")

    if args.demo_audio:
        try:
            import soundfile as sf
        except ImportError:
            print("\n[demo] install soundfile to render example clips")
            return
        sr = 44100
        wav, _ = librosa.load(args.demo_audio, sr=sr, mono=True,
                              offset=args.clip_start_seconds,
                              duration=args.clip_seconds)
        os.makedirs(args.demo_out, exist_ok=True)
        rng = np.random.default_rng(0)
        sf.write(os.path.join(args.demo_out, "orig.wav"), wav, sr)
        for j in range(3):
            aug = augment_waveform(wav, sr, rng)
            sf.write(os.path.join(args.demo_out, f"aug{j}.wav"), aug, sr)
        print(f"\n[demo] wrote orig + 3 augmentations to {args.demo_out}")


if __name__ == "__main__":
    _demo()
