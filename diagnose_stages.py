"""Which stage of the pipeline loses the sad -> happy edit?

Every edit crosses four stages, and each one could be the bottleneck:

  raw       the recording
  voc       mel -> BigVGAN, no autoencoder: the vocoder alone. Isolates BigVGAN.
  ae        mel -> AE encoder -> decoder -> BigVGAN (inference.reconstruct).
            ae - voc isolates the autoencoder.
  null      SDEdit with the UNCONDITIONAL model only (cfg_scale 0). What the
            diffusion prior does to a clip when nobody asks for a mood.
            null - ae isolates the diffusion prior.
  happy/sad SDEdit with the mood caption at the eval guidance. edit - null is
            what the text actually contributed.

The stages are cumulative, so each delta attributes a change to exactly one
component. Two questions are answered for each stage:

  1. Does mood information SURVIVE it? The happy-vs-sad gap on the CLAP
     sadness axis, measured across real happy and real sad songs, is compared
     stage by stage. A stage that shrinks the gap is erasing the features
     that distinguish the moods — no conditioning can steer toward features
     the codec cannot represent.
  2. Does it push every clip one WAY? The common-mode shift in sadness,
     valence, and plain acoustic features (brightness, high-band energy,
     onset strength, loudness). Happy music is brighter and more percussive
     than sad music, so a stage that dulls and smears audio moves everything
     toward sad — which would make happy edits fight the pipeline while sad
     edits ride it.

The acoustic features are judge-free: they do not depend on CLAP or on the
valence probe having been fit on real audio, so they are the check on whether
a CLAP/probe shift reflects a real change in the sound.

Usage (GPU node):
  python diagnose_stages.py --ckpt_dir output/job_perchan \
      --audio_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/MEMD_audio \
      --annotations_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt \
      --n_songs 30 --edit_strengths 0.4 0.6 --cfg_scale 7.0
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
import librosa

from config import DiffusionConfig
from pipeline import load_bigvgan, bigvgan_mel_spectrogram, FixedMelNormalizer
from inference import edit_mood, reconstruct
from annotations import load_annotations, mood_from_va, song_id_from_filename
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS)
from valence_probe import ValenceProbe

HAPPY, SAD = MOODS  # annotations.MOOD_PROMPTS order


# ---------------------------------------------------------------------------
# Judge-free acoustic features
# ---------------------------------------------------------------------------
def acoustic_features(wav: np.ndarray, sr: int) -> dict:
    S = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = S.sum() + 1e-12
    rms = librosa.feature.rms(y=wav)[0]
    return {
        # brightness; happy music skews brighter
        "centroid_hz": float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        # share of energy above 4 kHz: where a blurry codec loses first
        "hf_ratio": float(S[freqs >= 4000].sum() / total),
        # percussive attack strength; smearing onsets lowers it
        "onset": float(librosa.onset.onset_strength(y=wav, sr=sr).mean()),
        # temporal contrast: std of frame loudness (dB); smoothing lowers it
        "rms_db": float(20 * np.log10(rms.mean() + 1e-9)),
        "rms_var_db": float(np.std(20 * np.log10(rms + 1e-9))),
    }


# ---------------------------------------------------------------------------
# Vocoder-only path
# ---------------------------------------------------------------------------
@torch.no_grad()
def vocode_only(waveform: torch.Tensor, bigvgan_model, device) -> tuple:
    """mel -> BigVGAN with the same FixedMelNormalizer clamp the AE path uses,
    so voc differs from ae ONLY by the autoencoder. Also reports how much of
    the mel the clamp cuts off, since a clamp that bites is a codec loss that
    would otherwise be blamed on the AE or BigVGAN."""
    mel = bigvgan_mel_spectrogram(waveform, bigvgan_model)
    norm = FixedMelNormalizer()
    clipped_hi = float((mel > norm.mel_max).float().mean())
    mel_c = norm.denormalize(norm.normalize(mel))
    wav = bigvgan_model(mel_c.to(device)).squeeze().cpu().clamp(-1, 1).numpy()
    return wav, clipped_hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--probe_path", default=None,
                   help="Cached raw-domain valence_probe.npz (evaluate.py "
                        "writes one per ckpt dir). Optional.")
    p.add_argument("--n_songs", type=int, default=30,
                   help="Balanced across moods")
    p.add_argument("--edit_strengths", type=float, nargs="+", default=[0.4, 0.6])
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
    device = cfg.device
    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)
    bigvgan = load_bigvgan(device)
    sr = bigvgan.h.sampling_rate
    clap = Clap(args.clap_ckpt)

    probe = None
    probe_path = args.probe_path or os.path.join(args.ckpt_dir, "valence_probe.npz")
    if os.path.exists(probe_path):
        probe = ValenceProbe.load(probe_path)
        print(f"[DIAG] valence probe: {probe_path}")
    else:
        print("[DIAG] no valence probe found; reporting CLAP + acoustics only")

    files = pick_annotated_songs(args.audio_dir, va, args.n_songs, balanced=True)
    sample = load_clip(files[0], sr, cfg.clip_start_seconds, cfg.clip_seconds)
    models = load_models(cfg, args.ckpt_dir, sample, bigvgan, clap=clap)
    ae, dit, melody_enc, text_enc, diffusion, lat_mean, lat_std = models

    def score(wav):
        emb = clap.audio_embed(wav, sr)
        cos = clap.cos_to_moods(emb)
        out = {"sadness": float(cos[1] - cos[0]),
               "cos_happy": float(cos[0]), "cos_sad": float(cos[1])}
        if probe is not None:
            out["valence"] = float(probe.predict(emb))
        out.update(acoustic_features(wav, sr))
        return out

    def run_edit(waveform, mood, strength, cfg_scale, seed):
        cfg.edit_strength = strength
        cfg.cfg_scale = cfg_scale
        torch.manual_seed(seed)
        return edit_mood(waveform, mood, ae, dit, melody_enc, text_enc,
                         diffusion, bigvgan, cfg, lat_mean, lat_std
                         ).squeeze(0).numpy()

    rows = []
    for si, path in enumerate(files):
        sid = song_id_from_filename(path)
        gt = mood_from_va(*va[sid])
        wav_raw = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        waveform = torch.FloatTensor(wav_raw).unsqueeze(0)

        versions = {"raw": wav_raw}
        versions["voc"], clipped = vocode_only(waveform, bigvgan, device)
        versions["ae"] = reconstruct(waveform, ae, bigvgan, cfg).squeeze(0).numpy()
        for s in args.edit_strengths:
            # Same seed for every arm of this song => only the conditioning
            # differs between null / happy / sad at a given strength.
            seed = args.seed + si
            versions[f"null@{s}"] = run_edit(waveform, HAPPY, s, 0.0, seed)
            versions[f"happy@{s}"] = run_edit(waveform, HAPPY, s, args.cfg_scale, seed)
            versions[f"sad@{s}"] = run_edit(waveform, SAD, s, args.cfg_scale, seed)

        for name, wav in versions.items():
            r = {"song": os.path.basename(path), "gt": gt.split()[0],
                 "version": name, "mel_clipped_hi": clipped,
                 "chroma_vs_raw": chroma_similarity(wav_raw, wav, sr)}
            r.update(score(wav))
            rows.append(r)
        print(f"  {os.path.basename(path)} gt={gt.split()[0]} [{si+1}/{len(files)}]",
              flush=True)

    out_csv = os.path.join(args.ckpt_dir, "diagnose_stages.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[DIAG] wrote {out_csv}")
    summarize(rows, args.edit_strengths)


def summarize(rows, strengths):
    by = defaultdict(dict)                     # by[song][version] = row
    gt = {}
    for r in rows:
        by[r["song"]][r["version"]] = r
        gt[r["song"]] = r["gt"]
    songs = sorted(by)
    metrics = [m for m in ["sadness", "valence", "centroid_hz", "hf_ratio",
                           "onset", "rms_db", "rms_var_db", "chroma_vs_raw"]
               if m in rows[0]]

    def mean_se(xs):
        xs = np.asarray(xs, float)
        return xs.mean(), (2 * xs.std(ddof=1) / np.sqrt(len(xs))
                           if len(xs) > 1 else float("nan"))

    print("\n" + "=" * 78)
    print(" 1. DOES MOOD SURVIVE EACH STAGE?  (real happy vs real sad songs)")
    print("=" * 78)
    print(" gap = mean sadness(sad songs) - mean sadness(happy songs); raw = 100%")
    base_gap = None
    for v in ["raw", "voc", "ae"] + [f"null@{s}" for s in strengths]:
        h = [by[s][v]["sadness"] for s in songs if gt[s] == "happy"]
        sd = [by[s][v]["sadness"] for s in songs if gt[s] == "sad"]
        gap = np.mean(sd) - np.mean(h)
        base_gap = gap if base_gap is None else base_gap
        # AUC: how often a random sad song scores sadder than a random happy one
        auc = np.mean([a > b for a in sd for b in h])
        print(f"   {v:10s} gap {gap:+.4f}  ({100*gap/base_gap:5.1f}% of raw)"
              f"   AUC {auc:.2f}")

    print("\n" + "=" * 78)
    print(" 2. WHAT DOES EACH STAGE DO TO EVERY CLIP?  (mean delta +/- 2 SE)")
    print("=" * 78)
    steps = [("BigVGAN   voc-raw", "raw", "voc"),
             ("AE        ae-voc", "voc", "ae")]
    for s in strengths:
        steps += [(f"prior@{s}  null-ae", "ae", f"null@{s}"),
                  (f"text@{s}   happy-null", f"null@{s}", f"happy@{s}"),
                  (f"text@{s}   sad-null", f"null@{s}", f"sad@{s}")]
    hdr = "".join(f"{m[:11]:>14s}" for m in metrics)
    print(f"   {'stage':22s}{hdr}")
    for label, a, b in steps:
        cells = []
        for m in metrics:
            mu, se = mean_se([by[s][b][m] - by[s][a][m] for s in songs])
            cells.append(f"{mu:+8.3f}±{se:<5.3f}" if abs(mu) < 100
                         else f"{mu:+8.0f}±{se:<5.0f}")
        print(f"   {label:22s}" + "".join(f"{c:>14s}" for c in cells))
    print("   (sadness>0 = toward the sad caption; valence<0 = sadder)")

    print("\n" + "=" * 78)
    print(" 3. THE ASYMMETRY  (sadness of edit relative to the codec recon 'ae')")
    print("=" * 78)
    for s in strengths:
        for src, tgt in [("sad", "happy"), ("happy", "sad"),
                         ("happy", "happy"), ("sad", "sad")]:
            d = [by[x][f"{tgt}@{s}"]["sadness"] - by[x]["ae"]["sadness"]
                 for x in songs if gt[x] == src]
            dn = [by[x][f"null@{s}"]["sadness"] - by[x]["ae"]["sadness"]
                  for x in songs if gt[x] == src]
            mu, se = mean_se(d)
            print(f"   @{s}  {src:5s} -> {tgt:5s}: {mu:+.4f} ±{se:.4f}"
                  f"   (null prior alone: {np.mean(dn):+.4f})")
    print("   Desired sign: -> happy should be NEGATIVE, -> sad POSITIVE.")
    mc = np.mean([r["mel_clipped_hi"] for r in rows if r["version"] == "raw"])
    print(f"\n   mel frames above the normalizer ceiling (2.5): {100*mc:.3f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
