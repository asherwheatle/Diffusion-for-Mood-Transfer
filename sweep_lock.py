"""Does the melody lock + SDEdit start block sad -> happy?

Hypothesis: musical valence rides largely on mode (major/minor) and tempo.
The ControlNet melody embedding pins the top-k CQT pitch classes — i.e. the
mode — and SDEdit from a partial noise level keeps the input's timing. If so,
"sad" is reachable through timbre and dynamics while "happy" needs a mode
change the model is told not to make, which would explain why the happy text
push is about half the sad one (diagnose_stages.py, job 43193223).

Prediction if the lock is the cause: loosening it (melody_scale -> 0, higher
edit_strength) grows the happy push much more than the sad push, and the
sad -> happy edit moves the key toward MAJOR. Prediction if it is not: the
happy push stays small and asymmetric at every setting — the conditioning
itself is weak and loosening the lock only costs melody.

Grid: melody_scale x edit_strength, and at every cell three arms per song
sharing one noise seed: null (cfg 0), happy, sad. Text effect = arm - null,
so the prior's own drift at that setting is cancelled.

Judges:
  sadness   CLAP cos(sad caption) - cos(happy caption)
  mode      Krumhansl-Kessler major-minus-minor key correlation on the chroma
            (judge-free: >0 major-leaning). The direct test of the mechanism.
  onset_rate onsets per second (judge-free tempo/activity proxy; 5 s is too
            short for a stable BPM)
  chroma    similarity to the raw clip — what the lock is buying

Usage (GPU node):
  python sweep_lock.py --ckpt_dir output/job_perchan \
      --audio_dir .../MEMD_audio --annotations_dir .../DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt --n_songs 30
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
import librosa

from config import DiffusionConfig
from pipeline import load_bigvgan
from inference import edit_mood, reconstruct
from annotations import load_annotations, mood_from_va, song_id_from_filename
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS)

HAPPY, SAD = MOODS

# Krumhansl-Kessler key profiles (C major / C minor)
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mode_score(wav: np.ndarray, sr: int) -> float:
    """Best major-key correlation minus best minor-key correlation over all 12
    tonics. >0 = major-leaning, <0 = minor-leaning."""
    c = librosa.feature.chroma_cqt(y=wav, sr=sr).mean(axis=1)
    def best(profile):
        return max(np.corrcoef(c, np.roll(profile, k))[0, 1] for k in range(12))
    return float(best(_KK_MAJOR) - best(_KK_MINOR))


def onset_rate(wav: np.ndarray, sr: int) -> float:
    on = librosa.onset.onset_detect(y=wav, sr=sr, units="time")
    return len(on) / (len(wav) / sr)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=30)
    p.add_argument("--melody_scales", type=float, nargs="+", default=[1.0, 0.0])
    p.add_argument("--edit_strengths", type=float, nargs="+",
                   default=[0.6, 0.8, 0.95])
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)
    bigvgan = load_bigvgan(cfg.device)
    sr = bigvgan.h.sampling_rate
    clap = Clap(args.clap_ckpt)
    files = pick_annotated_songs(args.audio_dir, va, args.n_songs, balanced=True)
    sample = load_clip(files[0], sr, cfg.clip_start_seconds, cfg.clip_seconds)
    ae, dit, mel_enc, text_enc, diffusion, lat_mean, lat_std = load_models(
        cfg, args.ckpt_dir, sample, bigvgan, clap=clap)

    def measure(wav, wav_raw):
        cos = clap.cos_to_moods(clap.audio_embed(wav, sr))
        return {"sadness": float(cos[1] - cos[0]),
                "pred": MOODS[int(np.argmax(cos))].split()[0],
                "mode": mode_score(wav, sr), "onset_rate": onset_rate(wav, sr),
                "chroma": chroma_similarity(wav_raw, wav, sr)}

    rows = []
    for si, path in enumerate(files):
        gt = mood_from_va(*va[song_id_from_filename(path)]).split()[0]
        wav_raw = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        waveform = torch.FloatTensor(wav_raw).unsqueeze(0)
        base = {"song": os.path.basename(path), "gt": gt}
        rows.append({**base, "arm": "raw", "melody_scale": "", "strength": "",
                     **measure(wav_raw, wav_raw)})
        wav_ae = reconstruct(waveform, ae, bigvgan, cfg).squeeze(0).numpy()
        rows.append({**base, "arm": "ae", "melody_scale": "", "strength": "",
                     **measure(wav_ae, wav_raw)})
        for ms in args.melody_scales:
            for s in args.edit_strengths:
                cfg.edit_strength = s
                for arm, mood, g in [("null", HAPPY, 0.0),
                                     ("happy", HAPPY, args.cfg_scale),
                                     ("sad", SAD, args.cfg_scale)]:
                    cfg.cfg_scale = g
                    torch.manual_seed(args.seed + si)  # same noise per arm
                    wav = edit_mood(waveform, mood, ae, dit, mel_enc, text_enc,
                                    diffusion, bigvgan, cfg, lat_mean, lat_std,
                                    melody_scale=ms).squeeze(0).numpy()
                    rows.append({**base, "arm": arm, "melody_scale": ms,
                                 "strength": s, **measure(wav, wav_raw)})
        print(f"  {base['song']} gt={gt} [{si+1}/{len(files)}]", flush=True)

    out_csv = os.path.join(args.ckpt_dir, "sweep_lock.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SWEEP] wrote {out_csv}")
    summarize(rows, args.melody_scales, args.edit_strengths)


def _ms(xs):
    xs = np.asarray(xs, float)
    return xs.mean(), 2 * xs.std(ddof=1) / np.sqrt(len(xs))


def summarize(rows, scales, strengths):
    idx = {}
    for r in rows:
        idx[(r["song"], r["arm"], r["melody_scale"], r["strength"])] = r
    gt = {r["song"]: r["gt"] for r in rows}
    songs = sorted(gt)
    sad_songs = [s for s in songs if gt[s] == "sad"]
    happy_songs = [s for s in songs if gt[s] == "happy"]
    raw_gap = (np.mean([idx[(s, "raw", "", "")]["sadness"] for s in sad_songs]) -
               np.mean([idx[(s, "raw", "", "")]["sadness"] for s in happy_songs]))
    raw_mode_gap = (np.mean([idx[(s, "raw", "", "")]["mode"] for s in happy_songs]) -
                    np.mean([idx[(s, "raw", "", "")]["mode"] for s in sad_songs]))

    print("\n" + "=" * 96)
    print(" TEXT EFFECT vs the null arm at the same setting (mean +/- 2 SE)")
    print(f" real happy-vs-sad CLAP gap {raw_gap:+.4f}; real major-minus-minor "
          f"mode gap {raw_mode_gap:+.4f}")
    print("=" * 96)
    print(f" {'mel':>4s} {'str':>5s} | {'happy push (sad songs)':>22s} "
          f"{'sad push (happy songs)':>23s} {'ratio':>6s} | "
          f"{'mode shift s->h':>16s} {'mode h->s':>10s} | "
          f"{'s->h xfer':>9s} {'chroma':>6s}")
    for ms in scales:
        for st in strengths:
            k = lambda s, a: idx[(s, a, ms, st)]
            # happy push: how far toward happy (negative sadness) sad songs go
            hp = [k(s, "null")["sadness"] - k(s, "happy")["sadness"] for s in sad_songs]
            sp = [k(s, "sad")["sadness"] - k(s, "null")["sadness"] for s in happy_songs]
            hm = [k(s, "happy")["mode"] - k(s, "null")["mode"] for s in sad_songs]
            sm = [k(s, "null")["mode"] - k(s, "sad")["mode"] for s in happy_songs]
            xfer = np.mean([k(s, "happy")["pred"] == "happy" for s in sad_songs])
            chroma = np.mean([k(s, a)["chroma"] for s in songs
                              for a in ("happy", "sad")])
            (a, ae), (b, be) = _ms(hp), _ms(sp)
            (c, ce), (d, de) = _ms(hm), _ms(sm)
            print(f" {ms:4.1f} {st:5.2f} | {a:+.4f}±{ae:.4f} ({100*a/raw_gap:3.0f}%)"
                  f"  {b:+.4f}±{be:.4f} ({100*b/raw_gap:3.0f}%) {a/b if b else float('nan'):6.2f} |"
                  f" {c:+.4f}±{ce:.4f} {d:+.4f} | {100*xfer:8.0f}% {chroma:6.3f}")
    print(" happy/sad push: CLAP distance moved toward the target by the TEXT,")
    print("   in % of the real gap. ratio = happy push / sad push (1.0 = symmetric).")
    print(" mode shift: change in major-minus-minor score caused by the text, in")
    print("   the target's direction (>0 = right way).  s->h xfer: sad songs whose")
    print("   happy edit CLAP classifies as happy.  chroma: similarity to the raw clip.")

    print("\n PRIOR ALONE (null arm vs codec recon), sad songs:")
    for ms in scales:
        for st in strengths:
            d = [idx[(s, "null", ms, st)]["sadness"] - idx[(s, "ae", "", "")]["sadness"]
                 for s in sad_songs]
            m = [idx[(s, "null", ms, st)]["mode"] - idx[(s, "ae", "", "")]["mode"]
                 for s in sad_songs]
            print(f"   mel {ms:.1f} str {st:.2f}: sadness {np.mean(d):+.4f}  "
                  f"mode {np.mean(m):+.4f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
