"""Mood push vs audio quality: which strength x guidance setting is usable?

sweep_lock.py found that edit_strength 0.95 at cfg 7 "unlocks" sad -> happy
by CLAP. Heard in listen/ (job 43200245), those edits are not music: the
harmonic lines break into short fragments over broadband fizz and the melody
is unrecognizable, yet the sweep's chroma similarity still read 0.85. Chroma
between two UNRELATED songs is ~0.67, so 0.85 is already most of the way to
"a different song". The codec is not the cause: 1_recon keeps its harmonics,
nothing clips, and the AE decoder is Sigmoid-bounded.

So CLAP push alone rewards garble. This sweep scores every setting on push
AND on two structure measures, so a push that costs the music shows up as one:

  harm       HPSS harmonic-energy fraction. Sustained pitched notes are
             harmonic; the fizz is not. Reported relative to the song's own
             codec recon (1.0 = as tonal as the recon).
  melody     chroma similarity to the raw clip, rescaled so recon = 100% and
             an unrelated song (the next song in the list) = 0%.

Grid: edit_strength x (cfg_scale, cfg_rescale). Per song and strength, a
null arm (cfg 0) plus one edit toward the OTHER mood per guidance setting,
all sharing one noise seed, so push = edit - null cancels the prior's drift.
cfg_rescale is guidance rescale (Lin et al. 2023; see inference.edit_mood).

--inits adds a starting-latent axis (inference.edit_mood, cfg.edit_init):
"noise" is SDEdit, "invert" is DDIM inversion. With inversion, the null arm
(cfg 0, null prompt) replays the inversion's own predictions backward, so its
melody-kept should sit near 100%. If it does not, the inversion itself is
lossy and the edit arms cannot be judged. --invert_prompt source inverts with
the clip's own mood caption instead; the null arm is then no longer an exact
round trip.

WAVs for the first --n_listen songs of each mood go to
<ckpt_dir>/listen_quality/<song>_<gt>/, named by setting, so the table can be
checked by ear.

Usage (GPU node):
  python sweep_quality.py --ckpt_dir output/job_43200245 \
      --audio_dir .../MEMD_audio --annotations_dir .../DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt
"""

import os
import csv
import argparse

import numpy as np
import soundfile as sf
import torch
import librosa

from config import DiffusionConfig
from pipeline import load_bigvgan
from inference import edit_mood, reconstruct
from annotations import load_annotations, mood_from_va, song_id_from_filename
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS)

HAPPY, SAD = MOODS


def harmonic_ratio(wav: np.ndarray) -> float:
    """Fraction of spectral energy HPSS assigns to the harmonic component."""
    S = np.abs(librosa.stft(wav))
    H, P = librosa.decompose.hpss(S)
    h, p = float((H ** 2).sum()), float((P ** 2).sum())
    return h / (h + p + 1e-12)


def parse_guidance(specs):
    """'7' -> (7.0, 0.0); '7:0.7' -> (7.0, 0.7)."""
    out = []
    for s in specs:
        g, _, r = s.partition(":")
        out.append((float(g), float(r) if r else 0.0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--melody_scale", type=float, default=1.0)
    p.add_argument("--edit_strengths", type=float, nargs="+",
                   default=[0.5, 0.6, 0.8])
    p.add_argument("--guidance", nargs="+",
                   default=["1.5", "3", "7", "7:0.7"],
                   help="cfg_scale[:cfg_rescale] settings for the edit arm")
    p.add_argument("--inits", nargs="+", default=["noise", "invert"],
                   choices=["noise", "invert"],
                   help="Starting latent: SDEdit noise and/or DDIM inversion")
    p.add_argument("--invert_prompt", choices=["null", "source"],
                   default="null",
                   help="Caption the inversion is conditioned on")
    p.add_argument("--n_listen", type=int, default=3,
                   help="Songs per mood to also write as WAVs")
    args = p.parse_args()
    guidance = parse_guidance(args.guidance)

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

    raws = [load_clip(f, sr, cfg.clip_start_seconds, cfg.clip_seconds)
            for f in files]
    listen_root = os.path.join(args.ckpt_dir, "listen_quality")
    n_listened = {"happy": 0, "sad": 0}

    rows = []
    for si, path in enumerate(files):
        gt = mood_from_va(*va[song_id_from_filename(path)]).split()[0]
        target = HAPPY if gt == "sad" else SAD
        wav_raw = raws[si]
        unrelated = raws[(si + 1) % len(raws)]
        waveform = torch.FloatTensor(wav_raw).unsqueeze(0)
        song = os.path.basename(path)
        base = {"song": song, "gt": gt}

        listen_dir = None
        if n_listened[gt] < args.n_listen:
            n_listened[gt] += 1
            listen_dir = os.path.join(
                listen_root, f"{os.path.splitext(song)[0]}_{gt}")
            os.makedirs(listen_dir, exist_ok=True)

        def measure(wav):
            cos = clap.cos_to_moods(clap.audio_embed(wav, sr))
            return {"sadness": float(cos[1] - cos[0]),
                    "pred": MOODS[int(np.argmax(cos))].split()[0],
                    "harm": harmonic_ratio(wav),
                    "chroma": chroma_similarity(wav_raw, wav, sr)}

        wav_ae = reconstruct(waveform, ae, bigvgan, cfg).squeeze(0).numpy()
        rows.append({**base, "arm": "raw", "init": "", "strength": "", "cfg": "",
                     "rescale": "", **measure(wav_raw),
                     "chroma_unrelated": chroma_similarity(wav_raw, unrelated, sr)})
        rows.append({**base, "arm": "ae", "init": "", "strength": "", "cfg": "",
                     "rescale": "", **measure(wav_ae), "chroma_unrelated": ""})
        if listen_dir:
            sf.write(os.path.join(listen_dir, "0_raw.wav"), wav_raw, sr)
            sf.write(os.path.join(listen_dir, "1_recon.wav"), wav_ae, sr)

        source = SAD if gt == "sad" else HAPPY
        invert_mood = source if args.invert_prompt == "source" else None
        for init in args.inits:
          cfg.edit_init = init
          for s in args.edit_strengths:
            cfg.edit_strength = s
            arms = [("null", HAPPY, 0.0, 0.0)] + [
                (target.split()[0], target, g, r) for g, r in guidance]
            for arm, mood, g, r in arms:
                cfg.cfg_scale, cfg.cfg_rescale = g, r
                torch.manual_seed(args.seed + si)   # same noise per arm
                wav = edit_mood(waveform, mood, ae, dit, mel_enc, text_enc,
                                diffusion, bigvgan, cfg, lat_mean, lat_std,
                                melody_scale=args.melody_scale,
                                invert_mood=invert_mood).squeeze(0).numpy()
                rows.append({**base, "arm": arm, "init": init, "strength": s,
                             "cfg": g, "rescale": r, **measure(wav),
                             "chroma_unrelated": ""})
                if listen_dir:
                    tag = "2_null" if arm == "null" else f"3_{arm}_cfg{g:g}"
                    if r:
                        tag += f"_rs{r:g}"
                    sf.write(os.path.join(listen_dir, f"{tag}_{init}@{s}.wav"),
                             wav, sr)
        print(f"  {song} gt={gt} [{si+1}/{len(files)}]", flush=True)

    out_csv = os.path.join(args.ckpt_dir, "sweep_quality.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SWEEP] wrote {out_csv}")
    if any(n_listened.values()):
        print(f"[SWEEP] listening WAVs in {listen_root}")
    summarize(rows, args.inits, args.edit_strengths, guidance)


def summarize(rows, inits, strengths, guidance):
    idx = {(r["song"], r["arm"], r["init"], r["strength"], r["cfg"],
            r["rescale"]): r for r in rows}
    gt = {r["song"]: r["gt"] for r in rows}
    songs = sorted(gt)
    sad_songs = [s for s in songs if gt[s] == "sad"]
    happy_songs = [s for s in songs if gt[s] == "happy"]
    raw = lambda s: idx[(s, "raw", "", "", "", "")]
    ae = lambda s: idx[(s, "ae", "", "", "", "")]
    raw_gap = (np.mean([raw(s)["sadness"] for s in sad_songs]) -
               np.mean([raw(s)["sadness"] for s in happy_songs]))

    def rel_harm(s, r):
        return r["harm"] / ae(s)["harm"]

    def melody_kept(s, r):
        lo, hi = raw(s)["chroma_unrelated"], ae(s)["chroma"]
        return (r["chroma"] - lo) / (hi - lo)

    print("\n" + "=" * 107)
    print(" MOOD PUSH vs QUALITY  (push in % of the real happy-vs-sad CLAP gap "
          f"{raw_gap:+.4f})")
    print(f" codec recon: melody kept 100% by definition; raw harm/recon harm "
          f"{np.mean([raw(s)['harm'] / ae(s)['harm'] for s in songs]):.2f}")
    print("=" * 107)
    print(f" {'init':>6s} {'str':>4s} {'cfg':>4s} {'rs':>4s} | {'happy push':>10s} "
          f"{'s->h xfer':>9s} {'sad push':>9s} {'h->s xfer':>9s} | "
          f"{'harm vs recon':>13s} {'melody kept':>11s}")
    for init, st in ((i, s) for i in inits for s in strengths):
        null = lambda s: idx[(s, "null", init, st, 0.0, 0.0)]
        print(f" {init:>6s} {st:4.2f} null      | {'':10s} {'':9s} {'':9s} {'':9s} | "
              f"{np.mean([rel_harm(s, null(s)) for s in songs]):13.2f} "
              f"{100*np.mean([melody_kept(s, null(s)) for s in songs]):10.0f}%")
        for g, r in guidance:
            e = lambda s, a: idx[(s, a, init, st, g, r)]
            hp = np.mean([null(s)["sadness"] - e(s, "happy")["sadness"]
                          for s in sad_songs])
            sp = np.mean([e(s, "sad")["sadness"] - null(s)["sadness"]
                          for s in happy_songs])
            hx = np.mean([e(s, "happy")["pred"] == "happy" for s in sad_songs])
            sx = np.mean([e(s, "sad")["pred"] == "sad" for s in happy_songs])
            edits = ([rel_harm(s, e(s, "happy")) for s in sad_songs] +
                     [rel_harm(s, e(s, "sad")) for s in happy_songs])
            mel = ([melody_kept(s, e(s, "happy")) for s in sad_songs] +
                   [melody_kept(s, e(s, "sad")) for s in happy_songs])
            print(f" {init:>6s} {st:4.2f} {g:4.1f} {r:4.1f} | {100*hp/raw_gap:9.0f}% "
                  f"{100*hx:8.0f}% {100*sp/raw_gap:8.0f}% {100*sx:8.0f}% | "
                  f"{np.mean(edits):13.2f} {100*np.mean(mel):10.0f}%")
    print(" push: CLAP distance the TEXT moved clips toward the target vs the")
    print("   null arm (same noise). xfer: edits CLAP classifies as the target.")
    print(" harm vs recon: HPSS harmonic fraction / the song's codec recon's")
    print("   (<1 = less tonal, more noise-like). melody kept: chroma to raw,")
    print("   100% = codec recon, 0% = an unrelated song.")
    print(" invert: the null row is the inversion round trip; near 100% melody")
    print("   kept means the inversion is faithful.")
    print("=" * 107)


if __name__ == "__main__":
    main()
