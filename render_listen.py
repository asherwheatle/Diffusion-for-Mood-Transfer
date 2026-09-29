"""Render sweep_lock.py's edits to WAV so they can be judged by ear.

CLAP says the sad -> happy edits work at high edit_strength, but the
judge-free key-mode score never moved, so the edits need listening to. This
re-renders the exact edits the sweep scored: same song list, same per-song
seed (args.seed + song index), same melody_scale, so each file's CLAP numbers
in sweep_lock.csv describe the audio actually heard.

Per song, into <ckpt_dir>/listen/<song>_<gt>/:
  0_raw.wav             the recording
  1_recon.wav           codec round-trip only (the fair baseline: every edit
                        pays this resynthesis cost, so compare edits to this)
  2_null@<s>.wav        no prompt at strength s — what the prior alone does
  3_<target>@<s>.wav    the edit toward the OTHER mood at each strength
and an INDEX.txt ranking the songs by CLAP push, so the best, median and
worst cases can be picked out rather than only the flattering ones.

Usage (GPU node):
  python render_listen.py --ckpt_dir output/job_43200245 \
      --audio_dir .../MEMD_audio --annotations_dir .../DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt
"""

import os
import csv
import argparse

import numpy as np
import soundfile as sf
import torch

from config import DiffusionConfig
from pipeline import load_bigvgan
from inference import edit_mood, reconstruct
from annotations import load_annotations, mood_from_va, song_id_from_filename
from evaluate import Clap, load_models, load_clip, pick_annotated_songs, MOODS

HAPPY, SAD = MOODS


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    # Must match the sweep so the song list and seeds line up with its CSV.
    p.add_argument("--n_songs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--melody_scale", type=float, default=1.0)
    p.add_argument("--edit_strengths", type=float, nargs="+",
                   default=[0.6, 0.8, 0.95])
    p.add_argument("--null_strength", type=float, default=0.8)
    p.add_argument("--gt", choices=["sad", "happy", "both"], default="sad",
                   help="Which source songs to render (sad = the sad->happy "
                        "direction under test)")
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

    out_root = os.path.join(args.ckpt_dir, "listen")
    os.makedirs(out_root, exist_ok=True)

    def edit(waveform, mood, s, g, si):
        cfg.edit_strength, cfg.cfg_scale = s, g
        torch.manual_seed(args.seed + si)   # sweep_lock's per-song seed
        return edit_mood(waveform, mood, ae, dit, mel_enc, text_enc, diffusion,
                         bigvgan, cfg, lat_mean, lat_std,
                         melody_scale=args.melody_scale).squeeze(0).numpy()

    rendered = []
    for si, path in enumerate(files):
        gt = mood_from_va(*va[song_id_from_filename(path)]).split()[0]
        if args.gt != "both" and gt != args.gt:
            continue
        target = HAPPY if gt == "sad" else SAD
        song = os.path.splitext(os.path.basename(path))[0]
        d = os.path.join(out_root, f"{song}_{gt}")
        os.makedirs(d, exist_ok=True)
        wav_raw = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        waveform = torch.FloatTensor(wav_raw).unsqueeze(0)
        sf.write(os.path.join(d, "0_raw.wav"), wav_raw, sr)
        sf.write(os.path.join(d, "1_recon.wav"),
                 reconstruct(waveform, ae, bigvgan, cfg).squeeze(0).numpy(), sr)
        sf.write(os.path.join(d, f"2_null@{args.null_strength}.wav"),
                 edit(waveform, HAPPY, args.null_strength, 0.0, si), sr)
        for s in args.edit_strengths:
            sf.write(os.path.join(d, f"3_{target.split()[0]}@{s}.wav"),
                     edit(waveform, target, s, args.cfg_scale, si), sr)
        rendered.append((f"{song}.mp3", gt, target.split()[0], d))
        print(f"  rendered {d}", flush=True)

    write_index(args, rendered, out_root)


def write_index(args, rendered, out_root):
    """Rank rendered songs by the sweep's CLAP push at each strength."""
    sweep = os.path.join(args.ckpt_dir, "sweep_lock.csv")
    lines = [f"Rendered at melody_scale={args.melody_scale}, "
             f"cfg_scale={args.cfg_scale}. Compare edits to 1_recon.wav, not "
             f"0_raw.wav:", "every edit pays the codec's resynthesis cost.", ""]
    if not os.path.exists(sweep):
        lines.append("(no sweep_lock.csv found; no CLAP ranking)")
    else:
        rows = list(csv.DictReader(open(sweep)))
        idx = {(r["song"], r["arm"], float(r["melody_scale"] or -1),
                float(r["strength"] or -1)): r for r in rows}
        s_rank = args.edit_strengths[len(args.edit_strengths) // 2]
        lines.append("push = CLAP distance the TEXT moved the clip toward its "
                     "target vs the no-prompt edit")
        lines.append("(real happy-vs-sad gap is ~0.124). Sorted by push at "
                     f"strength {s_rank}.")
        lines.append("")
        table = []
        for song, gt, tgt, d in rendered:
            pushes, preds = [], []
            for s in args.edit_strengths:
                e = idx.get((song, tgt, args.melody_scale, s))
                n = idx.get((song, "null", args.melody_scale, s))
                if e is None or n is None:
                    pushes.append(float("nan")); preds.append("?")
                    continue
                sign = 1 if tgt == "happy" else -1   # happy = lower sadness
                pushes.append(sign * (float(n["sadness"]) - float(e["sadness"])))
                preds.append(e["pred"])
            table.append((pushes[len(pushes) // 2], song, gt, tgt, pushes,
                          preds, d))
        table.sort(reverse=True)
        hdr = "  ".join(f"push@{s}  CLAP@{s}" for s in args.edit_strengths)
        lines.append(f"{'rank':4s} {'song':10s} {'direction':13s} {hdr}")
        for i, (_, song, gt, tgt, pushes, preds, d) in enumerate(table):
            cells = "  ".join(f"{p:+.4f}  {q:>7s}"
                              for p, q in zip(pushes, preds))
            lines.append(f"{i+1:<4d} {song:10s} {gt+'->'+tgt:13s} {cells}"
                         f"   {os.path.relpath(d, args.ckpt_dir)}")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "INDEX.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
