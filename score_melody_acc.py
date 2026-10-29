"""Score already-rendered edits with Hou et al. (2024)'s melody accuracy.

Walks the listen*/ folders that render_listen.py and sweep_quality.py write
(one folder per song: 0_raw.wav, 1_recon.wav, then one WAV per setting) and
compares every WAV against that song's 0_raw.wav with
evaluate.melody_accuracy: the fraction of frames whose dominant pitch class
matches. Needs no GPU or checkpoint, only the saved audio.

Two reference rows frame the numbers:
  1_recon      codec round trip, no diffusion. The ceiling: every edit pays
               this cost before diffusion changes anything.
  (unrelated)  the next song's 0_raw.wav. The floor: what a clip that shares
               nothing with the source scores by accident.

Hou et al. report 56.6% for their model and 42-45% for MusicGen-melody, on the
Song Describer no-singing subset. These clips are 5 s DEAM excerpts, so the
comparison is same metric, different data.

Usage:
  python score_melody_acc.py output/job_43200245
Writes melody_acc.csv next to each listen*/ folder it scores.
"""

import os
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import librosa

from evaluate import melody_accuracy

SR = 44100


def score_folder(listen_root):
    song_dirs = sorted(d for d in glob.glob(os.path.join(listen_root, "*"))
                       if os.path.isfile(os.path.join(d, "0_raw.wav")))
    if not song_dirs:
        return []
    raws = [librosa.load(os.path.join(d, "0_raw.wav"), sr=SR, mono=True)[0]
            for d in song_dirs]
    rows = []
    for si, d in enumerate(song_dirs):
        song = os.path.basename(d)
        rows.append({"song": song, "setting": "(unrelated song)",
                     "melody_acc": melody_accuracy(
                         raws[si], raws[(si + 1) % len(raws)], SR)})
        for wav_path in sorted(glob.glob(os.path.join(d, "*.wav"))):
            setting = os.path.splitext(os.path.basename(wav_path))[0]
            if setting == "0_raw":
                continue
            wav = librosa.load(wav_path, sr=SR, mono=True)[0]
            rows.append({"song": song, "setting": setting,
                         "melody_acc": melody_accuracy(raws[si], wav, SR)})
    with open(os.path.join(listen_root, "melody_acc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["song", "setting", "melody_acc"])
        w.writeheader(); w.writerows(rows)
    return rows


def print_table(listen_root, rows):
    by_setting = defaultdict(list)
    for r in rows:
        by_setting[r["setting"]].append(r["melody_acc"])
    n_songs = len({r["song"] for r in rows})
    print(f"\n{'='*64}\n {listen_root}  ({n_songs} songs)\n{'='*64}")
    print(f" {'setting':36s} {'n':>3s} {'melody acc':>11s} {'min-max':>12s}")
    for setting in sorted(by_setting):
        v = by_setting[setting]
        print(f" {setting:36s} {len(v):>3d} {100*np.mean(v):>10.1f}% "
              f"{100*min(v):>5.0f}-{100*max(v):<5.0f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ckpt_dir", help="run folder holding listen*/ subfolders")
    args = p.parse_args()
    roots = sorted(d for d in glob.glob(os.path.join(args.ckpt_dir, "listen*"))
                   if os.path.isdir(d))
    for root in roots:
        rows = score_folder(root)
        if rows:
            print_table(root, rows)
    print("\n Same metric as Hou et al. (56.6% theirs, 42-45% MusicGen-melody);"
          "\n different data. 1_recon is this model's ceiling, (unrelated song)"
          "\n its floor; chance is ~8%.")


if __name__ == "__main__":
    main()
