"""Dry-run mood distribution counter.

Replicates the labelling half of `build_dataset` — the same 6 windows,
the same per-window valence averaging, the same happy/sad mapping and the
same dead-band drop — without decoding any audio or building mels/melodies.
Use it to see how many clips of each mood a retrain will actually use.

    python3 count_moods.py --audio_dir data/DEAM_audio/MEMD_audio

--audio_dir is optional: when given, only songs whose .mp3 is present are
counted (matching build_dataset's file filter); without it, every song
that has annotations is counted assuming all `clips_per_song` windows are
kept. Short-song window dropping is NOT modelled here, so on a corpus of
full-length DEAM excerpts (>=45 s) this is exact; for shorter songs it is
a slight over-count of the trailing windows.
"""

import argparse
import glob
import os
from collections import Counter

# annotations.py is numpy-only; we avoid importing config.py so this dry
# run needs no torch. Defaults below mirror config.DiffusionConfig.
from annotations import (load_annotation_windows, mood_from_va,
                         song_id_from_filename, VALENCE_DEAD_BAND)


def main():
    ap = argparse.ArgumentParser()
    # On HiPerGator the DEAM download lives under $DATA_ROOT (see the job
    # scripts); locally it's under data/. Honour DATA_ROOT if set.
    data_root = os.environ.get(
        "DATA_ROOT",
        "/orange/ufdatastudios/asherwheatle/DEAM_audio")
    default_annot = (os.path.join(data_root, "DEAM_Annotations")
                     if os.path.isdir(data_root)
                     else os.path.join("data", "DEAM_Annotations"))
    default_audio = (os.path.join(data_root, "MEMD_audio")
                     if os.path.isdir(data_root) else None)
    ap.add_argument("--audio_dir", type=str, default=default_audio,
                    help="if given, restrict to songs whose .mp3 exists here")
    ap.add_argument("--annotations_dir", type=str, default=default_annot)
    ap.add_argument("--clips_per_song", type=int, default=6)
    ap.add_argument("--clip_seconds", type=float, default=5)
    ap.add_argument("--clip_start_seconds", type=float, default=15)
    args = ap.parse_args()

    windows = [(args.clip_start_seconds + k * args.clip_seconds,
                args.clip_start_seconds + (k + 1) * args.clip_seconds)
               for k in range(args.clips_per_song)]
    print(f"[COUNT] {len(windows)} windows/song, "
          f"{args.clip_seconds:g}s each, "
          f"{windows[0][0]:.0f}-{windows[-1][1]:.0f}s")

    va = load_annotation_windows(args.annotations_dir, windows)

    if args.audio_dir:
        present = {song_id_from_filename(f)
                   for f in glob.glob(os.path.join(args.audio_dir, "*.mp3"))}
        before = len(va)
        va = {sid: pairs for sid, pairs in va.items() if sid in present}
        print(f"[COUNT] {len(va)}/{before} annotated songs have audio present")

    # Per-clip mood counts, plus which window index each mood comes from.
    # Clips inside the valence dead band have no mood; they are counted
    # separately because build_dataset drops them entirely.
    clip_moods = Counter()
    per_window = [Counter() for _ in range(args.clips_per_song)]
    dropped_per_window = [0] * args.clips_per_song
    n_total = 0
    for sid, pairs in va.items():
        for k, (v, a) in enumerate(pairs):
            n_total += 1
            mood = mood_from_va(v)
            if mood is None:
                dropped_per_window[k] += 1
                continue
            clip_moods[mood] += 1
            per_window[k][mood] += 1

    n_dropped = sum(dropped_per_window)
    n_clips = sum(clip_moods.values())
    print(f"\n=== Per-CLIP distribution ({n_clips} labeled of {n_total} "
          f"clips, {len(va)} songs x up to {args.clips_per_song}) ===")
    for mood, c in clip_moods.most_common():
        print(f"  {c:6d}  ({100*c/n_clips:4.1f}%)  {mood}")
    print(f"  {n_dropped:6d}  ({100*n_dropped/n_total:4.1f}% of all clips)  "
          f"DROPPED: |valence| < {VALENCE_DEAD_BAND:g}")

    if len(clip_moods) > 1:
        (maj, maj_n), (mino, min_n) = (clip_moods.most_common()[0],
                                       clip_moods.most_common()[-1])
        print(f"\nImbalance {maj} vs {mino}: {maj_n/min_n:.1f}x")

    print("\n=== By window position (are the labels spread out?) ===")
    for k in range(args.clips_per_song):
        s = k * args.clip_seconds + args.clip_start_seconds
        kept = ", ".join(f"{m.split()[0]} {per_window[k][m]}"
                         for m, _ in clip_moods.most_common())
        print(f"  window {k} ({s:.0f}-{s+args.clip_seconds:.0f}s): "
              f"{kept}, dropped {dropped_per_window[k]}")


if __name__ == "__main__":
    main()
