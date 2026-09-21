"""DEAM valence/arousal annotation loading and mood-text mapping.

Mood labels are binary and come from valence only (see `mood_from_va`);
arousal is still loaded because the valence probe and the analysis scripts
use it, but it plays no part in labeling.

DEAM provides two annotation formats, both supported here:

  dynamic (per-second):  arousal.csv / valence.csv
      one row per song: song_id, sample_15000ms, sample_15500ms, ...
      values in roughly [-1, 1], 500 ms steps, starting at 15 s
      (the first 15 s of each excerpt is unannotated)

  static (song-level):   static_annotations_averaged_songs_*.csv
      columns: song_id, valence_mean, arousal_mean, ... on a 1-9 scale

The loader searches `annotations_dir` recursively, prefers the dynamic
files (they let us average over the exact clip window we train on), and
falls back to the static ones. Everything is rescaled to [-1, 1].
"""

import csv
import glob
import os

import numpy as np


# Two moods, split on valence alone. Arousal is deliberately ignored: the
# arousal-derived labels ("energetic and powerful", "calm and peaceful") are
# the ones CLAP separates worst, so carrying them cost more CLAP gain than
# the extra vocabulary was worth. The smallest possible vocabulary gives the
# conditioning signal the best chance of being learned.
MOOD_HAPPY = "happy and uplifting"
MOOD_SAD = "sad and melancholic"

# Caption-like prompts for each mood. CLAP was trained on natural captions,
# not bare tags, so "a sad and melancholic piece of music" lands in a much
# better place than "sad and melancholic". This is the single source of truth:
# training, inference and evaluation must all embed the SAME string for a
# mood, or the model is conditioned on one vector and judged against another.
MOOD_PROMPTS = {
    MOOD_HAPPY: "a happy and uplifting piece of music",
    MOOD_SAD:   "a sad and melancholic piece of music",
}


def mood_prompt(mood: str) -> str:
    """Caption for a mood label; passes unknown strings through unchanged."""
    return MOOD_PROMPTS.get(mood, mood)


# Half-width of the dead band around valence 0. Clips landing inside it are
# dropped rather than labeled: DEAM's valence ratings sit on a positive shift,
# so a clip at v = -0.05 is "slightly below the corpus mean", not something a
# listener would call sad. Labeling those as sad is what made the two classes
# overlap. The band costs ~28% of the clips and leaves ~7.6k genuinely-labeled
# ones at roughly 70/30 happy/sad.
VALENCE_DEAD_BAND = 0.1

# Dynamic annotations start at 15 s into each excerpt.
DYNAMIC_START_MS = 15000
DYNAMIC_STEP_MS = 500


def mood_from_va(valence: float, arousal: float = None):
    """Map a valence on the [-1, 1] scale to a mood string.

    `arousal` is accepted (so existing `mood_from_va(*va[sid])` calls keep
    working) but ignored. Returns None for clips inside the dead band, i.e.
    ones whose label is too ambiguous to train or evaluate on; every caller
    must handle that by dropping the clip.
    """
    if valence > VALENCE_DEAD_BAND:
        return MOOD_HAPPY
    if valence < -VALENCE_DEAD_BAND:
        return MOOD_SAD
    return None


def _find_file(annotations_dir: str, filename: str):
    matches = glob.glob(os.path.join(annotations_dir, "**", filename),
                        recursive=True)
    return matches[0] if matches else None


def _read_dynamic_csv(path: str):
    """Read one dynamic annotation CSV in full.

    Returns:
        sample_ms: (T,) sample times in milliseconds
        series: {song_id: (T,) values, NaN-padded to the header length}
    """
    series = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_ms = np.array([
            int(col.split("_")[1].replace("ms", "")) for col in header[1:]
        ])
        for row in reader:
            if not row or not row[0].strip():
                continue
            song_id = int(float(row[0]))
            vals = np.array([
                float(v) if v.strip() else np.nan for v in row[1:]
            ])
            # Rows can be shorter (short songs) or longer than the header
            if len(vals) < len(sample_ms):
                vals = np.pad(vals, (0, len(sample_ms) - len(vals)),
                              constant_values=np.nan)
            else:
                vals = vals[:len(sample_ms)]
            series[song_id] = vals
    return sample_ms, series


def _window_mean(sample_ms: np.ndarray, vals: np.ndarray,
                 start_s: float, end_s: float):
    """Mean of vals over [start_s, end_s); falls back to the whole-song
    mean when the window has no annotated samples (e.g. clips taken from
    the unannotated first 15 s). None if the song has no data at all."""
    windowed = vals[(sample_ms >= start_s * 1000) &
                    (sample_ms < end_s * 1000)]
    if np.isfinite(windowed).any():
        return float(np.nanmean(windowed))
    if np.isfinite(vals).any():
        return float(np.nanmean(vals))
    return None


def _load_static_csv(paths: list) -> dict:
    """Read static song-level CSVs -> {song_id: (valence, arousal)} in [-1, 1]."""
    va = {}
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            # DEAM headers sometimes carry stray spaces (" valence_mean")
            field = {name.strip(): name for name in reader.fieldnames}
            for row in reader:
                song_id = int(float(row[field["song_id"]]))
                v = float(row[field["valence_mean"]])
                a = float(row[field["arousal_mean"]])
                # 1-9 scale -> [-1, 1]
                va[song_id] = ((v - 5.0) / 4.0, (a - 5.0) / 4.0)
    return va


def load_annotation_windows(annotations_dir: str, windows: list) -> dict:
    """
    Load DEAM annotations from anywhere under annotations_dir, averaged
    per clip window.

    Args:
        windows: list of (start_s, end_s) clip windows.

    Returns:
        {song_id: [(valence, arousal), ...]} with one pair per window,
        both values in [-1, 1]. With dynamic annotations each window gets
        its own mean; with the static fallback every window shares the
        song-level value.

    Raises FileNotFoundError if no known annotation files are present.
    """
    valence_path = _find_file(annotations_dir, "valence.csv")
    arousal_path = _find_file(annotations_dir, "arousal.csv")
    if valence_path and arousal_path:
        print(f"[ANNOT] Dynamic annotations: {valence_path}")
        v_ms, v_series = _read_dynamic_csv(valence_path)
        a_ms, a_series = _read_dynamic_csv(arousal_path)
        va = {}
        for sid in v_series.keys() & a_series.keys():
            pairs = []
            for start_s, end_s in windows:
                v = _window_mean(v_ms, v_series[sid], start_s, end_s)
                a = _window_mean(a_ms, a_series[sid], start_s, end_s)
                if v is None or a is None:
                    pairs = None
                    break
                pairs.append((v, a))
            if pairs is not None:
                va[sid] = pairs
        print(f"[ANNOT] {len(va)} songs with valence+arousal "
              f"({len(windows)} windows, "
              f"{windows[0][0]:.0f}-{windows[-1][1]:.0f}s)")
        return va

    static_paths = glob.glob(os.path.join(
        annotations_dir, "**", "static_annotations_averaged*.csv"),
        recursive=True)
    if static_paths:
        print(f"[ANNOT] Static annotations: {static_paths}")
        static = _load_static_csv(static_paths)
        va = {sid: [pair] * len(windows) for sid, pair in static.items()}
        print(f"[ANNOT] {len(va)} songs with valence+arousal "
              f"(song-level, repeated per window)")
        return va

    raise FileNotFoundError(
        f"No DEAM annotations found under {annotations_dir}. Expected "
        f"valence.csv + arousal.csv (dynamic) or "
        f"static_annotations_averaged*.csv (song-level).")


def load_annotations(annotations_dir: str, clip_start_seconds: float = 15,
                     clip_seconds: float = 15) -> dict:
    """Single-window convenience wrapper: {song_id: (valence, arousal)}."""
    window = (clip_start_seconds, clip_start_seconds + clip_seconds)
    va = load_annotation_windows(annotations_dir, [window])
    return {sid: pairs[0] for sid, pairs in va.items()}


def song_id_from_filename(filename: str):
    """DEAM audio files are named '<song_id>.mp3'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except ValueError:
        return None
