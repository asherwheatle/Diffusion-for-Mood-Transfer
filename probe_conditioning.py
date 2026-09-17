"""Does the conditioning path work, and does the text path reach it?

Training conditions the DiT on CLAP *audio* embeddings
(cfg.clap_cond_source="audio"); inference prompts it with CLAP *text*
embeddings. The two live in offset cones, so a bad edit could mean either
"conditioning is broken" or "conditioning was handed a vector from the wrong
cone". Same checkpoint, same songs, same sampling noise — only the
conditioning VECTOR changes between arms:

  text      the mood caption through the checkpoint's own alignment
            (ClapTextEncoder.align_text). Exactly what inference does.
  text_raw  the same caption with the alignment bypassed. For a checkpoint
            trained without alignment this equals `text`; for one trained with
            it, text - text_raw is what the alignment buys.
  audio     the centroid of real clips of the target mood, in the model's
            aligned audio space. The kind of vector the model trained on, so
            the modality gap is removed entirely. THE REFERENCE ARM.

Primary metric: cond_margin
---------------------------
Every edit passes through the autoencoder + BigVGAN, which lowers CLAP cosine
to *any* music caption. That penalty is large (~-0.14 on job_41644276) and
nearly identical across arms and target moods, so absolute clap_gain mostly
measures the resynthesis, not the mood. The earlier version of this probe
judged on absolute gain and called conditioning "broken" even though, in its
own data, the audio arm's happy/sad edits of a song diverged in 5/10 songs,
every time in the correct direction.

cond_margin compares a song's edits against EACH OTHER, which cancels the
shared penalty: for target mood m,

    cond_margin = cos(edit_m, m) - mean over all targets m' of cos(edit_m', m)

"How much more does the edit aimed at m sound like m than this song's edits
do on average?" > 0 means conditioning pushed toward the target. The paired
column counts songs where every mood's edit is classified as its own target.

Usage (GPU node):
  python probe_conditioning.py \
      --ckpt_dir output/job_41644276 \
      --audio_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/MEMD_audio \
      --annotations_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt \
      --n_songs 10 --n_ref 24
"""

import os
import csv
import argparse

import numpy as np
import torch

from config import DiffusionConfig
from pipeline import load_bigvgan
from inference import edit_mood
from annotations import (load_annotations, mood_from_va,
                         song_id_from_filename)
# reuse the evaluation helpers so this script and evaluate.py agree on the judge
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS, _l2)

ARMS = ["text", "text_raw", "audio"]


def collect_reference_audio(clap, files, va, sr, start_s, dur_s, n_ref):
    """Raw CLAP audio embeddings of up to n_ref real clips per mood.

    Returns {mood: (k, D) array}. Raw on purpose: the caller maps them into the
    checkpoint's aligned space, so the same references serve any checkpoint.
    """
    per_mood = {m: [] for m in MOODS}
    for path in files:
        gt = mood_from_va(*va[song_id_from_filename(path)])
        if gt is None or len(per_mood[gt]) >= n_ref:
            continue
        wav = load_clip(path, sr, start_s, dur_s)
        per_mood[gt].append(clap.audio_embed(wav, sr))
        if all(len(v) >= n_ref for v in per_mood.values()):
            break
    for m in MOODS:
        if not per_mood[m]:
            raise RuntimeError(f"No reference clips found for mood {m!r}. "
                               f"Raise --n_ref_pool.")
        print(f"[REF] {m:24s} {len(per_mood[m]):3d} reference clips")
    return {m: np.stack(v) for m, v in per_mood.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=10,
                   help="Test songs to edit (each x every mood x every arm)")
    p.add_argument("--n_ref", type=int, default=24,
                   help="Clips per mood averaged into the audio centroid")
    p.add_argument("--n_ref_pool", type=int, default=160,
                   help="Candidate songs scanned to fill the reference set")
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="Override cfg.cfg_scale (default: config value)")
    p.add_argument("--edit_strength", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
    cfg.edit_strength = args.edit_strength
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    if args.clap_ckpt:
        cfg.clap_ckpt = args.clap_ckpt
    sr, start_s, dur_s = cfg.sample_rate, cfg.clip_start_seconds, cfg.clip_seconds

    print(f"[PROBE] ckpt={args.ckpt_dir}  cfg_scale={cfg.cfg_scale}  "
          f"edit_strength={cfg.edit_strength}  moods={len(MOODS)}")

    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)
    clap = Clap(cfg.clap_ckpt)
    bigvgan = load_bigvgan(device=cfg.device)

    # Reference songs (for the audio centroids) must NOT overlap the test
    # songs, or the centroid has seen the very clip it will be used to edit.
    pool = pick_annotated_songs(args.audio_dir, va, args.n_ref_pool)
    test_files = pick_annotated_songs(args.audio_dir, va, args.n_songs)
    test_ids = {song_id_from_filename(f) for f in test_files}
    ref_files = [f for f in pool if song_id_from_filename(f) not in test_ids]
    print(f"[PROBE] {len(test_files)} test songs, {len(ref_files)} reference "
          f"candidates (disjoint)")

    models = load_models(cfg, args.ckpt_dir,
                         load_clip(test_files[0], sr, start_s, dur_s),
                         bigvgan, clap=clap)
    ae, dit, melody_enc, text_enc, diffusion, lat_mean, lat_std = models
    ck = torch.load(os.path.join(args.ckpt_dir, "diffusion.pt"),
                    map_location="cpu", weights_only=True)
    print(f"[PROBE] checkpoint alignment={ck.get('clap_align', 'none (pre-alignment)')}"
          f"  paraphrases={ck.get('clap_text_paraphrases', False)}")

    ref_raw = collect_reference_audio(clap, ref_files, va, sr, start_s, dur_s,
                                      args.n_ref)
    dev = text_enc.audio_mean.device
    with torch.no_grad():
        # Centroid in the space the model trained on (aligned audio).
        cents = {m: _l2(text_enc.align_audio(torch.from_numpy(v).to(dev).float())
                        .mean(0).cpu().numpy())
                 for m, v in ref_raw.items()}
        # Gap readout: how the caption sits relative to real audio, raw vs aligned.
        all_raw = torch.from_numpy(np.concatenate(list(ref_raw.values()))).float()
        labels = [m for m, v in ref_raw.items() for _ in range(len(v))]
        report = text_enc.alignment_report(
            all_raw, labels, torch.from_numpy(clap.text_emb).float(), MOODS)
    for m in MOODS:
        raw, aligned = report[m]
        print(f"[GAP] caption {m[:22]:22s} margin toward own audio centroid: "
              f"raw {raw:+.3f} -> aligned {aligned:+.3f}")

    def cond_for(arm, mood):
        """(vector, cond_space) fed to edit_mood for this arm/mood."""
        if arm == "text":
            return None, "text"                    # the real inference path
        if arm == "text_raw":
            return torch.from_numpy(clap.text_emb[MOODS.index(mood)]), "as_is"
        if arm == "audio":
            return torch.from_numpy(cents[mood]), "as_is"
        raise ValueError(arm)

    rows = []
    for si, path in enumerate(test_files):
        sid = song_id_from_filename(path)
        gt = mood_from_va(*va[sid])
        wav = load_clip(path, sr, start_s, dur_s)
        cos_orig = clap.cos_to_moods(clap.audio_embed(wav, sr))
        wav_t = torch.FloatTensor(wav).unsqueeze(0)

        for arm in ARMS:
            cos_by_target, song_rows = {}, []
            for mi, mood in enumerate(MOODS):
                # Identical noise across arms AND moods for a given song, so
                # any difference is caused purely by the conditioning vector.
                torch.manual_seed(args.seed + si)
                vec, space = cond_for(arm, mood)
                wav_e = edit_mood(
                    wav_t, mood, ae, dit, melody_enc, text_enc, diffusion,
                    bigvgan, cfg, lat_mean, lat_std,
                    cond_emb=vec, cond_space=space,
                ).squeeze().numpy()

                cos_edit = clap.cos_to_moods(clap.audio_embed(wav_e, sr))
                cos_by_target[mood] = cos_edit
                song_rows.append({
                    "song": os.path.basename(path), "song_id": sid,
                    "gt_mood": gt, "target_mood": mood, "arm": arm,
                    "clap_cos_original": round(float(cos_orig[mi]), 4),
                    "clap_cos_edited": round(float(cos_edit[mi]), 4),
                    "clap_gain": round(float(cos_edit[mi] - cos_orig[mi]), 4),
                    "clap_pred_edited": MOODS[int(np.argmax(cos_edit))],
                    "transfer_success": int(MOODS[int(np.argmax(cos_edit))] == mood),
                    "chroma_sim": round(chroma_similarity(wav, wav_e, sr), 4),
                })
            # Needs every target's edit of this song, hence computed afterward.
            for r in song_rows:
                mi = MOODS.index(r["target_mood"])
                sibling_mean = np.mean([cos_by_target[m][mi] for m in MOODS])
                r["cond_margin"] = round(
                    float(cos_by_target[r["target_mood"]][mi] - sibling_mean), 4)
            rows += song_rows
        print(f"  {os.path.basename(path)} (gt={gt}) done  [{si+1}/{len(test_files)}]")

    out_csv = os.path.join(args.ckpt_dir, "probe_conditioning.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[PROBE] Wrote {out_csv}")

    summarize(rows)


def _song_effect(rows, arm):
    """Mean cond_margin per song for one arm -> (mean, standard error, n).

    Songs are the independent unit: the moods of one song share its audio and
    noise, so treating each edit as independent would overstate confidence.
    """
    songs = sorted({r["song"] for r in rows})
    per_song = [np.mean([r["cond_margin"] for r in rows
                         if r["arm"] == arm and r["song"] == s]) for s in songs]
    n = len(per_song)
    se = float(np.std(per_song, ddof=1) / np.sqrt(n)) if n >= 2 else float("inf")
    return float(np.mean(per_song)), se, n


def _clearly_positive(mean, se, n):
    """Heuristic ~2-SE test. With < 3 songs nothing counts as established."""
    return n >= 3 and mean > 2 * se


def summarize(rows) -> str:
    """Print the probe summary table and verdict; returns the verdict key
    ("broken" | "inconclusive" | "gap" | "partial" | "closed")."""
    def agg(arm, key, mood=None):
        v = [r[key] for r in rows if r["arm"] == arm
             and (mood is None or r["target_mood"] == mood)]
        return float(np.mean(v)) if v else float("nan")

    def paired_correct(arm):
        """Songs where every mood's edit is classified as its own target."""
        songs = {r["song"] for r in rows}
        return sum(all(r["transfer_success"] for r in rows
                       if r["arm"] == arm and r["song"] == s) for s in songs)

    n = len({r["song"] for r in rows})
    eff = {arm: _song_effect(rows, arm) for arm in ARMS}
    print(f"\n{'='*80}\n PROBE SUMMARY  ({n} songs x {len(MOODS)} moods x "
          f"{len(ARMS)} arms)\n{'='*80}")
    print(f" {'arm':<10}{'cond_margin':>12}{'+/- 2SE':>9}{'margin>0':>10}"
          f"{'paired':>8}{'abs gain':>10}{'transfer%':>11}{'chroma':>8}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        pos = sum(1 for r in sub if r["cond_margin"] > 0)
        mean, se, _ = eff[arm]
        print(f" {arm:<10}{mean:>+12.4f}{2*se:>9.4f}{f'{pos}/{len(sub)}':>10}"
              f"{f'{paired_correct(arm)}/{n}':>8}{agg(arm,'clap_gain'):>+10.4f}"
              f"{100*agg(arm,'transfer_success'):>10.1f}%{agg(arm,'chroma_sim'):>8.3f}")

    print("\n per-mood cond_margin")
    print(f" {'arm':<10}" + "".join(f"{m[:16]:>18}" for m in MOODS))
    for arm in ARMS:
        print(f" {arm:<10}" + "".join(f"{agg(arm,'cond_margin',m):>+18.4f}"
                                      for m in MOODS))

    (m_text, se_text, _), (m_raw, se_raw, _), (m_audio, se_audio, _) = (
        eff[a] for a in ARMS)
    audio_pos = _clearly_positive(m_audio, se_audio, n)
    text_pos = _clearly_positive(m_text, se_text, n)
    gains = [agg(a, "clap_gain") for a in ARMS]
    print(f"\n{'='*80}\n VERDICT  (effect = per-song mean cond_margin, counted "
          f"only if > 2 SE; n={n} songs)\n{'='*80}")
    if not audio_pos and m_audio <= 0 and n >= 3:
        key = "broken"
        print(" Even the in-distribution audio vector does not push edits toward\n"
              " their target. The fault is in how conditioning reaches the latent\n"
              " (dit.py cross-attention / the projection), or mood was never\n"
              " learned. No embedding-space fix will help.")
    elif not audio_pos:
        key = "inconclusive"
        print(f" INCONCLUSIVE: the audio arm's effect ({m_audio:+.4f} +/- "
              f"{2*se_audio:.4f}) is not\n distinguishable from noise with {n} "
              f"songs. Re-run with more --n_songs\n before drawing any conclusion "
              f"about conditioning or the gap.")
    elif not text_pos:
        key = "gap"
        print(" Conditioning WORKS (audio arm clearly > 0) but the text path does\n"
              " not reliably reach it. The modality gap is still the blocker.")
    elif m_text >= 0.8 * m_audio:
        key = "closed"
        print(" The text path drives mood about as well as the in-distribution\n"
              " audio vector: the gap is effectively closed.")
    else:
        key = "partial"
        print(f" The text path works but recovers {100*m_text/m_audio:.0f}% of the\n"
              f" audio arm's effect — some gap remains.")
    if m_text != m_raw:
        print(f"\n Alignment effect on the caption: text_raw {m_raw:+.4f} -> "
              f"text {m_text:+.4f} ({m_text - m_raw:+.4f}).")
    if all(g < 0 for g in gains) and max(gains) - min(gains) < 0.05:
        print(f"\n Absolute gain is negative for every arm ({min(gains):+.3f} to "
              f"{max(gains):+.3f}) and nearly\n identical across them: that is "
              f"the autoencoder + vocoder resynthesis\n penalty, not mood. Read "
              f"cond_margin, not abs gain.")
    print("="*80)
    return key


if __name__ == "__main__":
    main()
