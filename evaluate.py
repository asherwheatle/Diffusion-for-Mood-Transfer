"""Objective evaluation of the mood-editing model.

Two independent judges, neither of which the model was trained against:

  1. CLAP (LAION music checkpoint) — an external audio<->text model. Measures
     whether an edit actually moved the audio toward the target mood text.
     Reported as:
       * clap_margin = cos(edit_m, m) - mean over targets m' of cos(edit_m', m)
                       THE PRIMARY METRIC. Every edit pays the same
                       autoencoder+vocoder resynthesis cost, which lowers CLAP
                       cosine to *any* caption; comparing a song's edits
                       against each other cancels it, leaving only what the
                       conditioning text changed.
       * clap_gain_recon = cos(edited, target) - cos(reconstructed, target),
                       where "reconstructed" is the same clip round-tripped
                       through the codec with no diffusion. Same correction,
                       measured directly rather than by cancellation.
       * clap_gain   = cos(edited, target) - cos(original, target). CONFOUNDED:
                       it sums the mood effect and the resynthesis cost, and
                       the cost is the larger term, so this is negative almost
                       regardless of how well conditioning works. Kept only so
                       older runs stay comparable — do not draw verdicts from it.
       * transfer    = does the target mood rank #1 among all mood prompts?

  2. Chroma similarity — cosine between the original and edited chroma (pitch-class)
     features. Measures whether the melody/harmony was preserved (the ControlNet
     claim). CLAP is blind to this, so the two together capture the real
     trade-off: mood changed AND tune kept.

Before trusting CLAP, we VALIDATE it: run it on the *original* DEAM clips against
their ground-truth valence mood labels. If CLAP can't tell the moods
apart on real audio (accuracy near the 50% chance line of the happy/sad
vocabulary), it can't judge edits either, and the CLAP numbers below should be
discarded.

Usage (on a GPU node, inside the venv):
  python evaluate.py \
      --ckpt_dir output/job_39423912 \
      --audio_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/MEMD_audio \
      --annotations_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt \
      --n_songs 20 --n_val 100 --edit_strength 0.6 --cfg_scale 5.0

Outputs (written to --ckpt_dir):
  eval_edits.csv        one row per (song, target_mood): CLAP gain, transfer, chroma
  clap_validation.csv   one row per validation song: gt mood vs CLAP-predicted mood
  eval_summary.txt      aggregated verdict
"""

import os
import csv
import glob
import argparse

from collections import Counter

import numpy as np
import torch
import librosa

from config import DiffusionConfig
from autoencoder import LatentAutoencoder
from dit import MoodDiT
from melody import MelodyEncoder, MelodyExtractor
from text_encoder import ClapTextEncoder
from diffusion import GaussianDiffusion
from pipeline import (load_bigvgan, bigvgan_mel_spectrogram, FixedMelNormalizer,
                      pad_spectrogram)
from inference import edit_mood, reconstruct
from annotations import (load_annotations, mood_from_va, song_id_from_filename,
                         MOOD_PROMPTS)
from valence_probe import (train_probe_from_clip_files, MOOD_VALENCE_SIGN)


# Mood prompts now live in annotations.py so training, inference and this
# evaluator all embed the identical string per mood.
MOODS = list(MOOD_PROMPTS.keys())

CLAP_SR = 48000  # LAION-CLAP expects 48 kHz mono


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_clip(path: str, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    """Load a fixed-length mono clip. Matches dataset._load_clip so the audio
    lines up with the 15-30 s window its VA annotation covers."""
    wav, _ = librosa.load(path, sr=sr, mono=True, offset=start_s, duration=dur_s)
    if len(wav) == 0:
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=dur_s)
    n = int(dur_s * sr)
    if len(wav) > n:
        wav = wav[:n]
    elif len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    return wav.astype(np.float32)


def chroma_similarity(wav_a: np.ndarray, wav_b: np.ndarray, sr: int) -> float:
    """Mean per-frame cosine similarity of chroma (pitch-class) features.
    ~1.0 = melody/harmony preserved, ~0 = unrelated. This is the melody-
    preservation axis CLAP cannot see."""
    ca = librosa.feature.chroma_cqt(y=wav_a, sr=sr)  # (12, T)
    cb = librosa.feature.chroma_cqt(y=wav_b, sr=sr)
    t = min(ca.shape[1], cb.shape[1])
    ca, cb = ca[:, :t], cb[:, :t]
    ca = ca / (np.linalg.norm(ca, axis=0, keepdims=True) + 1e-8)
    cb = cb / (np.linalg.norm(cb, axis=0, keepdims=True) + 1e-8)
    return float((ca * cb).sum(axis=0).mean())


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# CLAP wrapper
# ---------------------------------------------------------------------------
class Clap:
    def __init__(self, ckpt_path: str):
        try:
            import laion_clap
        except ImportError as e:
            raise ImportError(
                "laion_clap is not installed. Run:  uv pip install laion-clap\n"
                "and download the music checkpoint:\n"
                "  wget https://huggingface.co/lukewys/laion_clap/resolve/main/"
                "music_audioset_epoch_15_esc_90.14.pt"
            ) from e
        # The music checkpoint uses the HTSAT-base audio encoder, no fusion.
        self.model = laion_clap.CLAP_Module(enable_fusion=False,
                                            amodel="HTSAT-base")
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[CLAP] Loading music checkpoint: {ckpt_path}")
            self.model.load_ckpt(ckpt_path)
        else:
            print("[CLAP] WARNING: no music checkpoint given, loading the "
                  "default general-audio checkpoint (weaker at musical mood).")
            self.model.load_ckpt()
        self.model.eval()
        # Precompute the mood-prompt text embeddings once.
        self.text_emb = _l2(self.model.get_text_embedding(
            [MOOD_PROMPTS[m] for m in MOODS], use_tensor=False))  # (n_moods, D)

    @torch.no_grad()
    def audio_embed(self, wav_44k: np.ndarray, sr: int) -> np.ndarray:
        wav = librosa.resample(wav_44k, orig_sr=sr, target_sr=CLAP_SR)
        x = wav[None, :].astype(np.float32)  # (1, samples)
        emb = self.model.get_audio_embedding_from_data(x=x, use_tensor=False)
        return _l2(emb)[0]  # (D,)

    def cos_to_moods(self, audio_emb: np.ndarray) -> np.ndarray:
        """Cosine of one audio embedding against every mood prompt -> (n_moods,)."""
        return self.text_emb @ audio_emb


# ---------------------------------------------------------------------------
# Model loading (mirrors mood_diffusion.py edit mode)
# ---------------------------------------------------------------------------
def load_models(cfg: DiffusionConfig, ckpt_dir: str, sample_wav: np.ndarray,
                bigvgan_model, clap=None):
    device = cfg.device

    ae = LatentAutoencoder(cfg.ae_channels).to(device)
    ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "autoencoder.pt"),
                                  map_location=device, weights_only=True))
    ae.eval()

    # Infer latent shape from a real clip.
    normalizer = FixedMelNormalizer()
    mel = bigvgan_mel_spectrogram(
        torch.FloatTensor(sample_wav).unsqueeze(0), bigvgan_model)
    mel_norm = normalizer.normalize(mel)
    mel_padded, _ = pad_spectrogram(mel_norm.unsqueeze(0))
    with torch.no_grad():
        z = ae.encoder(mel_padded.to(device))
    _, C_lat, H_lat, W_lat = z.shape

    dit = MoodDiT(
        latent_channels=C_lat, latent_h=H_lat,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_blocks=cfg.n_dit_blocks, n_control_blocks=cfg.n_controlnet_blocks,
    ).to(device)
    melody_enc = MelodyEncoder(cfg.d_model, cfg.melody_top_k).to(device)
    # Reuse the already-loaded CLAP (its .model is the frozen text tower) so we
    # don't load a second copy; fall back to loading one if none was passed.
    text_enc = ClapTextEncoder(
        cfg.d_model, clap_model=(clap.model if clap is not None else None),
        clap_ckpt=cfg.clap_ckpt, n_tokens=cfg.text_n_tokens,
        device=device).to(device)

    ckpt = torch.load(os.path.join(ckpt_dir, "diffusion.pt"),
                      map_location=device, weights_only=True)
    dit.load_state_dict(ckpt["dit"])
    melody_enc.load_state_dict(ckpt["melody_enc"])
    text_enc.load_state_dict(ckpt["text_enc"])
    latent_mean = ckpt["latent_mean"].to(device)
    latent_std = ckpt["latent_std"].to(device)
    dit.eval(); melody_enc.eval(); text_enc.eval()

    diffusion = GaussianDiffusion(cfg.num_train_timesteps, device)
    return ae, dit, melody_enc, text_enc, diffusion, latent_mean, latent_std


def pick_annotated_songs(audio_dir: str, va: dict, n: int,
                         require_label: bool = True,
                         balanced: bool = False) -> list:
    """Return n song file paths (spread evenly) that have VA annotations.

    With require_label (the default), songs inside the valence dead band —
    which have no mood label, `mood_from_va` returns None — are excluded, so
    every mood-scoring stage can assume a usable ground truth. The valence
    probe passes require_label=False: it regresses on continuous valence and
    wants the middle of the range in its training set.

    With balanced, take n/len(MOODS) songs per ground-truth mood instead of
    spreading evenly over the whole corpus. DEAM leans positive (~70/30 even
    after the dead band), so the even spread inherits that skew — a 20-song
    pick came out 16 happy / 4 sad, which left the sad->happy edit direction
    resting on 4 songs. Edit evaluation wants both directions tested equally;
    CLAP validation and the probe do not, since they measure the judges
    against the corpus as it really is.
    """
    files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    files = [f for f in files if song_id_from_filename(f) in va]
    if require_label:
        files = [f for f in files
                 if mood_from_va(*va[song_id_from_filename(f)]) is not None]
    if not files:
        raise FileNotFoundError(f"No annotated MP3s found in {audio_dir}")

    def _spread(pool, k):
        if k >= len(pool):
            return list(pool)
        idxs = np.unique(np.linspace(0, len(pool) - 1, k).astype(int))
        return [pool[i] for i in idxs]

    if balanced and require_label:
        per_mood = max(1, n // len(MOODS))
        picked = []
        for m in MOODS:
            pool = [f for f in files
                    if mood_from_va(*va[song_id_from_filename(f)]) == m]
            got = _spread(pool, per_mood)
            if len(got) < per_mood:
                print(f"[PICK] WARNING: only {len(got)} songs available for "
                      f"'{m}', wanted {per_mood} — the split is not balanced.")
            picked += got
        return sorted(picked)

    return _spread(files, n)


# ---------------------------------------------------------------------------
# Stage 1: validate CLAP on original clips vs ground-truth mood
# ---------------------------------------------------------------------------
def validate_clap(clap: Clap, files: list, va: dict, sr: int,
                  start_s: float, dur_s: float, out_csv: str) -> float:
    print(f"\n{'='*60}\n CLAP VALIDATION on {len(files)} original clips\n{'='*60}")
    rows, correct = [], 0
    per_true = {m: [0, 0] for m in MOODS}  # mood -> [correct, total]
    for path in files:
        sid = song_id_from_filename(path)
        gt = mood_from_va(*va[sid])
        wav = load_clip(path, sr, start_s, dur_s)
        cos = clap.cos_to_moods(clap.audio_embed(wav, sr))
        pred = MOODS[int(np.argmax(cos))]
        ok = (pred == gt)
        correct += ok
        per_true[gt][0] += ok
        per_true[gt][1] += 1
        rows.append({"song": os.path.basename(path), "song_id": sid,
                     "gt_mood": gt, "clap_pred": pred, "correct": int(ok)})

    acc = correct / len(files)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["song", "song_id", "gt_mood",
                                          "clap_pred", "correct"])
        w.writeheader(); w.writerows(rows)

    print(f"[CLAP-VAL] Top-1 accuracy: {acc:.3f}  (chance = {1/len(MOODS):.3f})")
    for m in MOODS:
        c, tot = per_true[m]
        if tot:
            print(f"           {m:24s}: {c}/{tot} = {c/tot:.2f}")
    # Chance is 1/len(MOODS) — 0.5 for the two-mood vocabulary — so the bar
    # for "clearly above chance" has to move with the vocabulary size.
    verdict = ("LEGITIMATE — clearly above chance"
               if acc >= 1 / len(MOODS) + 0.15 else
               "WEAK — near chance, treat CLAP edit scores with caution")
    print(f"[CLAP-VAL] Verdict: {verdict}")
    print(f"[CLAP-VAL] Wrote {out_csv}")
    return acc


# ---------------------------------------------------------------------------
# Stage 1c: how far apart are REAL songs of each mood? (the yardstick)
# ---------------------------------------------------------------------------
def measure_mood_gap(clap: Clap, files: list, va: dict, sr: int,
                     start_s: float, dur_s: float) -> dict:
    """Mean mood-axis position of real clips of each mood.

    Only meaningful for a two-mood vocabulary, which is what this project
    uses. The axis is cos(x, MOODS[1]) - cos(x, MOODS[0]) — for the
    happy/sad split, "how much more this sounds sad than happy".

    The gap between the two means is the scale a mood EDIT should be read
    against: an edit that moves a song a tenth of the distance between real
    happy and real sad music has barely changed its mood, however reliably
    positive its margin is. Without this, margins have no units.
    """
    if len(MOODS) != 2:
        return {}
    print(f"\n{'='*60}\n MOOD-AXIS YARDSTICK on {len(files)} real clips\n{'='*60}")
    per_mood = {m: [] for m in MOODS}
    for path in files:
        gt = mood_from_va(*va[song_id_from_filename(path)])
        if gt not in per_mood:
            continue
        cos = clap.cos_to_moods(clap.audio_embed(
            load_clip(path, sr, start_s, dur_s), sr))
        per_mood[gt].append(float(cos[1] - cos[0]))
    out = {}
    for m, vals in per_mood.items():
        if vals:
            out[m] = (float(np.mean(vals)), float(np.std(vals, ddof=1))
                      if len(vals) > 1 else float("nan"), len(vals))
            print(f" {m:24s} n={out[m][2]:3d}  axis mean {out[m][0]:+.4f}  "
                  f"sd {out[m][1]:.4f}")
    if len(out) == 2:
        gap = out[MOODS[1]][0] - out[MOODS[0]][0]
        var = sum(out[m][1] ** 2 / out[m][2] for m in out)
        out["gap"] = (gap, float(np.sqrt(var)))
        print(f" REAL GAP between the two moods : {gap:+.4f} "
              f"+/- {2*np.sqrt(var):.4f} (2 SE)")
        print(" An edit's mood change is reported as a percentage of this.")
    return out


# ---------------------------------------------------------------------------
# Stage 2: edit N songs x every mood, score each edit
# ---------------------------------------------------------------------------
def evaluate_edits(cfg, clap, probe, files, va, sr, models, out_csv, seed=0):
    """Edit every song toward every mood and score each edit.

    Each edit is scored three ways, in increasing order of trustworthiness:

      *_original  vs the raw input clip. Confounded: the edit also paid the
                  autoencoder+vocoder resynthesis cost, which lowers CLAP
                  cosine to every caption because CLAP was fit on real audio.
                  Measured at -0.118 on job_42463485, i.e. nearly all of the
                  -0.133 "gain" that run reported.
      *_recon     vs the same clip round-tripped through the codec with no
                  diffusion (inference.reconstruct). Subtracts that cost.
                  NOTE the codec's effect on the two judges is NOT the same
                  sign: measured on job_42463485 it moved CLAP cosine -0.118
                  but valence +0.088. So a common-mode valence shift cannot be
                  written off as resynthesis — whatever the recon control does
                  not explain belongs to the diffusion stage, which unlike the
                  codec regenerates the latent instead of round-tripping it.
      *_margin    vs this song's *other* mood edits. Cancels everything the
                  edits share — codec, song, and noise — leaving only what the
                  conditioning text changed. This is the primary metric, and
                  it mirrors probe_conditioning.cond_margin.

    Sampling noise is fixed per song (the seed is reset before every mood), so
    within a song the only thing that differs between edits is the text.
    """
    ae, dit, melody_enc, text_enc, diffusion, lat_mean, lat_std = models
    fieldnames = ["song", "song_id", "gt_mood", "target_mood",
                  "clap_cos_original", "clap_cos_recon", "clap_cos_edited",
                  "clap_gain", "clap_gain_recon", "clap_margin",
                  "clap_pred_edited", "transfer_success",
                  "chroma_sim", "chroma_recon",
                  "valence_original", "valence_recon", "valence_edited",
                  "valence_shift", "valence_shift_recon", "valence_margin",
                  "valence_target_sign", "valence_correct_dir",
                  "valence_correct_dir_margin",
                  "cfg_scale", "edit_strength"]
    # Full cosine vector of each edit, one column per mood. Storing these
    # (not just the cosine toward the edit's own target) makes the mood-axis
    # separation directly computable from the CSV afterwards.
    cos_cols = [f"cos_to_{m.split()[0]}" for m in MOODS]
    fieldnames += cos_cols
    rows = []
    print(f"\n{'='*60}\n EDIT EVALUATION: {len(files)} songs x {len(MOODS)} "
          f"moods = {len(files)*len(MOODS)} edits\n{'='*60}")

    for si, path in enumerate(files):
        sid = song_id_from_filename(path)
        gt = mood_from_va(*va[sid])
        wav_orig = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        waveform = torch.FloatTensor(wav_orig).unsqueeze(0)

        orig_emb = clap.audio_embed(wav_orig, sr)
        orig_cos = clap.cos_to_moods(orig_emb)  # (n_moods,)
        v_orig = probe.predict(orig_emb)

        # Reconstruction control: the codec cost with no mood edit at all.
        wav_recon = reconstruct(waveform, ae, _bigvgan, cfg).squeeze(0).numpy()
        recon_emb = clap.audio_embed(wav_recon, sr)
        recon_cos = clap.cos_to_moods(recon_emb)
        v_recon = probe.predict(recon_emb)
        chroma_recon = chroma_similarity(wav_orig, wav_recon, sr)

        song_seed = seed + si
        song_rows, cos_by_target, v_by_target = [], {}, {}
        for target in MOODS:
            # SAME noise for every mood of this song => only the text differs.
            torch.manual_seed(song_seed)
            wav_edit = edit_mood(
                waveform, target, ae, dit, melody_enc, text_enc, diffusion,
                _bigvgan, cfg, lat_mean, lat_std,
            ).squeeze(0).numpy()

            edit_emb = clap.audio_embed(wav_edit, sr)
            edit_cos = clap.cos_to_moods(edit_emb)
            ti = MOODS.index(target)
            pred = MOODS[int(np.argmax(edit_cos))]

            v_edit = probe.predict(edit_emb)
            v_shift = v_edit - v_orig
            desired = MOOD_VALENCE_SIGN[target]
            cos_by_target[target] = edit_cos
            v_by_target[target] = v_edit
            song_rows.append({
                "song": os.path.basename(path), "song_id": sid, "gt_mood": gt,
                "target_mood": target,
                "clap_cos_original": round(float(orig_cos[ti]), 4),
                "clap_cos_recon": round(float(recon_cos[ti]), 4),
                "clap_cos_edited": round(float(edit_cos[ti]), 4),
                "clap_gain": round(float(edit_cos[ti] - orig_cos[ti]), 4),
                "clap_gain_recon": round(float(edit_cos[ti] - recon_cos[ti]), 4),
                "clap_pred_edited": pred,
                "transfer_success": int(pred == target),
                "chroma_sim": round(chroma_similarity(wav_orig, wav_edit, sr), 4),
                "chroma_recon": round(float(chroma_recon), 4),
                "valence_original": round(float(v_orig), 4),
                "valence_recon": round(float(v_recon), 4),
                "valence_edited": round(float(v_edit), 4),
                "valence_shift": round(float(v_shift), 4),
                "valence_shift_recon": round(float(v_edit - v_recon), 4),
                "valence_target_sign": desired,
                # did valence move in the target's intended direction?
                "valence_correct_dir": int(v_shift * desired > 0),
                "cfg_scale": cfg.cfg_scale,
                "edit_strength": cfg.edit_strength,
                **{c: round(float(edit_cos[j]), 4)
                   for j, c in enumerate(cos_cols)},
            })

        # Margins need every target's edit of this song, hence computed here.
        # Subtracting the sibling mean removes the codec cost, the song, and
        # the noise, all of which are shared across this song's edits.
        v_sibling_mean = float(np.mean([v_by_target[m] for m in MOODS]))
        for r in song_rows:
            ti = MOODS.index(r["target_mood"])
            cos_sibling_mean = float(np.mean([cos_by_target[m][ti]
                                              for m in MOODS]))
            r["clap_margin"] = round(
                float(cos_by_target[r["target_mood"]][ti] - cos_sibling_mean), 4)
            v_margin = v_by_target[r["target_mood"]] - v_sibling_mean
            r["valence_margin"] = round(float(v_margin), 4)
            r["valence_correct_dir_margin"] = int(
                v_margin * r["valence_target_sign"] > 0)
        rows += song_rows
        print(f"  {os.path.basename(path)} (gt={gt}) done  [{si+1}/{len(files)}]")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"[EDIT] Wrote {out_csv}")
    return rows


def _per_song_effect(rows, key, signed=False):
    """Mean of `key` per song, then across songs -> (mean, standard error, n).

    Songs are the independent unit: a song's mood edits share its audio and
    its sampling noise, so treating each edit as independent would overstate
    confidence. `signed` multiplies by valence_target_sign first, turning a
    raw shift into a shift *toward the intended direction*.
    """
    songs = sorted({r["song"] for r in rows})
    per_song = []
    for s in songs:
        vals = [r[key] * (r["valence_target_sign"] if signed else 1)
                for r in rows if r["song"] == s]
        if vals:
            per_song.append(float(np.mean(vals)))
    n = len(per_song)
    if n == 0:
        return float("nan"), float("inf"), 0
    se = float(np.std(per_song, ddof=1) / np.sqrt(n)) if n >= 2 else float("inf")
    return float(np.mean(per_song)), se, n


def _axis_separation(rows):
    """Per song: how far apart its two edits land on the mood axis.

    axis(x) = cos(x, MOODS[1]) - cos(x, MOODS[0]); separation is
    axis(edit toward MOODS[1]) - axis(edit toward MOODS[0]). Same form as
    the real-song gap from measure_mood_gap, so the two are comparable and
    their ratio answers "how much of a real mood difference did we get?".
    """
    if len(MOODS) != 2:
        return None
    lo, hi = (f"cos_to_{m.split()[0]}" for m in MOODS)
    by_song = {}
    for r in rows:
        by_song.setdefault(r["song"], {})[r["target_mood"]] = r
    seps = []
    for song, d in by_song.items():
        if len(d) != 2:
            continue
        a0, a1 = (d[m][hi] - d[m][lo] for m in MOODS)
        seps.append(a1 - a0)
    if not seps:
        return None
    n = len(seps)
    se = float(np.std(seps, ddof=1) / np.sqrt(n)) if n >= 2 else float("inf")
    return float(np.mean(seps)), se, n


def _transfer_split(rows):
    """Transfer success split by whether the edit was asked to CHANGE mood.

    An edit whose target equals the song's own mood needs no change at all,
    so counting it as a transfer success inflates the headline. Only the
    cross-mood edits test the conditioning.
    """
    cross = [r for r in rows if r["target_mood"] != r["gt_mood"]]
    same = [r for r in rows if r["target_mood"] == r["gt_mood"]]
    f = lambda rs: (100 * float(np.mean([r["transfer_success"] for r in rs]))
                    if rs else float("nan"), len(rs))
    return f(cross), f(same)


def summarize(rows, clap_acc, probe, out_txt, mood_gap=None):
    pm = probe.metrics
    chance = 1 / len(MOODS)
    lines = ["=" * 72, " EVALUATION SUMMARY", "=" * 72,
             f" cfg_scale {rows[0]['cfg_scale']}   "
             f"edit_strength {rows[0]['edit_strength']}",
             f" CLAP validation top-1 accuracy : {clap_acc:.3f} "
             f"(chance {chance:.3f})",
             f" Valence probe held-out R^2     : {pm.get('heldout_r2', float('nan')):.3f} "
             f"(pearson {pm.get('heldout_pearson', float('nan')):.3f}, "
             f"n={int(pm.get('n_total', 0))})", ""]

    # ---- The codec floor, measured rather than assumed ----
    codec_clap = np.mean([r["clap_cos_recon"] - r["clap_cos_original"]
                          for r in rows])
    codec_val = np.mean([r["valence_recon"] - r["valence_original"]
                         for r in rows])
    lines += [" RESYNTHESIS FLOOR (autoencoder + vocoder, no diffusion)",
              f"   CLAP cosine cost   : {codec_clap:+.4f}  "
              f"(paid by every edit, toward every caption)",
              f"   valence probe cost : {codec_val:+.4f}",
              f"   chroma ceiling     : "
              f"{np.mean([r['chroma_recon'] for r in rows]):.3f}  "
              f"(recon vs original: no edit can score above this)",
              " Any metric measured against the RAW input inherits these.", ""]

    # ---- Primary: margins (codec/song/noise all cancelled) ----
    lines.append(" PRIMARY (conditioning-only: edit vs this song's other edits)")
    lines.append(f" {'target mood':24s} {'clap.margin':>12s} "
                 f"{'val.margin':>11s} {'val.dir%':>9s} {'transfer%':>10s} "
                 f"{'chroma':>8s}")
    for m in MOODS:
        sub = [r for r in rows if r["target_mood"] == m]
        if not sub:
            continue
        lines.append(
            f" {m:24s} {np.mean([r['clap_margin'] for r in sub]):>+12.4f} "
            f"{np.mean([r['valence_margin'] * r['valence_target_sign'] for r in sub]):>+11.4f} "
            f"{100*np.mean([r['valence_correct_dir_margin'] for r in sub]):>8.1f}% "
            f"{100*np.mean([r['transfer_success'] for r in sub]):>9.1f}% "
            f"{np.mean([r['chroma_sim'] for r in sub]):>8.3f}")

    cm, cm_se, n_songs = _per_song_effect(rows, "clap_margin")
    vm, vm_se, _ = _per_song_effect(rows, "valence_margin", signed=True)
    gr, gr_se, _ = _per_song_effect(rows, "clap_gain_recon")
    tr = 100 * np.mean([r["transfer_success"] for r in rows])

    def verdict(mean, se):
        if not np.isfinite(se):
            return "too few songs to say"
        if mean > 2 * se:
            return "POSITIVE (> 2 SE)"
        if mean < -2 * se:
            return "NEGATIVE (> 2 SE)"
        return "not distinguishable from zero"

    # ---- How BIG is the mood change, in units of a real mood difference? ----
    sep = _axis_separation(rows)
    if sep is not None:
        s_mean, s_se, s_n = sep
        lines += ["", " MOOD CHANGE MAGNITUDE (the 'is this a major change?' test)",
                  f"   a song's two edits land  : {s_mean:+.4f} +/- {2*s_se:.4f}"
                  f" apart on the mood axis (n={s_n})"]
        if mood_gap and "gap" in mood_gap:
            g, g_se = mood_gap["gap"]
            pct = 100 * s_mean / g if g else float("nan")
            lo = 100 * (s_mean - 2*s_se) / (g + 2*g_se) if g else float("nan")
            hi = 100 * (s_mean + 2*s_se) / max(g - 2*g_se, 1e-6) if g else float("nan")
            lines += [f"   real happy-vs-sad gap    : {g:+.4f} +/- {2*g_se:.4f}",
                      f"   => EDIT REACHES {pct:.0f}% OF A REAL MOOD DIFFERENCE"
                      f"  (range {lo:.0f}-{hi:.0f}%)",
                      "   100% = the two edits differ as much as real happy and",
                      "   real sad music do. Well under that means the mood was",
                      "   nudged, not changed, however positive the margin is."]
        else:
            lines.append("   (no real-song yardstick measured; run with "
                         "--n_gap > 0 to put this in units)")

    # ---- Transfer, split by whether a change was actually asked for ----
    (tr_cross, n_cross), (tr_same, n_same) = _transfer_split(rows)
    lines += ["",
              f" Transfer, edits asked to CHANGE mood : {tr_cross:.1f}% "
              f"(n={n_cross})  <- the real test",
              f" Transfer, edits asked to KEEP   mood : {tr_same:.1f}% "
              f"(n={n_same})  <- preservation, not transfer"]

    lines += ["",
              f" Overall CLAP margin        : {cm:+.4f} +/- {2*cm_se:.4f} (2 SE, "
              f"n={n_songs} songs)  {verdict(cm, cm_se)}",
              f" Overall valence margin(dir): {vm:+.4f} +/- {2*vm_se:.4f} (2 SE)"
              f"  {verdict(vm, vm_se)}",
              f" Overall gain vs recon      : {gr:+.4f} +/- {2*gr_se:.4f} (2 SE)"
              f"  {verdict(gr, gr_se)}",
              f" Overall transfer success   : {tr:.1f}%  (chance {100*chance:.1f}%)",
              f" Overall chroma preserved   : {np.mean([r['chroma_sim'] for r in rows]):.3f}",
              ""]

    # ---- Confounded, kept only so old runs stay comparable ----
    raw_gain = np.mean([r["clap_gain"] for r in rows])
    raw_vshift = np.mean([r["valence_shift"] * r["valence_target_sign"]
                          for r in rows])
    raw_vbias = np.mean([r["valence_shift"] for r in rows])
    # Split the prompt-independent (common-mode) valence shift into the part
    # the codec explains and the part left over. The leftover is the diffusion
    # stage: SDEdit regenerates the latent rather than round-tripping it, so
    # unlike the codec term this one IS a property of the model. Do not
    # attribute the whole common-mode shift to resynthesis without checking
    # this split — on job_42463485 the codec term came out POSITIVE.
    codec_part = codec_val
    diffusion_part = raw_vbias - codec_val
    lines += [" CONFOUNDED (vs raw input; includes the floor above —"
              " for back-comparison only)",
              f"   mean CLAP gain           : {raw_gain:+.4f}",
              f"   valence shift(dir)       : {raw_vshift:+.4f}",
              f"   valence shift(unsigned)  : {raw_vbias:+.4f}  <- common-mode,"
              f" prompt-independent",
              f"     of which codec         : {codec_part:+.4f}",
              f"     of which diffusion     : {diffusion_part:+.4f}  <- a real"
              f" property of the model",
              f"   valence dir correct      : "
              f"{100*np.mean([r['valence_correct_dir'] for r in rows]):.1f}%",
              "",
              " Read: clap.margin / val.margin > 0 => the TEXT moved the audio",
              " toward its target. These cancel the resynthesis floor, so unlike",
              " raw gain they can legitimately be positive. chroma near 1 => melody",
              " kept. Trust valence only if the probe's held-out R^2 is well over 0.",
              "=" * 72]
    text = "\n".join(lines)
    print("\n" + text)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    print(f"[SUMMARY] Wrote {out_txt}")


# ---------------------------------------------------------------------------
_bigvgan = None  # module-global so edit_mood gets the loaded vocoder


def main():
    global _bigvgan
    p = argparse.ArgumentParser(description="CLAP + chroma evaluation of mood edits")
    p.add_argument("--ckpt_dir", required=True,
                   help="Dir with autoencoder.pt + diffusion.pt (the job folder)")
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None,
                   help="Path to music_audioset_epoch_15_esc_90.pt")
    p.add_argument("--n_songs", type=int, default=20, help="Songs to edit")
    p.add_argument("--n_val", type=int, default=100,
                   help="Songs for CLAP validation (cheap, no editing)")
    p.add_argument("--n_probe", type=int, default=500,
                   help="Annotated songs to fit the valence probe on "
                        "(cheap: CLAP-embed only, no editing)")
    p.add_argument("--n_gap", type=int, default=80,
                   help="Real songs (balanced across moods) used to measure "
                        "the happy-vs-sad gap that edit magnitude is reported "
                        "as a percentage of. 0 disables. Cheap: embed only.")
    p.add_argument("--unbalanced_songs", action="store_true",
                   help="Pick edit songs by even spread over the corpus "
                        "(the old behaviour, which inherits DEAM's ~70/30 "
                        "skew) instead of balancing across moods.")
    p.add_argument("--edit_strength", type=float, default=0.5)
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="Override cfg.cfg_scale (default: the config value). "
                        "The config default is tuned for training-time "
                        "sampling; diagnostics run at 5.0+, so leaving this "
                        "unset evaluates at far weaker guidance than the "
                        "conditioning probes use.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DiffusionConfig()
    cfg.edit_strength = args.edit_strength
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    print(f"[EVAL] cfg_scale={cfg.cfg_scale}  edit_strength={cfg.edit_strength}")

    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)

    print("[STEP] Loading BigVGAN...")
    _bigvgan = load_bigvgan(cfg.device)
    sr = _bigvgan.h.sampling_rate

    print("[STEP] Loading CLAP...")
    clap = Clap(args.clap_ckpt)

    # Stage 1: validate CLAP (cheap — more songs for a better estimate)
    val_files = pick_annotated_songs(args.audio_dir, va, args.n_val)
    clap_acc = validate_clap(clap, val_files, va, sr, cfg.clip_start_seconds,
                             cfg.clip_seconds,
                             os.path.join(args.ckpt_dir, "clap_validation.csv"))

    # Stage 1b: fit the continuous valence probe on frozen CLAP embeddings
    # (cheap — embedding only, no editing). Cached to the ckpt dir.
    probe_files = pick_annotated_songs(args.audio_dir, va, args.n_probe,
                                       require_label=False)
    probe = train_probe_from_clip_files(
        clap, probe_files, va, sr, cfg.clip_start_seconds, cfg.clip_seconds,
        load_clip, song_id_from_filename,
        cache_path=os.path.join(args.ckpt_dir, "valence_probe.npz"))

    # Stage 1c: the yardstick — how far apart are REAL songs of each mood?
    mood_gap = {}
    if args.n_gap > 0 and len(MOODS) == 2:
        gap_files = pick_annotated_songs(args.audio_dir, va, args.n_gap,
                                         balanced=True)
        mood_gap = measure_mood_gap(clap, gap_files, va, sr,
                                    cfg.clip_start_seconds, cfg.clip_seconds)

    # Stage 2: load the model and evaluate edits. Balanced by default so both
    # edit directions get equal weight; DEAM's skew otherwise leaves the
    # minority->majority direction resting on a handful of songs.
    edit_files = pick_annotated_songs(args.audio_dir, va, args.n_songs,
                                      balanced=not args.unbalanced_songs)
    gt_counts = Counter(mood_from_va(*va[song_id_from_filename(f)])
                        for f in edit_files)
    print(f"[EVAL] Edit songs by ground-truth mood: {dict(gt_counts)}")
    sample = load_clip(edit_files[0], sr, cfg.clip_start_seconds, cfg.clip_seconds)
    print("[STEP] Loading mood-diffusion checkpoints...")
    models = load_models(cfg, args.ckpt_dir, sample, _bigvgan, clap=clap)

    rows = evaluate_edits(cfg, clap, probe, edit_files, va, sr, models,
                          os.path.join(args.ckpt_dir, "eval_edits.csv"),
                          seed=args.seed)
    summarize(rows, clap_acc, probe,
              os.path.join(args.ckpt_dir, "eval_summary.txt"),
              mood_gap=mood_gap)


if __name__ == "__main__":
    main()
