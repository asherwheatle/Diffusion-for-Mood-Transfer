"""Training loops for the latent autoencoder and diffusion model.

Data tensors (mels, latents, melodies) stay on CPU so the full DEAM set fits
in memory; only the active minibatch is moved to the GPU. Batches are served
through a pinned-memory DataLoader and copied with non_blocking=True so the
host->device transfer overlaps compute instead of stalling each step.
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import DiffusionConfig
from mood_paraphrases import paraphrases_for
from autoencoder import LatentAutoencoder
from text_encoder import ClapTextEncoder
from dit import MoodDiT
from melody import MelodyEncoder
from diffusion import GaussianDiffusion
from pipeline import pad_spectrogram


def _atomic_save(obj, path: str):
    """Write to a temp file then rename, so an eviction mid-write can never
    leave a truncated (unloadable) checkpoint at `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _mel_recon_loss(recon: torch.Tensor, target: torch.Tensor,
                    cfg: DiffusionConfig) -> torch.Tensor:
    """Sharpness-preserving reconstruction loss for mel autoencoding.

    Pure MSE rewards blur — it minimizes average error by smoothing — and
    BigVGAN turns a blurred mel into distorted audio. We replace it with:
      * L1        — sharper base term than MSE
      * gradient  — match time/freq derivatives, so harmonics and onsets are
                    preserved instead of averaged away (directly counters the
                    lost mel-gradient energy we measured in reconstructions)
      * multiscale— L1 at coarser resolutions to keep global structure
    This is the mel-space analog of a multi-resolution STFT loss; the AE never
    sees waveforms, so the terms are applied on the spectrogram itself.
    """
    l1 = F.l1_loss(recon, target)

    # gradient-difference (edge) loss along time and frequency
    grad = (F.l1_loss(recon[..., 1:] - recon[..., :-1],
                      target[..., 1:] - target[..., :-1]) +
            F.l1_loss(recon[..., 1:, :] - recon[..., :-1, :],
                      target[..., 1:, :] - target[..., :-1, :]))

    # multi-scale L1: compare at 1/2, 1/4, 1/8 resolution
    ms = recon.new_zeros(())
    r, t = recon, target
    for _ in range(3):
        r = F.avg_pool2d(r, 2)
        t = F.avg_pool2d(t, 2)
        ms = ms + F.l1_loss(r, t)

    return (getattr(cfg, "ae_l1_weight", 1.0) * l1 +
            getattr(cfg, "ae_grad_weight", 1.0) * grad +
            getattr(cfg, "ae_ms_weight", 0.5) * ms)


def train_autoencoder(mel_batch: torch.Tensor, cfg: DiffusionConfig):
    """
    Train the latent autoencoder on a batch of normalized mel spectrograms.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per clip (CPU)

    Preserves spatial structure (no flatten bottleneck) so the latent
    is suitable for 2D diffusion.

    Returns:
        ae: trained LatentAutoencoder
        orig_hw: (H, W) before padding, needed for unpadding later
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_batch)
    n_songs = mel_padded.shape[0]

    ae = LatentAutoencoder(cfg.ae_channels).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    # Resume from an interrupted run if a checkpoint is present, so an
    # eviction doesn't cost every completed epoch.
    ckpt_path = os.path.join(cfg.output_dir, "autoencoder_ckpt.pt")
    start_epoch = 1
    if getattr(cfg, "resume", True) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        orig_hw = ck.get("orig_hw", orig_hw)
        print(f"[RESUME] Autoencoder resumed from epoch {ck['epoch']} "
              f"-> continuing at {start_epoch} ({ckpt_path})")

    # DataLoader with pinned memory so H2D copies overlap compute and kill the
    # per-batch copy stall; workers prefetch/collate the next batch off the
    # training thread.
    dataset = TensorDataset(mel_padded)
    num_workers = getattr(cfg, "num_workers", 2)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"\n{'='*60}")
    print(f" Training Latent Autoencoder ({cfg.ae_epochs} epochs)")
    print(f" Input: {tuple(mel_padded.shape)} ({n_songs} songs)  Device: {device}")
    print(f" Batch size: {cfg.batch_size}")
    print(f"{'='*60}")

    ckpt_interval = getattr(cfg, "ae_ckpt_interval", cfg.ae_epochs)

    ae.train()
    for epoch in tqdm(range(start_epoch, cfg.ae_epochs + 1), desc="Autoencoder",
                      initial=start_epoch - 1, total=cfg.ae_epochs):
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon, z = ae(batch)
            if recon.shape != batch.shape:
                recon = F.interpolate(recon, size=batch.shape[2:],
                                      mode="bilinear", align_corners=False)
            loss = _mel_recon_loss(recon, batch, cfg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | recon loss: {epoch_loss / max(n_batches, 1):.6f}")

        if epoch % ckpt_interval == 0 or epoch == cfg.ae_epochs:
            # Resumable checkpoint (model + optimizer + epoch) ...
            _atomic_save({"model": ae.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "orig_hw": orig_hw}, ckpt_path)
            # ... plus the inference-ready copy the rest of the code loads,
            # so even an interrupted run leaves a usable (if under-trained) AE.
            _atomic_save(ae.state_dict(),
                         os.path.join(cfg.output_dir, "autoencoder.pt"))

    ae.eval()
    return ae, orig_hw


@torch.no_grad()
def _encode_latents(ae: LatentAutoencoder, mel_padded: torch.Tensor,
                    device: str, chunk: int = 32) -> torch.Tensor:
    """Encode all mels to latents in chunks; result stays on CPU."""
    zs = []
    for i in tqdm(range(0, mel_padded.shape[0], chunk), desc="Encoding latents"):
        zs.append(ae.encoder(mel_padded[i:i + chunk].to(device)).cpu())
    return torch.cat(zs, dim=0)


def train_diffusion(ae: LatentAutoencoder, mel_batch: torch.Tensor,
                    melody_all: torch.Tensor, mood_texts: list[str],
                    cfg: DiffusionConfig, clap_audio: torch.Tensor = None,
                    clap_model=None):
    """
    Train the DiT + ControlNet diffusion model in the autoencoder's latent space.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per clip (CPU)
        melody_all: (N, top_k, T_cqt) precomputed melody pitch indices (CPU)
        mood_texts: list of N mood strings, one per clip
        clap_audio: (N, clap_dim) per-clip CLAP audio embeddings. Required
            when cfg.clap_cond_source == "audio"; ignored otherwise.
        clap_model: an already-loaded CLAP module to reuse (avoids a second
            ~2 GB load when the caller built one for the dataset pass).

    Each step:
      1. Sample a minibatch of songs' latents z0
      2. Sample random timesteps, add noise -> z_t
      3. Predict v conditioned on each song's text + melody
      4. MSE loss against v-target

    CFG dropout randomly drops text conditioning with probability cfg_dropout
    to enable classifier-free guidance at inference.

    Returns:
        dit, melody_enc, text_enc, diffusion, (latent_mean, latent_std)
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_batch)
    n_songs = mel_padded.shape[0]

    ae.eval()
    z0_all = _encode_latents(ae, mel_padded, device)
    del mel_padded   # free the padded mel copy; only latents are needed now
    # Standardize latents so the diffusion's unit-variance noise assumption
    # holds (the encoder's GroupNorm+SiLU output is skewed, std != 1).
    #
    # PER-CHANNEL, not one global scalar. Measured on the job_42891316 AE the
    # raw per-channel std spans 0.031-0.644 (21x). Dividing all 32 channels by
    # a single sigma left them at std 0.095-1.995, and diffusion then adds
    # unit-variance noise to every one of them. Counting channels whose SNR at
    # the SDEdit start is below 1 (input content already destroyed, regenerated
    # from the prior) against those above it (pinned to the input):
    #
    #                    t=350 (strength .35)   t=600 (strength .6)
    #     global          21 gone / 11 kept      28 gone /  4 kept
    #     per_channel      0 gone / 32 kept      32 gone /  0 kept
    #
    # So under one global sigma, EVERY edit strength was a blend of regenerate
    # and preserve, and turning the knob only shifted the mix 21:11 -> 28:4.
    # That is why melody preservation and mood transfer were both mediocre at
    # the same time instead of trading against each other: different channels
    # sat at opposite ends of the trade no matter what strength was asked for.
    # Per-channel makes the latent act as one unit, so edit_strength finally
    # means one thing — at the cost of a sharper transition, since all 32
    # channels now cross together somewhere between 0.35 and 0.6.
    #
    # Four channels landed at std 0.10, crossing SNR=1 at t=55 (6% of the
    # schedule) — noise to the DiT almost always. Per-channel also removes a DC
    # offset of up to 1.279 in units of the noise (post-norm per-channel mean
    # goes to exactly 0), which the DiT otherwise spends capacity representing.
    #
    # Measured effect on the latent's shape: skew 2.55 -> 1.20, excess kurtosis
    # 13.6 -> 3.5. The remainder is the encoder's final GroupNorm+SiLU (a hard
    # floor at -0.2785 with 3.4% of mass pinned to it); only an autoencoder
    # change removes that, and it needs an AE retrain.
    #
    # Old checkpoints hold scalar latent_mean/latent_std. Nothing breaks: every
    # consumer only broadcasts them, so a (1,C,1,1) stat and a scalar are
    # interchangeable at load time.
    norm_mode = getattr(cfg, "latent_norm", "per_channel")
    if norm_mode == "per_channel":
        latent_mean = z0_all.mean(dim=(0, 2, 3), keepdim=True)   # (1, C, 1, 1)
        latent_std = z0_all.std(dim=(0, 2, 3), keepdim=True)
    elif norm_mode == "global":
        latent_mean = z0_all.mean()
        latent_std = z0_all.std()
    else:
        raise ValueError(f"unknown cfg.latent_norm {norm_mode!r} "
                         f"(expected 'per_channel' or 'global')")
    # A near-dead channel must never become a division by ~0.
    latent_std = latent_std.clamp_min(1e-6)
    z0_all = (z0_all - latent_mean) / latent_std
    # Report the spread, not just a single number: the whole point of the change
    # is that one number was hiding a 21x range. Post-norm std should be ~1 for
    # every channel — if it is not, the stats and the data disagree.
    post = z0_all.std(dim=(0, 2, 3))
    print(f"[DIFF] Latents: {tuple(z0_all.shape)}  latent_norm={norm_mode}")
    print(f"[DIFF]   pre-norm  mean [{latent_mean.min():+.4f}, {latent_mean.max():+.4f}]  "
          f"std [{latent_std.min():.4f}, {latent_std.max():.4f}] "
          f"({latent_std.max() / latent_std.min():.1f}x spread)")
    print(f"[DIFF]   post-norm per-channel std [{post.min():.4f}, {post.max():.4f}] "
          f"(want ~1.00 for all {post.numel()})")
    print(f"[DIFF] Melody shape: {tuple(melody_all.shape)}")

    _, C_lat, H_lat, W_lat = z0_all.shape
    dit = MoodDiT(
        latent_channels=C_lat, latent_h=H_lat,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_blocks=cfg.n_dit_blocks, n_control_blocks=cfg.n_controlnet_blocks,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
    ).to(device)

    melody_enc = MelodyEncoder(d_model=cfg.d_model, top_k=cfg.melody_top_k).to(device)
    # Frozen CLAP text tower + trainable projection (Lever A). Only the
    # projection is optimized; CLAP is frozen and off the training hot path.
    text_enc = ClapTextEncoder(cfg.d_model, clap_model=clap_model,
                               clap_ckpt=cfg.clap_ckpt,
                               n_tokens=cfg.text_n_tokens, device=device).to(device)
    diffusion = GaussianDiffusion(cfg.num_train_timesteps, device)

    # Null (CFG) embedding: deliberately left unaligned — inference builds it
    # the same way, which is the only thing that matters for it.
    null_clap_emb = text_enc.encode([""])[0].to(device)       # (clap_dim,)

    # What each clip is conditioned on. "audio" gives every clip its own CLAP
    # audio embedding; "text" conditions every step on a mood prompt, which
    # (without paraphrases) collapses the whole set onto 2 vectors and lets
    # the projection degenerate into a lookup table.
    cond_source = getattr(cfg, "clap_cond_source", "text")
    use_paraphrases = getattr(cfg, "clap_text_paraphrases", False)
    align_mode = getattr(cfg, "clap_align", "none")

    # Training prompts: every paraphrase of every mood, laid out contiguously
    # per mood, so "a random prompt of mood k" is
    # para_start[k] + floor(U[0,1) * para_count[k]). Sampling happens per
    # step, so the text path sees a distribution per mood, not a single point.
    uniq_moods = sorted(set(mood_texts))
    mood_row = torch.tensor([uniq_moods.index(m) for m in mood_texts])  # (N,)
    prompts, prompt_labels, para_start, para_count = [], [], [], []
    for m in uniq_moods:
        ps = paraphrases_for(m, use_paraphrases)
        para_start.append(len(prompts))
        para_count.append(len(ps))
        prompts += ps
        prompt_labels += [m] * len(ps)
    para_start = torch.tensor(para_start, device=device)
    para_count = torch.tensor(para_count, device=device)
    prompt_raw = text_enc.encode(prompts)                      # (P, clap_dim)

    if cond_source == "audio":
        if clap_audio is None:
            raise ValueError(
                "cfg.clap_cond_source='audio' needs per-clip CLAP audio "
                "embeddings. Pass clap_embedder=... to build_dataset so it "
                "returns them (and delete any pre-CLAP cache).")
        if clap_audio.shape[0] != n_songs:
            raise ValueError(
                f"clap_audio has {clap_audio.shape[0]} rows but there are "
                f"{n_songs} clips — stale cache?")
        # Fit the modality-gap alignment on this corpus + prompt set. It is
        # frozen from here on and saved in text_enc's state dict, so inference
        # applies exactly the transform training used.
        text_enc.fit_alignment(clap_audio, mood_texts, prompt_raw,
                               prompt_labels, mode=align_mode,
                               reg=getattr(cfg, "clap_map_reg", 0.1))
        text_mix = cfg.clap_text_mix
        print(f"[COND] Conditioning on per-clip CLAP AUDIO embeddings "
              f"{tuple(clap_audio.shape)}; noise={cfg.clap_audio_noise}, "
              f"text_mix={text_mix}")
        print(f"[ALIGN] mode={align_mode}"
              + (f" reg={cfg.clap_map_reg}" if align_mode == "map" else "")
              + f"; prompts={len(prompts)} "
              f"({', '.join(f'{c} {m.split()[0]}' for m, c in zip(uniq_moods, para_count.tolist()))})")
        # Margin toward the correct mood's audio centroid (in-sample; >0 means
        # the vector points at its own mood). The canonical caption is the one
        # inference and evaluation actually use.
        canon = text_enc.alignment_report(
            clap_audio, mood_texts, prompt_raw[para_start], uniq_moods)
        para = text_enc.alignment_report(
            clap_audio, mood_texts, prompt_raw, prompt_labels)
        print("[ALIGN] text->own-mood audio margin (raw -> aligned):")
        for m in uniq_moods:
            print(f"[ALIGN]   {m:22s} caption {canon[m][0]:+.3f} -> "
                  f"{canon[m][1]:+.3f}   paraphrases {para[m][0]:+.3f} -> "
                  f"{para[m][1]:+.3f}")
    else:
        # Legacy text conditioning: every clip gets a sampled prompt of its
        # mood on every step. With no audio embeddings there is no gap to
        # correct, so the alignment stays at identity.
        if align_mode != "none":
            print(f"[ALIGN] clap_align={align_mode!r} ignored: it needs "
                  f"clap_cond_source='audio'")
        text_mix = 1.0
        print(f"[COND] Conditioning on mood TEXT prompts "
              f"({len(prompts)} prompts for {len(uniq_moods)} moods)")

    all_params = (list(dit.parameters()) +
                  list(melody_enc.parameters()) +
                  list(text_enc.parameters()))     # projection only; CLAP frozen
    optimizer = optim.AdamW(all_params, lr=cfg.diff_lr)

    # Resume from an interrupted run. latent_mean/std are recomputed above
    # (deterministic given the same AE + data), so the resumed run stays
    # consistent with the checkpoint's normalization.
    ckpt_path = os.path.join(cfg.output_dir, "diffusion_ckpt.pt")
    start_epoch = 1
    if getattr(cfg, "resume", True) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        dit.load_state_dict(ck["dit"])
        melody_enc.load_state_dict(ck["melody_enc"])
        text_enc.load_state_dict(ck["text_enc"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        print(f"[RESUME] Diffusion resumed from step {ck['epoch']} "
              f"-> continuing at {start_epoch} ({ckpt_path})")

    # Apply the alignment only now, after any resume: a resumed checkpoint
    # carries the alignment its weights were trained under, and the vectors
    # fed from here on must match that, not a fresh fit.
    clap_emb_all = (text_enc.align_audio(clap_audio.float()).cpu()
                    if cond_source == "audio" else None)       # (N, clap_dim)
    prompt_emb = text_enc.align_text(prompt_raw)               # (P, clap_dim)

    # Mood-balanced sampling: DEAM is skewed toward positive valence
    # (~70/30 happy/sad even after the dead band), so uniform sampling
    # would under-train the sad side and bias the conditioning.
    # Weight each clip by the inverse of its mood's frequency so every
    # mood contributes ~equally to training batches.
    from collections import Counter
    mood_counts = Counter(mood_texts)
    sample_weights = torch.tensor(
        [1.0 / mood_counts[t] for t in mood_texts], dtype=torch.double)

    print(f"\n{'='*60}")
    print(f" Training Diffusion Model ({cfg.diff_epochs} steps, "
          f"{n_songs} clips, batch {cfg.batch_size})")
    print(f" DiT blocks: {cfg.n_dit_blocks} ({cfg.n_controlnet_blocks} w/ ControlNet)")
    print(f" d_model: {cfg.d_model}  heads: {cfg.n_heads}")
    print(f" Mood distribution (raw): {dict(mood_counts)}")
    print(f" Sampling: balanced — each of the {len(mood_counts)} moods "
          f"~{1.0 / len(mood_counts):.0%} of every batch")
    print(f"{'='*60}")

    dit.train()
    melody_enc.train()
    text_enc.train()

    ckpt_interval = getattr(cfg, "diff_ckpt_interval", cfg.diff_epochs)

    def _save_diff_ckpt(step):
        state = {
            "dit": dit.state_dict(),
            "melody_enc": melody_enc.state_dict(),
            "text_enc": text_enc.state_dict(),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            # Records what the projection was trained to accept, so a
            # checkpoint can be told apart from a legacy text-conditioned one.
            "clap_cond_source": cond_source,
            # Provenance only — the alignment itself lives in text_enc's
            # buffers and is applied by load_state_dict.
            "clap_align": align_mode if cond_source == "audio" else "none",
            "clap_text_paraphrases": use_paraphrases,
            # Provenance only: tells whether melody_scale=0 is in-distribution
            # for this checkpoint (0.0 = never trained without melody).
            "melody_dropout": getattr(cfg, "melody_dropout", 0.0),
        }
        # Inference-ready copy (what evaluate.py / edit mode load) ...
        _atomic_save(state, os.path.join(cfg.output_dir, "diffusion.pt"))
        # ... plus a resumable copy that also carries optimizer + step.
        _atomic_save({**state, "optimizer": optimizer.state_dict(),
                      "epoch": step}, ckpt_path)

    for epoch in tqdm(range(start_epoch, cfg.diff_epochs + 1), desc="Diffusion",
                      initial=start_epoch - 1, total=cfg.diff_epochs):
        optimizer.zero_grad()

        idx = torch.multinomial(sample_weights, cfg.batch_size,
                                replacement=True)
        z0 = z0_all[idx].to(device, non_blocking=True)
        t = torch.randint(0, cfg.num_train_timesteps,
                          (cfg.batch_size,), device=device)
        noise = torch.randn_like(z0)
        z_t = diffusion.q_sample(z0, t, noise)

        mel_emb = melody_enc(melody_all[idx].to(device, non_blocking=True), W_lat)
        # Melody dropout: zero some rows' melody embedding so "no melody" is a
        # state the model has trained on. Zero is exactly what inference's
        # melody_scale=0 feeds the ControlNet, so the two match with no
        # inference change. Without this, melody_scale=0 was off-distribution
        # and collapsed the text effect in both directions (sweep_lock.py, job
        # 43197309: happy push 3%, sad 14% of the real gap at strength 0.6),
        # so the melody lock could not be loosened to test whether it blocks
        # sad -> happy. Drawn independently of the text dropout below.
        mel_drop_p = getattr(cfg, "melody_dropout", 0.0)
        if mel_drop_p > 0:
            keep = torch.rand(cfg.batch_size, 1, 1, device=device) >= mel_drop_p
            mel_emb = mel_emb * keep

        if clap_emb_all is not None:
            clap_emb = clap_emb_all[idx].to(device)   # (B, clap_dim), a fresh copy
            # Bridge CLAP's modality gap: jitter the (aligned) audio embedding
            # so the model tolerates a shifted input.
            if cfg.clap_audio_noise > 0:
                n = torch.randn_like(clap_emb)
                n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
                clap_emb = clap_emb + cfg.clap_audio_noise * n
                clap_emb = clap_emb / (clap_emb.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            clap_emb = torch.empty(cfg.batch_size, prompt_emb.shape[1],
                                   device=device)

        # Hand some rows (audio mode) or all rows (text mode) a TEXT prompt of
        # the clip's mood, drawn fresh from its paraphrase set, so the
        # inference-time text path is trained directly on a distribution
        # rather than on one fixed caption.
        if text_mix > 0:
            swap = torch.rand(cfg.batch_size, device=device) < text_mix
            k = mood_row[idx].to(device)
            pick = (torch.rand(cfg.batch_size, device=device)
                    * para_count[k]).long()
            pick = torch.minimum(pick, para_count[k] - 1)
            clap_emb[swap] = prompt_emb[(para_start[k] + pick)[swap]]

        drop = torch.rand(cfg.batch_size, device=device) < cfg.cfg_dropout
        clap_emb[drop] = null_clap_emb
        text_emb = text_enc(clap_emb)

        v_pred = dit(z_t, t, text_emb, mel_emb)
        v_tgt = diffusion.v_target(z0, noise, t)

        loss = F.mse_loss(v_pred, v_tgt)
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if epoch % ckpt_interval == 0 or epoch == cfg.diff_epochs:
            _save_diff_ckpt(epoch)

    dit.eval()
    melody_enc.eval()
    text_enc.eval()
    return dit, melody_enc, text_enc, diffusion, (latent_mean, latent_std)
