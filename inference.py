"""Mood editing inference via SDEdit + classifier-free guidance."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DiffusionConfig
from annotations import mood_prompt
from autoencoder import LatentAutoencoder
from melody import MelodyExtractor, MelodyEncoder
from text_encoder import ClapTextEncoder
from dit import MoodDiT
from diffusion import GaussianDiffusion
from pipeline import bigvgan_mel_spectrogram, FixedMelNormalizer, pad_spectrogram, unpad_spectrogram


@torch.no_grad()
def edit_mood(
    waveform: torch.Tensor,
    mood_text: str,
    ae: LatentAutoencoder,
    dit: MoodDiT,
    melody_enc: MelodyEncoder,
    text_enc: ClapTextEncoder,
    diffusion: GaussianDiffusion,
    bigvgan_model,
    cfg: DiffusionConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    melody_scale: float = None,
    cond_emb: torch.Tensor = None,
    cond_space: str = "audio",
) -> torch.Tensor:
    """
    Edit the mood of an audio waveform using text conditioning.

    SDEdit approach (paper section IV-C):
      1. Encode input mel -> latent z0
      2. Add noise to z0 up to timestep t_start (controlled by edit_strength)
      3. Denoise from t_start -> 0 with text + melody conditioning
      4. CFG scale 7 on text only (melody unguided, per paper)
      5. Decode -> BigVGAN -> output waveform

    edit_strength=0: no change. edit_strength=1: full regen from noise.

    cond_emb overrides the conditioning vector. Normally the CLAP *text*
    embedding of `mood_text` is used; pass a (clap_dim,) tensor here to feed
    an arbitrary CLAP vector instead — e.g. a CLAP *audio* embedding, which
    lets you test the conditioning path without crossing the audio/text
    modality gap. `mood_text` is then used only for logging.

    cond_space says what kind of vector cond_emb is, i.e. which alignment to
    apply before it reaches the model:
      "audio"  a raw CLAP audio embedding -> text_enc.align_audio (default)
      "text"   a raw CLAP text embedding  -> text_enc.align_text
      "as_is"  used as given (only L2-normed) — bypasses the alignment, e.g.
               to measure what the alignment buys on a raw text vector.

    melody_scale rescales the melody embedding before it reaches the
    ControlNet branch (default cfg.melody_scale). 0 removes melody
    information entirely — use it to test whether melody control is so
    strong it pins the output to the input and blocks the mood edit.

    Returns:
        wav_out: (1, T_samples) output waveform tensor
    """
    device = cfg.device
    wav_np = waveform.squeeze().numpy()

    mel = bigvgan_mel_spectrogram(waveform, bigvgan_model)
    normalizer = FixedMelNormalizer()
    mel_norm = normalizer.normalize(mel)
    mel_padded, orig_hw = pad_spectrogram(mel_norm.unsqueeze(0))
    mel_padded = mel_padded.to(device)

    z0 = ae.encoder(mel_padded)
    # Same latent standardization used during diffusion training
    z0 = (z0 - latent_mean) / latent_std
    print(f"[EDIT] Latent z0: {z0.shape}")

    extractor = MelodyExtractor(
        sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
        bins_per_octave=cfg.cqt_bins_per_octave,
        hop_length=cfg.cqt_hop, fmin=cfg.cqt_fmin,
        top_k=cfg.melody_top_k, highpass_cutoff=cfg.highpass_cutoff,
    )
    melody_indices = extractor.extract(wav_np)
    melody_tensor = torch.from_numpy(melody_indices).unsqueeze(0).to(device)
    W_lat = z0.shape[-1]
    melody_emb = melody_enc(melody_tensor, W_lat)
    if melody_scale is None:
        melody_scale = getattr(cfg, "melody_scale", 1.0)
    if melody_scale != 1.0:
        melody_emb = melody_emb * melody_scale

    # Embed the same caption the trainer and the evaluator use for this mood
    # ("a sad and melancholic piece of music", not the bare tag) — a different
    # string is a different CLAP vector. Every conditioning vector goes through
    # the modality-gap alignment the checkpoint was trained with (identity for
    # checkpoints that predate it); see ClapTextEncoder.align_*.
    if cond_emb is not None:
        vec = cond_emb.detach().to(device).float().reshape(1, -1)
        if cond_space == "audio":
            vec = text_enc.align_audio(vec)
        elif cond_space == "text":
            vec = text_enc.align_text(vec)
        elif cond_space == "as_is":
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"unknown cond_space {cond_space!r}")
        text_emb = text_enc(vec)
    else:
        text_emb = text_enc(text_enc.align_text(
            text_enc.encode([mood_prompt(mood_text)])))
    # The null embedding is unaligned on purpose — training builds it the same
    # way, and matching training is all that matters for it.
    null_text_emb = text_enc(text_enc.encode([""]))

    # SDEdit: noise z0 up to t_start
    T = cfg.num_train_timesteps
    t_start = max(1, min(int(cfg.edit_strength * T), T - 1))

    # Evenly spaced timesteps from t_start down to 0. The old form used a
    # fixed stride (T // num_inference_steps), which made the step COUNT
    # depend on edit_strength — a gentle edit silently got far fewer steps
    # than a strong one, so the two weren't comparable.
    grid = np.linspace(t_start, 0, cfg.num_inference_steps + 1)
    timesteps = list(dict.fromkeys(grid.round().astype(int).tolist()))

    noise = torch.randn_like(z0)
    t_tensor = torch.tensor([t_start], device=device)
    z_t = diffusion.q_sample(z0, t_tensor, noise)

    print(f"[EDIT] SDEdit from t={t_start} ({len(timesteps) - 1} steps), "
          f"CFG scale={cfg.cfg_scale}, eta={getattr(cfg, 'ddim_eta', 0.0)}, "
          f"melody_scale={melody_scale}")
    print(f"[EDIT] Mood text: \"{mood_text}\"")

    # DDIM denoising with CFG on text only (paper section IV-C)
    for i in tqdm(range(len(timesteps) - 1), desc="Denoising"):
        t_cur = torch.tensor([timesteps[i]], device=device)
        t_prev = torch.tensor([timesteps[i + 1]], device=device)

        v_cond = dit(z_t, t_cur, text_emb, melody_emb)
        v_uncond = dit(z_t, t_cur, null_text_emb, melody_emb)
        v_guided = v_uncond + cfg.cfg_scale * (v_cond - v_uncond)

        z_t = diffusion.ddim_step(z_t, v_guided, t_cur, t_prev,
                                  eta=getattr(cfg, "ddim_eta", 0.0))

    # Undo standardization before the decoder (it expects raw encoder-scale latents)
    z_t = z_t * latent_std + latent_mean
    recon_mel_norm = ae.decoder(z_t)
    if recon_mel_norm.shape != mel_padded.shape:
        recon_mel_norm = F.interpolate(
            recon_mel_norm, size=mel_padded.shape[2:],
            mode="bilinear", align_corners=False
        )
    recon_mel_norm = unpad_spectrogram(recon_mel_norm, orig_hw)
    recon_mel = normalizer.denormalize(recon_mel_norm.squeeze(0).cpu())

    with torch.inference_mode():
        wav_out = bigvgan_model(recon_mel.to(device))
    # BigVGAN v2 has use_tanh_at_final=false, so output is unbounded
    return wav_out.squeeze(0).cpu().clamp(-1.0, 1.0)
