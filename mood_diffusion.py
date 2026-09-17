"""
Music Mood Diffusion: ControlNet-DiT for Text-Conditioned Audio Mood Editing
=============================================================================
Implements "Editing Music with Melody and Text: Using ControlNet for Diffusion
Transformer" (Hou et al., 2024 - arXiv:2410.05151) on top of the existing
VAE + BigVGAN pipeline.

Full pipeline:
  1. Audio waveform -> Mel spectrogram (BigVGAN mel config)
  2. Mel -> 2D latent feature map (LatentAutoencoder encoder)
  3. Melody extraction from audio (Top-k CQT, paper section III-B)
  4. User mood text -> text embedding (frozen CLAP text tower + projection)
  5. Latent Diffusion (DiT + ControlNet) conditioned on text + melody
  6. Modified latent -> Mel (LatentAutoencoder decoder)
  7. Mel -> Output WAV (BigVGAN vocoder)

Usage:
  # Phase 1: Train latent autoencoder on mel spectrograms
  python mood_diffusion.py --mode train_ae --audio_dir data/DEAM_audio/MEMD_audio

  # Phase 2: Train diffusion model (text + melody conditioned)
  python mood_diffusion.py --mode train_diff --audio_dir data/DEAM_audio/MEMD_audio

  # Phase 3: Edit mood of an audio file
  python mood_diffusion.py --mode edit --input output/original.wav \
      --text "sad and melancholic" --edit_strength 0.7

  # Full pipeline (train_ae -> train_diff -> edit)
  python mood_diffusion.py --mode full --audio_dir data/DEAM_audio/MEMD_audio
"""

import os
import argparse

import torch
import soundfile as sf
import librosa

from config import DiffusionConfig
from autoencoder import LatentAutoencoder
from dit import MoodDiT
from melody import MelodyEncoder, MelodyExtractor
from text_encoder import ClapTextEncoder, load_clap_text_model
from diffusion import GaussianDiffusion
from train import train_autoencoder, train_diffusion
from inference import edit_mood
from visualize import plot_mood_edit
from pipeline import (
    load_bigvgan,
    bigvgan_mel_spectrogram,
    load_deam_song,
    FixedMelNormalizer,
    pad_spectrogram,
    unpad_spectrogram,
)
from dataset import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Music Mood Diffusion: edit audio mood with text"
    )
    parser.add_argument("--mode", type=str, default="full",
                        choices=["train_ae", "train_diff", "edit", "full"])
    parser.add_argument("--audio_dir", type=str,
                        default=os.path.join("data", "DEAM_audio", "MEMD_audio"))
    parser.add_argument("--input", type=str, default=None,
                        help="Input WAV for edit mode")
    parser.add_argument("--text", type=str, default="sad and melancholic",
                        help="Mood description for editing")
    parser.add_argument("--edit_strength", type=float, default=0.35,
                        help="0=no change, 1=full regen from noise")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--song_index", type=int, default=0)
    parser.add_argument("--n_songs", type=int, default=None,
                        help="Number of songs for training "
                             "(default: all annotated songs; -1 also = all)")
    parser.add_argument("--annotations_dir", type=str, default=None,
                        help="Root of the DEAM annotations download "
                             "(default: data/DEAM_Annotations)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable the on-disk dataset cache")
    parser.add_argument("--ae_epochs", type=int, default=None)
    parser.add_argument("--diff_epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = DiffusionConfig()
    cfg.output_dir = args.output_dir
    cfg.edit_strength = args.edit_strength
    if args.ae_epochs is not None:
        cfg.ae_epochs = args.ae_epochs
    if args.diff_epochs is not None:
        cfg.diff_epochs = args.diff_epochs
    if args.n_songs is not None:
        cfg.n_train_songs = None if args.n_songs <= 0 else args.n_songs
    if args.annotations_dir is not None:
        cfg.annotations_dir = args.annotations_dir
    if args.no_cache:
        cfg.cache_dir = None

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print(" Music Mood Diffusion Pipeline")
    print(" ControlNet-DiT for Text-Conditioned Audio Mood Editing")
    print("=" * 60)
    print(f"  Mode            : {args.mode}")
    print(f"  Device          : {cfg.device}")
    print(f"  Mood text       : \"{args.text}\"")
    print(f"  Edit strength   : {cfg.edit_strength}")
    print("=" * 60)

    # --- Load BigVGAN ---
    print("\n[STEP 1] Loading BigVGAN vocoder...")
    bigvgan_model = load_bigvgan(cfg.device)

    # --- Load audio ---
    if args.input and args.mode == "edit":
        print(f"\n[STEP 2] Loading input: {args.input}")
        wav, sr = librosa.load(args.input, sr=cfg.sample_rate, mono=True)
        max_samples = cfg.clip_seconds * cfg.sample_rate
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        waveform = torch.FloatTensor(wav).unsqueeze(0)
    else:
        print("\n[STEP 2] Loading DEAM song...")
        waveform, sr = load_deam_song(
            args.audio_dir, args.song_index,
            sample_rate=bigvgan_model.h.sampling_rate,
            clip_seconds=cfg.clip_seconds,
        )
        orig_path = os.path.join(cfg.output_dir, "original.wav")
        sf.write(orig_path, waveform.squeeze(0).numpy(), cfg.sample_rate)
        print(f"  Saved original: {orig_path}")

    wav_np = waveform.squeeze(0).numpy()

    # --- Mel spectrogram of the edit-target song ---
    print("\n[STEP 3] Computing mel spectrogram...")
    mel = bigvgan_mel_spectrogram(waveform, bigvgan_model)
    normalizer = FixedMelNormalizer()
    mel_norm = normalizer.normalize(mel)
    print(f"  Mel shape: {mel.shape}")

    # --- Multi-song training set (text conditioning is only learnable if
    #     different mood texts are paired with different songs) ---
    if args.mode in ("train_ae", "train_diff", "full"):
        n_desc = "all" if cfg.n_train_songs is None else cfg.n_train_songs
        print(f"\n[STEP 4] Building training set ({n_desc} songs)...")
        extractor = MelodyExtractor(
            sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
            bins_per_octave=cfg.cqt_bins_per_octave,
            hop_length=cfg.cqt_hop, fmin=cfg.cqt_fmin,
            top_k=cfg.melody_top_k, highpass_cutoff=cfg.highpass_cutoff,
        )
        # Conditioning on per-clip CLAP audio embeddings needs CLAP during the
        # dataset pass. Build it once here and hand the same frozen module to
        # train_diffusion so it isn't loaded (~2 GB) twice.
        clap_model = clap_embedder = None
        if getattr(cfg, "clap_cond_source", "text") == "audio":
            print("[STEP 4] Loading CLAP for audio-embedding conditioning...")
            clap_model = load_clap_text_model(cfg.clap_ckpt, cfg.device)
            _clap_enc = ClapTextEncoder(
                cfg.d_model, clap_model=clap_model,
                n_tokens=cfg.text_n_tokens, device=cfg.device)
            clap_embedder = _clap_enc.encode_audio

        mel_batch, melodies, mood_texts, names, clap_audio = build_dataset(
            args.audio_dir, cfg.n_train_songs, bigvgan_model,
            cfg.clip_seconds, annotations_dir=cfg.annotations_dir,
            clip_start_seconds=cfg.clip_start_seconds,
            melody_extractor=extractor, cache_dir=cfg.cache_dir,
            clap_embedder=clap_embedder,
            clips_per_song=cfg.clips_per_song,
            augment_moods=cfg.augment_moods,
            augment_target=cfg.augment_target,
            max_aug_per_clip=cfg.max_aug_per_clip,
            aug_max_semitones=cfg.aug_max_semitones,
            aug_max_shift_frac=cfg.aug_max_shift_frac,
            aug_snr_db_range=cfg.aug_snr_db_range,
            augment_seed=cfg.augment_seed,
        )

    # =========================================================================
    # Phase 1: Train latent autoencoder
    # =========================================================================
    if args.mode in ("train_ae", "full"):
        print("\n[PHASE 1] Training latent autoencoder...")
        ae, orig_hw = train_autoencoder(mel_batch, cfg)

        ae_path = os.path.join(cfg.output_dir, "autoencoder.pt")
        torch.save(ae.state_dict(), ae_path)
        print(f"  Saved autoencoder: {ae_path}")

        # Verify reconstruction
        ae.eval()
        with torch.no_grad():
            mel_padded, hw = pad_spectrogram(mel_norm.unsqueeze(0))
            recon, z = ae(mel_padded.to(cfg.device))
            if recon.shape != mel_padded.shape:
                import torch.nn.functional as F
                recon = F.interpolate(recon, size=mel_padded.shape[2:],
                                      mode="bilinear", align_corners=False)
            recon = unpad_spectrogram(recon.cpu(), hw)
        recon_mel = normalizer.denormalize(recon.squeeze(0))
        with torch.inference_mode():
            wav_ae = bigvgan_model(recon_mel.to(cfg.device))
        ae_wav_path = os.path.join(cfg.output_dir, "ae_reconstruction.wav")
        sf.write(ae_wav_path, wav_ae.squeeze().cpu().clamp(-1.0, 1.0).numpy(),
                 cfg.sample_rate)
        print(f"  AE reconstruction saved: {ae_wav_path}")

        if args.mode == "train_ae":
            return

    # =========================================================================
    # Phase 2: Train diffusion model
    # =========================================================================
    if args.mode in ("train_diff", "full"):
        if args.mode == "train_diff":
            ae = LatentAutoencoder(cfg.ae_channels).to(cfg.device)
            ae_path = os.path.join(cfg.output_dir, "autoencoder.pt")
            ae.load_state_dict(torch.load(ae_path, map_location=cfg.device,
                                          weights_only=True))
            ae.eval()

        print("\n[PHASE 2] Training diffusion model...")
        dit, melody_enc, text_enc, diffusion, latent_stats = train_diffusion(
            ae, mel_batch, melodies, mood_texts, cfg,
            clap_audio=clap_audio, clap_model=clap_model,
        )
        latent_mean, latent_std = latent_stats

        diff_path = os.path.join(cfg.output_dir, "diffusion.pt")
        torch.save({
            "dit": dit.state_dict(),
            "melody_enc": melody_enc.state_dict(),
            "text_enc": text_enc.state_dict(),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "clap_cond_source": getattr(cfg, "clap_cond_source", "text"),
            # Provenance only; the alignment itself is in text_enc's buffers.
            "clap_align": (getattr(cfg, "clap_align", "none")
                           if getattr(cfg, "clap_cond_source", "text") == "audio"
                           else "none"),
            "clap_text_paraphrases": getattr(cfg, "clap_text_paraphrases", False),
        }, diff_path)
        print(f"  Saved diffusion model: {diff_path}")

        if args.mode == "train_diff":
            return

    # =========================================================================
    # Phase 3: Edit mood
    # =========================================================================
    if args.mode == "edit":
        ae = LatentAutoencoder(cfg.ae_channels).to(cfg.device)
        ae_path = os.path.join(cfg.output_dir, "autoencoder.pt")
        ae.load_state_dict(torch.load(ae_path, map_location=cfg.device,
                                      weights_only=True))
        ae.eval()

        import torch.nn.functional as F
        mel_padded, _ = pad_spectrogram(mel_norm.unsqueeze(0))
        with torch.no_grad():
            z_shape = ae.encoder(mel_padded.to(cfg.device))
        _, C_lat, H_lat, W_lat = z_shape.shape

        dit = MoodDiT(
            latent_channels=C_lat, latent_h=H_lat,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_blocks=cfg.n_dit_blocks, n_control_blocks=cfg.n_controlnet_blocks,
        ).to(cfg.device)
        melody_enc = MelodyEncoder(cfg.d_model, cfg.melody_top_k).to(cfg.device)
        text_enc = ClapTextEncoder(cfg.d_model, clap_ckpt=cfg.clap_ckpt,
                                   n_tokens=cfg.text_n_tokens,
                                   device=cfg.device).to(cfg.device)

        diff_path = os.path.join(cfg.output_dir, "diffusion.pt")
        ckpt = torch.load(diff_path, map_location=cfg.device, weights_only=True)
        dit.load_state_dict(ckpt["dit"])
        melody_enc.load_state_dict(ckpt["melody_enc"])
        text_enc.load_state_dict(ckpt["text_enc"])
        latent_mean = ckpt["latent_mean"].to(cfg.device)
        latent_std = ckpt["latent_std"].to(cfg.device)
        dit.eval()
        melody_enc.eval()
        text_enc.eval()

        diffusion = GaussianDiffusion(cfg.num_train_timesteps, cfg.device)

    # --- Edit and save ---
    print(f"\n[PHASE 3] Editing mood: \"{args.text}\"")
    print(f"  Edit strength: {cfg.edit_strength}")

    wav_edited = edit_mood(
        waveform, args.text,
        ae, dit, melody_enc, text_enc, diffusion,
        bigvgan_model, cfg,
        latent_mean, latent_std,
    )

    edited_path = os.path.join(cfg.output_dir,
                                f"edited_{args.text.replace(' ', '_')}.wav")
    sf.write(edited_path, wav_edited.squeeze().numpy(), cfg.sample_rate)
    print(f"  Saved edited audio: {edited_path}")

    mel_edited = bigvgan_mel_spectrogram(wav_edited, bigvgan_model)
    plot_path = os.path.join(cfg.output_dir, "mood_edit_comparison.png")
    plot_mood_edit(mel, mel_edited, args.text, plot_path)

    print("\n" + "=" * 60)
    print(" MOOD DIFFUSION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Original        : output/original.wav")
    print(f"  Mood-edited     : {edited_path}")
    print(f"  Comparison plot : {plot_path}")
    print()
    print("  The melody (pitch structure) is preserved via ControlNet,")
    print("  while the mood/texture changes based on your text prompt.")


if __name__ == "__main__":
    main()
