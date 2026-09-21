import os

import torch


class DiffusionConfig:
    sample_rate = 44100
    clip_seconds = 5
    # Training clips start 15 s in: DEAM's dynamic annotations only cover
    # 15 s onward, so this aligns the audio with its mood label
    clip_start_seconds = 15
    # 6 x 5 s covers 15-45 s, the whole annotated region of a DEAM
    # excerpt; each clip gets the mood of its own 5 s window
    clips_per_song = 6
    n_mels = 128

    # Mood balancing via waveform augmentation (see augment.py). With the
    # valence-only labels the split is ~70/30 happy/sad, so the minority
    # "sad and melancholic" is boosted to parity by pitch-shifting /
    # time-shifting / adding noise to the real sad clips. Empty tuple
    # disables augmentation.
    # NOTE both moods are listed on purpose. Augmenting only the minority mood
    # made the augmentation artifacts a proxy for that mood: they appeared in
    # 59% of sad clips and 0% of happy ones, and measured on real clips the
    # artifacts alone move CLAP +0.069 toward the sad caption and the valence
    # probe -0.061. The model could read "has pitch-shift ringing and added
    # noise" as "sad" instead of learning mood, and because
    # clap_cond_source="audio" the contamination reached the conditioning
    # vectors themselves. Class balance does NOT need augmentation:
    # train.train_diffusion already draws every batch with inverse-frequency
    # weights, so each mood is ~50% of a batch whatever the raw counts.
    augment_moods = (
        "happy and uplifting",
        "sad and melancholic",
    )
    # Equal variants-per-clip for every augmented mood, so the augmented
    # FRACTION is identical across moods and artifacts carry no mood
    # information. Set to None to fall back to augment_target (count-matching)
    # mode, which cannot equalize the fraction when the moods differ in size.
    augment_ratio = 1.0
    augment_target = None        # target clips/mood; None = match largest mood
    max_aug_per_clip = 12        # ceiling on variants per real clip
    aug_max_semitones = 2.0      # pitch shift range (+/-)
    aug_max_shift_frac = 0.2     # time shift range as fraction of clip
    aug_snr_db_range = (20.0, 35.0)  # gaussian-noise SNR range
    augment_seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    annotations_dir = os.path.join("data", "DEAM_Annotations")
    cache_dir = "cache"

    # Autoencoder
    # 3 downsample stages (was 4): latent is 32 x (H/8) x (W/8) = 32x16x54
    # instead of 32x8x27, i.e. ~2x compression instead of ~8x. Each latent cell
    # now covers an 8x8 mel patch (was 16x16), so the decoder no longer has to
    # hallucinate large blocks — this is the main fix for the reconstruction
    # distortion. NOTE: this quadruples the DiT sequence length (216 -> 864),
    # so diffusion training (Phase 2) is significantly more expensive per step.
    ae_channels = [1, 64, 128, 32]
    ae_lr = 1e-3
    ae_epochs = 100
    # Reconstruction-loss weights (see train._mel_recon_loss). Pure MSE rewards
    # blur; these sharpen it. Tune if the loss scale needs rebalancing.
    ae_l1_weight = 1.0       # L1 base term (sharper than MSE)
    ae_grad_weight = 1.0     # time/freq gradient match (edge/harmonic detail)
    ae_ms_weight = 0.5       # multi-scale L1 (global structure)

    # CQT / Melody (paper section III-B)
    cqt_bins = 128
    cqt_bins_per_octave = 12
    cqt_hop = 512
    cqt_fmin = 8.18
    melody_top_k = 4
    highpass_cutoff = 261.2

    # DiT + ControlNet
    d_model = 256
    n_heads = 4
    n_dit_blocks = 8
    n_controlnet_blocks = 4
    mlp_ratio = 4
    dropout = 0.0

    # Text encoder (Lever A: frozen CLAP text tower + trainable projection)
    text_max_len = 128           # legacy char encoder only; unused by CLAP
    clap_ckpt = "music_audioset_epoch_15_esc_90.14.pt"
    text_n_tokens = 4            # length of the projected conditioning sequence

    # What the diffusion model is conditioned on during training.
    #   "audio" — each clip's own CLAP *audio* embedding; at inference the
    #             CLAP *text* embedding of the prompt is swapped in. CLAP's
    #             towers are contrastively aligned, so this transfers, and it
    #             gives ~10k distinct conditioning vectors (one per clip)
    #             instead of 2 (one per mood label). With only 2 vectors the
    #             trainable projection degenerates into a lookup table with
    #             no reason to preserve CLAP's semantics — which is why the
    #             model moved audio in mood-irrelevant directions.
    #   "text"  — legacy: the CLAP text embedding of the clip's mood label.
    clap_cond_source = "audio"
    # Modality-gap handling for clap_cond_source="audio". CLAP's audio and
    # text embeddings sit in slightly offset cones, so a model trained purely
    # on audio vectors meets an out-of-distribution input at inference.
    #   clap_audio_noise: Gaussian noise added to each audio embedding as a
    #       fraction of its (unit) norm, making the model tolerant of a shift.
    #   clap_text_mix: probability of substituting the mood label's TEXT
    #       embedding for the audio one, so the text path is trained directly
    #       rather than only assumed to transfer.
    clap_audio_noise = 0.10
    clap_text_mix = 0.25
    # Sample the text-mix prompt from ~60 paraphrases per mood
    # (mood_paraphrases.py) instead of the one canonical caption, so the text
    # path sees a spread-out distribution per mood and the projection has to
    # learn the happy<->sad direction rather than two fixed addresses.
    # Inference and evaluation still use the canonical caption.
    clap_text_paraphrases = True
    # Modality-gap correction, fitted once before training and saved in the
    # checkpoint (ClapTextEncoder buffers), so inference applies the same map.
    #   "none"   — raw CLAP vectors (the previous behaviour)
    #   "center" — subtract each modality's mean and renormalise
    #   "map"    — center, then rotate text into audio space (Procrustes)
    # Measured on this corpus with held-out paraphrases, the margin of a text
    # vector toward its own mood's audio centroid was +0.07 raw, +0.29
    # centered, +0.54 mapped — the last matching real audio clips (+0.53).
    clap_align = "map"
    # Procrustes shrinkage toward identity (fraction of the cross-covariance
    # spectral norm). Pins the ~510 directions with no mood signal to
    # identity; 0.01-0.1 gave the same held-out margin, larger values rotate
    # less (0.5: +0.49, 2.0: +0.38).
    clap_map_reg = 0.1

    # Diffusion
    num_train_timesteps = 1000
    prediction_type = "v"

    # Training
    n_train_songs = None      # None = entire annotated dataset (~1800 songs)
    # Raised from 16: the old value badly underfilled a modern GPU, leaving it
    # copy-stalled rather than compute-bound.
    batch_size = 64
    num_workers = 4           # DataLoader workers for the autoencoder loop
    diff_lr = 1e-4
    diff_epochs = 10000
    cfg_scale = 1.5
    # Fraction of training steps that see the null text embedding. 0.1 is the
    # low end of the usual range and leaves the unconditional path thinly
    # trained, which makes the (cond - uncond) guidance direction noisy.
    cfg_dropout = 0.2

    # Inference
    num_inference_steps = 50
    edit_strength = 0.35
    # DDIM stochasticity. 0 = deterministic (the original behaviour); a small
    # value lets an SDEdit trajectory leave the input's basin instead of
    # retracing it. 1.0 recovers ancestral/DDPM sampling.
    ddim_eta = 0.0
    # Scales the melody embedding feeding the ControlNet branch. 1.0 = as
    # trained; 0 = no melody information (mood-edit headroom test, see
    # ablate_melody.py). Lower values trade melody preservation for
    # conditioning freedom.
    melody_scale = 1.0

    log_interval = 50
    output_dir = "output"

    # Checkpointing (eviction resilience — HiPerGator jobs can be killed
    # mid-run). Every interval we write a resumable *_ckpt.pt (model +
    # optimizer + step) and refresh the inference-ready autoencoder.pt /
    # diffusion.pt, so an interrupted run is both restartable and usable.
    ae_ckpt_interval = 25       # epochs between autoencoder checkpoints
    diff_ckpt_interval = 1000   # steps between diffusion checkpoints
    resume = True               # resume from *_ckpt.pt in output_dir if present
