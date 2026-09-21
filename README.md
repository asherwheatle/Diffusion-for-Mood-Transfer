# musicdiffusion

Implementation of [Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer](https://arxiv.org/abs/2410.05151) (Hou et al., 2024) on top of a VAE + BigVGAN audio pipeline.

## What it does

Takes an audio clip and a text mood description (`"happy and uplifting"` or `"sad and melancholic"`) and outputs a version of the audio with the mood altered while preserving the original melody structure. The ControlNet branch locks in pitch/melody via CQT features; the DiT is steered by the text prompt via cross-attention and classifier-free guidance.

## Data setup

Training now uses DEAM's official valence/arousal annotations (instead of the
old audio-feature heuristic) and defaults to the **entire** annotated dataset
(~1800 songs). Expected layout:

```
data/
  DEAM_audio/MEMD_audio/        2.mp3 ... 2058.mp3
  DEAM_Annotations/             the annotations download (any nesting is fine —
                                the loader searches recursively for
                                valence.csv + arousal.csv, or the
                                static_annotations_averaged*.csv files)
```

Clips are labeled by mean **valence** over their own annotation window
(DEAM's dynamic annotations start at 15 s, so training clips do too), mapped
to one of two mood strings: `v > +0.1` is happy, `v < -0.1` is sad. Arousal is
not used. Clips inside that dead band (~28% of them) are dropped rather than
labeled — DEAM's valence ratings sit on a positive shift, so a barely-negative
clip does not actually sound sad, and labeling it as such is what blurred the
two classes together. Songs without annotations are skipped; if no annotation
files are found at all, it falls back to the old heuristic with a warning.

The first run decodes all mp3s and extracts melodies (~30 min), then caches
everything to `cache/*.npz`; later runs load in seconds. Use `--no_cache` to
disable.

## Usage

```bash
# Full pipeline: train autoencoder -> train diffusion -> edit
python mood_diffusion.py --mode full --audio_dir data/DEAM_audio/MEMD_audio \
    --annotations_dir data/DEAM_Annotations --text "sad and melancholic"

# Quick smoke test on a 32-song subset
python mood_diffusion.py --mode full --n_songs 32 --diff_epochs 2000

# Train autoencoder only
python mood_diffusion.py --mode train_ae --audio_dir data/DEAM_audio/MEMD_audio

# Train diffusion model only (requires saved autoencoder.pt)
python mood_diffusion.py --mode train_diff --audio_dir data/DEAM_audio/MEMD_audio

# Edit mood of an existing WAV (requires saved autoencoder.pt + diffusion.pt)
python mood_diffusion.py --mode edit --input output/original.wav --text "happy and uplifting" --edit_strength 0.7

# Run the basic reconstruction pipeline (no mood editing)
python pipeline.py --audio_dir data/DEAM_audio/MEMD_audio
```

## File structure

```
mood_diffusion.py   CLI entry point and main() — reads args, orchestrates the three phases
pipeline.py         BigVGAN vocoder loading, DEAM data loading, mel spectrogram utils,
                    MelNormalizer, and the original VAE-GAN reconstruction pipeline

config.py           DiffusionConfig — all hyperparameters in one place (epochs, lr,
                    model dims, diffusion schedule, CFG scale, etc.)

annotations.py      DEAM valence/arousal CSV loading (dynamic + static formats)
                    and the valence -> mood text mapping (happy/sad + dead band)

dataset.py          build_dataset — cuts each DEAM song into 5 s clips, labels
                    each clip from its own annotation window, extracts melodies,
                    shuffles, caches to cache/*.npz

autoencoder.py      LatentEncoder / LatentDecoder / LatentAutoencoder — convolutional
                    autoencoder that compresses mel spectrograms into a 2D spatial latent
                    suitable for diffusion (no flattening bottleneck)

dit.py              MoodDiT — the full Diffusion Transformer with ControlNet branch
                    (paper §III-A). Contains DiTBlock (self-attn + cross-attn + MLP
                    with AdaLayerNorm timestep conditioning), ZeroLinear (zero-init
                    ControlNet output gates), and SinusoidalTimestepEmbedding

diffusion.py        GaussianDiffusion — cosine noise schedule, v-prediction objective
                    (Salimans & Ho 2022), q_sample, v_target, DDIM sampling step

melody.py           MelodyExtractor — high-pass filter + CQT + top-k bin selection
                    (paper §III-B). MelodyEncoder — embeds pitch indices and downsamples
                    to match the latent temporal resolution for ControlNet injection

text_encoder.py     ClapTextEncoder — frozen CLAP text tower + trainable projection into
                    DiT cross-attention tokens. Also owns the modality-gap alignment
                    (mean-centering + Procrustes text->audio map), fitted before training
                    and stored as buffers so every loader applies it automatically.
                    (Legacy character-level TextEncoder kept for ablation.)

mood_paraphrases.py ~60 CLAP-validated paraphrases per mood. Training samples one per
                    step for the text path; inference/eval keep the canonical caption.

train.py            train_autoencoder() — MSE training loop for the latent AE
                    train_diffusion() — v-prediction training loop for DiT+ControlNet
                    with CFG dropout on text conditioning and mood-balanced batch
                    sampling (inverse-frequency weights, so the minority
                    sad mood gets equal training signal)

inference.py        edit_mood() — SDEdit inference: encode -> add noise to t_start ->
                    DDIM denoise with CFG (text only, melody unguided) -> decode -> BigVGAN

visualize.py        plot_mood_edit() — side-by-side mel spectrogram comparison plot
                    (original vs. mood-edited)

musicmake.py        Standalone script using shikhr/music_maker to generate MIDI
```

## Outputs

All outputs go to `output/` by default:

| File | Description |
|---|---|
| `original.wav` | Source audio clip (15s) |
| `ae_reconstruction.wav` | AE encode→decode round-trip, to verify latent quality |
| `edited_<text>.wav` | Mood-edited output |
| `autoencoder.pt` | Saved AE weights |
| `diffusion.pt` | Saved DiT + MelodyEncoder + TextEncoder weights |
| `mood_edit_comparison.png` | Side-by-side spectrogram plot |

## Key hyperparameters (config.py)

| Parameter | Default | Effect |
|---|---|---|
| `ae_epochs` | 100 | Full passes over the dataset for the AE |
| `diff_epochs` | 10000 | Diffusion training steps; more = stronger mood conditioning |
| `n_train_songs` | None (all) | Subset size for quick experiments |
| `clip_start_seconds` | 15 | Clip offset, aligned to the annotated region |
| `edit_strength` | 0.7 | 0 = no change, 1 = full regeneration from noise |
| `cfg_scale` | 7.0 | Higher = stronger text adherence, less fidelity |
| `d_model` | 256 | DiT model width |
| `n_dit_blocks` | 8 | DiT depth |
| `clip_seconds` | 5 | Audio clip length fed into the pipeline |
| `clips_per_song` | 6 | 5 s clips cut from each song (covers the 15–45 s annotated region) |
