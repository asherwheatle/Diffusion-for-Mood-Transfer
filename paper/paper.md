# Text-Conditioned Music Mood Editing with a Latent Diffusion Transformer and ControlNet

Asher Wheatle, University of Florida

## Abstract

Changing the mood of a piece of music while keeping its melody recognizable is a hard editing problem: the musical features that carry mood, such as mode, timbre, loudness and rhythmic density, overlap with the features that make a melody what it is. We build a small text-guided mood editor based on the Diffusion Transformer (DiT) with a ControlNet melody branch proposed by Hou et al. (2024). Audio is converted to a mel spectrogram and compressed by a purpose-trained convolutional autoencoder, and a 20.3 M-parameter DiT is trained from scratch on 5-second clips from the DEAM dataset, labeled happy or sad from their annotated valence. Mood is supplied through CLAP embeddings: the model trains on each clip's CLAP audio embedding and is prompted at inference with a CLAP text embedding, after a simple alignment step that reduces the gap between CLAP's audio and text spaces. Edits start from a DDIM inversion of the input rather than from random noise, which in our tests roughly doubled how much of the melody survived. On 20 held-out songs, the happy edit of a song scored as happier than its sad edit for 19 songs by CLAP and all 20 by a valence probe, and on average the two edits ended up about as far apart as real happy and sad music. The output kept 57.2% frame-level melody accuracy against a ceiling of 63.9% imposed by our autoencoder and vocoder. The effect is clearly asymmetric: edits toward sad were classified as sad 95% of the time, but edits toward happy only 50% of the time. Because the same CLAP model conditions the editor and judges it, these results still need to be confirmed by listening tests.

## 1 Introduction

Diffusion models now produce convincing images and audio, and much recent work uses them to edit existing content rather than generate it from scratch. For music, one natural edit is to change the mood of a recording, for example making a sad piece sound happy, while keeping the melody so that a listener still recognizes the song. This is difficult because mood and melody are not separate layers of the audio. Mode, harmony and register affect both, so an edit that changes mood strongly tends to change the tune as well.

Hou et al. (2024) addressed a related problem, text-guided music editing with melody control, by pairing a Diffusion Transformer with a ControlNet branch that feeds the model a melody representation extracted from the input. Their system is large and built on a pretrained audio generation model. We ask whether the same design can be rebuilt at a much smaller scale, trained from scratch on a single public emotion dataset, and focused on one edit: moving a clip toward happy or toward sad.

Our main contributions are:

1. A compact version of the DiT + ControlNet editor (20.3 M parameters) that runs on mel spectrograms through a small autoencoder and a pretrained BigVGAN vocoder.
2. A way to condition on mood with CLAP that avoids training on only two text vectors. The model learns from per-clip CLAP audio embeddings, and a fixed alignment lets text prompts stand in for them at inference.
3. An evaluation that tries to separate the effect of the mood prompt from artifacts introduced by resynthesis, together with a set of diagnostics that explain several design choices: DDIM inversion instead of SDEdit noising, per-channel latent standardization, and augmenting both moods equally.

We also report where the system falls short. The push toward happy is consistently weaker than the push toward sad, and our only automatic mood judge is not independent of the model.

## 2 Background

**Diffusion editing.** SDEdit (Meng et al., 2022) edits an input by adding noise to it and then denoising under a new condition. The amount of noise sets a trade-off between staying close to the input and following the new condition. DDIM inversion (Song et al., 2021; Mokady et al., 2023) instead runs the deterministic DDIM sampler backwards from the input, producing a noisy latent that still encodes the input. Denoising it under the original condition returns approximately the same input.

**Diffusion Transformers and ControlNet.** A DiT (Peebles & Xie, 2023) replaces the usual U-Net denoiser with a transformer over latent patches. ControlNet (Zhang et al., 2023) adds a parallel copy of part of the network that receives an extra control signal and adds its output back through zero-initialized layers, so the control is learned without disrupting the base model at the start. Hou et al. (2024) apply this to music, using the most energetic constant-Q transform (CQT) pitch bins per frame as the melody control.

**CLAP and the modality gap.** CLAP (Wu et al., 2023) trains an audio encoder and a text encoder so that matching audio and captions have similar embeddings. In practice, embeddings from the two encoders occupy separate regions of the shared space (Liang et al., 2022). A model trained on one modality's embeddings may therefore treat the other modality's embeddings as unfamiliar input.

## 3 Method

### 3.1 Overview

Given a 5-second clip and a target mood (happy or sad), the system produces a new clip that should sound like the target mood and keep the original's pitch content. Figure 1 shows the pipeline. A clip is converted to a log-mel spectrogram and encoded to a latent by a convolutional autoencoder. A melody sequence is extracted from the waveform with a CQT. The DiT denoises the latent with mood supplied through cross-attention and melody supplied through the ControlNet branch. The latent is then decoded back to a spectrogram, and BigVGAN turns it into audio.

> [Figure 1 placeholder. Training: DEAM clip → mel → AE encoder → latent → add noise → DiT + ControlNet → v-prediction loss; CQT melody into the ControlNet branch; CLAP audio embedding → alignment → projection → cross-attention. Editing: input clip → latent → DDIM inversion → guided DDIM denoising with the CLAP text embedding of the target caption → AE decoder → BigVGAN → output.]

### 3.2 Data and labels

We use DEAM (Aljanaki et al., 2017), which provides 45-second song excerpts with valence and arousal ratings sampled every 500 ms from 15 s onward. We rescale the ratings to [−1, 1] and use all 1,802 songs that have both series. Each song's annotated region (15–45 s) is split into six non-overlapping 5-second windows, for 10,812 candidate clips. Each clip is labeled from the mean valence of its own window: happy above 0.1, sad below −0.1, and discarded in between. The dead band matters because DEAM ratings lean positive, so a slightly negative clip does not reliably sound sad. We use valence only. It is the dimension that separates happy from sad, and in early tests CLAP distinguished arousal-based categories such as "calm" and "energetic" less well than any other mood categories we tried.

The dead band removes 2,985 clips (27.6%), leaving 7,827 clips: 5,543 happy (70.8%) and 2,284 sad (29.2%). Every labeled clip then receives one augmented copy, made by a pitch shift of up to ±2 semitones, a time shift of up to ±20% of the clip (with zero fill, not wraparound), and white noise at 20–35 dB SNR. All features are recomputed from the augmented audio. The final training set has 15,654 clips (11,086 happy, 4,568 sad).

We augment both moods at the same rate on purpose. An earlier version augmented only the sad clips, to balance the classes. That made the augmentation artifacts a cue for the label: they appeared in 59% of sad clips and none of the happy ones, and on real clips the artifacts alone moved CLAP 0.069 toward the sad caption. Since the model is conditioned on CLAP audio embeddings, this leaked into the conditioning itself. Class balance is now handled only by the sampler (Section 3.6).

### 3.3 Spectrograms and the latent autoencoder

Spectrograms are computed with BigVGAN's own analysis settings (44.1 kHz, FFT and window 2048, hop 512, 128 mel bands), so they match what the vocoder expects. A 5-second clip gives 128 × 430 frames. Log magnitudes are clipped to [−12, 2.5] and scaled to [0, 1] with this fixed range for every song, and the spectrogram is padded to 128 × 432.

The autoencoder has three downsampling blocks (4 × 4 convolution, stride 2, GroupNorm, SiLU) with channel widths 1 → 64 → 128 → 32, and a mirrored decoder ending in a sigmoid. A spectrogram maps to a 32 × 16 × 54 latent, so each latent cell covers 8 mel bands and about 93 ms. This is only a 2× reduction in the number of values. A version with one more downsampling stage (8× reduction) produced audibly distorted audio after vocoding, so we kept the compression mild. The model has 0.40 M parameters and is trained for 100 epochs with Adam (learning rate 10⁻³, batch 64).

A plain mean-squared-error loss gave blurry spectrograms, which the vocoder turns into distorted audio. We instead combine an L1 loss on the spectrogram with L1 losses on its differences along time and frequency, which preserve onsets and harmonic lines, and L1 losses on 2×, 4× and 8× average-pooled versions, which preserve the overall spectral shape (weights 1, 1 and 0.5).

After training, the autoencoder is frozen and all latents are standardized per channel. We first used a single global mean and standard deviation, but the channels' standard deviations differ by a factor of 21 (0.095 to 1.995). With one scale, some channels were already pure noise at a given diffusion step while others still matched the input almost exactly, so the edit-strength setting did not behave consistently. Per-channel standardization fixed this.

### 3.4 Melody representation

We follow Hou et al. (2024) and describe melody by the strongest pitch bins in each frame. The waveform is high-pass filtered at 261.2 Hz (middle C) to reduce bass and kick-drum energy, and a 128-bin CQT (one bin per MIDI note, hop 512) is computed. For each of the 431 frames we keep the indices of the four strongest bins. Each index is embedded with a learned 64-dimensional table, the four embeddings are concatenated, and two strided 1-D convolutions followed by linear interpolation bring the sequence to the latent's 54 time steps, giving a 54 × 256 melody sequence. This encoder (0.53 M parameters) is trained with the DiT.

### 3.5 Mood conditioning with CLAP

We use the LAION-CLAP music checkpoint (HTSAT-base audio encoder), frozen throughout. All embeddings are 512-dimensional unit vectors.

**Audio embeddings in training, text at inference.** The obvious approach is to condition each training clip on the CLAP text embedding of its label. With two labels, however, the network only ever sees two conditioning vectors and can treat them as an arbitrary two-way switch. A model we trained this way did respond to the prompt, but the changes it made were unrelated to mood. We therefore condition each clip on its own CLAP audio embedding, which gives the model 15,654 different vectors spread across the happy and sad regions of CLAP's space. At inference, the text embedding of the target mood takes the place of the audio embedding.

**Alignment.** CLAP's audio and text embeddings are offset from each other: on our data, the mean audio embedding and the mean text embedding have a cosine similarity of only 0.27. We fit a fixed alignment before training. Each modality is first centered on the average of its happy mean and sad mean and renormalized. We then fit a rotation $R$ that maps the centered text mood directions onto the centered audio mood directions (orthogonal Procrustes; Schönemann, 1966). Because there are only two moods, the data constrain only a couple of directions out of 512, so we regularize the fit toward the identity:

$$
C = \frac{1}{2}\sum_{m \in \{\text{happy},\,\text{sad}\}} \bar{c}^{\,a}_m (\bar{c}^{\,t}_m)^\top + \lambda \lVert C_0 \rVert_2 I = U\Sigma V^\top, \qquad R = UV^\top, \qquad \lambda = 0.1,
$$

where $\bar{c}^{\,a}_m$ and $\bar{c}^{\,t}_m$ are the centered audio and text means for mood $m$ and $C_0$ is the first term. Text vectors are centered, rotated by $R$ and renormalized; audio vectors are only centered and renormalized. To check the alignment, we measure each caption's cosine to the audio mean of its own mood minus its cosine to the other mood's. On the training data, this margin rises from +0.22 to +1.24 for the happy caption and from +0.03 to +1.04 for the sad caption.

**Prompts.** The inference prompts are "a happy and uplifting piece of music" and "a sad and melancholic piece of music". So that the text path is trained directly and not only through the audio–text substitution, we wrote 59 paraphrases per mood. They describe valence rather than energy, they mention the same instruments and genres for both moods (so that, for example, "cello" does not become a cue for sad), and each was checked to be closer in CLAP to its own mood's caption than to the other's. Several gentle positive phrasings, such as "serene and blissful", initially fell on the sad side and were reworded.

**Projection.** A linear layer maps the aligned 512-dimensional vector to four 256-dimensional tokens, followed by LayerNorm. These tokens are the keys and values for cross-attention. For classifier-free guidance, the "no condition" input is the CLAP text embedding of the empty string, left unaligned.

### 3.6 Diffusion Transformer and training

Each of the 16 × 54 latent cells becomes a token, projected to 256 dimensions with a learned position embedding. The model has eight DiT blocks, each with self-attention, cross-attention to the mood tokens, and an MLP, all with adaptive LayerNorm driven by the diffusion timestep (Peebles & Xie, 2023). The first four blocks have ControlNet partners. Each partner receives the current tokens plus the melody sequence (the same melody vector is added to all 16 frequency rows of a time step) and adds its output back to the main stream through a zero-initialized linear gate. Unlike the usual ControlNet setup, there is no pretrained base model: the main blocks and the ControlNet branch are trained together from random initialization, and the zero gates simply keep early training stable. The model has 20.3 M trainable parameters.

We use a 1000-step cosine noise schedule (Nichol & Dhariwal, 2021) and train with the v-prediction target (Salimans & Ho, 2022). For each training clip, the conditioning vector is its aligned CLAP audio embedding plus a small random perturbation (norm 0.1), which helps the model tolerate the remaining difference between aligned text and audio vectors. With probability 0.25 it is replaced by the aligned text embedding of a random paraphrase of the clip's mood, and with probability 0.2 by the null condition. Separately, the melody input is set to zero with probability 0.15, so the model also learns to work without melody. Batches are drawn so that each mood makes up half of each batch in expectation. We train with AdamW (learning rate 10⁻⁴, batch 64) for 14,149 steps, about 58 passes over the data, with no learning-rate schedule, EMA or gradient clipping. Training takes about 44 minutes on one NVIDIA B200 GPU on the University of Florida HiPerGator cluster. Appendix A lists all hyperparameters.

### 3.7 Editing

To edit a clip, we encode it to a latent $z_0$, extract its melody, and build two conditions: the aligned text embedding of the target caption and the null condition. An edit strength $s$ sets the starting noise level $\tau_0 = sT$. We run the deterministic DDIM sampler in reverse for 50 steps from $z_0$ up to $\tau_0$, using the null condition and the melody, to obtain an inverted latent. We then run the same 50 steps back down with classifier-free guidance (Ho & Salimans, 2022) applied to the mood condition only:

$$
\hat{v} = f_\theta(z_\tau, \tau, h_\varnothing, \mu) + w\,\big(f_\theta(z_\tau, \tau, h, \mu) - f_\theta(z_\tau, \tau, h_\varnothing, \mu)\big),
$$

where $h$ and $h_\varnothing$ are the mood and null tokens and $\mu$ is the melody, which both terms receive so that guidance never amplifies it. With $w = 0$, denoising retraces the inversion and should return the input, which gives a direct check that inversion works. The final latent is decoded and vocoded. Unless stated otherwise, edits use $s = 0.8$ and $w = 7$.

Earlier versions used SDEdit, which starts from $z_0$ mixed with fresh random noise. As Section 5.2 shows, that noise removes much of the melody before the model has a chance to restore it. At the strengths where the mood change became audible, SDEdit edits no longer sounded like music: harmonic lines broke apart into broadband noise. Inversion avoids this because the starting latent still encodes the song.

**Differences from Hou et al. (2024).** We keep their DiT with a ControlNet melody branch, top-k CQT melody features, v-prediction, editing by partially re-running diffusion on the input, and guidance on text only. We differ in using a small purpose-trained autoencoder with BigVGAN instead of a large pretrained audio model, conditioning through CLAP audio embeddings with an alignment step instead of direct text conditioning, training everything from scratch, using DDIM instead of DPM-Solver++, and starting edits from DDIM inversion.

## 4 Evaluation

### 4.1 Test set

Each test song is represented by its 15–20 s window, labeled with the same rule as in training (songs in the dead band are excluded). The main evaluation uses 20 songs at evenly spaced positions in the corpus (10 happy, 10 sad), and each is edited toward both moods, for 40 edits. Some diagnostics use a separate set of 30 songs (15 per mood).

### 4.2 Mood judges

**CLAP classifier.** We classify a clip as whichever caption its CLAP audio embedding is closer to. On 100 unedited clips this agrees with the DEAM labels 83% of the time (56/67 happy, 27/33 sad). CLAP is also what conditions the model, so it is not an independent judge, and we treat its scores on edits as indicative rather than conclusive.

**Valence probe.** To measure how far valence moved rather than only which side an edit fell on, we fit a ridge regression from CLAP audio embeddings to DEAM valence on 500 songs, choosing the regularization on a 20% held-out split. It reaches a held-out $R^2$ of 0.27 (Pearson $r$ = 0.54), which is weak for single clips but enough to detect average shifts over many edits. Refitting the probe on audio that has passed through the autoencoder and vocoder gives $R^2$ = 0.31, so it still works on resynthesized audio.

### 4.3 Separating the prompt's effect from resynthesis

Every edit passes through the autoencoder and vocoder, which by itself lowers CLAP similarity to both captions by about 0.11. Comparing an edit only to the original therefore mixes the mood effect with this loss. Our main metrics instead compare the two edits of the same song, which share the same source, the same resynthesis and the same random seed. We define a clip's sadness score as its CLAP cosine to the sad caption minus its cosine to the happy caption.

- **CLAP margin.** For an edit toward mood $m$, its cosine to caption $m$ minus the average cosine of the song's two edits to caption $m$. Positive values mean the edit sounds more like $m$ than its sibling does.
- **Edit separation.** The difference in sadness score between a song's sad edit and its happy edit, compared with the gap in mean sadness score between real sad and real happy clips.
- **Transfer.** The fraction of edits that the CLAP classifier assigns to the target mood, reported separately for edits that must change the song's mood and edits that keep it.
- **Push.** In the diagnostic sweeps, the change in sadness score caused by the prompt, measured against a $w = 0$ edit with the same seed and given as a percentage of the real happy–sad gap.

We treat the song as the unit of analysis and report effects with two standard errors.

### 4.4 Melody preservation

We use two measures. Chroma similarity is the mean frame-wise cosine between 12-bin CQT chroma vectors of the original and the edit. It gives partial credit and is generous: two unrelated songs already score about 0.67. In the sweeps we therefore rescale it so that the autoencoder–vocoder reconstruction scores 100% and an unrelated song scores 0%, and call this "melody kept". To compare with Hou et al. (2024), we also compute their melody accuracy: the fraction of frames in which the strongest pitch class (the top CQT bin of the high-passed melody features, modulo 12) matches between the original and the edit.

As a rough check that edits stay musical, we also report the share of spectral energy that harmonic–percussive separation (Fitzgerald, 2010) assigns to the harmonic part, relative to the reconstruction. Garbled edits replace sustained notes with broadband noise and score below 1.

## 5 Results

### 5.1 Main evaluation

Table 2 summarizes the 40 edits at the default setting ($s = 0.8$, $w = 7$, null-prompt inversion).

**Table 2.** Main results (20 songs, 40 edits). Ranges are ± 2 standard errors over songs.

| Measure | Toward happy | Toward sad | Overall |
|---|---|---|---|
| CLAP margin | +0.047 | +0.021 | +0.034 ± 0.009 |
| Valence probe margin (target direction) | +0.101 | +0.101 | +0.101 ± 0.019 |
| Songs where valence moved the right way | 100% | 100% | 100% |
| Transfer (CLAP classifies as target) | 50% | 95% | 72.5% |
| Chroma similarity (ceiling 0.933) | 0.923 | 0.910 | 0.917 |
| Melody accuracy (ceiling 63.9%) | | | 57.2% |

The prompt clearly moves the audio. The happy edit of a song scored as less sad than its sad edit for 19 of 20 songs by CLAP and for all 20 by the valence probe, and both overall margins are well above two standard errors. The two edits of a song ended up 0.135 ± 0.036 apart in sadness score, compared with a gap of 0.115 ± 0.038 between 40 real happy and 40 real sad clips. The average distance between a song's two edits is therefore about 117% of a real mood difference, but the uncertainty is large: combining both error bars, the ratio could plausibly be anywhere from 65% to 223%.

Whether an edit actually crosses to the other mood is a harder test. When the target differed from the song's original mood, CLAP classified the edit as the target 60% of the time (12 of 20). When the target matched the original mood, 85% of edits kept their classification. Most of the failures were edits toward happy: only half were classified as happy, compared with 95% of edits toward sad.

Melody survives well relative to what the codec allows. Chroma similarity is 0.917 against a reconstruction ceiling of 0.933. Melody accuracy is 57.2%, while the autoencoder and vocoder alone, with no diffusion, reach 63.9%. Hou et al. report 56.6% for their model and 42–45% for MusicGen-melody. Those numbers come from a different dataset (Song Describer, longer clips, a different audio codec), so the two results should not be read as a direct comparison. The more useful reference is our own ceiling: about 90% of the melody accuracy the codec permits is preserved through the edit.

### 5.2 DDIM inversion versus SDEdit noise

Table 3 compares the two ways of starting an edit on the 30-song diagnostic set. The "round trip" column runs the edit with $w = 0$, so it measures only how much of the melody the starting point loses.

**Table 3.** SDEdit noise vs. DDIM inversion ($w = 7$, 30 songs). Push is given as a percentage of the real happy–sad gap.

| Start | Strength | Melody kept, round trip | Happy push | Sad push | Melody kept, edit |
|---|---|---|---|---|---|
| SDEdit | 0.6 | 78% | 27% | 61% | 78% |
| SDEdit | 0.8 | 71% | 69% | 77% | 64% |
| SDEdit | 1.0 | 49% | 73% | 46% | 37% |
| Inversion | 0.6 | 99% | 15% | 50% | 95% |
| Inversion | 0.8 | 98% | 54% | 92% | 89% |
| Inversion | 1.0 | 101% | 41% | 104% | 66% |

With SDEdit, the unguided round trip already loses 22–51% of the melody, and the loss grows with strength. With inversion, the round trip keeps 98–101% at every strength, which confirms that the inversion is faithful. At strength 0.8, inversion keeps 89% of the melody while pushing toward sad more strongly than SDEdit does (92% vs. 77%), but pushes toward happy less strongly (54% vs. 69%). Part of SDEdit's apparent push likely comes from discarding the source, which makes room for the prompt but also produces the garbled audio we heard. Melody accuracy on the six songs we saved for listening shows the same pattern: the round trip scores 68.0% with inversion and 43.3% with SDEdit at strength 0.8, against 68.6% for the plain reconstruction and 7.9% for an unrelated song.

### 5.3 Guidance and strength under inversion

Because the happy push was weak, we tested stronger guidance and strength, and also inverting under the source song's own mood caption instead of the null prompt (Table 4).

**Table 4.** Guidance scale and strength with inversion (30 songs). "Happy transfer" is the share of sad songs whose happy edit CLAP classifies as happy.

| Inversion prompt | $s$ | $w$ | Happy push | Sad push | Happy transfer | Harmonic ratio | Melody kept |
|---|---|---|---|---|---|---|---|
| null | 0.8 | 5 | 32% | 74% | 40% | 1.03 | 93% |
| null | 0.8 | 7 | 54% | 92% | 40% | 1.03 | 89% |
| null | 0.8 | 10 | 88% | 103% | 80% | 1.03 | 81% |
| null | 0.9 | 5 | 66% | 90% | 53% | 1.02 | 91% |
| null | 0.9 | 7 | 96% | 99% | 80% | 1.01 | 84% |
| null | 0.9 | 10 | 115% | 109% | 80% | 0.98 | 73% |
| source caption | 0.8 | 7 | 58% | 81% | 53% | 1.02 | 90% |
| source caption | 0.9 | 7 | 99% | 82% | 80% | 1.00 | 85% |

Stronger settings make the push more symmetric. At $s = 0.9$, $w = 7$, the happy and sad pushes are nearly equal (96% and 99%) and 80% of happy edits are classified as happy, at a cost of five points of melody kept compared with the default. The harmonic ratio stays near 1 up to $w = 7$ and only starts to fall at $w = 10$, which suggests the edits do not become noise-like within this range. Inverting under the source caption gave a slightly larger happy push and a smaller sad push than the null prompt, with similar melody retention, so it did not change the overall picture. These sweeps were run after the default was set, and $s = 0.9$, $w = 7$ is a reasonable alternative operating point if a stronger, more balanced edit is worth some melody loss. We have not yet run the full evaluation of Section 5.1 at that setting.

### 5.4 Where the happy/sad asymmetry does not come from

We ran three checks to locate the weak happy push.

**The codec.** We measured the real happy–sad gap in CLAP sadness score at each stage of the pipeline on 30 songs. The gap is 0.124 on the raw audio, 0.133 after the vocoder alone and 0.115 after the autoencoder and vocoder (93% of raw), with classification AUC falling only from 0.87 to 0.83. The codec keeps most of the mood information, so it is not the main bottleneck.

**The melody branch.** One possibility is that the ControlNet ties the output so closely to the input melody that the model cannot make it sound happier. Thanks to melody dropout in training, we can scale the melody input down to zero without leaving the training distribution. Using an SDEdit start at strength 0.8, removing the melody entirely changed the happy push only from 69% to 71% of the real gap, while chroma similarity fell from 0.882 to 0.859 (Table 5). The sad push rose from 77% to 97%. Melody conditioning therefore costs some sad push but does not explain the weak happy push.

**Table 5.** Scaling the melody input (SDEdit start, $s = 0.8$, $w = 7$, 30 songs).

| Melody scale | Happy push | Sad push | Chroma |
|---|---|---|---|
| 1.0 | 69% | 77% | 0.882 |
| 0.5 | 70% | 85% | 0.874 |
| 0.0 | 71% | 97% | 0.859 |

**The text pathway.** On an earlier checkpoint (before the augmentation and standardization fixes described in Section 3), we compared three conditioning vectors on 10 songs with everything else held fixed: the aligned caption, the caption without alignment, and the average aligned audio embedding of 24 real clips of the target mood. The aligned caption produced a CLAP margin of +0.020 ± 0.007, at least as large as the real-audio average (+0.016 ± 0.006), and larger than the unaligned caption (+0.014 ± 0.005). The text path therefore does about as well as in-distribution audio conditioning, and the alignment helps. In all three arms, however, the margin for happy targets was close to zero while the margin for sad targets was clearly positive. The asymmetry is present even with real happy audio as the condition, which points to the model or the data rather than to the prompt.

We also checked whether edits change the key mode, using a major-minus-minor chroma template score. Across all settings in the melody sweep, the prompt did not move this score reliably in either direction, while real happy and sad songs differ by 0.063 on it. Whatever the model changes to make a clip sound happier or sadder, it is mostly not the mode. This is consistent with the ControlNet keeping the pitch content fixed, but it also means the system cannot use one of the strongest musical cues for valence.

## 6 Discussion and Limitations

The system does what it was designed to do in a limited sense. The text prompt reliably moves CLAP's and the valence probe's readings of the audio in the requested direction, and it does so while keeping most of the melody that the codec allows. The edits are not symmetric, though. Making music sadder works much better than making it happier at the default setting, and only a stronger setting closes that gap, at some cost to melody. We think the training data contribute to this. Sad clips are less than a third of the labeled data, and many happy DEAM clips are only mildly positive, so the model may have a sharper picture of "sad" than of "happy". The checks in Section 5.4 rule out the codec, the melody branch and the text alignment as the main cause.

Several limitations apply to all of these results:

- **The mood judge is not independent.** CLAP both conditions the model and scores it. The valence probe is also built on CLAP embeddings. An edit could learn to move CLAP's embedding without changing how the music sounds to a person. A listening test is needed before any claim about perceived mood can be made.
- **Small evaluation.** The main evaluation covers 20 songs and the diagnostics 10–30, and the melody accuracy figures in Section 5.2 use only six songs. The uncertainty on the edit-separation ratio (65% to 223%) shows how little 20 songs can pin down.
- **Short clips and one dataset.** All clips are 5 seconds long and come from DEAM, whose valence ratings lean positive. Results may not hold for longer excerpts or other music.
- **The codec sets a ceiling.** The autoencoder and vocoder cost about 0.11 in CLAP similarity, 0.07 in chroma similarity and about 36 points of melody accuracy before diffusion changes anything. Some of the remaining loss in the edits is from resynthesis, not from editing.
- **Comparison with Hou et al.** We use their melody accuracy metric but not their data, clip length, codec or other metrics (FD, KL, CLAP score), so our numbers cannot be placed in their results table.
- **Diagnostics on older checkpoints.** The conditioning-vector comparison was run before the final training changes, and the melody sweep used SDEdit rather than inversion. Their conclusions are consistent with the final model's behavior, but they have not been repeated on it.

## 7 Conclusion

We rebuilt the DiT + ControlNet music editor of Hou et al. (2024) at a small scale and adapted it to happy/sad mood editing with CLAP conditioning. Three choices made the largest difference in our experiments: conditioning on per-clip CLAP audio embeddings with a simple alignment to text, augmenting both moods equally so that artifacts do not encode the label, and starting edits from DDIM inversion instead of random noise. With these, the prompt moves the audio toward the requested mood on nearly every test song, and the edited audio keeps about 90% of the melody accuracy that the codec allows. Edits toward happy remain weaker than edits toward sad. Next steps are a listening study to test whether the changes CLAP detects are audible as mood, a full evaluation at the stronger $s = 0.9$ setting, and training on more balanced or more strongly labeled data to address the asymmetry.

## References

- Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLOS ONE*, 12(3).
- Fitzgerald, D. (2010). Harmonic/percussive separation using median filtering. *Proceedings of the 13th International Conference on Digital Audio Effects (DAFx)*.
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *arXiv:2207.12598*.
- Hou, S., et al. (2024). Editing music with melody and text: Using ControlNet for diffusion transformer. *arXiv:2410.05151*.
- Lee, S., Ping, W., Ginsburg, B., Catanzaro, B., & Yoon, S. (2023). BigVGAN: A universal neural vocoder with large-scale training. *ICLR*.
- Liang, W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. (2022). Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. *NeurIPS*.
- Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). SDEdit: Guided image synthesis and editing with stochastic differential equations. *ICLR*.
- Mokady, R., Hertz, A., Aberman, K., Pritch, Y., & Cohen-Or, D. (2023). Null-text inversion for editing real images using guided diffusion models. *CVPR*.
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV*.
- Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. *ICLR*.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1).
- Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.
- Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. *ICASSP*.
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. *ICCV*.

## Appendix A: Hyperparameters

| Component | Setting | Value |
|---|---|---|
| Audio | Sample rate / clip length / clips per song | 44.1 kHz / 5 s / 6 (15–45 s) |
| Labels | Valence dead band | ±0.1 |
| Augmentation | Pitch / time shift / SNR / copies per clip | ±2 semitones / ±20% / 20–35 dB / 1 (both moods) |
| Mel | FFT / hop / bands / log range | 2048 / 512 / 128 / [−12, 2.5] |
| Autoencoder | Channels / latent shape / standardization | 1-64-128-32 / 32×16×54 / per channel |
| | Loss weights (L1 / gradient / multi-scale) | 1 / 1 / 0.5 |
| | Optimizer / LR / batch / epochs | Adam / 10⁻³ / 64 / 100 |
| Melody | High-pass / CQT bins / hop / top-k | 261.2 Hz / 128 / 512 / 4 |
| CLAP | Checkpoint / dimension / tokens | HTSAT-base music / 512 / 4 |
| | Procrustes regularization / jitter / text probability / paraphrases | 0.1 / 0.1 / 0.25 / 60 per mood |
| DiT | Width / heads / blocks / ControlNet blocks / MLP ratio | 256 / 4 / 8 / 4 / 4 |
| | Trainable parameters | 20.3 M |
| Diffusion | Steps / schedule / target | 1000 / cosine / v |
| | Condition dropout / melody dropout | 0.2 / 0.15 |
| | Optimizer / LR / batch / steps | AdamW / 10⁻⁴ / 64 / 14,149 |
| Editing | Start / sampler steps / strength / guidance | DDIM inversion (null prompt) / 50 / 0.8 / 7 |
| Evaluation | Songs edited / CLAP validation songs / probe songs | 20 / 100 / 500 |
