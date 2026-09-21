# 3. Methodology

## 3.1 Overview

We address *melody-preserving mood editing*: given a short music excerpt $x$ and a target mood $m \in \{\text{happy}, \text{sad}\}$, produce an excerpt $\hat{x}$ that is perceived as having mood $m$ while keeping the pitch content of $x$. Our system follows the design of Hou et al. (2024), who pair a Diffusion Transformer (DiT) with a ControlNet branch that carries melody. We rebuild that design around three additions: a compact mel-spectrogram latent space, mood conditioning drawn from a contrastive language–audio model (CLAP), and an explicit correction for the gap between CLAP's audio and text embedding spaces.

The pipeline has seven stages (Figure 1):

1. **Waveform → mel spectrogram.** A 5 s mono clip at 44.1 kHz is converted to a 128-band log-mel spectrogram using the analysis settings of the BigVGAN vocoder.
2. **Mel → latent.** A convolutional autoencoder compresses the spectrogram into a 2-D latent feature map.
3. **Melody extraction.** Top-$k$ constant-Q transform (CQT) pitch indices are extracted from the waveform and embedded as a frame-level melody sequence.
4. **Mood conditioning.** A frozen CLAP model maps audio (during training) or a text prompt (at inference) into a shared embedding space. The embedding is aligned across modalities and projected into cross-attention tokens.
5. **Latent diffusion.** A DiT with a ControlNet branch denoises the latent, conditioned on mood through cross-attention and on melody through the ControlNet branch.
6. **Latent → mel.** The autoencoder decoder reconstructs the spectrogram.
7. **Mel → waveform.** A pretrained BigVGAN vocoder synthesizes the output audio.

Editing uses SDEdit (Meng et al., 2022). The input latent is partially noised and then denoised under the target mood, with classifier-free guidance applied to the mood condition only.

> **Figure 1 (suggested).** Block diagram. Top row (training): DEAM clip → mel → AE encoder → $z_0$ → noise → DiT+ControlNet → $v$-loss. The CQT melody feeds the ControlNet branch. The CLAP audio embedding passes through alignment and projection into cross-attention. Bottom row (inference): input clip → $z_0$ → partial noising → guided DDIM → AE decoder → BigVGAN → output. The CLAP text embedding of the target caption replaces the audio embedding.

---

## 3.2 Data and Mood Labels

### 3.2.1 Corpus

We use the MediaEval Database for Emotional Analysis in Music (DEAM; Aljanaki et al., 2017). DEAM contains 45 s excerpts with continuous valence and arousal ratings. We use the *dynamic* annotations, which are averaged across annotators and sampled every 500 ms from 15 s onward. We rescale all values to $[-1, 1]$. Of the audio files, 1,802 songs have both valence and arousal series, and every experiment uses all of them.

### 3.2.2 Segmentation and window-level labels

The first 15 s of each excerpt has no annotations, so each song contributes the annotated region 15–45 s. We split this region into six consecutive, non-overlapping 5 s windows, giving 10,812 candidate clips. Each clip $j$ is labeled from its *own* window rather than from a song-level average. Let $\bar{v}_j$ be the mean dynamic valence over the clip's window. The label is

$$
m_j =
\begin{cases}
\text{happy} & \bar{v}_j > \delta \\
\text{sad} & \bar{v}_j < -\delta \\
\varnothing\ \text{(discarded)} & |\bar{v}_j| \le \delta
\end{cases}
\qquad \delta = 0.1 .
$$

Arousal is not used. We use valence alone for two reasons. It is the affective axis that separates "happy" from "sad". In preliminary experiments, CLAP separated arousal-derived categories (e.g., "calm", "energetic") worst of all the mood categories we tried. The dead band $\delta$ removes clips whose valence is too close to neutral to carry a reliable label. DEAM valence ratings are shifted toward positive values, so a clip with slightly negative valence is not perceptually sad. Because labels are computed per window, a single song can contribute clips of both moods.

The dead band removes 2,985 clips (27.6%). The remaining 7,827 labeled clips are imbalanced: 5,543 happy (70.8%) and 2,284 sad (29.2%).

### 3.2.3 Class balancing by waveform augmentation

We bring the minority class up to parity with new augmented clips rather than only reweighting. Each real sad clip receives $n_j$ augmented variants. Let $N_{\max}$ be the size of the majority class and $N_{\text{sad}}$ the number of real sad clips. The target mean is $r = \min(N_{\max}/N_{\text{sad}} - 1,\ 12)$, and $n_j$ is obtained by stochastic rounding of $r$ ($\lfloor r \rfloor$ or $\lceil r \rceil$, with expectation $r$), so the class total hits the target exactly. In our data, $r \approx 1.43$.

Each variant applies three length-preserving transformations in sequence, with parameters drawn independently for every variant:

- **Pitch shift** by $u \sim \mathcal{U}(-2, 2)$ semitones (phase-vocoder pitch shifting).
- **Time shift** by an integer number of samples drawn uniformly from $[-0.2L, 0.2L]$, where $L$ is the clip length. The vacated region is zero-filled rather than wrapped, so the end of the clip is never spliced onto its start.
- **Additive white Gaussian noise** at an SNR drawn from $\mathcal{U}(20, 35)$ dB.

The output is clipped to $[-1, 1]$. The perturbations are kept small so that they do not change perceived valence. Augmented clips go through the full feature pipeline (mel, melody, CLAP embedding) *after* augmentation, so every feature matches the audio it came from. Augmentation adds 3,237 sad clips. The final training set has **11,064 clips: 5,543 happy and 5,521 sad**. The dataset is shuffled once with a fixed seed and cached to disk.

### 3.2.4 Spectral front end

We compute log-mel spectrograms with BigVGAN's own analysis function so that the representation matches what the vocoder expects: 44.1 kHz sampling, STFT window and FFT size 2048, hop 512, and 128 mel bands spanning 0–22.05 kHz. A 5 s clip gives $128 \times 430$ frames. We map log-magnitudes to $[0, 1]$ with a *fixed* global range instead of per-clip statistics, so all songs share one scale:

$$
\tilde{S} = \frac{\operatorname{clip}(S, -12, 2.5) + 12}{14.5}.
$$

Spectrograms are reflect-padded to $128 \times 432$ so that both dimensions divide evenly by the autoencoder's downsampling factor.

---

## 3.3 Latent Autoencoder

A deterministic convolutional autoencoder $(E, D)$ maps spectrograms to a latent grid, and diffusion runs on that grid.

**Architecture.** The encoder has three blocks, each a $4\times4$ convolution with stride 2 followed by GroupNorm (8 groups) and SiLU. The channel widths are $1 \to 64 \to 128 \to 32$. The decoder mirrors the encoder with transposed convolutions ($32 \to 128 \to 64 \to 1$). Its last layer uses a sigmoid to match the $[0,1]$ input range. A $128 \times 432$ spectrogram maps to a latent $z \in \mathbb{R}^{32 \times 16 \times 54}$. Each latent cell covers an $8 \times 8$ patch of mel bins and frames, about 93 ms of audio and 8 mel bands. This is a $2\times$ reduction in the number of values.

We deliberately chose mild compression. In an earlier variant with a fourth downsampling stage ($32 \times 8 \times 27$, $8\times$ compression), the decoder had to invent detail over $16 \times 16$ patches, and the vocoded audio was audibly distorted. The autoencoder has no KL term and no bottleneck that flattens the grid, so the latent keeps its 2-D time–frequency layout for the transformer. It has 0.40 M parameters.

**Reconstruction loss.** A plain MSE loss favors blurry reconstructions, and the vocoder turns blurred spectrograms into distorted audio. We therefore train with a loss built to preserve sharp detail:

$$
\mathcal{L}_{\text{AE}} =
\lambda_1 \lVert \hat{S} - S \rVert_1
+ \lambda_g \big( \lVert \nabla_t \hat{S} - \nabla_t S \rVert_1 + \lVert \nabla_f \hat{S} - \nabla_f S \rVert_1 \big)
+ \lambda_{ms} \sum_{s=1}^{3} \lVert P_{2^s}(\hat{S}) - P_{2^s}(S) \rVert_1 ,
$$

Here $\nabla_t$ and $\nabla_f$ are first differences along time and frequency, $P_{2^s}$ is $2^s \times 2^s$ average pooling, and $\lambda_1 = \lambda_g = 1$, $\lambda_{ms} = 0.5$ (all norms are means). The gradient term keeps harmonic ridges and onsets from being averaged away. The multi-scale term keeps the global spectral envelope. Together they act as a spectrogram-domain counterpart of a multi-resolution STFT loss.

**Training.** Adam, learning rate $10^{-3}$, batch size 64, 100 epochs over all 11,064 training spectrograms. After training, the autoencoder is frozen. All latents are pre-encoded and standardized with a single global scalar mean and standard deviation ($\mu_z = 0.111$, $\sigma_z = 0.284$), so they approximately match the diffusion process's unit-variance noise assumption. At decoding time, the standardization is reversed.

---

## 3.4 Melody Representation

Following Hou et al. (2024, §III-B), we represent melody as the few most energetic pitch bins in each frame.

**Extraction.** The waveform first passes through a second-order Butterworth high-pass filter at 261.2 Hz (middle C), which suppresses bass and kick-drum energy that would otherwise dominate the pitch estimate. We then compute a CQT with 128 bins at 12 bins per octave, starting at $f_{\min} = 8.18$ Hz (MIDI note 0), with hop 512. This gives 431 frames for a 5 s clip, one bin per MIDI pitch. For each frame we keep the indices of the $k = 4$ largest-magnitude bins, ordered by magnitude. This yields $M \in \{1, \dots, 128\}^{4 \times 431}$, with index 0 reserved for padding.

**Encoding.** Each pitch index is embedded with a learned table ($129 \times 64$). The four embeddings of a frame are concatenated into a 256-dimensional vector. Two 1-D convolutions (kernel 4, stride 4, each followed by SiLU) downsample the sequence from 431 to 26 frames. Linear interpolation then resamples it to the latent's temporal width $W = 54$, giving the melody sequence $\mu \in \mathbb{R}^{54 \times 256}$. The melody encoder has 0.53 M parameters and is trained jointly with the DiT.

---

## 3.5 Mood Conditioning with CLAP

### 3.5.1 Conditioning on audio embeddings

Mood is represented in the embedding space of LAION-CLAP (Wu et al., 2023), using the HTSAT-base audio encoder and the publicly released music checkpoint (`music_audioset_epoch_15_esc_90.14`). CLAP is frozen throughout. Audio is resampled to 48 kHz before embedding, and all embeddings are $\ell_2$-normalized vectors in $\mathbb{R}^{512}$.

The obvious approach would condition each clip on the CLAP *text* embedding of its mood label. That hands the conditioning network only two distinct vectors. With two inputs, the trainable projection can degenerate into a two-entry lookup table with no reason to preserve CLAP's semantic geometry. In our early experiments, a model trained this way responded to the prompt but moved the audio in directions unrelated to mood. We instead condition each clip on its own **CLAP audio embedding** $a_j$. This gives 11,064 distinct conditioning vectors, spread across the mood-relevant regions of the space. At inference, the text embedding of the target mood takes the place of the audio embedding. This substitution works because CLAP trains its audio and text towers to map into a shared space.

### 3.5.2 Correcting the modality gap

Contrastively trained audio and text embeddings fall in two offset cones rather than overlapping (Liang et al., 2022). On our corpus, the cosine similarity between the mean audio embedding and the mean text embedding is only about 0.27. A model trained only on audio vectors would therefore receive an out-of-distribution vector at inference. Before diffusion training, we fit a fixed alignment and store it with the model, so training and inference apply the same transform.

*Centering.* For each modality, we compute a class-balanced mean, the average of the per-mood means, so that the majority mood does not pull the center toward itself. For audio,

$$
\mu_a = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}} \operatorname{mean}\{a_j : m_j = m\},
$$

and $\mu_t$ is defined the same way over the training prompt embeddings (§3.5.3). Vectors are centered and renormalized: $\bar{a} = \operatorname{norm}(a - \mu_a)$, $\bar{t} = \operatorname{norm}(t - \mu_t)$.

*Regularized orthogonal Procrustes map.* We learn a rotation $R$ that carries centered text vectors into centered audio space. The only pairing available between modalities is at the mood level: there are no per-clip captions. Under that pairing, the cross-covariance reduces to a sum over mood centroids:

$$
C = \frac{1}{|\mathcal{M}|}\sum_{m} \bar{c}^{\,a}_m \big(\bar{c}^{\,t}_m\big)^{\!\top},
\qquad
\bar{c}^{\,a}_m = \operatorname{mean}\{\bar{a}_j : m_j = m\}.
$$

With two moods, $C$ has rank at most 2 and carries almost no information outside the happy–sad axis. An unregularized Procrustes solution would fill the remaining ~510 dimensions with an arbitrary rotation. We therefore shrink toward the identity:

$$
C' = C + \lambda \lVert C \rVert_2 I, \qquad C' = U \Sigma V^\top, \qquad R = U V^\top, \qquad \lambda = 0.1 .
$$

The resulting map rotates the text mood axis onto the audio mood axis and leaves the other directions nearly unchanged. The two alignment operators are

$$
\phi_a(a) = \operatorname{norm}(a - \mu_a), \qquad \phi_t(t) = \operatorname{norm}\big(R\,(t - \mu_t)\big).
$$

To quantify alignment, we compute each text vector's *margin*: its cosine to the audio centroid of its own mood minus its cosine to the other mood's centroid. On the training corpus (in-sample), the canonical happy caption's margin rises from +0.22 before alignment to +1.24 after, and the sad caption's from +0.03 to +1.04. Averaged over all paraphrases, the margins rise from +0.16 to +0.76 (happy) and from +0.04 to +0.78 (sad).

### 3.5.3 Paraphrase-augmented text path

At inference, each mood is prompted with a fixed caption: "a happy and uplifting piece of music" or "a sad and melancholic piece of music". We use caption-style prompts rather than bare tags because CLAP was trained on natural-language captions. The same strings are used in training, inference, and evaluation. To train the text pathway directly instead of relying entirely on cross-modal transfer, we wrote 60 prompts per mood: the canonical caption plus 59 paraphrases. The paraphrase sets follow three rules:

1. **Paraphrases describe valence, not arousal.** The happy set spans calm-positive ("a happy, smiling and peaceful acoustic song") through energetic-positive, mirroring the valence-only labels.
2. **Instruments and genres are mirrored across moods** (e.g., a happy piano phrase and a sad piano phrase). Otherwise the projection could learn a timbre cue such as "cello ⇒ sad" instead of a mood cue.
3. **Each paraphrase is checked against CLAP.** Its text embedding must be closer to its own mood's canonical caption than to the other mood's. Gentle positive wordings (e.g., "serene and blissful") initially fell on the sad side of CLAP's space. We reworded them with explicit valence words rather than dropping them, so calm-happy music stays represented.

On each training step, a text-substituted example draws one paraphrase uniformly from its mood's set (§3.7).

### 3.5.4 Projection to cross-attention tokens

A trainable linear layer maps an aligned 512-dimensional CLAP vector $c$ to $n = 4$ tokens of width $d = 256$, followed by per-token LayerNorm:

$$
h = \operatorname{LayerNorm}\big(\operatorname{reshape}(W c + b,\ 4 \times 256)\big).
$$

This projection (0.53 M parameters) is the only trainable part of the conditioning pathway. For classifier-free guidance, the null condition $c_\varnothing$ is the $\ell_2$-normalized CLAP text embedding of the empty string. It is deliberately left unaligned. It is a "no condition" marker, not a point in either modality's cone, and the only requirement is that training and inference use the identical vector.

---

## 3.6 Diffusion Transformer with ControlNet

**Tokenization.** Each of the $16 \times 54 = 864$ latent cells becomes one token. Tokens are ordered frequency-row first, projected linearly from 32 to $d = 256$ channels, and summed with a learned absolute positional embedding (truncated-normal initialization, std 0.02).

**Timestep embedding.** The diffusion timestep $\tau$ is encoded with a 256-dimensional sinusoidal embedding (base $10^4$), followed by an MLP ($256 \to 1024 \to 256$, SiLU).

**DiT block.** Each block applies, in order, self-attention, cross-attention to the mood tokens $h$, and an MLP (expansion 4, GELU). Each sub-layer is preceded by an adaptive LayerNorm and wrapped in a residual connection. The adaptive LayerNorm uses no learned affine parameters of its own. Instead, a linear layer on the timestep embedding predicts a scale $\gamma$ and shift $\beta$:

$$
\operatorname{AdaLN}(x, \tau) = (1 + \gamma(\tau)) \odot \operatorname{LN}(x) + \beta(\tau).
$$

The layer is zero-initialized, so each block starts as a standard pre-norm transformer block. Attention uses 4 heads. In cross-attention, queries come from the latent tokens and keys and values from the four mood tokens. We use no dropout.

**ControlNet branch.** The main stream has $N = 8$ DiT blocks $B_1, \dots, B_8$. The first four are paired with ControlNet blocks $B^c_1, \dots, B^c_4$, which have the same architecture, and with zero-initialized linear gates $Z_i$. The melody sequence $\mu \in \mathbb{R}^{54 \times 256}$ is broadcast across all 16 frequency rows, giving $\mu' \in \mathbb{R}^{864 \times 256}$, because one time frame's melody applies to every frequency band. For $i \le 4$,

$$
x \leftarrow B_i(x, \tau, h) + Z_i\big(B^c_i(x + \mu', \tau, h)\big),
$$

and for $i > 4$, $x \leftarrow B_i(x, \tau, h)$. A final LayerNorm and a linear projection back to 32 channels give the prediction, which is reshaped to $32 \times 16 \times 54$. Because the gates $Z_i$ start at zero, the melody pathway contributes nothing at initialization and grows in as training proceeds.

The standard ControlNet setting attaches a trainable branch to a pretrained, frozen base network. Here, **the base DiT and the ControlNet branch are trained jointly from random initialization**. The zero gates are kept for their stabilizing effect early in training.

The DiT has 19.2 M parameters: 11.6 M in the main blocks, 5.8 M in the ControlNet blocks, and 0.26 M in the gates. Together with the melody encoder and conditioning projection, the model has **20.3 M trainable parameters**.

---

## 3.7 Diffusion Training

**Forward process and objective.** We use $T = 1000$ timesteps with the cosine noise schedule of Nichol & Dhariwal (2021) ($s = 0.008$, $\beta_\tau$ clipped to $[10^{-4}, 0.999]$). Given a standardized latent $z_0$, noise $\epsilon \sim \mathcal{N}(0, I)$, and $\tau \sim \mathcal{U}\{0, \dots, T-1\}$,

$$
z_\tau = \sqrt{\bar\alpha_\tau}\, z_0 + \sqrt{1 - \bar\alpha_\tau}\, \epsilon .
$$

The network is trained with the $v$-prediction objective (Salimans & Ho, 2022):

$$
v_\tau = \sqrt{\bar\alpha_\tau}\, \epsilon - \sqrt{1 - \bar\alpha_\tau}\, z_0,
\qquad
\mathcal{L} = \mathbb{E}\big\lVert f_\theta(z_\tau, \tau, h, \mu) - v_\tau \big\rVert_2^2 .
$$

**Conditioning mixture.** For each training example $j$ with mood $m_j$, the conditioning vector is built in three stages:

1. *Audio with jitter.* Starting from the aligned audio embedding, $c_j = \operatorname{norm}\big(\phi_a(a_j) + \sigma\, u\big)$, where $u$ is a random unit vector and $\sigma = 0.1$. The jitter makes the model tolerant of the small residual offset between aligned text and audio vectors.
2. *Text substitution.* With probability $p_{\text{text}} = 0.25$, $c_j$ is replaced by $\phi_t(\mathrm{CLAP}_T(p))$, where $p$ is drawn uniformly from the paraphrase set of mood $m_j$. This trains the text pathway directly.
3. *Condition dropout.* Independently, with probability $p_\varnothing = 0.2$, $c_j$ is replaced by $c_\varnothing$ to train the unconditional model used for guidance. We use a rate above the common 0.1 because an under-trained unconditional branch makes the guidance direction noisy.

In expectation, about 60% of examples are conditioned on jittered audio embeddings, 20% on text paraphrases, and 20% on the null condition. The melody condition is never dropped.

**Mood-balanced sampling.** Minibatches are drawn with replacement, and each clip is weighted by $1/N_{m_j}$, so both moods have equal expected share in every batch regardless of any residual imbalance.

**Optimization.** AdamW (learning rate $10^{-4}$, PyTorch default weight decay), batch size 64, 10,000 steps (about 58 passes over the training set), constant learning rate. The autoencoder and CLAP are frozen. The DiT, ControlNet branch, melody encoder, and conditioning projection are trained jointly. We use no EMA and no gradient clipping.

---

## 3.8 Mood Editing at Inference

Given an input clip $x$ (5 s, mono, 44.1 kHz) and a target mood $m$:

1. **Encode.** Compute $\tilde{S}$ (§3.2.4), $z_0 = (E(\tilde{S}) - \mu_z)/\sigma_z$, and the melody sequence $\mu$ (§3.4).
2. **Condition.** $h = g\big(\phi_t(\mathrm{CLAP}_T(\text{caption}_m))\big)$ and $h_\varnothing = g(c_\varnothing)$, where $g$ is the projection of §3.5.4.
3. **Partial noising (SDEdit).** Given an edit strength $s \in (0, 1)$, set $\tau_0 = \operatorname{clip}(\lfloor sT \rfloor, 1, T-1)$ and $z_{\tau_0} = \sqrt{\bar\alpha_{\tau_0}}\, z_0 + \sqrt{1 - \bar\alpha_{\tau_0}}\, \epsilon$. The strength $s$ trades fidelity to the input ($s \to 0$) against freedom to change it ($s \to 1$).
4. **Guided denoising.** We run $K = 50$ steps on a grid of timesteps evenly spaced from $\tau_0$ to 0. With a fixed step count, edits at different strengths receive the same number of solver steps and are directly comparable. At each step, classifier-free guidance (Ho & Salimans, 2022) with scale $w$ is applied **to the mood condition only**. The melody condition is supplied to both branches, so it is never amplified:
   $$
   \hat{v} = f_\theta(z_\tau, \tau, h_\varnothing, \mu) + w\big(f_\theta(z_\tau, \tau, h, \mu) - f_\theta(z_\tau, \tau, h_\varnothing, \mu)\big).
   $$
   The update is the deterministic DDIM step (Song et al., 2021, $\eta = 0$) written in $v$-parameterization:
   $$
   \hat{z}_0 = \sqrt{\bar\alpha_\tau}\, z_\tau - \sqrt{1-\bar\alpha_\tau}\, \hat{v}, \quad
   \hat{\epsilon} = \sqrt{1-\bar\alpha_\tau}\, z_\tau + \sqrt{\bar\alpha_\tau}\, \hat{v}, \quad
   z_{\tau'} = \sqrt{\bar\alpha_{\tau'}}\, \hat{z}_0 + \sqrt{1-\bar\alpha_{\tau'}}\, \hat{\epsilon}.
   $$
   The implementation also supports stochastic DDIM ($\eta > 0$) and a scalar $\alpha$ applied to $\mu$, which we use only in the ablations of §3.9.4.
5. **Decode.** De-standardize the latent, decode with $D$, crop to $128 \times 430$, invert the mel normalization, synthesize with the pretrained BigVGAN v2 vocoder (44.1 kHz, 128 bands, 512× upsampling; Lee et al., 2023), and clamp to $[-1, 1]$.

Unless stated otherwise, the edits in our evaluation use $s = 0.5$, $w = 1.5$, and $K = 50$.

**Relation to Hou et al. (2024).** We keep the core of their design: a DiT with a ControlNet branch, top-$k$ CQT melody conditioning, $v$-prediction, SDEdit-style editing, and guidance on text only. We depart from it in the following ways:

- A small, purpose-trained mel autoencoder and BigVGAN vocoder replace a large pretrained latent audio model.
- Conditioning on CLAP audio embeddings, with modality-gap alignment and paraphrase augmentation, replaces direct text conditioning.
- The base DiT and the ControlNet branch are trained jointly from scratch.
- A DDIM sampler replaces DPM-Solver++.
- The default guidance scale is much lower (1.5 rather than 7). Stronger guidance is examined in §3.9.4.

---

## 3.9 Evaluation Protocol

We evaluate on two axes that pull against each other: **mood transfer** (did the edit move toward the target mood?) and **melody preservation** (is the pitch content intact?). Objective judges are only trustworthy if they are validated first, so each judge is tested on real, unedited DEAM audio before its scores on edits are used.

**Evaluation clips.** Each evaluation song is represented by its 15–20 s window, the first annotated window. Its ground-truth mood comes from that window's mean valence under the same dead band as in training, and songs inside the band are excluded. Edit evaluation uses 20 songs chosen at evenly spaced indices from the annotated corpus. Each song is edited toward *both* moods, giving 40 edits.

### 3.9.1 Validating CLAP as a mood judge

We use the same frozen CLAP model and canonical captions as the mood judge. For an audio embedding $e$, the predicted mood is $\arg\max_{m} \cos(e, t_m)$, where $t_m$ is the CLAP text embedding of caption $m$. On 100 unedited clips, we report top-1 accuracy against the valence-derived labels. The judge is considered usable if accuracy exceeds chance (0.5) by at least 0.15. CLAP reaches **0.83**.

CLAP is also used for conditioning, so it is not independent of the model. The chroma metric (§3.9.3) is an independent check on melody preservation. There is no fully independent automatic check on mood, and listening tests remain the definitive measure.

### 3.9.2 A continuous valence probe

The argmax transfer metric only records which of two buckets an edit falls into. To measure *how far* valence moved, we train a linear probe that predicts continuous valence from frozen CLAP audio embeddings. It uses 500 annotated songs, including dead-band songs so that the middle of the valence range is covered. Embeddings are standardized per dimension. We fit ridge regression with an unpenalized bias, choosing $\lambda \in \{0.1, 1, 3, 10, 30, 100, 300\}$ by $R^2$ on a random 20% held-out split, then refit on all 500 songs with the selected $\lambda$. We report the held-out $R^2$ and Pearson $r$, and treat the probe as reliable only if $R^2 \ge 0.15$. The probe reaches held-out $R^2 = 0.27$ ($r = 0.54$). This is only moderate accuracy per clip, but it is enough to detect shifts in the *mean* valence across many edits.

### 3.9.3 Edit metrics

For an original clip $x$, its edit $\hat{x}_m$ toward mood $m$, CLAP audio embeddings $e_x$ and $e_{\hat{x}}$, and probe $f$:

| Metric | Definition | Measures |
|---|---|---|
| CLAP gain | $\cos(e_{\hat{x}}, t_m) - \cos(e_x, t_m)$ | Movement toward the target caption |
| Transfer success | $\mathbb{1}\big[\arg\max_{m'} \cos(e_{\hat{x}}, t_{m'}) = m\big]$ | Edit is classified as the target mood |
| Chroma similarity | Mean over frames of the cosine similarity between 12-bin CQT chroma vectors of $x$ and $\hat{x}$ | Melody and harmony preservation |
| Directed valence shift | $s_m\,\big(f(e_{\hat{x}}) - f(e_x)\big)$, with $s_{\text{happy}} = +1$, $s_{\text{sad}} = -1$ | Valence moved in the intended direction |
| Valence direction accuracy | $\mathbb{1}\big[s_m\,(f(e_{\hat{x}}) - f(e_x)) > 0\big]$ | Sign of the valence change |

We report all metrics per target mood and overall. A successful edit scores high on mood transfer *and* keeps chroma similarity close to 1.

### 3.9.4 Conditioning diagnostics

Every edit passes through the autoencoder and vocoder, and this resynthesis lowers CLAP similarity to *any* caption by a roughly constant amount. Absolute CLAP gain therefore mixes the mood effect with this resynthesis penalty. We use three controlled diagnostics to separate the two. In each, sampling noise is fixed per song (the random seed is reset before every edit of that song), so differences between edits are caused only by the variable being tested.

1. **Guidance sweep.** A grid over $w \in \{1, 3, 5, 7\}$ and $s \in \{0.6, 0.8\}$. For each setting, we measure how much edits of the same song toward different moods differ from each other: mean pairwise waveform RMS difference and mean pairwise CLAP cosine distance. We also report gain, transfer, and chroma. If the edits do not differ at all, the conditioning has no effect. If they differ but gain is not positive, the model responds to the condition but not along the mood direction.

2. **Melody-strength ablation.** The melody embedding is scaled by $\alpha \in \{0, 0.25, 0.5, 1\}$ (with $w = 5$, $s = 0.6$). We also report the ratio of the mean melody-token norm to the mean latent-token norm. This tests whether ControlNet melody conditioning pins the output to the input and blocks mood changes. If mood metrics improve as $\alpha$ falls, melody is the bottleneck. If they stay flat while chroma similarity falls, it is not.

3. **Conditioning-vector probe.** With the same checkpoint, songs, and noise, we vary only the conditioning vector, across three arms:
   - **text**: the canonical caption passed through $\phi_t$, exactly as at inference;
   - **text-raw**: the same caption with alignment bypassed;
   - **audio**: the aligned centroid of 24 real clips of the target mood, drawn from songs disjoint from the test songs. This is an in-distribution reference that removes the modality gap entirely.

   The primary metric compares a song's edits *with each other*, which cancels the shared resynthesis penalty:
   $$
   \text{cond\_margin}(m) = \cos(e_{\hat{x}_m}, t_m) - \frac{1}{|\mathcal{M}|}\sum_{m'} \cos(e_{\hat{x}_{m'}}, t_m).
   $$
   A positive value means the edit aimed at $m$ sounds more like $m$ than the song's edits do on average. Because the edits of one song share audio and noise, we treat the *song* as the independent unit. We report the per-song mean with its standard error and count an effect as established only when $n \ge 3$ songs and the mean exceeds two standard errors. We also report *paired correctness*: the number of songs for which every mood's edit is classified as its own target.

   Comparing the arms locates a failure. If the audio arm has no effect, the conditioning pathway itself is not working. If the audio arm works but the text arm does not, the modality gap is the blocker. If the text arm reaches at least 80% of the audio arm's effect, the gap is effectively closed. The difference between text and text-raw measures what the alignment contributes.

---

## 3.10 Implementation Details

Training runs on a single NVIDIA B200 GPU (183 GB) on the University of Florida HiPerGator cluster. Diffusion training proceeds at about 5.4 steps/s, so 10,000 steps take about 31 minutes. The software stack is PyTorch 2.7.0 (CUDA 12.8), librosa 0.11, and laion-clap. Decoded mel spectrograms, melody indices, and CLAP embeddings are cached after the first pass over the data. Training checkpoints (model, optimizer, and step) are written atomically at fixed intervals, so interrupted jobs resume exactly. Dataset shuffling, augmentation, the probe split, and evaluation sampling all use fixed seeds (0).

**Table 1. Hyperparameters.**

| Component | Setting | Value |
|---|---|---|
| Audio | Sample rate / clip length / clip offset | 44.1 kHz / 5 s / 15 s |
| | Clips per song | 6 (15–45 s) |
| Labels | Valence dead band $\delta$ | 0.1 |
| Augmentation | Pitch / time shift / SNR | ±2 st / ±20% / 20–35 dB |
| | Max variants per clip | 12 |
| Mel | $n_{\text{fft}}$ / hop / bands / range | 2048 / 512 / 128 / [−12, 2.5] |
| Autoencoder | Channels / latent shape | 1-64-128-32 / 32×16×54 |
| | Loss weights (L1 / grad / multi-scale) | 1 / 1 / 0.5 |
| | Optimizer / LR / batch / epochs | Adam / 1e−3 / 64 / 100 |
| Melody | High-pass / CQT bins / bins per octave / $f_{\min}$ / hop / $k$ | 261.2 Hz / 128 / 12 / 8.18 Hz / 512 / 4 |
| CLAP | Checkpoint / dim | HTSAT-base music / 512 |
| | Conditioning tokens | 4 |
| | Alignment / Procrustes $\lambda$ | center + map / 0.1 |
| | Audio jitter $\sigma$ / $p_{\text{text}}$ / paraphrases per mood | 0.1 / 0.25 / 60 |
| DiT | Width / heads / blocks / ControlNet blocks / MLP ratio | 256 / 4 / 8 / 4 / 4 |
| | Trainable parameters | 20.3 M |
| Diffusion | $T$ / schedule / target | 1000 / cosine / $v$ |
| | Condition dropout $p_\varnothing$ | 0.2 |
| | Optimizer / LR / batch / steps | AdamW / 1e−4 / 64 / 10,000 |
| Inference | Sampler / steps / $\eta$ | DDIM / 50 / 0 |
| | Edit strength $s$ / guidance $w$ | 0.5 / 1.5 |
| Evaluation | Songs edited / CLAP validation songs / probe songs | 20 / 100 / 500 |

---

## References

- Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLOS ONE*, 12(3).
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *arXiv:2207.12598*.
- Hou, S., et al. (2024). Editing music with melody and text: Using ControlNet for diffusion transformer. *arXiv:2410.05151*.
- Lee, S., Ping, W., Ginsburg, B., Catanzaro, B., & Yoon, S. (2023). BigVGAN: A universal neural vocoder with large-scale training. *ICLR*.
- Liang, W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. (2022). Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. *NeurIPS*.
- Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). SDEdit: Guided image synthesis and editing with stochastic differential equations. *ICLR*.
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV*.
- Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. *ICLR*.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1).
- Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.
- Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. *ICASSP*.
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. *ICCV*.
