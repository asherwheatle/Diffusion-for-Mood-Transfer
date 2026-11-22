# Text-Conditioned Music Mood Editing With a Latent Diffusion Transformer and ControlNet

Asher Wheatle

University of Florida

## Abstract

Changing a song’s mood while keeping its melody recognizable is difficult because the features that carry mood overlap with those that carry melody. We built a small text-guided mood editor based on the Diffusion Transformer with a ControlNet melody branch proposed by Hou et al. (2024). A 20.3-million-parameter model was trained from scratch on 5-s clips from the Database for Emotional Analysis in Music (DEAM), labeled happy or sad from annotated valence. Mood is supplied through embeddings from a contrastive language–audio model (CLAP): the model trains on each clip’s audio embedding and is prompted at inference with a text embedding, after an alignment step that narrows the gap between the two spaces. Edits begin from a denoising diffusion implicit model (DDIM) inversion of the input rather than from random noise, which preserved far more of the melody. On 20 evaluation songs drawn from the training corpus, the happy edit of a song scored happier than its sad edit for 19 songs by CLAP and for all 20 by a valence probe, and the two edits differed about as much as real happy and sad music. Edits kept 57.2% frame-level melody accuracy, against 63.9% for the autoencoder and vocoder alone. Turning sad songs happy was harder than the reverse: CLAP classified 3 of 10 such edits as happy, against 9 of 10 happy songs edited toward sad. Part of this gap comes from the codec itself, which already tilts CLAP’s reading of unedited clips toward sad.

*Keywords:* music mood transfer, diffusion transformer, ControlNet, CLAP, DDIM inversion

# Text-Conditioned Music Mood Editing With a Latent Diffusion Transformer and ControlNet

Diffusion models now produce convincing images and audio, and recent work uses them to edit existing content rather than generate it from scratch. One natural musical edit is to change the mood of a recording, such as making a sad piece sound happy, while keeping the melody so that a listener still recognizes the song. This is difficult because mood and melody are not separate layers of the audio.

Hou et al. (2024) addressed a related problem, text-guided music editing with melody control, by pairing a Diffusion Transformer (DiT; Peebles & Xie, 2023) with a ControlNet branch (Zhang et al., 2023) that receives a melody representation extracted from the input. A ControlNet adds a parallel branch that processes an extra control signal and feeds back through zero-initialized layers. Hou et al. used the most energetic constant-Q transform (CQT) pitch bins in each frame as the melody control. Their system is large and built on a pretrained audio generation model, Stable Audio Open. We asked whether the same design can be rebuilt at a much smaller scale, trained from scratch on one public emotion dataset, and focused on one edit: moving a clip toward happy or toward sad.

Two further ideas from prior work shape our approach. SDEdit (Meng et al., 2022) edits by adding random noise to the input and then denoising under a new condition, whereas DDIM inversion (Mokady et al., 2023; Song et al., 2021) runs the deterministic sampler backward from the input, producing a noisy latent that still encodes the song. In addition, mood can be specified through Contrastive Language–Audio Pretraining (CLAP; Wu et al., 2023), which trains audio and text encoders so that matching audio and captions have similar embeddings. The two encoders’ embeddings nevertheless occupy separate regions of the shared space, a phenomenon called the modality gap (Liang et al., 2022).

We make three contributions. First, we present a compact editor (20.3 million parameters) that operates on mel spectrograms through a small autoencoder and a pretrained BigVGAN vocoder (Lee et al., 2023). Second, we condition on mood with per-clip CLAP audio embeddings and a fixed alignment that lets text prompts stand in for them at inference, which avoids training on only two text vectors. Third, we evaluate the editor with measures that separate the effect of the prompt from artifacts of resynthesis.

## Method

Given a 5-s clip and a target mood (happy or sad), the system produces a clip that should sound like the target mood while keeping the original pitch content. The clip is converted to a log-mel spectrogram and encoded to a latent by a convolutional autoencoder, and a melody sequence is extracted with a CQT. The DiT denoises the latent with mood supplied through cross-attention and melody through the ControlNet branch, and BigVGAN converts the decoded spectrogram to audio. Figure 1 shows the pipeline at inference.

**Figure 1**

*Overview of the Mood-Editing Pipeline*

> [Figure: Block diagram of the mood-editing pipeline: encoding the input audio to a latent, latent editing with a Diffusion Transformer conditioned on a mood text prompt and a melody ControlNet branch, and decoding to output audio with BigVGAN.]

*Note.* DiT = Diffusion Transformer; CQT = constant-Q transform. In the final system, the partial-noising step is performed by DDIM inversion.

### Data and Labels

We used DEAM (Aljanaki et al., 2017), which provides 45-s excerpts with valence and arousal ratings sampled every 500 ms from 15 s onward. We rescaled ratings to [−1, 1] and used all 1,802 songs with both series. Each song’s annotated region (15–45 s) was split into six nonoverlapping 5-s windows, yielding 10,812 candidate clips. Each clip was labeled from the mean valence of its own window: happy above .10, sad below −.10, and discarded in between. We used valence only because it separates happy from sad, and in early tests CLAP distinguished arousal-based categories such as calm and energetic less well than other mood categories.

The dead band removed 2,985 clips (27.6%), leaving 7,827: 5,543 happy (70.8%) and 2,284 sad (29.2%). Every labeled clip then received one augmented copy made with a pitch shift of up to ±2 semitones, a time shift of up to ±20% (zero-filled), and white noise at a signal-to-noise ratio of 20–35 dB. The final training set contained 15,654 clips (11,086 happy and 4,568 sad).

Both moods were augmented equally on purpose. An earlier version augmented only sad clips to balance the classes, which made augmentation artifacts a cue for the label: they appeared in 59% of sad clips and no happy clips, and on real clips they alone moved CLAP 0.069 toward the sad caption. Because the model is conditioned on CLAP audio embeddings, this leaked into the conditioning. Class balance is now handled only by the sampler.

### Spectrograms, Autoencoder, and Melody Features

Spectrograms were computed with the analysis settings of BigVGAN (44.1 kHz, window and fast Fourier transform size 2048, hop 512, 128 mel bands), so a 5-s clip gave a 128 × 430 array. Log magnitudes were clipped to [−12, 2.5] and scaled to [0, 1] with this fixed range for every song.

The autoencoder has three downsampling blocks (4 × 4 convolution, stride 2, group normalization, SiLU activation) with channel widths of 64, 128, and 32 and a mirrored decoder ending in a sigmoid. A spectrogram maps to a 32 × 16 × 54 latent. This is only a 2× reduction in the number of values; an 8× version produced audibly distorted audio after vocoding. The model has 0.40 million parameters and was trained for 100 epochs with Adam (learning rate = 10⁻³, batch size = 64). Because mean-squared error produced blurry spectrograms, the loss combined L1 terms on the spectrogram, its time and frequency differences, and multiscale average-pooled versions.

After training, the autoencoder was frozen and all latents were standardized per channel. A single global scale failed because the channels’ standard deviations differ widely: by a factor of about 21 (0.031 to 0.643) in an earlier autoencoder and by a factor of 12 (0.040 to 0.483) in the final one. Under one scale, some channels were already pure noise at a diffusion step while others still matched the input, and the edit-strength setting behaved inconsistently.

Following Hou et al. (2024), we described melody by the strongest pitch bins in each frame. The waveform was high-pass filtered at 261.2 Hz (middle C) to reduce bass and drum energy, and a 128-bin CQT (one bin per Musical Instrument Digital Interface note, hop 512) was computed. For each of the 431 frames we kept the four strongest bin indices, embedded each with a learned 64-dimensional table, concatenated them, and used two strided 1-D convolutions and linear interpolation to reach the latent’s 54 time steps, giving a 54 × 256 melody sequence. This encoder (0.53 million parameters) was trained jointly with the DiT (Figure 2).

**Figure 2**

*Melody Extraction and Encoding*

> [Figure: Two-part diagram: (a) melody extraction by high-pass filtering, constant-Q transform, and top-K pitch selection; (b) melody encoding by pitch embedding, concatenation, strided convolutions, and interpolation to latent frames.]

*Note.* CQT = constant-Q transform. K = number of strongest bins kept per frame.

### Mood Conditioning With CLAP

We used the frozen LAION-CLAP music checkpoint with the HTSAT-base audio encoder; all embeddings are 512-dimensional unit vectors. The obvious approach is to condition each training clip on the CLAP text embedding of its label, but with two labels the network sees only two vectors and can treat them as an arbitrary switch. A model trained this way responded to the prompt, but its changes were unrelated to mood. We therefore conditioned each clip on its own CLAP audio embedding, giving 15,654 distinct vectors spread across the happy and sad regions of CLAP’s space. At inference, the text embedding of the target mood replaced the audio embedding.

#### Alignment

In our data, the mean audio embedding and the mean text embedding had a cosine similarity of only .27. Before training, we fit a fixed alignment. Each modality was centered on the average of its happy and sad means and renormalized. We then fit a rotation *R* that maps the centered text mood directions onto the centered audio mood directions (orthogonal Procrustes; Schönemann, 1966). Because there are only two moods, the data constrain only a couple of the 512 directions, so we regularized the fit toward the identity:

$$
C = \frac{1}{2}\sum_{m \in \{\text{happy},\,\text{sad}\}} c^{a}_{m}\,\big(c^{t}_{m}\big)^{\top} + \lambda \lVert C_0 \rVert_2\, I = U \Sigma V^{\top},
$$

$$
R = U V^{\top}, \qquad \lambda = 0.1,
$$

where $c^{a}_{m}$ and $c^{t}_{m}$ are the centered audio and text means for mood *m* and $C_0$ is the first term. Text vectors were centered, rotated by *R*, and renormalized; audio vectors were only centered and renormalized. As a check, we computed each caption’s margin: its cosine to the audio mean of its own mood minus its cosine to the other mood’s. On the training data, the margin rose from 0.10 to 1.25 for the happy caption and from 0.09 to 1.05 for the sad caption.

#### Prompts and Projection

The inference prompts were “a happy and uplifting piece of music” and “a sad and melancholic piece of music.” To train the text pathway directly, we also wrote 59 paraphrases per mood that describe valence rather than energy, mention the same instruments and genres for both moods, and were each verified to be closer in CLAP to their own mood’s caption than to the other’s. A linear layer maps the aligned vector to four 256-dimensional tokens, which serve as keys and values for cross-attention. For classifier-free guidance (Ho & Salimans, 2022), the null condition is the unaligned CLAP text embedding of the empty string.

### Diffusion Transformer and Training

Each of the 16 × 54 latent cells becomes a token, projected to 256 dimensions with a learned position embedding. The model has eight DiT blocks, each with self-attention, cross-attention to the mood tokens, and a multilayer perceptron, using adaptive layer normalization driven by the diffusion timestep (Peebles & Xie, 2023). The first four blocks have ControlNet partners that receive the current tokens plus the melody sequence and add their output to the main stream through a zero-initialized linear gate. Unlike the usual ControlNet setup, all blocks were trained together from random initialization. The model has 20.3 million trainable parameters.

We used a 1,000-step cosine noise schedule (Nichol & Dhariwal, 2021) and the v-prediction target (Salimans & Ho, 2022). Each training clip’s conditioning vector was its aligned CLAP audio embedding plus a small random perturbation (norm = 0.1), which helps the model tolerate the residual difference between aligned text and audio vectors. With probability .25 it was replaced by the aligned text embedding of a random paraphrase of the clip’s mood, and with probability .20 by the null condition; the melody input was zeroed with probability .15 so that the model also learned to work without melody. We trained with AdamW (learning rate = 10⁻⁴, batch size = 64) for 14,149 steps on one NVIDIA B200 graphics processing unit, which took about 44 min using the University of Florida’s HiPerGator.

### Editing Procedure

To edit a clip, we encoded it to a latent $z_0$, extracted its melody, and built two conditions: the aligned text embedding of the target caption and the null condition. An edit strength *s* sets the starting noise level $\tau_0 = sT$, where *T* = 1,000. We ran the deterministic DDIM sampler (Song et al., 2021) in reverse for 50 steps from $z_0$ to $\tau_0$, using the null condition and the melody, to obtain an inverted latent. We then ran the same 50 steps back down with classifier-free guidance applied to the mood condition only:

$$
\hat{v} = f_\theta(z_\tau, \tau, h_\varnothing, \mu) + w\,\big(f_\theta(z_\tau, \tau, h, \mu) - f_\theta(z_\tau, \tau, h_\varnothing, \mu)\big),
$$

where $z_\tau$ is the latent at timestep τ, *h* and $h_\varnothing$ are the mood and null tokens, μ is the melody, and *w* is the guidance scale. Both terms receive the melody, so guidance never amplifies it. With *w* = 0, denoising retraces the inversion and should return the input, which provides a direct check that inversion works. The final latent was decoded and vocoded.

Relative to Hou et al. (2024), we kept the DiT with a ControlNet melody branch, top-*k* CQT melody features, v-prediction, and guidance on text only. Hou et al. keep four bins per stereo channel; we use mono audio and four bins. We used a small purpose-trained autoencoder with BigVGAN (Lee et al., 2023) instead of the pretrained Stable Audio Open model, conditioned on CLAP audio embeddings with an alignment step, and trained everything from scratch. The editing procedure also differs. Hou et al. edit by generating from noise, conditioned on the input’s melody and the new text, so the input reaches the output only through its melody features. We start from the input’s own latent through DDIM inversion.

## Results

### Evaluation Design

The main evaluation used 20 songs at evenly spaced positions in the corpus (10 happy, 10 sad), each represented by its 15–20 s window and labeled with the training rule (songs in the dead band were excluded). Each song was edited toward both moods, yielding 40 edits. The diagnostic sweeps used a separate set of 30 songs (15 per mood). Each of those songs was edited only toward the other mood, so each direction rests on 15 songs. No songs were held out from training: the model was trained on all 1,802 songs, and the 15–20 s window is one of each song’s six training windows. The evaluation therefore measures editing of clips the model has seen, not generalization to new music.

#### Mood Judges

We classified a clip as whichever caption its CLAP audio embedding was closer to. On 100 unedited clips (67 happy, 33 sad), this agreed with the DEAM labels 83% of the time, compared with 67% for always answering happy. Accuracy was similar for both moods (56 of 67 happy, 27 of 33 sad). To measure how far valence moved, we also fit a ridge regression from CLAP audio embeddings to DEAM valence on 500 songs, choosing the regularization on a 20% held-out split. The probe reached held-out *R*² = .27 (*r* = .54), which is weak for single clips but sufficient to detect average shifts across many edits.

#### Metrics

Resynthesis alone lowers CLAP similarity to both captions by about 0.11, so comparing an edit with its original mixes the mood effect with this loss. Our main metrics therefore compare the two edits of the same song, which share the source, the resynthesis, and the random seed. A clip’s sadness score is its CLAP cosine to the sad caption minus its cosine to the happy caption. The CLAP margin of an edit toward mood *m* is its cosine to caption *m* minus the average cosine of the song’s two edits to that caption. The valence margin is defined the same way with the probe’s output, signed so that positive means toward the target. With two moods, a song’s happy and sad valence margins are equal by construction, since each is half the difference between the song’s two edits, so we report the valence margin only overall. Edit separation is the difference in sadness score between a song’s sad and happy edits, compared with the gap between real sad and real happy clips. With two moods, the overall CLAP margin equals one quarter of the edit separation, so these two metrics are the same evidence in different units. Transfer is the percentage of edits the CLAP classifier assigns to the target mood. We report it separately for edits that must change the song’s mood, and we compare it with the classification of the codec reconstruction with no edit. In the sweeps, push is the change in sadness score caused by the prompt relative to a *w* = 0 edit with the same seed, as a percentage of the real happy–sad gap.

#### Melody Preservation

Chroma similarity is the mean frame-wise cosine between 12-bin CQT chroma vectors of the original and the edit. It is generous, since two unrelated songs already score about .71 on average (.61 to .88 across the 30 diagnostic songs). In the sweeps, we rescaled it per song so that the autoencoder–vocoder reconstruction scores 100% and an unrelated song 0%, calling the result melody kept. The reconstruction is a reference point, not a strict ceiling. The melody branch reads pitch from the original audio and can restore some detail the codec loses, so an output can score above 100%. When the reconstruction and the unrelated song score close together, the per-song ratio is also noisy. For comparison with Hou et al. (2024), we also computed their melody accuracy, the proportion of frames in which the strongest pitch class matches between original and edit. As a rough check that edits stay musical, we report the harmonic ratio, the share of spectral energy that harmonic–percussive separation (FitzGerald, 2010) assigns to the harmonic part, relative to the reconstruction; garbled edits score below 1.

### Main Results

Table 1 summarizes the 40 edits at the default setting (*s* = .8, *w* = 7).

**Table 1**

*Main Results for 20 Songs and 40 Edits*

| Measure | Toward happy | Toward sad | Overall |
|---|---|---|---|
| CLAP margin | 0.047 | 0.021 | 0.034 ± 0.009 |
| Valence probe margin (target direction)ᵃ | | | 0.101 ± 0.019 |
| Songs whose happy edit scored happier than their sad edit (CLAP / valence probe)ᵃ | | | 19 of 20 / 20 of 20 |
| Valence shift toward target relative to the original clip | 80% | 80% | 80% |
| Transfer, all edits (CLAP classifies as target) | 50% | 95% | 72.5% |
| Codec reconstruction classified as target (no edit)ᵇ | 30% | 70% | 50% |
| Transfer, edits that must change the song’s mood | 30% (3 of 10) | 90% (9 of 10) | 60% (12 of 20) |
| Chroma similarity (reconstruction = .933) | .923 | .910 | .917 |
| Melody accuracy (reconstruction = 63.9%) | | | 57.2% |

*Note.* Ranges are ±2 standard errors over songs. CLAP = Contrastive Language–Audio Pretraining.
ᵃ Compares a song’s two edits with each other, so it has no separate value per direction.
ᵇ Share of the 20 codec reconstructions that CLAP assigns to each mood; the baseline against which transfer should be read.

The prompt clearly moved the audio relative to the song’s other edit. The happy edit of a song scored as less sad than its sad edit for 19 of 20 songs by CLAP and for all 20 by the valence probe, and both overall margins exceeded two standard errors. Measured against the original clip instead, which mixes in the codec and the diffusion prior, valence moved toward the target for 16 of 20 songs in each direction (18 of 20 relative to the codec reconstruction). The two edits of a song were 0.135 ± 0.036 apart in sadness score, compared with 0.115 ± 0.038 between 40 real happy and 40 real sad clips. The average separation between a song’s two edits was thus about 117% of a real mood difference, though the uncertainty is large (plausibly 65%–223%). The real-song gap was measured on raw audio, whereas the edits are resynthesized.

Crossing to the other mood was a harder test, and here the two directions differed. Happy songs edited toward sad were classified as sad 9 times out of 10, but sad songs edited toward happy were classified as happy only 3 times out of 10. The overall transfer split (50% toward happy vs. 95% toward sad) overstates this difference, because CLAP’s classification starts out biased toward sad. With no edit at all, CLAP classified the codec reconstruction as sad for 14 of 20 songs, including 4 of the 10 happy songs. Relative to that baseline, edits toward happy raised the happy share by 20 points (30% to 50%), and edits toward sad raised the sad share by 25 points (70% to 95%). The CLAP margin, which cancels the shared baseline, was larger toward happy (0.047) than toward sad (0.021).

Melody survived well relative to what the codec allows. Chroma similarity was .917, against .933 for the plain reconstruction. Melody accuracy was 57.2%, against 63.9% for the autoencoder and vocoder alone, so about 90% of the codec’s melody accuracy was preserved. Hou et al. (2024) reported 56.6% for their model and 42%–45% for MusicGen-stereo-melody baselines on the Song Describer dataset. Their dataset, clip length, and codec differ from ours, so the figures are not directly comparable.

### DDIM Inversion Versus SDEdit Noise

Table 2 compares the two ways of starting an edit. With SDEdit, the unguided round trip (*w* = 0) already lost 22%–51% of the melody, and the loss grew with strength; with inversion, the round trip kept 98%–101% at every strength, confirming that inversion is faithful. The 101% at *s* = 1.0 is not a real gain. It is a mean of per-song ratios driven by one song (142%) whose reconstruction and unrelated-song chroma differ by only .076, so a small chroma difference becomes a large ratio. The median across songs is 99.6%, the pooled ratio is 100.0%, and the round trip’s raw chroma differs from the reconstruction’s by −.0001. At *s* = .8, inversion kept 89% of the melody while pushing toward sad more strongly than SDEdit (92% vs. 77%) but toward happy less strongly (54% vs. 69%). Part of SDEdit’s apparent push likely comes from discarding the source, which makes room for the prompt but also produced the garbled audio we heard.

**Table 2**

*SDEdit Noise Versus DDIM Inversion as the Starting Point*

| Start | Strength | Melody kept, round trip | Happy push | Sad push | Melody kept, edit |
|---|---|---|---|---|---|
| SDEdit | .6 | 78% | 27% | 61% | 78% |
| SDEdit | .8 | 71% | 69% | 77% | 64% |
| SDEdit | 1.0 | 49% | 73% | 46% | 37% |
| Inversion | .6 | 99% | 15% | 50% | 95% |
| Inversion | .8 | 98% | 54% | 92% | 89% |
| Inversion | 1.0 | 101% | 41% | 104% | 66% |

*Note.* Values are from 30 songs with *w* = 7. Push is the change in sadness score caused by the prompt, as a percentage of the real happy–sad gap. Each push column rests on 15 songs (happy push: sad songs edited toward happy; sad push: the reverse). Melody kept is the mean of per-song ratios.

### Guidance and Strength Under Inversion

Because the happy push was weak, we tested stronger guidance and strength (Table 3). Stronger settings made the push more symmetric. At *s* = .9 and *w* = 7, the happy and sad pushes were nearly equal (96% and 99%), and 12 of 15 happy edits of sad songs were classified as happy, at a cost of five points of melody kept. The harmonic ratio stayed near 1 except at *s* = .9 and *w* = 10, so edits did not become noise-like in this range. We have not yet repeated the full evaluation at *s* = .9.

**Table 3**

*Guidance Scale and Edit Strength Under Inversion*

| *s* | *w* | Happy push | Sad push | Happy transfer | Harmonic ratio | Melody kept |
|---|---|---|---|---|---|---|
| .8 | 5 | 32% | 74% | 40% | 1.03 | 93% |
| .8 | 7 | 54% | 92% | 40% | 1.03 | 89% |
| .8 | 10 | 88% | 103% | 80% | 1.03 | 81% |
| .9 | 5 | 66% | 90% | 53% | 1.02 | 91% |
| .9 | 7 | 96% | 99% | 80% | 1.01 | 84% |
| .9 | 10 | 115% | 109% | 80% | 0.98 | 73% |

*Note.* Values are from 30 songs with DDIM inversion under the null prompt; each push column rests on 15 songs. Happy transfer is the percentage of the 15 sad songs whose happy edit CLAP classifies as happy.

### Sources of the Happy–Sad Asymmetry

We ran three checks to locate the weak happy push.

First, we checked the codec. On the 30 diagnostic songs, using an earlier autoencoder with the same architecture, the real happy–sad gap in sadness score was 0.124 on raw audio and 0.115 after the autoencoder and vocoder (93% of raw). Classification area under the curve fell only from .87 to .83, and the codec did not shift clips in either direction. The 0.115 matches the real-song gap in the main results by coincidence; that figure was measured on raw audio from a different set of 80 songs. The final autoencoder, retrained along with the final model, does shift clips. On the 20 evaluation songs, reconstruction moved the sadness score by 0.042 ± 0.022 toward sad, about a third of the real happy–sad gap. It also turned CLAP’s reading of two more happy songs from happy to sad. The valence probe does not show this shift (the reconstruction scored 0.04 happier), so the shift may be specific to CLAP’s caption comparison rather than to perceived mood. In either case, it gives the transfer metric a sad-leaning baseline. It does not affect the margins or the push, which compare edits that pass through the same codec.

Second, the melody branch does not explain the weak happy push. Because melody dropout was used in training, the melody input could be scaled to zero without leaving the training distribution. With an SDEdit start at *s* = .8 (15 songs per direction), removing the melody entirely changed the happy push only from 69% to 71% of the real gap (chroma similarity fell from .882 to .859), whereas the sad push rose from 77% to 97%. Melody conditioning thus costs some sad push but not happy push.

Third, we compared conditioning vectors on an earlier checkpoint: the aligned caption, the unaligned caption, and the average aligned audio embedding of 24 real clips of the target mood, on 10 songs. The aligned caption produced a CLAP margin of 0.020 ± 0.007, at least as large as the real-audio average (0.016 ± 0.006) and larger than the unaligned caption (0.014 ± 0.005). In all three conditions, however, the margin for happy targets was near zero while that for sad targets was clearly positive. That checkpoint was the one trained with sad-only augmentation, whose artifacts acted as a cue for sad in the conditioning. The leak alone could produce a sad-favoring asymmetry, so this check does not show that the final model’s asymmetry has the same cause.

## Discussion

The system did what it was designed to do in a limited sense. The text prompt reliably moved CLAP’s and the valence probe’s readings of the audio in the requested direction while keeping most of the melody the codec allows. The edits were not fully symmetric, however. At the default setting, moving a sad song toward happy worked less well than the reverse (a push of 54% vs. 92% of the real gap; 3 of 10 vs. 9 of 10 classified as the target), and only a stronger setting narrowed the gap, at some cost to melody. The asymmetry is smaller than the raw transfer rates suggest. Part of it is a sad-leaning bias that the final codec introduces into CLAP’s classification, and the margin that cancels this bias was larger toward happy. For the remainder, we suspect the training data: sad clips were less than a third of the labeled data, and many happy DEAM clips are only mildly positive. The checks above make the melody branch an unlikely cause. They are less conclusive about the codec and the text pathway, because the codec stage check used an earlier autoencoder and the text-pathway comparison used a checkpoint with a known sad-favoring leak.

Several limitations apply. First, the mood judge is not independent: CLAP both conditions and scores the model, and the valence probe is built on CLAP embeddings, so an edit could move CLAP’s embedding without changing how the music sounds to a person. Second, no songs were held out: the evaluation songs and windows were part of the training set, so the results describe editing of familiar clips. Third, the evaluation is small (20 songs in the main evaluation, 10 to 30 in diagnostics, with 15 per direction in the sweeps), and the wide interval on edit separation (65%–223%) shows how little 20 songs can pin down. Fourth, all clips were 5 s long and came from DEAM, whose valence ratings lean positive, so results may not generalize. Fifth, the codec sets a reference point rather than a strict bound. It costs about 0.11 in CLAP similarity and 36 points of melody accuracy before diffusion changes anything, and the final codec tilts CLAP’s reading toward sad. Finally, several diagnostics were run on other versions of the model and have not been repeated on the final one. The text-pathway comparison used the checkpoint with the augmentation leak, the codec stage check used an earlier autoencoder, and the melody sweep used SDEdit rather than inversion.

In conclusion, rebuilding the DiT-with-ControlNet editor of Hou et al. (2024) at small scale produced a system whose text prompt moved the audio toward the requested mood on nearly every evaluation song while preserving about 90% of the codec’s melody accuracy. Three choices mattered most: per-clip CLAP audio embeddings with an alignment to text, equal augmentation of both moods, and DDIM inversion. Turning sad songs happy remains harder than the reverse at the default setting, though part of the apparent gap comes from a sad-leaning bias of the codec under CLAP. Future work should include a listening study, a held-out evaluation set, a full evaluation at *s* = .9, and more balanced training data.

## References

Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLOS ONE, 12*(3), Article e0173392. https://doi.org/10.1371/journal.pone.0173392

FitzGerald, D. (2010). Harmonic/percussive separation using median filtering. In *Proceedings of the 13th International Conference on Digital Audio Effects (DAFx-10)* (pp. 246–253).

Ho, J., & Salimans, T. (2022). *Classifier-free diffusion guidance*. arXiv. https://doi.org/10.48550/arXiv.2207.12598

Hou, S., Liu, S., Yuan, R., Xue, W., Shan, Y., Zhao, M., & Zhang, C. (2024). *Editing music with melody and text: Using ControlNet for diffusion transformer*. arXiv. https://doi.org/10.48550/arXiv.2410.05151

Lee, S.-g., Ping, W., Ginsburg, B., Catanzaro, B., & Yoon, S. (2023). BigVGAN: A universal neural vocoder with large-scale training. In *Proceedings of the 11th International Conference on Learning Representations*. https://arxiv.org/abs/2206.04658

Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. Y. (2022). Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. *Advances in Neural Information Processing Systems, 35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/702f4db7543a7432431df588d57bc7c9-Abstract-Conference.html

Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). SDEdit: Guided image synthesis and editing with stochastic differential equations. In *Proceedings of the 10th International Conference on Learning Representations*. https://arxiv.org/abs/2108.01073

Mokady, R., Hertz, A., Aberman, K., Pritch, Y., & Cohen-Or, D. (2023). Null-text inversion for editing real images using guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 6038–6047). https://doi.org/10.1109/CVPR52729.2023.00585

Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 8162–8171). PMLR. https://proceedings.mlr.press/v139/nichol21a.html

Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 4195–4205). https://doi.org/10.1109/ICCV51070.2023.00387

Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. In *Proceedings of the 10th International Conference on Learning Representations*. https://arxiv.org/abs/2202.00512

Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika, 31*(1), 1–10. https://doi.org/10.1007/BF02289451

Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. In *Proceedings of the 9th International Conference on Learning Representations*. https://arxiv.org/abs/2010.02502

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In *Proceedings of the 2023 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 1–5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10095969

Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 3836–3847). https://doi.org/10.1109/ICCV51070.2023.00355
