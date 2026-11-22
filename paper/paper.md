# Text-Conditioned Music Mood Editing With a Latent Diffusion Transformer and ControlNet

Asher Wheatle

University of Florida

## Abstract

Changing a song’s mood while keeping its melody recognizable is difficult because the features that carry mood overlap with those that carry melody. We built a small text-guided mood editor based on the Diffusion Transformer with a ControlNet melody branch of Hou et al. (2024). A 20.3-million-parameter model was trained from scratch on 5-s clips from the Database for Emotional Analysis in Music (DEAM), labeled happy or sad from annotated valence. Mood is supplied through embeddings from a contrastive language–audio model (CLAP): the model trains on each clip’s audio embedding and is prompted at inference with a text embedding, after an alignment step that narrows the gap between the two spaces. Edits begin from a denoising diffusion implicit model (DDIM) inversion of the input, which preserved far more melody than random noise. On 20 evaluation songs from the training corpus, the happy edit of a song scored happier than its sad edit for 19 songs by CLAP and for all 20 by a valence probe. Edits kept 57.2% frame-level melody accuracy, against 63.9% for the autoencoder and vocoder alone. Turning sad songs happy was harder than the reverse (3 of 10 vs. 9 of 10 edits classified as the target), partly because the codec already tilts CLAP’s reading of unedited clips toward sad.

*Keywords:* music mood transfer, diffusion transformer, ControlNet, CLAP, DDIM inversion

# Text-Conditioned Music Mood Editing With a Latent Diffusion Transformer and ControlNet

Diffusion models are increasingly used to edit existing content rather than generate it from scratch. One natural musical edit is to make a sad piece sound happy, or the reverse, while keeping the melody recognizable. This is difficult because mood and melody are not separate layers of the audio.

Hou et al. (2024) addressed text-guided music editing with melody control by pairing a Diffusion Transformer (DiT; Peebles & Xie, 2023) with a ControlNet branch (Zhang et al., 2023), a parallel branch that processes a control signal and feeds back through zero-initialized layers. Their melody control is the most energetic constant-Q transform (CQT) pitch bins in each frame, and their system is built on the large pretrained Stable Audio Open model. We asked whether the same design can be rebuilt at a much smaller scale, trained from scratch on one public emotion dataset, for one edit: moving a clip toward happy or toward sad.

Two further ideas shape our approach. SDEdit (Meng et al., 2022) edits by adding random noise to the input and denoising under a new condition, whereas DDIM inversion (Mokady et al., 2023; Song et al., 2021) runs the deterministic sampler backward from the input, producing a noisy latent that still encodes the song. Mood can be specified through Contrastive Language–Audio Pretraining (CLAP; Wu et al., 2023), whose audio and text embeddings nevertheless occupy separate regions of the shared space, a phenomenon called the modality gap (Liang et al., 2022).

We make three contributions: a compact editor (20.3 million parameters) operating on mel spectrograms through a small autoencoder and the pretrained BigVGAN vocoder (Lee et al., 2023); mood conditioning on per-clip CLAP audio embeddings with a fixed alignment that lets text prompts replace them at inference; and an evaluation that separates the prompt’s effect from resynthesis artifacts.

## Method

A clip is converted to a log-mel spectrogram and encoded to a latent by a convolutional autoencoder, and a melody sequence is extracted with a CQT. The DiT denoises the latent with mood supplied through cross-attention and melody through the ControlNet branch, and BigVGAN converts the decoded spectrogram to audio (Figure 1).

**Figure 1**

*Overview of the Mood-Editing Pipeline*

> [Figure: Block diagram of the mood-editing pipeline: encoding the input audio to a latent, latent editing with a Diffusion Transformer conditioned on a mood text prompt and a melody ControlNet branch, and decoding to output audio with BigVGAN.]

*Note.* DiT = Diffusion Transformer; CQT = constant-Q transform. In the final system, the partial-noising step is performed by DDIM inversion.

### Data and Labels

DEAM (Aljanaki et al., 2017) provides 45-s excerpts with valence ratings every 500 ms from 15 s onward. We rescaled ratings to [−1, 1] and split each of the 1,802 songs’ annotated region (15–45 s) into six 5-s windows, yielding 10,812 clips. Each clip was labeled from its own mean valence: happy above .10, sad below −.10, discarded in between. We used valence only; in early tests CLAP separated arousal-based categories such as calm and energetic poorly. The dead band removed 27.6% of clips, leaving 7,827: 5,543 happy (70.8%) and 2,284 sad (29.2%).

Every labeled clip received one augmented copy (pitch shift up to ±2 semitones, time shift up to ±20%, white noise at 20–35 dB signal-to-noise ratio), giving 15,654 training clips. Both moods were augmented equally on purpose. An earlier version augmented only sad clips, so artifacts appeared in 59% of sad clips and no happy ones; on real clips they alone moved CLAP 0.069 toward the sad caption, and this leaked into the CLAP conditioning. Class balance is now handled only by the sampler.

### Spectrograms, Autoencoder, and Melody Features

Spectrograms use BigVGAN’s analysis settings (44.1 kHz, window 2048, hop 512, 128 mel bands), giving a 128 × 430 array per clip, with log magnitudes clipped to [−12, 2.5] and scaled to [0, 1]. The autoencoder has three downsampling blocks (4 × 4 convolution, stride 2, group normalization, SiLU) with widths 64, 128, and 32 and a mirrored decoder ending in a sigmoid, mapping a spectrogram to a 32 × 16 × 54 latent. This is only a 2× reduction; an 8× version sounded distorted after vocoding. It has 0.40 million parameters and was trained for 100 epochs with L1 losses on the spectrogram, its time and frequency differences, and pooled versions (mean-squared error gave blurry output).

Latents were then standardized per channel. A single global scale failed because channel standard deviations differ by a factor of 12 in the final autoencoder (about 21 in an earlier one), so some channels became pure noise at a diffusion step while others still matched the input.

Following Hou et al. (2024), the waveform was high-pass filtered at middle C (261.2 Hz) and a 128-bin CQT (one bin per MIDI note, hop 512) was computed. For each of 431 frames we kept the four strongest bin indices, embedded each in 64 dimensions, concatenated them, and used two strided 1-D convolutions and interpolation to reach a 54 × 256 melody sequence. This encoder (0.53 million parameters) was trained jointly with the DiT (Figure 2).

**Figure 2**

*Melody Extraction and Encoding*

> [Figure: Two-part diagram: (a) melody extraction by high-pass filtering, constant-Q transform, and top-K pitch selection; (b) melody encoding by pitch embedding, concatenation, strided convolutions, and interpolation to latent frames.]

*Note.* CQT = constant-Q transform. K = number of strongest bins kept per frame.

### Mood Conditioning With CLAP

We used the frozen LAION-CLAP music checkpoint (HTSAT-base); embeddings are 512-dimensional unit vectors. Conditioning each clip on the text embedding of its label gives the network only two vectors, which it can treat as an arbitrary switch; a model trained this way responded to the prompt with changes unrelated to mood. We therefore conditioned each clip on its own CLAP audio embedding (15,654 distinct vectors) and substituted the target mood’s text embedding at inference.

#### Alignment

In our data, the mean audio and mean text embeddings had a cosine similarity of only .27. Before training, each modality was centered on the average of its happy and sad means and renormalized, and we fit a rotation *R* mapping the centered text mood directions onto the audio ones (orthogonal Procrustes; Schönemann, 1966). Because two moods constrain only a few of 512 directions, we regularized toward the identity:

$$
C = \frac{1}{2}\sum_{m \in \{\text{happy},\,\text{sad}\}} c^{a}_{m}\,\big(c^{t}_{m}\big)^{\top} + \lambda \lVert C_0 \rVert_2\, I = U \Sigma V^{\top}, \qquad R = U V^{\top}, \qquad \lambda = 0.1,
$$

where $c^{a}_{m}$ and $c^{t}_{m}$ are the centered audio and text means for mood *m* and $C_0$ is the first term. A caption’s margin (cosine to its own mood’s audio mean minus cosine to the other’s) rose from 0.10 to 1.25 for the happy caption and from 0.09 to 1.05 for the sad caption.

#### Prompts and Projection

The inference prompts were “a happy and uplifting piece of music” and “a sad and melancholic piece of music.” To train the text pathway directly, we wrote 59 paraphrases per mood that describe valence rather than energy, share instruments and genres across moods, and are each closer in CLAP to their own caption. A linear layer maps the aligned vector to four 256-dimensional cross-attention tokens. For classifier-free guidance (Ho & Salimans, 2022), the null condition is the unaligned CLAP embedding of the empty string.

### Diffusion Transformer and Training

Each of the 16 × 54 latent cells becomes a 256-dimensional token with a learned position embedding. The model has eight DiT blocks (self-attention, cross-attention to the mood tokens, and a multilayer perceptron, with timestep-driven adaptive layer normalization). The first four have ControlNet partners that receive the tokens plus the melody sequence and add their output through zero-initialized gates. Unlike the usual ControlNet setup, all blocks were trained together from scratch.

We used a 1,000-step cosine schedule (Nichol & Dhariwal, 2021) and the v-prediction target (Salimans & Ho, 2022). Each clip’s conditioning vector was its aligned audio embedding plus a random perturbation of norm 0.1, to tolerate the residual text–audio difference. With probability .25 it was replaced by the aligned embedding of a random paraphrase of the clip’s mood and with probability .20 by the null condition; the melody was zeroed with probability .15. We trained with AdamW (learning rate 10⁻⁴, batch size 64) for 14,149 steps, about 44 min on one NVIDIA B200 GPU on the University of Florida’s HiPerGator.

### Editing Procedure

An edit encodes the clip to a latent $z_0$ and extracts its melody. An edit strength *s* sets the starting noise level $\tau_0 = sT$ (*T* = 1,000). We ran the DDIM sampler (Song et al., 2021) in reverse for 50 steps from $z_0$ to $\tau_0$ under the null condition and the melody, then ran the same steps back down with classifier-free guidance on the mood condition only:

$$
\hat{v} = f_\theta(z_\tau, \tau, h_\varnothing, \mu) + w\,\big(f_\theta(z_\tau, \tau, h, \mu) - f_\theta(z_\tau, \tau, h_\varnothing, \mu)\big),
$$

where *h* and $h_\varnothing$ are the mood and null tokens, μ is the melody, and *w* is the guidance scale. Both terms receive the melody, so guidance never amplifies it. With *w* = 0, denoising retraces the inversion and should return the input, a direct check that inversion works.

Relative to Hou et al. (2024), we kept the DiT with a ControlNet melody branch, top-*k* CQT features (they keep four bins per stereo channel; we use four on mono audio), v-prediction, and text-only guidance. We replaced Stable Audio Open with a small autoencoder and BigVGAN, conditioned through aligned CLAP audio embeddings, and trained from scratch. Editing also differs: Hou et al. generate from noise conditioned on the input’s melody and new text, whereas we start from the input’s own latent through DDIM inversion.

## Results

### Evaluation Design

The main evaluation used 20 songs evenly spaced through the corpus (10 happy, 10 sad), each represented by its 15–20 s window and labeled with the training rule, and edited toward both moods (40 edits). Diagnostic sweeps used 30 other songs (15 per mood), each edited only toward the other mood. No songs were held out: this window is one of each song’s training windows, so the evaluation measures editing of familiar clips, not generalization.

#### Mood Judges

CLAP classified a clip by whichever caption its audio embedding was closer to. On 100 unedited clips (67 happy, 33 sad), it agreed with DEAM labels 83% of the time (vs. 67% for always answering happy), with similar accuracy for both moods. We also fit a ridge regression from CLAP audio embeddings to DEAM valence on 500 songs; it reached held-out *R*² = .27 (*r* = .54), weak per clip but sufficient for average shifts.

#### Metrics

Resynthesis alone lowers CLAP similarity to both captions by about 0.11, so our main metrics compare a song’s two edits, which share source, codec, and seed. A clip’s sadness score is its CLAP cosine to the sad caption minus that to the happy caption. An edit’s CLAP margin toward mood *m* is its cosine to caption *m* minus the mean of the song’s two edits; the valence margin is analogous, and with two moods it is identical for both directions. Edit separation is the sadness-score difference between a song’s sad and happy edits (four times the CLAP margin), compared with the gap between real sad and happy clips. Transfer is the percentage of edits CLAP assigns to the target. In the sweeps, push is the prompt’s change in sadness score relative to a *w* = 0 edit with the same seed, as a percentage of the real gap.

#### Melody Preservation

Chroma similarity is the mean frame-wise cosine between chroma vectors of the original and the edit; unrelated songs already score about .71. In the sweeps we rescaled it per song so the codec reconstruction scores 100% and an unrelated song 0% (“melody kept”); the reconstruction is a reference, not a strict ceiling. We also report the melody accuracy of Hou et al. (2024), the share of frames whose strongest pitch class matches, and a harmonic ratio from harmonic–percussive separation (FitzGerald, 2010) relative to the reconstruction, which falls below 1 for garbled edits.

### Main Results

Table 1 summarizes the 40 edits at the default setting (*s* = .8, *w* = 7).

**Table 1**

*Main Results for 20 Songs and 40 Edits*

| Measure | Toward happy | Toward sad | Overall |
|---|---|---|---|
| CLAP margin | 0.047 | 0.021 | **0.034 ± 0.009** |
| Valence probe margin (target direction)ᵃ | | | **0.101 ± 0.019** |
| Songs whose happy edit scored happier than their sad edit (CLAP / valence probe)ᵃ | | | **19 of 20 / 20 of 20** |
| Valence shift toward target relative to the original clip | 80% | 80% | 80% |
| Transfer, all edits (CLAP classifies as target) | 50% | 95% | 72.5% |
| Codec reconstruction classified as target (no edit)ᵇ | 30% | 70% | 50% |
| Transfer, edits that must change the song’s mood | 30% (3 of 10) | 90% (9 of 10) | 60% (12 of 20) |
| Chroma similarity (reconstruction = .933) | .923 | .910 | **.917** |
| Melody accuracy (reconstruction = 63.9%) | | | **57.2%** |

*Note.* Ranges are ±2 standard errors over songs. CLAP = Contrastive Language–Audio Pretraining. Bold marks the headline overall results.
ᵃ Compares a song’s two edits with each other, so it has no separate value per direction.
ᵇ Share of the 20 codec reconstructions that CLAP assigns to each mood; the baseline against which transfer should be read.

The prompt clearly moved the audio relative to the song’s other edit: both overall margins exceeded two standard errors, and the happy edit scored less sad than the sad edit for 19 of 20 songs by CLAP and all 20 by the probe. Against the original clip, which mixes in the codec and diffusion prior, valence moved toward the target for 16 of 20 songs in each direction. A song’s two edits were 0.135 ± 0.036 apart in sadness score, against 0.115 ± 0.038 between 40 real happy and 40 real sad clips (raw audio), or about 117% of a real mood difference (plausibly 65%–223%).

Crossing moods was harder, and the directions differed: 9 of 10 happy songs edited toward sad were classified sad, but only 3 of 10 sad songs edited toward happy were classified happy. The overall transfer split (50% vs. 95%) overstates this, because CLAP already classified 14 of 20 unedited reconstructions as sad, including 4 happy songs. Relative to that baseline, edits raised the happy share by 20 points and the sad share by 25, and the CLAP margin, which cancels the baseline, was larger toward happy.

Melody survived well: chroma similarity was .917 against .933 for the reconstruction, and melody accuracy was 57.2% against 63.9%, about 90% of what the codec allows. Hou et al. (2024) reported 56.6% for their model and 42%–45% for MusicGen-stereo-melody on the Song Describer dataset, but their data, clip length, and codec differ, so the figures are not directly comparable.

### DDIM Inversion Versus SDEdit Noise

With SDEdit, the unguided round trip (*w* = 0) already lost 22%–51% of the melody; with inversion it kept 98%–100% at every strength, confirming that inversion is faithful (Table 2). At *s* = 1.0 the per-song mean is 101%, driven by one song whose reconstruction and unrelated-song chroma differ by only .076; Table 2 reports the pooled ratio (100%), and the median is 99.6%. At *s* = .8, inversion kept 89% of the melody while pushing toward sad more strongly than SDEdit (92% vs. 77%) but toward happy less strongly (54% vs. 69%). Part of SDEdit’s push likely comes from discarding the source, which also produced the garbled audio we heard.

**Table 2**

*SDEdit Noise Versus DDIM Inversion as the Starting Point*

| Start | Strength | Melody kept, round trip | Happy push | Sad push | Melody kept, edit |
|---|---|---|---|---|---|
| SDEdit | .6 | 78% | **27%** | **61%** | 78% |
| SDEdit | .8 | 71% | **69%** | 77% | 64% |
| SDEdit | 1.0 | 49% | **73%** | 46% | 37% |
| Inversion | .6 | **99%** | 15% | 50% | **95%** |
| Inversion | .8 | **98%** | 54% | **92%** | **89%** |
| Inversion | 1.0 | **100%** | 41% | **104%** | **66%** |

*Note.* Values are from 30 songs with *w* = 7; each push column rests on 15 songs. Push is the change in sadness score caused by the prompt, as a percentage of the real happy–sad gap. Melody kept is the mean of per-song ratios, except the inversion round trip at strength 1.0, which is the pooled ratio (the per-song mean is 101%; see text). Bold marks the better value in each column within each strength.

### Guidance and Strength Under Inversion

Because the happy push was weak, we tested stronger settings (Table 3), which made the push more symmetric. At *s* = .9 and *w* = 7, the happy and sad pushes were nearly equal (96% and 99%) and 12 of 15 happy edits of sad songs were classified as happy, at a cost of five points of melody kept. The harmonic ratio fell below 1 only at *s* = .9, *w* = 10. We have not yet repeated the full evaluation at *s* = .9.

**Table 3**

*Guidance Scale and Edit Strength Under Inversion*

| *s* | *w* | Happy push | Sad push | Happy transfer | Harmonic ratio | Melody kept |
|---|---|---|---|---|---|---|
| .8 | 5 | 32% | 74% | 40% | 1.03 | **93%** |
| .8 | 7 | 54% | 92% | 40% | 1.03 | 89% |
| .8 | 10 | 88% | 103% | **80%** | 1.03 | 81% |
| .9 | 5 | 66% | 90% | 53% | 1.02 | 91% |
| .9 | 7 | 96% | 99% | **80%** | **1.01** | 84% |
| .9 | 10 | **115%** | **109%** | **80%** | 0.98 | 73% |

*Note.* Values are from 30 songs with DDIM inversion under the null prompt; each push column rests on 15 songs. Happy transfer is the percentage of the 15 sad songs whose happy edit CLAP classifies as happy. Bold marks the best value in each column (for harmonic ratio, the value closest to 1).

### Sources of the Happy–Sad Asymmetry

We ran three checks. First, the codec: with an earlier autoencoder, the real happy–sad gap was 0.124 on raw audio and 0.115 after reconstruction (93%), with no directional shift. The final autoencoder, however, moved the sadness score of the 20 evaluation songs by 0.042 ± 0.022 toward sad, about a third of the real gap. The valence probe did not show this shift, so it may be specific to CLAP’s caption comparison; either way it gives transfer a sad-leaning baseline but cancels out of the margins and push.

Second, the melody branch. Because of melody dropout, the melody input could be removed without leaving the training distribution. With an SDEdit start at *s* = .8, removing it changed the happy push only from 69% to 71% (chroma similarity .882 to .859), whereas the sad push rose from 77% to 97%. Melody conditioning costs some sad push but does not explain the weak happy push.

Third, the text pathway. On an earlier checkpoint, the aligned caption produced a CLAP margin of 0.020 ± 0.007 on 10 songs, at least as large as the mean aligned audio embedding of 24 real clips (0.016 ± 0.006) and larger than the unaligned caption (0.014 ± 0.005). In all three conditions the happy margin was near zero. That checkpoint, however, was the one with the sad-only augmentation leak, which alone could favor sad, so this check does not identify the final model’s asymmetry.

## Discussion

The text prompt reliably moved CLAP’s and the valence probe’s readings in the requested direction while keeping most of the melody the codec allows. Moving sad songs toward happy worked less well than the reverse at the default setting, and only a stronger setting narrowed the gap, at some cost to melody. Part of the asymmetry is a sad-leaning bias of the final codec under CLAP. For the remainder we suspect the training data: sad clips were under a third of the labeled data, and many happy DEAM clips are only mildly positive. Our checks make the melody branch an unlikely cause but are less conclusive about the codec and text pathway, because they used an earlier autoencoder and a leaky checkpoint.

Several limitations apply. CLAP both conditions and scores the model, and the valence probe is built on CLAP, so an edit could move CLAP’s embedding without changing how the music sounds. No songs were held out. The evaluation is small (20 songs, 15 per direction in the sweeps), as the wide 65%–223% interval shows. All clips were 5 s long and from DEAM, whose ratings lean positive. The codec costs about 0.11 in CLAP similarity and 36 points of melody accuracy before diffusion changes anything. Finally, the text-pathway, codec, and melody diagnostics were run on earlier versions of the model or with SDEdit and have not been repeated on the final one.

In conclusion, a small-scale rebuild of the editor of Hou et al. (2024) moved audio toward the requested mood on nearly every evaluation song while preserving about 90% of the codec’s melody accuracy. Three choices mattered most: per-clip CLAP audio embeddings aligned to text, equal augmentation of both moods, and DDIM inversion. Future work should include a listening study, a held-out evaluation set, a full evaluation at *s* = .9, and more balanced training data.

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
