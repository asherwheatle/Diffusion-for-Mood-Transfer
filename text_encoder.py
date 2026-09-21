"""Text encoders for mood descriptions.

Two encoders live here:

  * ClapTextEncoder (Lever A, default): a *frozen* CLAP text tower followed by
    a small trainable projection. CLAP was contrastively trained on text<->audio
    pairs, so a prompt like "sad and melancholic" already lands near
    dark/mysterious-sounding audio — and unseen phrasings ("eerie", "ominous
    film score") generalize because CLAP knows they are semantically close.
    This is the encoder the pipeline now trains and infers with, and it uses the
    *same* CLAP representation the evaluation judges against.

  * TextEncoder (legacy): a character-level transformer trained from scratch.
    It only ever sees the fixed mood strings during training, so it learns
    a lookup from those exact character sequences to audio — no semantics, no
    generalization. Kept for reference / ablation only.
"""

import os

import numpy as np
import torch
import torch.nn as nn

CLAP_SR = 48000  # LAION-CLAP expects 48 kHz mono


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def _balanced_mean(x: torch.Tensor, labels) -> torch.Tensor:
    """Mean of per-label means, so a skewed label mix can't drag the centre
    toward the majority mood."""
    labels = list(labels)
    groups = sorted(set(labels))
    return torch.stack([
        x[torch.tensor([i for i, l in enumerate(labels) if l == g],
                       device=x.device)].mean(0)
        for g in groups]).mean(0)


def load_clap_text_model(clap_ckpt: str = None, device: str = "cpu"):
    """Load a LAION-CLAP module (music checkpoint) for text embedding.

    Mirrors evaluate.Clap's loading so training and evaluation share one CLAP
    representation. Returns the frozen CLAP_Module; the caller wraps it.
    """
    try:
        import laion_clap
    except ImportError as e:
        raise ImportError(
            "laion_clap is not installed. Run:  uv pip install laion-clap\n"
            "and download the music checkpoint:\n"
            "  wget https://huggingface.co/lukewys/laion_clap/resolve/main/"
            "music_audioset_epoch_15_esc_90.14.pt"
        ) from e
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base",
                                   device=device)
    if clap_ckpt and os.path.exists(clap_ckpt):
        print(f"[CLAP-TXT] Loading music checkpoint: {clap_ckpt}")
        model.load_ckpt(clap_ckpt)
    else:
        print("[CLAP-TXT] WARNING: no music checkpoint given, loading the "
              "default general-audio checkpoint (weaker at musical mood).")
        model.load_ckpt()
    model.eval()
    return model


class ClapTextEncoder(nn.Module):
    """Frozen CLAP text tower + a trainable projection into a DiT
    cross-attention sequence.

        prompt --(frozen CLAP text tower)--> (clap_dim,) L2-normed
               --(trainable Linear)--------> (n_tokens, d_model) sequence

    Only the projection (`proj` + `norm`) is trainable and saved in the state
    dict; CLAP is frozen and deliberately held OUTSIDE nn.Module registration
    (in a plain list) so it does not appear in `.parameters()` or
    `.state_dict()` and is not toggled by `.train()/.eval()`.

    Usage matches the old TextEncoder except CLAP does its own tokenization, so
    callers pass raw strings through `.encode(...)`:

        clap_emb = text_enc.encode(["sad and melancholic"])   # (B, clap_dim)
        text_emb = text_enc(clap_emb)                         # (B, n_tokens, d)

    `encode` (frozen, no grad) is cheap to precompute once per unique prompt.
    """

    def __init__(self, d_model: int, clap_model=None, clap_ckpt: str = None,
                 n_tokens: int = 4, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        clap = (clap_model if clap_model is not None
                else load_clap_text_model(clap_ckpt, device))
        clap.eval()
        for p in clap.parameters():
            p.requires_grad_(False)
        self._clap = [clap]            # hidden from parameters()/state_dict()
        self.device = device

        # Infer CLAP's text-embedding dim from a probe encode so the projection
        # shape is always correct (HTSAT-base music checkpoint -> 512).
        with torch.no_grad():
            self.clap_dim = int(self._encode_raw([""]).shape[-1])

        self.proj = nn.Linear(self.clap_dim, d_model * n_tokens)
        self.norm = nn.LayerNorm(d_model)

        # Modality-gap alignment, fitted by `fit_alignment` before training.
        # Registered as buffers so they ride along in text_enc.state_dict():
        # every loader (evaluate.py, edit mode, the probes) picks them up
        # through its existing load_state_dict call, and no inference path can
        # silently forget to apply the alignment the model was trained with.
        # Zero means + identity map is a no-op, i.e. the pre-alignment
        # behaviour — which is also what old checkpoints load as.
        self.register_buffer("audio_mean", torch.zeros(self.clap_dim))
        self.register_buffer("text_mean", torch.zeros(self.clap_dim))
        self.register_buffer("text_to_audio", torch.eye(self.clap_dim))

    _ALIGN_KEYS = ("audio_mean", "text_mean", "text_to_audio")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints from before alignment existed lack these buffers. Keep
        # the module's current values for them instead of failing a strict
        # load: a freshly built encoder holds identity (so the old checkpoint
        # behaves exactly as trained), and a trainer resuming an old
        # checkpoint keeps the alignment it just fitted.
        for name in self._ALIGN_KEYS:
            state_dict.setdefault(prefix + name, getattr(self, name))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def clap(self):
        return self._clap[0]

    # ---- modality-gap alignment ------------------------------------------
    # CLAP's audio and text embeddings occupy offset cones (measured on this
    # corpus: cos(mean_audio, mean_text) ~ 0.27). The DiT is trained on audio
    # embeddings but prompted with text ones, so without correction it meets
    # an out-of-distribution vector at inference. Two composable fixes:
    #
    #   center  subtract each modality's mean and renormalise, removing the
    #           constant translation between the cones (Liang et al. 2022,
    #           "Mind the Gap").
    #   map     additionally rotate centred text into centred audio space with
    #           an orthogonal Procrustes fit, so the text happy<->sad axis
    #           lines up with the audio one.
    #
    # The null (CFG) embedding is deliberately NOT aligned, in training or at
    # inference: it is a "no condition" marker, not a point in either cone,
    # and all that matters is that both sides use the identical vector.

    def align_audio(self, emb: torch.Tensor) -> torch.Tensor:
        """Raw CLAP audio embedding(s) -> the space the DiT is trained on."""
        return _l2(emb.to(self.audio_mean) - self.audio_mean)

    def align_text(self, emb: torch.Tensor) -> torch.Tensor:
        """Raw CLAP text embedding(s) -> the DiT's (audio-aligned) space."""
        return _l2((emb.to(self.text_mean) - self.text_mean)
                   @ self.text_to_audio.T)

    @torch.no_grad()
    def fit_alignment(self, audio_emb: torch.Tensor, audio_labels,
                      text_emb: torch.Tensor, text_labels,
                      mode: str = "map", reg: float = 0.1):
        """Fit the alignment buffers.

        Args:
            audio_emb: (N, D) raw CLAP audio embeddings of the training clips.
            audio_labels: N mood labels.
            text_emb: (P, D) raw CLAP text embeddings of the training prompts
                (all paraphrases of all moods).
            text_labels: P mood labels.
            mode: "none" | "center" | "map" (map implies center).
            reg: Procrustes shrinkage toward identity, as a fraction of the
                cross-covariance's spectral norm.

        Why the map is fitted on mood centroids: the text side has no per-clip
        captions, only per-mood prompts, so pairing every clip with a prompt of
        its mood gives a cross-covariance sum_i a_i t_m(i)^T that equals the
        outer products of the per-mood centroids exactly. With two balanced
        moods that is a rank-1 signal (the happy->sad axis on each side), and
        plain Procrustes would fill the other ~510 directions with an arbitrary
        rotation. Adding reg*I pins those directions to identity, so the map
        rotates the text mood axis onto the audio one and leaves everything
        else in place.
        """
        if mode not in ("none", "center", "map"):
            raise ValueError(f"unknown alignment mode {mode!r}")
        dev = self.audio_mean.device
        A = audio_emb.to(dev, torch.float64)
        T = text_emb.to(dev, torch.float64)
        eye = torch.eye(self.clap_dim, device=dev, dtype=torch.float64)

        self.audio_mean.zero_()
        self.text_mean.zero_()
        self.text_to_audio.copy_(eye)
        if mode == "none":
            return

        mu_a = _balanced_mean(A, audio_labels)
        mu_t = _balanced_mean(T, text_labels)
        self.audio_mean.copy_(mu_a)
        self.text_mean.copy_(mu_t)
        if mode == "center":
            return

        audio_labels, text_labels = list(audio_labels), list(text_labels)
        moods = sorted(set(audio_labels) & set(text_labels))
        if len(moods) < 2:
            raise ValueError("map alignment needs >= 2 moods present in both "
                             f"audio and text; got {moods}")
        Ac, Tc = _l2(A - mu_a), _l2(T - mu_t)

        def centroid(x, labels, m):
            idx = torch.tensor([i for i, l in enumerate(labels) if l == m],
                               device=dev)
            return x[idx].mean(0)

        C = sum(torch.outer(centroid(Ac, audio_labels, m),
                            centroid(Tc, text_labels, m))
                for m in moods) / len(moods)
        C = C + reg * torch.linalg.matrix_norm(C, ord=2) * eye
        U, _, Vh = torch.linalg.svd(C)
        self.text_to_audio.copy_(U @ Vh)

    @torch.no_grad()
    def alignment_report(self, audio_emb: torch.Tensor, audio_labels,
                         text_emb: torch.Tensor, text_labels) -> dict:
        """How well each mood's text lands on that mood's audio, raw vs aligned.

        For each text vector: margin = cos(x, own audio centroid) - mean cos
        to the other moods' centroids. Positive means the vector points toward
        its own mood. Returns {mood: (raw_margin, aligned_margin)}, averaged
        over that mood's text vectors.
        """
        audio_labels, text_labels = list(audio_labels), list(text_labels)
        moods = sorted(set(audio_labels) & set(text_labels))

        def margins(A, X):
            cent = {m: _l2(A[torch.tensor(
                [i for i, l in enumerate(audio_labels) if l == m],
                device=A.device)].mean(0)) for m in moods}
            out = {}
            for m in moods:
                xs = X[torch.tensor([i for i, l in enumerate(text_labels)
                                     if l == m], device=X.device)]
                own = xs @ cent[m]
                other = torch.stack([xs @ cent[o] for o in moods if o != m]
                                    ).mean(0)
                out[m] = float((own - other).mean())
            return out

        dev = self.audio_mean.device
        A, T = audio_emb.to(dev).float(), text_emb.to(dev).float()
        raw = margins(_l2(A), _l2(T))
        aligned = margins(self.align_audio(A), self.align_text(T))
        return {m: (raw[m], aligned[m]) for m in moods}

    @torch.no_grad()
    def _encode_raw(self, texts) -> torch.Tensor:
        emb = self.clap.get_text_embedding(list(texts), use_tensor=True)
        emb = emb.detach().float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)   # L2-norm
        return emb.to(self.device)

    @torch.no_grad()
    def encode_audio(self, wavs, sr: int, chunk: int = 64) -> torch.Tensor:
        """CLAP *audio* embeddings for a list of waveforms -> (B, clap_dim).

        Same space as `encode` (CLAP is contrastively aligned), so a model
        trained on these can be prompted with text embeddings at inference.
        Resampling matches evaluate.Clap.audio_embed exactly — training must
        not use a different audio front-end than the judge.
        """
        import librosa

        out = []
        for i in range(0, len(wavs), chunk):
            batch = [
                librosa.resample(np.asarray(w, dtype=np.float32),
                                 orig_sr=sr, target_sr=CLAP_SR)
                for w in wavs[i:i + chunk]
            ]
            n = min(len(b) for b in batch)
            x = np.stack([b[:n] for b in batch]).astype(np.float32)
            emb = self.clap.get_audio_embedding_from_data(x=x, use_tensor=False)
            emb = torch.from_numpy(np.asarray(emb)).float()
            out.append(emb / (emb.norm(dim=-1, keepdim=True) + 1e-8))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode(self, texts, chunk: int = 256) -> torch.Tensor:
        """Frozen CLAP text embeddings for a list of strings -> (B, clap_dim).

        Chunked so a large corpus (one string per training song) doesn't push
        the whole batch through CLAP's text tower at once.
        """
        texts = list(texts)
        if len(texts) <= chunk:
            return self._encode_raw(texts)
        return torch.cat([self._encode_raw(texts[i:i + chunk])
                          for i in range(0, len(texts), chunk)], dim=0)

    def forward(self, clap_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clap_emb: (B, clap_dim) frozen, L2-normed CLAP text embeddings
                      (from `.encode`).
        Returns:
            (B, n_tokens, d_model) conditioning sequence for cross-attention.
        """
        B = clap_emb.shape[0]
        h = self.proj(clap_emb).view(B, self.n_tokens, self.d_model)
        return self.norm(h)


class TextEncoder(nn.Module):
    """
    Legacy learned character-level text encoder for mood descriptions.

    Produces a sequence of text embeddings for cross-attention conditioning
    in the DiT. Self-contained — no pretrained model downloads needed, but it
    learns no word semantics (see module docstring). Superseded by
    ClapTextEncoder; kept for reference / ablation.
    """

    def __init__(self, d_model=256, max_len=128, n_layers=2, n_heads=4):
        super().__init__()
        self.max_len = max_len
        self.char_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def tokenize(text: str, max_len: int = 128) -> torch.LongTensor:
        tokens = [min(ord(c), 255) for c in text[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, seq_len) character indices 0-255
        Returns:
            (B, seq_len, d_model) text embeddings for cross-attention
        """
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device)
        x = self.char_embed(tokens) + self.pos_embed(pos)
        x = self.encoder(x)
        return self.norm(x)
