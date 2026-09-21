"""A valence probe: a ridge-regression head on frozen CLAP audio embeddings.

Motivation
----------
The main evaluation (evaluate.py) judges a mood edit two ways: CLAP's happy/sad
argmax (transfer_success) and chroma preservation. The mood argmax is
coarse — it only tells you which of the two buckets an edit landed in. This probe
adds a *continuous valence axis* so we can ask a sharper question: did the edit
move the audio's positivity (valence) in the intended direction, and by how
much?

Valence is the harder affective axis to read from audio (mode/harmony carry it,
and DEAM's valence labels are noisier than its arousal labels), so a per-song
prediction is only moderately accurate. But as an aggregate metric over many
edits, even a noisy-but-unbiased probe reliably detects a *shift in the mean*,
which is exactly the "did mood transfer, on the whole" question we care about.

Design
------
  frozen CLAP audio embedding (D-dim, L2-normed)  ->  ridge head  ->  valence
                                                      (fit on DEAM valence)

  * Regression, not classification: DEAM valence is continuous in [-1, 1];
    we keep it continuous and only threshold/aggregate afterward.
  * Ridge (L2) is essential because the embedding dim (~512) is comparable to
    or larger than the number of training songs (p >= n), so an unregularized
    fit would overfit badly. lambda is chosen by held-out R^2.
  * The probe MUST be reported with its held-out R^2 / correlation. If that is
    near zero, the valence-shift metric it produces is noise and should be
    disregarded — mirroring how evaluate.py validates CLAP before trusting it.

Caveat: the probe reuses CLAP embeddings, so it shares CLAP's blind spots with
evaluate.py's existing CLAP metric — it is a sharper readout on the same
representation, not a fully independent judge.
"""

import os

import numpy as np


# Desired valence direction for each target mood. The labels are now derived
# from valence alone (annotations.mood_from_va), so these signs are exactly
# the two sides of that split rather than an approximation of a quadrant.
MOOD_VALENCE_SIGN = {
    "happy and uplifting":  +1,
    "sad and melancholic":  -1,
}


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an unregularized bias. X: (n, d) -> w: (d+1,).

    Solves min_w ||[X 1]w - y||^2 + lam * ||w[:-1]||^2. The bias column is
    left out of the penalty so we don't shrink the mean prediction.
    """
    n, d = X.shape
    Xa = np.concatenate([X, np.ones((n, 1))], axis=1)      # (n, d+1)
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0                                       # don't penalize bias
    w = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)
    return w


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)


class ValenceProbe:
    """A fitted linear valence head. Consumes L2-normalized CLAP embeddings."""

    def __init__(self, w: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray,
                 metrics: dict = None):
        self.w = w                    # (d+1,)
        self.feat_mean = feat_mean    # (d,)
        self.feat_std = feat_std      # (d,)
        self.metrics = metrics or {}  # held-out R^2, pearson, n, lambda

    def predict(self, emb: np.ndarray) -> np.ndarray:
        """Predict valence for one (d,) or a batch (n, d) of CLAP embeddings.

        Returns a float for a single embedding, else a (n,) array. Values are
        raw (not clipped to [-1, 1]) so differences stay meaningful.
        """
        single = (emb.ndim == 1)
        X = np.atleast_2d(emb).astype(np.float64)
        X = (X - self.feat_mean) / self.feat_std
        Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        pred = Xa @ self.w
        return float(pred[0]) if single else pred

    def save(self, path: str):
        np.savez(path, w=self.w, feat_mean=self.feat_mean,
                 feat_std=self.feat_std,
                 metrics=np.array(list(self.metrics.items()), dtype=object))

    @classmethod
    def load(cls, path: str) -> "ValenceProbe":
        d = np.load(path, allow_pickle=True)
        metrics = ({k: float(v) for k, v in d["metrics"]}
                   if "metrics" in d else {})
        return cls(d["w"], d["feat_mean"], d["feat_std"], metrics)


def fit_valence_probe(embeddings: np.ndarray, valence: np.ndarray,
                      val_frac: float = 0.2, seed: int = 0,
                      lambdas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
                      ) -> ValenceProbe:
    """Fit a ridge valence probe with a held-out lambda sweep.

    Args:
        embeddings: (n, d) L2-normalized CLAP audio embeddings.
        valence:    (n,) ground-truth valence in [-1, 1].
        val_frac:   fraction held out to pick lambda and report honest metrics.

    Returns a ValenceProbe whose .metrics holds the held-out R^2 / pearson
    (the numbers that say whether the probe is trustworthy). The returned
    probe is refit on ALL songs with the chosen lambda, since more data makes
    a better metric — but the reported R^2 comes from the held-out split.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(valence, dtype=np.float64)
    n = len(y)

    # Standardize features (helps ridge condition the normal equations).
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0) + 1e-8
    Xn = (X - feat_mean) / feat_std

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    best = {"lam": lambdas[0], "r2": -np.inf, "pearson": 0.0}
    for lam in lambdas:
        w = _ridge_fit(Xn[tr_idx], y[tr_idx], lam)
        Xa_val = np.concatenate([Xn[val_idx], np.ones((n_val, 1))], axis=1)
        pred = Xa_val @ w
        r2 = _r2(y[val_idx], pred)
        if r2 > best["r2"]:
            best = {"lam": lam, "r2": r2, "pearson": _pearson(y[val_idx], pred)}

    # Refit on everything with the chosen lambda for the deployed probe.
    w_full = _ridge_fit(Xn, y, best["lam"])
    metrics = {"heldout_r2": best["r2"], "heldout_pearson": best["pearson"],
               "lambda": best["lam"], "n_train": float(n - n_val),
               "n_heldout": float(n_val), "n_total": float(n)}
    return ValenceProbe(w_full, feat_mean, feat_std, metrics)


def train_probe_from_clip_files(clap, files, va, sr, start_s, dur_s,
                                load_clip, song_id_from_filename,
                                cache_path: str = None,
                                val_frac: float = 0.2, seed: int = 0
                                ) -> ValenceProbe:
    """Embed each annotated clip with CLAP and fit the valence probe.

    `clap`, `load_clip`, `song_id_from_filename` are passed in from evaluate.py
    so this module stays free of the heavy CLAP/audio imports. Caches the fitted
    probe (small) to `cache_path` and reuses it on later runs.
    """
    if cache_path and os.path.exists(cache_path):
        probe = ValenceProbe.load(cache_path)
        m = probe.metrics
        print(f"[VPROBE] Loaded cached probe: {cache_path}")
        print(f"[VPROBE] held-out R^2={m.get('heldout_r2', float('nan')):.3f} "
              f"pearson={m.get('heldout_pearson', float('nan')):.3f} "
              f"(n={int(m.get('n_total', 0))}, lambda={m.get('lambda')})")
        return probe

    print(f"\n{'='*60}\n VALENCE PROBE: embedding {len(files)} annotated clips"
          f"\n{'='*60}")
    embs, vals = [], []
    for path in files:
        sid = song_id_from_filename(path)
        wav = load_clip(path, sr, start_s, dur_s)
        embs.append(clap.audio_embed(wav, sr))
        vals.append(va[sid][0])            # (valence, arousal) -> valence
    embs = np.stack(embs, axis=0)
    vals = np.asarray(vals, dtype=np.float64)

    probe = fit_valence_probe(embs, vals, val_frac=val_frac, seed=seed)
    m = probe.metrics
    print(f"[VPROBE] held-out R^2={m['heldout_r2']:.3f} "
          f"pearson={m['heldout_pearson']:.3f}  "
          f"(train={int(m['n_train'])}, held-out={int(m['n_heldout'])}, "
          f"lambda={m['lambda']:g})")
    verdict = ("LEGITIMATE — valence is learnable here" if m["heldout_r2"] >= 0.15
               else "WEAK — near zero; treat valence_shift as unreliable")
    print(f"[VPROBE] Verdict: {verdict}")
    if cache_path:
        probe.save(cache_path)
        print(f"[VPROBE] Cached probe to {cache_path}")
    return probe
