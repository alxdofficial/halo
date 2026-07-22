"""Tokenizer-quality metric suite (docs/design/TOKENIZER_ABLATION.md §"How to compare").

Head-free and config-axis metrics for comparing tokenizer arms, chosen so the comparison does NOT
rest only on a downstream predictor head — and, crucially, does NOT use the A1/filterbank-similarity
objective (confound #7: A1's target is the fixed filterbank, so it is home-field for Arm A).

Pure functions operate on already-computed embeddings ``Z (N,d)`` + labels + config tags, so they
are unit-testable without an encoder. ``collect_embeddings`` / ``run_suite`` wire a frozen encoder
to the ablation subset.

Metrics
  * knn_purity              — head-free: fraction of k-NN sharing the label (retrieval quality).
  * cross_config_retrieval  — head-free: retrieve same activity ACROSS a rate/placement change.
  * alignment / uniformity  — Wang-Isola SSL representation-quality (head-free).
  * effective_rank          — spectral entropy; catches dimensional collapse (head-free).
  * decodability            — linear probe BA predicting a CONFIG (rate/placement): how much the
                              tokenizer leaked it. Interpret with cross-config transfer, not alone.
  * linear_probe_ba / knn_ba — the downstream number (one signal among many).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- pure metric functions
def _l2(Z: np.ndarray) -> np.ndarray:
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def _cap(Z, *arrays, n=4000, seed=0):
    """Subsample to n rows (the O(N^2) similarity metrics stay cheap)."""
    if len(Z) <= n:
        return (Z, *arrays)
    idx = np.random.RandomState(seed).choice(len(Z), n, replace=False)
    return (Z[idx], *[a[idx] for a in arrays])


def knn_purity(Z: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each point's k nearest (cosine) neighbours that share its label."""
    Z, y = _cap(Z, y)
    S = _l2(Z) @ _l2(Z).T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, kth=min(k, len(Z) - 1) - 1, axis=1)[:, :k]
    return float((y[nn] == y[:, None]).mean())


def cross_config_retrieval(Z: np.ndarray, y: np.ndarray, cfg: np.ndarray, k: int = 10) -> float:
    """precision@k where each query retrieves only neighbours from a DIFFERENT config.

    Tests representation invariance to the config axis: can the same activity be retrieved across a
    rate/placement change? Only queries that HAVE a same-label instance in another config count.
    """
    Z, y, cfg = _cap(Z, y, cfg)
    Zn = _l2(Z)
    S = Zn @ Zn.T
    precs = []
    for i in range(len(Z)):
        other = cfg != cfg[i]
        if not (other & (y == y[i])).any():          # no cross-config positive exists -> skip
            continue
        cand = np.where(other)[0]
        top = cand[np.argsort(-S[i, cand])[:k]]
        precs.append(float((y[top] == y[i]).mean()))
    return float(np.mean(precs)) if precs else float("nan")


def alignment(Z: np.ndarray, y: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    """Wang-Isola alignment: mean ||z_i - z_j||^2 over same-label pairs (normalised). Lower=tighter."""
    Zn = _l2(Z)
    rng = np.random.RandomState(seed)
    labs, inv = np.unique(y, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(labs))]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return float("nan")
    d = []
    for _ in range(n_pairs):
        g = groups[rng.randint(len(groups))]
        i, j = rng.choice(g, 2, replace=False)
        d.append(((Zn[i] - Zn[j]) ** 2).sum())
    return float(np.mean(d))


def uniformity(Z: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    """Wang-Isola uniformity: log E[exp(-2 ||z_i - z_j||^2)] over random pairs. Lower=more uniform."""
    Zn = _l2(Z)
    rng = np.random.RandomState(seed)
    i = rng.randint(0, len(Zn), n_pairs)
    j = rng.randint(0, len(Zn), n_pairs)
    keep = i != j
    d = ((Zn[i[keep]] - Zn[j[keep]]) ** 2).sum(1)
    return float(np.log(np.mean(np.exp(-2.0 * d)) + 1e-12))


def effective_rank(Z: np.ndarray) -> float:
    """exp(spectral entropy) of the centred singular-value distribution. ~1 = collapsed, ~d = full."""
    s = np.linalg.svd(Z - Z.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def decodability(Z: np.ndarray, target: np.ndarray, seed: int = 0) -> float:
    """Balanced accuracy of a linear probe predicting a CONFIG label (rate/placement) from Z.

    High = the tokenizer baked the config in (a potential shortcut). Report ALONGSIDE cross-config
    transfer — some leakage (e.g. the filterbank's Nyquist mask exposing rate) may be benign.
    """
    return linear_probe_ba(*_stratified_split(Z, np.asarray(target), seed))


def linear_probe_ba(Ztr, ytr, Zte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr.tolist())) < 2:
        return float("nan")
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Ztr), ytr)
    return float(balanced_accuracy_score(yte, clf.predict(sc.transform(Zte))))


def _stratified_split(Z, y, seed=0, test_frac=0.3):
    """Returns (Ztr, ytr, Zte, yte) — the order linear_probe_ba expects (NOT sklearn's default)."""
    from sklearn.model_selection import train_test_split
    strat = y if min(np.bincount(np.unique(y, return_inverse=True)[1])) >= 2 else None
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=test_frac, random_state=seed, stratify=strat)
    return Ztr, ytr, Zte, yte


# --------------------------------------------------------------------- encoder -> embeddings harness
def collect_embeddings(enc, index, split: str, device, per_stream_cap: Optional[int] = None):
    """Embed a CorpusIndex split through the frozen encoder, tagged with config axes.

    Returns dict of np.arrays: Z (N,d), y (labels), rate, placement, dataset.
    """
    import torch
    from training.tokenizer.eval_transfer import encode_dataset
    from training.tokenizer.ablation_subset import iter_split_streams

    Zs, ys, rates, places, dsets = [], [], [], [], []
    for st in iter_split_streams(index, split):
        data, labels = st["data"], st["labels"]
        if per_stream_cap and len(data) > per_stream_cap:
            sel = np.random.RandomState(0).choice(len(data), per_stream_cap, replace=False)
            data, labels = data[sel], labels[sel]
        with torch.no_grad():
            z = encode_dataset(enc, data, st["texts"], device, st["rate"],
                               gravity_state=st["gravity"], channel_mask=st["channel_mask"])
        Zs.append(z.numpy())
        ys.append(labels)
        rates.append(np.full(len(labels), st["rate"]))
        places.append(np.full(len(labels), st["placement"], dtype=object))
        dsets.append(np.full(len(labels), st["dataset"], dtype=object))
    return {"Z": np.concatenate(Zs), "y": np.concatenate(ys), "rate": np.concatenate(rates),
            "placement": np.concatenate(places), "dataset": np.concatenate(dsets)}


def run_suite(enc, device, cap: int = 10_000, per_stream_cap: int = 2000) -> dict:
    """Full head-free + config-axis + downstream suite on the ablation subset (in-dist val + held-out)."""
    from training.tokenizer.ablation_subset import build_subset_index, build_heldout_index

    idx = build_subset_index(cap=cap)
    val = collect_embeddings(enc, idx, "val", device, per_stream_cap)
    ho = collect_embeddings(enc, build_heldout_index(cap=cap), "val", device, per_stream_cap)

    out = {
        "n_frontend_params": int(sum(p.numel() for p in enc.filterbank.parameters())),
        "n_encoder_params": int(sum(p.numel() for p in enc.parameters())),
        "val": {
            "knn_purity": round(knn_purity(val["Z"], val["y"]), 4),
            "cross_rate_retrieval": round(cross_config_retrieval(val["Z"], val["y"], val["rate"]), 4),
            "cross_placement_retrieval": round(
                cross_config_retrieval(val["Z"], val["y"], val["placement"]), 4),
            "alignment": round(alignment(val["Z"], val["y"]), 4),
            "uniformity": round(uniformity(val["Z"]), 4),
            "effective_rank": round(effective_rank(val["Z"]), 2),
            "rate_decodability": round(decodability(val["Z"], val["rate"]), 4),
            "placement_decodability": round(decodability(val["Z"], val["placement"]), 4),
        },
        "heldout_config": {
            "knn_purity": round(knn_purity(ho["Z"], ho["y"]), 4),
            "cross_placement_retrieval": round(
                cross_config_retrieval(ho["Z"], ho["y"], ho["placement"]), 4),
        },
    }
    # downstream: linear probe trained on val, tested on held-out config (transfer)
    out["transfer_probe_ba"] = round(
        linear_probe_ba(*_stratified_split(val["Z"], val["y"])), 4)
    return out
