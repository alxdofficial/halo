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


def knn_purity(Z: np.ndarray, y: np.ndarray, k: int = 10, macro: bool = True) -> float:
    """Fraction of each point's k nearest (cosine) neighbours that share its label.

    ``macro=True`` (default) averages per-label so common activities don't dominate (audit #7);
    ``macro=False`` is the window-micro average.
    """
    Z, y = _cap(Z, y)
    S = _l2(Z) @ _l2(Z).T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, kth=min(k, len(Z) - 1) - 1, axis=1)[:, :k]
    per_point = (y[nn] == y[:, None]).mean(axis=1)
    if macro:
        return float(np.mean([per_point[y == l].mean() for l in np.unique(y)]))
    return float(per_point.mean())


def cross_config_retrieval(Z: np.ndarray, y: np.ndarray, cfg: np.ndarray, k: int = 10,
                           macro: bool = True) -> float:
    """precision@k where each query retrieves only neighbours from a DIFFERENT config.

    Tests representation invariance to the config axis: can the same activity be retrieved across a
    rate/placement change? Only queries that HAVE a same-label instance in another config count.
    ``macro=True`` averages per-label (audit #7). NOTE: with synchronized-stream configs (e.g. xrf's
    6 simultaneous placements) this is *paired-placement* evidence, not independent cross-dataset
    generalization.
    """
    Z, y, cfg = _cap(Z, y, cfg)
    Zn = _l2(Z)
    S = Zn @ Zn.T
    precs, qlab = [], []
    for i in range(len(Z)):
        other = cfg != cfg[i]
        if not (other & (y == y[i])).any():          # no cross-config positive exists -> skip
            continue
        cand = np.where(other)[0]
        top = cand[np.argsort(-S[i, cand])[:k]]
        precs.append(float((y[top] == y[i]).mean()))
        qlab.append(y[i])
    if not precs:
        return float("nan")
    if macro:
        precs, qlab = np.array(precs), np.array(qlab)
        return float(np.mean([precs[qlab == l].mean() for l in np.unique(qlab)]))
    return float(np.mean(precs))


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
                               gravity_state=st["gravity"], channel_mask=st["channel_mask"],
                               sensor_text=st.get("sensor_text"))
        Zs.append(z.numpy())
        ys.append(labels)
        rates.append(np.full(len(labels), st["rate"]))
        places.append(np.full(len(labels), st["placement"], dtype=object))
        dsets.append(np.full(len(labels), st["dataset"], dtype=object))
    return {"Z": np.concatenate(Zs), "y": np.concatenate(ys), "rate": np.concatenate(rates),
            "placement": np.concatenate(places), "dataset": np.concatenate(dsets)}


def run_suite(enc, device, cap: int = 10_000, per_stream_cap: int = 2000,
              data_seed: Optional[int] = None) -> dict:
    """Full head-free + config-axis + downstream suite on the ablation subset (in-dist val + held-out).

    ``data_seed`` MUST match the checkpoint's training data_seed so the metric-val subjects are the
    SAME subject-disjoint split the model held out (audit 2026-07-23 #1: without this the harness used
    the default split regardless of the training seed, leaking ~19 train subjects into metric-val).
    """
    from training.tokenizer.ablation_subset import build_subset_index, build_heldout_index

    idx = build_subset_index(cap=cap, seed=data_seed)
    val = collect_embeddings(enc, idx, "val", device, per_stream_cap)
    ho = collect_embeddings(enc, build_heldout_index(cap=cap, seed=data_seed), "val", device, per_stream_cap)

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
    # downstream TRANSFER: probe TRAINED on subset-val, TESTED on the held-out config.
    ba, n_shared = transfer_probe_ba(val["Z"], val["y"], ho["Z"], ho["y"])
    out["transfer_probe_ba"] = round(ba, 4) if ba == ba else ba
    out["transfer_n_shared_labels"] = n_shared
    # in-distribution probe for contrast (this is NOT transfer)
    out["indist_probe_ba"] = round(linear_probe_ba(*_stratified_split(val["Z"], val["y"])), 4)
    return out


def transfer_probe_ba(train_Z, train_y, test_Z, test_y):
    """Probe TRAINED on train_*, TESTED on test_* — restricted to the labels shared by both.

    This is the genuine transfer number. The earlier ``run_suite`` mislabelled an in-distribution
    val-split probe as "transfer" (it never touched the held-out set); this makes train≠test explicit
    and is unit-tested. Returns (balanced_accuracy, n_shared_labels).
    """
    shared = sorted(set(np.asarray(train_y).tolist()) & set(np.asarray(test_y).tolist()))
    if len(shared) < 2:
        return float("nan"), len(shared)
    trm, tem = np.isin(train_y, shared), np.isin(test_y, shared)
    return linear_probe_ba(train_Z[trm], train_y[trm], test_Z[tem], test_y[tem]), len(shared)


def main() -> None:
    """CLI: run the suite on a checkpoint and write a JSON artifact (audit #8)."""
    import argparse
    import json
    from pathlib import Path

    import torch
    from training.tokenizer.eval_transfer import build_encoder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=10_000)
    ap.add_argument("--per-stream-cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    # Reconstruct the checkpoint's EXACT subject split so metric-val = the model's held-out subjects (#1).
    cfg = ckpt.get("config", {})
    data_seed = cfg.get("data_seed", cfg.get("seed"))   # fall back to seed for pre-data_seed checkpoints
    res = run_suite(enc, device, cap=args.cap, per_stream_cap=args.per_stream_cap, data_seed=data_seed)
    res["_meta"] = {"checkpoint": str(args.checkpoint), "seed": args.seed, "data_seed": data_seed,
                    "frontend": cfg.get("frontend"),
                    "train_datasets": sorted(cfg.get("train_datasets", []) or []),
                    "step": ckpt.get("step")}
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
