"""Model-agnostic scoring core for the ZS-XD evaluation protocol (v2).

Ported from the legacy ``eval_v2.py`` scoring functions, minus the legacy data
IO (that lives in :mod:`eval.data`, wired to the new grid format). Everything
here operates on plain arrays / label strings, so HALO and every baseline are
scored by the SAME code path.

Protocol summary (see docs):
  * **ZS-XD**: zero-shot vs the TARGET dataset's own pre-registered label
    strings (exact string match, no synonym groups anywhere).
  * **Primary metric: macro-F1** over classes present in ground truth UNION
    predictions (not accuracy) — the imbalance-aware ZSL canon.
  * **Subject-disjoint** splits for anything that trains on target data; a
    subject never appears in two splits.
  * **Subject-stratified bootstrap CIs**: windows within a subject are
    correlated, so we resample subjects, not windows.
  * **ConSE bridge** (Norouzi et al., 2014) for closed-vocabulary baselines:
    a probability-weighted convex combination of frozen-SBERT training-label
    embeddings, scored against the target vocabulary.

Fixes preserved from the legacy audit (do not regress):
  * macro-F1 is primary, not accuracy.
  * Splits are subject-DISJOINT.
  * No target-label leakage at inference: the ConSE bridge combines *training*
    labels and only projects onto the target vocabulary; the bootstrap FREEZES
    the scored class set on the full sample so every replicate scores one
    estimand.
  * Ground truth is offset-free: the new grid format stores per-window label
    strings directly (:mod:`eval.data`), so there is no code arithmetic here at
    all — only a drop of windows whose label is outside the candidate vocab.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Pre-registered protocol constants (do not tune post-hoc).
CONSE_TOP_T = 10
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 3431
SOFT_POOL_TAU = 0.07  # matches the training temperature


# =============================================================================
# Ground truth (offset-free, new grid format)
# =============================================================================

def filter_ground_truth(
    gt_names: Sequence[str],
    subjects: Sequence,
    candidates: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Restrict per-window ground truth to the dataset's candidate label vocab.

    The new grid format already stores a per-window label string and subject id
    (:mod:`eval.data`), so there is no majority vote over raw codes and NO offset
    arithmetic — the class of legacy bug (HARTH min-subtraction) is structurally
    impossible here. The only ground-truth decision left is which windows are
    in-vocabulary: a window whose label is not one of the target dataset's
    pre-registered candidate strings can never be predicted correctly (candidates
    are exactly `candidates`) and is dropped, mirroring the legacy `keep_idx`.

    Args:
        gt_names:   per-window ground-truth label strings, length N.
        subjects:   per-window subject ids, length N (any hashable; str or int).
        candidates: the target dataset's candidate label vocabulary.

    Returns:
        kept_gt:  labels of the windows kept (subset of `candidates`).
        kept_subjects: (N',) subject ids of the kept windows.
        keep_idx: (N',) indices into the original N windows that were kept, so
                  any per-window model output aligns via ``output[keep_idx]``.
    """
    gt_names = list(gt_names)
    subjects = np.asarray(subjects)
    vocab = set(candidates)
    keep_idx = np.array([i for i, g in enumerate(gt_names) if g in vocab], dtype=np.int64)
    kept_gt = [gt_names[i] for i in keep_idx]
    return kept_gt, subjects[keep_idx], keep_idx


# =============================================================================
# Subject-disjoint splitting
# =============================================================================

def subject_disjoint_split(
    subjects: np.ndarray,
    fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split window indices into train/val/test with DISJOINT subject sets.

    Subjects (not windows) are shuffled and partitioned by the given fractions.
    Every split receives at least one subject; requires >= 3 unique subjects.
    The val allocation is floored so the remainder falls to TEST — this hands
    test >= 2 subjects whenever the cohort is large enough (e.g. n=15 -> 12/1/2),
    giving a non-degenerate bootstrap CI. Small cohorts can still yield 1 test
    subject; that is flagged downstream by :func:`subject_bootstrap_ci`.

    Returns three index arrays into `subjects`.
    """
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    if len(uniq) < 3:
        raise ValueError(
            f"subject_disjoint_split needs >=3 unique subjects, got {len(uniq)}"
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)

    n = len(perm)
    n_train = max(1, int(round(n * fracs[0])))
    n_val = max(1, int(n * fracs[1]))
    if n_train + n_val >= n:  # guarantee test gets >= 1 subject
        n_train = max(1, n - 2)
        n_val = 1
    train_subj = set(perm[:n_train].tolist())
    val_subj = set(perm[n_train:n_train + n_val].tolist())
    test_subj = set(perm[n_train + n_val:].tolist())

    assert train_subj.isdisjoint(val_subj) and train_subj.isdisjoint(test_subj) \
        and val_subj.isdisjoint(test_subj), "subject splits must be disjoint"
    assert train_subj and val_subj and test_subj, "every split needs >=1 subject"

    in_train = np.isin(subjects, list(train_subj))
    in_val = np.isin(subjects, list(val_subj))
    in_test = np.isin(subjects, list(test_subj))
    return np.nonzero(in_train)[0], np.nonzero(in_val)[0], np.nonzero(in_test)[0]


def balanced_subsample_indices(
    indices: np.ndarray,
    gt_names: Sequence[str],
    rate: float,
    seed: int = BOOTSTRAP_SEED,
    return_counts: bool = False,
):
    """As-balanced-as-possible subsample of `indices` down to ~`rate` of its size.

    Water-filling: classes are filled scarce-first with an equal share of the
    remaining budget, and any deficit from a class that runs out of windows is
    redistributed to classes with spare capacity. This keeps the total at
    ~rate*N (a plain per-class cap silently under-samples and would make e.g. a
    few-shot 10% split both too small and imbalanced), while staying as balanced
    as the data allows. Achieved per-class counts are returned when
    `return_counts=True` so the caller can record them.
    """
    rng = np.random.RandomState(seed)
    indices = np.asarray(indices)
    names = np.asarray(gt_names)[indices]
    classes, _ = np.unique(names, return_counts=True)
    avail = {c: int((names == c).sum()) for c in classes.tolist()}
    n_total = max(len(classes), int(len(indices) * rate))

    order = sorted(classes.tolist(), key=lambda c: avail[c])  # scarce class first
    quota = {c: 0 for c in classes.tolist()}
    remaining = n_total
    k = len(order)
    for i, c in enumerate(order):
        share = remaining // (k - i)
        take = min(avail[c], share)
        quota[c] = take
        remaining -= take

    picked: List[int] = []
    achieved: Dict[str, int] = {}
    for c in classes.tolist():
        cls_idx = rng.permutation(indices[names == c])
        take = quota[c]
        picked.extend(cls_idx[:take].tolist())
        achieved[c] = int(take)
    picked = rng.permutation(np.array(picked, dtype=np.int64))
    if return_counts:
        return picked, achieved
    return picked


# =============================================================================
# Scoring: similarities -> predictions
# =============================================================================

def _as_2d_label_sims(sims: np.ndarray) -> np.ndarray:
    """Collapse multi-prototype similarity (N, L, K) -> (N, L) by max over K."""
    if sims.ndim == 3:
        return sims.max(axis=-1)
    return sims


def predict_from_similarity(
    sims: np.ndarray,
    candidates: Sequence[str],
) -> List[str]:
    """argmax over candidate labels. `sims`: (N, L) or (N, L, K)."""
    sims = _as_2d_label_sims(np.asarray(sims))
    idx = sims.argmax(axis=1)
    return [candidates[i] for i in idx]


def soft_pool_patch_scores(
    patch_sims: np.ndarray,
    patch_mask: np.ndarray,
    tau: float = SOFT_POOL_TAU,
) -> np.ndarray:
    """Soft logit-pooling of per-patch label scores into one segment score.

    ``score[k] = sum_t softmax(patch_sims[t] / tau)[k]`` over valid patches.
    Statistically stronger than hard per-patch majority voting (no vote
    fragmentation, no first-index tie bias).

    Args:
        patch_sims: (P, L) per-patch cosine similarities.
        patch_mask: (P,) bool, True = valid patch.
        tau: softmax temperature (pre-registered = training temperature).

    Returns:
        (L,) pooled segment scores.
    """
    sims = np.asarray(patch_sims, dtype=np.float64)[np.asarray(patch_mask, dtype=bool)]
    if sims.size == 0:
        raise ValueError("no valid patches to pool")
    z = sims / tau
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    return p.sum(axis=0)


def segment_predictions(
    patch_sims: np.ndarray,
    patch_masks: np.ndarray,
    candidates: Sequence[str],
    mode: str = "soft",
    tau: float = SOFT_POOL_TAU,
) -> List[str]:
    """Segment-level predictions from per-patch similarities.

    Args:
        patch_sims: (N, P, L) per-patch similarities (padded).
        patch_masks: (N, P) bool valid-patch masks.
        candidates: L label strings.
        mode: 'soft' (pre-registered default) or 'vote' (legacy diagnostic).
    """
    preds = []
    for i in range(patch_sims.shape[0]):
        if mode == "soft":
            scores = soft_pool_patch_scores(patch_sims[i], patch_masks[i], tau)
        elif mode == "vote":
            valid = np.asarray(patch_masks[i], dtype=bool)
            votes = np.asarray(patch_sims[i])[valid].argmax(axis=1)
            scores = np.bincount(votes, minlength=len(candidates))
        else:
            raise ValueError(f"unknown pooling mode: {mode}")
        preds.append(candidates[int(np.argmax(scores))])
    return preds


# =============================================================================
# Metrics
# =============================================================================

def macro_f1_classes(gt_names: Sequence[str], pred_names: Sequence[str]) -> List[str]:
    """Class set macro-F1 is averaged over: ground-truth classes UNION predicted
    classes. Union (sklearn's default) charges false positives that a model
    routes into candidate classes with zero test windows — GT-only would let
    those FPs escape; the full candidate vocab would inject automatic F1=0 for
    never-relevant classes and over-penalize."""
    return sorted(set(gt_names) | set(pred_names))


def classification_metrics(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    f1_classes: Optional[Sequence[str]] = None,
    recall_classes: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """v2 metric set.

    - macro-F1 (primary): averaged over `f1_classes` = GT ∪ predicted classes.
    - balanced accuracy = macro recall over `recall_classes` = GT classes only
      (recall is undefined for a class with no true samples).
    `f1_classes` / `recall_classes` may be pinned by the caller (the bootstrap
    freezes them on the full sample so every replicate scores the SAME estimand).
    """
    gt_names = list(gt_names)
    pred_names = list(pred_names)
    if f1_classes is None:
        f1_classes = macro_f1_classes(gt_names, pred_names)
    if recall_classes is None:
        recall_classes = sorted(set(gt_names))
    return {
        "f1_macro": f1_score(gt_names, pred_names, labels=list(f1_classes),
                             average="macro", zero_division=0) * 100,
        "balanced_accuracy": recall_score(gt_names, pred_names, labels=list(recall_classes),
                                          average="macro", zero_division=0) * 100,
        "accuracy": accuracy_score(gt_names, pred_names) * 100,
        "f1_weighted": f1_score(gt_names, pred_names, labels=list(f1_classes),
                                average="weighted", zero_division=0) * 100,
        "n_samples": len(gt_names),
        "n_gt_classes": len(set(gt_names)),
        "n_scored_classes": len(f1_classes),
    }


def per_class_f1(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
) -> Dict[str, float]:
    """Per-class F1 (percent) over GT ∪ predicted classes."""
    classes = macro_f1_classes(gt_names, pred_names)
    scores = f1_score(gt_names, pred_names, labels=classes,
                      average=None, zero_division=0)
    return {c: float(s) * 100 for c, s in zip(classes, scores)}


def subject_bootstrap_ci(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    subjects: np.ndarray,
    metric: str = "f1_macro",
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Subject-stratified bootstrap CI: resample SUBJECTS with replacement.

    Windows within a subject are correlated — resampling windows would
    understate variance, so we resample whole subjects.

    Two correctness guards from the audit:
      * The scored class set is FROZEN once on the full sample and reused for
        every replicate. Re-deriving it per replicate makes replicates that drop
        a subject-exclusive class average macro-F1 over fewer classes — a
        different estimand — so the interval need not bracket the point estimate.
      * With < 2 subjects a subject-bootstrap has no variance to resample; we
        return a NaN interval flagged `ci_degenerate` rather than a fake
        zero-width 95% CI.
    """
    gt = np.asarray(gt_names)
    pred = np.asarray(pred_names)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)

    if len(uniq) < 2:
        return {
            f"{metric}_ci_lo": float("nan"),
            f"{metric}_ci_hi": float("nan"),
            "bootstrap_B": 0,
            "n_subjects": int(len(uniq)),
            "ci_degenerate": True,
        }

    f1_classes = macro_f1_classes(gt.tolist(), pred.tolist())
    recall_classes = sorted(set(gt.tolist()))

    def score(g, p) -> float:
        if metric == "f1_macro":
            return f1_score(g, p, labels=f1_classes, average="macro", zero_division=0) * 100
        if metric == "balanced_accuracy":
            return recall_score(g, p, labels=recall_classes, average="macro", zero_division=0) * 100
        if metric == "accuracy":
            return accuracy_score(g, p) * 100
        raise ValueError(f"unsupported bootstrap metric: {metric}")

    subj_windows = {s: np.nonzero(subjects == s)[0] for s in uniq}
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(B):
        sample_subj = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([subj_windows[s] for s in sample_subj])
        stats.append(score(gt[idx].tolist(), pred[idx].tolist()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        f"{metric}_ci_lo": float(lo),
        f"{metric}_ci_hi": float(hi),
        "bootstrap_B": B,
        "n_subjects": int(len(uniq)),
        "ci_degenerate": False,
    }


# =============================================================================
# SBERT encoder + ConSE bridge (Norouzi et al., 2014)
# =============================================================================

_SBERT_CACHE: dict = {}


def get_sbert_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[Sequence[str]], np.ndarray]:
    """Frozen SBERT mean-pool encoder used by the ConSE bridge (the SAME encoder
    for every bridged model). Labels are de-underscored before encoding and the
    returned embeddings are L2-normalized."""
    if model_name not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    sbert = _SBERT_CACHE[model_name]

    def encode(labels: Sequence[str]) -> np.ndarray:
        texts = [l.replace("_", " ") for l in labels]
        return np.asarray(sbert.encode(texts, normalize_embeddings=True))

    return encode


def conse_embeddings(
    probs: np.ndarray,
    train_vocab_embs: np.ndarray,
    top_T: int = CONSE_TOP_T,
) -> np.ndarray:
    """ConSE semantic embedding: probability-weighted convex combination of the
    top-T training-label embeddings.

    Args:
        probs: (N, K) classifier softmax over its OWN training vocabulary.
        train_vocab_embs: (K, D) L2-normalized embeddings of the training labels.
        top_T: number of top classes to combine (pre-registered = 10).

    Returns:
        (N, D) L2-normalized semantic embeddings.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != train_vocab_embs.shape[0]:
        raise ValueError(
            f"probs {probs.shape} incompatible with vocab embeddings "
            f"{train_vocab_embs.shape}"
        )
    T = min(top_T, probs.shape[1])
    top_idx = np.argsort(-probs, axis=1)[:, :T]                      # (N, T)
    top_p = np.take_along_axis(probs, top_idx, axis=1)               # (N, T)
    denom = top_p.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    w = top_p / denom
    v = np.einsum("nt,ntd->nd", w, train_vocab_embs[top_idx])        # (N, D)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return v / norms


def conse_predict(
    probs: np.ndarray,
    train_vocab: Sequence[str],
    target_labels: Sequence[str],
    encode: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    top_T: int = CONSE_TOP_T,
) -> Tuple[List[str], Dict[str, object]]:
    """Full ConSE bridge: classifier softmax over the train vocab -> predictions
    over the target dataset's label strings, plus reachability stats.

    No target-label leakage: the bridge combines only *training* label
    embeddings; the target vocabulary enters solely as the projection space.

    Returns:
        pred_names: N predictions among `target_labels`.
        info: reachability stats. `reachable_nn_lb` is a T=1 nearest-neighbour
              LOWER BOUND on which target classes the bridge can output; the
              actual top-T convex combinations can also land on other classes,
              so `predicted_classes` (the classes actually hit) is reported too.
    """
    if encode is None:
        encode = get_sbert_encoder()
    train_embs = encode(train_vocab)
    target_embs = encode(target_labels)

    v = conse_embeddings(probs, train_embs, top_T=top_T)             # (N, D)
    sims = v @ target_embs.T                                         # (N, L)
    preds = [target_labels[i] for i in sims.argmax(axis=1)]

    nn_of_train = (train_embs @ target_embs.T).argmax(axis=1)        # (K,)
    reachable_lb = sorted({target_labels[i] for i in nn_of_train})
    predicted = sorted(set(preds))
    info = {
        "reachable_nn_lb": reachable_lb,
        "reachability_lb": len(reachable_lb) / len(target_labels),
        "predicted_classes": predicted,
        "n_predicted_classes": len(predicted),
        "top_T": top_T,
    }
    return preds, info
