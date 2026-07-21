"""Baseline adapter framework for the ZS-XD evaluation protocol (v2).

Each baseline is a small adapter that subclasses one of two tiers and is
registered with :func:`register`:

  * :class:`ConSEAdapter` — a closed-vocabulary classifier. It emits a per-window
    softmax over the GLOBAL training vocabulary
    (``data/labels/global_labels.json``); the base bridges that to the target
    dataset's label strings with ConSE (Norouzi et al., 2014).
  * :class:`CosineAdapter` — a text-aligned model with its own text tower. It
    emits per-window sensor embeddings and label embeddings in a shared space;
    the base scores by cosine similarity (no bridge).

An adapter declares its INPUT CONTRACT (channels / rate / window) so a per-baseline
resampler (added later) can honour it, implements ``setup`` plus its one tier
method, and is decorated with ``@register``. The base owns the shared plumbing —
ground truth, the ConSE/cosine scoring, subject-stratified CIs — via
:mod:`eval.scoring` and :mod:`eval.data`, so there is NO per-baseline dispatch
code and no per-baseline ground-truth handling (the source of the legacy label
bugs). No concrete baseline lives here yet; drop a ``<name>/adapter.py`` in to add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval import data as eval_data
from eval import scoring

# Populated by @register at import time: name -> adapter instance.
REGISTRY: Dict[str, "BaselineAdapter"] = {}


def register(cls):
    """Class decorator: instantiate and add to REGISTRY under ``cls.name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `name` to register")
    REGISTRY[cls.name] = cls()
    return cls


@dataclass(frozen=True)
class InputContract:
    """What a baseline expects its sensor input to look like.

    A ``None`` field means "accepts native" (the adapter handles it internally).
    Consumed by a per-baseline resampler (added with the concrete adapters); the
    base only records it so the eval driver can honour each baseline's contract
    instead of feeding every model the same tensor.
    """
    channels: Optional[Sequence[str]] = None  # required channel names/order, or None
    rate_hz: Optional[float] = None           # required sampling rate, or None
    window_sec: Optional[float] = None        # required window length in seconds, or None


class BaselineAdapter:
    """Base adapter. Subclass :class:`ConSEAdapter` or :class:`CosineAdapter`.

    Subclasses set `name` + `tier`, declare `contract`, and implement `setup`
    plus their tier's prediction method. The shared :meth:`evaluate` turns a
    dataset/stream into the v2 metric bundle.
    """
    name: str = ""
    tier: str = ""  # "conse" | "cosine"
    contract: InputContract = InputContract()

    def setup(self, device):
        """Load model + artifacts once; return an opaque state object."""
        raise NotImplementedError

    def is_incompatible(self, dataset: str) -> Optional[str]:
        """Return a short reason if this model CANNOT be validly scored on
        `dataset` (e.g. a gravity-dependent model on a gravity-removed set), else
        None. The driver records such a case as an explicit, disclosed N/A cell —
        not silently scored, and not counted as a failure."""
        return None

    def predict(self, stream: eval_data.EvalStream, state, device) -> Tuple[List[str], dict]:
        """Per-window predictions over ``stream.eval_labels`` (aligned 1:1 with
        ``stream.windows``) plus an info dict. Implemented by each tier."""
        raise NotImplementedError

    def evaluate(
        self,
        dataset: str,
        stream: str,
        *,
        alignment: str = "non_harmonised",
        device="cpu",
        state=None,
    ) -> dict:
        """Score this baseline on one dataset/stream -> v2 metric bundle.

        Loads the grid, checks compatibility, runs the tier's `predict`, restricts
        to windows whose ground truth is in the candidate vocabulary, and computes
        the metric bundle. Returns ``{"status": "n/a", "reason": ...}`` for a
        disclosed incompatible dataset (no invalid number is produced).
        """
        na = self.is_incompatible(dataset)
        if na is not None:
            return {"status": "n/a", "reason": na, "dataset": dataset, "stream": stream}

        s = eval_data.load_eval_stream(dataset, stream, alignment=alignment)
        if state is None:
            state = self.setup(device)

        preds, info = self.predict(s, state, device)
        if len(preds) != s.n_windows:
            raise ValueError(
                f"{self.name}: predicted {len(preds)} labels for {s.n_windows} windows "
                f"({dataset}/{stream}) — predictions must align 1:1 with windows."
            )

        gt, subjects, keep_idx = scoring.filter_ground_truth(s.gt, s.subjects, s.eval_labels)
        preds = [preds[i] for i in keep_idx]
        return score(gt, preds, subjects, extra=info)


class ConSEAdapter(BaselineAdapter):
    """Closed-vocabulary classifier scored via the ConSE bridge."""
    tier = "conse"

    def window_probs(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        """Per-window softmax over the GLOBAL training labels: (N, K), aligned
        1:1 with ``stream.windows``. K == len(global_labels())."""
        raise NotImplementedError

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        probs = np.asarray(self.window_probs(stream, state, device))
        vocab = global_labels()
        if probs.shape[1] != len(vocab):
            raise ValueError(
                f"{self.name}: window_probs has {probs.shape[1]} columns but the "
                f"global vocabulary has {len(vocab)} labels."
            )
        return scoring.conse_predict(probs, vocab, stream.eval_labels)


class CosineAdapter(BaselineAdapter):
    """Text-aligned model scored by cosine similarity (no bridge)."""
    tier = "cosine"

    def window_embeddings(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        """Per-window L2-normalized sensor embeddings: (N, D)."""
        raise NotImplementedError

    def encode_labels(self, labels: Sequence[str], state, device) -> np.ndarray:
        """Encode the target label strings into the same space: (L, D)."""
        raise NotImplementedError

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        emb = np.asarray(self.window_embeddings(stream, state, device))      # (N, D)
        lab = np.asarray(self.encode_labels(stream.eval_labels, state, device))  # (L, D)
        sims = emb @ lab.T                                                    # (N, L)
        preds = scoring.predict_from_similarity(sims, stream.eval_labels)
        return preds, {"predicted_classes": sorted(set(preds))}


# =============================================================================
# Shared ground truth + scoring (offset-free v2)
# =============================================================================

def load_gt(
    dataset: str,
    stream: str,
    alignment: str = "non_harmonised",
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Canonical ground truth for a dataset/stream: ``(eval_labels, gt_names,
    subjects, keep_idx)``.

    Reads the per-window label strings and subject ids directly from the grid
    (offset-free — never the legacy code-offset path) and drops windows whose
    label is outside the dataset's candidate vocabulary. `keep_idx` indexes the
    original N windows, so any per-window model output aligns via
    ``output[keep_idx]``.
    """
    s = eval_data.load_eval_stream(dataset, stream, alignment=alignment)
    gt_names, subjects, keep_idx = scoring.filter_ground_truth(s.gt, s.subjects, s.eval_labels)
    return s.eval_labels, gt_names, subjects, keep_idx


def score(gt_names, pred_names, subjects, extra: Optional[dict] = None) -> dict:
    """v2 metric bundle: macro-F1 (primary) + balanced-acc + subject CIs + per-class."""
    m = scoring.classification_metrics(gt_names, pred_names)
    # Small-cohort datasets (few subjects) get a jagged, over-wide subject bootstrap → use the
    # leave-one-subject-out jackknife CI instead; larger cohorts keep the bootstrap (#7).
    import numpy as _np
    if len(_np.unique(subjects)) < scoring.SMALL_COHORT_MAX_SUBJECTS:
        m.update(scoring.subject_groupkfold_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    else:
        m.update(scoring.subject_bootstrap_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    m["per_class_f1"] = scoring.per_class_f1(gt_names, pred_names)
    if extra:
        m.update(extra)
    return m


def fit_fingerprint(**parts) -> str:
    """Hash of everything that changes what a fitted ConSE head IS (Phase 1.6 / audit H6).

    The old caches validated only `labels` (and later corpus mode), so they silently survived
    changes to the subject split, the seed, the probe architecture, the backbone checkpoint, the
    per-stream cap and the optimizer settings — an audit demonstrated a fabricated cache with
    ``n_windows=1`` being accepted. Any change to a part here invalidates the cache and forces a
    refit, which is the safe default: a wrong cache produces a silently wrong number.
    """
    import hashlib
    import json
    blob = json.dumps({k: parts[k] for k in sorted(parts)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_probe(feat_dim: int, n_classes: int, hidden: int = 512):
    """The ONE probe architecture every ConSE-tier baseline fits on its frozen features.

    Phase 1.4. Previously harnet used the official 2-layer ``EvaClassifier``
    (feat→512→n, ~300k params) while halo, crosshar and limubert used a single
    ``nn.Linear(feat→n)`` (~24k) — so harnet was given **strictly more probe capacity than
    everyone else, including us**. That is not a "same probe" comparison.

    Unifying on the 2-layer form: leaves harnet exactly at its paper's head, and *strengthens*
    the other three (it also moves LiMU-BERT toward its paper's non-linear downstream classifier,
    where a linear probe understates it). Strengthening the baselines is the honest direction —
    if our numbers survive it they are better earned.

    Shape matches ssl-wearables ``EvaClassifier``: Linear(feat, hidden) → ReLU → Linear(hidden, n).
    """
    import torch.nn as nn
    return nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def global_labels() -> List[str]:
    """The global ConSE training-label vocabulary."""
    return eval_data.load_global_labels()
