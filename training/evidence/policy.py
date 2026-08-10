"""Single-source Phase-B recipe with derived, non-conflicting retrieval limits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# These are recipe constants, not independent experiment knobs. The active index is sampled in
# source-window units so a future trainable tokenizer can refresh it without re-encoding thousands
# of unrelated one-off patches.
ARCHIVE_BUDGET_WINDOWS = 250_000
ACTIVE_WINDOWS_PER_LABEL = 16
ACTIVE_REFRESH_STEPS = 100
RETRIEVAL_SUBSPACES = 4
RETRIEVAL_SUBSPACE_DIM = 64
RETRIEVAL_PROJECTION_EMA = 0.995
RETRIEVAL_TEMPERATURE = 0.07
RETRIEVAL_OVERSAMPLE = 8
MIN_TOPK_PER_SUBSPACE = 4
MAX_TOPK_PER_SUBSPACE = 32

# The adaptation conditions. These exist because `eval_enrollment` reports them: training has to
# cover the conditions we evaluate on or the model is off-distribution at test time. They are
# sampled uniformly per episode, NOT scheduled — an earlier revision cycled a shuffled copy of the
# tuple for exact long-run 25% coverage, which forced `episodes_per_step` to be a multiple of four
# and bought nothing at the 3,000-step default, where every condition is still sampled hundreds of
# times.
EPISODE_TYPES = (
    "semantic_zero_support",
    "ordinary_few_support",
    "cross_subject_few_support",
    "same_subject_enrollment",
)

# Inclusive ranges, sampled per episode. Previously enumerated tuples ((1,2,4,8) and
# (2,3,4,8,12,16)) that had to be cycled and padded to the batch size; a range says the same thing
# with less machinery and no batch-size coupling.
SUPPORT_COUNT_RANGE = (1, 8)
CANDIDATE_COUNT_RANGE = (2, 16)

# The fixed validation canaries enumerate instead of sampling, deliberately: they are the
# checkpoint-selection target and the generalization-gap probe, so they must be byte-identical
# across runs and across checkpoints. Training sampling and validation fixtures are different
# requirements and no longer share a constant.
VALIDATION_SUPPORT_COUNTS = (1, 2, 4, 8)
VALIDATION_CANDIDATE_COUNTS = (2, 3, 4, 8, 12, 16)

# One optimizer update is an intentionally small collection of independent adaptation tasks.  These
# two values describe compute shape, not model behavior: every episode below gets its own candidate
# set, support overlay, aliases, distractors, physical view, and query executions.  Keeping them as
# named recipe constants prevents the old ambiguous `batch` from being mistaken for episode count.
EPISODES_PER_STEP = 8
QUERIES_PER_EPISODE = 8

LABEL_TEXT_MODES = ("coherent", "random_alias")
PHYSICAL_VIEW_MODES = ("clean", "augmented")

# Probability that a supported episode relabels its candidates with meaningless episode-local
# aliases. This is the only condition that forces example-based rather than text-based inference, so
# it is load-bearing rather than decorative. Alias episodes are always fully enrolled: an
# un-enrolled candidate under a meaningless name carries no information at all.
ALIAS_PROBABILITY = 1 / 3

# How many of an episode's candidates carry enrolled examples is drawn uniformly from
# 1..candidate_count, so partial and full enrollment fall out of one draw rather than being named
# strata. The mixture matters: if no candidate is enrolled there is no substrate to answer from
# (Phase A is label-free, so nothing maps a window to a label name), and if every candidate is
# enrolled a prototype over the declared support is already optimal, so neither retrieval over
# background memory nor language is needed. Partial enrollment is the only regime in which the
# evidence engine is the sole method that can answer at all — a prototype is undefined for a
# candidate with zero support.

# Scale of the closed-form retrieval vote. Not on the forward path: it sets the units of the
# untrained retrieval + text-ensemble control that every Phase-B result is quoted against, so the
# control and the learned logits stay numerically comparable.
RETRIEVAL_VOTE_SCALE = 10.0

# External evaluation is deliberately split so repeated development runs do not consume every
# dataset that is supposed to support the final claim. The test roster is only selected explicitly.
PHASE_B_DEV_DATASETS = ("motionsense", "realworld", "shoaib")
PHASE_B_TEST_DATASETS = ("inclusivehar", "usc_had", "tnda_har", "ut_complex")

# Artifact guard for the complete predictor recipe. Bump this whenever a behavioral training path
# changes even if the decoder parameter schema stays compatible.
PHASE_B_TRAINING_REGIME = "episodic_memory_adaptation_v14_balanced_token_components"

# Evaluation artifacts are independently versioned from training artifacts. A training-only change
# must not invalidate a held-out protocol, while a cohort/scoring/runtime-memory change must never
# be allowed into a comparison table under an old protocol name.
PHASE_B_EVALUATION_REGIME = "enrollment_adaptation_v2_paired_execution_protocol"

# Fine-tuning constants. The online tokenizer uses a deliberately small LR, while an EMA copy
# supplies stable retrieval keys and the inference representation.
TOKENIZER_LR_SCALE = 0.05
TOKENIZER_EMA_DECAY = 0.999
TOKENIZER_FINETUNE_WARMUP_STEPS = 100


@dataclass(frozen=True)
class PhaseBPolicy:
    """The only public retrieval-capacity choice is the final evidence budget."""

    evidence_budget: int = 64
    tokenizer_mode: str = "frozen"

    def __post_init__(self) -> None:
        if self.evidence_budget < 8:
            raise ValueError("evidence_budget must be at least 8")
        if self.tokenizer_mode not in {"frozen", "ema_finetune"}:
            raise ValueError("tokenizer_mode must be 'frozen' or 'ema_finetune'")

    def topk_per_subspace(self, query_patches: int) -> int:
        """Derive selector K so longer/multi-sensor queries do not inflate raw retrieval."""
        q = max(1, int(query_patches))
        raw = math.ceil(
            RETRIEVAL_OVERSAMPLE * self.evidence_budget / (q * RETRIEVAL_SUBSPACES)
        )
        return max(MIN_TOPK_PER_SUBSPACE, min(MAX_TOPK_PER_SUBSPACE, raw))

    def as_dict(self, *, query_patches: int | None = None) -> dict:
        value = asdict(self)
        value.update({
            "active_windows_per_label": ACTIVE_WINDOWS_PER_LABEL,
            "active_refresh_steps": ACTIVE_REFRESH_STEPS,
            "n_subspaces": RETRIEVAL_SUBSPACES,
            "subspace_dim": RETRIEVAL_SUBSPACE_DIM,
            "subspace_ema": RETRIEVAL_PROJECTION_EMA,
            "max_evidence": self.evidence_budget,
            "tau_retr": RETRIEVAL_TEMPERATURE,
            "dynamic_topk_range": [MIN_TOPK_PER_SUBSPACE, MAX_TOPK_PER_SUBSPACE],
        })
        if query_patches is not None:
            value["topk_per_subspace"] = self.topk_per_subspace(query_patches)
        return value
