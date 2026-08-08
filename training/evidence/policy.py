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
SOFT_RETRIEVAL_TEMPERATURE_START = 0.20
SOFT_RETRIEVAL_TEMPERATURE_END = 0.07
SOFT_RETRIEVAL_ANNEAL_STEPS = 500
# The all-row route is a deliberately biased estimator for rows outside hard top-k. Real-bank
# probes found its raw retriever gradient 5-8x larger than the selected hard path. Keep the route,
# but scale it so it teaches missed rows without becoming the effective training objective.
SOFT_BACKWARD_SCALE = 0.10
RETRIEVAL_OVERSAMPLE = 8
MIN_TOPK_PER_SUBSPACE = 4
MAX_TOPK_PER_SUBSPACE = 32

DISTRACTOR_MODES = ("random", "language", "motion_family", "physical", "mixed")

# Consolidated adaptation curriculum. These are fixed recipe strata rather than independent
# probabilities: cycling a shuffled copy of the tuple gives exact long-run 25% coverage.
EPISODE_TYPES = (
    "semantic_zero_support",
    "ordinary_few_support",
    "cross_subject_few_support",
    "same_subject_enrollment",
)
SUPPORT_COUNTS = (1, 2, 4, 8)
CANDIDATE_COUNTS = (4, 8, 12, 16)
LABEL_TEXT_MODES = ("coherent", "random_alias")
PHYSICAL_VIEW_MODES = ("clean", "augmented")

# External evaluation is deliberately split so repeated development runs do not consume every
# dataset that is supposed to support the final claim. The test roster is only selected explicitly.
PHASE_B_DEV_DATASETS = ("motionsense", "realworld", "shoaib")
PHASE_B_TEST_DATASETS = ("inclusivehar", "usc_had", "tnda_har", "ut_complex")

# Artifact guard for the complete predictor recipe. Bump this whenever a behavioral training path
# changes even if the decoder parameter schema stays compatible.
PHASE_B_TRAINING_REGIME = "episodic_memory_adaptation_hard_forward_soft_backward_v5"

# Fine-tuning constants. The online tokenizer uses a deliberately small LR, while an EMA copy
# supplies stable retrieval keys and the inference representation.
TOKENIZER_LR_SCALE = 0.05
TOKENIZER_EMA_DECAY = 0.999
TOKENIZER_FINETUNE_WARMUP_STEPS = 100
TOKENIZER_KEY_REFRESH_STEPS = 100
TOKENIZER_KEY_REFRESH_SHARDS = 8


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

    @property
    def max_per_window(self) -> int:
        return max(1, self.evidence_budget // 16)

    @property
    def max_per_label(self) -> int:
        return max(2, int(round(self.evidence_budget * 3 / 16)))

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
            "max_per_window": self.max_per_window,
            "max_per_label": self.max_per_label,
            "tau_retr": RETRIEVAL_TEMPERATURE,
            "soft_tau_start": SOFT_RETRIEVAL_TEMPERATURE_START,
            "soft_tau_end": SOFT_RETRIEVAL_TEMPERATURE_END,
            "soft_tau_anneal_steps": SOFT_RETRIEVAL_ANNEAL_STEPS,
            "soft_backward_scale": SOFT_BACKWARD_SCALE,
            "dynamic_topk_range": [MIN_TOPK_PER_SUBSPACE, MAX_TOPK_PER_SUBSPACE],
        })
        if query_patches is not None:
            value["topk_per_subspace"] = self.topk_per_subspace(query_patches)
        return value
