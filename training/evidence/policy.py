"""Single-source Phase-B recipe with derived, non-conflicting retrieval limits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# These are recipe constants, not independent experiment knobs. The active index is sampled in
# source-window units so a future trainable tokenizer can refresh it without re-encoding thousands
# of unrelated one-off patches.
ARCHIVE_BUDGET_WINDOWS = 250_000
ACTIVE_WINDOWS_PER_LABEL = 16
# How often the retrievable memory bank is redrawn from the reservoir. At 100 a 3000-step run only
# ever made ~13% of the 248k-window reservoir reachable — the rest was encoded, stored and never
# retrievable. At 5 that becomes ~71%. The cadence used to be pinned by cost: redrawing took ~300 ms
# because the sampler rescanned every window once per label. That scan is now precomputed, so a
# refresh costs ~88 ms and the whole cadence adds ~26 s to a ~40 min run.
ACTIVE_REFRESH_STEPS = 5
RETRIEVAL_SUBSPACES = 4
RETRIEVAL_SUBSPACE_DIM = 64
RETRIEVAL_PROJECTION_EMA = 0.995
RETRIEVAL_TEMPERATURE = 0.07
RETRIEVAL_OVERSAMPLE = 8
MIN_TOPK_PER_SUBSPACE = 4
MAX_TOPK_PER_SUBSPACE = 32

# Training begins by exposing more of the already-retrieved hard shortlist to the differentiable
# evidence decoder.  Both values anneal to the exact deployment recipe over the first third of the
# run; validation and inference always use the deployment values above.  This gives plausible rows
# outside the eventual evidence roster a gradient early on without introducing full-bank soft
# retrieval or a train-time candidate-conditioned selector.
RETRIEVAL_EXPLORATION_FRACTION = 1 / 3
RETRIEVAL_EXPLORATION_BUDGET_MULTIPLIER = 4
RETRIEVAL_EXPLORATION_TEMPERATURE = 0.20

# Zero-support checkpoint selection is a guardrail around the adaptation objective. Fixed canaries
# are deterministic, but one percentage point avoids rejecting a checkpoint over tiny floating-point
# and stochastic-kernel differences while still preventing a meaningful semantic-transfer collapse.
ZERO_SUPPORT_GUARD_TOLERANCE = 0.01

# The adaptation conditions. These exist because `eval_enrollment` reports them: training has to
# cover the conditions we evaluate on or the model is off-distribution at test time. They are
# sampled independently outside one load-bearing counterfactual support/zero pair and one guaranteed
# alias episode per optimizer step. The independently sampled remainder retains broad stochastic
# coverage while those anchors ensure every balanced loss group is present.
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

# One optimizer update is an intentionally small collection of adaptation tasks. These
# two values describe compute shape, not model behavior: every episode below gets its own candidate
# set, support overlay, aliases, distractors, physical view, and query executions, except for the
# deliberately matched support/zero counterfactual pair. Keeping them as
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
PHASE_B_TRAINING_REGIME = "episodic_memory_adaptation_v17_k0_credit_wider_reservoir"

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

    def training_retrieval(self, step: int, total_steps: int) -> tuple[int, float]:
        """Return the training evidence budget and score temperature for this optimizer step.

        The selector still performs the same hard query-driven top-k lookup. Only the number of
        unique retrieved rows shown to the decoder, and the softness of their relative attention
        prior, differ during the early credit-assignment curriculum.
        """
        if step < 1 or total_steps < 1:
            raise ValueError("step and total_steps must be positive")
        anneal_steps = max(1, math.ceil(total_steps * RETRIEVAL_EXPLORATION_FRACTION))
        progress = min(1.0, max(0.0, (step - 1) / max(1, anneal_steps - 1)))
        multiplier = (
            RETRIEVAL_EXPLORATION_BUDGET_MULTIPLIER
            + (1.0 - RETRIEVAL_EXPLORATION_BUDGET_MULTIPLIER) * progress
        )
        budget = max(self.evidence_budget, int(round(self.evidence_budget * multiplier)))
        temperature = (
            RETRIEVAL_EXPLORATION_TEMPERATURE
            + (RETRIEVAL_TEMPERATURE - RETRIEVAL_EXPLORATION_TEMPERATURE) * progress
        )
        return budget, float(temperature)

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
            "training_retrieval_curriculum": {
                "anneal_fraction": RETRIEVAL_EXPLORATION_FRACTION,
                "initial_budget_multiplier": RETRIEVAL_EXPLORATION_BUDGET_MULTIPLIER,
                "initial_temperature": RETRIEVAL_EXPLORATION_TEMPERATURE,
                "final_temperature": RETRIEVAL_TEMPERATURE,
            },
        })
        if query_patches is not None:
            value["topk_per_subspace"] = self.topk_per_subspace(query_patches)
        return value
