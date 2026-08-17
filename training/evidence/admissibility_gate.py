"""The admissibility gate — a low-rank bilinear form over frozen sensor and concept text.

WHAT REPLACED WHAT
------------------
``resolvability.gate_tensor`` answered "can this configuration witness this concept" with a
dictionary lookup keyed by the literal string ``"<dataset>/<stream>"`` and an exact label match. That
has one fatal property: a genuinely novel label misses every key, every entry falls back to the
neutral default, and a uniform gate multiplies all candidates equally — so the gate provably cannot
change the argmax in exactly the open-vocabulary case the design exists for.

This module makes admissibility a *function of text* instead of a lookup:

    u = A · sbert(sensor_text)          A: (r, 384)   "where does this sensor sit"
    v = B · sbert(concept_text)         B: (r, 384)   "where does this concept move"
    w = sigmoid(u^T LAMBDA v + b)       LAMBDA: (r, r)

Both arguments are text the system already receives, so a never-seen sensor and a never-seen label
both produce a value with no table entry anywhere. Nothing is named: latent body parts and concept
groups are DISCOVERED, and whether they correspond to anatomy is checked afterwards rather than
assumed (see ``latent_measurement_correlation``).

WHY BILINEAR, AND NOT ADDITIVE
------------------------------
This is forced by measurement, not taste. ``resolvability.paired_contrast`` found a stream-profile
correlation of +0.42 with 97% of stream pairs INVERTING on simultaneous recordings: the pocket beats
the wrist on squats and loses to it on tricep extensions. An additive score ``f(sensor)+g(concept)``
assigns each sensor ONE quality number, so it must rank concepts identically for every sensor and can
never produce that flip. A signed rank-1 product can express a two-by-two inversion, so rank 1 remains
a valid ablation. Rank 8 is the current default: it remains a strong compression of the 384-wide
text space while allowing several independent configuration-concept factors. Lower ranks remain
registered ablations in the held-out extrapolation study.

WHY r STAYS SMALL
-----------------
There are ~20 streams in the bank and ~10 distinct placements. At large r, ``A`` has enough capacity
to hand every stream its own near-orthogonal code, LAMBDA becomes the old lookup table in disguise,
and the model reproduces the training cells while generalising nothing. The compression is what
forces the latents to mean something. ``r`` is not a guess: ``gate_extrapolation`` sweeps it under
held-out placements and concepts, and the right value is where held-out error stops improving.

LAMBDA IS NOT EXTRA CAPACITY
----------------------------
When ``A`` and ``B`` are both free, ``u^T LAMBDA v = s^T A^T LAMBDA B c`` and LAMBDA can be absorbed
into ``A``. It is kept for two reasons that are not about expressiveness: it is the small
well-conditioned object the PCA ablation fits (r*r parameters rather than a full projection), and
with ``A``/``B`` frozen it IS the readable coupling matrix — the figure a reviewer
can argue with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_DIM = 384
DEFAULT_RANK = 8
# Below this max-cosine to any training sensor text, the configuration is treated as unfamiliar and
# the gate is blended back toward the neutral value. Without it a linear map extrapolates confidently
# into regions it has never seen, which is strictly worse than the lookup's honest "never measured".
DEFAULT_NOVELTY_FLOOR = 0.55


def _leave_one_out_novelty_floor(embeddings: torch.Tensor) -> torch.Tensor:
    """Robust floor for centered-text nearest-neighbour familiarity.

    Centering removes shared template words ("accelerometer", "recorded", "on") before cosine is
    used as a novelty score. The fifth percentile of leave-one-description-out nearest-neighbour
    similarity is conservative: familiar paraphrases pass, while an unrelated configuration or
    concept is blended back to the neutral gate rather than extrapolated confidently.
    """
    unique = torch.unique(embeddings.detach(), dim=0)
    if len(unique) < 3:
        return torch.tensor(DEFAULT_NOVELTY_FLOOR, dtype=embeddings.dtype)
    centered = F.normalize(unique - unique.mean(0, keepdim=True), dim=-1)
    similarity = centered @ centered.t()
    similarity.fill_diagonal_(float("-inf"))
    nearest = similarity.max(1).values
    return torch.quantile(nearest, 0.05).clamp(-0.5, 0.95)


def _pca_projection(embeddings: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-``rank`` principal directions of ``embeddings`` (N, D) as an (rank, D) projection.

    UNSUPERVISED, so it spends none of the measured cells — which is the whole point: the warm start
    can then fit LAMBDA alone against every measured cell instead of splitting that budget with a
    1536-parameter projection it cannot afford.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got {tuple(embeddings.shape)}")
    n, d = embeddings.shape
    if rank > min(n, d):
        raise ValueError(f"rank {rank} exceeds min(n_samples={n}, dim={d})")
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    # SVD rather than torch.pca_lowrank: deterministic, and these matrices are tiny.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank].contiguous()


class AdmissibilityGate(nn.Module):
    """(R, C) admissibility weights from (R, 384) sensor text and (C, 384) concept text."""

    def __init__(self, rank: int = DEFAULT_RANK, text_dim: int = TEXT_DIM):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.sensor_proj = nn.Linear(text_dim, rank, bias=False)
        self.concept_proj = nn.Linear(text_dim, rank, bias=False)
        # Identity coupling = a plain dot product between the two latent codes ("the sensor is where
        # the motion is"). Off-diagonals let one latent location partially observe another, which is
        # what the model has to learn.
        self.coupling = nn.Parameter(torch.eye(rank))
        self.offset = nn.Parameter(torch.zeros(()))
        # Neutral value used for unfamiliar configurations. Set by `warm_start`; the measured median
        # is 0.33, so a hard-coded 0.5 would treat an unmeasured pair as MORE resolvable than a
        # typical measured one.
        self.register_buffer("neutral", torch.tensor(0.5))
        self.register_buffer("known_sensors", torch.zeros(0, text_dim))
        self.register_buffer("known_concepts", torch.zeros(0, text_dim))
        self.register_buffer("sensor_novelty_floor", torch.tensor(DEFAULT_NOVELTY_FLOOR))
        self.register_buffer("concept_novelty_floor", torch.tensor(DEFAULT_NOVELTY_FLOOR))

    # ------------------------------------------------------------------ forward
    def latents(self, sensor_text: torch.Tensor, concept_text: torch.Tensor):
        """(R,384),(C,384) -> (R,r),(C,r). Exposed so the post-hoc latent audit can read them."""
        return self.sensor_proj(sensor_text), self.concept_proj(concept_text)

    def forward(self, sensor_text: torch.Tensor, concept_text: torch.Tensor) -> torch.Tensor:
        u, v = self.latents(sensor_text, concept_text)
        w = torch.sigmoid(u @ self.coupling @ v.t() + self.offset)       # (R, C)
        return self.abstain(w, sensor_text, concept_text)

    def paired(
        self,
        sensor_text: torch.Tensor,
        concept_text: torch.Tensor,
        *,
        abstain: bool = False,
    ) -> torch.Tensor:
        """Score aligned ``(sensor[i], concept[i])`` pairs without an outer product."""
        if sensor_text.shape != concept_text.shape or sensor_text.ndim != 2:
            raise ValueError("paired inputs must have matching (N,D) shapes")
        u, v = self.latents(sensor_text, concept_text)
        w = torch.sigmoid((u * (v @ self.coupling.t())).sum(dim=-1) + self.offset)
        if not abstain:
            return w
        sensor_alpha = self._familiarity(
            sensor_text, self.known_sensors, self.sensor_novelty_floor
        )
        concept_alpha = self._familiarity(
            concept_text, self.known_concepts, self.concept_novelty_floor
        )
        return sensor_alpha * concept_alpha * w + (
            1.0 - sensor_alpha * concept_alpha
        ) * self.neutral

    @staticmethod
    def _familiarity(
        value: torch.Tensor, known: torch.Tensor, floor: torch.Tensor,
    ) -> torch.Tensor:
        if known.numel() == 0:
            return torch.ones(len(value), device=value.device, dtype=value.dtype)
        known = known.to(device=value.device, dtype=value.dtype)
        centroid = known.mean(0, keepdim=True)
        centered_value = F.normalize(value - centroid, dim=-1)
        centered_known = F.normalize(known - centroid, dim=-1)
        nearest = (centered_value @ centered_known.t()).max(dim=1).values
        floor = floor.to(device=value.device, dtype=value.dtype)
        return ((nearest - floor) / (1.0 - floor).clamp_min(1e-6)).clamp(0.0, 1.0)

    def abstain(
        self, w: torch.Tensor, sensor_text: torch.Tensor, concept_text: torch.Tensor,
    ) -> torch.Tensor:
        """Blend toward ``neutral`` when either side is unlike the fitted text supports.

        A novel configuration and a novel concept are separate extrapolation risks. Treating only the
        sensor side as novel made arbitrary episode aliases receive confident semantic penalties even
        though their words intentionally carry no activity meaning.
        """
        if self.known_sensors.numel() == 0 and self.known_concepts.numel() == 0:
            return w
        sensor_alpha = self._familiarity(
            sensor_text, self.known_sensors, self.sensor_novelty_floor
        )
        concept_alpha = self._familiarity(
            concept_text, self.known_concepts, self.concept_novelty_floor
        )
        alpha = sensor_alpha.unsqueeze(1) * concept_alpha.unsqueeze(0)
        return alpha * w + (1.0 - alpha) * self.neutral

    @torch.no_grad()
    def extend_known_support(
        self,
        sensor_text: torch.Tensor | None = None,
        concept_text: torch.Tensor | None = None,
    ) -> None:
        """Add Stage-2 training descriptions to the novelty reference support.

        The buffers are not learned parameters. Leaving them fixed after refining on a new sensor or
        wording would make ``abstain`` continue to suppress an input the gate has actually trained
        on, so every refinement artifact must update them before it is saved.
        """
        if sensor_text is not None and sensor_text.numel():
            values = sensor_text.detach().to(self.known_sensors.device, self.known_sensors.dtype)
            self.known_sensors = torch.unique(
                torch.cat((self.known_sensors, values), dim=0), dim=0
            )
            self.sensor_novelty_floor.copy_(
                _leave_one_out_novelty_floor(self.known_sensors).to(self.sensor_novelty_floor)
            )
        if concept_text is not None and concept_text.numel():
            values = concept_text.detach().to(self.known_concepts.device, self.known_concepts.dtype)
            self.known_concepts = torch.unique(
                torch.cat((self.known_concepts, values), dim=0), dim=0
            )
            self.concept_novelty_floor.copy_(
                _leave_one_out_novelty_floor(self.known_concepts).to(self.concept_novelty_floor)
            )

    # ------------------------------------------------------------------ telemetry
    @staticmethod
    def spread(w: torch.Tensor) -> dict[str, float]:
        """THE COLLAPSE GUARD. The cheapest way for training to reduce loss may be w == constant.

        A collapsed gate is a RESULT — "admissibility does not help" — and must be visible rather
        than silently absorbed, so this is logged every validation rather than inspected on demand.
        """
        w = w.detach()
        if w.numel() == 0:
            return {"gate/mean": 0.0, "gate/std": 0.0, "gate/min": 0.0, "gate/max": 0.0,
                    "gate/log_penalty_mean": 0.0, "gate/log_penalty_std": 0.0}
        log_penalty = -w.clamp_min(1e-8).log()
        return {
            "gate/mean": float(w.mean()), "gate/std": float(w.std(unbiased=False)),
            "gate/min": float(w.min()), "gate/max": float(w.max()),
            "gate/log_penalty_mean": float(log_penalty.mean()),
            "gate/log_penalty_std": float(log_penalty.std(unbiased=False)),
        }


# --------------------------------------------------------------------------------------------
# Warm start
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TableObservations:
    """Measured cells, flattened. ``value`` is resolvability in [0,1]."""

    sensor_text: list[str]
    concept_text: list[str]
    value: np.ndarray
    stream_key: list[str]
    concept: list[str]

    def __len__(self) -> int:
        return len(self.value)


def observations_from_table(
    table: dict, sensor_text_for: dict[str, str] | None = None,
) -> TableObservations:
    """Flatten ``resolvability.json`` into (sensor text, concept text, value) rows.

    ``sensor_text_for`` maps ``"<dataset>/<stream>::<modality>"`` to the sensor DESCRIPTION — the
    text the deployment supplies. Earlier drafts rebuilt a placement string from the curated
    deployment policy, which is a second interface smuggled in beside the text and defeats the point.
    Streams with no description are skipped rather than guessed at.
    """
    s_text, c_text, values, keys, concepts = [], [], [], [], []
    source = table.get("per_sensor")
    if table.get("schema_version") != 2 or source is None:
        raise ValueError(
            "admissibility fitting requires a schema-2 per-sensor resolvability table"
        )
    for key, payload in source.items():
        text = payload.get("sensor_text")
        if text is None and sensor_text_for is not None:
            text = sensor_text_for.get(key)
        if text is None:
            continue
        for concept, score in payload["labels"].items():
            s_text.append(text)
            c_text.append(concept.replace("_", " "))
            values.append(float(score))
            keys.append(key)
            concepts.append(concept)
    return TableObservations(s_text, c_text, np.asarray(values, dtype=np.float32), keys, concepts)


def fit_gate(
    sensor_emb: torch.Tensor,          # (N, 384) per observation
    concept_emb: torch.Tensor,         # (N, 384) per observation
    target: torch.Tensor,              # (N,) measured resolvability
    rank: int = DEFAULT_RANK,
    mode: str = "full",                # 'full' = free projections (default) | 'pca' = fixed
    steps: int = 600,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 20260812,
) -> AdmissibilityGate:
    """Fit a gate to measured cells.

    ``mode='full'`` is the default. Unsupervised PCA retains directions of greatest text variance,
    which need not be directions that predict physical observability. ``mode='pca'`` remains a
    low-capacity ablation; the current train-only table must be evaluated under held-out folds before
    either projection choice is treated as an empirical result.
    """
    if mode not in ("pca", "full"):
        raise ValueError("mode must be 'pca' or 'full'")
    torch.manual_seed(seed)
    gate = AdmissibilityGate(rank=rank, text_dim=sensor_emb.shape[1])

    with torch.no_grad():
        uniq_s = torch.unique(sensor_emb, dim=0)
        uniq_c = torch.unique(concept_emb, dim=0)
        gate.sensor_proj.weight.copy_(_pca_projection(uniq_s, rank))
        gate.concept_proj.weight.copy_(_pca_projection(uniq_c, rank))
        gate.neutral.fill_(float(target.median()))
        gate.known_sensors = uniq_s.clone()
        gate.known_concepts = uniq_c.clone()
        gate.sensor_novelty_floor.copy_(_leave_one_out_novelty_floor(uniq_s))
        gate.concept_novelty_floor.copy_(_leave_one_out_novelty_floor(uniq_c))

    if mode == "pca":
        for p in (gate.sensor_proj.weight, gate.concept_proj.weight):
            p.requires_grad_(False)
    params = [p for p in gate.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # Abstention is deliberately OFF during the fit: every training row is by construction a familiar
    # configuration, so blending toward neutral here would only damp the gradient.
    for _ in range(steps):
        opt.zero_grad()
        pred = gate.paired(sensor_emb, concept_emb)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
    return gate


@torch.no_grad()
def predict_cells(gate: AdmissibilityGate, sensor_emb: torch.Tensor,
                  concept_emb: torch.Tensor, abstain: bool = True) -> torch.Tensor:
    """Per-observation (paired, not outer-product) gate values — the form the fit is scored on."""
    return gate.paired(sensor_emb, concept_emb, abstain=abstain)


@torch.no_grad()
def latent_measurement_correlation(
    gate: AdmissibilityGate, obs: TableObservations,
    sensor_emb: torch.Tensor, concept_emb: torch.Tensor,
) -> dict:
    """Do the learned latents track the supplied physical measurement?

    This is an independent guard only when ``obs`` was held out from both warm-start fitting and any
    Stage-2 refinement. On fitted cells it is a fit diagnostic, not validation evidence.
    """
    pred = predict_cells(gate, sensor_emb, concept_emb, abstain=False).numpy()
    truth = obs.value
    if truth.std() == 0 or pred.std() == 0:
        return {"pearson_r": None, "note": "no variance to correlate"}
    r = float(np.corrcoef(pred, truth)[0, 1])
    # Does the gate reproduce INVERSIONS, or only the average difficulty of each concept? Subtracting
    # each concept's mean removes "some activities are just easier", leaving the configuration-
    # dependent part — which is the only part the contribution claims.
    by_concept: dict[str, list[int]] = {}
    for i, concept in enumerate(obs.concept):
        by_concept.setdefault(concept, []).append(i)
    dp, dt = [], []
    for idx in by_concept.values():
        if len(idx) < 2:
            continue
        idx_a = np.asarray(idx)
        dp.extend((pred[idx_a] - pred[idx_a].mean()).tolist())
        dt.extend((truth[idx_a] - truth[idx_a].mean()).tolist())
    within = None
    if len(dp) > 2 and np.std(dp) > 0 and np.std(dt) > 0:
        within = float(np.corrcoef(dp, dt)[0, 1])
    return {
        "pearson_r": round(r, 4),
        "within_concept_pearson_r": round(within, 4) if within is not None else None,
        "n_cells": int(len(truth)),
        "n_concepts_with_multiple_configs": sum(1 for v in by_concept.values() if len(v) >= 2),
        "note": ("within_concept_pearson_r is the load-bearing one: it removes per-concept difficulty "
                 "and measures only the configuration-dependent structure the gate exists to capture"),
    }
