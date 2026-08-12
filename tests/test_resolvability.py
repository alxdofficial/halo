"""Gate tests for the resolvability measurement.

The measurement decides whether the admissibility gate has anything to gate on, so its failure modes
matter more than its happy path:

  * the score must come from SIGNAL separability, never from label-string semantics — otherwise it
    would be re-measuring the text interface we already showed is inert, and would be circular;
  * a configuration that cannot separate a concept must score ~0, and one that separates it perfectly
    ~1, with chance mapped to 0 rather than 0.5;
  * the paired contrast must only compare SIMULTANEOUS streams;
  * concept-dependence must distinguish "a fixed placement-quality ordering" from "which placement
    wins depends on the concept" — the distinction the gate's third argument exists for;
  * unmeasured (stream, label) pairs must return the neutral default, never a veto or a licence.
"""

from __future__ import annotations

import numpy as np
import torch

from training.evidence.resolvability import (
    _concept_dependence, _one_vs_rest_resolvability, gate_tensor, paired_contrast,
)


def _separable(n=200, d=8, sep=6.0, seed=0):
    """Two well-separated clusters -> the configuration CAN witness the concept."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d)).astype(np.float32)
    labels = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    z[labels == "a", 0] += sep
    subjects = np.array([f"s{i % 4}" for i in range(n)])
    return z, labels, subjects


def test_separable_concept_scores_high():
    z, labels, subjects = _separable()
    scores = _one_vs_rest_resolvability(z, labels, subjects)
    # 0.75 corresponds to one-vs-rest balanced accuracy 0.875 — unambiguously "this configuration
    # witnesses this concept", without pinning the test to the exact kNN tie-breaking behaviour.
    assert scores["a"] >= 0.75 and scores["b"] >= 0.75, scores


def test_indistinguishable_concept_scores_near_zero():
    """Chance maps to 0, not 0.5 — 'this configuration cannot witness this concept'."""
    z, labels, subjects = _separable(sep=0.0)          # identical distributions
    scores = _one_vs_rest_resolvability(z, labels, subjects)
    assert all(v < 0.25 for v in scores.values()), scores


def test_score_is_signal_not_label_semantics():
    """Renaming the labels must not move the score. Guards against circularity with the text path."""
    z, labels, subjects = _separable()
    base = _one_vs_rest_resolvability(z, labels, subjects)
    renamed = np.where(labels == "a", "zzz_nonsense_token", "qqq_other_token")
    after = _one_vs_rest_resolvability(z, renamed, subjects)
    assert sorted(base.values()) == sorted(after.values())


def test_single_subject_stream_is_unscorable():
    """Subject-disjoint scoring is impossible with one subject; report nothing rather than a number."""
    z, labels, _ = _separable()
    assert _one_vs_rest_resolvability(z, labels, np.array(["only"] * len(labels))) == {}


# ------------------------------------------------------------------------- paired contrast
def _per_stream(dataset, streams: dict[str, dict[str, float]]) -> dict:
    return {f"{dataset}/{s}": {"dataset": dataset, "stream": s, "n_windows": 100, "labels": lab}
            for s, lab in streams.items()}


def test_paired_contrast_only_covers_simultaneous_datasets():
    """A non-paired dataset's streams differ in subjects and protocol, so the contrast is invalid."""
    per_stream = _per_stream("wisdm", {"phone_pocket": {"walk": 0.9}, "watch_wrist": {"walk": 0.2}})
    assert paired_contrast(per_stream) == {}


def test_paired_contrast_reports_the_gap():
    per_stream = _per_stream("mmfit", {"left_wrist": {"curls": 1.0, "squats": 0.3},
                                       "right_pocket": {"curls": 0.1, "squats": 0.95}})
    out = paired_contrast(per_stream)["mmfit"]
    assert out["labels"]["curls"]["best_stream"] == "left_wrist"
    assert out["labels"]["squats"]["best_stream"] == "right_pocket"
    assert out["labels"]["curls"]["gap"] > 0.8


def test_paired_contrast_skips_labels_seen_by_one_stream():
    per_stream = _per_stream("mmfit", {"left_wrist": {"curls": 1.0, "solo": 0.5},
                                       "right_pocket": {"curls": 0.1}})
    assert "solo" not in paired_contrast(per_stream)["mmfit"]["labels"]


# ---------------------------------------------------------------------- concept dependence
def test_fixed_quality_ordering_gives_high_correlation():
    """One stream uniformly better: a scalar per placement would suffice, no third argument needed."""
    by_label = {f"l{i}": {"good": 0.9 - 0.05 * i, "bad": 0.5 - 0.05 * i} for i in range(6)}
    out = _concept_dependence(by_label)
    assert out["mean_correlation"] > 0.9
    assert out["inverting_pairs"] == 0


def test_concept_dependent_ordering_inverts():
    """Each stream wins on a different concept -> the gate genuinely needs (config, config, concept)."""
    by_label = {
        "curls": {"wrist": 1.0, "pocket": 0.1},
        "squats": {"wrist": 0.2, "pocket": 0.95},
        "pushups": {"wrist": 0.8, "pocket": 0.15},
        "walking": {"wrist": 0.3, "pocket": 0.9},
    }
    out = _concept_dependence(by_label)
    assert out["mean_correlation"] < 0.0
    assert out["inverting_pairs"] == 1


def test_concept_dependence_needs_enough_labels():
    assert _concept_dependence({"a": {"x": 0.5, "y": 0.4}})["mean_correlation"] is None


# --------------------------------------------------------------------------- gate consumption
def test_gate_tensor_reads_row_config_against_candidate():
    table = {"per_stream": {"mmfit/left_ear": {"labels": {"bicep_curls": 0.13, "squats": 0.7}}}}
    gate = gate_tensor(["mmfit/left_ear"], ["bicep_curls", "squats"], table)
    assert gate.shape == (1, 1, 2)
    assert abs(float(gate[0, 0, 0]) - 0.13) < 1e-6
    assert abs(float(gate[0, 0, 1]) - 0.70) < 1e-6


def test_unmeasured_pairs_get_the_neutral_default():
    """Absent evidence must not silently become a veto (0) or a licence (1)."""
    table = {"per_stream": {}}
    gate = gate_tensor(["unknown/stream"], ["novel_label"], table, default=0.5)
    assert float(gate[0, 0, 0]) == 0.5


def test_gate_tensor_broadcasts_over_query_sensors():
    table = {"per_stream": {"a/b": {"labels": {"x": 0.9}}}}
    gate = gate_tensor(["a/b"], ["x"], table, n_query_sensors=3)
    assert gate.shape == (3, 1, 1)
    assert torch.allclose(gate, torch.full((3, 1, 1), 0.9))
