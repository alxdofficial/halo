"""Render a compact Phase-B telemetry dashboard from one run-specific JSONL history."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") != "run_start":
            rows.append(value)
    return rows


def _series(rows, key, field="mean"):
    points = []
    for row in rows:
        value = row.get("metrics", {}).get(key, {}).get(field)
        if value is not None:
            points.append((row["step"], value))
    return points


def _validation_series(rows, key):
    points, previous = [], object()
    for row in rows:
        value = row.get("validation", {}).get(key)
        if value is not None and value != previous:
            points.append((row["step"], value)); previous = value
    return points


def _line(axis, points, label, **kwargs):
    if points:
        x, y = zip(*points)
        axis.plot(x, y, label=label, **kwargs)


def render(telemetry_dir: Path, output: Path | None = None) -> Path:
    telemetry_dir = Path(telemetry_dir)
    latest = json.loads((telemetry_dir / "phase_b_telemetry_latest.json").read_text())
    rows = _load(Path(latest["history_file"]))
    output = output or telemetry_dir / "telemetry.png"
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    stage = latest.get("stage")
    if stage == "predictor":
        _line(axes[0, 0], _series(rows, "loss_over_random"), "CE / random CE")
        _line(axes[0, 0], _series(rows, "chance_normalized_train_accuracy"), "chance-normalized acc")
        axes[0, 0].set_title("Training objective")
        for key, label in (
            ("decoder_grad_norm", "decoder"), ("retriever_grad_norm", "retriever"),
            ("tokenizer_grad_norm", "tokenizer"), ("preclip_grad_norm", "total pre-clip"),
        ):
            _line(axes[0, 1], _series(rows, key), label)
        axes[0, 1].set_yscale("symlog", linthresh=1e-4); axes[0, 1].set_title("Gradient norms")
        for key, label in (
            ("provided_support_recall_at_k", "provided-support recall"),
            ("provided_support_pool_mass", "support pool mass"),
            ("pool_weight_max_share", "max pool share"),
            ("topk_retained_soft_mass", "retained soft mass"),
        ):
            _line(axes[0, 2], _series(rows, key), label)
        axes[0, 2].set_title("Evidence use")
        for key, label in (
            ("hard_soft_retriever_grad_cosine", "hard/soft grad cosine"),
            ("effective_soft_to_hard_retriever_grad_ratio", "effective soft/hard ratio"),
            ("retrieval_normalized_entropy", "soft retrieval entropy"),
            ("pool_normalized_entropy", "pool entropy"),
        ):
            _line(axes[1, 0], _series(rows, key), label)
        axes[1, 0].set_title("Retrieval health")
        for key, label in (
            ("macro_cell_ba", "held-out"), ("identity_macro_cell_ba", "identity"),
            ("train_macro_cell_ba", "matched train"),
            ("random_alias_ba", "random alias"),
        ):
            _line(axes[1, 1], _validation_series(rows, key), label, marker="o")
        axes[1, 1].set_title("Fixed canaries")
    else:
        _line(axes[0, 0], _series(rows, "loss"), "train BCE")
        _line(axes[0, 0], _validation_series(rows, "bce"), "validation BCE", marker="o")
        axes[0, 0].set_title("Confidence loss")
        _line(axes[0, 1], _series(rows, "grad_norm"), "head")
        axes[0, 1].set_title("Gradient norm")
        for key, label in (
            ("target_positive_rate", "target positive"),
            ("predicted_confidence_mean", "mean confidence"),
            ("predicted_confidence_std", "confidence std"),
        ):
            _line(axes[0, 2], _series(rows, key), label)
        axes[0, 2].set_title("Output distribution")
        for key, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("ece", "ECE")):
            _line(axes[1, 0], _validation_series(rows, key), label, marker="o")
        axes[1, 0].set_title("Calibration validation")
        for key, label in (
            ("mean_confidence_positive", "positive"),
            ("mean_confidence_negative", "negative"),
            ("positive_rate", "base rate"),
        ):
            _line(axes[1, 1], _validation_series(rows, key), label, marker="o")
        axes[1, 1].set_title("Confidence separation")
    for key, label in (
        ("step_seconds", "step seconds"),
        ("gpu_allocated_gib", "allocated GiB"),
        ("gpu_reserved_gib", "reserved GiB"),
    ):
        _line(axes[1, 2], _series(rows, key), label)
    axes[1, 2].set_title("Performance")
    for axis in axes.flat:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.suptitle(f"Phase B {stage} · {latest.get('run_id')}", fontsize=11)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    temporary.replace(output)
    return output
