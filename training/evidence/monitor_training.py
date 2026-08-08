"""CPU-only health monitor for Phase-B predictor and confidence training."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
SEVERITY = {"green": 0, "warning": 1, "critical": 2}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metric(snapshot: dict, name: str, field: str = "mean") -> float:
    try:
        return float(snapshot["metrics"][name][field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def assess(telemetry_dir: Path, stale_seconds: float = 150.0) -> dict:
    latest_path = Path(telemetry_dir) / "phase_b_telemetry_latest.json"
    snapshot = _load(latest_path)
    alerts: list[dict[str, str]] = []

    def alert(level: str, code: str, message: str) -> None:
        alerts.append({"severity": level, "code": code, "message": message})

    now = time.time()
    generated = float(snapshot.get("generated_unix_seconds", 0.0))
    age = now - generated if generated else float("inf")
    step = int(snapshot.get("step", 0))
    metadata = snapshot.get("metadata", {})
    planned = int(metadata.get("planned_steps", step or 0))
    final = snapshot.get("event") == "run_end" or (planned > 0 and step >= planned)
    stage = snapshot.get("stage", "unknown")

    if not snapshot:
        alert("critical", "no_telemetry", "No Phase-B telemetry snapshot is available.")
    elif age > stale_seconds and not final:
        alert("critical", "stale_heartbeat", f"Telemetry has not updated for {age:.0f} seconds.")
    if snapshot.get("nonfinite"):
        bad = ", ".join(f"{key}={count}" for key, count in snapshot["nonfinite"].items())
        alert("critical", "nonfinite_values", f"Non-finite values occurred: {bad}.")

    if stage == "predictor" and snapshot.get("metrics"):
        hard_error = _metric(snapshot, "hard_forward_max_abs_error", "max")
        if math.isfinite(hard_error) and hard_error > 1e-6:
            alert("critical", "hard_forward_mismatch",
                  f"Straight-through forward differs from hard inference by {hard_error:.3g}.")
        warmup = int(metadata.get("warmup_steps", 0))
        if step > warmup:
            for name in ("decoder_grad_norm", "retriever_grad_norm"):
                maximum = _metric(snapshot, name, "max")
                if math.isfinite(maximum) and maximum == 0.0:
                    alert("critical", "dead_gradient", f"{name} remained zero in the latest window.")
        clipped = _metric(snapshot, "gradient_clipped_fraction")
        if math.isfinite(clipped) and clipped > 0.8:
            alert("warning", "persistent_clipping",
                  f"{clipped:.0%} of recent steps exceeded the gradient clip threshold.")
        pool_share = _metric(snapshot, "pool_weight_max_share")
        if math.isfinite(pool_share) and pool_share > 0.8:
            alert("warning", "pool_collapse",
                  f"One evidence item receives {pool_share:.0%} mean maximum pooling mass.")
        row_share = _metric(snapshot, "selected_row_max_share")
        if math.isfinite(row_share) and row_share > 0.25:
            alert("warning", "memory_row_collapse",
                  f"One memory row supplies {row_share:.0%} of recent selections.")
        hard_soft_cos = _metric(snapshot, "hard_soft_retriever_grad_cosine")
        if math.isfinite(hard_soft_cos) and hard_soft_cos < -0.25:
            alert("warning", "surrogate_gradient_conflict",
                  f"Hard and soft retriever gradients have cosine {hard_soft_cos:.2f}.")
        validation = snapshot.get("validation", {})
        gain = validation.get("adaptation_macro_cell_ba_gain")
        if gain is not None and float(gain) < -0.05:
            alert("warning", "worse_than_identity",
                  f"Held-out adaptation score is {float(gain):.3f} below identity control.")

    if stage == "confidence" and snapshot.get("metrics"):
        positive_rate = _metric(snapshot, "target_positive_rate")
        if math.isfinite(positive_rate) and not 0.02 <= positive_rate <= 0.98:
            alert("warning", "confidence_target_imbalance",
                  f"Recent confidence-target positive rate is {positive_rate:.1%}.")
        score_mean = _metric(snapshot, "predicted_confidence_mean")
        score_std = _metric(snapshot, "predicted_confidence_std")
        if math.isfinite(score_mean) and math.isfinite(score_std) \
                and score_std < 0.005 and (score_mean < 0.05 or score_mean > 0.95):
            alert("warning", "confidence_saturation",
                  f"Confidence outputs are nearly constant at {score_mean:.3f}.")

    status = "green"
    for item in alerts:
        if SEVERITY[item["severity"]] > SEVERITY[status]:
            status = item["severity"]
    elapsed = float(snapshot.get("elapsed_seconds", 0.0))
    rate = step / elapsed if step > 0 and elapsed > 0 else float("nan")
    eta = (planned - step) / rate / 60.0 if rate > 0 and planned >= step else float("nan")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "stage": stage,
        "run_id": snapshot.get("run_id"),
        "step": step,
        "planned_steps": planned,
        "progress": step / planned if planned else 0.0,
        "heartbeat_age_s": age,
        "steps_per_s": rate,
        "eta_minutes": eta,
        "alerts": alerts,
        "metrics": snapshot.get("metrics", {}),
        "mix": snapshot.get("mix", {}),
        "strata": snapshot.get("strata", {}),
        "validation": snapshot.get("validation", {}),
        "telemetry": str(latest_path),
    }


def render_text(report: dict) -> str:
    rate = report["steps_per_s"]
    eta = report["eta_minutes"]
    lines = [
        f"STATUS: {report['status'].upper()} ({report['stage']})",
        f"run: {report.get('run_id')}",
        f"step: {report['step']:,}/{report['planned_steps']:,} ({report['progress']:.1%})",
        f"heartbeat: {report['heartbeat_age_s']:.0f}s ago",
        f"throughput: {rate:.2f} step/s; ETA: {eta:.1f} min",
    ]
    if report["alerts"]:
        lines.append("alerts:")
        lines.extend(
            f"- [{item['severity']}] {item['code']}: {item['message']}"
            for item in report["alerts"]
        )
    else:
        lines.append("alerts: none")
    return "\n".join(lines) + "\n"


def write_report(telemetry_dir: Path) -> dict:
    report = assess(telemetry_dir)
    _atomic_text(Path(telemetry_dir) / "health.json", json.dumps(report, indent=2) + "\n")
    text = render_text(report)
    _atomic_text(Path(telemetry_dir) / "health.txt", text)
    print(text, end="", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0)
    parser.add_argument("--stale-seconds", type=float, default=150.0)
    args = parser.parse_args()
    while True:
        report = assess(args.telemetry_dir, stale_seconds=args.stale_seconds)
        _atomic_text(args.telemetry_dir / "health.json", json.dumps(report, indent=2) + "\n")
        text = render_text(report)
        _atomic_text(args.telemetry_dir / "health.txt", text)
        print(text, end="", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
