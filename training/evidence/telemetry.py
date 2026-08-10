"""Low-overhead, atomic Phase-B health telemetry."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "last": float(values[-1]),
    }


class PhaseBTelemetry:
    """Accumulate short windows of scalar health metrics and publish atomic snapshots.

    A run-specific history file prevents resumed/repeated experiments from being mixed. ``latest``
    is intentionally shared inside the run directory so a monitor has one stable path to poll; the
    explicit startup record replaces stale output before the first minute of training elapses.
    """

    def __init__(
        self,
        output_dir: Path,
        interval_seconds: float = 60.0,
        *,
        stage: str = "predictor",
        run_id: str | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.stage = str(stage)
        self.jsonl = self.output_dir / f"phase_b_telemetry_{self.run_id}.jsonl"
        self.latest = self.output_dir / "phase_b_telemetry_latest.json"
        self.interval_seconds = float(interval_seconds)
        self.started_unix_seconds = time.time()
        self.last_emit = time.monotonic()
        self.numeric: dict[str, list[float]] = defaultdict(list)
        self.categories: dict[str, Counter] = defaultdict(Counter)
        self.strata: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.nonfinite: Counter = Counter()
        self.total_nonfinite: Counter = Counter()
        self.last_validation: dict = {}
        self.metadata: dict = {}
        self.sequence = 0
        self.started = False
        self.last_metrics: dict = {}
        self.last_mix: dict = {}
        self.last_strata: dict = {}
        self.last_nonfinite: dict = {}

    def due(self) -> bool:
        return time.monotonic() - self.last_emit >= self.interval_seconds

    def start(
        self,
        *,
        step: int,
        elapsed_seconds: float,
        metadata: dict | None = None,
    ) -> dict:
        """Publish a startup heartbeat immediately, replacing any stale ``latest`` file."""
        self.metadata = dict(metadata or {})
        self.started = True
        return self._publish(
            step=step,
            elapsed_seconds=elapsed_seconds,
            event="run_start",
            window_seconds=0.0,
            metrics={},
            mix={},
            strata={},
            nonfinite={},
        )

    def update(
        self,
        metrics: dict,
        *,
        categories: dict | None = None,
        strata: dict | None = None,
    ) -> None:
        finite: dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            number = float(value)
            if np.isfinite(number):
                self.numeric[key].append(number)
                finite[key] = number
            else:
                self.nonfinite[key] += 1
                self.total_nonfinite[key] += 1
        for key, value in (categories or {}).items():
            values = value if isinstance(value, (list, tuple)) else (value,)
            self.categories[key].update(str(item) for item in values)
        # Each declared dimension receives the same finite scalar set. This permits cheap comparisons
        # such as loss-by-candidate-count or support-recall-by-episode-type without a combinatorial
        # cross-product of every curriculum field.
        for dimension, category in (strata or {}).items():
            bucket = self.strata[str(dimension)][str(category)]
            for key, value in finite.items():
                bucket[key].append(value)

    def set_validation(self, metrics: dict) -> None:
        def finite_tree(value):
            if isinstance(value, dict):
                return {key: clean for key, item in value.items()
                        if (clean := finite_tree(item)) is not None}
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if np.isfinite(number) else None

        self.last_validation = finite_tree(metrics) or {}

    def _publish(
        self,
        *,
        step: int,
        elapsed_seconds: float,
        event: str,
        window_seconds: float,
        metrics: dict,
        mix: dict,
        strata: dict,
        nonfinite: dict,
    ) -> dict:
        self.sequence += 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event,
            "sequence": self.sequence,
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "window_seconds": float(window_seconds),
            "metrics": metrics,
            "mix": mix,
            "strata": strata,
            "nonfinite": nonfinite,
            "validation": self.last_validation,
            "metadata": self.metadata,
            "history_file": str(self.jsonl),
            "started_unix_seconds": self.started_unix_seconds,
            "generated_unix_seconds": time.time(),
        }
        line = json.dumps(payload, sort_keys=True)
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        temporary = self.latest.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.latest)
        return payload

    def emit(
        self,
        *,
        step: int,
        elapsed_seconds: float,
        force: bool = False,
        final: bool = False,
    ) -> dict | None:
        if not self.started:
            self.start(step=step, elapsed_seconds=elapsed_seconds)
        if not force and not self.due():
            return None
        if not self.numeric and not self.categories and not self.nonfinite \
                and not self.last_validation and not final:
            return None
        window_seconds = time.monotonic() - self.last_emit
        metrics = {
            key: _summary(values) for key, values in sorted(self.numeric.items()) if values
        }
        mix = {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(self.categories.items())
        }
        strata = {
            dimension: {
                category: {
                    key: _summary(values)
                    for key, values in sorted(metric_values.items()) if values
                }
                for category, metric_values in sorted(categories.items())
            }
            for dimension, categories in sorted(self.strata.items())
        }
        nonfinite = dict(sorted(self.total_nonfinite.items()))
        if final and not metrics and not mix and not self.nonfinite:
            metrics, mix, strata = self.last_metrics, self.last_mix, self.last_strata
        payload = self._publish(
            step=step,
            elapsed_seconds=elapsed_seconds,
            event="run_end" if final else "heartbeat",
            window_seconds=window_seconds,
            metrics=metrics,
            mix=mix,
            strata=strata,
            nonfinite=nonfinite,
        )
        self.last_metrics, self.last_mix = metrics, mix
        self.last_strata, self.last_nonfinite = strata, nonfinite
        self.numeric.clear()
        self.categories.clear()
        self.strata.clear()
        self.nonfinite.clear()
        self.last_emit = time.monotonic()
        return payload
