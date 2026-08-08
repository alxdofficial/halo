"""Low-overhead, atomic Phase-B health telemetry."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class PhaseBTelemetry:
    def __init__(self, output_dir: Path, interval_seconds: float = 60.0):
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.output_dir / "phase_b_telemetry.jsonl"
        self.latest = self.output_dir / "phase_b_telemetry_latest.json"
        self.interval_seconds = float(interval_seconds)
        self.last_emit = time.monotonic()
        self.numeric: dict[str, list[float]] = defaultdict(list)
        self.categories: dict[str, Counter] = defaultdict(Counter)
        self.last_validation: dict[str, float] = {}

    def due(self) -> bool:
        return time.monotonic() - self.last_emit >= self.interval_seconds

    def update(self, metrics: dict, *, categories: dict | None = None) -> None:
        for key, value in metrics.items():
            if value is not None and np.isfinite(float(value)):
                self.numeric[key].append(float(value))
        for key, value in (categories or {}).items():
            values = value if isinstance(value, (list, tuple)) else (value,)
            self.categories[key].update(str(item) for item in values)

    def set_validation(self, metrics: dict[str, float]) -> None:
        self.last_validation = {
            key: float(value) for key, value in metrics.items()
            if np.isfinite(float(value))
        }

    def emit(self, *, step: int, elapsed_seconds: float, force: bool = False) -> dict | None:
        if not force and not self.due():
            return None
        if not self.numeric and not self.categories and self.latest.exists():
            return None
        payload = {
            "step": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "window_seconds": float(time.monotonic() - self.last_emit),
            "metrics": {
                key: {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": float(values[-1]),
                }
                for key, values in sorted(self.numeric.items()) if values
            },
            "mix": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(self.categories.items())
            },
            "validation": dict(sorted(self.last_validation.items())),
            "generated_unix_seconds": time.time(),
        }
        line = json.dumps(payload, sort_keys=True)
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        temporary = self.latest.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.latest)
        self.numeric.clear()
        self.categories.clear()
        self.last_emit = time.monotonic()
        return payload
