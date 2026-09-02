"""Short fit/gradient/evaluation smoke matrix for released frozen encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from applications.motion_monitoring.baseline_encoder import (
    OPTIONAL_APPLICATION_BASELINES,
    PRIMARY_APPLICATION_BASELINES,
    BaselineMotionEncoder,
)
from applications.motion_monitoring.smoke import (
    _task1_smoke,
    _task2_smoke,
    _task3_smoke,
)


_TASKS = {
    "task1": _task1_smoke,
    "task2": _task2_smoke,
    "task3": _task3_smoke,
}


def run(
    baselines: list[str],
    tasks: list[str],
    *,
    steps: int,
    device: str,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for name in baselines:
        started = perf_counter()
        encoder = BaselineMotionEncoder(name, device=device)
        task_results: dict[str, object] = {}
        for task in tasks:
            task_started = perf_counter()
            task_results[task] = {
                **_TASKS[task](encoder, steps=steps, train_encoder=False),
                "elapsed_sec": perf_counter() - task_started,
            }
        results[name] = {
            "status": "mechanical_smoke_only",
            "encoder": encoder.provenance(),
            "tasks": task_results,
            "elapsed_sec": perf_counter() - started,
        }
    return {
        "steps": steps,
        "device": device,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(PRIMARY_APPLICATION_BASELINES),
        choices=(*PRIMARY_APPLICATION_BASELINES, *OPTIONAL_APPLICATION_BASELINES),
    )
    parser.add_argument(
        "--tasks", nargs="+", default=list(_TASKS), choices=tuple(_TASKS)
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    payload = run(args.baselines, args.tasks, steps=args.steps, device=args.device)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
