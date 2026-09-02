"""Assemble protocol-compatible application evaluation JSON files into Markdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from applications.motion_monitoring.evaluation import (
    ApplicationEvaluation,
    render_markdown,
)


def render_grouped(results: list[ApplicationEvaluation]) -> str:
    groups = defaultdict(list)
    for result in results:
        groups[
            (
                result.task,
                result.dataset,
                str(result.protocol.get("readout", "unspecified")),
                result.protocol_fingerprint,
            )
        ].append(result)
    lines = [
        "# Complete application evaluation",
        "",
        "Status: complete deterministic evaluation, but provisional for publication. Subject-level",
        "confidence intervals and raw-signal or physical-feature controls are not yet included.",
        "",
        "All thresholds were selected once on development data. Each table reports the",
        "complete sealed test split for one dataset and one readout; results are not pooled",
        "across datasets.",
        "",
        "For Task 3, B-cubed scores are conditional on matched predicted events. They must be",
        "interpreted together with occurrence precision, occurrence recall, and false occurrences",
        "per hour; they are not an end-to-end recurrence-discovery score on their own.",
    ]
    for (task, dataset, readout, _), rows in sorted(groups.items()):
        lines.extend(
            [
                "",
                f"## {task.upper()} - {dataset} - {readout}",
                "",
                render_markdown(rows).rstrip(),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    results = [ApplicationEvaluation.from_json(path) for path in args.results]
    rendered = render_grouped(results) if args.grouped else render_markdown(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
