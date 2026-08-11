"""Render the three planned RESULTS.md artifacts (T1/T2/T3) plus the v20 training diagnostic.

These are the figures `docs/results/RESULTS.md` sections 3-5 specify, drawn from committed result
JSON rather than from numbers retyped into a doc. Every panel obeys the section 8 reporting rules:

  * `k` is the number of enrolled WINDOWS per candidate, one per source recording.
  * Cross-subject and same-subject enrollment are never pooled onto one line.
  * Every adaptation series is drawn against its two canaries — support-removed and
    support-label-shuffled. A series that does not separate from its canaries is not reading its
    enrollment, whatever its absolute score.
  * Cells are aggregated QUERY-WEIGHTED, never as an unweighted mean over datasets: RealWorld
    contributes a few percent of queries with a 2-way decision and would otherwise carry a third of
    a three-dataset mean.

Run:
    python -m training.evidence.plot_results_artifacts \
        --enrollment trained=<path> closed_form=<path> \
        --train-log training/evidence/outputs/phase_b_v20_20260811/train.log \
        --out-dir docs/results/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Series drawn on the adaptation panels. The canaries are deliberately styled as dashed grey so a
#: series that sits on top of them reads as a failure at a glance rather than as another method.
SERIES = [
    ("f1_macro", "HALO memory (this arm)", "#1f77b4", "-", 2.2),
    ("identity_f1_macro", "closed-form retrieval vote", "#d62728", "-", 2.0),
    ("prototype_f1_macro", "prototype (closed form)", "#2ca02c", "-", 1.8),
    ("ridge_head_f1_macro", "ridge head (fitted)", "#ff7f0e", "-", 1.8),
    ("support_removed_f1_macro", "canary: support removed", "#8c8c8c", "--", 1.3),
    ("support_label_shuffled_f1_macro", "canary: labels shuffled", "#bfbfbf", ":", 1.3),
]


def load_cells(path: Path) -> dict:
    return json.loads(path.read_text()).get("results", {})


def curve(cells: dict, field: str, *, subject_relation: str, shape: str = "full") -> dict:
    """Query-weighted mean of `field` per support count k, over matching cells."""
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key, cell in cells.items():
        if cell.get("subject_relation") != subject_relation:
            continue
        if cell.get("configuration_relation") != "same_configuration":
            continue
        if cell.get("enrollment_shape") not in (shape, "zero"):
            continue
        value, queries = cell.get(field), cell.get("queries")
        if value is None or not queries:
            continue
        k = int(key.rsplit("/k", 1)[1]) if "/k" in key else 0
        totals[k][0] += float(value) * int(queries)
        totals[k][1] += int(queries)
    return {k: num / den for k, (num, den) in sorted(totals.items()) if den}


def plot_t3(arms: dict[str, dict], out: Path) -> Path:
    """T3 — label efficiency. x = k, y = macro-F1, one line per (encoder, mechanism)."""
    fig, axes = plt.subplots(1, len(arms), figsize=(6.4 * len(arms), 5.0), sharey=True,
                             squeeze=False)
    for ax, (arm, cells) in zip(axes[0], arms.items()):
        for field, label, colour, style, width in SERIES:
            points = curve(cells, field, subject_relation="cross_subject")
            if not points:
                continue
            # In the closed_form arm the readout IS the retrieval vote, so those two series
            # coincide exactly. Draw the arm's own readout wider and semi-transparent underneath
            # so the overlap reads as "identical by construction" rather than "line missing".
            arm_readout = field == "f1_macro"
            ax.plot(list(points), list(points.values()), style, color=colour,
                    linewidth=width + 2.5 if arm_readout else width,
                    alpha=0.45 if arm_readout else 1.0, zorder=1 if arm_readout else 2,
                    marker="o" if style == "-" else None, markersize=4, label=label)
        ax.set_title(f"{arm} — cross-subject enrollment", fontsize=11)
        ax.set_xlabel("enrolled windows per candidate (k)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xticks(sorted({k for c in arms.values() for k in curve(c, "f1_macro",
                                                                     subject_relation="cross_subject")}))
    axes[0][0].set_ylabel("macro-F1")
    axes[0][-1].legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.suptitle("T3 — label efficiency: adaptation per enrolled example (dev roster, "
                 "query-weighted)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_t2(arms: dict[str, dict], out: Path, *, at_k: int = 8) -> Path:
    """T2 — mechanism comparison at one budget, each mechanism beside its canaries."""
    labels, groups = [], []
    for arm, cells in arms.items():
        values = []
        for field, label, *_ in SERIES:
            points = curve(cells, field, subject_relation="cross_subject")
            values.append(points.get(at_k, float("nan")))
        groups.append(values)
        labels.append(arm)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.8 / max(len(groups), 1)
    x = np.arange(len(SERIES))
    for index, (arm, values) in enumerate(zip(labels, groups)):
        ax.bar(x + index * width, values, width, label=arm,
               color=[s[2] for s in SERIES], alpha=0.55 + 0.45 * index,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([s[1].replace(" (", "\n(") for s in SERIES], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_title(f"T2 — adaptation mechanisms at k={at_k}, cross-subject "
                 f"(bars beyond the two grey canaries are reading their enrollment)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training(train_log: Path, out: Path) -> Path:
    """The v20 story: enrollment dependence appearing, against the control it had to catch."""
    rows = [json.loads(line) for line in train_log.read_text().splitlines()
            if line.startswith('{"step"')]
    steps = [r["step"] for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    left.plot(steps, [r.get("support_removal_true_probability_drop", 0) for r in rows],
              color="#1f77b4", linewidth=2, label="removing support")
    left.plot(steps, [r.get("support_label_shuffle_true_probability_drop", 0) for r in rows],
              color="#2ca02c", linewidth=2, label="deranging support labels")
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xlabel("optimizer step")
    left.set_ylabel("drop in true-label probability")
    left.set_title("Enrollment dependence — v19 sat at 0.000 for its whole run", fontsize=10)
    left.legend(fontsize=9)
    left.grid(alpha=0.25, linewidth=0.6)

    right.plot(steps, [r.get("selection_low_k_ba", 0) for r in rows], color="#1f77b4",
               linewidth=2, label="trained relational decoder")
    right.plot(steps, [r.get("selection_low_k_identity_ba", 0) for r in rows], color="#d62728",
               linewidth=2, label="closed-form retrieval vote (control)")
    right.set_xlabel("optimizer step")
    right.set_ylabel("balanced accuracy, held-family k=1/2 canaries")
    right.set_title("Eligibility gate — the decoder must match the control to ship", fontsize=10)
    right.legend(fontsize=9)
    right.grid(alpha=0.25, linewidth=0.6)

    for axis in (left, right):
        axis.axvline(400, color="#888888", linestyle="--", linewidth=1)
        axis.annotate("retriever unfreezes", xy=(400, axis.get_ylim()[1]),
                      xytext=(6, -12), textcoords="offset points", fontsize=8, color="#555555")

    fig.suptitle("Phase-B v20 training diagnostic", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_t1(table: Path, out: Path) -> Path | None:
    """T1 — zero-shot mean macro-F1 per model, read from the assembled protocol table."""
    if not table.exists():
        return None
    rows = []
    for line in table.read_text().splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        try:
            rows.append((parts[0], float(parts[1])))
        except ValueError:
            continue
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.6))
    colours = ["#1f77b4" if "halo" in name.lower() else "#9e9e9e" for name, _ in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colours, edgecolor="black",
            linewidth=0.5)
    for index, (_, value) in enumerate(rows):
        ax.text(value + 0.4, index, f"{value:.1f}", va="center", fontsize=9)
    ax.set_xlabel("mean macro-F1 over the held-out datasets")
    ax.set_title("T1 — zero-shot transfer across models (HALO rows in blue)", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrollment", nargs="+", required=True,
                        help="NAME=PATH pairs, one per arm")
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--protocol-table", type=Path,
                        default=REPO / "training/diagnostics/outputs/current_protocol_table.md")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs/results/figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = {}
    for pair in args.enrollment:
        name, _, path = pair.partition("=")
        arms[name] = load_cells(Path(path))
        print(f"{name}: {len(arms[name])} cells from {path}")

    written = [
        plot_t3(arms, args.out_dir / "t3_label_efficiency.png"),
        plot_t2(arms, args.out_dir / "t2_adaptation_mechanisms.png"),
    ]
    if args.train_log and args.train_log.exists():
        written.append(plot_training(args.train_log, args.out_dir / "phase_b_v20_training.png"))
    t1 = plot_t1(args.protocol_table, args.out_dir / "t1_zero_shot.png")
    if t1:
        written.append(t1)

    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
