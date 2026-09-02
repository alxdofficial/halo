"""Render the workflow, scenario-icon, and dataset-mix diagrams as PNG figures.

Two independent color axes, used consistently across every figure in this deck:
  - pipeline ROLE (frozen encoder / trained head / task-specific decoder / output)
  - task IDENTITY (Task 1 Detect / Task 2 Compare / Task 3 Discover), a validated
    three-hue categorical set reused from the project's prior dataviz work.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path("comms/2026-08-31-pivot-brief/figures")
ROOT.mkdir(parents=True, exist_ok=True)

INK, MUTED, BLUE, GREEN = "#1A1A1A", "#666666", "#34618E", "#3F6B4A"
CREAM, CARD, LINE = "#FBF3DF", "#F4F6F9", "#CCCCCC"
GREEN_TINT = "#E7F0EA"
TASK = {1: "#2A64A8", 2: "#7B4B94", 3: "#B5651D"}  # detect / compare / discover
TASK_TINT = {1: "#EAF0F7", 2: "#F1EAF4", 3: "#F8ECE0"}

plt.rcParams["font.family"] = "DejaVu Sans"


def _fig(w, h, dpi=220):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rounded_box(ax, x, y, w, h, *, fill, edge, lw=1.8, rounding=0.10):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    return box


def pill(ax, cx, cy, text, *, color):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.3, fontweight="bold",
             color="white", bbox=dict(boxstyle="round,pad=0.32", fc=color, ec="none"))


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=2.2, rad=0.0, style="-|>", alpha=1.0):
    patch = mpatches.FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
        color=color, linewidth=lw, alpha=alpha, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


# ---------------------------------------------------------------- icons
def icon_frozen(ax, cx, cy, r, color, lw=2.4):
    for k in range(3):
        a = math.radians(60 * k)
        dx, dy = r * math.cos(a), r * math.sin(a)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=lw, solid_capstyle="round")
        for sign in (-1, 1):
            tx, ty = cx + sign * dx * 0.62, cy + sign * dy * 0.62
            perp = math.radians(60 * k + 90)
            ex, ey = 0.28 * r * math.cos(perp), 0.28 * r * math.sin(perp)
            ax.plot([tx - ex, tx + ex], [ty - ey, ty + ey], color=color, lw=lw * 0.8, solid_capstyle="round")


def icon_trained(ax, cx, cy, r, color, lw=2.4):
    fracs = (0.30, 0.62, 0.44)
    for i, frac in enumerate(fracs):
        y = cy + r * (0.62 - i * 0.62)
        ax.plot([cx - r, cx + r], [y, y], color=LINE, lw=lw * 0.9, solid_capstyle="round")
        kx = cx - r + 2 * r * frac
        ax.add_patch(plt.Circle((kx, y), r * 0.16, fc=color, ec="none", zorder=3))


def icon_output(ax, cx, cy, r, color, lw=2.6):
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=color, lw=lw))
    ax.plot([cx - r * 0.45, cx - r * 0.08, cx + r * 0.55],
            [cy - r * 0.02, cy - r * 0.42, cy + r * 0.35],
            color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")


def icon_detect(ax, cx, cy, r, color, lw=2.4):
    gx, gy = cx - r * 0.18, cy + r * 0.18
    ax.add_patch(plt.Circle((gx, gy), r * 0.58, fill=False, ec=color, lw=lw))
    a = math.radians(45)
    hx, hy = gx + r * 0.58 * math.cos(-a), gy + r * 0.58 * math.sin(-a)
    ex, ey = gx + r * 1.15 * math.cos(-a), gy + r * 1.15 * math.sin(-a)
    ax.plot([hx, ex], [hy, ey], color=color, lw=lw * 1.3, solid_capstyle="round")


def icon_compare(ax, cx, cy, r, color, lw=2.2):
    xs = np.linspace(cx - r, cx + r, 60)
    y1 = cy + r * 0.28 + r * 0.22 * np.sin((xs - cx) * 3.0)
    y2 = cy - r * 0.28 + r * 0.22 * np.sin((xs - cx) * 3.0 + 0.9)
    ax.plot(xs, y1, color=color, lw=lw, solid_capstyle="round")
    ax.plot(xs, y2, color=color, lw=lw, ls=(0, (3, 2)), solid_capstyle="round")
    arrow(ax, cx, cy + r * 0.10, cx, cy - r * 0.02, color=color, lw=lw * 0.8, style="-", alpha=0.0)


def icon_discover(ax, cx, cy, r, color, lw=2.0):
    pts = {
        "a": (cx - r * 0.55, cy + r * 0.10),
        "b": (cx - r * 0.05, cy + r * 0.55),
        "c": (cx - r * 0.15, cy - r * 0.35),
        "d": (cx + r * 0.60, cy + r * 0.30),
        "e": (cx + r * 0.45, cy - r * 0.45),
    }
    edges = [("a", "b"), ("b", "c"), ("a", "c")]
    for u, v in edges:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=color, lw=lw, alpha=0.85)
    for i, (name, (x, y)) in enumerate(pts.items()):
        clustered = name in ("a", "b", "c")
        ax.add_patch(plt.Circle((x, y), r * 0.13, fc=color if clustered else LINE,
                                  ec=color, lw=1.3, zorder=3))


TASK_ICON = {1: icon_detect, 2: icon_compare, 3: icon_discover}
TASK_LABEL = {1: "Task 1 · Detect", 2: "Task 2 · Compare", 3: "Task 3 · Discover"}


def icon_badge(path, icon_fn, color, size=1.15):
    fig, ax = _fig(size, size)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    icon_fn(ax, size / 2, size / 2, size * 0.36, color)
    fig.savefig(path, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- workflow pipeline
def workflow(path, task_id, stages):
    """stages: list of (role, header, body_lines, icon_fn)."""
    w, h = 12.13, 3.35
    fig, ax = _fig(w, h)
    n = len(stages)
    arrow_w, gap = 0.46, 0.16
    box_w = (w - (n - 1) * (arrow_w + 2 * gap)) / n
    box_h, box_y = 2.55, 0.42
    task_color = TASK[task_id]

    x = 0.0
    for i, (role, header, body, icon_fn) in enumerate(stages):
        if role == "frozen":
            fill, edge, icon_color = CARD, BLUE, BLUE
        elif role == "trained":
            fill, edge, icon_color = GREEN_TINT, GREEN, GREEN
        elif role == "taskhead":
            fill, edge, icon_color = TASK_TINT[task_id], task_color, task_color
        else:
            fill, edge, icon_color = CREAM, INK, INK
        rounded_box(ax, x, box_y, box_w, box_h, fill=fill, edge=edge, lw=2.0)

        cx = x + box_w / 2
        icon_cy = box_y + box_h - 0.62
        icon_fn(ax, cx, icon_cy, 0.30, icon_color)
        if role == "trained":
            pill(ax, x + box_w - 0.52, box_y + box_h - 0.24, "TRAINED", color=GREEN)
        elif role == "frozen":
            pill(ax, x + box_w - 0.48, box_y + box_h - 0.24, "FIXED", color=BLUE)

        ax.text(cx, box_y + box_h - 1.02, header, ha="center", va="top",
                 fontsize=11.3, fontweight="bold", color=INK)
        body_text = "\n".join(body)
        ax.text(cx, box_y + box_h - 1.34, body_text, ha="center", va="top",
                 fontsize=9.1, color=INK, linespacing=1.55)

        x += box_w
        if i < n - 1:
            x += gap
            arrow(ax, x, box_y + box_h / 2, x + arrow_w, box_y + box_h / 2, color=MUTED, lw=2.4)
            x += arrow_w + gap

    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- dataset mix (bowtie)
# (dataset, task_id, tier) — tier from the registered role / documented primary-use text
# in ANNOTATION_INVENTORY.md, APPLICATION_DATASETS.md section 5-6, and SOURCE_INVENTORY.json.
# "primary": a full training or primary-evaluation role. "secondary": an explicitly weaker
# role (control / weak supervision / transfer / reconstruct-pending).
TRAIN_EDGES = [
    ("OpenPack", 1, "primary"), ("OpenPack", 3, "primary"),
    ("CrossFit", 1, "primary"), ("CrossFit", 3, "primary"),
    ("AIDLAB-HAR", 1, "secondary"), ("AIDLAB-HAR", 3, "secondary"),
    ("RecoFit", 1, "primary"), ("RecoFit", 3, "secondary"),
    ("HARMES", 1, "secondary"), ("HARMES", 3, "secondary"),
]
EVAL_EDGES = [
    ("C-MHAD", 1, "primary"), ("C-MHAD", 3, "secondary"),
    ("WEAR", 1, "primary"), ("WEAR", 3, "secondary"),
    ("MoniPar", 1, "primary"),
    ("OCA", 3, "primary"), ("OCA", 1, "secondary"),
]


def dataset_mix(path):
    w, h = 12.13, 4.85
    fig, ax = _fig(w, h)

    train = ["OpenPack", "CrossFit", "AIDLAB-HAR", "RecoFit", "HARMES"]
    ev = ["C-MHAD", "WEAR", "MoniPar", "OCA"]

    tbw, tbh = 2.35, 0.56
    tx = 0.35
    tys = np.linspace(h - 0.95, 0.65, len(train))
    train_pos = dict(zip(train, tys))

    ebw, ebh = 2.1, 0.56
    ex_ = w - 0.35 - ebw
    eys = np.linspace(h - 1.05, 0.70, len(ev))
    eval_pos = dict(zip(ev, eys))

    kbw, kbh = 2.05, 0.92
    kx = w / 2 - kbw / 2
    kys = np.linspace(h - 1.15, 0.75, 3)
    task_pos = dict(zip((1, 2, 3), kys))

    # COHORT_V1.json (the frozen application manifest) currently indexes 7 of these 9
    # sources; MoniPar and HARMES still use the older representation-corpus storage and
    # are not yet in the reviewed RawRecording cache contract (manifests/README.md).
    PENDING = {"MoniPar", "HARMES"}

    def source_box(x, y, bw, bh, name):
        pending = name in PENDING
        edge = MUTED if pending else LINE
        rounded_box(ax, x, y, bw, bh, fill=CARD, edge=edge,
                    lw=1.3, rounding=0.10)
        if pending:
            box = mpatches.FancyBboxPatch(
                (x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.10",
                linewidth=1.3, edgecolor=MUTED, facecolor="none", linestyle=(0, (3, 2)),
            )
            ax.add_patch(box)
            ax.text(x + bw / 2, y + bh * 0.64, name, ha="center", va="center",
                     fontsize=9.3, color=INK)
            ax.text(x + bw / 2, y + bh * 0.24, "pending cohort integration",
                     ha="center", va="center", fontsize=6.2, style="italic", color=MUTED)
        else:
            ax.text(x + bw / 2, y + bh / 2, name, ha="center", va="center",
                     fontsize=9.6, color=INK)

    for name, y in train_pos.items():
        source_box(tx, y, tbw, tbh, name)
    for name, y in eval_pos.items():
        source_box(ex_, y, ebw, ebh, name)
    for tid, y in task_pos.items():
        if tid == 2:
            rounded_box(ax, kx, y, kbw, kbh, fill="white", edge=TASK[tid], lw=1.8, rounding=0.10)
            box = mpatches.FancyBboxPatch(
                (kx, y), kbw, kbh, boxstyle="round,pad=0,rounding_size=0.10",
                linewidth=0, facecolor=TASK[tid], alpha=0.10,
            )
            ax.add_patch(box)
            ax.text(kx + kbw / 2, y + kbh / 2 + 0.13, TASK_LABEL[tid], ha="center", va="center",
                     fontsize=10.3, fontweight="bold", color=TASK[tid])
            ax.text(kx + kbw / 2, y + kbh / 2 - 0.19, "no sealed source yet",
                     ha="center", va="center", fontsize=8.0, style="italic", color=MUTED)
        else:
            rounded_box(ax, kx, y, kbw, kbh, fill=TASK[tid], edge=TASK[tid], lw=0)
            ax.text(kx + kbw / 2, y + kbh / 2, TASK_LABEL[tid], ha="center", va="center",
                     fontsize=10.3, fontweight="bold", color="white")

    def draw(edges, source_pos, source_box_w, box_h, *, from_right, target_x):
        for name, tid, tier in edges:
            sy = source_pos[name] + box_h / 2
            sx = ex_ if from_right else tx + source_box_w
            ty = task_pos[tid] + kbh / 2
            # Curve direction mirrors for the right-hand side so both fans read as
            # matching arcs rather than one side bowing the opposite way.
            base_rad = 0.14 if ty > sy else -0.14
            rad = -base_rad if from_right else base_rad
            lw = 1.9 if tier == "primary" else 1.3
            alpha = 0.62 if tier == "primary" else 0.34
            linestyle = (0, (4, 2)) if from_right else ("solid" if tier == "primary" else (0, (2, 2)))
            patch = mpatches.FancyArrowPatch(
                (sx, sy), (target_x, ty), arrowstyle="-|>", mutation_scale=13,
                color=TASK[tid], linewidth=lw, alpha=alpha, shrinkA=0, shrinkB=0,
                linestyle=linestyle, connectionstyle=f"arc3,rad={rad}",
            )
            ax.add_patch(patch)

    draw(TRAIN_EDGES, train_pos, tbw, tbh, from_right=False, target_x=kx)
    draw(EVAL_EDGES, eval_pos, ebw, ebh, from_right=True, target_x=kx + kbw)

    ax.text(tx + tbw / 2, h - 0.22, "TRAIN — fits the learnable heads",
             ha="center", va="center", fontsize=10, fontweight="bold", color=MUTED)
    ax.text(ex_ + ebw / 2, h - 0.22, "EVAL — sealed, frozen comparison",
             ha="center", va="center", fontsize=10, fontweight="bold", color=MUTED)

    legend_y = 0.42
    lx = w / 2 - 2.55
    ax.plot([lx, lx + 0.32], [legend_y, legend_y], color=MUTED, lw=2.0)
    ax.text(lx + 0.40, legend_y, "primary role, train", va="center", fontsize=7.8, color=MUTED)
    ax.plot([lx + 1.85, lx + 2.17], [legend_y, legend_y], color=MUTED, lw=1.4, ls=(0, (2, 2)))
    ax.text(lx + 2.25, legend_y, "secondary (control / weak / transfer)", va="center",
             fontsize=7.8, color=MUTED)
    ax.plot([lx + 5.15, lx + 5.47], [legend_y, legend_y], color=MUTED, lw=2.0, ls=(0, (4, 2)))
    ax.text(lx + 5.55, legend_y, "eval source, dashed = sealed", va="center",
             fontsize=7.8, color=MUTED)
    ax.text(w / 2, legend_y - 0.30,
             "Real Task-2 candidates (ALAMEDA PD, COPS, WATCH-PD, KneE-PAD) are conditional or "
             "gated — none is a ready sealed source yet.",
             ha="center", va="center", fontsize=8.0, style="italic", color=MUTED)

    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    for tid, fn in TASK_ICON.items():
        icon_badge(ROOT / f"icon_task{tid}.png", fn, TASK[tid])

    workflow(ROOT / "workflow_task1.png", 1, [
        ("frozen", "Baseline encoder", ["Frozen HALO patch embeddings", "for reference + full query", "timeline, native rate."], icon_frozen),
        ("trained", "Learnable head", ["Small trained projection,", "identity-initialized, for", "same-motion similarity."], icon_trained),
        ("taskhead", "Task head", ["Subsequence DTW search,", "open begin/end, bounded", "local warp, full timeline."], icon_detect),
        ("output", "Prediction", ["Event intervals + scores", "after temporal NMS,", "with retrieved evidence."], icon_output),
    ])
    workflow(ROOT / "workflow_task2.png", 2, [
        ("frozen", "Baseline encoder", ["Frozen HALO patch embeddings", "for the reference and", "comparison executions."], icon_frozen),
        ("trained", "Learnable head", ["Small diagonal-reweight", "projection tuned for", "same-task comparison."], icon_trained),
        ("taskhead", "Task head", ["Resample both to phase;", "compute phase-local residual", "+ signed latent delta."], icon_compare),
        ("output", "Prediction", ["Change score, phase-local", "deviation curve, interpretable", "duration/intensity deltas."], icon_output),
    ])
    workflow(ROOT / "workflow_task3.png", 3, [
        ("frozen", "Baseline encoder", ["Frozen HALO patch embeddings,", "multiscale-pooled across", "the full recording."], icon_frozen),
        ("trained", "Learnable head", ["Small projection + calibrated", "cosine affinity, trained on", "same/different identities."], icon_trained),
        ("taskhead", "Task head", ["Mutual-nearest-neighbour", "graph, IoU dedup, connected-", "component clustering."], icon_discover),
        ("output", "Prediction", ["Ranked motif clusters: count,", "duration, cadence, examples,", "for human confirmation."], icon_output),
    ])
    dataset_mix(ROOT / "dataset_mix.png")
    print("wrote", sorted(p.name for p in ROOT.glob("*.png")))
