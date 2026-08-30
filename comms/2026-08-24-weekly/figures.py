"""Render the weekly-deck figures in the HALO deck house style.

House style taken from HALO_Weekly_20260811.pptx: white ground, muted office palette anchored on
the deck's blue. The chart trio is a separable re-step of that blue -- the deck's own blue/green
pair fails CVD separation at dE 5 and was only ever used for semantic accents, never for
adjacent series.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("comms/2026-08-24-weekly/figures")
DATA = json.load(open("comms/2026-08-24-weekly/data/deck.json"))
KS = [1, 2, 4, 8, 16]
KSS = [str(k) for k in KS]

# Validated on a white surface: every check PASS, worst adjacent dE 21.4 (protan).
C1, C2, C3 = "#2A64A8", "#B5651D", "#7B4B94"
INK, MID, GRID, MUTED, BG = "#1A1A1A", "#555555", "#E4E4E4", "#9A9A96", "#FFFFFF"
BAND = "#EFF4EE"
REGIMES = {"ordinary": "Ordinary activities", "specialized_novel": "Specialized / clinical"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MID, "axes.edgecolor": GRID,
    "xtick.color": MID, "ytick.color": MID, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def series(block, regime):
    return [block.get(regime, {}).get(k, np.nan) for k in KSS]


def headline(regime):
    h = DATA["headline"]["nearest"]
    order = sorted([m for m in h if m != "HALO"],
                   key=lambda m: -np.nanmean(series(h[m], regime)))
    fig, ax = plt.subplots(figsize=(7.5, 4.75))
    x = np.arange(len(KS))
    for name in reversed(order[2:]):
        ax.plot(x, series(h[name], regime), color=MUTED, lw=1.3, zorder=1)
        ax.annotate(name, (x[-1], series(h[name], regime)[-1]), xytext=(7, 0),
                    textcoords="offset points", color=MUTED, fontsize=8.5, va="center")
    for name, colour in zip(order[:2], (C2, C3)):
        ax.plot(x, series(h[name], regime), color=colour, lw=1.9, marker="o", ms=4.5, zorder=2)
        ax.annotate(name, (x[-1], series(h[name], regime)[-1]), xytext=(7, 0),
                    textcoords="offset points", color=colour, fontsize=9.5,
                    fontweight="bold", va="center")
    y = series(h["HALO"], regime)
    ax.plot(x, y, color=C1, lw=3.0, marker="o", ms=6.5, zorder=3)
    ax.annotate("HALO", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                color=C1, fontsize=11.5, fontweight="bold", va="center")
    ax.set_xticks(x); ax.set_xticklabels(KS)
    ax.set_xlabel("enrolled executions per class  (k)")
    ax.set_ylabel("macro F1")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-0.2, len(KS) - 1 + 1.45)
    save(fig, f"headline_{regime}")


def zero_shot():
    z = DATA["zero_shot"]
    names = sorted(z, key=lambda m: -z[m].get("ordinary", 0))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    y = np.arange(len(names))
    hi = names.index("HALO")
    ax.axhspan(hi - 0.45, hi + 0.45, color=BAND, zorder=0)
    ax.barh(y - 0.19, [z[n].get("ordinary", 0) for n in names], height=0.34,
            color=C1, label="ordinary", zorder=2)
    ax.barh(y + 0.19, [z[n].get("specialized_novel", 0) for n in names], height=0.34,
            color=C2, label="specialized / clinical", zorder=2)
    for i, n in enumerate(names):
        for off, key in ((-0.19, "ordinary"), (0.19, "specialized_novel")):
            ax.text(z[n].get(key, 0) + 0.6, i + off, f"{z[n].get(key, 0):.1f}",
                    va="center", fontsize=9, color=MID)
    ax.set_yticks(y); ax.set_yticklabels(names)
    for tick, n in zip(ax.get_yticklabels(), names):
        if n == "HALO":
            tick.set_fontweight("bold"); tick.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlabel("macro F1")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, labelcolor=MID)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.margins(x=0.08)
    save(fig, "zero_shot")


RUNGS = [
    ("support_raw_1nn",   "1. stop pooling: keep every recording row"),
    ("full_raw_1nn",      "2. add the 512-row corpus memory"),
    ("full_reranked_1nn", "3. apply the learned reranker  (= full native engine)"),
]


def ladder():
    colours = [C1, C2, C3]
    lad = DATA["ladder"]
    base = {r: series(lad["pooled_execution_1nn"], r) for r in REGIMES}
    fig, axes = plt.subplots(1, 2, figsize=(12.13, 4.3), sharey=True)
    width = 0.26
    for ax, (regime, title) in zip(axes, REGIMES.items()):
        x = np.arange(len(KS))
        prev = np.array(base[regime], dtype=float)
        for i, (key, label) in enumerate(RUNGS):
            cur = np.array(series(lad[key], regime), dtype=float)
            delta = cur - prev
            ax.bar(x + (i - 1) * width, delta, width * 0.9, color=colours[i],
                   label=label if regime == "ordinary" else None)
            for xi, d in zip(x + (i - 1) * width, delta):
                lift = 4 + (9 * i if abs(d) < 0.25 else 0)
                ax.annotate(f"{d:+.2f}", (xi, d),
                            xytext=(0, lift if d >= 0 else -(lift + 8)),
                            textcoords="offset points", ha="center", fontsize=8, color=MID)
            prev = cur
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(KS)
        ax.set_xlabel("enrolled executions per class  (k)")
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.margins(y=0.18)
    axes[0].set_ylabel("macro F1 added by this step")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=MID,
                   handlelength=1.1, borderpad=0.1)
    save(fig, "ladder")


VERSIONS = [("best (long_4h)", "long_4h\n(pre-engine)"), ("PB-01", "PB-01\nmix + vote"),
            ("PB-02", "PB-02\nscalar + vote"), ("PB-03", "PB-03\nrerank + 1-NN")]


def trajectory():
    s = DATA["series"]
    fig, axes = plt.subplots(1, 2, figsize=(12.13, 4.3))
    x = np.arange(len(VERSIONS))
    for ax, (regime, title) in zip(axes, REGIMES.items()):
        enc = [np.nanmean(series(s[k]["nearest"], regime)) for k, _ in VERSIONS]
        eng = [np.nanmean(series(s[k]["engine"], regime)) if s[k]["engine"] else np.nan
               for k, _ in VERSIONS]
        ax.plot(x, enc, color=C1, lw=2.8, marker="o", ms=7, label="encoder (1-NN readout)")
        ax.plot(x, eng, color=C2, lw=2.2, marker="s", ms=6, ls="--",
                label="native engine readout")
        span = np.nanmax(enc + eng) - np.nanmin(enc + eng)
        close = [abs(a - b) < 0.09 * span if not np.isnan(b) else False
                 for a, b in zip(enc, eng)]
        for xi, v, tight in zip(x, enc, close):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 21 if tight else 12),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=C1, fontweight="bold")
        for xi, v, tight in zip(x, eng, close):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, -27 if tight else -20),
                            textcoords="offset points", ha="center", fontsize=9.5, color=C2)
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in VERSIONS], fontsize=9)
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.margins(y=0.19, x=0.09)
    axes[0].set_ylabel("macro F1, mean over k = 1…16")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=MID,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    save(fig, "trajectory")


for f in OUT.glob("*.png"):
    f.unlink()
for regime in REGIMES:
    headline(regime)
zero_shot(); ladder(); trajectory()
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
