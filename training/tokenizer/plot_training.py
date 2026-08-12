"""Render the Phase-A JSONL telemetry as a CPU-only live dashboard.

Pass ``--watch N`` to refresh periodically. Rendering is atomic, so another process can safely read
the PNG while it is being updated.

Run:
  python -m training.tokenizer.plot_training \
    --log training/tokenizer/outputs/pretrain_native/log.jsonl [--watch 30]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def load(log_path: Path):
    step_recs, val_recs = [], []
    for line in log_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        (val_recs if "val_knn_ba" in r else step_recs).append(r)
    return step_recs, val_recs


def series(recs, key):
    xs, ys = [], []
    for r in recs:
        if r.get(key) is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def dict_series(recs, key):
    """{sub_key: ([steps], [values])} from records carrying a {sub_key: value} dict under `key`."""
    out: dict = {}
    for r in recs:
        for s, v in (r.get(key) or {}).items():
            out.setdefault(s, ([], []))
            out[s][0].append(r["step"])
            out[s][1].append(v)
    return out


def ewma(values, alpha=0.2):
    out = []
    for value in values:
        out.append(value if not out else alpha * value + (1.0 - alpha) * out[-1])
    return out


def _line(axis, records, key, label=None, *, smooth=False, **kwargs):
    x, y = series(records, key)
    if not x:
        return
    if smooth and len(y) > 2:
        # The faint raw underlay sets its OWN alpha/width, so drop any caller-supplied width or
        # colour rather than passing them twice -- every `smooth=True` caller passes lw, which
        # made --render a hard TypeError.
        underlay = {k: v for k, v in kwargs.items()
                    if k not in ("color", "lw", "linewidth", "alpha")}
        axis.plot(x, y, alpha=0.2, lw=0.6, **underlay)
        y = ewma(y)
    axis.plot(x, y, label=label or key, **kwargs)


def _legend(axis, **kwargs):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def render(log_path: Path, out_path: Path):
    S, V = load(log_path)
    fig, ax = plt.subplots(4, 3, figsize=(18, 15))

    loss_keys = (("total", "total"), ("loss_weighted/jepa", "JEPA family"),
                 ("loss_weighted/descriptor", "descriptor part"),
                 ("loss_weighted/vicreg", "VICReg weighted"))
    for key, label in loss_keys:
        fallback = {"loss_weighted/jepa": "jepa", "loss_weighted/vicreg": "vicreg"}.get(key)
        _line(ax[0, 0], S, key if series(S, key)[0] else fallback, label, smooth=True, lw=1.1)
    ax[0, 0].set_title("weighted losses (EWMA)"); _legend(ax[0, 0], fontsize=7)

    for key, label in (("vicreg/invariance", "invariance"), ("vicreg/variance", "variance"),
                       ("vicreg/covariance", "covariance"), ("vicreg/min_std", "min std")):
        _line(ax[0, 1], S, key, label, smooth=True, lw=1.0)
    ax[0, 1].set_title("VICReg components"); _legend(ax[0, 1], fontsize=7)

    for key, label in (("jepa/margin", "JEPA margin"), ("vicreg/margin", "VICReg margin"),
                       ("descriptor/top1", "descriptor top-1"),
                       ("descriptor/chance_top1", "descriptor chance")):
        _line(ax[0, 2], S, key, label, smooth=True, lw=1.1)
    ax[0, 2].axhline(0.0, color="black", lw=0.6)
    ax[0, 2].set_title("pair margins and descriptor retrieval")
    _legend(ax[0, 2], fontsize=7)

    for key, label in (("grad/total_preclip", "total"), ("grad/encoder", "encoder"),
                       ("grad/jepa_predictor", "JEPA head"),
                       ("grad/vicreg_projector", "VICReg head"),
                       ("grad/descriptor_head", "descriptor head"),
                       ("grad/bias_projection", "bias projection")):
        _line(ax[1, 0], S, key, label, lw=0.9)
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("gradient norms (pre-clip)")
    _legend(ax[1, 0], fontsize=7)

    for key, label in (("grad_objective/jepa", "JEPA"),
                       ("grad_objective/vicreg", "VICReg")):
        _line(ax[1, 1], S, key, label, marker="o", ms=3, lw=1.0)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_title("objective gradients on encoder")
    _legend(ax[1, 1], fontsize=7)
    geometry = ax[1, 1].twinx()
    _line(geometry, S, "grad_objective/jepa_share", "JEPA share", color="tab:green",
          marker=".", lw=0.8)
    _line(geometry, S, "grad_cosine/jepa_vs_vicreg", "cosine", color="tab:red",
          marker=".", lw=0.8)
    geometry.set_ylim(-1.05, 1.05)
    _legend(geometry, fontsize=6, loc="lower right")

    for key, label in (("repr_encoder/effective_rank", "encoder"),
                       ("repr_projector/effective_rank", "projector"),
                       ("repr_teacher/effective_rank", "EMA teacher")):
        _line(ax[1, 2], S, key, label, lw=1.0)
    ax[1, 2].set_title("representation effective rank"); _legend(ax[1, 2], fontsize=7)

    for key, label in (("repr_encoder/min_std", "encoder min"),
                       ("repr_encoder/mean_std", "encoder mean"),
                       ("repr_projector/min_std", "projector min"),
                       ("repr_projector/mean_std", "projector mean")):
        _line(ax[2, 0], S, key, label, smooth=True, lw=0.9)
    ax[2, 0].set_title("representation spread"); _legend(ax[2, 0], fontsize=7)

    for key, label, style in (
        ("val_knn_label_stream_ba", "kNN label/stream (selection)", "k-o"),
        ("val_knn_ba", "kNN global", "k--"),
        ("val_conse_label_stream_ba", "ConSE label/stream", "b-s"),
        ("val_conse_ba", "ConSE global", "b--"),
    ):
        x, y = series(V, key)
        if x:
            ax[2, 1].plot(x, y, style, ms=3, lw=1.3, label=label)
    for _, (xs, ys) in sorted(dict_series(V, "val_ba_by_source").items()):
        ax[2, 1].plot(xs, ys, lw=0.6, alpha=0.25, color="gray")
    ax[2, 1].set_title("validation (gray = per-source kNN)"); _legend(ax[2, 1], fontsize=6)

    latest = S[-1] if S else {}
    observed = latest.get("data/source_share_window", {})
    target = latest.get("data/source_share_target", {})
    names = sorted(set(observed) | set(target))
    if names:
        y = list(range(len(names)))
        ax[2, 2].barh([v - 0.18 for v in y], [observed.get(k, 0) for k in names], 0.36,
                      label="observed")
        ax[2, 2].barh([v + 0.18 for v in y], [target.get(k, 0) for k in names], 0.36,
                      label="target")
        ax[2, 2].set_yticks(y, names, fontsize=6)
    ax[2, 2].set_title("rolling dataset share"); _legend(ax[2, 2], fontsize=7)

    aug = latest.get("data/augmentation_rate_window", {})
    if aug:
        names = sorted(aug)
        ax[3, 0].barh(range(len(names)), [aug[k] for k in names])
        display = {
            "channel_dropout": "channel drop",
            "channel_text_dropout": "channel-text drop",
            "channel_text_phrase": "text paraphrase",
            "rotation_3d": "SO(3) rotation",
            "sensor_text_dropout": "sensor-text drop",
            "window_crop": "window crop",
        }
        ax[3, 0].set_yticks(range(len(names)), [display.get(name, name) for name in names], fontsize=6)
        ax[3, 0].set_xlim(0, 1)
    ax[3, 0].set_title("realized augmentation rate")

    _line(ax[3, 1], S, "perf/steps_per_s", "steps/s", lw=1.0, color="tab:green")
    ax[3, 1].set_title("throughput"); ax[3, 1].set_ylabel("steps/s")
    mem = ax[3, 1].twinx()
    _line(mem, S, "memory/allocated_gib", "allocated GiB", lw=0.9, color="tab:purple")
    _line(mem, S, "memory/reserved_gib", "reserved GiB", lw=0.9, color="tab:orange")
    mem.set_ylabel("GiB")
    _legend(ax[3, 1], fontsize=7, loc="upper left")
    _legend(mem, fontsize=7, loc="upper right")

    ax[3, 2].axis("off")
    best = max((r.get("val_knn_label_stream_ba", r.get("val_knn_ba", float("nan")))
                for r in V), default=float("nan"))
    best_c = max((r.get("val_conse_label_stream_ba", r.get("val_conse_ba", float("nan")))
                  for r in V), default=float("nan"))
    clip_values = series(S, "grad/clip_coefficient")[1]
    clipped_fraction = (sum(v < 1.0 for v in clip_values) / len(clip_values)
                        if clip_values else float("nan"))
    txt = (f"step          : {S[-1]['step'] if S else 0}\n"
           f"val points   : {len(V)}\n"
           f"best selection: {best:.3f}\n"
           f"best ConSE LS : {best_c:.3f}\n"
           f"last total    : {series(S, 'total')[1][-1] if series(S, 'total')[1] else float('nan'):.3f}\n"
           f"steps/s       : {latest.get('perf/steps_per_s', float('nan')):.2f}\n"
           f"ETA minutes   : {latest.get('perf/eta_minutes', float('nan')):.1f}\n"
           f"zero targets  : {latest.get('jepa_zero_target_frac_window', float('nan')):.2%}\n"
           f"descriptor tgt: {latest.get('descriptor/target_window_fraction', float('nan')):.2%}\n"
           f"descriptor top: {latest.get('descriptor/top1', float('nan')):.2%}\n"
           f"descriptor rnd: {latest.get('descriptor/chance_top1', float('nan')):.2%}\n"
           f"input finite  : {latest.get('data/input_finite_fraction', float('nan')):.6f}\n"
           f"input abs max : {latest.get('data/input_abs_max', float('nan')):.2f}\n"
           f"AMP skips     : {latest.get('amp/skipped_updates_total', 0)}\n"
           f"logged clipped: {clipped_fraction:.1%}")
    ax[3, 2].text(0.03, 0.95, txt, va="top", family="monospace", fontsize=10)

    for axis in ax.flat:
        if axis.axison:
            axis.set_xlabel("step" if not axis.patches else "")
            axis.grid(alpha=0.18)

    fig.suptitle(f"training telemetry — {log_path.parent.name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--watch", type=float, default=0.0, help="re-render every N seconds (0 = once)")
    args = ap.parse_args()
    out = args.out or args.log.parent / "telemetry.png"
    while True:
        if args.log.exists():
            render(args.log, out)
            print(f"-> {out}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
