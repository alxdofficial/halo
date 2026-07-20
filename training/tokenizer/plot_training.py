"""Live training-telemetry plots — read a run's log.jsonl and render a multi-panel PNG.

Panels: per-objective + total loss · learning rate · per-module gradient norms (log) · val kNN-BA
(overall + per data source) · per-source A1 loss. Pass --watch N to re-render every N seconds while a
run is live (leave a terminal running it next to training).

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


def render(log_path: Path, out_path: Path):
    S, V = load(log_path)
    fig, ax = plt.subplots(3, 2, figsize=(15, 11))

    for k, lbl in (("total", "total"), ("a1_masked", "A1"), ("a2_supcon", "A2"),
                   ("a3_grounding", "A3")):
        x, y = series(S, k)
        if x:
            ax[0, 0].plot(x, y, lw=0.9, label=lbl)
    ax[0, 0].set_title("losses"); ax[0, 0].set_xlabel("step"); ax[0, 0].legend(fontsize=8)

    x, y = series(S, "lr")
    ax[0, 1].plot(x, y); ax[0, 1].set_title("learning rate"); ax[0, 1].set_xlabel("step")

    for k in ("grad/encoder", "grad/a1", "grad/a2", "grad/a3_cad", "grad/a3_eig"):
        x, y = series(S, k)
        if x:
            ax[1, 0].plot(x, y, lw=0.9, label=k.split("/")[1])
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("per-module grad norm (log)")
    ax[1, 0].set_xlabel("step"); ax[1, 0].legend(fontsize=8)

    x, y = series(V, "val_knn_ba")
    if x:
        ax[1, 1].plot(x, y, "k-o", ms=3, lw=1.5, label="kNN-BA")
    xc, yc = series(V, "val_conse_ba")
    if xc:
        ax[1, 1].plot(xc, yc, "b-s", ms=3, lw=1.5, label="ConSE text-cos")
    for s, (xs, ys) in sorted(dict_series(V, "val_ba_by_source").items()):
        ax[1, 1].plot(xs, ys, lw=0.7, alpha=0.4, color="gray")
    ax[1, 1].set_title("val accuracy (bold=overall kNN & ConSE, thin gray=per-source kNN)")
    ax[1, 1].set_xlabel("step"); ax[1, 1].legend(fontsize=8)

    for s, (xs, ys) in sorted(dict_series(S, "a1_by_source").items()):
        ax[2, 0].plot(xs, ys, lw=0.8, label=s)
    ax[2, 0].set_title("A1 loss per data source"); ax[2, 0].set_xlabel("step")
    ax[2, 0].legend(fontsize=6, ncol=2)

    ax[2, 1].axis("off")
    best = max((r["val_knn_ba"] for r in V), default=float("nan"))
    best_c = max((r.get("val_conse_ba", float("nan")) for r in V), default=float("nan"))
    txt = (f"steps logged : {S[-1]['step'] if S else 0}\n"
           f"val points   : {len(V)}\n"
           f"best kNN-BA   : {best:.3f}\n"
           f"best ConSE-BA : {best_c:.3f}\n"
           f"last total    : {series(S, 'total')[1][-1] if series(S, 'total')[1] else float('nan'):.3f}")
    ax[2, 1].text(0.03, 0.92, txt, va="top", family="monospace", fontsize=12)

    fig.suptitle(f"training telemetry — {log_path.parent.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
