#!/usr/bin/env python3
"""Data-QA: randomly sample sessions from a dataset and plot their sensor channels over
time with the activity label, so the data + labels can be visually sanity-checked.

    python datascripts/shared/plot_sessions.py <dataset> [--n 6] [--seed 0]
    python datascripts/shared/plot_sessions.py --all [--n 6]

Output: test_output/data_qa/<dataset>_sessions.png (gitignored).
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "test_output" / "data_qa"

# subject id from session name (best-effort; None -> unknown)
def _subject(sid, ds):
    try:
        pat = {"uci_har": r"_(\d+)_", "pamap2": r"subject(\d+)", "kuhar": r"s(\d+)",
               "hapt": r"user(\d+)", "mhealth": r"subject(\d+)"}
        if ds == "wisdm":     return sid.split("_")[3]
        if ds == "dsads":     return sid.split("_")[1]
        if ds in ("hhar", "recgym", "capture24", "shoaib", "harth", "mobiact"):
            return sid.split("_")[1]
        if ds == "inclusivehar": return sid.split("_")[1]
        m = re.search(pat.get(ds, r"$^"), sid)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


def plot_dataset(ds, n=6, seed=0, max_ch=12):
    sdir = DATA / ds / "sessions"
    labels = json.load(open(DATA / ds / "labels.json"))
    sids = sorted(labels.keys())
    rng = np.random.RandomState(seed)
    pick = [sids[i] for i in rng.choice(len(sids), size=min(n, len(sids)), replace=False)]

    fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.3 * len(pick)), squeeze=False)
    for ax, sid in zip(axes[:, 0], pick):
        df = pd.read_parquet(sdir / sid / "data.parquet")
        t = df["timestamp_sec"].to_numpy() if "timestamp_sec" in df.columns else np.arange(len(df))
        t = t - t[0]
        chans = [c for c in df.columns if c != "timestamp_sec"]
        for c in chans[:max_ch]:
            ax.plot(t, df[c].to_numpy(), lw=0.7, label=c)
        lab = labels[sid]
        lab = lab[0] if isinstance(lab, list) and lab else lab
        dur = float(t[-1]) if len(t) > 1 else 0.0
        rate = (len(df) - 1) / dur if dur > 0 else float("nan")
        ax.set_title(f"{ds} | {sid}  |  label={lab}  |  subj={_subject(sid, ds)}  |  "
                     f"{rate:.0f} Hz  {dur:.1f}s  {len(df)} samp  |  {len(chans)} ch"
                     + ("" if len(chans) <= max_ch else f" (first {max_ch})"),
                     fontsize=8, loc="left")
        ax.legend(fontsize=5, ncol=8, loc="upper right", framealpha=0.4)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        ax.margins(x=0.005)
    fig.suptitle(f"{ds}: {len(pick)} random sessions (seed {seed})", fontsize=12, y=0.997)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{ds}_sessions.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    targets = sorted([p.name for p in DATA.iterdir() if (p / "sessions").is_dir()]) if a.all else [a.dataset]
    for ds in targets:
        try:
            print("wrote", plot_dataset(ds, n=a.n, seed=a.seed))
        except Exception as e:
            print(f"FAILED {ds}: {type(e).__name__}: {e}")
