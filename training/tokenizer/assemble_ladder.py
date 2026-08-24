"""Assemble the native-e2e arm ladder into one table.

Every arm ran the identical loop from the identical Phase-A checkpoint. What differs is which
modules were live, which encoder depth was stored, and how much evidence the mixer could attend
over. The table exists so those three axes can be read separately instead of as one "e2e" number.

Columns, and why each one is here:

  hard mean   macro-F1 under the DEPLOYMENT rule (per-candidate top-k), averaged over support
              counts. The headline. Grouped by k rather than pooled everywhere else, because k=0
              and k=4 are different tasks with different floors and a pooled mean hides a zero-shot
              collapse behind few-shot gains.
  k=0         held-out CONCEPTS with no support. The only cell that tests the semantic bridge, and
              the one the previous learned decoder pushed below chance.
  transfer    Phase-A development-transfer canary. The anchor arm established that the task gain
              and the representation damage were the same quantity, so an arm that wins the
              headline while dropping this has not won anything.
  gain        pair_gain at the end. Still at 0.02 means the mixer never turned itself on.
  corr        mean |correction| in log-weight units at the end. Runaway growth here with a flat
              headline is the mixer overwriting retrieval rather than refining it.

    python -m training.tokenizer.assemble_ladder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("training/tokenizer/outputs")


def _last(rows: list[dict], key: str):
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def collect(directory: Path) -> dict | None:
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    log = [json.loads(line) for line in (directory / "log.jsonl").read_text().splitlines() if line]
    final = summary["final_validation"]
    initial = summary["initial_validation"]
    curve = final["hard_curve"]
    config_path = directory / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return {
        "arm": directory.name.split("_", 1)[-1] if "_" in directory.name else directory.name,
        "config": config,
        "hard_mean": final["hard_mean_macro_f1"],
        "initial_hard_mean": initial["hard_mean_macro_f1"],
        "curve": {k: cell["macro_f1"] for k, cell in curve.items()},
        "transfer": _last(log, "transfer/mean"),
        "pair_gain": _last(log, "mixer/pair_gain"),
        "correction": _last(log, "mixer/mean_abs_residual"),
        "pool_rows": _last(log, "mixer/pool_rows"),
        "reached": _last(log, "dead_path/memory_rows_with_gradient"),
        "top_k_share": _last(log, "dead_path/top_k_gradient_share"),
        "agreement": _last(log, "deploy/argmax_agreement"),
        "minutes": summary["elapsed_seconds"] / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prefix", default="ladder_")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    arms = [a for a in (collect(d) for d in sorted(args.root.glob(f"{args.prefix}*"))
                        if d.is_dir()) if a]
    if not arms:
        raise SystemExit(f"no completed arms under {args.root}/{args.prefix}*")
    ks = sorted({k for a in arms for k in a["curve"]}, key=lambda s: int(s.split("=")[1]))

    header = (f"{'arm':18s} {'hard mean':>10s} {'Δ ctrl':>8s} "
              + " ".join(f"{k:>7s}" for k in ks)
              + f" {'transfer':>9s} {'gain':>7s} {'corr':>7s} {'min':>6s}")
    lines = [header, "-" * len(header)]
    control = next((a["hard_mean"] for a in arms if a["arm"] == "control_full"), None)
    for arm in sorted(arms, key=lambda a: -a["hard_mean"]):
        delta = f"{arm['hard_mean'] - control:+.4f}" if control is not None else "     n/a"
        lines.append(
            f"{arm['arm']:18s} {arm['hard_mean']:>10.4f} {delta:>8s} "
            + " ".join(f"{arm['curve'].get(k, float('nan')):>7.4f}" for k in ks)
            + f" {arm['transfer'] if arm['transfer'] is not None else float('nan'):>9.4f}"
            + f" {arm['pair_gain'] if arm['pair_gain'] is not None else float('nan'):>7.4f}"
            + f" {arm['correction'] if arm['correction'] is not None else float('nan'):>7.3f}"
            + f" {arm['minutes']:>6.1f}"
        )
    lines.append("")
    lines.append("what each arm actually ran (decoded from run_config.json, not from the name):")
    for arm in sorted(arms, key=lambda a: -a["hard_mean"]):
        config = arm["config"]
        if config is None:
            lines.append(f"  {arm['arm']:18s} (predates run_config.json — see the name)")
            continue
        bits = ["trains " + ("+".join(config["trains"]) or "nothing"),
                f"depth {config['retrieval_depth']}",
                "gate on" if config["gate"] else "GATE OFF"]
        if config.get("mixer_forms"):
            bits.append("forms " + "+".join(config["mixer_forms"]))
            bits.append(f"pool {config['mixer_pool'] or 'uncapped'}")
            if config.get("semantic_refine", "off") != "off":
                bits.append(f"REFINE TEXT: {config['semantic_refine']}")
        if config.get("scramble_text"):
            bits.append("SCRAMBLED VOCAB")
        bits.append(f"seed {config['seed']}")
        lines.append(f"  {arm['arm']:18s} " + " · ".join(bits))
    lines.append("")
    lines.append("dead path (last diagnostic step of each arm that trained a mixer):")
    for arm in arms:
        if arm["pool_rows"] is None:
            continue
        lines.append(
            f"  {arm['arm']:18s} pool {arm['pool_rows']:>6.0f} rows · "
            f"memory reached {arm['reached']:.3f} · top-k gradient share {arm['top_k_share']:.3f} "
            f"· soft/hard agreement {arm['agreement']:.3f}"
        )
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
