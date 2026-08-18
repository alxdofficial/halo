"""Compare Phase-A fleet arms on the held-out development-transfer selection metric.

Reads each arm's `log.jsonl` and reports its BEST selection score with the posture canaries at
that step, against two measured reference points:

    PHASE_A_RANDOM_INIT_FLOOR   a random-init encoder of the same architecture
    OLD_GOOD                    the best checkpoint ever trained (phase_a_sensor_v1_20260813_v2)

An arm below the floor has learned nothing usable for retrieval. Screening noise on this metric is
about 0.012 (docs: 3k-step screening sd 0.0065), so differences under ~0.012 between arms are not
rankable from one seed and are printed as ties.

    python -m training.tokenizer.compare_phase_a_fleet [--glob 'phase_a_*_20260818']
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.tokenizer.eval_transfer import PHASE_A_RANDOM_INIT_FLOOR

OLD_GOOD = 0.8577
NOISE = 0.012
OUT_ROOT = Path(__file__).resolve().parent / "outputs"


def arm_rows(run_dir: Path) -> list[dict]:
    log = run_dir / "log.jsonl"
    if not log.exists():
        return []
    rows = []
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("development_transfer_knn_ba") is not None:
            rows.append(record)
    return rows


def summarize(run_dir: Path) -> dict | None:
    rows = arm_rows(run_dir)
    if not rows:
        return None
    best = max(rows, key=lambda r: r["development_transfer_knn_ba"])
    by = best.get("development_transfer_by_dataset", {})
    final = rows[-1]
    return {
        "arm": run_dir.name,
        "n_selections": len(rows),
        "best_step": best["step"],
        "best": best["development_transfer_knn_ba"],
        "final": final["development_transfer_knn_ba"],
        "final_step": final["step"],
        "posture_rw": by.get("posture/realworld"),
        "posture_sh": by.get("posture/shoaib"),
        "posture_ms": by.get("posture/motionsense"),
        "posture_ex": by.get("posture/extrasensory"),
        "extrasensory": by.get("extrasensory"),
        # Peaking early then decaying is the signature the old run showed; worth seeing per arm.
        "peaked_early": best["step"] < 0.5 * final["step"] if final["step"] else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="phase_a_*_20260818")
    parser.add_argument("--root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    arms = [s for s in (summarize(d) for d in sorted(args.root.glob(args.glob)) if d.is_dir()) if s]
    if not arms:
        print(f"no arms with selection scores under {args.root}/{args.glob}")
        return

    print(f"reference: random-init floor {PHASE_A_RANDOM_INIT_FLOOR:.4f} | "
          f"old-good 4k {OLD_GOOD:.4f} | screening noise ~{NOISE:.3f}\n")
    header = (f"{'arm':40} {'n':>3} {'best':>7} {'@step':>7} {'vs floor':>9} {'vs old':>8}  "
              f"{'post rw':>7} {'post sh':>7} {'post ex':>7}")
    print(header)
    print("-" * len(header))
    for a in sorted(arms, key=lambda r: -r["best"]):
        def fmt(value):
            return f"{value:7.3f}" if value is not None else f"{'-':>7}"
        print(f"{a['arm'][:40]:40} {a['n_selections']:>3} {a['best']:7.4f} {a['best_step']:>7} "
              f"{a['best'] - PHASE_A_RANDOM_INIT_FLOOR:+9.4f} {a['best'] - OLD_GOOD:+8.4f}  "
              f"{fmt(a['posture_rw'])} {fmt(a['posture_sh'])} {fmt(a['posture_ex'])}")

    print()
    best = max(arms, key=lambda r: r["best"])
    contenders = [a for a in arms if best["best"] - a["best"] <= NOISE]
    print(f"leader: {best['arm']} at {best['best']:.4f} (step {best['best_step']})")
    if len(contenders) > 1:
        print(f"  NOT separable from {len(contenders) - 1} other arm(s) at the {NOISE:.3f} noise "
              f"floor: {', '.join(a['arm'] for a in contenders if a is not best)}")
    print(f"  vs random-init floor : {best['best'] - PHASE_A_RANDOM_INIT_FLOOR:+.4f}"
          f"{'  <-- BELOW FLOOR, learned nothing usable' if best['best'] < PHASE_A_RANDOM_INIT_FLOOR else ''}")
    print(f"  vs old-good 4k       : {best['best'] - OLD_GOOD:+.4f}"
          f"{'  <-- does not recover the best checkpoint we have had' if best['best'] < OLD_GOOD - NOISE else ''}")
    for a in arms:
        if a["peaked_early"]:
            print(f"  note: {a['arm']} peaked at step {a['best_step']} of {a['final_step']} "
                  f"(final {a['final']:.4f}) -- early peak then decay")


if __name__ == "__main__":
    main()
