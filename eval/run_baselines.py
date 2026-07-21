"""Generic baseline evaluation RUNNER for the ZS-XD protocol (v2).

One loop over the adapter ``baselines.REGISTRY`` — no per-baseline dispatch.
Each registered adapter is scored on every held-out eval dataset/stream via its
shared :meth:`BaselineAdapter.evaluate`, which returns either a v2 metric bundle
or a disclosed ``{"status": "n/a", ...}`` for an incompatible dataset. Every
(baseline, dataset, stream) cell is written to its own JSON under
``eval/results/`` (gitignored) so the table assembler can reject a missing or
partial grid loudly instead of blank-filling it.

Fail-loud + complete discipline (ported from the legacy driver):
  * A cell's FINAL JSON is never written incrementally: results stream to a
    ``.partial.json`` sidecar and are atomically promoted to the final path only
    once the cell has a definite outcome — so a mid-write crash can never leave a
    file that reads as a completed result.
  * A stale final JSON is deleted BEFORE a cell runs, so a crash can't leave an
    old complete-looking file standing in for the fresh one.
  * A disclosed-incompatible dataset is recorded as an explicit ``n/a`` cell
    (``_status="na"``) — not silently skipped, not scored as a real number.
  * A crash is RECORDED as a failed cell (``_status="failed"`` + the error), not
    swallowed, and the run exits non-zero. assemble_table then rejects that cell
    loudly rather than treating it as absent.

Usage::

    python -m eval.run_baselines
    python -m eval.run_baselines --baselines crosshar --datasets motionsense shoaib
    python -m eval.run_baselines --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import baselines as B
from data.scripts.curate import deployment_policy as policy

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "eval" / "results"


# =============================================================================
# Cell enumeration (shared with assemble_table so the two agree exactly)
# =============================================================================

def resolve_eval_cells(datasets: Sequence[str]) -> List[Tuple[str, str]]:
    """Expand each dataset to its primary ``(dataset, stream_id)`` eval cells.

    A dataset with no primary stream in the deployment policy is a caller error
    (fail loud) rather than a silently empty column.
    """
    cells: List[Tuple[str, str]] = []
    for ds in datasets:
        specs = policy.stream_specs(ds, "primary")
        if not specs:
            raise ValueError(
                f"dataset {ds!r} has no primary eval stream in deployment_policy "
                f"(known primary eval datasets: {list(policy.PRIMARY_EVAL_DATASETS)})"
            )
        for spec in specs:
            cells.append((ds, spec.stream_id))
    return cells


def result_path(results_dir: Path, baseline: str, dataset: str, stream: str) -> Path:
    return results_dir / f"{baseline}__{dataset}__{stream}.json"


def _atomic_write(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically via a ``.partial.json`` sidecar."""
    partial = path.with_suffix(".partial.json")
    with open(partial, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    partial.replace(path)   # atomic promote — final exists only once fully written


# =============================================================================
# One cell
# =============================================================================

def run_cell(
    baseline: str,
    dataset: str,
    stream: str,
    *,
    alignment: str,
    device,
    state,
    results_dir: Path,
) -> str:
    """Score one (baseline, dataset, stream) cell and write its JSON.

    Returns the cell ``_status``: ``"complete"``, ``"na"`` or ``"failed"``.
    """
    out_path = result_path(results_dir, baseline, dataset, stream)
    # Drop any stale final BEFORE running so a crash can't leave last run's file.
    if out_path.exists():
        out_path.unlink()

    # Stamp the protocol into EVERY cell (including failures and n/a), so the assembler can tell
    # a 59-label result from a 93-label one. Without this, stale results are indistinguishable
    # from current ones and a table silently mixes protocols.
    from eval.protocol import protocol_fingerprint
    base = {"_baseline": baseline, "_dataset": dataset, "_stream": stream,
            "_alignment": alignment, "_protocol": protocol_fingerprint()}
    adapter = B.REGISTRY[baseline]
    try:
        result = adapter.evaluate(dataset, stream, alignment=alignment,
                                  device=device, state=state)
    except Exception as e:  # a crash is a RECORDED failure, never a silent skip
        import traceback
        traceback.print_exc()
        _atomic_write(out_path, {**base, "_status": "failed", "error": repr(e)})
        print(f"  {dataset:14} {stream:22} FAILED: {e}")
        return "failed"

    if isinstance(result, dict) and result.get("status") == "n/a":
        _atomic_write(out_path, {**base, "_status": "na",
                                 "na_reason": result.get("reason", "")})
        print(f"  {dataset:14} {stream:22} N/A ({result.get('reason', '')})")
        return "na"

    _atomic_write(out_path, {**base, "_status": "complete", "metrics": result})
    f1 = result.get("f1_macro")
    if result.get("ci_degenerate"):
        ci = "[degenerate]"
    else:
        ci = f"[{result.get('f1_macro_ci_lo', float('nan')):.1f}," \
             f"{result.get('f1_macro_ci_hi', float('nan')):.1f}]"
    print(f"  {dataset:14} {stream:22} F1={f1:5.1f} {ci}  "
          f"bAcc={result.get('balanced_accuracy', float('nan')):5.1f}")
    return "complete"


# =============================================================================
# Driver
# =============================================================================

def run(
    baselines: Sequence[str],
    datasets: Sequence[str],
    *,
    alignment: str = "non_harmonised",
    device="cpu",
    results_dir: Path = RESULTS_DIR,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Run every (baseline × cell). Returns ``(ran, failed_cells)`` where
    ``failed_cells`` is a list of ``(baseline, dataset, stream)``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cells = resolve_eval_cells(datasets)

    unknown = [b for b in baselines if b not in B.REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown baseline(s) {unknown}; registered: {sorted(B.REGISTRY)}"
        )

    ran: List[str] = []
    failed_cells: List[Tuple[str, str, str]] = []
    for name in baselines:
        adapter = B.REGISTRY[name]
        print(f"\n{'#' * 60}\n# {name.upper()} (tier={adapter.tier})\n{'#' * 60}")
        # setup once per baseline; a setup crash fails every cell for it (recorded).
        state = None
        setup_error = None
        try:
            state = adapter.setup(device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            setup_error = e
            print(f"  !! setup failed: {e}")

        ran.append(name)
        for ds, stream in cells:
            if setup_error is not None:
                out_path = result_path(results_dir, name, ds, stream)
                if out_path.exists():
                    out_path.unlink()
                _atomic_write(out_path, {"_baseline": name, "_dataset": ds,
                                         "_stream": stream, "_alignment": alignment,
                                         "_status": "failed",
                                         "error": f"setup failed: {setup_error!r}"})
                failed_cells.append((name, ds, stream))
                continue
            status = run_cell(name, ds, stream, alignment=alignment, device=device,
                              state=state, results_dir=results_dir)
            if status == "failed":
                failed_cells.append((name, ds, stream))

    return ran, failed_cells


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="baselines to run (default: all registered)")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS),
                    help="eval datasets (default: PRIMARY_EVAL_DATASETS)")
    ap.add_argument("--alignment", default="non_harmonised",
                    choices=["non_harmonised", "harmonised"])
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    baselines = args.baselines if args.baselines is not None else sorted(B.REGISTRY)
    if not baselines:
        print("!! no baselines to run (registry is empty and none requested).")
        return 1
    device = args.device or _default_device()
    results_dir = Path(args.results_dir)

    print(f"Protocol v2 | device={device} | alignment={args.alignment} | "
          f"registry={sorted(B.REGISTRY)} | run={baselines}")

    ran, failed_cells = run(baselines, args.datasets, alignment=args.alignment,
                            device=device, results_dir=results_dir)

    if failed_cells:
        print(f"\n!! {len(failed_cells)} cell(s) FAILED: "
              f"{[f'{b}/{d}/{s}' for b, d, s in failed_cells]}")
        return 1
    n_cells = len(ran) * len(resolve_eval_cells(args.datasets))
    print(f"\nOK all {len(ran)} baseline(s) x cells complete "
          f"({n_cells} cells, every requested dataset present).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
