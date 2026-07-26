"""Generic baseline evaluation runner for the current ZS-XD protocol.

One loop over the adapter ``baselines.REGISTRY`` — no per-baseline dispatch.
Each registered adapter is scored on every held-out eval dataset/stream via its
shared :meth:`BaselineAdapter.evaluate`, which returns either the current metric bundle
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
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Sequence, Tuple

import baselines as B
from data.scripts.curate import deployment_policy as policy
from eval.protocol import protocol_fingerprint

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


@contextmanager
def _exclusive_cell(path: Path):
    """Prevent two current runners from evaluating/writing the same result cell."""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"result cell is already being evaluated by another process: {path}"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} acquired={datetime.now(timezone.utc).isoformat()}\n")
        lock.flush()
        yield


def _run_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "id": str(uuid.uuid4()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "git_commit": commit,
        "git_dirty": dirty,
    }


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
    setup_elapsed_s: float = 0.0,
    run_provenance: dict | None = None,
) -> str:
    """Score one (baseline, dataset, stream) cell and write its JSON.

    Returns the cell ``_status``: ``"complete"``, ``"na"`` or ``"failed"``.
    """
    out_path = result_path(results_dir, baseline, dataset, stream)
    with _exclusive_cell(out_path):
        # Drop any stale final BEFORE running so a crash can't leave last run's file.
        if out_path.exists():
            out_path.unlink()

        # Stamp the protocol into EVERY cell (including failures and n/a), so the assembler can tell
        # a 59-label result from a 93-label one. Without this, stale results are indistinguishable
        # from current ones and a table silently mixes protocols.
        adapter = B.REGISTRY[baseline]
        base = {"_baseline": baseline, "_dataset": dataset, "_stream": stream,
                "_alignment": alignment, "_protocol": protocol_fingerprint(),
                "_run": run_provenance or _run_provenance(),
                "_adapter": {"module": type(adapter).__module__, "class": type(adapter).__name__,
                             "tier": adapter.tier},
                "_timing": {"setup_s": round(float(setup_elapsed_s), 3)}}
        cell_started = time.time()
        try:
            result = adapter.evaluate(dataset, stream, alignment=alignment,
                                      device=device, state=state)
        except Exception as e:  # a crash is a RECORDED failure, never a silent skip
            import traceback
            traceback.print_exc()
            base["_timing"]["cell_s"] = round(time.time() - cell_started, 3)
            _atomic_write(out_path, {**base, "_status": "failed", "error": repr(e)})
            print(f"  {dataset:14} {stream:22} FAILED: {e}")
            return "failed"

        base["_timing"]["cell_s"] = round(time.time() - cell_started, 3)
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
    provenance = _run_provenance()

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
        setup_started = time.time()
        try:
            state = adapter.setup(device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            setup_error = e
            print(f"  !! setup failed: {e}")
        setup_elapsed_s = time.time() - setup_started

        ran.append(name)
        for ds, stream in cells:
            if setup_error is not None:
                out_path = result_path(results_dir, name, ds, stream)
                with _exclusive_cell(out_path):
                    if out_path.exists():
                        out_path.unlink()
                    _atomic_write(out_path, {"_baseline": name, "_dataset": ds,
                                             "_stream": stream, "_alignment": alignment,
                                             "_protocol": protocol_fingerprint(),
                                             "_run": provenance,
                                             "_adapter": {
                                                 "module": type(adapter).__module__,
                                                 "class": type(adapter).__name__,
                                                 "tier": adapter.tier,
                                             },
                                             "_timing": {
                                                 "setup_s": round(setup_elapsed_s, 3),
                                                 "cell_s": 0.0,
                                             },
                                             "_status": "failed",
                                             "error": f"setup failed: {setup_error!r}"})
                failed_cells.append((name, ds, stream))
                continue
            status = run_cell(name, ds, stream, alignment=alignment, device=device,
                              state=state, results_dir=results_dir,
                              setup_elapsed_s=setup_elapsed_s,
                              run_provenance=provenance)
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

    protocol = protocol_fingerprint()
    print(f"Protocol v{protocol['version']} ({protocol['n_labels']} labels) | "
          f"device={device} | alignment={args.alignment} | "
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
