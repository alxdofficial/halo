"""Assemble the ZS-XD comparison table from the per-cell result JSONs written by
:mod:`eval.run_baselines`.

Rows = baselines, columns = eval datasets/streams, cells = macro-F1 (with a
subject-stratified 95% CI), plus a mean column. Prints a markdown table and,
optionally, writes it to a file.

Fail-loud discipline (ported from the legacy assembler): a cell is USED only
when its result JSON exists and is ``_status="complete"``. A missing cell, a
``_status="failed"`` cell, or a ``_status`` that is not one of the known values
is REJECTED loudly (the assembler raises) rather than blank-filled — a hole in
the grid must never be smoothed over into a printable number. A disclosed
``_status="na"`` cell is the one legitimate non-number: it is marked ``n/a`` and
excluded from the row mean's support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from data.scripts.curate import deployment_policy as policy
from eval.protocol import protocol_fingerprint, protocol_mismatch
from eval.run_baselines import RESULTS_DIR, resolve_eval_cells, result_path

KNOWN_STATUS = {"complete", "na", "failed"}


def _col_label(dataset: str, stream: str, dataset_multistream: set) -> str:
    """A dataset column is labelled by the dataset; disambiguate with the stream
    only when that dataset contributes more than one primary stream."""
    return f"{dataset}/{stream}" if dataset in dataset_multistream else dataset


def _load_cell(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


_CURRENT_PROTOCOL = None      # resolved lazily in collect(); see eval/protocol.py


def collect(
    baselines: Sequence[str],
    datasets: Sequence[str],
    results_dir: Path,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[Tuple[str, str], dict]], List[str]]:
    """Read every expected cell. Returns ``(cells, table, rejected)`` where
    ``table[baseline][cell]`` is the parsed JSON of a complete/na cell and
    ``rejected`` is a list of human-readable reasons a cell was rejected."""
    global _CURRENT_PROTOCOL
    _CURRENT_PROTOCOL = protocol_fingerprint()
    print(f"[assemble] protocol v{_CURRENT_PROTOCOL['version']} · "
          f"{_CURRENT_PROTOCOL['n_labels']} labels · vocab {_CURRENT_PROTOCOL['vocab_fp']} · "
          f"split {_CURRENT_PROTOCOL['split_fp']}")
    cells = resolve_eval_cells(datasets)
    table: Dict[str, Dict[Tuple[str, str], dict]] = {b: {} for b in baselines}
    rejected: List[str] = []

    for b in baselines:
        for ds, stream in cells:
            path = result_path(results_dir, b, ds, stream)
            data = _load_cell(path)
            if data is not None:
                # A cell from a different protocol is worse than a missing cell: it looks valid
                # and silently mixes 59-label and 93-label numbers in one table.
                why = protocol_mismatch(data.get("_protocol"), _CURRENT_PROTOCOL)
                if why:
                    rejected.append(f"STALE    {b}/{ds}/{stream} — {why}")
                    continue
            if data is None:
                rejected.append(f"MISSING  {b}/{ds}/{stream} (no {path.name})")
                continue
            status = data.get("_status")
            if status not in KNOWN_STATUS:
                rejected.append(f"BAD      {b}/{ds}/{stream} (_status={status!r})")
                continue
            if status == "failed":
                rejected.append(
                    f"FAILED   {b}/{ds}/{stream} ({data.get('error', 'no error recorded')})")
                continue
            if status == "complete" and "f1_macro" not in data.get("metrics", {}):
                rejected.append(f"NOMETRIC {b}/{ds}/{stream} (complete but no f1_macro)")
                continue
            table[b][(ds, stream)] = data
    return cells, table, rejected


def _fmt_cell(data: dict) -> Tuple[str, Optional[float]]:
    """Return ``(display, f1_or_None)`` for one cell's JSON."""
    if data.get("_status") == "na":
        return "n/a", None
    m = data["metrics"]
    f1 = m["f1_macro"]
    if m.get("ci_degenerate"):
        return f"{f1:.1f} [degen]", f1
    lo = m.get("f1_macro_ci_lo")
    hi = m.get("f1_macro_ci_hi")
    if lo is None or hi is None:
        return f"{f1:.1f}", f1
    return f"{f1:.1f} [{lo:.1f},{hi:.1f}]", f1


def render(
    baselines: Sequence[str],
    cells: Sequence[Tuple[str, str]],
    table: Dict[str, Dict[Tuple[str, str], dict]],
) -> str:
    counts: Dict[str, int] = {}
    for _ds, _ in cells:
        counts[_ds] = counts.get(_ds, 0) + 1
    multistream = {ds for ds, c in counts.items() if c > 1}
    col_labels = [_col_label(ds, s, multistream) for ds, s in cells]
    T = len(cells)

    lines = []
    hdr = "| Model | " + " | ".join(col_labels) + f" | **mean (k/{T})** |"
    sep = "|" + "---|" * (T + 2)
    lines.append(hdr)
    lines.append(sep)

    for b in baselines:
        row_cells = []
        f1s: List[float] = []
        for cell in cells:
            data = table[b].get(cell)
            if data is None:
                # collect() already rejected loudly; this path is unreachable in a
                # successful assemble, but keep it explicit rather than blank.
                row_cells.append("MISSING")
                continue
            disp, f1 = _fmt_cell(data)
            row_cells.append(disp)
            if f1 is not None:
                f1s.append(f1)
        if f1s:
            mean = sum(f1s) / len(f1s)
            k = len(f1s)
            meancell = f"**{mean:.1f}** ({k}/{T})" + ("" if k == T else " (incomplete)")
        else:
            meancell = "n/a"
        lines.append(f"| {b} | " + " | ".join(row_cells) + f" | {meancell} |")

    lines.append("")
    lines.append("Cells: macro-F1 [subject-stratified 95% CI]. `n/a` = disclosed "
                 "incompatible dataset (excluded from the mean's support k). "
                 "`(incomplete)` mean = averaged over fewer than the full column "
                 "set (an n/a cell) — read per-dataset cells, not the mean.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="baseline rows (default: every baseline with a result JSON)")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS),
                    help="dataset columns (default: PRIMARY_EVAL_DATASETS)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=None, help="also write the markdown table here")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)

    if args.baselines is not None:
        baselines = sorted(args.baselines)
    else:
        # Every baseline that produced at least one result file for these datasets.
        baselines = sorted({p.name.split("__", 1)[0]
                            for p in results_dir.glob("*__*__*.json")})
        if not baselines:
            print(f"!! no result JSONs in {results_dir}; run eval.run_baselines first.")
            return 1

    cells, table, rejected = collect(baselines, args.datasets, results_dir)

    if rejected:
        print("!! INCOMPLETE RESULT GRID — refusing to assemble a table with holes:")
        for r in rejected:
            print(f"   - {r}")
        print(f"\n{len(rejected)} cell(s) missing/failed. Re-run eval.run_baselines "
              "for these before assembling.")
        return 1

    md = render(baselines, cells, table)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
