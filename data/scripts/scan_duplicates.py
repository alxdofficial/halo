"""Scan the built grids for BYTE-IDENTICAL repeated windows and cache their indices.

Two windows of real motion are never bit-for-bit equal. When they are, the source device
re-emitted a stale buffer instead of sampling. ExtraSensory's Pebble watch does this in long
runs: subject ``0A986513`` has one group of 178 identical recordings spanning 10,620 s, and
subject ``3600D531`` is 87.7% duplicates. NHANES does it too, but benignly — its repeats are
motionless non-wear blocks quantised to the same 0.001 g triple.

Two rules, because the two cases differ in what we can still trust:

* **All members share one label** -> exactly one of them is a genuine observation. Keep the
  first, drop the rest. The signal is real; only the repeats are fabricated.
* **Members disagree on the label** -> the buffer went stale across a label change, so the
  window is real but its label is unknowable. Drop the WHOLE group. Keeping an arbitrary
  member injects a known-wrong label at least half the time. On ExtraSensory's wrist stream
  this is 36.9% of duplicate pairs, dominated by lying<->sitting (4,413 pairs).

Duplicates are only ever matched WITHIN one stream. Simultaneous multi-placement streams
(xrf_v2, nfi_fared) are meant to describe the same event from different sensors.

Like ``scan_implausible``, results are cached to a small JSON so ``CorpusIndex`` stays lazy.

Run:  python -m data.scripts.scan_duplicates          # writes data/quality/duplicate_windows.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[2] / "data" / "quality" / "duplicate_windows.json"

#: Windows are hashed in blocks so a 4 GB grid is never fully resident.
BLOCK = 4096


def scan_stream(data, labels) -> tuple[list[int], int, int]:
    """Return (indices to drop, n duplicate groups, n groups dropped whole for label conflict)."""
    groups: dict[bytes, list[int]] = {}
    n = data.shape[0]
    for start in range(0, n, BLOCK):
        block = np.asarray(data[start:start + BLOCK], dtype=np.float32)
        for j, window in enumerate(block):
            digest = hashlib.blake2b(window.tobytes(), digest_size=16).digest()
            groups.setdefault(digest, []).append(start + j)

    drop: list[int] = []
    n_groups = n_conflict = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        if len({labels[i] for i in members}) > 1:    # stale across a label change
            n_conflict += 1
            drop.extend(members)                     # label unknowable -> drop the whole group
        else:
            drop.extend(members[1:])                 # keep one genuine observation
    return sorted(drop), n_groups, n_conflict


def scan(alignment: str = "native") -> dict:
    from data.scripts.eda.grid_io import discover_grids, grid_corpus_fingerprint

    refs = discover_grids(alignment)
    bad: dict[str, list[int]] = {}
    stats = []
    for ref in sorted(refs, key=lambda r: r.key):
        if ref.n_windows == 0:
            continue
        drop, n_groups, n_conflict = scan_stream(ref.load_data(), ref.labels)
        if drop:
            bad[ref.key] = drop
            stats.append((ref.key, len(drop), ref.n_windows, n_groups, n_conflict))
    return {"alignment": alignment, "grid_fingerprint": grid_corpus_fingerprint(alignment, refs),
            "windows": bad, "summary": stats}


def load(alignment: str = "native", *, require: bool = False) -> dict[str, set[int]]:
    """stream key -> set of window indices to exclude.

    ``require=True`` refuses to return an empty screen. A missing or wrong-alignment cache is
    indistinguishable from "this corpus has no duplicates", so on a remote snapshot that shipped
    without ``data/quality/duplicate_windows.json`` training would silently readmit every stale
    ExtraSensory buffer. Callers that depend on the screen should pass require=True.
    """
    if not OUT.exists():
        if require:
            raise FileNotFoundError(
                f"{OUT} is missing — run `python -m data.scripts.scan_duplicates`. Training "
                "without it silently readmits byte-identical stale-buffer windows."
            )
        return {}
    blob = json.loads(OUT.read_text())
    if blob.get("alignment") != alignment:
        if require:
            raise ValueError(
                f"{OUT} was built for alignment {blob.get('alignment')!r}, not {alignment!r}; "
                "re-run data.scripts.scan_duplicates for this alignment."
            )
        return {}
    from data.scripts.eda.grid_io import grid_corpus_fingerprint
    current = grid_corpus_fingerprint(alignment)
    if blob.get("grid_fingerprint") != current:
        if require:
            raise ValueError(
                f"{OUT} does not match the current {alignment} grids; re-run "
                "`python -m data.scripts.scan_duplicates` after every grid rebuild."
            )
        return {}
    return {k: set(v) for k, v in blob.get("windows", {}).items()}


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    blob = scan()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    total = sum(len(v) for v in blob["windows"].values())
    for key, n, tot, groups, conflict in blob["summary"]:
        print(f"  {key:28s} {n:6d}/{tot:7d} windows dropped "
              f"({n / tot:5.1%}) · {groups} groups, {conflict} label-conflicted")
    print(f"-> {total} duplicate windows cached for exclusion in {OUT}")


if __name__ == "__main__":
    main()
