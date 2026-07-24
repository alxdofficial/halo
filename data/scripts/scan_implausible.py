"""Scan the built grids for PHYSICALLY IMPOSSIBLE windows and cache their indices.

Some source releases contain windows whose gyroscope exceeds any consumer MEMS full-scale range.
The widest range shipped on phones/watches is +/-2000 dps (34.907 rad/s); a sample beyond that is
not vigorous motion, it is a scale/acquisition artifact. Training on it teaches the filterbank a
band energy that no real device can produce.

We DROP such windows rather than clip them — clipping fabricates a plausible value for data we know
is wrong. The scan is cached to a small JSON so ``CorpusIndex`` stays lazy (it must not read every
grid at construction time).

Run:  python -m data.scripts.scan_implausible          # writes data/quality/implausible_windows.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Widest consumer gyroscope full-scale range (+/-2000 degrees/second), in rad/s. A window whose
#: |gyro| exceeds this on ANY axis/sample cannot have been produced by the declared hardware.
GYRO_RAIL_RAD_S = float(np.radians(2000.0))          # 34.907 rad/s

OUT = Path(__file__).resolve().parents[2] / "data" / "quality" / "implausible_windows.json"


def scan(alignment: str = "native") -> dict:
    from data.scripts.eda.grid_io import discover_grids
    bad: dict[str, list[int]] = {}
    stats = []
    for ref in sorted(discover_grids(alignment), key=lambda r: r.key):
        mask = np.asarray(ref.mask, dtype=bool)
        if not mask[3:].any():                        # accel-only stream: no gyro to check
            continue
        data = np.asarray(ref.load_data())
        peak = np.abs(data[:, :, 3:]).max(axis=(1, 2))
        idx = np.nonzero(peak > GYRO_RAIL_RAD_S)[0]
        if idx.size:
            bad[ref.key] = sorted(int(i) for i in idx)
            stats.append((ref.key, int(idx.size), ref.n_windows, float(peak.max())))
    return {"alignment": alignment, "gyro_rail_rad_s": GYRO_RAIL_RAD_S,
            "windows": bad, "summary": stats}


def load(alignment: str = "native") -> dict[str, set[int]]:
    """stream key -> set of window indices to exclude (empty if the scan was never run)."""
    if not OUT.exists():
        return {}
    blob = json.loads(OUT.read_text())
    if blob.get("alignment") != alignment:
        return {}
    return {k: set(v) for k, v in blob.get("windows", {}).items()}


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    blob = scan()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    total = sum(len(v) for v in blob["windows"].values())
    print(f"gyro rail = {GYRO_RAIL_RAD_S:.3f} rad/s (2000 dps)")
    for key, n, tot, peak in blob["summary"]:
        print(f"  {key:28s} {n:5d}/{tot:6d} windows exceed the rail (peak {peak:.2f} rad/s)")
    print(f"-> {total} windows cached for exclusion in {OUT}")


if __name__ == "__main__":
    main()
