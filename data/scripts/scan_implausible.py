"""Scan the built grids for PHYSICALLY IMPOSSIBLE windows and cache their indices.

Some source releases contain windows beyond any consumer MEMS full-scale range. The widest ranges
shipped on phones/watches are +/-2000 dps (34.907 rad/s) and +/-16 g; a sample beyond either is not
vigorous motion, it is a scale/acquisition artifact. Training on it teaches the filterbank a band
energy that no real device can produce. BOTH modalities are screened, so accelerometer-only streams
(capture24, extrasensory, nhanes, unimib_shar) are covered — they used to be skipped entirely.

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

#: Widest consumer accelerometer full-scale range (+/-16 g). Same argument as the gyro rail: a
#: sample beyond it is a scale/acquisition artifact, not motion. This is the only screen an
#: ACCEL-ONLY stream can get, and until it existed those streams (capture24, extrasensory, nhanes,
#: unimib_shar) were passed through entirely unscreened.
ACCEL_RAIL_G = 16.0

OUT = Path(__file__).resolve().parents[2] / "data" / "quality" / "implausible_windows.json"


def scan(alignment: str = "native") -> dict:
    from data.scripts.eda.grid_io import discover_grids
    bad: dict[str, list[int]] = {}
    stats = []
    for ref in sorted(discover_grids(alignment), key=lambda r: r.key):
        if ref.n_windows == 0:
            continue
        mask = np.asarray(ref.mask, dtype=bool)
        data = ref.load_data()
        over = np.zeros(ref.n_windows, dtype=bool)
        gyro_peak = accel_peak = 0.0
        # Block-wise so a multi-GB grid is never fully resident (capture24 is 1.5M windows).
        for start in range(0, ref.n_windows, 4096):
            block = np.asarray(data[start:start + 4096], dtype=np.float32)
            a_peak = np.abs(block[:, :, :3]).max(axis=(1, 2))
            accel_peak = max(accel_peak, float(a_peak.max()))
            hit = a_peak > ACCEL_RAIL_G
            if mask[3:].any():                        # accel-only streams have no gyro to check
                g_peak = np.abs(block[:, :, 3:]).max(axis=(1, 2))
                gyro_peak = max(gyro_peak, float(g_peak.max()))
                hit |= g_peak > GYRO_RAIL_RAD_S
            over[start:start + len(hit)] = hit
        idx = np.nonzero(over)[0]
        if idx.size:
            bad[ref.key] = sorted(int(i) for i in idx)
            stats.append((ref.key, int(idx.size), ref.n_windows, gyro_peak, accel_peak))
    return {"alignment": alignment, "gyro_rail_rad_s": GYRO_RAIL_RAD_S,
            "accel_rail_g": ACCEL_RAIL_G, "windows": bad, "summary": stats}


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
    print(f"gyro rail = {GYRO_RAIL_RAD_S:.3f} rad/s (2000 dps) · accel rail = {ACCEL_RAIL_G} g")
    for key, n, tot, gyro_peak, accel_peak in blob["summary"]:
        print(f"  {key:28s} {n:5d}/{tot:6d} windows exceed a rail "
              f"(peak gyro {gyro_peak:.2f} rad/s, peak |accel| {accel_peak:.2f} g)")
    print(f"-> {total} windows cached for exclusion in {OUT}")


if __name__ == "__main__":
    main()
