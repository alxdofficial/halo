"""Estimate conservative Phase-B virtual-subject ranges from native-grid data.

This diagnostic never changes the training recipe. It reports between-subject spread relative to
within-subject spread for pace proxies, dynamic acceleration, gyro energy, and smoothness.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.scripts.eda.grid_io import discover_grids


def window_features(window: np.ndarray, rate_hz: float, mask) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    acc = x[:, :3] if mask[:3].all() else np.zeros((len(x), 3))
    gyro = x[:, 3:] if mask[3:].all() else np.zeros((len(x), 3))
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    centered = acc_mag - acc_mag.mean()
    frequency = np.fft.rfftfreq(len(centered), 1.0 / rate_hz)
    power = np.abs(np.fft.rfft(centered)) ** 2
    valid = (frequency >= 0.3) & (frequency <= min(5.0, rate_hz / 2))
    pace = float(frequency[valid][power[valid].argmax()]) if valid.any() else 0.0
    jerk = np.diff(acc, axis=0) * rate_hz if len(acc) > 1 else np.zeros_like(acc)
    return np.asarray([
        pace,
        float(np.std(acc_mag)),
        float(np.sqrt(np.mean(gyro_mag ** 2))),
        float(np.sqrt(np.mean(jerk ** 2))),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-windows-per-stream", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "outputs/subject_style_calibration.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    by_label_config_subject = defaultdict(list)
    for ref in discover_grids("native"):
        n = min(ref.n_windows, args.max_windows_per_stream)
        rows = rng.choice(ref.n_windows, size=n, replace=False)
        data = ref.load_data()
        for row in rows:
            key = (ref.key, str(ref.labels[row]), str(ref.subjects[row]))
            by_label_config_subject[key].append(
                window_features(data[row], float(ref.rate_hz), ref.mask)
            )
    stratum_subjects = defaultdict(list)
    for (stream, label, _subject), values in by_label_config_subject.items():
        if len(values) >= 2:
            stratum_subjects[(stream, label)].append(np.mean(values, axis=0))
    names = ("pace_hz", "dynamic_acc_std_g", "gyro_rms_rad_s", "jerk_rms_g_s")
    relative = [[] for _ in names]
    usable_strata = 0
    subject_groups = 0
    for subject_values in stratum_subjects.values():
        if len(subject_values) < 3:
            continue
        values = np.stack(subject_values)
        center = np.median(values, axis=0)
        usable_strata += 1
        subject_groups += len(values)
        for i in range(len(names)):
            valid = values[:, i] > 1e-8
            if center[i] > 1e-8 and valid.sum() >= 3:
                relative[i].extend((values[valid, i] / center[i]).tolist())
    if not all(relative):
        raise SystemExit(
            "insufficient repeated subjects per stream/label; increase --max-windows-per-stream"
        )
    payload = {
        "subject_groups": subject_groups,
        "label_configuration_strata": usable_strata,
        "features": {
            name: {
                "relative_p10": float(np.percentile(relative[i], 10)),
                "relative_median": float(np.median(relative[i])),
                "relative_p90": float(np.percentile(relative[i], 90)),
                "observations": len(relative[i]),
            }
            for i, name in enumerate(names)
        },
        "note": (
            "Ratios are subject means divided by the median within the same stream and label. "
            "Use them to review, not automatically widen, subject_style.py."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
