"""Convert PHYTMO (Zenodo 6319979) to the HALO session format.

Source: `inertial/{upper,lower}/{A..E}/<segment>/<name>.csv`. 30 volunteers aged 20-69 performed
6 physiotherapy exercises and 3 gait variations, each **twice correctly and twice incorrectly**,
wearing four magneto-inertial units.

Everything below is taken from the dataset paper (Garcia-de-Villa, Jimenez-Martin &
Garcia-Dominguez, *Scientific Data* 9:266, 2022) rather than inferred:

  * **Filename** `GNNEEELP_S`: `G` = age group, `NN` = subject, `EEE` = exercise, `L` = limb (only
    on the lateral exercises), `P` = **0 correctly performed / 1 wrongly performed**, `S` = series
    (1 or 2). `*Calib.csv`, `*LCalib.csv`, `*RCalib.csv` are calibration motions and are skipped.
  * **Age groups**: A 20-29, B 30-39, C 40-49, D 50-59, E 60-69.
  * **Placements**: lower-limb recordings use the anterior surface of both shins and both thighs;
    upper-limb recordings use the exterior lateral surface of both arms and both forearms.
  * **Rate**: 100 Hz for accelerometers and gyroscopes (magnetometers run at 20 Hz and are dropped).
  * **Units**: the CSV header states them per column — `Accelerometer X (g)`,
    `Gyroscope X (deg/s)`, `Magnetometer X (uT)`. Angular rate is converted to rad/s here.

The correct/incorrect distinction is carried IN THE LABEL (`..._performed_incorrectly`), because
that is the discrimination a rehabilitation-monitoring claim actually has to make, and PHYTMO is one
of only two sources in the corpus that annotates it (the other is KneE-PAD). Limb side is in the
label too for the lateral exercises: at a given sensor, a left-knee flexion and a right-knee flexion
are different motions, not the same one.

Upper-limb and lower-limb trials are separate recordings with different sensor sets, so a session
carries either the four upper columns or the four lower ones; the deployment policy routes them by
the `_upper_` / `_lower_` token in the session id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x" / "inertial"
NATIVE_RATE = 100.0
DEG_TO_RAD = np.pi / 180.0

# Directory name -> column prefix. The two sets never appear in the same recording.
SEGMENTS = {
    "upper": {"Rarm": "right_arm", "Larm": "left_arm",
              "Rforearm": "right_forearm", "Lforearm": "left_forearm"},
    "lower": {"Rthigh": "right_thigh", "Lthigh": "left_thigh",
              "Rshin": "right_shin", "Lshin": "left_shin"},
}

# Paper, exercise codes.
EXERCISES = {
    "EFE": "elbow_flexion_extension",
    "EAH": "extension_of_the_arms_over_the_head",
    "SQZ": "squeezing",
    "KFE": "knee_flexion_extension",
    "SQT": "squat",
    "HAA": "hip_abduction",
    "GAT": "natural_gait",
    "GIS": "gait_along_an_infinity_shaped_path",
    "GHT": "heel_to_toe_gait",
}
SIDES = {"L": "left", "R": "right"}
# GIS and GHT carry no correct/incorrect digit: they ARE the gait variations, so there is no
# "wrongly performed" counterpart to record.
NO_QUALITY_FLAG = frozenset({"GIS", "GHT"})

ACC_G_RANGE = (0.7, 1.4)         # median |acc| per file
GYRO_DEGS_MIN = 20.0             # p99.9 |w| BEFORE conversion; rad/s would never reach this
RATE_TOLERANCE_HZ = 2.0
MAX_GAP_SEC = 0.5
# 8 of the 4,520 inertial CSVs contain dropped packets written as all-NaN rows. They are ALL from
# one subject on one exercise (C02, left-knee flexion-extension): the right-thigh unit drops isolated
# single samples (3-4% of rows, harmless), but the left-shin unit is 40% missing in runs of 3-5
# samples. Interpolating 40% of a signal would be fabrication, so a trial is SKIPPED when any of its
# four units exceeds either bound; isolated single-sample dropouts are interpolated. Everything else
# from C02 survives. See docs/data/DATASET_EXPANSION_2026-08.md section 8b.
MAX_NAN_RUN = 2
MAX_NAN_FRACTION = 0.05

_HEADER = ("Time (s)",
           "Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)",
           "Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)")


def parse_name(stem: str) -> tuple[str, str, str, str] | None:
    """`A08SQZ1_2` -> (subject, exercise_code, side, label). None for calibration files."""
    if "Calib" in stem:
        return None
    body, _, series = stem.partition("_")
    group, number, rest = body[0], body[1:3], body[3:]
    subject = f"{group.lower()}{int(number):02d}"
    code, rest = rest[:3], rest[3:]
    if code not in EXERCISES:
        raise ValueError(f"{stem}: unknown exercise code {code!r}")
    side = ""
    if rest and rest[0] in SIDES:
        side, rest = rest[0], rest[1:]
    name = EXERCISES[code]
    if side:
        name = f"{name}_of_the_{SIDES[side]}_limb"
    if code in NO_QUALITY_FLAG:
        if rest:
            raise ValueError(f"{stem}: {code} should carry no correct/incorrect digit, got {rest!r}")
    else:
        if rest not in ("0", "1"):
            raise ValueError(f"{stem}: expected a 0/1 execution digit, got {rest!r}")
        if rest == "1":
            name = f"{name}_performed_incorrectly"
    return subject, code, series or "1", name


def create_manifest() -> dict:
    prefixes = sorted({p for group in SEGMENTS.values() for p in group.values()})
    return {
        "dataset_name": "PHYTMO",
        "description": (
            "Physical therapy monitoring database. 30 volunteers aged 20-69 (five 10-year age "
            "groups) performed 6 prescribed exercises and 3 gait variations, each in two correct "
            "and two deliberately incorrect series of at least 8 repetitions, wearing four "
            "magneto-inertial units at 100 Hz on the limbs, with optical motion-capture reference. "
            "Garcia-de-Villa, Jimenez-Martin & Garcia-Dominguez, Scientific Data 9:266, 2022."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{prefix}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"magneto-inertial unit on the {prefix.replace('_', ' ')}; "
                             f"{'g (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for prefix in prefixes for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 30,
        "activities": [],          # filled in main() from what actually converted
        "placements": prefixes,
        "device_profile": "device",
        "gravity_state": "present",
        "license": "CC-BY (Zenodo record 6319979)",
        "citation": ("Garcia-de-Villa S, Jimenez-Martin A, Garcia-Dominguez JJ. A database of "
                     "physical therapy exercises with variability of execution collected by "
                     "wearable sensors. Scientific Data 9:266, 2022. doi:10.1038/s41597-022-01387-2"),
        "unit_note": "Source gyroscope is deg/s and is converted to rad/s in this converter. "
                     "Magnetometer columns (20 Hz) are dropped.",
        "execution_note": (
            "The filename's 0/1 digit (correct vs wrongly performed) is carried into the LABEL, so "
            "an exercise and its incorrect variant are distinct concepts."
        ),
        "age_groups": {"a": "20-29", "b": "30-39", "c": "40-49", "d": "50-59", "e": "60-69"},
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw inertial tree not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks_deg: list[float] = []
    rates: list[float] = []
    incomplete = 0
    nan_rows = 0
    dropouts: list[str] = []
    n_rows = 0
    subjects: set[str] = set()

    for limb, segments in SEGMENTS.items():
        # Enumerate trials from the first segment, then require the other three to exist.
        anchor = next(iter(segments))
        for group_dir in sorted((RAW / limb).glob("[A-E]")):
            for path in sorted((group_dir / anchor).glob("*.csv")):
                parsed = parse_name(path.stem)
                if parsed is None:
                    continue
                subject, _code, series, activity = parsed
                paths = {prefix: group_dir / directory / path.name
                         for directory, prefix in segments.items()}
                if not all(p.exists() for p in paths.values()):
                    incomplete += 1
                    continue

                frame = None
                clock = None
                rejected = ""
                for prefix, source in paths.items():
                    table = pd.read_csv(source)
                    missing = [c for c in _HEADER if c not in table.columns]
                    if missing:
                        raise ValueError(f"{source}: missing columns {missing}")
                    values = table[list(_HEADER)].to_numpy(dtype=np.float64)
                    holes = ~np.isfinite(values[:, 1:]).all(axis=1)
                    if holes.any():
                        runs = np.diff(np.r_[0, np.flatnonzero(np.diff(holes.astype(int)) != 0) + 1,
                                             len(holes)])
                        longest = int(max(r for r, h in zip(runs, holes[np.r_[
                            0, np.cumsum(runs)[:-1]]]) if h))
                        share = float(holes.mean())
                        if longest > MAX_NAN_RUN or share > MAX_NAN_FRACTION:
                            rejected = (f"{prefix}: {100 * share:.1f}% missing, longest run "
                                        f"{longest}")
                            break
                        nan_rows += int(holes.sum())
                        good = ~holes
                        for column in range(1, values.shape[1]):
                            values[holes, column] = np.interp(
                                values[holes, 0], values[good, 0], values[good, column])
                    if not np.isfinite(values).all():
                        raise ValueError(f"{source}: non-finite samples remain after interpolation")
                    if clock is None:
                        clock = values[:, 0]
                        steps = np.diff(clock)
                        rates.append(1.0 / float(np.median(steps)))
                        if float(np.max(steps)) > MAX_GAP_SEC:
                            raise ValueError(f"{source}: clock gap "
                                             f"{float(np.max(steps)):.2f}s exceeds {MAX_GAP_SEC}s")
                        frame = pd.DataFrame({"timestamp_sec": clock - clock[0]})
                    n = min(len(frame), len(values))
                    if len(values) != len(frame):
                        # The four units are logged independently; trim to the shortest.
                        frame = frame.iloc[:n].reset_index(drop=True)
                    magnitudes.append(float(np.median(np.linalg.norm(values[:n, 4:7], axis=1))))
                    gyro_peaks_deg.append(float(np.percentile(np.abs(values[:n, 1:4]), 99.9)))
                    for offset, axis in enumerate("xyz"):
                        frame[f"{prefix}_acc_{axis}"] = values[:n, 4 + offset].astype(np.float32)
                    for offset, axis in enumerate("xyz"):
                        frame[f"{prefix}_gyro_{axis}"] = (values[:n, 1 + offset] * DEG_TO_RAD
                                                          ).astype(np.float32)

                if rejected:
                    dropouts.append(f"{path.stem} ({rejected})")
                    continue
                session_id = f"{subject}_{limb}_{activity}_s{series}"
                frame["activity"] = activity
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)
                labels[session_id] = [activity]
                subjects.add(subject)
                n_rows += len(frame)
            print(f"  {limb}/{group_dir.name}: done", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_G_RANGE[0] <= reference <= ACC_G_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the g range {ACC_G_RANGE}; the "
                         "CSV header documents g.")
    peak_deg = float(np.max(gyro_peaks_deg))
    if peak_deg < GYRO_DEGS_MIN:
        raise ValueError(
            f"raw |gyro| p99.9 max = {peak_deg:.1f} never exceeds {GYRO_DEGS_MIN}; the source looks "
            "like rad/s now, so the deg/s -> rad/s conversion here would shrink it 57x.")
    measured_rate = float(np.median(rates))
    if abs(measured_rate - NATIVE_RATE) > RATE_TOLERANCE_HZ:
        raise ValueError(f"median clock rate {measured_rate:.3f} Hz != documented {NATIVE_RATE} Hz")

    activities = sorted({v[0] for v in labels.values()})
    manifest = create_manifest()
    manifest["activities"] = activities
    manifest["subjects"] = len(subjects)
    manifest["excluded_trials"] = sorted(dropouts)
    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "phytmo", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "eval_labels.json").write_text(json.dumps({"labels": activities}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[phytmo] units verified: median |acc| {reference:.3f} g (gravity present), raw |gyro| "
          f"p99.9 max {peak_deg:.1f} deg/s -> {peak_deg * DEG_TO_RAD:.2f} rad/s, clock "
          f"{measured_rate:.2f} Hz", flush=True)
    print(f"[phytmo] {len(labels)} sessions, {len(subjects)} subjects, {n_rows:,} samples, "
          f"{hours:.2f} h, {len(activities)} labels; {incomplete} trials skipped for an incomplete "
          f"4-sensor set, {len(dropouts)} for excessive sensor dropout, {nan_rows:,} isolated "
          f"dropped-packet rows interpolated -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
