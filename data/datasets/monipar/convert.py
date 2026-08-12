"""Convert MONIPAR (Zenodo 8104853) to the HALO session format.

Source: `MONIPAR_PD_{SUPERVISED,REMOTE}.mat` + `MONIPAR_HEALTHYCONTROL.mat`, MATLAB cell arrays
indexed [subject, week]. 21 Parkinson's-disease patients + 7 healthy controls each performed the
same 8-exercise protocol **once a week**, usually on the same weekday at a similar hour.

Why this dataset is here (load-bearing): it is the ONLY source in the corpus with verified
**across-session** structure for the same person and the same exercise. Every other rehabilitation
set records one continuous bout per subject per exercise, which makes "enrollment" a within-session
measurement (see docs/data/DATASET_EXPANSION_2026-08.md section 9). Here, one converted session =
one weekly visit, so the grid's execution ids (session-level, see eval/data.py) separate week 1 from
week 5 by seven days. Enrol in week 1, recognize in week N.

Its cost is that the device is **accelerometer-only** — a 3-channel stream through the canonical
6-slot pad+mask.

Raw layout per trial cell, 5 columns (Monipar_README.txt, verified at convert time):
    0  timestamp, milliseconds since trial start
    1  accelerometer x, m/s^2
    2  accelerometer y, m/s^2
    3  accelerometer z, m/s^2
    4  exercise label, 0-8

Labels keep the dataset's own clinical wording; this is an evaluation source, so its concepts must
stay outside the training vocabulary to score as genuinely unseen.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x"
NATIVE_RATE = 50.0
CHANNELS = ("acc_x", "acc_y", "acc_z")

# (file stem, MATLAB variable, subject-id prefix, cohort description)
SUBGROUPS = (
    ("MONIPAR_PD_SUPERVISED", "SUPERVISED_RAWDATA", "sup", "Parkinson's patient, supervised visit"),
    ("MONIPAR_PD_REMOTE", "REMOTE_RAWDATA", "rem", "Parkinson's patient, unsupervised at home"),
    ("MONIPAR_HEALTHYCONTROL", "HEALTHYCONTROL_RAWDATA", "hc", "healthy control"),
)

# Label code -> published wording (Monipar_README.txt).
EXERCISES = {
    0: "postural_transition",
    1: "resting",
    2: "postural_hand_tremor",
    3: "moving_hands_to_the_chest",
    4: "finger_tapping",
    5: "rapid_hand_opening_and_closing",
    6: "pronation_supination",
    7: "arising_from_a_chair",
    8: "gait",
}

# Unit guard. The README documents m/s^2; a still wrist reads ~9.81. g would land near 1.0 and
# milli-g near 1000, and either would silently corrupt HALO's signed DC/gravity feature.
ACC_MS2_RANGE = (8.0, 11.5)      # dataset median |acc|
RATE_RANGE = (45.0, 55.0)        # per-trial mean rate from the real millisecond clock
MAX_GAP_SEC = 1.0                # largest tolerated hole in a trial's clock

# The README says 50 Hz, and the real clock disagrees: measured over all 174 trials the acquisition
# rate is BIMODAL — the supervised subgroup runs at 52.85 Hz and parts of the remote/control
# subgroups at 49.87-52.85 Hz. Treating 52.85 Hz samples as 50 Hz would shift every frequency HALO
# reads by 5.7%, which matters here because the whole point of this dataset is tremor band structure
# (rest tremor 4-6 Hz vs postural/action tremor 6-12 Hz). Each trial is therefore anti-alias
# resampled from its OWN measured rate onto a true 50 Hz grid.
RATE_TOLERANCE_HZ = 0.05         # per-trial clock jitter tolerated before resampling is required


def _trial_rate(ts_ms: np.ndarray) -> float:
    """Mean sampling rate implied by the trial's own millisecond clock."""
    span = (ts_ms[-1] - ts_ms[0]) / 1000.0
    return (len(ts_ms) - 1) / span if span > 0 else float("nan")


def _to_native_rate(acc: np.ndarray, codes: np.ndarray, source_hz: float):
    """Resample one trial from its measured rate to exactly NATIVE_RATE.

    The millisecond clock alternates 19/20 ms because it is a ROUNDED readout of a constant device
    rate, not because acquisition jitters, so the samples are treated as uniform at `source_hz` and
    passed through the same anti-aliased polyphase filter the grid builder uses. Labels are
    categorical and ride along by nearest neighbour.
    """
    if abs(source_hz - NATIVE_RATE) <= RATE_TOLERANCE_HZ:
        return acc.astype(np.float32), codes
    ratio = Fraction(NATIVE_RATE / source_hz).limit_denominator(2000)
    out = resample_poly(acc, up=ratio.numerator, down=ratio.denominator, axis=0)
    index = np.clip(np.round(np.linspace(0, len(codes) - 1, len(out))).astype(int),
                    0, len(codes) - 1)
    return out.astype(np.float32), codes[index]


def create_manifest() -> dict:
    return {
        "dataset_name": "MONIPAR",
        "description": (
            "Weekly at-home and in-clinic monitoring of Parkinson's disease with a TicWatch S2 "
            "consumer smartwatch, worn on the wrist with the greatest presence of motor symptoms "
            "(the dominant hand for controls). 21 PD patients (supervised and remote subgroups) + 7 healthy "
            "controls performed the same 8-exercise MDS-UPDRS-derived protocol once a week for up "
            "to 9 weeks. Triaxial acceleration only, 50 Hz. Source: Zenodo record 8104853."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"acc_{a}", "description":
             f"accelerometer {a}-axis in m/s^2 (gravity present), wrist-worn smartwatch",
             "sampling_rate_hz": NATIVE_RATE} for a in "xyz"
        ],
        "subjects": 28,
        "activities": sorted(EXERCISES.values()),
        # Patients used the more-affected wrist; controls used their dominant wrist. The shared
        # acquisition configuration therefore has no single impairment-side description.
        "placements": ["the wrist"],
        "device_profile": "watch",
        "gravity_state": "present",
        "license": "see the Zenodo record (8104853)",
        "citation": "MONIPAR database v1.0, Zenodo, doi:10.5281/zenodo.8104853",
        "enrollment_note": (
            "One session = one weekly visit, so sessions of the same subject and exercise are "
            "genuinely independent occasions one week apart. This is the corpus's only verified "
            "across-session enrollment source."
        ),
        "sensor_note": "Accelerometer only — no gyroscope. Emitted as a 3-channel stream.",
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw .mat files not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    rates: list[float] = []
    gaps: list[float] = []
    n_rows = 0
    n_subjects = 0

    for stem, variable, prefix, _cohort in SUBGROUPS:
        path = RAW / f"{stem}.mat"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        cells = sio.loadmat(path)[variable]
        n_subjects += cells.shape[0]
        for subject_index in range(cells.shape[0]):
            subject = f"{prefix}{subject_index + 1:02d}"
            for week_index in range(cells.shape[1]):
                trial = cells[subject_index, week_index]
                if np.size(trial) == 0:
                    continue                      # that subject missed that week
                trial = np.asarray(trial, dtype=np.float64)
                if trial.shape[1] != 5:
                    raise ValueError(f"{subject} week {week_index + 1}: expected 5 columns, "
                                     f"got {trial.shape[1]}")
                ts_ms, acc, codes = trial[:, 0], trial[:, 1:4], trial[:, 4]
                if not np.isfinite(trial).all():
                    raise ValueError(f"{subject} week {week_index + 1}: non-finite samples")
                if not np.all(np.diff(ts_ms) > 0):
                    raise ValueError(f"{subject} week {week_index + 1}: clock is not monotonic")
                unknown = set(np.unique(codes).astype(int)) - set(EXERCISES)
                if unknown:
                    raise ValueError(f"{subject} week {week_index + 1}: unknown label codes {unknown}")

                source_hz = _trial_rate(ts_ms)
                rates.append(source_hz)
                gaps.append(float(np.max(np.diff(ts_ms)) / 1000.0))
                magnitudes.append(float(np.median(np.linalg.norm(acc, axis=1))))
                acc, codes = _to_native_rate(acc, codes, source_hz)

                session_id = f"{subject}_w{week_index + 1:02d}"
                frame = pd.DataFrame({
                    "timestamp_sec": np.arange(len(acc), dtype=np.float64) / NATIVE_RATE,
                    **{f"acc_{a}": acc[:, i].astype(np.float32) for i, a in enumerate("xyz")},
                })
                # Per-SAMPLE labels: one weekly trial walks through all 8 exercises in sequence,
                # so a session-level label would be a lie. The assembler majority-votes per window.
                frame["activity"] = [EXERCISES[int(c)] for c in codes]
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)

                labels[session_id] = sorted({EXERCISES[int(c)] for c in np.unique(codes)})
                n_rows += len(frame)

    reference = float(np.median(magnitudes))
    if not ACC_MS2_RANGE[0] <= reference <= ACC_MS2_RANGE[1]:
        raise ValueError(
            f"dataset median |acc| = {reference:.3f}, outside the m/s^2 range {ACC_MS2_RANGE}. The "
            "source units changed (g would be ~1.0, milli-g ~1000); do not silently rescale.")
    mean_rate = float(np.median(rates))
    if not RATE_RANGE[0] <= min(rates) and max(rates) <= RATE_RANGE[1]:
        raise ValueError(f"trial rates span {min(rates):.2f}-{max(rates):.2f} Hz, outside {RATE_RANGE}")
    worst_gap = float(np.max(gaps))
    if worst_gap > MAX_GAP_SEC:
        raise ValueError(f"largest clock gap {worst_gap:.2f} s exceeds {MAX_GAP_SEC} s; the "
                         "converter would have to split sessions at acquisition holes.")

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "monipar", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "eval_labels.json").write_text(json.dumps(
        {"labels": sorted(EXERCISES.values())}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[monipar] units verified: median |acc| {reference:.3f} m/s^2 (gravity present); source "
          f"clock {min(rates):.2f}-{max(rates):.2f} Hz (median {mean_rate:.2f}) resampled to "
          f"{NATIVE_RATE:g} Hz; largest clock gap {worst_gap:.3f} s", flush=True)
    print(f"[monipar] {len(labels)} weekly sessions, {n_subjects} subjects, {n_rows:,} samples, "
          f"{hours:.2f} h, {len(EXERCISES)} exercises -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
