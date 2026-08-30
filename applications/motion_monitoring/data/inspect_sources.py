#!/usr/bin/env python3
"""Empirically validate acquired application datasets and record their contracts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from applications.motion_monitoring.data.adapters.recofit import (
    iter_recordings as iter_recofit,
)


HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
HALO_DATASETS = HERE.parents[2] / "data" / "datasets"
OUTPUT = HERE / "inspection" / "summary.json"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def describe(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def read_xlsx_rows(path: Path) -> list[list[str | float | None]]:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall(f"{NS}si")]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(f".//{NS}row"):
        values: dict[int, str | float | None] = {}
        for cell in row.findall(f"{NS}c"):
            ref = cell.attrib["r"]
            letters = re.match(r"[A-Z]+", ref).group(0)
            column = 0
            for letter in letters:
                column = column * 26 + ord(letter) - ord("A") + 1
            value_node = cell.find(f"{NS}v")
            if value_node is None:
                value: str | float | None = None
            elif cell.attrib.get("t") == "s":
                value = shared[int(value_node.text)]
            else:
                value = float(value_node.text)
            values[column - 1] = value
        rows.append([values.get(index) for index in range(max(values, default=-1) + 1)])
    return rows


def inspect_c_mhad() -> dict[str, Any]:
    root = SOURCES / "c_mhad" / "raw"
    csv_paths = sorted(root.rglob("*.csv"))
    workbooks = sorted(root.rglob("*.xlsx"))
    row_counts, durations, rates, acc_norms, gyro_abs = [], [], [], [], []
    duplicate_steps = 0
    nonfinite = 0
    for path in csv_paths:
        frame = pd.read_csv(path, skiprows=[1, 2])
        values = frame.to_numpy(dtype=np.float64)
        nonfinite += int((~np.isfinite(values)).sum())
        timestamp = values[:, 0] / 1000.0
        delta = np.diff(timestamp)
        duplicate_steps += int((delta <= 0).sum())
        row_counts.append(len(frame))
        durations.append(float(timestamp[-1] - timestamp[0]))
        rates.append(float(1.0 / np.median(delta[delta > 0])))
        acc_norms.append(float(np.median(np.linalg.norm(values[:, 1:4], axis=1))))
        gyro_abs.append(float(np.quantile(np.abs(values[:, 4:7]), 0.99)))

    event_counts: Counter[str] = Counter()
    event_durations: list[float] = []
    invalid_intervals = 0
    label_maps = {
        "TVGestureApplication": {
            1: "swipe_left",
            2: "swipe_right",
            3: "wave",
            4: "circle_clockwise",
            5: "circle_counterclockwise",
        },
        "TransitionMovementsApplication": {
            1: "stand_to_sit",
            2: "sit_to_stand",
            3: "sit_to_lie",
            4: "lie_to_sit",
            5: "lie_to_stand",
            6: "stand_to_lie",
            7: "stand_to_fall",
        },
    }
    for path in workbooks:
        rows = read_xlsx_rows(path)
        if rows[0] != ["Video", "Action", "StartTime(Seconds)", "EndTime(Seconds)"]:
            raise ValueError(f"unexpected C-MHAD workbook header in {path}")
        application = path.parts[-3]
        for row in rows[1:]:
            if len(row) < 4 or row[0] is None:
                continue
            label_id = int(row[1])
            start, end = float(row[2]), float(row[3])
            label = label_maps[application].get(label_id)
            if label is None or start < 0 or end <= start or end > 121:
                invalid_intervals += 1
                continue
            event_counts[label] += 1
            event_durations.append(end - start)
    return {
        "status": "pass",
        "files": {"csv": len(csv_paths), "annotation_workbooks": len(workbooks)},
        "rows_per_stream": describe(row_counts),
        "duration_seconds": describe(durations),
        "measured_rate_hz": describe(rates),
        "median_acceleration_norm_m_s2": describe(acc_norms),
        "gyro_abs_q99_deg_s": describe(gyro_abs),
        "nonfinite_values": nonfinite,
        "non_increasing_timestamp_steps": duplicate_steps,
        "event_counts": dict(sorted(event_counts.items())),
        "event_duration_seconds": describe(event_durations),
        "invalid_annotation_intervals": invalid_intervals,
        "conversion": "acceleration / 9.80665 to g; gyroscope * pi / 180 to rad/s",
    }


def inspect_wear() -> dict[str, Any]:
    root = SOURCES / "wear" / "raw"
    paths = sorted((root / "inertial_50hz").glob("*.csv"))
    rows, durations, null_fractions, norms = [], [], [], []
    labels: Counter[str] = Counter()
    nonfinite = 0
    missing_by_column: Counter[str] = Counter()
    id_mismatch = 0
    expected_columns = [
        "sbj_id",
        "right_arm_acc_x",
        "right_arm_acc_y",
        "right_arm_acc_z",
        "right_leg_acc_x",
        "right_leg_acc_y",
        "right_leg_acc_z",
        "left_leg_acc_x",
        "left_leg_acc_y",
        "left_leg_acc_z",
        "left_arm_acc_x",
        "left_arm_acc_y",
        "left_arm_acc_z",
        "label",
    ]
    for path in paths:
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        if list(frame.columns) != expected_columns:
            raise ValueError(f"unexpected WEAR columns in {path}")
        file_id = int(re.search(r"\d+", path.stem).group(0))
        if set(frame["sbj_id"].unique()) != {file_id}:
            id_mismatch += 1
        numeric_frame = frame.iloc[:, 1:13].apply(pd.to_numeric, errors="coerce")
        missing_by_column.update(
            {
                column: int(count)
                for column, count in numeric_frame.isna().sum().items()
                if count
            }
        )
        numeric = numeric_frame.to_numpy(dtype=np.float64)
        nonfinite += int((~np.isfinite(numeric)).sum())
        for start in (0, 3, 6, 9):
            device = numeric[:, start : start + 3]
            valid = np.isfinite(device).all(axis=1)
            if valid.any():
                norms.append(float(np.median(np.linalg.norm(device[valid], axis=1))))
        rows.append(len(frame))
        durations.append(len(frame) / 50.0)
        counts = frame["label"].value_counts(dropna=False)
        labels.update({str(key): int(value) for key, value in counts.items()})
        null_fractions.append(float((frame["label"] == "null").mean()))

    annotation_paths = sorted((root / "annotations_60fps").glob("*.json"))
    split = json.loads((root / "annotations_60fps" / "wear_split_1.json").read_text())
    test = json.loads((root / "annotations_60fps" / "wear_test_2.json").read_text())
    interval_labels = sorted(split["label_dict"], key=split["label_dict"].get)
    invalid_intervals = 0
    for payload in (split, test):
        for recording in payload["database"].values():
            duration = recording["duration"]
            for annotation in recording["annotations"]:
                start, end = annotation["segment"]
                invalid_intervals += int(
                    start < 0 or end <= start or end > duration + 0.05
                )
    return {
        "status": "pass_with_identity_alias_and_missing_channel_mask_required",
        "files": {"inertial_csv": len(paths), "annotation_json": len(annotation_paths)},
        "published_unique_participants": 22,
        "released_recording_files": len(paths),
        "identity_aliases": {
            "sbj_18": "second_session_of_sbj_0",
            "sbj_19": "second_session_of_sbj_14",
        },
        "new_test_participants": ["sbj_20", "sbj_21", "sbj_22", "sbj_23"],
        "rows_per_recording": describe(rows),
        "duration_seconds": describe(durations),
        "null_fraction": describe(null_fractions),
        "median_acceleration_norm_g_across_devices": describe(norms),
        "labels": dict(sorted(labels.items())),
        "interval_label_order": interval_labels,
        "nonfinite_values": nonfinite,
        "missing_values_by_column": dict(sorted(missing_by_column.items())),
        "subject_id_mismatches": id_mismatch,
        "invalid_annotation_intervals": invalid_intervals,
        "timing": "fixed 50 Hz; derive time from row index because CSV has no timestamp",
    }


def _edf_text(data: bytes, offset: int, width: int) -> str:
    return data[offset : offset + width].decode("ascii").strip()


def read_edf_header(data: bytes) -> dict[str, Any]:
    header_bytes = int(_edf_text(data, 184, 8))
    records = int(_edf_text(data, 236, 8))
    record_duration = float(_edf_text(data, 244, 8))
    signals = int(_edf_text(data, 252, 4))
    offset = 256
    fields = []
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        fields.append(
            [
                data[offset + index * width : offset + (index + 1) * width]
                .decode("ascii")
                .strip()
                for index in range(signals)
            ]
        )
        offset += signals * width
    (
        labels,
        _,
        dimensions,
        physical_min,
        physical_max,
        digital_min,
        digital_max,
        _,
        samples,
        _,
    ) = fields
    return {
        "header_bytes": header_bytes,
        "records": records,
        "record_duration": record_duration,
        "labels": labels,
        "dimensions": dimensions,
        "physical_min": [float(value) for value in physical_min],
        "physical_max": [float(value) for value in physical_max],
        "digital_min": [float(value) for value in digital_min],
        "digital_max": [float(value) for value in digital_max],
        "samples": [int(value) for value in samples],
    }


def inspect_aidlab_har() -> dict[str, Any]:
    path = SOURCES / "aidlab_har" / "downloads" / "AIDLAB-HAR-DATASET_v3.zip"
    activities: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    source_codes = set()
    durations = []
    rates = set()
    channel_contracts = set()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        edf_names = sorted(name for name in names if name.endswith(".edf"))
        annotation_names = sorted(
            name for name in names if "/data/" in name and name.endswith(".csv")
        )
        for name in edf_names:
            match = re.search(r"(SUB\d+)_(.+)_S\d+\.edf$", name)
            source_codes.add(match.group(1))
            activities[match.group(2).lower()] += 1
            header = read_edf_header(archive.read(name))
            durations.append(header["records"] * header["record_duration"])
            rates.update(
                round(count / header["record_duration"], 6)
                for count in header["samples"][:7]
            )
            channel_contracts.add(
                tuple(zip(header["labels"][:7], header["dimensions"][:7]))
            )
        for name in annotation_names:
            rows = csv.DictReader(archive.read(name).decode("utf-8").splitlines())
            event_counts.update(row["event"] for row in rows)
    return {
        "status": "pass_with_annotation_limit",
        "edf_recordings": len(edf_names),
        "annotation_csv": len(annotation_names),
        "source_subject_codes": len(source_codes),
        "subject_identity_warning": "SUBxx codes are not globally unique participant identities.",
        "activity_recordings": dict(sorted(activities.items())),
        "duration_seconds": describe(durations),
        "sample_rates_hz": sorted(rates),
        "channel_contracts": [list(contract) for contract in sorted(channel_contracts)],
        "annotation_events": dict(sorted(event_counts.items())),
        "annotation_warning": "repetition_marker intervals are fiducial windows, not complete repetitions",
        "adapter": "acceleration-only; quaternion is metadata and raw gyroscope is absent",
    }


def inspect_oca() -> dict[str, Any]:
    path = SOURCES / "oca" / "downloads" / "OCA.zip"
    rows, durations, rates, acc_norms, gyro_q99 = [], [], [], [], []
    gyro_dps_quantization_residual = []
    labels: Counter[int] = Counter()
    gaps = []
    nonfinite = 0
    with zipfile.ZipFile(path) as archive:
        csv_names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        metadata = json.loads(archive.read("metadata.json"))
        for name in csv_names:
            frame = pd.read_csv(archive.open(name))
            values = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
            nonfinite += int((~np.isfinite(values)).sum())
            timestamp = frame["timestamp"].to_numpy(dtype=np.float64) / 1000.0
            delta = np.diff(timestamp)
            positive = delta[delta > 0]
            rows.append(len(frame))
            durations.append(float(timestamp[-1] - timestamp[0]))
            rates.append(float(1.0 / np.median(positive)))
            for index in range(4):
                start = 1 + index * 6
                acc_norms.append(
                    float(
                        np.median(np.linalg.norm(values[:, start : start + 3], axis=1))
                    )
                )
                gyro = values[:, start + 3 : start + 6]
                gyro_q99.append(float(np.quantile(np.abs(gyro), 0.99)))
                # BNO055 degree/s output uses 16 LSB per degree/s. This exact
                # release-wide quantization disambiguates it from rad/s mode.
                gyro_dps_quantization_residual.append(
                    float(np.max(np.abs(gyro * 16.0 - np.rint(gyro * 16.0))))
                )
            for gap in delta[delta > 2.0]:
                gaps.append({"file": name, "seconds": float(gap)})
            labels.update(int(value) for value in frame["label"])
    split_files = {
        split: sorted(files) for split, files in metadata["benchmark_splits"].items()
    }
    return {
        "status": "pass_with_clock_gap_split_required",
        "files": len(csv_names),
        "rows_per_session": describe(rows),
        "duration_seconds": describe(durations),
        "measured_rate_hz": describe(rates),
        "median_acceleration_norm_m_s2": describe(acc_norms),
        "gyro_abs_q99_deg_s": describe(gyro_q99),
        "gyro_dps_1_over_16_quantization_max_residual": max(
            gyro_dps_quantization_residual
        ),
        "nonfinite_values": nonfinite,
        "timestamp_gaps_over_2s": gaps,
        "label_rows": {str(key): value for key, value in sorted(labels.items())},
        "official_splits": split_files,
        "conversion": "acceleration / 9.80665 to g; gyroscope * pi / 180 to rad/s",
    }


def inspect_crossfit() -> dict[str, Any]:
    root = (
        SOURCES
        / "crossfit"
        / "raw"
        / "HAR_Crossfit_Sensors_Data"
        / "data"
        / "constrained_workout"
        / "preprocessed_numpy_data"
    )
    exercise_paths = sorted((root / "np_exercise_data").rglob("*.npy"))
    all_repetition_paths = sorted((root / "np_reps_data").rglob("*.npy"))
    repetition_paths = [
        path for path in all_repetition_paths if path.parent.name != "Null"
    ]
    null_repetition_paths = [
        path for path in all_repetition_paths if path.parent.name == "Null"
    ]
    participant_map_path = (
        root.parents[3] / "HAR_Crossfit_Sensors_Code" / "participant_ex_code_map.txt"
    )
    participant_map = json.loads(participant_map_path.read_text())
    index_to_participant = {
        int(index): participant
        for participant, indices in participant_map.items()
        for index in indices
    }
    unmapped_exercise_indices = []
    participants_with_non_null_exercise = set()
    participants_with_null_only = set()
    exercise_shapes, repetition_lengths = [], []
    imu_nonfinite = 0
    orientation_nonfinite = 0
    acc_norms, gyro_q99 = [], []
    imu_rows = np.r_[0:6, 9:15]
    orientation_rows = np.r_[6:9, 15:18]
    for path in exercise_paths:
        array = np.load(path, mmap_mode="r")
        exercise_shapes.append(list(array.shape))
        exercise_index = int(path.stem.rsplit("_", 1)[-1])
        if exercise_index not in index_to_participant:
            unmapped_exercise_indices.append(exercise_index)
        elif path.parent.name == "Null":
            participants_with_null_only.add(index_to_participant[exercise_index])
        else:
            participants_with_non_null_exercise.add(
                index_to_participant[exercise_index]
            )
        if array.ndim != 2 or array.shape[0] != 18:
            continue
        imu_nonfinite += int((~np.isfinite(array[imu_rows])).sum())
        orientation_nonfinite += int((~np.isfinite(array[orientation_rows])).sum())
        for offset in (0, 9):
            acc_norms.append(
                float(np.median(np.linalg.norm(array[offset : offset + 3].T, axis=1)))
            )
            gyro_q99.append(
                float(np.quantile(np.abs(array[offset + 3 : offset + 6]), 0.99))
            )
    for path in repetition_paths:
        array = np.load(path, mmap_mode="r")
        repetition_lengths.append(int(array.shape[-1]))
        if array.ndim == 2 and array.shape[0] == 18:
            imu_nonfinite += int((~np.isfinite(array[imu_rows])).sum())
            orientation_nonfinite += int((~np.isfinite(array[orientation_rows])).sum())
    return {
        "status": "pass_with_identity_count_and_short_fragment_caveats",
        "publication_participants": 54,
        "released_participant_codes": len(participant_map),
        "participant_codes_with_non_null_exercise": len(
            participants_with_non_null_exercise
        ),
        "participant_codes_with_null_only": sorted(
            participants_with_null_only - participants_with_non_null_exercise
        ),
        "participant_identity_warning": "the released participant map has 57 codes, while the paper reports 54 participants; freeze source-code splits and do not infer demographics or distinct-person counts from code names alone",
        "exercise_arrays": len(exercise_paths),
        "non_null_exercise_arrays": sum(
            path.parent.name != "Null" for path in exercise_paths
        ),
        "null_exercise_arrays": sum(
            path.parent.name == "Null" for path in exercise_paths
        ),
        "repetition_arrays": len(repetition_paths),
        "null_pseudo_repetition_arrays": len(null_repetition_paths),
        "short_repetition_arrays_under_0_2_seconds": sum(
            length < 20 for length in repetition_lengths
        ),
        "unmapped_exercise_indices": sorted(set(unmapped_exercise_indices)),
        "exercise_labels": sorted(
            path.name for path in (root / "np_exercise_data").iterdir() if path.is_dir()
        ),
        "unique_exercise_shapes": len({tuple(shape) for shape in exercise_shapes}),
        "exercise_length_samples": describe([shape[-1] for shape in exercise_shapes]),
        "repetition_length_samples": describe(repetition_lengths),
        "median_acceleration_norm_m_s2": describe(acc_norms),
        "gyro_abs_q99_rad_s": describe(gyro_q99),
        "imu_nonfinite_values": imu_nonfinite,
        "orientation_nonfinite_values": orientation_nonfinite,
        "orientation_warning": "orientation rows contain missing values in at least one released recording; HALO uses only the finite accelerometer and gyroscope rows",
        "channel_order": "wrist acc/gyro/orientation then ankle acc/gyro/orientation",
        "timing": "arrays are interpolated to 100 Hz by the released preprocessing code",
    }


def inspect_openpack() -> dict[str, Any]:
    paths = sorted((SOURCES / "openpack" / "downloads").glob("U*.zip"))
    expected_ids = [f"U01{index:02d}" for index in range(1, 12)] + [
        f"U02{index:02d}" for index in range(1, 11)
    ]
    identity_aliases = {
        "U0202": "U0105",
        "U0203": "U0108",
        "U0204": "U0110",
        "U0205": "U0107",
        "U0210": "U0103",
    }
    suffixes: Counter[str] = Counter()
    member_counts, archive_sizes = [], []
    sample_headers: dict[str, list[str]] = {}
    imu_rows, imu_durations, imu_rates, acc_norms, gyro_q99 = [], [], [], [], []
    imu_nonfinite = 0
    empty_imu_files = 0
    imu_non_increasing_steps = 0
    imu_gaps_over_0_2s: list[dict[str, Any]] = []
    imu_files = 0
    sensor_files: Counter[str] = Counter()
    action_rows = 0
    operation_rows = 0
    invalid_annotation_intervals = 0
    invalid_annotation_examples: list[dict[str, Any]] = []
    action_labels: Counter[str] = Counter()
    operation_labels: Counter[str] = Counter()
    for path in paths:
        archive_sizes.append(path.stat().st_size)
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            member_counts.append(len(members))
            for item in members:
                suffixes[Path(item.filename).suffix.lower() or "<none>"] += 1
                if item.filename.endswith(".csv") and len(sample_headers) < 8:
                    with archive.open(item) as stream:
                        header = (
                            stream.readline().decode("utf-8-sig").strip().split(",")
                        )
                    sample_headers[item.filename] = header
                if re.fullmatch(r"atr/atr0[1-4]/S\d{4}\.csv", item.filename):
                    frame = pd.read_csv(archive.open(item))
                    expected_columns = [
                        "unixtime",
                        "acc_x",
                        "acc_y",
                        "acc_z",
                        "gyro_x",
                        "gyro_y",
                        "gyro_z",
                        "quat_w",
                        "quat_x",
                        "quat_y",
                        "quat_z",
                    ]
                    if list(frame.columns) != expected_columns:
                        raise ValueError(
                            f"unexpected OpenPack IMU columns in {path.name}:{item.filename}"
                        )
                    if frame.empty:
                        empty_imu_files += 1
                        continue
                    values = frame.to_numpy(dtype=np.float64)
                    imu_nonfinite += int((~np.isfinite(values[:, :7])).sum())
                    timestamp = values[:, 0] / 1000.0
                    delta = np.diff(timestamp)
                    positive = delta[delta > 0]
                    imu_non_increasing_steps += int((delta <= 0).sum())
                    for index in np.where(delta > 0.2)[0]:
                        imu_gaps_over_0_2s.append(
                            {
                                "archive": path.name,
                                "member": item.filename,
                                "after_row": int(index),
                                "seconds": float(delta[index]),
                            }
                        )
                    imu_rows.append(len(frame))
                    imu_durations.append(float(timestamp[-1] - timestamp[0]))
                    imu_rates.append(float(1.0 / np.median(positive)))
                    acc_norms.append(
                        float(np.median(np.linalg.norm(values[:, 1:4], axis=1)))
                    )
                    gyro_q99.append(float(np.quantile(np.abs(values[:, 4:7]), 0.99)))
                    imu_files += 1
                    sensor_files[item.filename.split("/")[1]] += 1
                elif re.fullmatch(
                    r"annotation/openpack-actions/S\d{4}\.csv", item.filename
                ):
                    frame = pd.read_csv(archive.open(item))
                    start = pd.to_datetime(
                        frame["start"], utc=True, errors="coerce", format="mixed"
                    )
                    end = pd.to_datetime(
                        frame["end"], utc=True, errors="coerce", format="mixed"
                    )
                    invalid = start.isna() | end.isna() | (end <= start)
                    invalid_annotation_intervals += int(invalid.sum())
                    for index in np.flatnonzero(invalid.to_numpy())[:5]:
                        invalid_annotation_examples.append(
                            {
                                "archive": path.name,
                                "member": item.filename,
                                "row": int(index),
                                "label": str(frame.iloc[index]["action"]),
                                "start": str(frame.iloc[index]["start"]),
                                "end": str(frame.iloc[index]["end"]),
                            }
                        )
                    action_rows += len(frame)
                    action_labels.update(frame["action"].astype(str))
                elif re.fullmatch(
                    r"annotation/openpack-operations/S\d{4}\.csv", item.filename
                ):
                    frame = pd.read_csv(archive.open(item))
                    start = pd.to_datetime(
                        frame["start"], utc=True, errors="coerce", format="mixed"
                    )
                    end = pd.to_datetime(
                        frame["end"], utc=True, errors="coerce", format="mixed"
                    )
                    invalid = start.isna() | end.isna() | (end <= start)
                    invalid_annotation_intervals += int(invalid.sum())
                    for index in np.flatnonzero(invalid.to_numpy())[:5]:
                        invalid_annotation_examples.append(
                            {
                                "archive": path.name,
                                "member": item.filename,
                                "row": int(index),
                                "label": str(frame.iloc[index]["operation"]),
                                "start": str(frame.iloc[index]["start"]),
                                "end": str(frame.iloc[index]["end"]),
                            }
                        )
                    operation_rows += len(frame)
                    operation_labels.update(frame["operation"].astype(str))
    return {
        "status": (
            "pass_with_two_clock_gaps_and_one_zero_duration_label"
            if len(paths) == 21
            else "incomplete"
        ),
        "subject_archives": len(paths),
        "released_identifiers": [path.stem for path in paths],
        "missing_expected_identifiers": sorted(
            set(expected_ids) - {path.stem for path in paths}
        ),
        "distinct_people": 16,
        "identity_aliases": identity_aliases,
        "archive_bytes": sum(archive_sizes),
        "members_per_archive": describe(member_counts),
        "member_suffix_counts": dict(sorted(suffixes.items())),
        "sample_csv_headers": sample_headers,
        "imu_files": imu_files,
        "imu_files_by_sensor": dict(sorted(sensor_files.items())),
        "imu_rows_per_file": describe(imu_rows),
        "imu_duration_seconds": describe(imu_durations),
        "measured_rate_hz": describe(imu_rates),
        "median_acceleration_norm_g": describe(acc_norms),
        "gyro_abs_q99_deg_s": describe(gyro_q99),
        "imu_nonfinite_values": imu_nonfinite,
        "empty_imu_files": empty_imu_files,
        "imu_non_increasing_timestamp_steps": imu_non_increasing_steps,
        "imu_timestamp_gaps_over_0_2_seconds": imu_gaps_over_0_2s,
        "action_intervals": action_rows,
        "action_labels": len(action_labels),
        "operation_intervals": operation_rows,
        "operation_labels": len(operation_labels),
        "invalid_annotation_intervals": invalid_annotation_intervals,
        "invalid_annotation_examples": invalid_annotation_examples[:10],
        "release_count_note": "the inspected v1.0 subject archives contain 53,760 action and 20,264 operation intervals; these exceed the paper's 52,529 and 20,129 counts",
        "conversion": "acceleration already g; gyroscope degree/s must be converted to rad/s",
        "warning": "21 released identifiers represent 16 distinct people; collapse the five documented aliases before any subject split.",
    }


def inspect_recofit() -> dict[str, Any]:
    path = SOURCES / "recofit" / "downloads" / "exercise_data.50.0000_multionly.mat"
    variables = [
        {"name": name, "shape": list(shape), "class": cls}
        for name, shape, cls in whosmat(path)
    ]
    sample_rate = float(loadmat(path, variable_names=["Fs"], squeeze_me=True)["Fs"])

    subject_ids = set()
    durations, rates, acc_norms, gyro_q99 = [], [], [], []
    activity_rows: Counter[str] = Counter()
    nonfinite_imu = 0
    invalid_intervals = 0
    incomplete_visits = 0
    visit_count = 0
    for recording in iter_recofit(path):
        visit_count += 1
        subject_ids.add(int(recording.subject_id))
        incomplete_visits += int(recording.metadata["source_incomplete"])
        stream = recording.streams[0]
        timestamp = stream.timestamps_sec
        delta = np.diff(timestamp)
        positive = delta[delta > 0]
        durations.append(float(timestamp[-1] - timestamp[0]))
        rates.append(float(1.0 / np.median(positive)))
        values = stream.values
        nonfinite_imu += int((~np.isfinite(values[stream.valid])).sum())
        acc_norms.append(float(np.median(np.linalg.norm(values[:, :3], axis=1))))
        gyro_q99.append(float(np.rad2deg(np.quantile(np.abs(values[:, 3:6]), 0.99))))
        for event in recording.events:
            activity_rows[event.label] += 1
            invalid_intervals += int(event.end_sec <= event.start_sec)

    return {
        "status": "pass_with_set_level_annotations",
        "bytes": path.stat().st_size,
        "variables": variables,
        "sample_rate_field_hz": sample_rate,
        "selected_file_subjects": len(subject_ids),
        "visits": visit_count,
        "incomplete_visits": incomplete_visits,
        "duration_seconds": describe(durations),
        "total_hours": float(sum(durations) / 3600.0),
        "measured_rate_hz": describe(rates),
        "median_acceleration_norm_g": describe(acc_norms),
        "gyro_abs_q99_deg_s": describe(gyro_q99),
        "nonfinite_primary_imu_values": nonfinite_imu,
        "annotation_rows": int(sum(activity_rows.values())),
        "activity_labels": len(activity_rows),
        "most_common_annotation_rows": dict(activity_rows.most_common(15)),
        "invalid_annotation_intervals": invalid_intervals,
        "annotation_limit": "set intervals and counts only; no complete per-repetition timestamps",
        "conversion": "acceleration already g; gyroscope degree/s must be converted to rad/s",
        "placement": "primary accelDataMatrix/gyroDataMatrix is the documented right-forearm armband; slave matrices are not used without a separate placement contract",
    }


def inspect_existing_halo_source(name: str) -> dict[str, Any]:
    root = HALO_DATASETS / name
    paths = sorted((root / "sessions").glob("*/data.parquet"))
    sample_indices = np.linspace(0, len(paths) - 1, min(100, len(paths)), dtype=int)
    sampled_paths = [paths[index] for index in sample_indices]
    rows, durations, rates, acc_norms = [], [], [], []
    schemas = set()
    nonfinite = 0
    subjects = set()
    for path in sampled_paths:
        frame = pd.read_parquet(path)
        schemas.add(tuple(frame.columns))
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        nonfinite += int((~np.isfinite(numeric)).sum())
        timestamp = frame["timestamp_sec"].to_numpy(dtype=np.float64)
        delta = np.diff(timestamp)
        rows.append(len(frame))
        durations.append(float(timestamp[-1] - timestamp[0]))
        rates.append(float(1.0 / np.median(delta[delta > 0])))
        acceleration = frame[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float64)
        acc_norms.append(float(np.median(np.linalg.norm(acceleration, axis=1))))
        subjects.update(str(value) for value in frame["subject"].unique())
    labels = json.loads((root / "labels.json").read_text())
    label_values = sorted({label for values in labels.values() for label in values})
    return {
        "status": "pass_reused_in_place",
        "session_files": len(paths),
        "sampled_session_files": len(sampled_paths),
        "sampled_subjects": len(subjects),
        "labels": label_values,
        "rows_per_sampled_session": describe(rows),
        "duration_seconds": describe(durations),
        "measured_rate_hz": describe(rates),
        "median_acceleration_norm_m_s2": describe(acc_norms),
        "sampled_nonfinite_values": nonfinite,
        "column_schemas": [list(schema) for schema in sorted(schemas)],
        "reuse_note": "read from data/datasets in place; do not copy into the application data tree",
    }


INSPECTORS = {
    "aidlab_har": inspect_aidlab_har,
    "c_mhad": inspect_c_mhad,
    "crossfit": inspect_crossfit,
    "oca": inspect_oca,
    "openpack": inspect_openpack,
    "recofit": inspect_recofit,
    "wear": inspect_wear,
    "existing_harmes": lambda: inspect_existing_halo_source("harmes"),
    "existing_monipar": lambda: inspect_existing_halo_source("monipar"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - set(INSPECTORS))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    selected = args.datasets or list(INSPECTORS)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "inspected_on": "2026-08-30",
        "datasets": {},
    }
    for name in selected:
        inspector = INSPECTORS[name]
        print(f"Inspecting {name}")
        try:
            summary["datasets"][name] = inspector()
        except FileNotFoundError as error:
            summary["datasets"][name] = {
                "status": "not_downloaded",
                "error": str(error),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
