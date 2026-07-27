"""Focused tests for the H-MOG nested-archive converter."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from data.datasets.hmog.convert import (
    RATE_HZ,
    WINDOW_SAMPLES,
    _activity_table,
    convert,
    regularize_activity,
)


def test_hmog_is_opt_in_and_has_six_channel_phone_hand_policy():
    from data.scripts.curate.deployment_policy import (
        OPTIONAL_PHASE_A_DATASETS as POLICY_OPTIONAL,
        get_stream_spec,
    )
    from training.tokenizer.pretrain_data import (
        OPTIONAL_PHASE_A_DATASETS as TRAINER_OPTIONAL,
        stream_channel_descriptions,
        stream_sensor_texts,
    )

    assert "hmog" in POLICY_OPTIONAL
    assert set(POLICY_OPTIONAL) == set(TRAINER_OPTIONAL)
    spec = get_stream_spec("hmog", "phone_hand")
    assert spec.role == "phase_a_scale"
    assert spec.device_profile == "phone"
    assert spec.gravity_state == "present"
    assert tuple(spec.required) == (
        "acc_x",
        "acc_y",
        "acc_z",
    )
    assert tuple(spec.optional) == (
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )
    descriptions = stream_channel_descriptions("hmog", "phone_hand")
    assert len(descriptions) == 6
    assert all("hand" in value and "phone" in value for value in descriptions)
    _, sensors, sensor_id = stream_sensor_texts("hmog", "phone_hand")
    assert len(sensors) == 2
    assert sensor_id == [0, 0, 0, 1, 1, 1]


def test_regularize_activity_synchronizes_and_splits_long_gaps():
    first = np.arange(0.0, 7.0, 1.0 / RATE_HZ)
    second = np.arange(8.0, 15.0, 1.0 / RATE_HZ)
    time = np.r_[first, second]
    acc = np.column_stack((np.sin(time), np.cos(time), np.full(len(time), 9.81)))
    gyro = np.column_stack((time * 0.01, time * 0.02, time * 0.03))

    blocks, stats = regularize_activity(time, acc, time, gyro)

    assert [len(block) for block in blocks] == [WINDOW_SAMPLES, WINDOW_SAMPLES]
    assert stats["retained_samples"] == 2 * WINDOW_SAMPLES
    assert stats["invalid_grid_samples"] > 0
    assert np.isfinite(np.concatenate(blocks)).all()


def test_activity_table_maps_and_deduplicates_consistent_rows(tmp_path):
    path = tmp_path / "subject.zip"
    row = "100669011000001,100669,1,0,10000,0,10000,1,7,1\n"
    with ZipFile(path, "w") as archive:
        archive.writestr("100669/100669_session_1/Activity.csv", row + row)
    with ZipFile(path) as archive:
        frame = _activity_table(
            archive,
            "100669/100669_session_1/Activity.csv",
            "100669",
        )
    assert len(frame) == 1
    assert int(frame.iloc[0]["gesture_scenario"]) == 1


def test_convert_synthetic_nested_archive(tmp_path):
    subject = "100669"
    base = f"{subject}/{subject}_session_2"
    activity_id = 100669021000001
    n = 2 * WINDOW_SAMPLES
    event_ns = np.arange(n, dtype=np.int64) * 10_000_000
    system_ms = 1_400_000_000_000 + event_ns // 1_000_000
    acc = pd.DataFrame(
        {
            0: system_ms,
            1: event_ns,
            2: activity_id,
            3: np.zeros(n),
            4: np.zeros(n),
            5: np.full(n, 9.80665),
            6: np.zeros(n, dtype=int),
        }
    )
    gyro = acc.copy()
    gyro[[3, 4, 5]] = 0.0
    activity = (
        f"{activity_id},{subject},2,{system_ms[0]},{system_ms[-1]},"
        f"{event_ns[0] // 1_000_000},{event_ns[-1] // 1_000_000},2,2,1\n"
    )

    nested = BytesIO()
    with ZipFile(nested, "w") as subject_archive:
        subject_archive.writestr(f"{base}/Activity.csv", activity)
        subject_archive.writestr(
            f"{base}/Accelerometer.csv",
            acc.to_csv(index=False, header=False),
        )
        subject_archive.writestr(
            f"{base}/Gyroscope.csv",
            gyro.to_csv(index=False, header=False),
        )
    outer_path = tmp_path / "hmog_dataset.zip"
    with ZipFile(outer_path, "w") as outer:
        outer.writestr(f"public_dataset/{subject}.zip", nested.getvalue())

    output = tmp_path / "converted"
    assert convert(outer_path, output)
    sessions = list((output / "sessions").glob("*/data.parquet"))
    assert len(sessions) == 1
    frame = pd.read_parquet(sessions[0])
    assert len(frame) == 2 * WINDOW_SAMPLES
    assert frame["subject"].astype(str).unique().tolist() == [subject]
    assert frame[["acc_x", "acc_y"]].to_numpy().max() == 0.0
    assert np.allclose(frame["acc_z"], 9.80665)
    assert '"walking"' in (output / "labels.json").read_text()


def test_convert_uses_unique_internal_subject_id_and_records_archive_alias(tmp_path):
    archive_subject = "207969"
    internal_subject = "207696"
    base = f"{archive_subject}/{archive_subject}_session_1"
    activity_id = 207696011000001
    n = WINDOW_SAMPLES
    event_ns = np.arange(n, dtype=np.int64) * 10_000_000
    system_ms = 1_400_000_000_000 + event_ns // 1_000_000
    sensor = pd.DataFrame(
        {
            0: system_ms,
            1: event_ns,
            2: activity_id,
            3: np.zeros(n),
            4: np.zeros(n),
            5: np.full(n, 9.80665),
            6: np.zeros(n, dtype=int),
        }
    )
    activity = (
        f"{activity_id},{internal_subject},1,{system_ms[0]},{system_ms[-1]},"
        f"{event_ns[0] // 1_000_000},{event_ns[-1] // 1_000_000},1,1,1\n"
    )
    nested = BytesIO()
    with ZipFile(nested, "w") as subject_archive:
        subject_archive.writestr(f"{base}/Activity.csv", activity)
        subject_archive.writestr(
            f"{base}/Accelerometer.csv",
            sensor.to_csv(index=False, header=False),
        )
        subject_archive.writestr(
            f"{base}/Gyroscope.csv",
            sensor.to_csv(index=False, header=False),
        )
    outer_path = tmp_path / "hmog_dataset.zip"
    with ZipFile(outer_path, "w") as outer:
        outer.writestr(
            f"public_dataset/{archive_subject}.zip",
            nested.getvalue(),
        )

    output = tmp_path / "converted"
    assert convert(outer_path, output)
    frame = pd.read_parquet(next((output / "sessions").glob("*/data.parquet")))
    assert frame["subject"].astype(str).unique().tolist() == [internal_subject]
    manifest = (output / "manifest.json").read_text()
    assert f'"{archive_subject}": "{internal_subject}"' in manifest
