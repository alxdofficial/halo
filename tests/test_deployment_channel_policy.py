"""Regression tests for the phone/watch deployment-scoped data view."""

import numpy as np
import pandas as pd
import pytest

from data.scripts.deployment_policy import (
    EXCLUDED_PRIMARY_DATASETS,
    PRIMARY_EVAL_DATASETS,
    PRIMARY_TRAIN_DATASETS,
    STANDARD_CHANNEL_ORDER,
    all_source_channels,
    channel_description,
    curate_frame,
    get_stream_spec,
    session_stream_specs,
    stream_specs,
)


EXPECTED_PRIMARY_CHANNELS = {
    "uci_har": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "hhar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "pamap2": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    # WISDM has two primary logical devices; both share the same intended schema.
    "wisdm": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "kuhar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "unimib_shar": ("acc_x", "acc_y", "acc_z"),
    "hapt": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "mhealth": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "capture24": ("acc_x", "acc_y", "acc_z"),
    "motionsense": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "realworld": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "mobiact": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "shoaib": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "inclusivehar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
}


def _full_source_frame(dataset, spec, n=8):
    columns = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    for source in all_source_channels(dataset, role=spec.role):
        columns[source] = np.ones(n, dtype=float)
    return pd.DataFrame(columns)


@pytest.mark.parametrize("dataset", PRIMARY_TRAIN_DATASETS + PRIMARY_EVAL_DATASETS)
def test_primary_streams_prune_to_three_or_six_channels(dataset):
    specs = stream_specs(dataset, role="primary")
    assert specs, f"{dataset} needs at least one primary deployment stream"
    for spec in specs:
        curated, metadata = curate_frame(_full_source_frame(dataset, spec), spec)
        sensor_columns = tuple(c for c in curated.columns if c != "timestamp_sec")
        assert sensor_columns == EXPECTED_PRIMARY_CHANNELS[dataset]
        assert metadata.channels == sensor_columns
        assert len(sensor_columns) in (3, 6)
        assert sensor_columns == tuple(c for c in STANDARD_CHANNEL_ORDER if c in sensor_columns)
        assert not any("mag" in c or "ecg" in c or "temp" in c for c in curated.columns)


def test_uci_uses_total_acceleration_not_body_acceleration():
    spec = get_stream_spec("uci_har", "phone_waist")
    assert spec.required["acc_x"] == ("total_acc_x",)
    assert "body_acc_x" not in all_source_channels("uci_har")


def test_pamap2_keeps_only_wrist_acc16_and_gyro():
    sources = set(all_source_channels("pamap2"))
    assert sources == {
        "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
        "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    }


def test_mhealth_keeps_right_wrist_imu_acc_and_gyro():
    sources = set(all_source_channels("mhealth"))
    assert sources == {
        "arm_acc_x", "arm_acc_y", "arm_acc_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    }
    assert not any("chest" in s or "ankle" in s or "ecg" in s or "mag" in s for s in sources)


def test_ios_total_acceleration_is_reconstructed_before_pruning():
    spec = get_stream_spec("motionsense", "phone_front_pocket")
    frame = _full_source_frame("motionsense", spec)
    curated, metadata = curate_frame(frame, spec)
    # userAcceleration=1 plus gravity=1 for each synthetic axis.
    assert np.allclose(curated[["acc_x", "acc_y", "acc_z"]].to_numpy(), 2.0)
    assert metadata.gravity_state == "present"
    assert not any(name.startswith("gravity_") or name.startswith("attitude_") for name in curated.columns)


def test_wisdm_legacy_sessions_keep_accel_sessions_and_drop_gyro_only_sessions():
    phone = session_stream_specs("wisdm", "phone_accel_1600_A_0000")
    watch = session_stream_specs("wisdm", "watch_accel_1600_A_0000")
    assert [s.stream_id for s in phone] == ["phone_pocket"]
    assert [s.stream_id for s in watch] == ["watch_wrist"]
    assert session_stream_specs("wisdm", "phone_gyro_1600_A_0000") == ()
    assert session_stream_specs("wisdm", "watch_gyro_1600_A_0000") == ()


def test_wisdm_missing_gyro_remains_accelerometer_only():
    spec = get_stream_spec("wisdm", "phone_pocket")
    frame = pd.DataFrame({
        "timestamp_sec": [0.0, 0.05],
        "phone_accel_x": [1.0, 1.1],
        "phone_accel_y": [2.0, 2.1],
        "phone_accel_z": [3.0, 3.1],
    })
    curated, metadata = curate_frame(frame, spec)
    assert metadata.channels == ("acc_x", "acc_y", "acc_z")
    assert tuple(curated.columns) == ("timestamp_sec", "acc_x", "acc_y", "acc_z")


def test_shoaib_primary_and_diagnostic_placements_are_separate():
    assert [s.stream_id for s in stream_specs("shoaib", "primary")] == ["phone_right_pocket"]
    assert {s.stream_id for s in stream_specs("shoaib", "diagnostic")} == {
        "phone_left_pocket", "phone_belt", "watch_wrist_proxy",
    }
    assert not any("upper_arm" in source for source in all_source_channels("shoaib", "diagnostic"))


def test_non_deployment_datasets_are_not_primary():
    assert "dsads" not in PRIMARY_TRAIN_DATASETS
    assert "harth" not in PRIMARY_EVAL_DATASETS
    assert stream_specs("dsads", "primary") == ()
    assert stream_specs("harth", "primary") == ()
    assert {"dsads", "harth"}.issubset(EXCLUDED_PRIMARY_DATASETS)


def test_channel_text_names_device_placement_axis_and_gravity_state():
    spec = get_stream_spec("kuhar", "phone_waist")
    _, metadata = curate_frame(_full_source_frame("kuhar", spec), spec)
    assert channel_description(metadata, "acc_x") == (
        "phone accelerometer X-axis at waist; gravity removed"
    )
    assert channel_description(metadata, "gyro_z") == "phone gyroscope Z-axis at waist"
