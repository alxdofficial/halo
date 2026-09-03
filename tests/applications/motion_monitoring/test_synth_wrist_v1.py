"""Synthetic Task-1 corpus: determinism, placement rules, provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters import synth_wrist_v1 as synth
from applications.motion_monitoring.data.contracts import SensorStream


RATE = 50.0


def _donor(clip_id: str, label: str, subject: str, seconds: float, seed: int) -> synth.DonorClip:
    rng = np.random.default_rng(seed)
    time = np.arange(int(seconds * 100)) / 100
    values = np.zeros((len(time), 6), dtype=np.float32)
    values[:, 2] = 1.0 + 0.5 * np.sin(2 * np.pi * time)  # gravity on z + dynamic part
    values[:, 0] = 0.4 * np.cos(2 * np.pi * time)
    values[:, 3:6] = rng.normal(0.0, 0.5, (len(time), 3))
    return synth.DonorClip(
        clip_id=clip_id,
        label=label,
        subject_id=subject,
        exercise_id=hash(label) % 10,
        repetition_index=0,
        values=values,
        rate_hz=100.0,
    )


def _session(recording_id: str, subject: str, seconds: float, seed: int, *, sets=()):
    rng = np.random.default_rng(seed)
    time = np.arange(int(seconds * RATE)) / RATE
    values = np.zeros((len(time), 6), dtype=np.float32)
    values[:, 1] = 1.0  # gravity on y: a different frame than the donors
    values += rng.normal(0.0, 0.01, values.shape).astype(np.float32)
    # a burst of background motion inside every "set"
    for start, end, _ in sets:
        mask = (time >= start) & (time < end)
        values[mask, 0] += 0.8 * np.sin(2 * np.pi * 2 * time[mask])
    stream = SensorStream(
        stream_id="right_forearm_imu",
        placement="right forearm",
        device="test",
        timestamps_sec=time,
        values=values,
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        valid=np.ones_like(values, dtype=bool),
        gravity_state="present",
        nominal_rate_hz=RATE,
    )
    return synth.BackgroundSession(
        cache_index=0,
        recording_id=recording_id,
        subject_id=subject,
        timestamps_sec=time,
        values=values,
        valid=np.ones_like(values, dtype=bool),
        rate_hz=RATE,
        set_intervals=tuple(sets),
        junk_intervals=(),
        stream=stream,
    )


@pytest.fixture
def bank():
    donors = tuple(
        _donor(f"{label}-{subject}-{index}", label, subject, 1.0 + 0.5 * index, seed)
        for seed, (label, subject, index) in enumerate(
            (label, subject, index)
            for label in ("squat", "pushup", "burpee")
            for subject in ("A", "B", "C")
            for index in range(2)
        )
    )
    sessions = (
        _session("bg-1", "p1", 180.0, 1, sets=((40.0, 55.0, "row"),)),
        _session("bg-2", "p2", 200.0, 2),
    )
    weights = np.array([180.0, 200.0]) / 380.0
    return donors, sessions, weights


def _synthesize(bank, seed: int, index: int = 0):
    donors, sessions, weights = bank
    rng = np.random.default_rng(seed)
    return synth.synthesize_recording(
        index, rng, donors=donors, sessions=sessions, session_weights=weights
    )


def test_synthesis_is_deterministic_under_seed(bank):
    first = _synthesize(bank, 7)
    second = _synthesize(bank, 7)
    other = _synthesize(bank, 8)
    assert first is not None and second is not None
    assert np.array_equal(first.streams[0].values, second.streams[0].values)
    assert [e.metadata for e in first.events] == [e.metadata for e in second.events]
    assert other is None or not np.array_equal(first.streams[0].values, other.streams[0].values)


def test_inserts_respect_margins_spacing_sets_and_carry_provenance(bank):
    config = synth.SYNTHESIS_CONFIG
    seen_inserts = 0
    for seed in range(12):
        recording = _synthesize(bank, seed, index=seed)
        if recording is None:
            continue
        stream = recording.streams[0]
        length = float(stream.timestamps_sec[-1]) + 1.0 / RATE
        inserts = [e for e in recording.events if e.annotation_kind == "inserted_execution"]
        sets = [e for e in recording.events if e.annotation_kind == "background_activity"]
        seen_inserts += len(inserts)
        assert recording.metadata["synthetic"] is True
        assert recording.metadata["synthesis_config_sha256"] == synth.config_digest()
        assert recording.metadata["inserted_primary_count"] == sum(
            e.metadata["role"] == "primary" for e in inserts
        )
        assert recording.subject_id.startswith("bg:")
        assert stream.metadata["whole_query_rotation_deg"] <= config["whole_query_rotation_max_deg"]
        assert bool(np.asarray(stream.valid).all())
        # rotations preserve |acc|; background gravity is still ~1 g
        norms = np.linalg.norm(stream.values[:, :3], axis=1)
        assert abs(float(np.median(norms)) - 1.0) < 0.05
        for event in inserts:
            assert event.start_sec >= config["edge_margin_sec"] - 1e-6
            assert event.end_sec <= length - config["edge_margin_sec"] + 1e-6
            assert config["time_warp_bounds"][0] <= event.metadata["time_warp"] <= config["time_warp_bounds"][1]
            assert config["amplitude_bounds"][0] <= event.metadata["amplitude"] <= config["amplitude_bounds"][1]
            assert event.metadata["guard_sec"] == event.metadata["crossfade_sec"]
            assert event.metadata["donor_clip_id"].startswith(event.label)
            if event.metadata["role"] == "primary":
                assert event.label == recording.metadata["primary_label"]
            else:
                assert event.label != recording.metadata["primary_label"]
            for background in sets:
                assert event.end_sec + config["insert_spacing_sec"] <= background.start_sec + 1e-6 or (
                    event.start_sec - config["insert_spacing_sec"] >= background.end_sec - 1e-6
                )
        for left, right in zip(inserts, inserts[1:]):
            assert right.start_sec - left.end_sec >= config["insert_spacing_sec"] - 1e-6
    assert seen_inserts > 0


def test_same_subject_primary_fraction_and_distractor_labels(bank):
    same = total = 0
    for seed in range(40):
        recording = _synthesize(bank, seed, index=seed)
        if recording is None:
            continue
        reference_subject = recording.metadata["reference_subject_id"]
        for event in recording.events:
            if event.annotation_kind != "inserted_execution" or event.metadata["role"] != "primary":
                continue
            total += 1
            same += event.metadata["donor_subject_id"] == reference_subject
    assert total > 20
    # Most primaries come from a different person than the reference.
    assert same / total < 0.5


def test_derived_provenance_binds_generator_and_source_caches():
    from applications.motion_monitoring.data.cache import cache_provenance

    sources = Path(synth.__file__).resolve().parents[1] / "sources"
    if not (sources / "crossfit" / "processed" / "canonical_v1" / "cache.json").is_file():
        pytest.skip("crossfit canonical cache is not built here")
    provenance = cache_provenance("synth_wrist_v1")
    assert set(provenance["source_caches"]) == {"crossfit"}
    assert "payload_tree_sha256" not in provenance
    assert provenance["adapter_module"].endswith("synth_wrist_v1")
    assert len(provenance["adapter_sha256"]) == 64


def test_clean_donor_record_is_enrollment_only_and_configuration_matched(bank):
    clip = bank[0][0]
    recording = synth.donor_recording(0, clip)
    event = recording.events[0]
    stream = recording.streams[0]
    assert recording.metadata["task1_reference_only"] is True
    assert event.annotation_kind == "enrollment_execution"
    assert event.metadata["donor_clip_id"] == clip.clip_id
    assert stream.placement == "wrist"
    assert stream.device == "off-the-shelf smartwatch"
    assert stream.channels == (
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
    )


def test_every_synthetic_query_has_primary_and_distractor_insertions(bank):
    for seed in range(20):
        recording = _synthesize(bank, seed, index=seed)
        assert recording is not None
        assert recording.metadata["inserted_primary_count"] >= 1
        assert recording.metadata["inserted_distractor_count"] >= 1
