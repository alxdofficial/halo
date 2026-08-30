from __future__ import annotations

from pathlib import Path

import numpy as np

from applications.motion_monitoring.data.cache import (
    CachedRecordingDataset,
    write_recording,
)
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


def test_recording_cache_round_trip_and_map_access(tmp_path: Path) -> None:
    timestamps = np.arange(12, dtype=np.float64) / 20.0
    values = np.arange(72, dtype=np.float32).reshape(12, 6)
    valid = np.ones_like(values, dtype=bool)
    valid[3, 4] = False
    recording = RawRecording(
        dataset="fixture",
        recording_id="person/visit:1",
        subject_id="person",
        session_id="visit",
        streams=(
            SensorStream(
                stream_id="watch",
                placement="right_wrist",
                device="fixture",
                timestamps_sec=timestamps,
                values=values,
                channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
                valid=valid,
                gravity_state="present",
                nominal_rate_hz=20.0,
                metadata={"tuple": (1, 2)},
            ),
        ),
        events=(EventInterval(0.1, 0.4, "event", metadata={"count": np.int64(2)}),),
        split="train",
        metadata={"array": np.asarray([1, 2])},
    )

    record_dir = tmp_path / "record"
    write_recording(recording, record_dir)
    (tmp_path / "manifest.jsonl").write_text(
        '{"directory":"record","recording_id":"person/visit:1"}\n', encoding="utf-8"
    )
    cached = CachedRecordingDataset(tmp_path)
    restored = cached[0]

    assert len(cached) == 1
    assert restored.recording_id == recording.recording_id
    assert restored.metadata["array"] == [1, 2]
    assert restored.streams[0].metadata["tuple"] == [1, 2]
    assert restored.events[0].metadata["count"] == 2
    assert isinstance(restored.streams[0].values.base, np.memmap)
    np.testing.assert_array_equal(restored.streams[0].values, values)
    np.testing.assert_array_equal(restored.streams[0].valid, valid)
