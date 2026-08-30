"""Lazy adapter for the corrected AIDLAB-HAR v3 EDF release."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_ARCHIVE_NAME = "AIDLAB-HAR-DATASET_v3.zip"
_ARCHIVE_PREFIX = "AIDLAB-HAR-DATASET-v3"
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "aidlab_har"
_RECORDING_RE = re.compile(
    r"^(?P<subject>SUB\d+)_(?P<activity>[A-Z]+)_S(?P<series>\d+)$"
)
_ACCELERATION_LABELS = ("acceleration_x", "acceleration_y", "acceleration_z")
_CANONICAL_ACCELERATION_CHANNELS = ("acc_x", "acc_y", "acc_z")
_ACTIVITY_LABELS = {
    "ABDOMINALTENSE": "abdominal_tense",
    "BEND": "bend",
    "BROADJUMP": "broad_jump",
    "BURPEES": "burpees",
    "CHAIRSTANDANDSIT": "chair_stand_and_sit",
    "CRUNCHES": "crunches",
    "DOWNWARDDOG": "downward_dog",
    "LUNGES": "lunges",
    "LYINGHIPRISES": "lying_hip_rises",
    "PLANK": "plank",
    "PUSHUPS": "push_ups",
    "ROTATINGTOETOUCHES": "rotating_toe_touches",
    "RUNNINGPLANK": "running_plank",
    "SIDELUNGES": "side_lunges",
    "SQUATS": "squats",
    "WALK": "walk",
}


@dataclass(frozen=True)
class _EdfSignal:
    label: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: int


def _find_archive(root: Path | None) -> Path:
    source = _DEFAULT_ROOT if root is None else Path(root)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"expected a ZIP archive, got: {source}")
        return source

    candidates = (
        source / "downloads" / _ARCHIVE_NAME,
        source / _ARCHIVE_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {_ARCHIVE_NAME} under AIDLAB-HAR root {source}"
    )


def _ascii_field(raw: bytes, *, name: str) -> str:
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"EDF field {name!r} is not ASCII") from exc


def _number(value: str, *, name: str, kind: type[int] | type[float]) -> int | float:
    try:
        return kind(value)
    except ValueError as exc:
        raise ValueError(f"invalid EDF {name}: {value!r}") from exc


def _parse_signal_headers(raw: bytes, signal_count: int) -> tuple[_EdfSignal, ...]:
    position = 256
    field_specs = (
        ("label", 16),
        ("transducer", 80),
        ("physical_dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefiltering", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    )
    fields: dict[str, list[str]] = {}
    for name, width in field_specs:
        values = []
        for index in range(signal_count):
            end = position + width
            if end > len(raw):
                raise ValueError("truncated EDF signal header")
            values.append(_ascii_field(raw[position:end], name=f"{name}[{index}]"))
            position = end
        fields[name] = values

    signals = []
    for index in range(signal_count):
        signal = _EdfSignal(
            label=fields["label"][index],
            physical_dimension=fields["physical_dimension"][index],
            physical_min=float(
                _number(
                    fields["physical_min"][index],
                    name=f"physical_min[{index}]",
                    kind=float,
                )
            ),
            physical_max=float(
                _number(
                    fields["physical_max"][index],
                    name=f"physical_max[{index}]",
                    kind=float,
                )
            ),
            digital_min=int(
                _number(
                    fields["digital_min"][index],
                    name=f"digital_min[{index}]",
                    kind=int,
                )
            ),
            digital_max=int(
                _number(
                    fields["digital_max"][index],
                    name=f"digital_max[{index}]",
                    kind=int,
                )
            ),
            samples_per_record=int(
                _number(
                    fields["samples_per_record"][index],
                    name=f"samples_per_record[{index}]",
                    kind=int,
                )
            ),
        )
        if signal.samples_per_record <= 0:
            raise ValueError(f"EDF signal {signal.label!r} has no samples")
        if signal.digital_max <= signal.digital_min:
            raise ValueError(f"EDF signal {signal.label!r} has invalid digital range")
        if signal.physical_max <= signal.physical_min:
            raise ValueError(f"EDF signal {signal.label!r} has invalid physical range")
        signals.append(signal)
    return tuple(signals)


def _unit_to_g_scale(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized == "g":
        return 1.0
    if normalized == "mg":
        return 1.0 / 1000.0
    if normalized in {"m/s2", "m/s^2", "m/s²"}:
        return 1.0 / 9.80665
    raise ValueError(f"unsupported AIDLAB-HAR acceleration unit: {unit!r}")


def _read_acceleration_edf(raw: bytes) -> tuple[np.ndarray, np.ndarray, float]:
    if len(raw) < 256:
        raise ValueError("truncated EDF fixed header")
    if _ascii_field(raw[0:8], name="version") != "0":
        raise ValueError("unsupported EDF version")

    header_bytes = int(
        _number(
            _ascii_field(raw[184:192], name="header_bytes"),
            name="header_bytes",
            kind=int,
        )
    )
    record_count = int(
        _number(
            _ascii_field(raw[236:244], name="record_count"),
            name="record_count",
            kind=int,
        )
    )
    record_duration_sec = float(
        _number(
            _ascii_field(raw[244:252], name="record_duration"),
            name="record_duration",
            kind=float,
        )
    )
    signal_count = int(
        _number(
            _ascii_field(raw[252:256], name="signal_count"),
            name="signal_count",
            kind=int,
        )
    )
    if signal_count <= 0 or record_duration_sec <= 0:
        raise ValueError("EDF has invalid signal count or record duration")
    minimum_header_bytes = 256 * (signal_count + 1)
    if header_bytes < minimum_header_bytes or header_bytes > len(raw):
        raise ValueError("EDF header byte count is inconsistent with the file")

    signals = _parse_signal_headers(raw[:header_bytes], signal_count)
    samples_per_record = sum(signal.samples_per_record for signal in signals)
    bytes_per_record = samples_per_record * np.dtype("<i2").itemsize
    data_bytes = len(raw) - header_bytes
    if record_count == -1:
        if data_bytes % bytes_per_record:
            raise ValueError("unknown EDF record count cannot be inferred exactly")
        record_count = data_bytes // bytes_per_record
    if record_count <= 0 or data_bytes != record_count * bytes_per_record:
        raise ValueError("EDF data length does not match its record layout")

    label_to_index = {signal.label: index for index, signal in enumerate(signals)}
    if len(label_to_index) != len(signals):
        raise ValueError("EDF signal labels are not unique")
    missing = [label for label in _ACCELERATION_LABELS if label not in label_to_index]
    if missing:
        raise ValueError(f"EDF is missing acceleration signals: {missing}")

    digital = np.frombuffer(raw, dtype="<i2", offset=header_bytes)
    records = digital.reshape(record_count, samples_per_record)
    signal_offsets = np.cumsum([0, *(signal.samples_per_record for signal in signals)])
    acceleration_length: int | None = None
    acceleration: np.ndarray | None = None
    rates = []
    for axis, label in enumerate(_ACCELERATION_LABELS):
        index = label_to_index[label]
        signal = signals[index]
        start = int(signal_offsets[index])
        stop = int(signal_offsets[index + 1])
        samples = records[:, start:stop].reshape(-1)
        if acceleration_length is None:
            acceleration_length = len(samples)
            acceleration = np.empty(
                (acceleration_length, len(_ACCELERATION_LABELS)), dtype=np.float32
            )
        elif len(samples) != acceleration_length:
            raise ValueError("acceleration axes contain different sample counts")

        physical_scale = (signal.physical_max - signal.physical_min) / (
            signal.digital_max - signal.digital_min
        )
        unit_scale = _unit_to_g_scale(signal.physical_dimension)
        scale = physical_scale * unit_scale
        offset = (
            signal.physical_min - signal.digital_min * physical_scale
        ) * unit_scale
        assert acceleration is not None
        np.multiply(samples, scale, out=acceleration[:, axis], casting="unsafe")
        np.add(acceleration[:, axis], offset, out=acceleration[:, axis])
        rates.append(signal.samples_per_record / record_duration_sec)

    if any(abs(rate - rates[0]) > 1e-9 for rate in rates[1:]):
        raise ValueError(f"acceleration axes have different sample rates: {rates}")
    assert acceleration is not None
    rate_hz = float(rates[0])
    timestamps_sec = np.arange(len(acceleration), dtype=np.float64) / rate_hz
    return timestamps_sec, acceleration, rate_hz


def _read_quality_intervals(
    archive: ZipFile,
) -> dict[str, tuple[tuple[int, int], ...]]:
    member = f"{_ARCHIVE_PREFIX}/quality_intervals.csv"
    rows = csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")))
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row["signal"] != "acceleration":
            continue
        start = int(row["start_sample"])
        stop = int(row["end_sample_exclusive"])
        if start < 0 or stop <= start:
            raise ValueError(f"invalid acceleration quality interval: {row}")
        intervals[row["recording_id"]].append((start, stop))
    return {key: tuple(value) for key, value in intervals.items()}


def _pair_source_events(
    archive: ZipFile,
    *,
    csv_member: str,
    activity: str,
) -> tuple[EventInterval, ...]:
    rows = csv.DictReader(io.StringIO(archive.read(csv_member).decode("utf-8-sig")))
    open_intervals: dict[str, float] = {}
    events = []
    for row in rows:
        timestamp = float(row["timestamp_s"])
        source_event = row["event"].strip()
        if source_event.endswith("_onset"):
            interval_type = source_event.removesuffix("_onset")
            if interval_type in open_intervals:
                raise ValueError(f"duplicate {interval_type} onset in {csv_member}")
            open_intervals[interval_type] = timestamp
            continue
        if not source_event.endswith("_offset"):
            raise ValueError(f"unknown source event {source_event!r} in {csv_member}")
        interval_type = source_event.removesuffix("_offset")
        try:
            start = open_intervals.pop(interval_type)
        except KeyError as exc:
            raise ValueError(f"orphan {interval_type} offset in {csv_member}") from exc
        annotation_kind = (
            "repetition_fiducial" if interval_type == "repetition_marker" else "series"
        )
        events.append(
            EventInterval(
                start_sec=start,
                end_sec=timestamp,
                label=activity,
                annotation_kind=annotation_kind,
                metadata={"source_interval_type": interval_type},
            )
        )
    if open_intervals:
        raise ValueError(f"unclosed source intervals in {csv_member}: {open_intervals}")
    return tuple(sorted(events, key=lambda event: (event.start_sec, event.end_sec)))


def _events_within_slice(
    events: tuple[EventInterval, ...],
    timestamps_sec: np.ndarray,
    rate_hz: float,
) -> tuple[EventInterval, ...]:
    start = float(timestamps_sec[0])
    stop = float(timestamps_sec[-1] + 1.0 / rate_hz)
    return tuple(
        event
        for event in events
        if event.start_sec >= start and event.end_sec <= stop + 1e-9
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield native-rate, recording-scoped AIDLAB-HAR accelerometer streams.

    ``SUBxx`` is retained only as source metadata. Since the release explicitly
    says those codes are not global participant identities, each source file is
    assigned a collision-free recording-scoped ``subject_id``.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return

    archive_path = _find_archive(root)
    yielded = 0
    with ZipFile(archive_path) as archive:
        quality_intervals = _read_quality_intervals(archive)
        members = set(archive.namelist())
        edf_members = sorted(
            member
            for member in members
            if member.startswith(f"{_ARCHIVE_PREFIX}/data/")
            and member.lower().endswith(".edf")
        )
        for edf_member in edf_members:
            source_recording_id = Path(edf_member).stem
            match = _RECORDING_RE.fullmatch(source_recording_id)
            if match is None:
                raise ValueError(f"unexpected AIDLAB-HAR recording name: {edf_member}")
            source_activity = match.group("activity")
            try:
                activity = _ACTIVITY_LABELS[source_activity]
            except KeyError as exc:
                raise ValueError(
                    f"unknown AIDLAB-HAR activity: {source_activity}"
                ) from exc

            timestamps, acceleration, rate_hz = _read_acceleration_edf(
                archive.read(edf_member)
            )
            valid = np.ones(acceleration.shape, dtype=bool)
            source_quality_intervals = quality_intervals.get(source_recording_id, ())
            for start, stop in source_quality_intervals:
                if stop > len(valid):
                    raise ValueError(
                        f"quality interval [{start}, {stop}) exceeds {source_recording_id}"
                    )
                valid[start:stop, :] = False

            csv_member = str(Path(edf_member).with_suffix(".csv")).replace("\\", "/")
            events = (
                _pair_source_events(
                    archive,
                    csv_member=csv_member,
                    activity=activity,
                )
                if csv_member in members
                else ()
            )
            clock_slices = split_at_clock_gaps(
                timestamps,
                max_gap_sec=1.5 / rate_hz,
            )
            for part_index, clock_slice in enumerate(clock_slices):
                part_timestamps = timestamps[clock_slice]
                suffix = "" if len(clock_slices) == 1 else f"::part{part_index:03d}"
                recording_id = f"aidlab_har::{source_recording_id}{suffix}"
                yield RawRecording(
                    dataset="aidlab_har",
                    recording_id=recording_id,
                    subject_id=f"aidlab_har_recording::{source_recording_id}{suffix}",
                    session_id=recording_id,
                    streams=(
                        SensorStream(
                            stream_id="chest_accelerometer",
                            placement="chest",
                            device="AIDLAB wearable IMU",
                            timestamps_sec=part_timestamps,
                            values=acceleration[clock_slice],
                            channels=_CANONICAL_ACCELERATION_CHANNELS,
                            valid=valid[clock_slice],
                            gravity_state="present",
                            nominal_rate_hz=rate_hz,
                            metadata={
                                "source_member": edf_member,
                                "source_unit": "g",
                                "raw_gyroscope_available": False,
                            },
                        ),
                    ),
                    events=_events_within_slice(events, part_timestamps, rate_hz),
                    metadata={
                        "source_recording_id": source_recording_id,
                        "source_subject_code": match.group("subject"),
                        "subject_identity_scope": "recording_only",
                        "source_series_code": f"S{match.group('series')}",
                        "source_activity": source_activity,
                        "activity": activity,
                        "time_origin": "sample_relative_edf",
                        "acceleration_quality_interval_count": len(
                            source_quality_intervals
                        ),
                        "annotation_available": bool(events),
                    },
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
