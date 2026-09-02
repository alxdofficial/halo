"""Synthetic wrist-IMU training corpus for Task 1 (spec section C).

Every recording is a 60-120 s slice of a real RecoFit right-forearm session
into which single-repetition CrossFit wrist clips are spliced. The inserted
extents are the only ``inserted_execution`` events, so target labels are
execution-level by construction; the background's own RecoFit annotations are
kept as ``background_activity`` events for provenance and never become targets.

Design constraints (all measurable downstream):

* raw-level synthesis in canonical units (g, rad/s) at the background's native
  50 Hz, so every encoder sees the same waveform once;
* the donor is time-warped, amplitude-scaled on its dynamic part, resampled,
  rotated so its mean gravity direction matches the background's local gravity,
  then crossfaded in over 0.2-0.4 s at each seam;
* insertions never overlap each other, a RecoFit exercise set, or a source-junk
  interval (device taps / device on table), and prefer low-motion background;
* a small whole-recording axis rotation (<= 15 degrees) is applied last so the
  background and every insert share one sensor frame;
* the generator is a pure function of ``SYNTHESIS_CONFIG`` and the two source
  caches, so the cache builder reproduces it bit-for-bit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


DATASET = "synth_wrist_v1"
_DATA_ROOT = Path(__file__).resolve().parents[1]

SYNTHESIS_CONFIG: dict[str, Any] = {
    "version": 1,
    "seed": 20260902,
    "donor_dataset": "crossfit",
    "background_dataset": "recofit",
    "target_hours": 40.0,
    "background_seconds": [60.0, 120.0],
    "donor_excluded_labels": ["Null"],
    "donor_duration_bounds_sec": [1.0, 8.0],
    # Number of inserted executions of the recording's primary exercise.
    "primary_insert_counts": [0, 1, 2, 3, 4],
    "primary_insert_weights": [0.20, 0.25, 0.25, 0.15, 0.15],
    # Number of inserted executions of other exercises (same procedure).
    "distractor_insert_counts": [0, 1, 2],
    "distractor_insert_weights": [0.40, 0.35, 0.25],
    "same_subject_primary_fraction": 0.2,
    "time_warp_bounds": [0.85, 1.15],
    "amplitude_bounds": [0.8, 1.2],
    "noise_sd_acc_g": 0.01,
    "noise_sd_gyro_rad_s": 0.02,
    "crossfade_bounds_sec": [0.2, 0.4],
    "edge_margin_sec": 2.0,
    "insert_spacing_sec": 1.5,
    "candidate_grid_sec": 0.25,
    "low_motion_quantile": 0.5,
    "whole_query_rotation_max_deg": 15.0,
    "background_junk_kinds": ["source_junk"],
    "background_set_kind": "set",
}


@dataclass(frozen=True)
class DonorClip:
    clip_id: str
    label: str
    subject_id: str
    exercise_id: int
    repetition_index: int
    values: np.ndarray  # [T, 6] at ``rate_hz``
    rate_hz: float


@dataclass(frozen=True)
class BackgroundSession:
    cache_index: int
    recording_id: str
    subject_id: str
    timestamps_sec: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    rate_hz: float
    set_intervals: tuple[tuple[float, float, str], ...]
    junk_intervals: tuple[tuple[float, float], ...]
    stream: SensorStream


def config_digest(config: dict[str, Any] = SYNTHESIS_CONFIG) -> str:
    return sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_root(dataset: str, root: Path | None) -> Path:
    base = Path(root) if root is not None else _DATA_ROOT / "sources"
    return base / dataset / "processed" / "canonical_v1"


# --------------------------------------------------------------------------- banks


def load_donor_bank(
    cache: CachedRecordingDataset, config: dict[str, Any] = SYNTHESIS_CONFIG
) -> tuple[DonorClip, ...]:
    """Standalone single-repetition clips with a full, valid, bounded extent."""

    low, high = config["donor_duration_bounds_sec"]
    excluded = set(config["donor_excluded_labels"])
    clips: list[DonorClip] = []
    for recording in cache:
        if not recording.metadata.get("duplicates_parent_exercise_signal"):
            continue
        if recording.metadata.get("pseudo_repetition_fragment"):
            continue
        if len(recording.events) != 1 or len(recording.streams) != 1:
            continue
        event = recording.events[0]
        stream = recording.streams[0]
        if event.annotation_kind != "repetition" or event.label in excluded:
            continue
        duration = float(event.end_sec - event.start_sec)
        if not low <= duration <= high:
            continue
        if not bool(np.asarray(stream.valid).all()):
            continue
        clips.append(
            DonorClip(
                clip_id=recording.recording_id,
                label=event.label,
                subject_id=recording.subject_id,
                exercise_id=int(recording.metadata.get("exercise_id", -1)),
                repetition_index=int(recording.metadata.get("repetition_index", -1)),
                values=np.asarray(stream.values, dtype=np.float32),
                rate_hz=float(stream.nominal_rate_hz),
            )
        )
    clips.sort(key=lambda clip: clip.clip_id)
    if not clips:
        raise ValueError("donor bank is empty")
    return tuple(clips)


def load_background_index(
    cache: CachedRecordingDataset, config: dict[str, Any] = SYNTHESIS_CONFIG
) -> tuple[BackgroundSession, ...]:
    junk_kinds = set(config["background_junk_kinds"])
    set_kind = config["background_set_kind"]
    minimum = float(config["background_seconds"][0]) + 2 * float(config["edge_margin_sec"])
    sessions: list[BackgroundSession] = []
    for cache_index, recording in enumerate(cache):
        if len(recording.streams) != 1:
            continue
        stream = recording.streams[0]
        timestamps = np.asarray(stream.timestamps_sec, dtype=np.float64)
        if timestamps[-1] - timestamps[0] < minimum:
            continue
        sessions.append(
            BackgroundSession(
                cache_index=cache_index,
                recording_id=recording.recording_id,
                subject_id=recording.subject_id,
                timestamps_sec=timestamps,
                values=np.asarray(stream.values, dtype=np.float32),
                valid=np.asarray(stream.valid),
                rate_hz=float(stream.nominal_rate_hz),
                set_intervals=tuple(
                    (float(event.start_sec), float(event.end_sec), event.label)
                    for event in recording.events
                    if event.annotation_kind == set_kind
                ),
                junk_intervals=tuple(
                    (float(event.start_sec), float(event.end_sec))
                    for event in recording.events
                    if event.annotation_kind in junk_kinds
                ),
                stream=stream,
            )
        )
    if not sessions:
        raise ValueError("background index is empty")
    return tuple(sessions)


# ------------------------------------------------------------------ signal helpers


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``source`` onto unit vector ``target``."""

    a = source / max(np.linalg.norm(source), 1e-9)
    b = target / max(np.linalg.norm(target), 1e-9)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees about any axis orthogonal to ``a``.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _rotate(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[:, :3] = out[:, :3] @ rotation.T
    out[:, 3:6] = out[:, 3:6] @ rotation.T
    return out


def _resample(values: np.ndarray, source_rate: float, target_rate: float, warp: float) -> np.ndarray:
    """Linear resampling of a donor whose duration is stretched by ``warp``."""

    duration = len(values) / source_rate * warp
    count = max(2, int(round(duration * target_rate)))
    source_times = np.arange(len(values)) / source_rate * warp
    target_times = np.arange(count) / target_rate
    target_times = np.clip(target_times, 0.0, source_times[-1])
    return np.stack(
        [np.interp(target_times, source_times, values[:, channel]) for channel in range(values.shape[1])],
        axis=1,
    )


def _motion_energy(values: np.ndarray, rate_hz: float, window_sec: float = 1.0) -> np.ndarray:
    """Per-sample rolling mean of |acc - local gravity| (g)."""

    acc = np.asarray(values[:, :3], dtype=np.float64)
    width = max(1, int(round(window_sec * rate_hz)))
    kernel = np.ones(width) / width
    gravity = np.stack([np.convolve(acc[:, i], kernel, mode="same") for i in range(3)], axis=1)
    dynamic = np.linalg.norm(acc - gravity, axis=1)
    return np.convolve(dynamic, kernel, mode="same")


# ----------------------------------------------------------------- one recording


def _overlaps(start: float, end: float, intervals: Sequence[tuple[float, float]], pad: float = 0.0) -> bool:
    return any(end + pad > left and start - pad < right for left, right in intervals)


def _choose_window(
    rng: np.random.Generator,
    sessions: Sequence[BackgroundSession],
    session_weights: np.ndarray,
    config: dict[str, Any],
) -> tuple[BackgroundSession, float, float] | None:
    low, high = config["background_seconds"]
    for _ in range(50):
        session = sessions[int(rng.choice(len(sessions), p=session_weights))]
        length = float(rng.uniform(low, high))
        span = session.timestamps_sec[-1] - session.timestamps_sec[0]
        if span <= length:
            continue
        start = session.timestamps_sec[0] + float(rng.uniform(0.0, span - length))
        end = start + length
        if _overlaps(start, end, session.junk_intervals):
            continue
        left = int(np.searchsorted(session.timestamps_sec, start, side="left"))
        right = int(np.searchsorted(session.timestamps_sec, end, side="left"))
        if right - left < int(low * session.rate_hz) or not session.valid[left:right].all():
            continue
        return session, start, end
    return None


def _draw_donor(
    rng: np.random.Generator,
    clips_by_label_subject: dict[str, dict[str, list[DonorClip]]],
    label: str,
    *,
    subject: str | None = None,
    exclude_subject: str | None = None,
) -> DonorClip:
    by_subject = clips_by_label_subject[label]
    subjects = sorted(by_subject)
    if subject is not None and subject in by_subject:
        chosen_subject = subject
    else:
        pool = [item for item in subjects if item != exclude_subject] or subjects
        chosen_subject = pool[int(rng.integers(len(pool)))]
    clips = by_subject[chosen_subject]
    return clips[int(rng.integers(len(clips)))]


def synthesize_recording(
    index: int,
    rng: np.random.Generator,
    *,
    donors: Sequence[DonorClip],
    sessions: Sequence[BackgroundSession],
    session_weights: np.ndarray,
    config: dict[str, Any] = SYNTHESIS_CONFIG,
) -> RawRecording | None:
    labels = sorted({clip.label for clip in donors})
    clips_by_label_subject: dict[str, dict[str, list[DonorClip]]] = {}
    for clip in donors:
        clips_by_label_subject.setdefault(clip.label, {}).setdefault(clip.subject_id, []).append(clip)

    window = _choose_window(rng, sessions, session_weights, config)
    if window is None:
        return None
    session, window_start, window_end = window
    rate = session.rate_hz
    left = int(np.searchsorted(session.timestamps_sec, window_start, side="left"))
    right = int(np.searchsorted(session.timestamps_sec, window_end, side="left"))
    timestamps = session.timestamps_sec[left:right] - session.timestamps_sec[left]
    signal = np.asarray(session.values[left:right], dtype=np.float64).copy()
    length_sec = float(timestamps[-1]) + 1.0 / rate

    # Background annotations clipped to the window (provenance; never targets).
    background_events: list[EventInterval] = []
    blocked: list[tuple[float, float]] = []
    for set_start, set_end, set_label in session.set_intervals:
        clipped_start = max(set_start, window_start) - window_start
        clipped_end = min(set_end, window_end) - window_start
        if clipped_end - clipped_start <= 0.05:
            continue
        blocked.append((clipped_start, clipped_end))
        background_events.append(
            EventInterval(
                start_sec=float(clipped_start),
                end_sec=float(clipped_end),
                label=set_label,
                annotation_kind="background_activity",
                metadata={
                    "source_dataset": config["background_dataset"],
                    "source_recording_id": session.recording_id,
                    "clipped_to_window": bool(
                        set_start < window_start or set_end > window_end
                    ),
                },
            )
        )

    primary = labels[int(rng.integers(len(labels)))]
    reference_subject = sorted(clips_by_label_subject[primary])[
        int(rng.integers(len(clips_by_label_subject[primary])))
    ]
    primary_count = int(
        rng.choice(config["primary_insert_counts"], p=config["primary_insert_weights"])
    )
    distractor_count = int(
        rng.choice(config["distractor_insert_counts"], p=config["distractor_insert_weights"])
    )
    plan: list[tuple[str, DonorClip]] = []
    for _ in range(primary_count):
        same = rng.uniform() < config["same_subject_primary_fraction"]
        plan.append(
            (
                "primary",
                _draw_donor(
                    rng,
                    clips_by_label_subject,
                    primary,
                    subject=reference_subject if same else None,
                    exclude_subject=None if same else reference_subject,
                ),
            )
        )
    others = [label for label in labels if label != primary]
    for _ in range(distractor_count):
        label = others[int(rng.integers(len(others)))]
        plan.append(("distractor", _draw_donor(rng, clips_by_label_subject, label)))
    order = rng.permutation(len(plan))
    plan = [plan[i] for i in order]

    energy = _motion_energy(signal, rate)
    grid = float(config["candidate_grid_sec"])
    margin = float(config["edge_margin_sec"])
    spacing = float(config["insert_spacing_sec"])
    warp_low, warp_high = config["time_warp_bounds"]
    amp_low, amp_high = config["amplitude_bounds"]
    fade_low, fade_high = config["crossfade_bounds_sec"]
    inserted_events: list[EventInterval] = []
    occupied: list[tuple[float, float]] = []
    for role, clip in plan:
        warp = float(rng.uniform(warp_low, warp_high))
        amplitude = float(rng.uniform(amp_low, amp_high))
        fade = float(rng.uniform(fade_low, fade_high))
        donor = _resample(clip.values, clip.rate_hz, rate, warp)
        duration = len(donor) / rate
        starts = np.arange(margin, length_sec - margin - duration, grid)
        candidates = [
            s
            for s in starts
            if not _overlaps(s, s + duration, occupied, pad=spacing)
            and not _overlaps(s, s + duration, blocked, pad=spacing)
        ]
        if not candidates:
            continue
        scores = np.array(
            [
                float(
                    energy[
                        int(round(s * rate)) : int(round((s + duration) * rate))
                    ].mean()
                )
                for s in candidates
            ]
        )
        threshold = float(np.quantile(scores, config["low_motion_quantile"]))
        quiet = [s for s, score in zip(candidates, scores) if score <= threshold]
        start_sec = float(quiet[int(rng.integers(len(quiet)))])
        start = int(round(start_sec * rate))
        stop = start + len(donor)
        if stop > len(signal):
            continue

        # Amplitude scaling on the dynamic part only; gravity stays 1 g.
        donor_gravity = donor[:, :3].mean(axis=0)
        donor[:, :3] = donor_gravity + amplitude * (donor[:, :3] - donor_gravity)
        donor[:, 3:6] *= amplitude
        donor[:, :3] += rng.normal(0.0, config["noise_sd_acc_g"], donor[:, :3].shape)
        donor[:, 3:6] += rng.normal(0.0, config["noise_sd_gyro_rad_s"], donor[:, 3:6].shape)

        # Gravity alignment: donor mean gravity -> background local gravity.
        local_gravity = signal[start:stop, :3].mean(axis=0)
        rotation = _rotation_between(donor_gravity, local_gravity)
        donor = _rotate(donor, rotation)
        alignment_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
        )

        fade_samples = max(1, min(int(round(fade * rate)), len(donor) // 2))
        weight = np.ones(len(donor))
        ramp = np.linspace(0.0, 1.0, fade_samples + 2)[1:-1]
        weight[:fade_samples] = ramp
        weight[-fade_samples:] = ramp[::-1]
        signal[start:stop] = (
            weight[:, None] * donor + (1.0 - weight[:, None]) * signal[start:stop]
        )
        occupied.append((start_sec, start_sec + duration))
        inserted_events.append(
            EventInterval(
                start_sec=float(start / rate),
                end_sec=float(stop / rate),
                label=clip.label,
                annotation_kind="inserted_execution",
                metadata={
                    "role": role,
                    "donor_dataset": config["donor_dataset"],
                    "donor_clip_id": clip.clip_id,
                    "donor_subject_id": clip.subject_id,
                    "donor_exercise_id": clip.exercise_id,
                    "donor_repetition_index": clip.repetition_index,
                    "time_warp": warp,
                    "amplitude": amplitude,
                    "crossfade_sec": fade,
                    "gravity_alignment_deg": alignment_deg,
                    "guard_sec": fade,
                },
            )
        )

    # Whole-recording rotation so background and inserts share a sensor frame.
    axis = rng.normal(size=3)
    angle = float(np.radians(rng.uniform(0.0, config["whole_query_rotation_max_deg"])))
    signal = _rotate(signal, _axis_angle(axis, angle))

    events = sorted(inserted_events + background_events, key=lambda e: (e.start_sec, e.label))
    stream = SensorStream(
        stream_id="wrist_imu",
        placement=session.stream.placement,
        device="synthetic wrist IMU (RecoFit background, CrossFit donors)",
        timestamps_sec=timestamps,
        values=signal.astype(np.float32),
        channels=session.stream.channels,
        valid=np.ones(signal.shape, dtype=bool),
        gravity_state="present",
        nominal_rate_hz=rate,
        metadata={
            "synthetic": True,
            "background_dataset": config["background_dataset"],
            "background_recording_id": session.recording_id,
            "background_cache_index": session.cache_index,
            "background_window_sec": [float(window_start), float(window_end)],
            "whole_query_rotation_deg": float(np.degrees(angle)),
            "acceleration_unit": "g",
            "gyroscope_unit": "rad/s",
        },
    )
    recording_id = f"{DATASET}:{index:05d}"
    return RawRecording(
        dataset=DATASET,
        recording_id=recording_id,
        subject_id=f"bg:{session.subject_id}",
        session_id=recording_id,
        streams=(stream,),
        events=tuple(events),
        metadata={
            "synthetic": True,
            "synthesis_config_sha256": config_digest(config),
            "primary_label": primary,
            "reference_subject_id": reference_subject,
            "inserted_primary_count": sum(
                1 for e in inserted_events if e.metadata["role"] == "primary"
            ),
            "inserted_distractor_count": sum(
                1 for e in inserted_events if e.metadata["role"] == "distractor"
            ),
        },
    )


# ------------------------------------------------------------------- adapter API


def iter_recordings(
    *, root: Path | None = None, limit: int | None = None
) -> Iterator[RawRecording]:
    config = SYNTHESIS_CONFIG
    donors = load_donor_bank(
        CachedRecordingDataset(_canonical_root(config["donor_dataset"], root)), config
    )
    sessions = load_background_index(
        CachedRecordingDataset(_canonical_root(config["background_dataset"], root)),
        config,
    )
    spans = np.array([s.timestamps_sec[-1] - s.timestamps_sec[0] for s in sessions])
    session_weights = spans / spans.sum()
    rng = np.random.default_rng(int(config["seed"]))
    budget = float(config["target_hours"]) * 3600.0
    produced_sec = 0.0
    produced = 0
    index = 0
    while produced_sec < budget and (limit is None or produced < limit):
        recording = synthesize_recording(
            index,
            rng,
            donors=donors,
            sessions=sessions,
            session_weights=session_weights,
            config=config,
        )
        index += 1
        if recording is None:
            continue
        stream = recording.streams[0]
        produced_sec += float(stream.timestamps_sec[-1] - stream.timestamps_sec[0])
        produced += 1
        yield recording
