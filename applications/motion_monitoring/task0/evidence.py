"""Native-time physical motion evidence for Task 0."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import lfilter

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.task0.contracts import (
    EvidenceConfig,
    EvidenceSequence,
    FEATURE_NAMES,
    RobustFeatureScaler,
)


_ACC_CHANNELS = ("acc_x", "acc_y", "acc_z")
_GYRO_CHANNELS = ("gyro_x", "gyro_y", "gyro_z")


def _triad(
    stream: SensorStream, names: tuple[str, str, str]
) -> tuple[np.ndarray, np.ndarray] | None:
    if not all(name in stream.channels for name in names):
        return None
    indices = [stream.channels.index(name) for name in names]
    values = np.asarray(stream.values[:, indices], dtype=np.float64)
    valid = np.asarray(stream.valid[:, indices], dtype=bool) & np.isfinite(values)
    return values, valid


def _true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    yield from zip(edges[::2], edges[1::2])


def _one_pole(
    values: np.ndarray, timestamps: np.ndarray, cutoff_hz: float
) -> np.ndarray:
    differences = np.diff(timestamps)
    median_dt = float(np.median(differences))
    if np.allclose(differences, median_dt, rtol=1e-3, atol=1e-9):
        alpha = np.exp(-2.0 * np.pi * cutoff_hz * median_dt)
        initial_state = (alpha * values[0])[None, :]
        output, _ = lfilter(
            [1.0 - alpha], [1.0, -alpha], values, axis=0, zi=initial_state
        )
        return output
    output = np.empty_like(values, dtype=np.float64)
    output[0] = values[0]
    for index in range(1, len(values)):
        dt = max(float(timestamps[index] - timestamps[index - 1]), np.finfo(float).eps)
        alpha = np.exp(-2.0 * np.pi * cutoff_hz * dt)
        output[index] = alpha * output[index - 1] + (1.0 - alpha) * values[index]
    return output


def _gravity_estimate(
    values: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate gravity without filtering across missing-data runs."""

    row_valid = valid.all(axis=1)
    gravity = np.zeros_like(values, dtype=np.float64)
    estimate_valid = np.zeros(len(values), dtype=bool)
    for start, stop in _true_runs(row_valid):
        if stop - start < 2:
            continue
        run_values = values[start:stop]
        run_times = timestamps[start:stop]
        forward = _one_pole(run_values, run_times, cutoff_hz)
        backward = _one_pole(run_values[::-1], -run_times[::-1], cutoff_hz)[::-1]
        gravity[start:stop] = 0.5 * (forward + backward)
        estimate_valid[start:stop] = True
    return gravity, estimate_valid


def _native_rate(stream: SensorStream) -> float:
    differences = np.diff(stream.timestamps_sec)
    finite = differences[np.isfinite(differences) & (differences > 0)]
    if not len(finite):
        raise ValueError(f"{stream.stream_id}: cannot infer a native sampling rate")
    return float(1.0 / np.median(finite))


def _window_bounds(
    timestamps: np.ndarray, config: EvidenceConfig
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) < config.min_samples:
        return np.empty(0), np.empty(0)
    median_dt = float(np.median(np.diff(timestamps)))
    final_edge = float(timestamps[-1] + median_dt)
    duration = final_edge - float(timestamps[0])
    if duration <= 0:
        return np.empty(0), np.empty(0)
    if duration <= config.window_seconds:
        return np.asarray([timestamps[0]]), np.asarray([final_edge])
    count = (
        int(np.floor((duration - config.window_seconds) / config.stride_seconds)) + 1
    )
    starts = (
        float(timestamps[0])
        + np.arange(count, dtype=np.float64) * config.stride_seconds
    )
    return starts, starts + config.window_seconds


def extract_physical_evidence(
    recording: RawRecording,
    stream: SensorStream,
    config: EvidenceConfig | None = None,
) -> EvidenceSequence:
    """Compute compact physical summaries without resampling or imputation."""

    config = config or EvidenceConfig()
    if not any(candidate is stream for candidate in recording.streams):
        raise ValueError("stream does not belong to the supplied recording")
    acceleration = _triad(stream, _ACC_CHANNELS)
    gyroscope = _triad(stream, _GYRO_CHANNELS)
    if acceleration is None and gyroscope is None:
        raise ValueError(
            f"{stream.stream_id}: Task 0 requires a complete vector-sensor triad"
        )

    timestamps = np.asarray(stream.timestamps_sec, dtype=np.float64)
    starts, ends = _window_bounds(timestamps, config)
    count = len(starts)
    features = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64)
    feature_valid = np.zeros_like(features, dtype=bool)
    valid_fraction = np.zeros(count, dtype=np.float64)
    constant_fraction = np.zeros(count, dtype=np.float64)

    dynamic_acc: np.ndarray | None = None
    dynamic_valid: np.ndarray | None = None
    if acceleration is not None:
        acc_values, acc_valid = acceleration
        row_valid = acc_valid.all(axis=1)
        if stream.gravity_state == "removed":
            dynamic_acc = acc_values
            dynamic_valid = row_valid
        else:
            gravity, gravity_valid = _gravity_estimate(
                acc_values, acc_valid, timestamps, config.gravity_cutoff_hz
            )
            dynamic_acc = acc_values - gravity
            dynamic_valid = row_valid & gravity_valid

    gyro_values: np.ndarray | None = None
    gyro_valid: np.ndarray | None = None
    if gyroscope is not None:
        gyro_values, gyro_mask = gyroscope
        gyro_valid = gyro_mask.all(axis=1)

    rate_hz = _native_rate(stream)
    left = np.searchsorted(timestamps, starts, side="left")
    right = np.searchsorted(timestamps, ends, side="left")
    expected = np.maximum(
        config.min_samples, np.rint((ends - starts) * rate_hz).astype(np.int64)
    )
    modality_fractions: list[np.ndarray] = []
    modality_constant: list[np.ndarray] = []
    modality_window_valid: list[np.ndarray] = []

    def summarize_modality(
        values: np.ndarray, row_valid: np.ndarray, feature_index: int
    ) -> None:
        counts_prefix = np.concatenate(([0], np.cumsum(row_valid, dtype=np.int64)))
        counts = counts_prefix[right] - counts_prefix[left]
        fractions = np.minimum(1.0, counts / expected)
        valid_windows = (
            (counts >= config.min_samples)
            & (fractions >= config.min_valid_fraction)
            & ((right - left) >= config.min_samples)
        )

        safe_values = np.where(row_valid[:, None], values, 0.0)
        norm_squared = np.sum(safe_values * safe_values, axis=1)
        energy_prefix = np.concatenate(([0.0], np.cumsum(norm_squared)))
        energy = energy_prefix[right] - energy_prefix[left]
        features[valid_windows, feature_index] = np.sqrt(
            energy[valid_windows] / counts[valid_windows]
        )
        feature_valid[:, feature_index] = valid_windows

        value_prefix = np.vstack((np.zeros((1, 3)), np.cumsum(safe_values, axis=0)))
        square_prefix = np.vstack(
            (np.zeros((1, 3)), np.cumsum(safe_values * safe_values, axis=0))
        )
        sums = value_prefix[right] - value_prefix[left]
        square_sums = square_prefix[right] - square_prefix[left]
        denominator = np.maximum(counts, 1)[:, None]
        variance = np.maximum(
            0.0, square_sums / denominator - (sums / denominator) ** 2
        )
        constant = np.ones(count, dtype=np.float64)
        constant[valid_windows] = np.mean(
            variance[valid_windows] <= config.constant_tolerance**2, axis=1
        )
        modality_fractions.append(fractions)
        modality_constant.append(constant)
        modality_window_valid.append(valid_windows)

    if dynamic_acc is not None and dynamic_valid is not None:
        summarize_modality(dynamic_acc, dynamic_valid, 0)
    if gyro_values is not None and gyro_valid is not None:
        summarize_modality(gyro_values, gyro_valid, 1)

    if modality_fractions:
        valid_fraction[:] = np.min(np.vstack(modality_fractions), axis=0)
        constant_values = np.vstack(modality_constant)
        constant_valid = np.vstack(modality_window_valid)
        valid_modality_count = constant_valid.sum(axis=0)
        has_valid_modality = valid_modality_count > 0
        constant_fraction[:] = 1.0
        constant_fraction[has_valid_modality] = (
            np.sum(
                constant_values[:, has_valid_modality]
                * constant_valid[:, has_valid_modality],
                axis=0,
            )
            / valid_modality_count[has_valid_modality]
        )
    else:
        constant_fraction[:] = 1.0

    return EvidenceSequence(
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        stream_id=stream.stream_id,
        placement=stream.placement,
        window_start_sec=starts,
        window_end_sec=ends,
        features=features,
        feature_valid=feature_valid,
        valid_fraction=valid_fraction,
        constant_fraction=constant_fraction,
        metadata={
            "device": stream.device,
            "channels": list(stream.channels),
            "gravity_state": stream.gravity_state,
            "measured_rate_hz": rate_hz,
            "nominal_rate_hz": stream.nominal_rate_hz,
            "evidence_config": {
                "window_seconds": config.window_seconds,
                "stride_seconds": config.stride_seconds,
                "gravity_cutoff_hz": config.gravity_cutoff_hz,
                "min_valid_fraction": config.min_valid_fraction,
            },
        },
    )


def fit_robust_scaler(
    sequences: Iterable[EvidenceSequence], *, max_values_per_feature: int = 1_000_000
) -> RobustFeatureScaler:
    """Fit development-only median/MAD scaling with deterministic memory bounds."""

    if max_values_per_feature <= 0:
        raise ValueError("max_values_per_feature must be positive")
    buckets: list[list[np.ndarray]] = [[] for _ in FEATURE_NAMES]
    bucket_sizes = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    for sequence in sequences:
        for feature_index in range(len(FEATURE_NAMES)):
            mask = sequence.feature_valid[:, feature_index]
            values = sequence.features[mask, feature_index]
            if len(values):
                buckets[feature_index].append(np.asarray(values, dtype=np.float64))
                bucket_sizes[feature_index] += len(values)
                if bucket_sizes[feature_index] > 2 * max_values_per_feature:
                    combined = np.concatenate(buckets[feature_index])
                    indices = np.linspace(
                        0,
                        len(combined) - 1,
                        max_values_per_feature,
                        dtype=np.int64,
                    )
                    buckets[feature_index] = [combined[indices]]
                    bucket_sizes[feature_index] = max_values_per_feature

    center = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    observed = np.zeros(len(FEATURE_NAMES), dtype=bool)
    for index, parts in enumerate(buckets):
        if not parts:
            continue
        values = np.concatenate(parts)
        if len(values) > max_values_per_feature:
            sample_indices = np.linspace(
                0, len(values) - 1, max_values_per_feature, dtype=np.int64
            )
            values = values[sample_indices]
        median = float(np.median(values))
        estimate = 1.4826 * float(np.median(np.abs(values - median)))
        if estimate <= np.finfo(float).eps:
            q25, q75 = np.percentile(values, [25.0, 75.0])
            estimate = float((q75 - q25) / 1.349)
        if estimate <= np.finfo(float).eps:
            estimate = float(np.std(values))
        center[index] = median
        scale[index] = estimate if estimate > np.finfo(float).eps else 1.0
        observed[index] = True
    if not observed.any():
        raise ValueError(
            "cannot fit Task-0 scaling: no valid physical evidence was observed"
        )
    return RobustFeatureScaler(center=center, scale=scale, observed=observed)


def standardized_motion_score(
    sequence: EvidenceSequence, scaler: RobustFeatureScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Combine positive robust deviations while respecting unavailable modalities."""

    standardized = scaler.transform(sequence.features, sequence.feature_valid)
    available = sequence.feature_valid & scaler.observed
    positive = np.maximum(standardized, 0.0)
    scores = np.zeros(len(sequence.features), dtype=np.float64)
    valid = available.any(axis=1)
    scores[valid] = np.sqrt(np.sum(positive[valid] ** 2, axis=1))
    return scores, valid
