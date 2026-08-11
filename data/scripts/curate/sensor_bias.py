"""``sensor_bias`` — activity-invariant channel physics, per sensor.

The third vector of the design of record (docs/design/DESIGN_OF_RECORD.md). Where
``text_descriptor`` says what the technician *declared* a sensor to be, ``sensor_bias`` says what the
signal *measurably* is. The two disagree more often than the field assumes — our own corpus audit
found recgym min-max normalised to [0,1], kuhar and uci_har shipping gravity-removed accelerometry,
and a g-vs-m/s^2 unit fingerprint leaking through the DC feature — and every one of those silently
corrupted cross-stream comparability.

WHY EVERY FIELD IS ACTIVITY-INVARIANT
-------------------------------------
The descriptor is pooled over *all* of a sensor's recordings. Datasets have skewed activity
distributions (a gym study's wrist sees mostly exercise; a free-living study's sees mostly
sedentary), so any field that responds to *what the person did* would make this vector a dataset
fingerprint, and retrieval conditioned on it would degenerate into "retrieve from the same dataset" —
a leakage trap of exactly the shape as hapt==uci_har. Every field below is therefore a property of
the acquisition CHANNEL, estimated on quiescent windows or from the signal's quantisation/range
structure, never from motion content.

``scan_fingerprint_risk`` is the falsifiable guard: a descriptor that predicts dataset identity
better than placement alone does has failed, and must be pruned before use.

WHY EVERY FIELD IS A NORM OR AN AGGREGATE
-----------------------------------------
So that the SO(3) rotation augmentation leaves the descriptor unchanged. That is a design constraint,
not a coincidence: it is what lets the augmentation pushforward (``transform_bias``) stay exact.

Run:
    python -m data.scripts.curate.sensor_bias --build      # write per-sensor descriptors
    python -m data.scripts.curate.sensor_bias --check      # fingerprint-risk guard
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from data.scripts.eda.grid_io import discover_grids

OUT_PATH = Path("data/scripts/curate/sensor_bias.json")

# ----------------------------------------------------------------------------------------------
# Constants — each pinned to a physical reason, not tuned.
# ----------------------------------------------------------------------------------------------

# Human motion is band-limited to roughly 20 Hz (the filterbank's analysis band tops out at 15 Hz
# for this reason). Energy above MOTION_BAND_HZ is therefore instrument, not person — which is what
# makes it a channel property rather than an activity property.
MOTION_BAND_HZ = 20.0

# A window is "quiescent" if its per-channel std sits in the lowest QUIESCENT_PCT of the stream.
# Estimating gravity magnitude and noise floor on quiescent windows is what removes activity
# dependence: at rest, an accelerometer reads gravity plus instrument noise and nothing else.
QUIESCENT_PCT = 10.0

# Cap on windows scanned per stream. These statistics converge fast (they are medians and low-order
# aggregates over per-window scalars), and the corpus is ~305k windows; scanning all of it buys
# nothing but wall-clock.
MAX_WINDOWS_PER_STREAM = 4000

# Rails detection: a sample within RAIL_TOL of the stream's observed extreme counts as clipped.
RAIL_TOL = 1e-3

ACCEL_SLICE = slice(0, 3)
GYRO_SLICE = slice(3, 6)

# Field order is the vector layout. Keep it stable: the bank stores these by position.
FIELDS = (
    "gravity_magnitude",     # median ||acc|| on quiescent windows      (accel only)
    "gravity_presence",      # median |per-channel mean| = |DC|         (accel only)
    "noise_floor",           # rms energy above MOTION_BAND_HZ, quiescent
    "quantization_step",     # smallest positive gap between adjacent distinct values
    "clip_fraction",         # fraction of samples at an observed rail
    "rate_fidelity",         # measured/declared effective bandwidth ratio
    "dropout_fraction",      # fraction of masked/absent samples
    "rest_bias",             # median ||signal|| on quiescent windows   (gyro: rest offset)
    "dynamic_range",         # p99.9 - p0.1 of magnitude
)


@dataclass(frozen=True)
class SensorBias:
    """One sensor's descriptor. ``modality`` is 'accel' or 'gyro'."""
    dataset: str
    stream: str
    modality: str
    n_windows: int
    values: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Field estimators
# ----------------------------------------------------------------------------------------------

def _quiescent_index(data: np.ndarray) -> np.ndarray:
    """Rows whose total per-window std is in the lowest QUIESCENT_PCT.

    Uses the summed per-channel std as the activity proxy. Returns at least one row so a stream of
    uniformly-active windows still yields an estimate (with the honest consequence that its
    'quiescent' statistics are an upper bound, which the caller can see via n_windows).
    """
    if data.size == 0:
        return np.zeros(0, dtype=np.int64)
    activity = data.std(axis=1).sum(axis=1)          # (N,)
    cut = np.percentile(activity, QUIESCENT_PCT)
    idx = np.flatnonzero(activity <= cut)
    return idx if idx.size else np.array([int(np.argmin(activity))])


def _high_band_rms(data: np.ndarray, rate_hz: float) -> float:
    """RMS of spectral content above MOTION_BAND_HZ — instrument noise, not motion.

    Returns NaN when the Nyquist frequency sits at or below the motion band: at <= 40 Hz there is no
    out-of-band content to measure, and inventing a number there would be exactly the kind of
    unsupported cell this project reports as unsupported.
    """
    nyquist = rate_hz / 2.0
    if nyquist <= MOTION_BAND_HZ or data.size == 0:
        return float("nan")
    n = data.shape[1]
    spec = np.abs(np.fft.rfft(data - data.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    band = freqs > MOTION_BAND_HZ
    if not band.any():
        return float("nan")
    # Parseval-normalised so the value is comparable across window lengths and rates.
    return float(np.sqrt((spec[:, band, :] ** 2).sum(axis=1).mean()) / np.sqrt(n))


def _quantization_step(values: np.ndarray) -> float:
    """Smallest positive gap between adjacent distinct sample values.

    A 16-bit sensor at +-8 g quantises to ~2.4e-4 g; a float-normalised stream shows ~1e-7 or the
    resolution of whatever transform was applied. This is the field that catches "someone has already
    processed this data" independent of any activity content.
    """
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size < 2:
        return float("nan")
    # Subsample for cost; the minimum gap is a robust statistic under subsampling because it is
    # determined by the ADC grid, which every sample lies on.
    if flat.size > 200_000:
        flat = flat[:: flat.size // 200_000]
    uniq = np.unique(flat)
    if uniq.size < 2:
        return float("nan")
    gaps = np.diff(uniq)
    gaps = gaps[gaps > 0]
    return float(np.min(gaps)) if gaps.size else float("nan")


def _clip_fraction(values: np.ndarray) -> float:
    """Fraction of samples sitting at the observed extremes — saturation behaviour."""
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    lo, hi = float(flat.min()), float(flat.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    span = hi - lo
    at_rail = (flat >= hi - RAIL_TOL * span) | (flat <= lo + RAIL_TOL * span)
    return float(at_rail.mean())


def _rate_fidelity(data: np.ndarray, rate_hz: float) -> float:
    """Fraction of spectral energy below the declared Nyquist that is genuinely occupied.

    A stream declared at 100 Hz but actually upsampled from 20 Hz has near-zero energy above 10 Hz.
    That is a *declared-vs-real* rate discrepancy, and it is invisible to the metadata.
    """
    if data.size == 0 or rate_hz <= 0:
        return float("nan")
    n = data.shape[1]
    spec = np.abs(np.fft.rfft(data - data.mean(axis=1, keepdims=True), axis=1)) ** 2
    total = spec.sum(axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    # Energy in the upper half of the declared band. A faithfully-sampled stream has some; an
    # upsampled one has essentially none.
    upper = freqs > (rate_hz / 4.0)
    if not upper.any():
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, spec[:, upper, :].sum(axis=1) / total, np.nan)
    return float(np.nanmedian(frac))


def _dropout_fraction(data: np.ndarray) -> float:
    """Fraction of non-finite or exactly-zero-run samples — transmission character."""
    flat = data.reshape(-1)
    if flat.size == 0:
        return float("nan")
    return float((~np.isfinite(flat)).mean())


# ----------------------------------------------------------------------------------------------
# Per-sensor assembly
# ----------------------------------------------------------------------------------------------

def compute_sensor_bias(data: np.ndarray, rate_hz: float, modality: str) -> tuple[float, ...]:
    """Descriptor for one modality's 3 channels of one stream.

    ``data`` is (N, T, 3) for this modality only. Fields that cannot be estimated for this modality
    or at this rate are NaN — an unsupported cell is reported as unsupported, never filled.
    """
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"expected (N, T, 3) for one sensor, got {data.shape}")
    quiet = _quiescent_index(data)
    q = data[quiet]
    magnitude = np.linalg.norm(data, axis=2)                      # (N, T)
    q_magnitude = np.linalg.norm(q, axis=2)

    # Gravity fields are accelerometer-only: a gyroscope has no gravity component, so reporting a
    # number there would be a fabrication rather than a measurement.
    if modality == "accel":
        gravity_magnitude = float(np.nanmedian(q_magnitude))
        gravity_presence = float(np.nanmedian(np.abs(q.mean(axis=1))))
    else:
        gravity_magnitude = float("nan")
        gravity_presence = float("nan")

    finite_mag = magnitude[np.isfinite(magnitude)]
    dynamic_range = (float(np.percentile(finite_mag, 99.9) - np.percentile(finite_mag, 0.1))
                     if finite_mag.size else float("nan"))

    return (
        gravity_magnitude,
        gravity_presence,
        _high_band_rms(q, rate_hz),
        _quantization_step(data),
        _clip_fraction(data),
        _rate_fidelity(data, rate_hz),
        _dropout_fraction(data),
        float(np.nanmedian(q_magnitude)),
        dynamic_range,
    )


def build(limit_streams: int | None = None) -> list[SensorBias]:
    """Scan the native grids and emit one descriptor per (stream, present modality)."""
    out: list[SensorBias] = []
    refs = sorted(discover_grids("native"), key=lambda r: (r.dataset, r.stream))
    if limit_streams:
        refs = refs[:limit_streams]
    for ref in refs:
        data = ref.load_data()
        if data.shape[0] > MAX_WINDOWS_PER_STREAM:
            step = data.shape[0] // MAX_WINDOWS_PER_STREAM
            data = data[::step][:MAX_WINDOWS_PER_STREAM]
        mask = np.asarray(ref.mask, dtype=bool)
        for modality, sl in (("accel", ACCEL_SLICE), ("gyro", GYRO_SLICE)):
            if not mask[sl].any():
                continue                                  # modality genuinely absent (capture24)
            values = compute_sensor_bias(data[:, :, sl].astype(np.float64),
                                         float(ref.rate_hz), modality)
            out.append(SensorBias(dataset=ref.dataset, stream=ref.stream, modality=modality,
                                  n_windows=int(data.shape[0]), values=values))
            print(f"  {ref.dataset}/{ref.stream} [{modality}]: "
                  + ", ".join(f"{k}={v:.4g}" for k, v in zip(FIELDS, values)), flush=True)
    return out


def standardize(rows: list[SensorBias]) -> dict:
    """z-score each field across sensors, so the vector is scale-free at consumption.

    Statistics are computed once here and FROZEN into the artifact. A descriptor whose normalisation
    drifts with the corpus is not a measurement, and Phase B compares descriptors across streams.
    """
    raw = np.array([r.values for r in rows], dtype=np.float64)     # (S, F)
    mu = np.nanmean(raw, axis=0)
    sd = np.nanstd(raw, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (raw - mu) / sd
    # NaN (unsupported for this modality/rate) becomes 0 = "at the corpus mean", i.e. carries no
    # evidence either way. The mask records WHICH fields were unsupported so a consumer can tell
    # "average" from "unknown" — they are not the same claim.
    supported = np.isfinite(raw)
    z = np.where(supported, z, 0.0)
    return {
        "fields": list(FIELDS),
        "mu": [None if not np.isfinite(v) else float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "sensors": [
            {**{k: v for k, v in asdict(r).items() if k != "values"},
             "raw": [None if not np.isfinite(v) else float(v) for v in r.values],
             "z": [float(v) for v in z[i]],
             "supported": [bool(v) for v in supported[i]]}
            for i, r in enumerate(rows)
        ],
    }


# ----------------------------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------------------------

def scan_fingerprint_risk(payload: dict) -> dict:
    """How much dataset identity does the descriptor carry beyond placement?

    THE GUARD. ``sensor_bias`` enters the trunk, so if it is really a dataset ID the model can use it
    as a shortcut and retrieval conditioned on it degenerates to same-dataset retrieval — inflating
    every number while looking like the mechanism working.

    Reported as nearest-neighbour dataset purity in descriptor space. A value near 1.0 means the
    descriptor separates datasets cleanly and is a fingerprint. Compare against the placement-only
    baseline: sensors of the same modality at the same declared placement SHOULD look alike, so some
    clustering is physics, not leakage. The excess is the risk.
    """
    rows = payload["sensors"]
    z = np.array([r["z"] for r in rows], dtype=np.float64)
    datasets = [r["dataset"] for r in rows]
    modality = [r["modality"] for r in rows]
    if len(rows) < 3:
        return {"n_sensors": len(rows), "nn_dataset_purity": None, "note": "too few sensors"}

    def _placement(row) -> str:
        try:
            from data.scripts.curate.deployment_policy import get_stream_spec
            return get_stream_spec(row["dataset"], row["stream"]).placement
        except Exception:
            return row["stream"]

    placements = [_placement(r) for r in rows]
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    def _purity(candidate_filter) -> tuple[float | None, int]:
        hits = scored = 0
        for i in range(len(rows)):
            # Only compare within modality: accel and gyro descriptors are not commensurable (the
            # gravity fields are NaN for gyro), so a cross-modality neighbour is meaningless.
            cand = [j for j in range(len(rows))
                    if j != i and modality[j] == modality[i] and candidate_filter(i, j)]
            if not cand:
                continue
            nn = min(cand, key=lambda j: dist[i, j])
            hits += datasets[nn] == datasets[i]
            scored += 1
        return (hits / scored if scored else None), scored

    raw, n_raw = _purity(lambda i, j: True)
    # THE DISCRIMINATING CONTROL. Sensors sharing hardware SHOULD look alike — that is the physics
    # the descriptor exists to capture, and it is not leakage. What would be leakage is the
    # descriptor identifying the dataset even when the two sensors sit at DIFFERENT placements,
    # because then it is keying on provenance rather than on channel physics. Restricting the
    # neighbour pool to different-placement candidates isolates that.
    xplace, n_xplace = _purity(lambda i, j: placements[j] != placements[i])
    n_datasets = len(set(datasets))
    chance = 1.0 / max(n_datasets, 1)

    # MEASURED 2026-08-11: cross-placement purity came back IDENTICAL to raw (0.5337, all 163
    # sensors scored), because the exclusion removes nothing — every stream in this corpus carries
    # its own placement, so "different placement, same dataset" is still "same hardware, same
    # processing pipeline". Dataset identity and acquisition hardware are CONFOUNDED by
    # construction: 34 datasets, one device model each.
    #
    # So this guard cannot separate "descriptor captures channel physics" (intended) from
    # "descriptor is a dataset ID" (leakage) on this corpus, and reporting a pass would be
    # false assurance. It stays as a monitor — a sharp INCREASE across corpus revisions is still
    # informative — but the decisive guards are downstream, where the confound does not apply:
    #
    #   1. Encoder probe: linear probe from the trunk's `feature` output to dataset ID must sit at
    #      chance. `sensor_bias` enters the trunk, so this is what tests whether the representation
    #      absorbed provenance.
    #   2. Phase-B retrieval-provenance guard: does enabling the bias blend shift retrieval toward
    #      the query's own dataset, against a placement-matched baseline? That measures the harm
    #      directly rather than proxying it.
    confounded = (xplace is not None and raw is not None and abs(xplace - raw) < 1e-9)
    return {
        "n_sensors": len(rows),
        "n_datasets": n_datasets,
        "chance": round(chance, 4),
        "nn_dataset_purity": round(raw, 4) if raw is not None else None,
        "nn_dataset_purity_cross_placement": round(xplace, 4) if xplace is not None else None,
        "n_scored_cross_placement": n_xplace,
        "verdict": "INCONCLUSIVE — dataset and hardware are confounded in this corpus",
        "note": ("Placement exclusion removed no candidates, so the two purities are identical: "
                 "each stream has a unique placement and each dataset a single device model. "
                 "Decisive guards are the encoder dataset-ID probe and the Phase-B "
                 "retrieval-provenance check; see module docstring."
                 if confounded else
                 "Placement exclusion was informative; compare the two purities."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="scan grids and write the artifact")
    ap.add_argument("--check", action="store_true", help="run the fingerprint-risk guard")
    ap.add_argument("--limit-streams", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.build:
        rows = build(args.limit_streams)
        payload = standardize(rows)
        payload["guard"] = scan_fingerprint_risk(payload)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n-> {args.out}  ({len(rows)} sensors)")
        print(json.dumps(payload["guard"], indent=2))
    elif args.check:
        payload = json.loads(args.out.read_text())
        print(json.dumps(scan_fingerprint_risk(payload), indent=2))
    else:
        ap.error("pass --build or --check")


if __name__ == "__main__":
    main()
