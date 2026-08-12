"""Convert FORTH-TRACE (SPL, ICS-FORTH) to the HALO session format.

Source: github.com/spl-icsforth/FORTH_TRACE_DATASET, `partX/partXdevN.csv`, 15 participants x 5
Shimmer nodes. No header; 12 columns, per the dataset README:

    0   device id            5   gyroscope y            10  timestamp, milliseconds
    1   accelerometer x      6   gyroscope z            11  activity label, 1..16
    2   accelerometer y      7   magnetometer x
    3   accelerometer z      8   magnetometer y
    4   gyroscope x          9   magnetometer z

Why this dataset is here: it carries a **left wrist and a right wrist simultaneously** — the corpus
is thin on bilateral pairs — plus a torso, a thigh and an ankle, and it labels **9 postural
transitions** explicitly rather than discarding them.

Two things measured here that the README does not say, and one it says that a naive reading would
get wrong:

  * Units. Acceleration is m/s^2 with gravity present (still |a| ~ 9.8). Angular rate is
    **degrees per second** (walking p99 ~ 130), so it is converted to rad/s here — the canonical
    contract is rad/s and the gyroscope is never rescaled downstream.
  * Rate. The README's 51.2 Hz is correct, but the millisecond timestamps ROUND to alternating
    19/20 ms, so a median-of-diff estimate says 50 Hz. The mean over a whole file gives 51.198 Hz.
  * The released timestamp column is NOT usable as an interpolation abscissa — it is serialised to
    six significant figures and degrades to 100 ms steps late in every recording, and part3's is
    outright non-monotonic. Time is therefore reconstructed from the sample index at 51.2 Hz and
    the stamps are consulted only to locate genuine dropouts, which split the recording. See
    `GAP_TOLERANCE_SEC`.
  * The five nodes share one clock and one recording span but return slightly different sample
    counts, so each is truncated to the shortest and they share one reconstructed timebase. That is
    what makes the five placements genuinely simultaneous downstream: windows of one session id
    share an event id and their ordinals line up.

Transitions are short by construction (measured: 2.5-10 s per block, against a 6 s analysis window),
so most transition blocks produce no grid rows. They are still converted — the session store is
lossless and the window length is not a property of the data. See
docs/data/DATASET_EXPANSION_2026-08.md section 8b.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "repo"
NATIVE_RATE = 51.2
DEG_TO_RAD = np.pi / 180.0

# README Table 1, device id -> our column prefix.
DEVICES = {
    1: "left_wrist",
    2: "right_wrist",
    3: "torso",
    4: "right_thigh",
    5: "left_ankle",
}

# README Table 2. Arrows are spelled out; "->" would not survive into label text cleanly.
ACTIVITIES = {
    1: "standing",
    2: "sitting",
    3: "sitting_and_talking",
    4: "walking",
    5: "walking_and_talking",
    6: "climbing_stairs",
    7: "climbing_stairs_and_talking",
    8: "transition_from_standing_to_sitting",
    9: "transition_from_sitting_to_standing",
    10: "transition_from_standing_to_sitting_and_talking",
    11: "transition_from_sitting_and_talking_to_standing",
    12: "transition_from_standing_to_walking",
    13: "transition_from_walking_to_standing",
    14: "transition_from_standing_to_climbing_stairs",
    15: "transition_from_climbing_stairs_to_walking",
    16: "transition_from_climbing_stairs_and_talking_to_walking_and_talking",
}

ACC_MS2_RANGE = (8.0, 11.5)      # dataset median |acc|
GYRO_DEGS_MIN = 20.0             # p99.9 |w| BEFORE conversion; rad/s would never reach this
RATE_TOLERANCE_HZ = 0.5
WINDOW_SECONDS = 6.0             # only used to report how many blocks are too short to window

# The released timestamp column CANNOT be handed to np.interp. It is serialised to six significant
# figures, so its resolution decays as elapsed time grows: past t = 1,000,000 ms the quantum is
# 100 ms while the node keeps emitting 51.18 samples/s, producing runs of five identical stamps.
# That is 3.7% of every ~1040 s recording. Worse, all five nodes of part3 are non-monotonic from
# t = 1.6 s (947-2,350 backwards steps each; the stamps read like two interleaved sub-sequences
# emitted out of order), and np.interp is simply undefined for non-increasing xp.
#
# Measured 2026-08-11, interpolated-against-raw band power over a whole file:
#
#     band        part0dev1  part4dev1  part3dev5
#     0.3-3 Hz      1.00       1.00       1.01
#     3-8 Hz        0.96       0.96       0.98
#     8-15 Hz       0.86       0.89       0.82      <- inside the filterbank's analysis band
#
# In the quantised tail alone, 8-15 Hz lands at 0.47 / 1.74 / 0.29.
#
# The clock is therefore RECONSTRUCTED from the sample index at the documented rate, which is what
# the hardware actually did, and the released stamps are used only to locate genuine dropouts. A
# gap is detected where the stamps advance by more than one sample period plus a tolerance; the
# recording is split there so nothing is interpolated across it. Detection uses a monotone running
# maximum of the stamp column so the quantisation and part3's ordering defect cannot invent gaps.
GAP_TOLERANCE_SEC = 1.0
# Under quantisation the stamps only ever lag reconstructed time; a systematic LEAD would mean the
# true rate is not 51.2 Hz and index-derived time would drift out of the labels.
MAX_CLOCK_LEAD_SEC = 5.0

# All five nodes carry the same annotation track, so their labels must agree once the clocks are
# aligned. Gap-aware alignment retains 14 of 15 participants at >=0.84 agreement. Part8's earlier
# 0.13 score was caused by interpolating across a 1,828 s dropout; splitting that gap restores 0.974.
# Only part4 remains invalid at 0.147, indicating annotation tracks from different takes. Emitting it
# as one frame would assert a simultaneity the data does not support, so it is named in the manifest.
MIN_NODE_AGREEMENT = 0.80


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: one ~1040 s recording per participant.

    Session ids are `{subject}_{activity}_{block}`, one per contiguous label run. Every block of one
    participant comes from the SAME single recording, so two blocks of the same activity are
    minutes apart within one bout and are not independent enrollment executions. Measured
    2026-08-11: 1.7x overcount. FORTH-TRACE therefore supports no same-subject enrollment curve,
    which is a result, not a gap to fill (docs/data/DATASET_EXPANSION_2026-08.md section 8b).
    """
    return session_id.split("_")[0]


def _short_holes_are_covered(occupied: np.ndarray, tolerance: int) -> np.ndarray:
    """Treat an empty run of at most `tolerance` slots as covered, a longer one as a real dropout.

    Slot occupancy is jittery by construction: each node's stamps quantise independently, so their
    occupied-slot sets differ by a sample or two. Intersecting exact occupancy across five nodes
    would shred a clean recording into thousands of one-slot holes. What matters is whether a node
    stopped sampling, which shows up as a run of empty slots seconds long.
    """
    covered = occupied.copy()
    edges = np.flatnonzero(np.diff(np.r_[True, occupied, True].astype(np.int8)))
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start <= tolerance:
            covered[start:stop] = True
    return covered


def _runs(codes: np.ndarray):
    """Yield (label_code, start, stop) for each maximal run of one label."""
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(codes)]):
        yield int(codes[start]), int(start), int(stop)


def create_manifest() -> dict:
    return {
        "dataset_name": "FORTH-TRACE",
        "description": (
            "15 participants wearing 5 Shimmer inertial nodes (left wrist, right wrist, torso, "
            "right thigh, left ankle) performing 7 basic activities and 9 explicitly labelled "
            "postural transitions at 51.2 Hz. Karagiannaki, Panousopoulou & Tsakalides, UbiComp "
            "2016. Source: github.com/spl-icsforth/FORTH_TRACE_DATASET."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{device}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"Shimmer node on the {device.replace('_', ' ')}; "
                             f"{'m/s^2 (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for device in DEVICES.values() for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 15,
        "activities": sorted(ACTIVITIES.values()),
        "placements": sorted(DEVICES.values()),
        "device_profile": "device",
        "gravity_state": "present",
        "license": "cite Karagiannaki et al., UbiComp 2016",
        "citation": ("Karagiannaki K, Panousopoulou A, Tsakalides P. A Benchmark Study on Feature "
                     "Selection for Human Activity Recognition. UbiComp 2016 adjunct."),
        "unit_note": "Source gyroscope is deg/s and is converted to rad/s in this converter.",
        "synchronisation_note": (
            "The five nodes share one recording clock; each file is interpolated onto a common "
            "51.2 Hz timebase so the five placements are simultaneous sample-for-sample."
        ),
    }


def _load_participant(part: int):
    """Return (uniform timebase, {device_prefix: (T,6) acc+gyro}, per-sample label codes)."""
    frames = {}
    for device_id, prefix in DEVICES.items():
        path = RAW / f"part{part}" / f"part{part}dev{device_id}.csv"
        if not path.exists():
            return None
        table = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        if table.shape[1] != 12:
            raise ValueError(f"{path.name}: expected 12 columns, got {table.shape[1]}")
        frames[prefix] = table

    # Reconstruct each node's SAMPLE POSITION on one ideal 51.2 Hz timeline. Between dropouts the
    # node sampled uniformly, so position advances by one per row; at a dropout it advances by the
    # number of samples the node failed to emit, which the released stamps can still tell us even at
    # 100 ms resolution because every real gap here is seconds long. Nothing is interpolated: rows
    # are SCATTERED to their positions and the uncovered positions are marked as holes.
    #
    # Index alignment alone is not enough. The five nodes return different row counts precisely
    # because some of them dropped out mid-recording (part1's ankle node loses 13.1 s), so aligning
    # on row index would shift every label after the dropout and destroy the node-to-node agreement
    # that guarantees the placements are simultaneous.
    positions: dict[str, np.ndarray] = {}
    for prefix, table in frames.items():
        # A monotone envelope of the released stamps. Quantisation and part3's out-of-order stamps
        # both only ever push a sample's stamp DOWN relative to its neighbours, so the running
        # maximum recovers the intended timeline without inventing gaps at either defect.
        stamps = np.maximum.accumulate(table[:, 10] - table[0, 10])
        # The stamps' RESOLUTION collapses but their ACCURACY does not: a value quantised to 100 ms
        # is still within 100 ms (~5 samples) of the truth. That is useless for interpolation, which
        # needs local timing, and perfectly adequate for placement. So each row is assigned the
        # nearest integer slot on the ideal 51.2 Hz timeline and left otherwise untouched. Reading
        # positions off the stamps rather than off the row index also survives nodes that drop
        # samples SILENTLY, without recording a gap — part9 and part11 do exactly that, and an
        # index-derived timeline shifts every label after such a drop.
        # Samples sharing a quantised stamp were emitted uniformly across that quantum, so they are
        # spread evenly from their stamp to the next distinct one instead of being de-tied onto
        # consecutive slots. De-tying leaves a skipped slot every time the quantum (100 ms) is not
        # an exact multiple of the sample period (19.53 ms), which fragments the recording into
        # thousands of one-sample "gaps"; even spreading reproduces the emission the node actually
        # made. The per-sample step is capped at 1.5x nominal so the group before a REAL dropout is
        # not smeared across it.
        period = 1000.0 / NATIVE_RATE
        first = np.r_[True, np.diff(stamps) > 0]
        group = np.cumsum(first) - 1
        group_stamp = stamps[first]
        counts = np.bincount(group)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        following = np.r_[group_stamp[1:], group_stamp[-1] + counts[-1] * period]
        step = np.minimum((following - group_stamp) / counts, 1.5 * period)
        rank = np.arange(len(stamps)) - starts[group]
        raw = np.round((group_stamp[group] + rank * step[group]) / period).astype(np.int64)
        # Any residual collision resolves to the next free slot.
        index = np.arange(len(raw))
        positions[prefix] = np.maximum.accumulate(raw - index) + index
        drift = float(positions[prefix][-1] / NATIVE_RATE - stamps[-1] / 1000.0)
        if drift > MAX_CLOCK_LEAD_SEC:
            raise ValueError(
                f"part{part} {prefix}: de-tied sample positions run {drift:.1f}s past the released "
                f"stamps, more than {MAX_CLOCK_LEAD_SEC}s. That means the node emitted faster than "
                f"{NATIVE_RATE} Hz over a sustained stretch, so the documented rate is wrong and "
                "placement onto a uniform grid is unsafe.")

    n = min(int(p[-1]) + 1 for p in positions.values())
    base = np.arange(n) / NATIVE_RATE * 1000.0

    signals: dict[str, np.ndarray] = {}
    label_votes = []
    covered = np.ones(n, dtype=bool)
    tolerance = int(round(GAP_TOLERANCE_SEC * NATIVE_RATE))
    for prefix, table in frames.items():
        place = positions[prefix]
        take = place < n
        occupied = np.zeros(n, dtype=bool)
        occupied[place[take]] = True
        block = np.zeros((n, 6), dtype=np.float32)
        block[place[take]] = table[take, 1:7]
        votes = np.zeros(n, dtype=int)
        votes[place[take]] = table[take, 11].astype(int)
        # A slot left empty by placement jitter holds the preceding sample. Empty slots are
        # unavoidable whenever the 100 ms quantum is not a whole number of 19.53 ms periods, and
        # they are at most a couple of samples long — holding for under 40 ms is a far smaller
        # distortion than resampling the whole recording. Only runs longer than the gap tolerance
        # are treated as real dropouts, and those are excluded rather than filled.
        source = np.maximum.accumulate(np.where(occupied, np.arange(n), 0))
        block, votes = block[source], votes[source]
        block[:, 3:] *= DEG_TO_RAD                       # deg/s -> rad/s
        signals[prefix] = block
        label_votes.append(votes)
        covered &= _short_holes_are_covered(occupied, tolerance)

    # The five nodes carry the same annotation track; disagreement means a clock problem or, worse,
    # tracks annotated against different takes. The statistic is the WORST node, not the mean over
    # nodes: one broken node out of five only drags a mean down to ~0.83, which would sail past any
    # sane threshold while the bilateral-wrist pair it is supposed to guard is meaningless.
    # Agreement is measured only where every node actually has a sample; a hole would otherwise
    # score as agreement on the zero-fill.
    stacked = np.stack(label_votes)
    agreement = float(min(np.mean(row[covered] == stacked[0][covered]) for row in stacked))
    # A position no node covers is a hole in the five-placement frame, so the recording is cut at
    # both of its edges and the hole itself never reaches a session.
    cuts = np.flatnonzero(np.diff(covered.astype(np.int8)) != 0) + 1
    return base, signals, stacked[0], agreement, dict(frames), cuts, covered


def main() -> None:
    parts = sorted(int(p.name.removeprefix("part")) for p in RAW.glob("part*") if p.is_dir())
    if not parts:
        raise SystemExit(f"raw participant folders not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks_deg: list[float] = []
    rates: list[float] = []
    agreements: dict[str, float] = {}
    excluded: list[tuple[str, float]] = []
    short_blocks = 0
    n_gaps = 0
    dropped_gap_rows = 0
    n_rows = 0

    for part in parts:
        loaded = _load_participant(part)
        if loaded is None:
            print(f"  part{part}: incomplete device set, skipped", flush=True)
            continue
        base, signals, codes, agreement, raw_tables, cuts, covered = loaded
        agreements[f"part{part}"] = agreement
        if agreement < MIN_NODE_AGREEMENT:
            excluded.append((f"part{part}", agreement))
            print(f"  part{part}: EXCLUDED, node label agreement {agreement:.3f} < "
                  f"{MIN_NODE_AGREEMENT}", flush=True)
            continue
        for prefix, table in raw_tables.items():
            clock = table[:, 10]
            rates.append((len(clock) - 1) / ((clock[-1] - clock[0]) / 1000.0))
            magnitudes.append(float(np.median(np.linalg.norm(table[:, 1:4], axis=1))))
            gyro_peaks_deg.append(float(np.percentile(np.abs(table[:, 4:7]), 99.9)))
        # Only where a node actually sampled: uncovered positions carry a zero fill, not a label.
        unknown = set(np.unique(codes[covered])) - set(ACTIVITIES)
        if unknown:
            raise ValueError(f"part{part}: unknown activity codes {sorted(unknown)}")

        # A dropout ends a block just as a label change does, so no emitted session spans a gap.
        n_gaps += int(len(cuts) // 2)
        boundaries = np.zeros(len(codes), dtype=np.int64)
        boundaries[cuts[cuts < len(codes)]] = 1
        # One integer key per (activity, gap-era) so `_runs` splits on either. The multiplier is
        # larger than any possible gap count, so the two components never collide.
        stride = len(codes) + 1
        segments = np.asarray(codes, dtype=np.int64) * stride + np.cumsum(boundaries)

        subject = f"part{part:02d}"
        seen: dict[int, int] = {}
        for segment, start, stop in _runs(segments):
            # Coverage is constant inside a segment: every coverage transition is already a cut.
            if not covered[start]:
                dropped_gap_rows += stop - start
                continue
            code = int(segment) // stride
            block = seen.get(code, 0)
            seen[code] = block + 1
            activity = ACTIVITIES[code]
            if (stop - start) < WINDOW_SECONDS * NATIVE_RATE:
                short_blocks += 1
            session_id = f"{subject}_{activity}_{block:02d}"
            frame = pd.DataFrame({"timestamp_sec": (base[start:stop] - base[start]) / 1000.0})
            for prefix, signal in signals.items():
                chunk = signal[start:stop]
                for offset, name in enumerate(("acc_x", "acc_y", "acc_z",
                                               "gyro_x", "gyro_y", "gyro_z")):
                    frame[f"{prefix}_{name}"] = chunk[:, offset]
            frame["activity"] = activity
            frame["subject"] = subject
            target = sessions_dir / session_id
            target.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target / "data.parquet", index=False)
            labels[session_id] = [activity]
            n_rows += len(frame)
        print(f"  part{part}: {len(seen)} activities, {sum(seen.values())} blocks", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_MS2_RANGE[0] <= reference <= ACC_MS2_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the m/s^2 range {ACC_MS2_RANGE}")
    peak_deg = float(np.max(gyro_peaks_deg))
    if peak_deg < GYRO_DEGS_MIN:
        raise ValueError(
            f"raw |gyro| p99.9 max = {peak_deg:.1f} never exceeds {GYRO_DEGS_MIN}; the source looks "
            "like rad/s now, so the deg/s -> rad/s conversion here would shrink it 57x.")
    measured_rate = float(np.median(rates))
    if abs(measured_rate - NATIVE_RATE) > RATE_TOLERANCE_HZ:
        raise ValueError(f"median clock rate {measured_rate:.3f} Hz != documented {NATIVE_RATE} Hz")

    manifest = create_manifest()
    manifest["subjects"] = len(agreements) - len(excluded)
    manifest["node_label_agreement"] = {k: round(v, 4) for k, v in sorted(agreements.items())}
    manifest["excluded_participants"] = {
        name: (f"node label agreement {value:.3f} < {MIN_NODE_AGREEMENT}; the five annotation "
               f"tracks describe different takes, so the placements are not simultaneous")
        for name, value in excluded
    }
    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "forth_trace", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[forth_trace] units verified: median |acc| {reference:.3f} m/s^2 (gravity present), "
          f"raw |gyro| p99.9 max {peak_deg:.1f} deg/s -> {peak_deg * DEG_TO_RAD:.2f} rad/s, "
          f"clock {measured_rate:.3f} Hz; node label agreement min "
          f"{min(v for k, v in agreements.items() if k not in dict(excluded)):.4f} over the kept "
          f"participants", flush=True)
    print(f"[forth_trace] {len(labels)} blocks from {len(parts) - len(excluded)}/{len(parts)} "
          f"participants ({', '.join(n for n, _ in excluded) or 'none'} excluded), {n_rows:,} "
          f"samples, {hours:.2f} h; timebase reconstructed from the sample index at "
          f"{NATIVE_RATE} Hz, split at {n_gaps} acquisition gaps "
          f"({dropped_gap_rows / NATIVE_RATE:.1f}s of unobserved time dropped); "
          f"{short_blocks} blocks shorter "
          f"than {WINDOW_SECONDS:g}s will yield no grid rows -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
