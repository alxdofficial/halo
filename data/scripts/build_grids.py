"""Build the corpus grids from converted sessions — the harmonisation entry point.

Run AFTER the per-dataset converters have produced `data/datasets/<ds>/sessions/*.parquet`
(+ `manifest.json` with the native rate, + `labels.json`):

    python -m data.scripts.build_grids                     # all streams: harmonised + non-harmonised

`harmonised-strict` (phones only) is NOT a separate build — it is the phone subset of the harmonised
grids, selected at training time via `deployment_streams(placement_strict=True)`.

For every device stream (`deployment_policy.deployment_streams`), each session is assembled into a
windowed `Grid`:

  * **harmonised**     — resampled to 60 Hz, fixed 6-ch [acc,gyro] pad+mask, **canonical** labels
  * **non_harmonised** — native rate, native 3/6-ch, **native** labels

Grids are written per dataset+stream under `data/datasets/<ds>/grids/` (gitignored). The assembly
core (`stream_grid`) is unit-tested on synthetic sessions; only the disk I/O below needs real data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.scripts.assembly import baseline_view
from data.scripts.assembly.assemble import Grid, assemble, resample_signal
from data.scripts.curate.deployment_policy import (STANDARD_CHANNEL_ORDER, StreamSpec,
                                                   deployment_streams, session_stream_specs)
from data.scripts.labels.canonical_labels import canonicalize

REPO = Path(__file__).resolve().parents[2]
HARMONISED_RATE_HZ = 60
WINDOW_SECONDS = 6.0

# A session as the assembler needs it: the raw frame + its native rate + the subject id (for splits).
Session = Tuple[pd.DataFrame, float, object]


def stream_grid(dataset: str, spec: StreamSpec, sessions: Iterable[Session], *,
                alignment: str, harmonised: bool, pre_windowed: bool = False,
                window_seconds: float = WINDOW_SECONDS) -> Tuple[Grid, List]:
    """Assemble every session of ONE device stream into a single stacked `Grid`.

    ``harmonised=True`` → resample to 60 Hz + map labels to the canonical vocabulary.
    ``harmonised=False`` → keep the native rate and native labels.
    ``pre_windowed=True`` → the dataset is distributed as fixed-length segments (e.g. UCI HAR's
    2.56 s windows), too short for the 6 s corpus window; treat each segment as exactly ONE window
    (window = the segment's own resampled length) instead of cutting 6 s windows from it.
    Returns ``(grid, subjects)`` where ``subjects`` is one entry per window (for subject-disjoint splits).
    """
    sessions = list(sessions)
    resample_to = HARMONISED_RATE_HZ if harmonised else None
    datas: List[np.ndarray] = []
    labels: List = []
    subjects: List = []
    channels: Optional[Tuple[str, ...]] = None
    mask = None
    rate_out = float(HARMONISED_RATE_HZ) if harmonised else None

    forced_window: Optional[int] = None
    if pre_windowed and sessions:
        f0, r0, _ = sessions[0]
        n0 = len(f0)
        # The segment's length after (optional) resampling — one whole window, no truncation.
        forced_window = (len(resample_signal(np.zeros((n0, 1), np.float32), r0, HARMONISED_RATE_HZ))
                         if harmonised else n0)

    for frame, native_rate, subject in sessions:
        out_rate = HARMONISED_RATE_HZ if harmonised else native_rate
        window = forced_window if forced_window is not None else max(1, round(window_seconds * out_rate))
        g = assemble(frame, dataset, spec, alignment=alignment, window=window,
                     rate_hz=native_rate, resample_to=resample_to)
        if len(g.data) == 0:
            continue
        datas.append(g.data)
        channels, mask, rate_out = g.channels, g.mask, g.rate_hz
        if harmonised:
            labels.extend(canonicalize(l) for l in g.labels)   # unify synonyms in the harmonised corpus
        else:
            labels.extend(g.labels)                            # native labels for the non-harmonised corpus
        subjects.extend([subject] * len(g.data))

    if not datas:  # nothing assembled — return a well-formed empty grid at the right width
        declared = tuple(c for c in STANDARD_CHANNEL_ORDER
                         if c in (set(spec.required) | set(spec.optional or {})))
        _, out_channels, empty_mask = baseline_view.to_view(
            np.zeros((0, len(declared)), np.float32), declared, alignment)
        return Grid(np.zeros((0, 0, len(out_channels)), np.float32), empty_mask, out_channels, [],
                    alignment, dataset, float(HARMONISED_RATE_HZ if harmonised else 0.0)), []
    return Grid(np.concatenate(datas, axis=0), mask, channels, labels, alignment, dataset,
                float(rate_out)), subjects


# --------------------------------------------------------------------------------------------------
# Disk I/O — reads what the converters produce. Thin by design; the logic above is what's tested.
# --------------------------------------------------------------------------------------------------

def iter_sessions(dataset: str, spec: StreamSpec,
                  keep_ids: Optional[set] = None) -> Iterable[Session]:
    """Yield the converted sessions belonging to `spec`'s device stream.

    Expects `data/datasets/<ds>/sessions/*.parquet` + `manifest.json` (native `sampling_rate_hz`).
    For multi-stream datasets (wisdm phone/watch) sessions are routed by `session_stream_specs`.
    ``keep_ids`` (if given) restricts to that set of session ids — used by the class-balanced cap for
    huge free-living datasets (capture24) so we never load the full corpus into memory.
    """
    ds_dir = REPO / "data" / "datasets" / dataset
    # Native rate is a dataset property (metadata.json). The session manifest stores rate per-channel;
    # a single deployment stream shares one rate, so the dataset rate is authoritative.
    native_rate = float(json.loads((ds_dir / "metadata.json").read_text())["sampling_rate_hz"])
    # Converters write per-session activity to labels.json (session_id -> [activity]); the parquet has
    # only sensor channels, so inject `activity` here for the assembler to majority-vote.
    labels_map = json.loads((ds_dir / "labels.json").read_text()) if (ds_dir / "labels.json").exists() else {}
    # One session per subdir: sessions/<session_id>/data.parquet.
    for pq in sorted((ds_dir / "sessions").glob("*/data.parquet")):
        sid = pq.parent.name
        if keep_ids is not None and sid not in keep_ids:
            continue
        if spec.stream_id not in {s.stream_id for s in session_stream_specs(dataset, sid, role=spec.role)}:
            continue
        frame = pd.read_parquet(pq)
        if "activity" not in frame.columns and sid in labels_map:
            act = labels_map[sid]
            frame = frame.assign(activity=act[0] if isinstance(act, list) else act)
        # Subject for subject-disjoint splits: a `subject` column if present, else the session-id
        # prefix (converters encode the subject in the id, e.g. "sub01_..." / "subject3_...").
        subject = frame["subject"].iloc[0] if "subject" in frame.columns else sid.split("_")[0]
        yield frame, native_rate, subject


def _pre_windowed(dataset: str) -> bool:
    """Whether a dataset is distributed as fixed short segments (metadata `pre_windowed: true`)."""
    meta = REPO / "data" / "datasets" / dataset / "metadata.json"
    return bool(meta.exists() and json.loads(meta.read_text()).get("pre_windowed", False))


def _max_hours_per_class(dataset: str) -> Optional[float]:
    """Per-class hour cap for a huge free-living dataset (metadata `max_hours_per_class`), else None."""
    meta = REPO / "data" / "datasets" / dataset / "metadata.json"
    if not meta.exists():
        return None
    v = json.loads(meta.read_text()).get("max_hours_per_class")
    return float(v) if v else None


def _greedy_class_cap(per_class: dict, max_hours: float) -> set:
    """Select session ids so each class contributes at most `max_hours` of data.

    `per_class` maps canonical-label -> list of (session_id, duration_seconds). Within a class,
    sessions are taken in a deterministic hash order (mixes subjects, reproducible) until the hour
    budget is reached — always keeping at least one session so a rare class never vanishes.
    Pure/deterministic so it is unit-testable without disk.
    """
    budget = max_hours * 3600.0
    keep: set = set()
    for items in per_class.values():
        ordered = sorted(items, key=lambda it: hashlib.md5(str(it[0]).encode()).hexdigest())
        acc = 0.0
        for i, (sid, dur) in enumerate(ordered):
            if i > 0 and acc >= budget:
                break
            keep.add(sid)
            acc += dur
    return keep


def _capped_session_ids(dataset: str, spec: StreamSpec, max_hours: float) -> set:
    """Class-balanced keep-set for `dataset`'s stream, reading only parquet row-counts (cheap)."""
    import pyarrow.parquet as pq_meta  # metadata-only read; no data loaded
    from collections import defaultdict
    ds_dir = REPO / "data" / "datasets" / dataset
    native_rate = float(json.loads((ds_dir / "metadata.json").read_text())["sampling_rate_hz"])
    labels_map = json.loads((ds_dir / "labels.json").read_text()) if (ds_dir / "labels.json").exists() else {}
    per_class = defaultdict(list)
    for pqf in sorted((ds_dir / "sessions").glob("*/data.parquet")):
        sid = pqf.parent.name
        if spec.stream_id not in {s.stream_id for s in session_stream_specs(dataset, sid, role=spec.role)}:
            continue
        act = labels_map.get(sid)
        label = canonicalize(act[0] if isinstance(act, list) else act) if act else "?"
        nrows = pq_meta.ParquetFile(pqf).metadata.num_rows
        per_class[label].append((sid, nrows / native_rate))
    return _greedy_class_cap(per_class, max_hours)


def build(out_root: Optional[Path] = None, datasets: Optional[Sequence[str]] = None) -> None:
    """Assemble + save harmonised and non-harmonised grids for EVERY device stream (phone + watch).

    We build the full phone+watch set once; `harmonised-strict` is the phone-only subset selected at
    training time (`deployment_streams(placement_strict=True)`), not a separate materialization.
    Pass `datasets` to build only those (e.g. after a single converter re-run).
    """
    out_root = out_root or (REPO / "data" / "datasets")
    want = set(datasets) if datasets else None
    for spec in deployment_streams(placement_strict=False):
        if want and spec.dataset not in want:
            continue
        pw = _pre_windowed(spec.dataset)
        cap = _max_hours_per_class(spec.dataset)
        keep_ids = _capped_session_ids(spec.dataset, spec, cap) if cap else None
        if keep_ids is not None:
            print(f"  {spec.dataset}/{spec.stream_id}: class-balanced cap {cap}h/class → {len(keep_ids)} sessions")
        sessions = list(iter_sessions(spec.dataset, spec, keep_ids=keep_ids))
        for harmonised, alignment in ((True, "harmonised"), (False, "non_harmonised")):
            grid, subjects = stream_grid(spec.dataset, spec, sessions,
                                         alignment=alignment, harmonised=harmonised, pre_windowed=pw)
            _save(out_root, spec, grid, subjects)


def _save(out_root: Path, spec: StreamSpec, grid: Grid, subjects: List) -> None:
    d = out_root / spec.dataset / "grids" / grid.alignment / spec.stream_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data.npy", grid.data)
    np.save(d / "mask.npy", grid.mask)
    (d / "meta.json").write_text(json.dumps({
        "dataset": grid.dataset, "stream_id": spec.stream_id, "alignment": grid.alignment,
        "rate_hz": grid.rate_hz, "channels": list(grid.channels),
        "labels": list(map(str, grid.labels)), "subjects": list(map(str, subjects)),
    }))
    print(f"  {spec.dataset}/{spec.stream_id}/{grid.alignment}: {grid.data.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Build only these datasets (default: all primary phone+watch streams).")
    args = p.parse_args()
    build(datasets=args.dataset)


if __name__ == "__main__":
    main()
