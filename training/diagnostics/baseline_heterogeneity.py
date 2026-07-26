"""Model-agnostic zero-shot heterogeneity stress test.

Unlike ``zeroshot_difficulty`` (which opens HALO's representation and separates encoder from text
bridge), this harness treats every registered adapter as a black-box zero-shot predictor. It scores
the SAME held-out windows before and after one controlled physical perturbation:

  * rate: anti-aliased downsample with the new rate supplied to the adapter;
  * channel: remove the gyro triad and update the channel mask;
  * orientation: one shared SO(3) rotation for accel + gyro per window;
  * gravity: remove accelerometer gravity and update HALO's diagnostic metadata override.

Per-stream retention is shifted macro-F1 / matched macro-F1. Aggregate summaries primarily report
the ratio of mean F1 and absolute F1 movement: averaging per-stream ratios is unstable when matched
performance is near zero. Prediction consistency is also reported because F1 retention alone can hide
compensating label swaps. These are bounded diagnostic subsets, not headline benchmark cells; use
``eval.run_baselines`` for full-data results and confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps

import baselines
from data.scripts.augmentations import _random_so3
from data.scripts.curate import deployment_policy as policy
from data.scripts.eda.grid_io import discover_grids
from eval import data as eval_data
from eval import scoring
from eval.protocol import protocol_fingerprint

DEFAULT_OUT = Path("training/diagnostics/outputs/baseline_heterogeneity.json")
DEFAULT_PLACEMENT = ("xrf_v2", "left_wrist", "left_pocket")
AXES = ("rate", "channel", "orientation", "gravity")
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
DEVICE_WORDS = {"phone": "phone", "watch": "watch", "watch_proxy": "phone",
                "device": "wearable device"}


@dataclass
class WindowSet:
    data: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    texts: list[str]
    rate: float
    gravity_state: str | None
    channel_mask: tuple[bool, ...]
    dataset: str
    stream: str
    tag: str


def _stream_metadata(dataset: str, stream: str) -> tuple[list[str], str | None]:
    """Build HALO-equivalent per-channel text without importing HALO's ``model`` package."""
    try:
        spec = policy.get_stream_spec(dataset, stream)
    except (KeyError, ValueError):
        return [f"{channel} at the body" for channel in CHANNELS], None
    place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
        else f"the {spec.placement}"
    device = DEVICE_WORDS.get(spec.device_profile, spec.device_profile.replace("_", " "))
    where = f"{place} ({device})"
    grav = "; gravity removed" if spec.gravity_state == "removed" else "; includes gravity"
    texts = ([f"accelerometer {axis}-axis worn at {where}{grav}" for axis in "xyz"]
             + [f"gyroscope {axis}-axis worn at {where}" for axis in "xyz"])
    return texts, spec.gravity_state


def _shift_rate(ws: WindowSet, target_hz: float) -> WindowSet:
    if not 0 < target_hz < ws.rate:
        raise ValueError(
            f"rate shift requires 0 < target_hz < native_hz; got {target_hz} vs {ws.rate}"
        )
    ratio = Fraction(target_hz / ws.rate).limit_denominator(50)
    data = sps.resample_poly(ws.data, ratio.numerator, ratio.denominator, axis=1).astype(
        np.float32
    )
    return WindowSet(
        **{**ws.__dict__, "data": data,
           "rate": ws.rate * ratio.numerator / ratio.denominator,
           "tag": ws.tag + "|rate"}
    )


def _shift_channel(ws: WindowSet) -> WindowSet:
    data = ws.data.copy()
    data[..., 3:] = 0.0
    return WindowSet(
        **{**ws.__dict__, "data": data,
           "channel_mask": tuple((*ws.channel_mask[:3], False, False, False)),
           "tag": ws.tag + "|acc_only"}
    )


def _shift_orientation(ws: WindowSet, seed: int) -> WindowSet:
    torch.manual_seed(seed)
    data = ws.data.copy()
    for index in range(len(data)):
        rotation = _random_so3().to(torch.float32)
        for start in (0, 3):
            if all(ws.channel_mask[start:start + 3]):
                triad = torch.from_numpy(data[index, :, start:start + 3])
                data[index, :, start:start + 3] = torch.einsum(
                    "ij,tj->ti", rotation, triad
                ).numpy()
    return WindowSet(**{**ws.__dict__, "data": data, "tag": ws.tag + "|so3"})


def _shift_gravity(ws: WindowSet) -> WindowSet:
    data = ws.data.copy()
    wn = 0.4 / (ws.rate / 2.0)
    if not 0 < wn < 1 or data.shape[1] <= 12:
        data[..., :3] -= data[..., :3].mean(axis=1, keepdims=True)
    else:
        b, a = sps.butter(2, wn, btype="low")
        for channel in range(3):
            data[..., channel] -= sps.filtfilt(b, a, data[..., channel], axis=1)
    texts = []
    for index, text in enumerate(ws.texts):
        if index < 3:
            text = text.replace("; includes gravity", "").rstrip(" ;") + " (gravity removed)"
        texts.append(text)
    return WindowSet(
        **{**ws.__dict__, "data": data.astype(np.float32), "texts": texts,
           "gravity_state": "removed", "tag": ws.tag + "|grav_removed"}
    )


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _subset_indices(labels, cap: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if cap is None or len(labels) <= cap:
        return np.arange(len(labels), dtype=np.int64)
    unique = sorted(set(labels.tolist()))
    if cap < len(unique):
        raise ValueError(f"max_windows={cap} cannot cover all {len(unique)} labels")
    rng = np.random.default_rng(seed)
    per_label = max(1, cap // len(unique))
    selected = []
    for label in unique:
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        selected.extend(idx[:per_label].tolist())
    if len(selected) < cap:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: cap - len(selected)].tolist())
    return np.sort(np.asarray(selected[:cap], dtype=np.int64))


def subset_stream(stream: eval_data.EvalStream, cap: int, seed: int) -> eval_data.EvalStream:
    eligible = np.flatnonzero(np.isin(np.asarray(stream.gt), stream.eval_labels))
    local = _subset_indices(np.asarray(stream.gt)[eligible], cap, seed)
    idx = eligible[local]
    return eval_data.EvalStream(
        dataset=stream.dataset,
        stream=stream.stream,
        alignment=stream.alignment,
        windows=np.asarray(stream.windows)[idx],
        gt=np.asarray(stream.gt, dtype=object)[idx].tolist(),
        subjects=np.asarray(stream.subjects)[idx],
        channels=list(stream.channels),
        rate_hz=float(stream.rate_hz),
        mask=np.asarray(stream.mask, dtype=bool).copy(),
        eval_labels=list(stream.eval_labels),
        gravity_state=stream.gravity_state,
        channel_descriptions=stream.channel_descriptions,
    )


def _to_windowset(stream: eval_data.EvalStream) -> WindowSet:
    default_texts, default_gravity = _stream_metadata(stream.dataset, stream.stream)
    gravity = stream.gravity_state if stream.gravity_state is not None else default_gravity
    texts = stream.channel_descriptions if stream.channel_descriptions is not None else default_texts
    return WindowSet(
        data=np.asarray(stream.windows, dtype=np.float32),
        labels=np.asarray(stream.gt, dtype=object),
        subjects=np.asarray(stream.subjects),
        texts=list(texts),
        rate=float(stream.rate_hz),
        gravity_state=gravity,
        channel_mask=tuple(np.asarray(stream.mask, dtype=bool).tolist()),
        dataset=stream.dataset,
        stream=stream.stream,
        tag=f"{stream.dataset}/{stream.stream}",
    )


def _from_windowset(base: eval_data.EvalStream, ws: WindowSet) -> eval_data.EvalStream:
    return eval_data.EvalStream(
        dataset=base.dataset,
        stream=base.stream,
        alignment=base.alignment,
        windows=ws.data,
        gt=list(ws.labels),
        subjects=np.asarray(ws.subjects),
        channels=list(base.channels),
        rate_hz=float(ws.rate),
        mask=np.asarray(ws.channel_mask, dtype=bool),
        eval_labels=list(base.eval_labels),
        gravity_state=ws.gravity_state,
        channel_descriptions=list(ws.texts),
    )


def transform_stream(stream: eval_data.EvalStream, axis: str, rate_hz: float,
                     seed: int) -> eval_data.EvalStream:
    ws = _to_windowset(stream)
    if axis == "rate":
        shifted = _shift_rate(ws, rate_hz)
    elif axis == "channel":
        if not any(ws.channel_mask[3:]):
            raise ValueError("stream is already accelerometer-only")
        shifted = _shift_channel(ws)
    elif axis == "orientation":
        shifted = _shift_orientation(ws, seed)
    elif axis == "gravity":
        if ws.gravity_state != "present":
            raise ValueError(f"gravity_state={ws.gravity_state!r}; source is not gravity-present")
        shifted = _shift_gravity(ws)
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return _from_windowset(stream, shifted)


def predict_and_score(adapter, stream, state, device) -> tuple[np.ndarray, np.ndarray, dict]:
    preds, info = adapter.predict(stream, state, device)
    if len(preds) != stream.n_windows:
        raise ValueError(
            f"{adapter.name} returned {len(preds)} predictions for {stream.n_windows} windows"
        )
    gt, _subjects, keep = scoring.filter_ground_truth(
        stream.gt, stream.subjects, stream.eval_labels
    )
    pred = np.asarray(preds, dtype=object)[keep]
    metrics = scoring.classification_metrics(gt, pred.tolist())
    metrics["n_windows"] = int(len(keep))
    metrics["predicted_classes"] = sorted(set(pred.tolist()))
    if info.get("reachability_lb") is not None:
        metrics["reachability_lb"] = float(info["reachability_lb"])
    return pred, np.asarray(keep, dtype=np.int64), metrics


def _subject_variability(stream, pred, keep) -> dict:
    gt = np.asarray(stream.gt, dtype=object)[keep]
    subjects = np.asarray(stream.subjects)[keep]
    values = []
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        values.append(scoring.classification_metrics(gt[mask].tolist(), pred[mask].tolist())[
            "f1_macro"
        ])
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_subjects": len(values),
        "mean_subject_f1": round(float(arr.mean()), 3),
        "std_subject_f1": round(float(arr.std()), 3),
        "worst_subject_f1": round(float(arr.min()), 3),
        "best_subject_f1": round(float(arr.max()), 3),
    }


def _novelty_slice(stream, pred, keep, train_vocab: set[str]) -> dict:
    gt = np.asarray(stream.gt, dtype=object)[keep]
    rows = {}
    for name, novel in (("seen_in_halo_corpus", False), ("novel_to_halo_corpus", True)):
        mask = np.asarray([(label not in train_vocab) == novel for label in gt])
        if not mask.any():
            rows[name] = {"n_windows": 0, "n_labels": 0, "balanced_accuracy": None,
                          "macro_f1": None}
            continue
        metrics = scoring.classification_metrics(gt[mask].tolist(), pred[mask].tolist())
        rows[name] = {
            "n_windows": int(mask.sum()),
            "n_labels": int(len(set(gt[mask].tolist()))),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["f1_macro"],
        }
    return rows


def _axis_row(axis, dataset, stream, matched_pred, shifted_pred, matched_metrics,
              shifted_metrics) -> dict:
    matched_f1 = float(matched_metrics["f1_macro"])
    shifted_f1 = float(shifted_metrics["f1_macro"])
    return {
        "axis": axis,
        "dataset": dataset,
        "stream": stream,
        "n_windows": int(matched_metrics["n_windows"]),
        "matched_f1": round(matched_f1, 3),
        "shifted_f1": round(shifted_f1, 3),
        "absolute_drop": round(matched_f1 - shifted_f1, 3),
        "retention": round(shifted_f1 / matched_f1, 4) if matched_f1 > 0 else None,
        "prediction_consistency": round(float((matched_pred == shifted_pred).mean()), 4),
        "matched_balanced_accuracy": matched_metrics["balanced_accuracy"],
        "shifted_balanced_accuracy": shifted_metrics["balanced_accuracy"],
    }


def _aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "complete"]
    by_axis = {}
    for axis in AXES:
        selected = [row for row in valid if row["axis"] == axis]
        matched = np.asarray([r["matched_f1"] for r in selected], dtype=np.float64)
        shifted = np.asarray([r["shifted_f1"] for r in selected], dtype=np.float64)
        mean_matched = float(matched.mean()) if selected else None
        mean_shifted = float(shifted.mean()) if selected else None
        by_axis[axis] = {
            "n_streams": len(selected),
            "mean_matched_f1": round(mean_matched, 3) if mean_matched is not None else None,
            "mean_shifted_f1": round(mean_shifted, 3) if mean_shifted is not None else None,
            "mean_absolute_drop": round(mean_matched - mean_shifted, 3)
            if mean_matched is not None else None,
            "retention_from_mean_f1": round(mean_shifted / mean_matched, 4)
            if mean_matched is not None and mean_matched > 0 else None,
            "mean_per_stream_retention_unstable": round(float(np.mean(
                [r["retention"] for r in selected if r["retention"] is not None]
            )), 4) if any(r["retention"] is not None for r in selected) else None,
            "n_near_floor_streams": int((matched < 5.0).sum()) if selected else 0,
            "mean_prediction_consistency": round(float(np.mean(
                [r["prediction_consistency"] for r in selected]
            )), 4) if selected else None,
        }
    return by_axis


def _paired_placement_streams(cap: int, seed: int, placement) -> tuple[
        eval_data.EvalStream, eval_data.EvalStream, bool]:
    refs = {(ref.dataset, ref.stream): ref for ref in discover_grids("native")}
    key_a = (placement[0], placement[1])
    key_b = (placement[0], placement[2])
    if key_a not in refs or key_b not in refs:
        raise FileNotFoundError(f"placement grids missing: {key_a}, {key_b}")
    aligned = (
        len(refs[key_a].labels) == len(refs[key_b].labels)
        and refs[key_a].labels == refs[key_b].labels
        and refs[key_a].subjects == refs[key_b].subjects
    )
    if aligned:
        idx_a = idx_b = _subset_indices(refs[key_a].labels, cap, seed)
    else:
        idx_a = _subset_indices(refs[key_a].labels, cap, seed)
        idx_b = _subset_indices(refs[key_b].labels, cap, seed)

    def build(ref, idx):
        texts, gravity = _stream_metadata(ref.dataset, ref.stream)
        return WindowSet(
            data=np.asarray(ref.load_data()[idx], dtype=np.float32),
            labels=np.asarray(ref.labels, dtype=object)[idx],
            subjects=np.asarray(ref.subjects)[idx],
            texts=texts,
            rate=float(ref.rate_hz),
            gravity_state=gravity,
            channel_mask=tuple(ref.mask),
            dataset=ref.dataset,
            stream=ref.stream,
            tag=ref.key,
        )

    wa, wb = build(refs[key_a], idx_a), build(refs[key_b], idx_b)
    candidates = sorted(set(wa.labels.tolist()) | set(wb.labels.tolist()))

    def convert(ws):
        return eval_data.EvalStream(
            dataset=ws.dataset, stream=ws.stream, alignment="native", windows=ws.data,
            gt=list(ws.labels), subjects=np.asarray(ws.subjects), channels=[
                "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
            ], rate_hz=ws.rate, mask=np.asarray(ws.channel_mask, dtype=bool),
            eval_labels=candidates, gravity_state=ws.gravity_state,
            channel_descriptions=list(ws.texts),
        )

    return convert(wa), convert(wb), aligned


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2))
    partial.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="*", default=None,
                        help="registered models (default: all)")
    parser.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument("--max-windows", type=int, default=400)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--placement", nargs=3, default=list(DEFAULT_PLACEMENT),
                        metavar=("DATASET", "A", "B"))
    parser.add_argument("--no-placement", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-partial", action="store_true",
                        help="exit zero even when a requested model/cell fails")
    args = parser.parse_args(argv)
    if args.max_windows < 2:
        parser.error("--max-windows must be >= 2")
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")

    names = args.baselines if args.baselines is not None else sorted(baselines.REGISTRY)
    unknown = [name for name in names if name not in baselines.REGISTRY]
    if unknown:
        parser.error(f"unknown baselines {unknown}; registered={sorted(baselines.REGISTRY)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_vocab = set(eval_data.load_global_labels())
    started = time.time()
    failures = []
    models = {}

    streams = []
    for dataset in args.datasets:
        specs = policy.stream_specs(dataset, "primary")
        if not specs:
            raise SystemExit(f"{dataset!r} has no primary evaluation stream")
        for spec in specs:
            raw = eval_data.load_eval_stream(dataset, spec.stream_id, "non_harmonised")
            streams.append(subset_stream(raw, args.max_windows, args.seed))

    placement_pair = None
    if not args.no_placement:
        placement_pair = _paired_placement_streams(
            args.max_windows, args.seed, tuple(args.placement)
        )

    for name in names:
        model_started = time.time()
        adapter = baselines.REGISTRY[name]
        print(f"\n[{name}] setup (tier={adapter.tier})", flush=True)
        try:
            state = adapter.setup(device)
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[{name}] SETUP FAILED: {reason}", flush=True)
            models[name] = {"status": "failed", "error": reason,
                            "elapsed_s": round(time.time() - model_started, 3)}
            failures.append(f"{name}/setup: {reason}")
            continue

        rows = []
        matched = {}
        subject_rows = {}
        novelty_rows = {}
        for stream in streams:
            cell = f"{stream.dataset}/{stream.stream}"
            incompatible = adapter.is_incompatible(stream.dataset)
            if incompatible is not None:
                for axis in args.axes:
                    rows.append({"axis": axis, "dataset": stream.dataset,
                                 "stream": stream.stream, "status": "na",
                                 "reason": incompatible})
                continue
            try:
                base_pred, keep, base_metrics = predict_and_score(adapter, stream, state, device)
                matched[cell] = base_metrics
                subject_rows[cell] = _subject_variability(stream, base_pred, keep)
                novelty_rows[cell] = _novelty_slice(stream, base_pred, keep, train_vocab)
            except BaseException as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(f"{name}/{cell}/matched: {reason}")
                for axis in args.axes:
                    rows.append({"axis": axis, "dataset": stream.dataset,
                                 "stream": stream.stream, "status": "failed", "error": reason})
                continue

            for axis in args.axes:
                try:
                    shifted = transform_stream(stream, axis, args.rate_hz, args.seed)
                    shift_pred, shift_keep, shift_metrics = predict_and_score(
                        adapter, shifted, state, device
                    )
                    if not np.array_equal(keep, shift_keep):
                        raise AssertionError("matched and shifted scoring indices differ")
                    row = _axis_row(axis, stream.dataset, stream.stream, base_pred, shift_pred,
                                    base_metrics, shift_metrics)
                    row["status"] = "complete"
                    rows.append(row)
                    print(f"[{name}] {cell:42s} {axis:11s} "
                          f"{row['matched_f1']:6.2f}->{row['shifted_f1']:6.2f} "
                          f"ret={row['retention']} agree={row['prediction_consistency']}",
                          flush=True)
                except ValueError as exc:
                    rows.append({"axis": axis, "dataset": stream.dataset,
                                 "stream": stream.stream, "status": "na", "reason": str(exc)})
                except BaseException as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{name}/{cell}/{axis}: {reason}")
                    rows.append({"axis": axis, "dataset": stream.dataset,
                                 "stream": stream.stream, "status": "failed", "error": reason})

        placement_result = None
        if placement_pair is not None:
            stream_a, stream_b, aligned = placement_pair
            try:
                pred_a, keep_a, metrics_a = predict_and_score(adapter, stream_a, state, device)
                pred_b, keep_b, metrics_b = predict_and_score(adapter, stream_b, state, device)
                if not np.array_equal(keep_a, keep_b):
                    raise AssertionError("placement scoring indices differ")
                placement_result = _axis_row(
                    "placement", stream_a.dataset, f"{stream_a.stream}->{stream_b.stream}",
                    pred_a, pred_b, metrics_a, metrics_b
                )
                corpus_overlap = stream_a.dataset in policy.PRIMARY_TRAIN_DATASETS
                placement_result.update(status="complete", row_aligned=bool(aligned),
                                        paired_same_instant=bool(aligned),
                                        corpus_overlap=corpus_overlap,
                                        held_out_dataset=not corpus_overlap,
                                        valid_for_cross_dataset_robustness=not corpus_overlap,
                                        exact_memory_overlap_possible=(
                                            corpus_overlap and name == "halo_evidence"
                                        ))
            except BaseException as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(f"{name}/placement: {reason}")
                placement_result = {"status": "failed", "error": reason}

        models[name] = {
            "status": "complete" if not any(r.get("status") == "failed" for r in rows)
            and (placement_result is None or placement_result.get("status") != "failed") else "partial",
            "tier": adapter.tier,
            "rows": rows,
            "aggregate": _aggregate(rows),
            "matched": matched,
            "subject_variability": subject_rows,
            "label_novelty": novelty_rows,
            "placement": placement_result,
            "elapsed_s": round(time.time() - model_started, 3),
        }

    payload = {
        "schema_version": 2,
        "protocol": protocol_fingerprint(),
        "git": _git_state(),
        "device": str(device),
        "seed": args.seed,
        "max_windows": args.max_windows,
        "rate_target_hz": args.rate_hz,
        "datasets": args.datasets,
        "axes": args.axes,
        "placement": None if args.no_placement else args.placement,
        "placement_interpretation": (
            "The default xrf_v2 pair is a same-instant controlled placement stress test, but "
            "xrf_v2 is in the Phase-A/head-fit corpus. It is not a held-out robustness estimate; "
            "retrieval models may contain the exact query windows in memory."
            if not args.no_placement and args.placement[0] in policy.PRIMARY_TRAIN_DATASETS
            else "The requested placement pair is outside the primary HALO training corpus."
        ),
        "models": models,
        "failures": failures,
        "elapsed_s": round(time.time() - started, 3),
    }
    _atomic_json(args.out, payload)
    print(f"\n-> {args.out} ({len(failures)} failure(s), {payload['elapsed_s']} s)", flush=True)
    return 0 if not failures or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
