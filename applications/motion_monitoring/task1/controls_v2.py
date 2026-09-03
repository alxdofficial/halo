"""Task-1 synthetic-corpus controls (TASK1_REFERENCE_RESOLUTION_SPEC.md section C.4).

Both controls train on the synthetic train manifest with the background
wearer as the leakage group: a deterministic fraction of background subjects
is held out as *synthetic dev*, which exists only inside this script (the
cohort keeps ``synth_wrist_v1`` train-only; nothing here calibrates a
reported number).

* ``splice_leak``: every unit keeps its query and targets but receives a
  reference of a *different* exercise. If a head trained this way scores
  above chance on synthetic dev, the seams are a feature and the synthesis
  recipe is rejected. Chance is the untrained matcher under the same
  wrong references; the correct-reference arm trained on the same units
  gives the scale.
* ``reference_identity``: the reference is the clean source *same donor clip*
  as one of the query's positives, paired against the cross-clip rule on
  exactly the same synthetic units. The same-clip arm should score higher on
  synthetic held-out subjects. This diagnoses source-clip identity leakage;
  it does not make a claim about natural same-subject enrollment.

Thresholds here are oracle F1 searches on synthetic dev — diagnostics, never
the deployed operating point.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.task1.train_full import (
    calibrate,
    fit_head,
    split_by_subject,
)


def _digest(*parts: object) -> str:
    return sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def wrong_references(
    units: list[Task1EvaluationUnit], *, seed: int
) -> list[Task1EvaluationUnit]:
    """Give every unit a reference of another exercise; queries and targets stay."""

    by_label: dict[str, list[Task1EvaluationUnit]] = defaultdict(list)
    for unit in units:
        by_label[unit.label].append(unit)
    swapped = []
    for unit in units:
        pool = [
            other
            for label, others in sorted(by_label.items())
            if label != unit.label
            for other in others
            if other.reference_recording_id != unit.query_recording_id
        ]
        donor = random.Random(
            _digest(seed, "wrong", unit.query_recording_id, unit.query_stream_id, unit.label)
        ).choice(pool)
        swapped.append(
            replace(
                unit,
                reference_cache_index=donor.reference_cache_index,
                reference_recording_id=donor.reference_recording_id,
                reference_subject_id=donor.reference_subject_id,
                reference_stream_id=donor.reference_stream_id,
                reference_event_index=donor.reference_event_index,
                reference_interval_sec=donor.reference_interval_sec,
                reference_rule=donor.reference_rule,
                label=donor.label,
            )
        )
    return swapped


def clean_reference_index(
    cache,
) -> dict[str, list[tuple[int, str, str, int, tuple[float, float], str]]]:
    """Map donor clip id to clean enrollment-only source records."""

    references: dict[
        str, list[tuple[int, str, str, int, tuple[float, float], str]]
    ] = defaultdict(list)
    for cache_index, recording in enumerate(cache):
        for event_index, event in enumerate(recording.events):
            clip = event.metadata.get("donor_clip_id")
            if event.annotation_kind != "enrollment_execution" or clip is None:
                continue
            references[str(clip)].append(
                (
                    cache_index,
                    recording.recording_id,
                    recording.subject_id,
                    event_index,
                    (float(event.start_sec), float(event.end_sec)),
                    recording.streams[0].stream_id,
                )
            )
    return references


def same_clip_references(
    units: list[Task1EvaluationUnit], cache, references, *, seed: int
) -> tuple[list[Task1EvaluationUnit], list[Task1EvaluationUnit]]:
    """Pair each present unit with a reference inserted from the same donor clip.

    Returns ``(same_clip_units, cross_clip_units)`` over the same query set:
    present units without any same-clip insert elsewhere in the corpus are
    dropped from both arms; absent units are kept unchanged in both.
    """

    same_clip, cross_clip = [], []
    for unit in units:
        if not unit.target_present:
            same_clip.append(unit)
            cross_clip.append(unit)
            continue
        recording = cache[unit.query_cache_index]
        target_clips = {
            str(event.metadata.get("donor_clip_id"))
            for event in recording.events
            if event.annotation_kind == "inserted_execution"
            and event.label == unit.label
            and event.metadata.get("role") == "primary"
        }
        candidates = [
            item
            for clip in sorted(target_clips)
            for item in references.get(clip, ())
            if item[1] != unit.query_recording_id
        ]
        if not candidates:
            continue
        cache_index, recording_id, subject_id, event_index, interval, stream_id = min(
            candidates,
            key=lambda item: _digest(seed, "same_clip", unit.query_recording_id, unit.label, item[1], item[3]),
        )
        same_clip.append(
            replace(
                unit,
                reference_cache_index=cache_index,
                reference_recording_id=recording_id,
                reference_subject_id=subject_id,
                reference_stream_id=stream_id,
                reference_event_index=event_index,
                reference_interval_sec=interval,
                reference_rule="source_event+same_donor_clip",
            )
        )
        cross_clip.append(unit)
    return same_clip, cross_clip


class _UnitManifest:
    """Minimal stand-in accepted by ``calibrate`` (it only reads ``.units``)."""

    def __init__(self, units: list[Task1EvaluationUnit]) -> None:
        self.units = tuple(asdict(unit) for unit in units)


def _summ(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": calibration["threshold"],
        "event_f1": calibration["metrics"]["event_f1"],
        "event_precision": calibration["metrics"]["event_precision"],
        "event_recall": calibration["metrics"]["event_recall"],
        "false_alarms_per_hour": calibration["metrics"]["false_alarms_per_hour"],
        "per_dataset_event_f1": {
            dataset: metrics["event_f1"] for dataset, metrics in calibration["per_dataset"].items()
        },
        "eligible_units": calibration["eligible_units"],
        "rejected_units": calibration["rejected_units"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK1_V2.json")
    parser.add_argument("--train-manifest", type=Path, default=root / "manifests/TASK1_TRAIN_V2.json")
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("splice_leak", "reference_identity", "all"), default="all")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-seconds", type=float, default=60.0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    cohort = read_cohort_manifest(args.cohort)
    representations = open_representations(args.representations, cohort=cohort)
    train_manifest = read_task_manifest(args.train_manifest)
    units = [Task1EvaluationUnit(**row) for row in train_manifest.units]
    datasets = sorted({unit.dataset for unit in units})
    if datasets != ["synth_wrist_v1"]:
        raise ValueError(f"controls expect a synthetic-only train manifest, got {datasets}")
    caches = {dataset: open_cache(dataset) for dataset in datasets}
    validate_task_manifest(train_manifest, cohort, caches)
    if train_manifest.task != "task1" or train_manifest.protocol.get("split") != "train":
        raise ValueError("--train-manifest must be a Task-1 train manifest")
    synth_train, synth_dev = split_by_subject(
        units, seed=args.seed, heldout_fraction=args.heldout_fraction
    )
    feature_dim = representations.get(
        units[0].dataset, units[0].query_recording_id, units[0].query_stream_id
    ).embeddings.shape[1]
    fit_kwargs = dict(
        feature_dim=feature_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        query_seconds=args.query_seconds,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        telemetry_every=max(1, args.steps // 5),
        seed=args.seed,
        device=args.device,
    )
    report: dict[str, Any] = {
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": train_manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "synthetic_split": {
            "heldout_fraction": args.heldout_fraction,
            "train_units": len(synth_train),
            "dev_units": len(synth_dev),
            "train_subjects": len({u.query_subject_id for u in synth_train}),
            "dev_subjects": len({u.query_subject_id for u in synth_dev}),
        },
        "steps": args.steps,
    }
    args.output.mkdir(parents=True, exist_ok=True)

    def fit(train_units):
        started = time.time()
        model, telemetry, rejections = fit_head(train_units, caches, representations, **fit_kwargs)
        return model, {"telemetry": telemetry, "episode_rejections": rejections, "seconds": time.time() - started}

    def score(dev_units, model):
        return _summ(calibrate(_UnitManifest(dev_units), caches, representations, model))

    if args.control in ("splice_leak", "all"):
        wrong_train = wrong_references(synth_train, seed=args.seed)
        wrong_dev = wrong_references(synth_dev, seed=args.seed + 1)
        print("splice-leak: training wrong-reference head")
        wrong_model, wrong_fit = fit(wrong_train)
        print("splice-leak: training correct-reference head")
        correct_model, correct_fit = fit(synth_train)
        rows = {
            "chance_direct_wrong_reference": score(wrong_dev, None),
            "wrong_head_on_wrong_reference": score(wrong_dev, wrong_model),
            "wrong_head_on_correct_reference": score(synth_dev, wrong_model),
            "direct_correct_reference": score(synth_dev, None),
            "correct_head_on_correct_reference": score(synth_dev, correct_model),
        }
        leak = rows["wrong_head_on_wrong_reference"]["event_f1"] - rows["chance_direct_wrong_reference"]["event_f1"]
        signal = rows["correct_head_on_correct_reference"]["event_f1"] - rows["direct_correct_reference"]["event_f1"]
        report["splice_leak"] = {
            **rows,
            "fits": {"wrong": wrong_fit, "correct": correct_fit},
            "leak_delta_event_f1": leak,
            "signal_delta_event_f1": signal,
            # Seams are "a feature" when the wrong-reference head learns anything
            # a wrong reference should not know about the targets.
            "passed": bool(leak <= 0.02),
        }
        for name, row in rows.items():
            print(f"  {name:36s} F1={row['event_f1']:.3f} FA/h={row['false_alarms_per_hour']:.1f}")
        print(f"  leak Δ={leak:+.3f}  signal Δ={signal:+.3f}  passed={report['splice_leak']['passed']}")
        torch.save(wrong_model.state_dict(), args.output / "splice_leak_wrong_head.pt")

    if args.control in ("reference_identity", "all"):
        references = clean_reference_index(caches["synth_wrist_v1"])
        same_train, cross_train = same_clip_references(
            synth_train, caches["synth_wrist_v1"], references, seed=args.seed
        )
        same_dev, cross_dev = same_clip_references(
            synth_dev, caches["synth_wrist_v1"], references, seed=args.seed + 1
        )
        print(
            f"reference-identity: paired units train={len(same_train)} dev={len(same_dev)} "
            f"(present train={sum(u.target_present for u in same_train)})"
        )
        print("reference-identity: training same-clip head")
        same_model, same_fit = fit(same_train)
        print("reference-identity: training cross-clip head")
        cross_model, cross_fit = fit(cross_train)
        rows = {
            "same_clip_direct_synth_dev": score(same_dev, None),
            "cross_clip_direct_synth_dev": score(cross_dev, None),
            "same_clip_head_synth_dev": score(same_dev, same_model),
            "cross_clip_head_synth_dev": score(cross_dev, cross_model),
        }
        synth_gap = rows["same_clip_head_synth_dev"]["event_f1"] - rows["cross_clip_head_synth_dev"]["event_f1"]
        report["reference_identity"] = {
            **rows,
            "paired_units": {"train": len(same_train), "dev": len(same_dev)},
            "fits": {"same_clip": same_fit, "cross_clip": cross_fit},
            "synthetic_dev_gap_event_f1": synth_gap,
            "natural_check": "not applicable: same-subject enrollment is not a same-clip control",
            "passed": bool(synth_gap > 0),
        }
        for name, row in rows.items():
            print(f"  {name:32s} F1={row['event_f1']:.3f} FA/h={row['false_alarms_per_hour']:.1f}")
        print(f"  synth gap={synth_gap:+.3f}  passed={report['reference_identity']['passed']}")

    (args.output / "controls_v2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"saved -> {args.output / 'controls_v2.json'}")


if __name__ == "__main__":
    main()
