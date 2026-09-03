"""Task-2 evaluation: reliability, responsiveness and nuisance false alarms.

Reads the frozen test manifest, scores every comparison through the same
deployment path for both arms (phase residual, personal envelope, per-person
reference-only limit), and reports each source separately. Nothing here is tuned:
the limit is computed only from that person's accepted reference set.

Three questions are answered per source (design doc section 7):

* **reliability** -- ICC(2,1), SEM and MDC95 over the accepted repeats, which is
  what the change score's noise floor actually is;
* **responsiveness** -- paired within-person AUROC and standardised response mean
  between accepted and changed queries, computed inside each person so that
  between-person offsets cannot do the work;
* **nuisance false alarms** -- the fraction of accepted repeats that exceed their
  own person's threshold.

The untrained floor is a mandatory row: with ``--head-directory`` omitted the same
report is produced for the closed-form arm, and both must appear in any table.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    Task2EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.representation_cache import (
    bounded_representation_id,
    open_representations,
)
from .contracts import BoundedExecution, ExecutionEpisode, collate_execution_episodes
from .metrics import (
    nuisance_false_alarm_rate,
    paired_within_series_auroc,
    reliability,
    standardised_response_mean,
    subject_bootstrap,
)
from .model import ChangeRuler
from .records import crop_sequence
from .scoring import personal_change_report, physical_change_report


def _execution(
    cache, representations, dataset: str, cache_index: int, event_index: int, stream_id: str
) -> BoundedExecution | None:
    recording = cache[cache_index]
    event = recording.events[event_index]
    stream = [item for item in recording.streams if item.stream_id == stream_id][0]
    try:
        sequence = representations.get(
            dataset,
            bounded_representation_id(recording.recording_id, event_index),
            stream_id,
        )
        cropped = crop_sequence(sequence, float(event.start_sec), float(event.end_sec))
        return BoundedExecution(
            embeddings=cropped.embeddings,
            patch_intervals_sec=cropped.intervals_sec,
            patch_mask=cropped.valid,
            dataset=dataset,
            subject_id=str(recording.subject_id),
            session_id=str(recording.session_id),
            execution_id=f"{recording.recording_id}:{event_index}",
            task_id=str(event.label),
            # Carry the real key: without it the episode's compatibility guard
            # compares None with None and can never fire.
            sensor_config=sensor_compatibility_key(
                device=stream.device,
                placement=stream.placement,
                channels=stream.channels,
                gravity_state=stream.gravity_state,
            ),
            physical_features=cropped.physical_features,
            physical_feature_mask=cropped.physical_feature_mask,
            physical_feature_names=cropped.physical_feature_names,
        )
    except (KeyError, FileNotFoundError, ValueError):
        return None


def score_units(
    units: list[Task2EvaluationUnit], caches, representations, model, *, strict: bool = True
) -> list[dict[str, Any]]:
    """Score each frozen comparison; skip and count units the encoder cannot serve."""

    rows: list[dict[str, Any]] = []
    skipped = 0
    for unit in units:
        cache = caches[unit.dataset]
        references = [
            _execution(cache, representations, unit.dataset, index, event_index, unit.stream_id)
            for index, event_index in zip(
                unit.reference_cache_indices, unit.reference_event_indices
            )
        ]
        query = _execution(
            cache,
            representations,
            unit.dataset,
            unit.query_cache_index,
            unit.query_event_index,
            unit.stream_id,
        )
        if query is None or any(item is None for item in references):
            if strict:
                raise ValueError(
                    f"common Task-2 unit is not representable: "
                    f"{unit.dataset}/{unit.query_recording_id}/{unit.stream_id}"
                )
            skipped += 1
            continue
        episode = ExecutionEpisode(
            accepted_references=tuple(references),
            query=query,
            episode_kind="unlabeled_query",
        )
        batch = collate_execution_episodes([episode])
        if model is not None:
            batch = batch.to(next(model.parameters()).device)
        report = personal_change_report(batch, model)
        physical = physical_change_report(tuple(references), query)
        rows.append(
            {
                "dataset": unit.dataset,
                "subject_id": unit.subject_id,
                "task_id": unit.task_id,
                "stream_id": unit.stream_id,
                "role": unit.role,
                "relation": unit.relation,
                "cell": unit.cell,
                "deviation": float(report.joint_deviation[0]),
                "personal_limit95": float(report.personal_limit95[0]),
                "exceeds_personal_limit": bool(report.exceeds_personal_limit[0]),
                "reference_limited": bool(report.reference_limited[0]),
                "occasion_id": str(
                    unit.change_evidence.get(
                        "week",
                        unit.change_evidence.get(
                            "trial_index", unit.query_recording_id
                        ),
                    )
                ),
                "evidence": dict(unit.change_evidence),
                "physical_deviation": physical["joint_deviation"],
                "physical_personal_limit95": physical["personal_limit95"],
                "physical_exceeds_personal_limit": physical[
                    "exceeds_personal_limit"
                ],
            }
        )
    if skipped:
        rows.append({"dataset": "__skipped__", "count": skipped})
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One report per source, plus the strict-margin subgroup where it exists."""

    scored = [row for row in rows if row["dataset"] != "__skipped__"]
    report: dict[str, Any] = {}
    # One report per (source, declared analysis cell). MoniPar answers two
    # different questions with two different reference rules, and pooling them
    # would average a reliability floor together with a responsiveness result.
    for dataset, cell in sorted({(row["dataset"], row["cell"]) for row in scored}):
        subset = [
            row for row in scored if row["dataset"] == dataset and row["cell"] == cell
        ]
        accepted = [row for row in subset if row["role"] == "accepted_query"]
        changed = [row for row in subset if row["role"] == "changed_query"]
        entry: dict[str, Any] = {
            "units": len(subset),
            "accepted": len(accepted),
            "changed": len(changed),
            "subjects": len({row["subject_id"] for row in subset}),
            "reference_limited": sum(row["reference_limited"] for row in subset),
        }
        entry["accepted_subjects"] = len({row["subject_id"] for row in accepted})
        reliability_rows: dict[str, Any] = {}
        strata = sorted({(row["task_id"], row["stream_id"]) for row in accepted})
        for task_id, stream_id in strata:
            stratum = [
                row
                for row in accepted
                if row["task_id"] == task_id and row["stream_id"] == stream_id
            ]
            key = f"{task_id}/{stream_id}"
            try:
                reliability_rows[key] = vars(
                    reliability(
                        [row["deviation"] for row in stratum],
                        [row["subject_id"] for row in stratum],
                        [row["occasion_id"] for row in stratum],
                        condition=stratum[0]["relation"],
                    )
                )
            except ValueError as error:
                reliability_rows[key] = {"unsupported": str(error)}
        if reliability_rows:
            entry["reliability_by_task_stream"] = reliability_rows
        if accepted:
            try:
                entry["nuisance_false_alarms"] = nuisance_false_alarm_rate(
                    [row["deviation"] for row in accepted],
                    [row["personal_limit95"] for row in accepted],
                )
            except ValueError as error:
                # Every reference set was too small for a threshold; an
                # unsupported cell is a result, not something to force.
                entry["nuisance_false_alarms"] = {"unsupported": str(error)}
            try:
                entry["accepted_deviation"] = subject_bootstrap(
                    [row["deviation"] for row in accepted],
                    [row["subject_id"] for row in accepted],
                )
            except ValueError as error:
                entry["accepted_deviation"] = {"unsupported": str(error)}
        if accepted and changed:
            def by_series(source):
                grouped: dict[str, list[float]] = defaultdict(list)
                for row in source:
                    key = f"{row['subject_id']}/{row['task_id']}/{row['stream_id']}"
                    grouped[key].append(row["deviation"])
                return grouped

            try:
                entry["responsiveness"] = paired_within_series_auroc(
                    by_series(accepted), by_series(changed)
                )
            except ValueError as error:
                entry["responsiveness"] = {"unsupported": str(error)}
            paired_accepted, paired_changed = [], []
            for series, values in sorted(by_series(accepted).items()):
                other = by_series(changed).get(series)
                if other:
                    paired_accepted.append(sum(values) / len(values))
                    paired_changed.append(sum(other) / len(other))
            if len(paired_accepted) >= 2:
                entry["standardised_response_mean"] = standardised_response_mean(
                    paired_accepted, paired_changed
                )
            else:
                entry["standardised_response_mean"] = {
                    "unsupported": "fewer than two subjects have both accepted and changed queries"
                }
        strict = [row for row in changed if row["evidence"].get("strict_change")]
        if strict and accepted:
            strict_entry: dict[str, Any] = {
                "changed": len(strict),
                "subjects": len({row["subject_id"] for row in strict}),
            }
            try:
                strict_entry["responsiveness"] = paired_within_series_auroc(
                    by_series(accepted), by_series(strict)
                )
            except ValueError as error:
                strict_entry["responsiveness"] = {"unsupported": str(error)}
            entry["strict_change_subgroup"] = strict_entry
        physical_accepted = [row for row in accepted if row.get("physical_deviation") is not None]
        physical_changed = [row for row in changed if row.get("physical_deviation") is not None]
        physical_entry: dict[str, Any] = {
            "accepted": len(physical_accepted),
            "changed": len(physical_changed),
        }
        if physical_accepted:
            physical_entry["accepted_mean_deviation"] = sum(
                row["physical_deviation"] for row in physical_accepted
            ) / len(physical_accepted)
            try:
                physical_entry["nuisance_false_alarms"] = nuisance_false_alarm_rate(
                    [row["physical_deviation"] for row in physical_accepted],
                    [row["physical_personal_limit95"] for row in physical_accepted],
                )
            except ValueError as error:
                physical_entry["nuisance_false_alarms"] = {"unsupported": str(error)}
        if physical_accepted and physical_changed:
            physical_entry["changed_mean_deviation"] = sum(
                row["physical_deviation"] for row in physical_changed
            ) / len(physical_changed)
            physical_entry["responsiveness"] = paired_within_series_auroc(
                by_series(
                    [{**row, "deviation": row["physical_deviation"]} for row in physical_accepted]
                ),
                by_series(
                    [{**row, "deviation": row["physical_deviation"]} for row in physical_changed]
                ),
            )
        entry["raw_physical_control"] = physical_entry
        entry["cell"] = cell
        report[f"{dataset}/{cell}"] = entry
    skipped = [row for row in rows if row["dataset"] == "__skipped__"]
    report["skipped_units"] = skipped[0]["count"] if skipped else 0
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK2_V1.json")
    parser.add_argument("--test-manifest", type=Path, default=root / "manifests/TASK2_TEST_V1.json")
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--common-units", type=Path, required=True)
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK2_TRAIN_V1.json"
    )
    parser.add_argument("--head-directory", type=Path, help="omit for the untrained floor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("evaluation limit must be positive")

    cohort = read_cohort_manifest(args.cohort)
    representations = open_representations(args.representations, cohort=cohort)
    manifest = read_task_manifest(args.test_manifest)
    train_manifest = read_task_manifest(args.train_manifest)
    if manifest.task != "task2" or manifest.protocol.get("split") != "test":
        raise ValueError("--test-manifest must be the Task-2 test manifest")
    caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for row in manifest.units})
    }
    validate_task_manifest(manifest, cohort, caches)
    common_text = args.common_units.read_text(encoding="utf-8")
    common = json.loads(common_text)
    if common.get("test_manifest_fingerprint") != manifest.fingerprint:
        raise ValueError("common Task-2 units belong to another test manifest")
    if common.get("train_manifest_fingerprint") != train_manifest.fingerprint:
        raise ValueError("common Task-2 units belong to another train manifest")
    provenance = representations.metadata["encoder_provenance"]
    if not any(
        row.get("encoder_provenance") == provenance
        for row in common.get("representations", {}).values()
    ):
        raise ValueError("common Task-2 units were not built for this encoder")

    model = None
    if args.head_directory is not None:
        checkpoint = torch.load(
            args.head_directory / "task2_head.pt", map_location="cpu", weights_only=False
        )
        if checkpoint.get("cohort_fingerprint") != cohort.fingerprint:
            raise ValueError("Task-2 checkpoint belongs to another cohort")
        if checkpoint.get("representation_provenance") != representations.metadata[
            "encoder_provenance"
        ]:
            raise ValueError("Task-2 checkpoint used another encoder representation")
        if checkpoint.get("train_manifest_fingerprint") != train_manifest.fingerprint:
            raise ValueError("Task-2 checkpoint used another train manifest")
        if checkpoint.get("common_unit_fingerprint") != sha256(
            common_text.encode("utf-8")
        ).hexdigest():
            raise ValueError("Task-2 checkpoint used another common-unit set")
        model = ChangeRuler(
            int(checkpoint["embedding_dim"]),
            phase_bins=int(checkpoint["phase_bins"]),
            projection_dim=checkpoint.get("projection_dim"),
        ).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

    indices = [int(index) for index in common["selected_test_unit_indices"]]
    units = [Task2EvaluationUnit(**manifest.units[index]) for index in indices][: args.limit]
    started = time.time()
    rows = score_units(units, caches, representations, model)
    report = {
        "task": "task2",
        "arm": "untrained_floor" if model is None else "learned_ruler",
        "cohort_fingerprint": cohort.fingerprint,
        "test_manifest_fingerprint": manifest.fingerprint,
        "common_unit_fingerprint": sha256(common_text.encode("utf-8")).hexdigest(),
        "representation_provenance": representations.metadata["encoder_provenance"],
        "protocol": dict(manifest.protocol),
        "status": "reportable" if args.limit is None else "pilot_limited_nonreportable",
        "available_common_test_units": len(indices),
        "evaluated_test_units": len(units),
        "selection_limit": args.limit,
        "seconds": time.time() - started,
        "summary": summarise(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True, default=str))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
