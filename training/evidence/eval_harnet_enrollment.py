"""Evaluate frozen HARNet features on the Phase-B enrollment protocol.

This is a representation control, not a HARNet paper-number reproduction and not a use of HALO's
learned evidence engine.  It reuses the exact execution-disjoint subject plans from
``eval_enrollment`` and fits only three support-set rules:

* nearest labeled support window;
* one normalized prototype per candidate;
* a deterministic L2 ridge head.

The resulting k-curves can be compared directly with the HALO prototype/ridge controls because the
candidate sets, support executions, query windows, seed, and metrics are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.base import global_labels
from baselines.harnet.adapter import (
    ACC_CHANNELS,
    FEAT_DIM,
    GRAVITY_MIN_G,
    HARNET_NAME,
    SSL_HUB_REPO,
    SSL_HUB_TAG,
    TARGET_HZ,
    TARGET_LEN,
    _extract_feats,
    _gravity_dc,
    _load_harnet,
    _select_accel,
    _to_30hz_150,
)
from data.scripts.curate import deployment_policy
from eval.data import load_eval_stream
from eval.scoring import align_ground_truth_labels, classification_metrics
from training.evidence.device import resolve_device
from training.evidence.eval_enrollment import (
    build_paired_enrollment_plans,
    paired_subject_summary,
)
from training.evidence.policy import PHASE_B_DEV_DATASETS, PHASE_B_TEST_DATASETS


_REPO = Path(__file__).resolve().parents[2]
_OUT = Path(__file__).resolve().parent / "outputs" / "representation_controls"
_SOURCE_PATHS = (
    "training/evidence/eval_harnet_enrollment.py",
    "training/evidence/eval_enrollment.py",
    "training/evidence/policy.py",
    "baselines/harnet/adapter.py",
    "eval/data.py",
    "eval/scoring.py",
    "data/scripts/curate/deployment_policy.py",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _json_fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in _SOURCE_PATHS:
        path = _REPO / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _backbone_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.feature_extractor.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def encode_harnet_stream(model, stream, device) -> torch.Tensor:
    """Return one normalized HARNet-5 feature per evaluation window."""
    if any(channel not in stream.channels for channel in ACC_CHANNELS):
        missing = [channel for channel in ACC_CHANNELS if channel not in stream.channels]
        raise ValueError(f"missing accelerometer channels: {missing}")
    channel_indices = [stream.channels.index(channel) for channel in ACC_CHANNELS]
    if not bool(np.asarray(stream.mask, dtype=bool)[channel_indices].all()):
        raise ValueError("one or more accelerometer axes are masked")
    dc = _gravity_dc(np.asarray(stream.windows), list(stream.channels))
    if dc < GRAVITY_MIN_G:
        raise ValueError(
            f"gravity-removed accelerometer (median |DC|={dc:.3f} g); HARNet requires gravity"
        )
    accel = _select_accel(np.asarray(stream.windows), list(stream.channels))
    inputs = _to_30hz_150(accel, float(stream.rate_hz))
    features = _extract_feats(model, inputs, device)
    if features.shape != (len(stream.windows), FEAT_DIM):
        raise RuntimeError(
            f"HARNet emitted {features.shape}, expected {(len(stream.windows), FEAT_DIM)}"
        )
    return F.normalize(torch.from_numpy(features).float(), dim=-1)


def support_predictions(
    features: torch.Tensor,
    support_rows: np.ndarray,
    support_positions: np.ndarray,
    query_rows: np.ndarray,
    n_candidates: int,
    device,
) -> dict[str, np.ndarray]:
    """Predict candidate positions with matched nearest/prototype/ridge support rules."""
    if len(support_rows) == 0:
        raise ValueError("support_predictions requires positive support")
    x = features[torch.as_tensor(support_rows, dtype=torch.long)].to(device)
    q = features[torch.as_tensor(query_rows, dtype=torch.long)].to(device)
    position = torch.as_tensor(support_positions, dtype=torch.long, device=device)
    if any(not bool(position.eq(candidate).any()) for candidate in range(n_candidates)):
        raise ValueError("every candidate must have support")

    similarity = q @ x.T
    nearest = position[similarity.argmax(1)]
    centroids = torch.stack([
        F.normalize(x[position.eq(candidate)].mean(0), dim=0)
        for candidate in range(n_candidates)
    ])
    prototype = (q @ centroids.T).argmax(1)

    target = F.one_hot(position, n_candidates).float()
    gram = x @ x.T
    alpha = torch.linalg.solve(
        gram + torch.eye(len(x), device=device, dtype=x.dtype), target
    )
    ridge = (q @ (x.T @ alpha)).argmax(1)
    return {
        "nearest": nearest.cpu().numpy(),
        "prototype": prototype.cpu().numpy(),
        "ridge": ridge.cpu().numpy(),
    }


def score_cell(features, labels, plans, support_count: int, device, seed: int) -> dict:
    """Score one fixed nested-support cell and retain subjects as independent units."""
    all_true: list[str] = []
    all_predictions = {name: [] for name in ("nearest", "prototype", "ridge")}
    subject_results = {}
    for plan in plans:
        support_rows = np.asarray([
            row
            for candidate_rows in plan.support_rows
            for row in candidate_rows[:support_count]
        ], dtype=np.int64)
        support_positions = np.repeat(
            np.arange(len(plan.candidate_names), dtype=np.int64), support_count
        )
        query_rows = np.asarray(plan.query_rows, dtype=np.int64)
        if not len(query_rows):
            continue
        predicted = support_predictions(
            features, support_rows, support_positions, query_rows,
            len(plan.candidate_names), device,
        )
        truth = labels[query_rows].tolist()
        names = np.asarray(plan.candidate_names, dtype=object)
        subject_record = {"queries": len(query_rows)}
        all_true.extend(truth)
        for method, positions in predicted.items():
            values = names[positions].tolist()
            all_predictions[method].extend(values)
            subject_record[f"{method}_f1_macro"] = classification_metrics(
                truth, values
            )["f1_macro"]
        subject_results[str(plan.subject)] = subject_record

    if not all_true:
        return {"status": "insufficient_independent_executions", "queries": 0}
    result = {
        "queries": len(all_true),
        "subjects": len(subject_results),
        "support_count": support_count,
        "subject_results": subject_results,
    }
    for method, values in all_predictions.items():
        result[f"{method}_f1_macro"] = classification_metrics(all_true, values)["f1_macro"]

    # Reuse the same paired subject-bootstrap implementation by exposing HARNet's prototype as the
    # nominal prediction and its other support rules under the existing comparator field names.
    bootstrap_records = {
        subject: {
            "f1_macro": record["prototype_f1_macro"],
            "identity_f1_macro": record["nearest_f1_macro"],
            "prototype_f1_macro": record["prototype_f1_macro"],
            "ridge_head_f1_macro": record["ridge_f1_macro"],
            "support_removed_f1_macro": None,
            "support_label_shuffled_f1_macro": None,
        }
        for subject, record in subject_results.items()
    }
    result["subject_macro"] = paired_subject_summary(
        bootstrap_records, seed=seed + 700_001 + support_count
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-role", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--support", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if any(value <= 0 for value in args.support):
        parser.error("HARNet representation curves require positive support counts")
    explicit_datasets = args.datasets is not None
    datasets = args.datasets or {
        "dev": list(PHASE_B_DEV_DATASETS),
        "test": list(PHASE_B_TEST_DATASETS),
        "all": list(deployment_policy.PRIMARY_EVAL_DATASETS),
    }[args.protocol_role]
    device = resolve_device(args.device)
    model = _load_harnet(len(global_labels()), device)
    model.eval()
    backbone_fp = _backbone_fingerprint(model)

    results = {}
    protocol = {}
    for dataset in datasets:
        for spec in deployment_policy.stream_specs(dataset, "primary"):
            relation_id = (
                f"{dataset}/{spec.stream_id}/from_{spec.stream_id}/"
                "same_configuration/cross_subject"
            )
            try:
                stream = load_eval_stream(dataset, spec.stream_id, alignment="native")
                labels = np.asarray(
                    align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object
                )
                subjects = np.asarray(stream.subjects, dtype=object)
                executions = np.asarray(stream.execution_ids, dtype=object)
                relation_seed = args.seed + int(
                    hashlib.sha256(
                        f"{dataset}/{spec.stream_id}/from_{spec.stream_id}/same_configuration".encode()
                    ).hexdigest()[:8], 16
                )
                plans, coverage = build_paired_enrollment_plans(
                    labels, subjects, executions, list(stream.eval_labels),
                    requested_support=args.support, mode="cross_subject", seed=relation_seed,
                )
                coverage.update({
                    "query_stream": spec.stream_id,
                    "support_stream": spec.stream_id,
                    "configuration_relation": "same_configuration",
                    "subject_relation": "cross_subject",
                    "query_execution_granularity": stream.execution_granularity,
                    "support_execution_granularity": stream.execution_granularity,
                })
                protocol[relation_id] = coverage
                features = encode_harnet_stream(model, stream, device)
            except (FileNotFoundError, ValueError) as error:
                protocol[relation_id] = {"status": "incompatible", "reason": str(error)}
                print(f"{relation_id}: skipped ({error})", flush=True)
                continue

            for support_count in sorted(set(args.support)):
                key = f"{relation_id}/full/k{support_count}"
                if not plans:
                    result = {**coverage, "queries": 0}
                elif support_count > int(coverage["support_ceiling"]):
                    result = {
                        **coverage,
                        "status": "above_paired_support_ceiling",
                        "queries": 0,
                        "subjects": len(plans),
                    }
                else:
                    result = score_cell(
                        features, labels, plans, support_count, device, relation_seed
                    )
                    result["paired_protocol"] = coverage
                result.update({
                    "dataset": dataset,
                    "query_stream": spec.stream_id,
                    "support_stream": spec.stream_id,
                    "configuration_relation": "same_configuration",
                    "subject_relation": "cross_subject",
                    "enrollment_shape": "full",
                    "support_count": support_count,
                })
                results[key] = result
                if result.get("status"):
                    print(f"{key}: skipped ({result['status']})", flush=True)
                else:
                    print(
                        f"{key}: nearest={result['nearest_f1_macro']:.1f} "
                        f"prototype={result['prototype_f1_macro']:.1f} "
                        f"ridge={result['ridge_f1_macro']:.1f} n={result['queries']}",
                        flush=True,
                    )

    evaluation_protocol = {
        "protocol_role": args.protocol_role,
        "dataset_selection": "explicit" if explicit_datasets else "protocol_roster",
        "datasets": list(datasets),
        "support": sorted(set(args.support)),
        "modes": ["cross_subject"],
        "configuration_modes": ["same"],
        "enrollment_shapes": ["full"],
        "seed": args.seed,
        "curve_policy": "fixed_subject_candidate_query_cohort_with_nested_execution_support",
        "protocol": protocol,
    }
    output = args.out or _OUT / f"harnet_enrollment_{args.protocol_role}.json"
    payload = {
        "schema_version": 1,
        "representation": "harnet5_official_frozen_trunk",
        "backbone": {
            "name": HARNET_NAME,
            "repository": SSL_HUB_REPO,
            "tag": SSL_HUB_TAG,
            "fingerprint": backbone_fp,
            "feature_dim": FEAT_DIM,
            "input_contract": {
                "channels": list(ACC_CHANNELS),
                "rate_hz": TARGET_HZ,
                "samples": TARGET_LEN,
                "gravity": "retained",
            },
        },
        "protocol_role": args.protocol_role,
        "evaluation_protocol": evaluation_protocol,
        "evaluation_protocol_fp": _json_fingerprint(evaluation_protocol),
        "evaluation_source_fp": _source_fingerprint(),
        "results": results,
    }
    _write_json(output, payload)
    print(f"-> {output}", flush=True)


if __name__ == "__main__":
    main()
