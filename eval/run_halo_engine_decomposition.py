"""Decompose HALO enrollment into retrieval, corpus-vote, and mixer stages.

Every arm consumes the same serialized adaptation manifest and the same encoded query/support rows.
This is a diagnostic of one trained checkpoint, not a new training or test-set selection path.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import baselines
from eval.data import load_eval_stream
from eval.enrollment_protocol import iter_cells, load_manifest
from eval.run_adaptation_baselines import (
    _atomic_write,
    _file_fingerprint,
    _git_provenance,
    _source_fingerprint,
)
from eval.scoring import align_ground_truth_labels, classification_metrics
from model.evidence.rows import SensorRows


# Keep method names explicit: these strings become the durable result-table row labels.
METHODS = (
    "support_patch_1nn",
    "support_soft_vote",
    "support_mixer",
    "corpus_only_semantic_vote",
    "full_uniform_corpus_vote",
    "full_semantic_topk_vote",
    "full_semantic_bank_vote",
    "full_engine",
)
QUERY_CHUNK = 4096


def _take(rows: SensorRows, index: torch.Tensor) -> SensorRows:
    return SensorRows(**{
        field: None if getattr(rows, field) is None else getattr(rows, field)[index]
        for field in rows.__dataclass_fields__
    })


def _recording_logits(row_mass: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    n_recordings = int(inverse.max().item()) + 1
    mass = row_mass.new_zeros((n_recordings, row_mass.shape[1]))
    mass.index_add_(0, inverse, row_mass)
    counts = torch.bincount(inverse, minlength=n_recordings).to(mass.dtype).unsqueeze(1)
    return (mass / counts.clamp_min(1)).clamp_min(1e-8).log()


def _support_hard_logits(result: dict, support: SensorRows, candidates: int) -> torch.Tensor:
    nearest = result["scores"].argmax(dim=1)
    bound = support.enrolled_candidate.to(nearest.device)[nearest]
    if bool(bound.lt(0).any()) or bool(bound.ge(candidates).any()):
        raise ValueError("support-only hard retrieval encountered an unbound support row")
    row_mass = F.one_hot(bound, candidates).to(result["scores"].dtype)
    return _recording_logits(row_mass, result["query_inverse"])


def _uniform_corpus_logits(result: dict, memory: SensorRows, candidates: int) -> torch.Tensor:
    """Use normal retrieval but make every corpus row an uninformative uniform vote."""
    bound = memory.enrolled_candidate.to(result["scores"].device)
    enrolled = F.one_hot(bound.clamp_min(0), candidates).to(result["scores"].dtype)
    uniform = torch.full_like(enrolled, 1.0 / candidates)
    row_vote = torch.where(bound.unsqueeze(1).ge(0), enrolled, uniform)
    row_mass = torch.softmax(result["scores"].float(), dim=1) @ row_vote.float()
    return _recording_logits(row_mass, result["query_inverse"])


def _topk_base_logits(
    result: dict,
    memory: SensorRows,
    candidate_text: torch.Tensor,
    label_text: torch.Tensor,
) -> torch.Tensor:
    """Production top-k vote computed from the production engine's selected rows and scores."""
    from model.evidence.engine import vote
    from model.evidence.rows import evidence_label_tokens

    selected = result["selected"][result["query_inverse"]]
    labels = evidence_label_tokens(memory, candidate_text, label_text)
    row_mass = vote(
        result["scores"].gather(1, selected),
        labels[selected],
        candidate_text,
        memory.enrolled_candidate.to(selected.device)[selected],
    )
    return _recording_logits(row_mass, result["query_inverse"])


def predict_plan(
    adapter,
    query_stream,
    support_stream,
    plan: dict,
    support_count: int,
    state: dict,
    *,
    seed: int,
) -> tuple[dict[str, list[str]], dict]:
    query_rows, query_window, _, _ = adapter._stream_rows(query_stream, state)
    support_rows, support_window, _, _ = adapter._stream_rows(support_stream, state)
    memory, support_windows, enrolled_rows = adapter._append_enrollment(
        state["bank"], support_rows, support_window, plan, support_count,
    )
    support_index = torch.nonzero(memory.enrolled_candidate.ge(0), as_tuple=True)[0]
    support = _take(memory, support_index)
    if len(support.feature) != enrolled_rows:
        raise ValueError("the support-only decomposition did not isolate every enrolled row")

    candidates = list(plan["candidate_names"])
    candidate_text = F.normalize(
        torch.from_numpy(state["sbert"]([name.replace("_", " ") for name in candidates])).float(),
        dim=-1,
    ).to(state["device"])
    requested = [int(row) for row in plan["query_rows"]]
    predicted: dict[str, dict[int, str]] = {method: {} for method in METHODS}
    full_generator = torch.Generator().manual_seed(int(seed))
    support_generator = torch.Generator().manual_seed(int(seed))
    corpus_generator = torch.Generator().manual_seed(int(seed))

    with torch.no_grad():
        for start in range(0, len(requested), QUERY_CHUNK):
            chunk_windows = requested[start:start + QUERY_CHUNK]
            wanted = torch.as_tensor(
                chunk_windows, dtype=torch.long, device=query_window.device,
            )
            row_index = torch.nonzero(torch.isin(query_window, wanted), as_tuple=True)[0]
            if not len(row_index):
                raise ValueError("a decomposition query selected no sensor rows")
            query = _take(query_rows, row_index)

            full = state["engine"](
                query, memory, candidate_text, state["label_text"], generator=full_generator,
            )
            support_result = state["engine"](
                query, support, candidate_text, state["label_text"],
                generator=support_generator,
            )
            corpus_result = state["engine"](
                query, state["bank"], candidate_text, state["label_text"],
                generator=corpus_generator,
            )
            logits = {
                "support_patch_1nn": _support_hard_logits(
                    support_result, support, len(candidates),
                ),
                "support_soft_vote": support_result["base_logits"],
                "support_mixer": support_result["logits"],
                "corpus_only_semantic_vote": corpus_result["base_logits"],
                "full_uniform_corpus_vote": _uniform_corpus_logits(
                    full, memory, len(candidates),
                ),
                "full_semantic_topk_vote": _topk_base_logits(
                    full, memory, candidate_text, state["label_text"],
                ),
                "full_semantic_bank_vote": full["base_logits"],
                "full_engine": full["logits"],
            }
            windows = full["query_window"].cpu().tolist()
            for method, values in logits.items():
                positions = values.argmax(dim=1).cpu().tolist()
                for window, position in zip(windows, positions, strict=True):
                    predicted[method][int(window)] = candidates[int(position)]

    missing = {
        method: [window for window in requested if window not in values]
        for method, values in predicted.items()
    }
    if any(missing.values()):
        raise ValueError(f"decomposition omitted query windows: {missing}")
    return {
        method: [values[window] for window in requested]
        for method, values in predicted.items()
    }, {
        "corpus_rows": int(len(state["bank"].feature)),
        "enrolled_rows": int(enrolled_rows),
        "enrolled_windows": int(support_windows),
        "total_memory_rows": int(len(memory.feature)),
    }


def _score_cell(
    adapter,
    query_stream,
    support_stream,
    query_labels: np.ndarray,
    plans: list[dict],
    support_count: int,
    state: dict,
    seed: int,
) -> dict:
    truth: list[str] = []
    predictions: dict[str, list[str]] = defaultdict(list)
    subjects = {}
    infos = []
    for plan_index, plan in enumerate(plans):
        plan_predictions, info = predict_plan(
            adapter, query_stream, support_stream, plan, support_count, state,
            seed=seed + plan_index,
        )
        rows = np.asarray(plan["query_rows"], dtype=np.int64)
        plan_truth = query_labels[rows].tolist()
        truth.extend(plan_truth)
        record = {"queries": len(rows)}
        for method, values in plan_predictions.items():
            predictions[method].extend(values)
            record[f"{method}_f1_macro"] = classification_metrics(
                plan_truth, values,
            )["f1_macro"]
        subjects[str(plan["subject"])] = record
        infos.append(info)
    result = {
        method: classification_metrics(truth, values)
        for method, values in predictions.items()
    }
    return {"queries": len(truth), "subject_results": subjects, "adapter_info": infos, **result}


def run(manifest_path: Path, out: Path, device: str) -> dict:
    manifest = load_manifest(manifest_path, validate_grids=True)
    adapter = baselines.REGISTRY["halo_compact"]
    resolved_device = torch.device(device)
    started = time.time()
    state = adapter.setup(resolved_device)
    streams = {}

    def stream(dataset: str, stream_id: str):
        key = (dataset, stream_id)
        if key not in streams:
            streams[key] = load_eval_stream(
                dataset, stream_id, alignment=manifest["alignment"],
            )
        return streams[key]

    results = {}
    for cell_id, cell in iter_cells(manifest):
        if cell["kind"] != "enrollment" or cell["status"] != "ok":
            continue
        dataset = cell["dataset"]
        query_stream = stream(dataset, cell["query_stream"])
        support_stream = stream(dataset, cell["support_stream"])
        query_labels = np.asarray(
            align_ground_truth_labels(query_stream.gt, query_stream.eval_labels), dtype=object,
        )
        for seed_text, seed_payload in cell["seeds"].items():
            seed = int(seed_text)
            for support_count in manifest["support_counts"]:
                if support_count <= 0:
                    continue
                payload = seed_payload
                ceiling = int(cell["support_ceiling"])
                cohort = "main"
                if support_count > ceiling:
                    secondary = cell.get("secondary_high_support", {})
                    if secondary.get("status") != "ok" or support_count > int(
                        secondary.get("support_ceiling", 0)
                    ):
                        continue
                    payload = secondary["seeds"][seed_text]
                    cohort = "secondary_high_support"
                scored = _score_cell(
                    adapter, query_stream, support_stream, query_labels,
                    payload["plans"], support_count, state, seed,
                )
                results[f"{cell_id}/coherent/seed{seed}/k{support_count}"] = {
                    **scored,
                    "dataset": dataset,
                    "regime": cell["regime"],
                    "support_count": support_count,
                    "seed": seed,
                    "cohort": cohort,
                }
        print(f"[halo-decomposition] {cell_id}: complete", flush=True)

    result = {
        "schema_version": 1,
        "experiment": "halo_engine_decomposition",
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "methods": list(METHODS),
        "adapter_source_fingerprint": _source_fingerprint(adapter),
        "decomposition_source": _file_fingerprint(Path(__file__)),
        "evaluation_artifacts": {
            name: _file_fingerprint(path)
            for name, path in adapter.evaluation_artifacts(state).items()
        },
        "adapter_config": adapter.evaluation_config(state),
        "elapsed_seconds": time.time() - started,
        **_git_provenance(),
        "results": results,
    }
    _atomic_write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.manifest, args.out, args.device)
    print(f"wrote {len(result['results'])} cells to {args.out}")


if __name__ == "__main__":
    main()
