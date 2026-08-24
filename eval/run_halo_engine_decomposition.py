"""Decompose HALO into raw and learned recording-level nearest-neighbor retrieval.

Every arm consumes the same serialized adaptation manifest, query recordings, enrolled recordings,
and frozen corpus bank.  The only intervention is whether memory contains support, corpus, or both,
and whether the learned per-row correction is enabled.  This is a checkpoint diagnostic rather than
a separate classifier or training path.
"""

from __future__ import annotations

import argparse
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


METHODS = (
    "support_raw_1nn",
    "support_reranked_1nn",
    "corpus_raw_1nn",
    "corpus_reranked_1nn",
    "full_raw_1nn",
    "full_reranked_1nn",
)
QUERY_CHUNK = 1024


def _take(rows: SensorRows, index: torch.Tensor) -> SensorRows:
    return SensorRows(**{
        field: None if getattr(rows, field) is None else getattr(rows, field)[index]
        for field in rows.__dataclass_fields__
    })


def _raw_and_reranked(result: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact raw and corrected nearest-candidate logits from one engine call."""
    raw = result["base_logits"]
    reranked = result["logits"]
    if raw.shape != reranked.shape or raw.dim() != 2:
        raise ValueError("engine decomposition expected query-by-candidate recording logits")
    return raw, reranked


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
    del seed  # the active reranker has no stochastic identity or selection path
    query_rows, query_window, _, _ = adapter._stream_rows(query_stream, state)
    support_rows, support_window, _, _ = adapter._stream_rows(support_stream, state)
    memory, support_windows, enrolled_rows = adapter._append_enrollment(
        state["bank"], support_rows, support_window, plan, support_count,
    )
    support_index = torch.nonzero(memory.enrolled_candidate.ge(0), as_tuple=True)[0]
    support = _take(memory, support_index)
    if len(support.feature) != enrolled_rows:
        raise ValueError("support-only memory did not isolate every enrolled recording")

    candidates = list(plan["candidate_names"])
    candidate_text = F.normalize(
        torch.from_numpy(state["sbert"]([name.replace("_", " ") for name in candidates])).float(),
        dim=-1,
    ).to(state["device"])
    requested = [int(row) for row in plan["query_rows"]]
    predicted: dict[str, dict[int, str]] = {method: {} for method in METHODS}

    with torch.no_grad():
        for start in range(0, len(requested), QUERY_CHUNK):
            chunk_windows = requested[start:start + QUERY_CHUNK]
            wanted = torch.as_tensor(
                chunk_windows, dtype=torch.long, device=query_window.device,
            )
            row_index = torch.nonzero(torch.isin(query_window, wanted), as_tuple=True)[0]
            if not len(row_index):
                raise ValueError("a decomposition query selected no recording rows")
            query = _take(query_rows, row_index)

            support_result = state["engine"](
                query, support, candidate_text, state["label_text"],
            )
            corpus_result = state["engine"](
                query, state["bank"], candidate_text, state["label_text"],
            )
            full_result = state["engine"](
                query, memory, candidate_text, state["label_text"],
            )
            support_raw, support_reranked = _raw_and_reranked(support_result)
            corpus_raw, corpus_reranked = _raw_and_reranked(corpus_result)
            full_raw, full_reranked = _raw_and_reranked(full_result)
            logits = {
                "support_raw_1nn": support_raw,
                "support_reranked_1nn": support_reranked,
                "corpus_raw_1nn": corpus_raw,
                "corpus_reranked_1nn": corpus_reranked,
                "full_raw_1nn": full_raw,
                "full_reranked_1nn": full_reranked,
            }
            windows = full_result["query_window"].cpu().tolist()
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
        "corpus_recordings": int(len(state["bank"].feature)),
        "enrolled_recordings": int(enrolled_rows),
        "enrolled_windows": int(support_windows),
        "total_memory_recordings": int(len(memory.feature)),
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
        "schema_version": 2,
        "experiment": "halo_recording_reranker_decomposition",
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
