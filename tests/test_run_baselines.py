"""End-to-end test for the baseline RUNNER + TABLE ASSEMBLER.

Concrete baseline adapters may not all be registered yet, so this exercises the
full run -> JSON -> assemble -> table path against a tiny synthetic
``CosineAdapter`` (random embeddings) on a real eval grid (motionsense). It also
locks in the fail-loud discipline: a crashing adapter is recorded as a failed
cell, and the assembler refuses to build a table over a failed/missing cell.
"""

import json

import numpy as np
import pytest

import baselines as B
from baselines.base import CosineAdapter
from eval import assemble_table, run_baselines
from eval.run_baselines import result_path

DATASET = "motionsense"
STREAM = "phone_front_pocket"


def _grid_exists() -> bool:
    return (run_baselines.REPO / "data" / "datasets" / DATASET / "grids"
            / "non_harmonised" / STREAM / "data.npy").exists()


requires_grid = pytest.mark.skipif(
    not _grid_exists(), reason=f"no {DATASET}/{STREAM} eval grid available")


class _CosineRand(CosineAdapter):
    """Random-embedding cosine baseline — real end-to-end path, no model."""
    name = "cosine_rand_test"
    D = 16

    def setup(self, device):
        return {"rng": np.random.RandomState(0)}

    def window_embeddings(self, stream, state, device):
        e = state["rng"].randn(stream.n_windows, self.D).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def encode_labels(self, labels, state, device):
        e = state["rng"].randn(len(labels), self.D).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


class _Crashes(CosineAdapter):
    name = "crashes_test"
    D = 8

    def setup(self, device):
        return {}

    def window_embeddings(self, stream, state, device):
        raise RuntimeError("boom")

    def encode_labels(self, labels, state, device):
        return np.zeros((len(labels), self.D), np.float32)


@pytest.fixture
def registered():
    """Register the synthetic adapters for the test, then restore the registry."""
    saved = dict(B.REGISTRY)
    B.REGISTRY[_CosineRand.name] = _CosineRand()
    B.REGISTRY[_Crashes.name] = _Crashes()
    try:
        yield
    finally:
        B.REGISTRY.clear()
        B.REGISTRY.update(saved)


@requires_grid
def test_run_then_assemble_end_to_end(registered, tmp_path):
    ran, failed = run_baselines.run(
        [_CosineRand.name], [DATASET],
        alignment="non_harmonised", device="cpu", results_dir=tmp_path)

    assert ran == [_CosineRand.name]
    assert failed == []

    out = result_path(tmp_path, _CosineRand.name, DATASET, STREAM)
    assert out.exists() and not out.with_suffix(".partial.json").exists()
    payload = json.loads(out.read_text())
    assert payload["_status"] == "complete"
    assert 0.0 <= payload["metrics"]["f1_macro"] <= 100.0
    # a real subject-stratified CI was computed (motionsense has many subjects)
    assert not payload["metrics"]["ci_degenerate"]

    rc = assemble_table.main(
        ["--baselines", _CosineRand.name, "--datasets", DATASET,
         "--results-dir", str(tmp_path)])
    assert rc == 0

    cells, tbl, rejected = assemble_table.collect(
        [_CosineRand.name], [DATASET], tmp_path)
    assert rejected == []
    md = assemble_table.render([_CosineRand.name], cells, tbl)
    assert DATASET in md and "mean" in md


@requires_grid
def test_crash_is_recorded_and_assembler_rejects_it(registered, tmp_path):
    ran, failed = run_baselines.run(
        [_Crashes.name], [DATASET],
        alignment="non_harmonised", device="cpu", results_dir=tmp_path)

    # the crash is RECORDED, not swallowed
    assert failed == [(_Crashes.name, DATASET, STREAM)]
    out = result_path(tmp_path, _Crashes.name, DATASET, STREAM)
    assert json.loads(out.read_text())["_status"] == "failed"

    # ...and the assembler refuses to build a table with that hole (fail loud)
    rc = assemble_table.main(
        ["--baselines", _Crashes.name, "--datasets", DATASET,
         "--results-dir", str(tmp_path)])
    assert rc == 1


@requires_grid
def test_missing_cell_is_rejected_loudly(registered, tmp_path):
    # never ran the runner -> no JSON at all -> assembler must reject, not blank-fill
    _cells, _tbl, rejected = assemble_table.collect(
        [_CosineRand.name], [DATASET], tmp_path)
    assert len(rejected) == 1 and rejected[0].startswith("MISSING")


def test_resolve_eval_cells_unknown_dataset_fails_loud():
    with pytest.raises(ValueError):
        run_baselines.resolve_eval_cells(["not_a_dataset"])
