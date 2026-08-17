import json

import numpy as np
import pytest
import torch

from eval.data import EvalStream
from eval.enrollment_protocol import (
    SCHEMA_VERSION,
    _zero_shot_cell,
    _positive_cell,
    load_manifest,
    save_manifest,
    stream_fingerprint,
)
from eval.run_adaptation_baselines import _macro_f1_positions, support_predictions


def _stream(name="wrist"):
    labels, subjects, executions = [], [], []
    windows = []
    for subject in ("s1", "s2", "s3"):
        for label in ("walk", "sit"):
            for execution in range(4):
                labels.append(label)
                subjects.append(subject)
                executions.append(f"{subject}:{label}:{execution}")
                windows.append(np.full((12, 3), execution, dtype=np.float32))
    return EvalStream(
        dataset="synthetic",
        stream=name,
        alignment="native",
        windows=np.stack(windows),
        gt=labels,
        subjects=np.asarray(subjects, dtype=object),
        channels=["acc_x", "acc_y", "acc_z"],
        rate_hz=2.0,
        mask=np.ones(3, dtype=bool),
        eval_labels=["walk", "sit"],
        execution_ids=np.asarray(executions, dtype=object),
    )


def test_fast_position_macro_f1_matches_shared_metric():
    from eval.scoring import classification_metrics

    candidates = np.asarray(["walk", "sit", "run"], dtype=object)
    truth = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    prediction = np.asarray([0, 2, 1, 0, 2], dtype=np.int64)
    expected = classification_metrics(
        candidates[truth].tolist(), candidates[prediction].tolist()
    )["f1_macro"]
    assert _macro_f1_positions(truth, prediction, len(candidates)) == pytest.approx(expected)


def test_positive_manifest_uses_fixed_roster_nested_disjoint_executions():
    stream = _stream()
    cell = _positive_cell(
        stream,
        stream,
        regime="ordinary",
        subject_relation="same_subject",
        configuration_relation="same_configuration",
        support_counts=[0, 1, 2],
        seeds=[3, 4],
    )
    assert cell["status"] == "ok"
    assert cell["candidate_names"] == ["walk", "sit"]
    assert cell["support_ceiling"] == 2
    for payload in cell["seeds"].values():
        for plan in payload["plans"]:
            assert plan["candidate_names"] == ["walk", "sit"]
            assert all(len(rows) == 2 for rows in plan["support_execution_rows"])
            support = {value for values in plan["support_execution_ids"] for value in values}
            assert support.isdisjoint(plan["query_execution_ids"])


def test_positive_manifest_lowers_ceiling_without_shrinking_candidates():
    stream = _stream()
    cell = _positive_cell(
        stream,
        stream,
        regime="ordinary",
        subject_relation="same_subject",
        configuration_relation="same_configuration",
        support_counts=[0, 1, 2, 4, 8],
        seeds=[3],
    )
    # Four executions cannot provide k=4 support and a disjoint query, but k=2 can.
    assert cell["support_ceiling"] == 2
    assert cell["candidate_names"] == ["walk", "sit"]


def test_manifest_content_hash_rejects_tampering(tmp_path):
    stream = _stream()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": "test",
        "alignment": "native",
        "datasets": [],
        "action_regimes": {},
        "support_counts": [0],
        "seeds": [1],
        "stream_fingerprints": {},
        "cells": {},
    }
    from eval.enrollment_protocol import _json_hash
    manifest["manifest_fingerprint"] = _json_hash(manifest)
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)
    assert load_manifest(path, validate_grids=False)["protocol_name"] == "test"
    blob = json.loads(path.read_text())
    blob["support_counts"] = [0, 1]
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="content fingerprint"):
        load_manifest(path, validate_grids=False)
    assert len(stream_fingerprint(stream)) == 64


def test_frozen_adaptation_controls_learn_separable_support():
    features = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
    ], dtype=np.float32)
    predicted = support_predictions(
        features,
        np.asarray([0, 2]),
        np.asarray([0, 1]),
        np.asarray([1, 3]),
        2,
        device=torch.device("cpu"),
        seed=7,
    )
    assert set(predicted) == {"nearest", "prototype", "ridge", "linear_head"}
    for values in predicted.values():
        np.testing.assert_array_equal(values, [0, 1])


def test_external_runner_consumes_manifest_without_rebuilding_episodes(tmp_path, monkeypatch):
    import baselines
    import eval.run_adaptation_baselines as runner
    from baselines.base import BaselineAdapter

    stream = _stream()
    positive = _positive_cell(
        stream, stream, regime="ordinary", subject_relation="cross_subject",
        configuration_relation="same_configuration", support_counts=[0, 1, 2], seeds=[3],
    )
    manifest = {
        "manifest_fingerprint": "manifest-test",
        "alignment": "native",
        "support_counts": [0, 1, 2],
        "cells": {
            "synthetic/wrist/zero_shot": _zero_shot_cell(stream, regime="ordinary"),
            "synthetic/wrist/from_wrist/same_configuration/cross_subject": positive,
        },
    }

    class FakeAdapter(BaselineAdapter):
        name = "manifest_fake"
        tier = "fake"

        def setup(self, device):
            return None

        def predict_candidates(self, value, candidates, state, device):
            return list(value.gt), {}

        def predict_candidates_from_features(self, features, candidates, state, device):
            names = np.asarray(list(candidates), dtype=object)
            return names[np.asarray(features).argmax(1)].tolist(), {}

        def window_features(self, value, state, device):
            return np.asarray(
                [[1.0, 0.0] if label == "walk" else [0.0, 1.0] for label in value.gt],
                dtype=np.float32,
            )

    baselines.REGISTRY[FakeAdapter.name] = FakeAdapter()
    monkeypatch.setattr(runner, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(runner, "load_eval_stream", lambda *args, **kwargs: stream)
    out = tmp_path / "result.json"
    try:
        payload = runner.run(
            baseline_name=FakeAdapter.name,
            manifest_path=tmp_path / "unused.json",
            device="cpu",
            out=out,
        )
    finally:
        baselines.REGISTRY.pop(FakeAdapter.name, None)
    assert out.exists()
    assert payload["manifest_fingerprint"] == "manifest-test"
    assert any(key.endswith("/coherent/k0") for key in payload["results"])
    positive_results = [
        value for value in payload["results"].values()
        if value.get("kind") == "enrollment"
    ]
    assert positive_results
    assert all(value["prototype"]["f1_macro"] == 100.0 for value in positive_results)
