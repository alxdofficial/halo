"""Wiring tests: gate persistence, the bank adapter, and the three-step predictor end to end.

The properties that matter here are contract properties, not numerics:

  * a schema-3 bank is REFUSED rather than silently scored through the pooled table, because
    "which sensor produced this row" — the gate's whole input — is undefined for pooled vectors;
  * a round-tripped gate produces bit-identical values, including its variable-width
    ``known_sensors`` buffer, which a naive load_state_dict silently drops;
  * evaluating the gate once per DISTINCT sensor description and indexing out is EXACTLY equal to
    evaluating it per row, so the de-duplication is a speedup and not an approximation;
  * admissibility remains a continuous retrieval prior at inference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from training.evidence.admissible_retrieval import (
    admissibility_from_gate, admissibility_from_unique,
)
from training.evidence.admissibility_gate import AdmissibilityGate
from training.evidence.gate_predictor import (
    BankRows, SensorBankStore, _SENSOR_DESCRIPTOR_EMBEDDINGS,
    _sensor_descriptor_embeddings, _stream_sensor_descriptions, concatenate_bank_rows,
    load_evidence_engine, load_gate, predict_bank, predict_bank_grouped, save_gate,
    sensor_rows_from_bank,
)
from training.evidence import bank_guard
from training.evidence.admissible_retrieval import SensorRows

TEXT, D, BIAS = 384, 16, 14
SEED = 20260812


def _gate(rank: int = 4, n_known: int = 5) -> AdmissibilityGate:
    torch.manual_seed(SEED)
    gate = AdmissibilityGate(rank=rank)
    g = torch.Generator().manual_seed(SEED)
    gate.known_sensors = F.normalize(torch.randn(n_known, TEXT, generator=g), dim=-1)
    with torch.no_grad():
        gate.neutral.fill_(0.33)
    return gate.eval()


def _bank_rows(R: int = 24, U: int = 3) -> BankRows:
    g = torch.Generator().manual_seed(SEED)
    uniq = F.normalize(torch.randn(U, TEXT, generator=g), dim=-1)
    descriptor_id = torch.arange(R) % U
    rows = SensorRows(
        feature=torch.randn(R, D, generator=g),
        descriptor=uniq.index_select(0, descriptor_id),
        bias=torch.randn(R, BIAS, generator=g),
        modality=torch.zeros(R, dtype=torch.long),
        gravity=torch.zeros(R, dtype=torch.long),
        label=torch.arange(R) % 4,
        dataset=torch.arange(R) % 2,
        enrolled_candidate=torch.full((R,), -1, dtype=torch.long),
    )
    return BankRows(rows=rows, unique_descriptor=uniq, descriptor_id=descriptor_id)


def _texts(C: int = 3, V: int = 4):
    g = torch.Generator().manual_seed(SEED + 1)
    return (F.normalize(torch.randn(C, TEXT, generator=g), dim=-1),
            F.normalize(torch.randn(V, TEXT, generator=g), dim=-1))


def test_enrollment_concatenation_preserves_distinct_recording_groups():
    from dataclasses import replace

    left, right = _bank_rows(6, 2), _bank_rows(4, 2)
    left = BankRows(replace(left.rows, source_window=torch.tensor([0, 0, 1, 1, 2, 2])),
                    left.unique_descriptor, left.descriptor_id)
    right = BankRows(replace(right.rows, source_window=torch.tensor([0, 0, 8, 8])),
                     right.unique_descriptor, right.descriptor_id)
    combined = concatenate_bank_rows(left, right)
    assert combined.rows.source_window.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_engine_checkpoint_round_trips_its_full_configuration(tmp_path):
    """Every field, not a hand-listed subset. A checkpoint rebuilt at the wrong shape reports the
    wrong model under the right name, and that is worse than failing to load."""
    from dataclasses import asdict
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine

    cfg = EngineConfig(
        spec=AttentionSpec(d_model=16, n_heads=4, ffn_mult=3, dropout=0.05),
        trunk_layers=2,
        semantic_scale=0.75,
        surrogate_temperature=0.08,
        telemetry_neighbors=5,
    )
    source = EvidenceEngine(None, cfg)
    path = tmp_path / "engine.pt"
    torch.save({"evidence_engine": source.state_dict(),
                "config": {"engine_config": asdict(cfg)}}, path)
    loaded = load_evidence_engine(path)
    assert loaded.cfg == cfg
    for name, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name


def test_an_engine_checkpoint_without_its_config_refuses_to_load(tmp_path):
    from model.evidence.engine import EvidenceEngine

    path = tmp_path / "engine.pt"
    torch.save({"evidence_engine": EvidenceEngine(None).state_dict(), "config": {}}, path)
    with pytest.raises(ValueError, match="engine_config"):
        load_evidence_engine(path)


def test_an_engine_checkpoint_with_missing_model_weights_refuses_to_load(tmp_path):
    """`strict=False` is allowed only to omit an unattached encoder, not engine parameters."""
    from dataclasses import asdict
    from model.evidence.engine import EvidenceEngine

    engine = EvidenceEngine(None)
    state = engine.state_dict()
    state.pop(next(key for key in state if key.startswith("reranker.")))
    path = tmp_path / "broken_engine.pt"
    torch.save({"evidence_engine": state,
                "config": {"engine_config": asdict(engine.cfg)}}, path)
    with pytest.raises(RuntimeError, match="missing"):
        load_evidence_engine(path)


def _schema_five_bank() -> dict:
    return {
        "schema_version": 5,
        "cfg_names": {0: "alpha/watch", 1: "beta/phone"},
        "sensor_descriptions": {
            0: [{"slot": 0, "text": "wrist accelerometer"}],
            1: [
                {"slot": 0, "text": "pocket accelerometer"},
                {"slot": 1, "text": "pocket gyroscope"},
            ],
        },
        "sensor": {
            "Z": torch.arange(4 * D, dtype=torch.float16).reshape(4, D),
            "bias": torch.arange(4 * BIAS, dtype=torch.float32).reshape(4, BIAS),
            "cfg": torch.tensor([0, 1, 1, 0]),
            "slot": torch.tensor([0, 0, 1, 0]),
            "modality": torch.tensor([0, 0, 1, 0]),
            "gravity": torch.tensor([0, 1, 0, 0]),
            "y": torch.tensor([1, 2, 3, 1]),
        },
    }


# ------------------------------------------------------------------------------ persistence
def test_gate_round_trips_including_the_known_sensor_buffer(tmp_path):
    """`known_sensors` has a corpus-dependent width; a load that ignores it silently disables
    abstention, turning every novel configuration into a confident extrapolation."""
    gate = _gate(n_known=7)
    path = save_gate(gate, tmp_path / "gate.pt", {"n_cells": 990})
    restored = load_gate(path)
    assert restored.known_sensors.shape == (7, TEXT)
    assert torch.allclose(restored.known_sensors, gate.known_sensors)
    assert float(restored.neutral) == float(gate.neutral)
    assert restored.rank == gate.rank
    s, c = _bank_rows().unique_descriptor, _texts()[0]
    assert torch.allclose(restored(s, c), gate(s, c))


def test_loading_a_missing_gate_fails_loudly(tmp_path):
    try:
        load_gate(tmp_path / "absent.pt")
    except FileNotFoundError:
        return
    raise AssertionError("a missing gate must fail, not fall back to an untrained one")


# -------------------------------------------------------------------------- de-duplication
def test_unique_evaluation_equals_per_row_evaluation():
    """The speedup must be EXACT — the gate reads only text, so duplicate rows are identical."""
    gate, bank = _gate(), _bank_rows()
    cand, _ = _texts()
    query_descriptor = bank.unique_descriptor[:2]
    per_row = admissibility_from_gate(gate, bank.rows, cand, query_descriptor)
    deduped = admissibility_from_unique(gate, bank.unique_descriptor, bank.descriptor_id,
                                        cand, query_descriptor)
    assert torch.allclose(per_row, deduped, atol=1e-6)


def test_query_and_evidence_gates_use_a_scale_preserving_geometric_mean():
    class FixedGate:
        def __call__(self, sensor, candidate):
            value = 0.25 if len(sensor) == 2 else 0.09
            return torch.full((len(sensor), len(candidate)), value)

    cand, _ = _texts(C=1)
    query = _bank_rows(U=2).unique_descriptor
    rows = _bank_rows(R=3, U=3)
    out = admissibility_from_unique(
        FixedGate(), rows.unique_descriptor, rows.descriptor_id, cand, query
    )
    assert torch.allclose(out, torch.full_like(out, 0.15), atol=1e-6)


def test_arbitrary_aliases_receive_neutral_admissibility():
    gate, bank = _gate(), _bank_rows()
    cand, _ = _texts()
    out = admissibility_from_unique(
        gate, bank.unique_descriptor, bank.descriptor_id, cand,
        bank.unique_descriptor[:2], semantic_labels=False,
    )
    assert torch.equal(out, torch.ones_like(out))


def test_arbitrary_alias_prediction_reuses_the_exact_gate_disabled_path():
    gate, bank = _gate(), _bank_rows()
    candidate, label_text = _texts()
    logits, aux = predict_bank(
        gate, bank,
        query_feature=torch.randn(2, D), query_bias=torch.randn(2, BIAS),
        query_descriptor=bank.unique_descriptor[:2],
        query_modality=torch.zeros(2, dtype=torch.long),
        query_gravity=torch.zeros(2, dtype=torch.long),
        candidate_text=candidate, label_text=label_text, top_k=6,
        semantic_labels=False,
    )
    assert torch.equal(logits, aux["identity_logits"])


# --------------------------------------------------------------------------- bank contract
def test_schema_three_bank_is_refused():
    """Pooled session vectors have no per-sensor identity, so admissibility is undefined for them."""
    for bank in ({"schema_version": 3, "cfg_names": {}},
                 {"schema_version": 4, "cfg_names": {}}):     # schema 4 but no sensor table
        try:
            sensor_rows_from_bank(bank)
        except ValueError as exc:
            assert "sensor" in str(exc).lower()
            continue
        raise AssertionError("a bank without per-sensor rows must be refused, not reinterpreted")


def test_sensor_descriptor_embeddings_are_computed_once(monkeypatch):
    calls = []

    def encoder(values):
        calls.append(list(values))
        return torch.stack([
            torch.full((TEXT,), float(index + 1)) for index, _ in enumerate(values)
        ]).numpy()

    import eval.scoring

    _SENSOR_DESCRIPTOR_EMBEDDINGS.clear()
    monkeypatch.setattr(eval.scoring, "get_sbert_encoder", lambda: encoder)
    first = _sensor_descriptor_embeddings(["alpha", "beta"])
    second = _sensor_descriptor_embeddings(["beta", "alpha"])
    assert calls == [["alpha", "beta"]]
    assert torch.equal(second, first.flip(0))
    _SENSOR_DESCRIPTOR_EMBEDDINGS.clear()


def test_gpu_row_store_matches_the_existing_bank_adapter(monkeypatch):
    def encoder(values):
        return torch.stack([
            F.normalize(torch.arange(TEXT, dtype=torch.float32) + index + 1, dim=0)
            for index, _ in enumerate(values)
        ]).numpy()

    import eval.scoring

    _SENSOR_DESCRIPTOR_EMBEDDINGS.clear()
    monkeypatch.setattr(eval.scoring, "get_sbert_encoder", lambda: encoder)
    bank = _schema_five_bank()
    selected = torch.tensor([3, 1, 2])
    expected = sensor_rows_from_bank(bank, row_indices=selected)
    actual = SensorBankStore.from_bank(bank, "cpu").select(selected)
    for name in expected.rows.__dataclass_fields__:
        got, want = getattr(actual.rows, name), getattr(expected.rows, name)
        if got is None or want is None:
            # `source_window` is optional provenance; both paths must agree on having it or not.
            assert got is want, f"{name}: one path carries provenance and the other does not"
            continue
        assert torch.equal(got, want), name
    assert torch.equal(
        actual.unique_descriptor.index_select(0, actual.descriptor_id),
        expected.unique_descriptor.index_select(0, expected.descriptor_id),
    )
    _SENSOR_DESCRIPTOR_EMBEDDINGS.clear()


def test_archive_descriptor_preserves_partner_modality_presence():
    accel_only = _stream_sensor_descriptions("capture24", "watch_wrist", ("accel",))
    paired = _stream_sensor_descriptions("uci_har", "phone_waist", ("accel", "gyro"))
    assert len(accel_only) == 1
    assert "without a gyroscope" in accel_only[0]
    assert "alongside a gyroscope" in paired[0]
    assert "alongside an accelerometer" in paired[1]


# ------------------------------------------------------------------------------- prediction
def test_predict_bank_returns_candidate_logits_and_gate_telemetry():
    gate, bank = _gate(), _bank_rows()
    cand, lab = _texts()
    logits, aux = predict_bank(
        gate, bank,
        query_feature=torch.randn(2, D), query_bias=torch.randn(2, BIAS),
        query_descriptor=bank.unique_descriptor[:2],
        query_modality=torch.zeros(2, dtype=torch.long),
        query_gravity=torch.zeros(2, dtype=torch.long),
        candidate_text=cand, label_text=lab, top_k=6,
    )
    assert logits.shape == (cand.shape[0],)
    assert torch.isfinite(logits).all()
    assert aux["identity_logits"].shape == (cand.shape[0],)
    for key in ("gate/mean", "gate/std", "gate/log_penalty_mean"):
        assert key in aux, "the collapse guard must be reported on every prediction"
    for key in ("evidence/all_candidates_zero", "evidence/zero_candidate_fraction"):
        assert key in aux, "evidence availability must be visible on every prediction"


def test_grouped_prediction_matches_individual_query_windows():
    gate, bank = _gate(), _bank_rows()
    candidate, label_text = _texts()
    g = torch.Generator().manual_seed(SEED + 9)
    query_feature = torch.randn(5, D, generator=g)
    query_bias = torch.randn(5, BIAS, generator=g)
    query_descriptor = F.normalize(torch.randn(5, TEXT, generator=g), dim=-1)
    query_modality = torch.zeros(5, dtype=torch.long)
    query_gravity = torch.zeros(5, dtype=torch.long)
    owner = torch.tensor([0, 0, 1, 2, 2])
    grouped, grouped_aux = predict_bank_grouped(
        gate, bank,
        query_feature, query_bias, query_descriptor, query_modality, query_gravity,
        candidate, label_text, top_k=6, query_group=owner, group_count=3,
    )
    expected, expected_identity = [], []
    for group in range(3):
        selected = owner.eq(group)
        logits, aux = predict_bank(
            gate, bank,
            query_feature[selected], query_bias[selected], query_descriptor[selected],
            query_modality[selected], query_gravity[selected], candidate, label_text, top_k=6,
        )
        expected.append(logits)
        expected_identity.append(aux["identity_logits"])
    assert torch.allclose(grouped, torch.stack(expected), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        grouped_aux["identity_logits"], torch.stack(expected_identity), atol=1e-6, rtol=1e-6,
    )


def test_low_admissibility_is_reported_but_does_not_hard_delete_evidence():
    gate, bank = _gate(), _bank_rows()
    cand, lab = _texts()
    kw = dict(query_feature=torch.randn(1, D), query_bias=torch.randn(1, BIAS),
              query_descriptor=bank.unique_descriptor[:1],
              query_modality=torch.zeros(1, dtype=torch.long),
              query_gravity=torch.zeros(1, dtype=torch.long),
              candidate_text=cand, label_text=lab, top_k=6)
    with torch.no_grad():
        gate.coupling.zero_()
        gate.offset.fill_(-20.0)
    logits, aux = predict_bank(gate, bank, **kw)
    assert torch.isfinite(logits).all()
    assert aux["gate/mean"] < 1e-6
    assert aux["gate/log_penalty_mean"] > 10.0


def test_text_embedding_probe_guard_detects_runtime_drift(monkeypatch):
    stored = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bank = {"text_embed_probe": stored.clone()}
    monkeypatch.setattr(bank_guard, "text_embedding_probe", lambda: stored.clone())
    bank_guard.assert_text_embedding_path_current(bank, context="test")

    monkeypatch.setattr(bank_guard, "text_embedding_probe", lambda: stored + 1.0)
    try:
        bank_guard.assert_text_embedding_path_current(bank, context="test")
    except SystemExit as exc:
        assert "TEXT-EMBEDDING PATH CHANGED" in str(exc)
        return
    raise AssertionError("a changed runtime text encoder must invalidate the bank")


def test_a_wholly_confident_gate_leaves_every_row_eligible():
    gate, bank = _gate(), _bank_rows()
    cand, lab = _texts()
    with torch.no_grad():
        gate.coupling.zero_()
        gate.offset.fill_(20.0)
    _, aux = predict_bank(
        gate, bank,
        query_feature=torch.randn(1, D), query_bias=torch.randn(1, BIAS),
        query_descriptor=bank.unique_descriptor[:1],
        query_modality=torch.zeros(1, dtype=torch.long),
        query_gravity=torch.zeros(1, dtype=torch.long),
        candidate_text=cand, label_text=lab, top_k=6)
    assert aux["gate/log_penalty_mean"] < 1e-6
    assert aux["gate/mean"] > 0.99


def test_topk_path_backpropagates_into_both_gate_projections():
    gate, bank = AdmissibilityGate(rank=2), _bank_rows()
    cand, lab = _texts()
    logits, _ = predict_bank(
        gate, bank,
        query_feature=torch.randn(2, D), query_bias=torch.randn(2, BIAS),
        query_descriptor=bank.unique_descriptor[:2],
        query_modality=torch.zeros(2, dtype=torch.long),
        query_gravity=torch.zeros(2, dtype=torch.long),
        candidate_text=cand, label_text=lab, top_k=12,
    )
    (-torch.log_softmax(logits, dim=0)[0]).backward()
    for name, parameter in (
        ("sensor projection", gate.sensor_proj.weight),
        ("concept projection", gate.concept_proj.weight),
        ("coupling", gate.coupling),
    ):
        assert parameter.grad is not None and float(parameter.grad.norm()) > 0.0, name


def test_the_raw_nearest_control_stays_closed_form_under_a_trained_engine():
    """The reranker exposes the unchanged cosine-nearest control beside learned logits."""
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine
    from training.evidence.admissible_retrieval import SensorRows

    torch.manual_seed(1)
    Q, M, C, V, d = 2, 8, 3, 7, 16
    query = SensorRows(
        feature=F.normalize(torch.randn(Q, d), dim=-1),
        descriptor=F.normalize(torch.randn(Q, 384), dim=-1), bias=torch.zeros(Q, 1),
        modality=torch.zeros(Q, dtype=torch.long), gravity=torch.zeros(Q, dtype=torch.long),
        label=torch.full((Q,), -1), dataset=torch.zeros(Q, dtype=torch.long),
        enrolled_candidate=torch.full((Q,), -1), source_window=torch.arange(Q),
    )
    memory = SensorRows(
        feature=F.normalize(torch.randn(M, d), dim=-1),
        descriptor=F.normalize(torch.randn(M, 384), dim=-1), bias=torch.zeros(M, 1),
        modality=torch.zeros(M, dtype=torch.long), gravity=torch.zeros(M, dtype=torch.long),
        label=torch.randint(0, V, (M,)), dataset=torch.zeros(M, dtype=torch.long),
        enrolled_candidate=torch.full((M,), -1), source_window=torch.arange(M),
    )
    candidate = F.normalize(torch.randn(C, 384), dim=-1)
    label_text = F.normalize(torch.randn(V, 384), dim=-1)
    engine = EvidenceEngine(None, EngineConfig(spec=AttentionSpec(d_model=d, n_heads=4))).eval()
    with torch.no_grad():
        result = engine(query, memory, candidate, label_text)
    expected = F.normalize(query.feature, dim=-1) @ F.normalize(memory.feature, dim=-1).T
    assert torch.allclose(result["base_scores"], expected, atol=1e-6)
    assert result["logits"].shape == result["base_logits"].shape == (Q, C)
    assert torch.isfinite(result["logits"]).all()


def test_a_bank_without_recording_provenance_is_refused_by_the_engine_path():
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine
    from training.evidence.admissible_retrieval import SensorRows

    torch.manual_seed(2)
    S, R, C, V, d = 2, 16, 3, 5, 16
    query = SensorRows(
        feature=F.normalize(torch.randn(S, d), dim=-1),
        descriptor=F.normalize(torch.randn(S, 384), dim=-1), bias=torch.zeros(S, 1),
        modality=torch.zeros(S, dtype=torch.long), gravity=torch.zeros(S, dtype=torch.long),
        label=torch.full((S,), -1), dataset=torch.zeros(S, dtype=torch.long),
        enrolled_candidate=torch.full((S,), -1), source_window=torch.arange(S),
    )
    memory = SensorRows(
        feature=F.normalize(torch.randn(R, d), dim=-1),
        descriptor=F.normalize(torch.randn(R, 384), dim=-1), bias=torch.zeros(R, 1),
        modality=torch.zeros(R, dtype=torch.long), gravity=torch.zeros(R, dtype=torch.long),
        label=torch.randint(0, V, (R,)), dataset=torch.zeros(R, dtype=torch.long),
        enrolled_candidate=torch.full((R,), -1, dtype=torch.long), source_window=None,
    )
    engine = EvidenceEngine(None, EngineConfig(spec=AttentionSpec(d_model=d, n_heads=4))).eval()
    with pytest.raises(ValueError, match="source_window"):
        engine(query, memory, F.normalize(torch.randn(C, 384), dim=-1),
               F.normalize(torch.randn(V, 384), dim=-1))


def test_engine_checkpoints_survive_a_config_default_change(tmp_path):
    """A default reranker checkpoint can omit its nested config without changing behaviour."""
    from dataclasses import asdict
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine

    cfg = EngineConfig(spec=AttentionSpec(d_model=16, n_heads=4))
    source = EvidenceEngine(None, cfg)
    payload = asdict(cfg)
    del payload["reranker"]
    path = tmp_path / "engine.pt"
    torch.save({"evidence_engine": source.state_dict(), "config": {"engine_config": payload}}, path)
    loaded = load_evidence_engine(path)
    assert loaded.cfg == cfg
