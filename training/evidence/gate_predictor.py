"""Wiring: fit an ``AdmissibilityGate``, persist it, and run the three-step predictor on a bank.

This is the seam between the gate (a function of text) and Phase-B evaluation (a bank of per-sensor
rows). Three jobs:

  1. ``fit_from_table``   — warm-start a gate on measured resolvability and write it to disk with the
                            provenance needed to tell later which table and encoder produced it.
  2. ``sensor_rows_from_bank`` — turn a schema-5 memory bank into the ``SensorRows`` the predictor
                            consumes, using the exact sensor text and gravity convention captured
                            when the bank was built.
  3. ``predict_bank``     — the closed-form three-step prediction for a batch of query sensors.

WHY THE DESCRIPTOR TEXT IS STORED ONCE, NOT PER ROW
---------------------------------------------------
A bank holds ~10^5 sensor rows drawn from ~40 distinct sensor descriptions. Storing a 384-d
descriptor on every row would add ~150 MB of near-perfect duplication. The bank stores one exact text
record per ``(cfg, slot)`` and fingerprints that map. The gate embeds each distinct text once and
indexes it out, which is exact rather than approximate and cannot silently change after a policy edit.

SCHEMA 4 IS REQUIRED, AND THE FAILURE IS LOUD
---------------------------------------------
Per-sensor rows exist only when the bank was built with ``--sensor-rows`` against a sensor-granularity
encoder. A schema-3 bank has pooled session vectors, for which "which sensor is this" is undefined and
admissibility is meaningless. Falling back to the pooled table would silently score a different
quantity, so it raises instead.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from training.evidence.admissible_retrieval import (
    engine_vote,
    BIAS_BLEND_WEIGHT, SensorRows, admissibility_from_unique,
    compatibility_mask, rank_scores, vote,
)
from training.evidence.admissibility_gate import (
    DEFAULT_RANK, AdmissibilityGate, fit_gate, observations_from_table,
)

GATE_PATH = Path("training/evidence/outputs/admissibility_gate.pt")
GATE_ARTIFACT_VERSION = 4
ADMISSIBILITY_TRAINING_REGIME = "admissibility_gate_train_only_sensor_v1"
ADMISSIBILITY_REFINEMENT_REGIME = "admissibility_gate_stage2_soft_retrieval_v1"
NATIVE_JOINT_EPISODIC_REGIME = "admissibility_gate_native_joint_episodic_e2e_v1"
_SENSOR_DESCRIPTOR_EMBEDDINGS: dict[str, torch.Tensor] = {}


def _sensor_descriptor_embeddings(texts: Sequence[str]) -> torch.Tensor:
    """Embed each immutable bank descriptor once per process.

    Phase-B repeatedly selects rows carrying the same small descriptor vocabulary. Calling MiniLM
    for every query is both exact duplication and substantially slower than retrieval itself.
    """
    values = [str(text) for text in texts]
    missing = list(dict.fromkeys(
        text for text in values if text not in _SENSOR_DESCRIPTOR_EMBEDDINGS
    ))
    if missing:
        from eval.scoring import get_sbert_encoder

        encoded = np.asarray(get_sbert_encoder()(missing), dtype=np.float32)
        _SENSOR_DESCRIPTOR_EMBEDDINGS.update({
            text: torch.from_numpy(vector.copy()) for text, vector in zip(missing, encoded)
        })
    return torch.stack([_SENSOR_DESCRIPTOR_EMBEDDINGS[text] for text in values])


# --------------------------------------------------------------------------------- persistence
def save_gate(
    gate: AdmissibilityGate,
    path: Path = GATE_PATH,
    provenance: dict | None = None,
    bank: dict | None = None,
    training_regime: str = ADMISSIBILITY_TRAINING_REGIME,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_kind": "halo_admissibility_gate",
        "artifact_version": GATE_ARTIFACT_VERSION,
        "predictor_mode": "admissibility_gate",
        "training_regime": training_regime,
        "state_dict": gate.state_dict(),
        "rank": gate.rank,
        "text_dim": gate.sensor_proj.in_features,
        "provenance": provenance or {},
    }
    if bank is not None:
        from training.evidence.bank_guard import bank_fingerprint

        checkpoint_sha = (provenance or {}).get("resolvability_checkpoint_sha256")
        backbone_sha = (bank.get("backbone") or {}).get("fingerprint")
        if checkpoint_sha != backbone_sha:
            raise ValueError(
                "the gate's resolvability checkpoint and memory-bank backbone differ: "
                f"{checkpoint_sha!r} != {backbone_sha!r}"
            )
        payload.update({
            "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
            "vocab": list(bank.get("vocab", ())),
            "backbone_fp": backbone_sha,
        })
    torch.save(payload, path)
    return path


def load_evidence_engine(path, encoder=None, device="cpu"):
    """The trained evidence engine from a Phase-B checkpoint, or None if it has none.

    Returning None rather than raising is deliberate: a checkpoint written before the engine
    existed is still legitimately scoreable under the closed-form rule. A checkpoint that DOES
    carry engine weights but cannot be reconstructed is an error, because silently dropping to the
    closed-form path would report the wrong model under the right name.
    """
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine
    from model.evidence.evidence_mixer import EvidenceMixerConfig
    from model.evidence.evidence_reranker import EvidenceRerankerConfig
    from model.evidence.retrieval_scorer import PairScorerConfig

    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("evidence_engine")
    if state is None:
        return None
    saved = dict(blob.get("config", {}).get("engine_config") or {})
    if not saved:
        raise ValueError(
            "checkpoint carries engine weights but no engine_config; it cannot be rebuilt without "
            "guessing its shape, and a guess would report the wrong model under the right name"
        )
    has_reranker = "reranker.row_head.weight" in state
    has_historical_mixer = "mixer.residual_head.weight" in state
    if not has_reranker and not has_historical_mixer:
        raise ValueError(
            "checkpoint has evidence-engine weights but neither the current scalar reranker nor "
            "the supported historical candidate-residual mixer"
        )
    scorer_cfg = dict(saved.get("scorer", {}))
    if "learned" not in scorer_cfg:
        # Checkpoints written before the `learned` field default-drifted when the field's default
        # later changed to False: a model TRAINED with the learned scorer would silently be rebuilt
        # with the fixed cosine. The weights say which one was trained; believe them.
        scorer_cfg["learned"] = "scorer.base_gain" in state
    saved = {**saved, "scorer": scorer_cfg}
    cfg = EngineConfig(
        spec=AttentionSpec(**saved["spec"]),
        trunk_layers=int(saved.get("trunk_layers", 3)),
        top_k=int(saved["top_k"]),
        scorer=PairScorerConfig(**saved.get("scorer", {})),
        mixer=EvidenceMixerConfig(**saved.get("mixer", {})),
        reranker=EvidenceRerankerConfig(**saved.get("reranker", {})),
        mixing=saved.get("mixing", "rerank" if has_reranker else "attention"),
        vote_scope=saved.get("vote_scope", "bank"),
    )
    engine = EvidenceEngine(encoder, cfg)
    if encoder is not None:
        engine.load_state_dict(state, strict=True)
    else:
        incompatible = engine.load_state_dict(state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.startswith("encoder.")]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "evidence-engine checkpoint does not match its saved configuration: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )
    return engine.to(device).eval()


def load_gate(path: Path = GATE_PATH) -> AdmissibilityGate:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — fit it with `python -m training.evidence.gate_predictor --fit`")
    # weights_only=True: the artifact is only tensors, scalars and a string-keyed provenance dict,
    # so there is no reason to permit arbitrary unpickling here (unlike the Phase-A checkpoints,
    # which carry richer objects).
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("artifact_kind") != "halo_admissibility_gate" or \
            blob.get("artifact_version") != GATE_ARTIFACT_VERSION:
        raise ValueError(
            f"{path} is a legacy admissibility artifact. Refit it from the current train-only "
            "per-sensor resolvability table."
        )
    gate = AdmissibilityGate(rank=int(blob["rank"]), text_dim=int(blob["text_dim"]))
    # `known_sensors` is a buffer whose width varies with the fitting corpus, so it must be resized
    # before load_state_dict rather than left at its zero-row initialisation.
    for name in ("known_sensors", "known_concepts"):
        known = blob["state_dict"].get(name)
        if known is not None:
            setattr(gate, name, torch.zeros_like(known))
    gate.load_state_dict(blob["state_dict"])
    gate.eval()
    return gate


# ------------------------------------------------------------------------------------ fitting
def fit_from_table(
    rank: int = DEFAULT_RANK,
    mode: str = "full",
    table: dict | None = None,
    text_views: int = 4,
) -> tuple:
    """Warm-start a gate on the measured resolvability table. Returns ``(gate, provenance)``."""
    from training.evidence.gate_extrapolation import _embed
    from training.evidence.policy import PHASE_B_DEV_DATASETS, PHASE_B_TEST_DATASETS
    from training.evidence.resolvability import load as load_table

    table = load_table() if table is None else table
    if table.get("schema_version") != 2 or table.get("scope") != \
            "phase_a_training_subjects_only_per_sensor":
        raise ValueError("gate fitting requires the current train-only per-sensor table")
    roster = set(table.get("training_datasets", ()))
    measured = {value.get("dataset") for value in table.get("per_sensor", {}).values()}
    unexpected = sorted(value for value in measured - roster if value is not None)
    if None in measured or unexpected:
        raise ValueError(
            f"resolvability table contains sensor rows outside its recorded roster: {unexpected}"
        )
    leaked = sorted(
        set(table.get("training_datasets", ()))
        & set(PHASE_B_DEV_DATASETS + PHASE_B_TEST_DATASETS)
    )
    if leaked:
        raise ValueError(
            f"resolvability table includes Phase-B development/test datasets: {leaked}"
        )
    obs = observations_from_table(table)
    if len(obs) < 50:
        raise ValueError(f"only {len(obs)} usable cells — too few to warm-start a gate")
    from training.evidence.admissibility_text import (
        embed_text_views, flatten_training_views, observation_text_views,
    )

    views = observation_text_views(obs, count=text_views, seed=20260812)
    sensor_views, concept_views = embed_text_views(views, _embed)
    sensor_fit, concept_fit, target_fit = flatten_training_views(
        sensor_views, concept_views, obs.value
    )
    gate = fit_gate(
        sensor_fit, concept_fit, target_fit, rank=rank, mode=mode,
    )
    provenance = {
        "n_cells": len(obs), "n_text_views_per_cell": text_views,
        "n_fit_rows": len(target_fit), "rank": rank, "mode": mode,
        "n_sensors": len(set(obs.stream_key)), "n_concepts": len(set(obs.concept)),
        "resolvability_checkpoint": table.get("checkpoint"),
        "resolvability_checkpoint_sha256": table.get("checkpoint_sha256"),
        "resolvability_checkpoint_step": table.get("checkpoint_step"),
        "resolvability_table_sha256": hashlib.sha256(
            json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "training_datasets": list(table.get("training_datasets", ())),
        "training_concepts": sorted(set(obs.concept)),
        "validation_subjects_sha256": table.get("validation_subjects_sha256"),
        "fit_seed": 20260812,
        "fit_steps": 600,
        "fit_lr": 0.05,
        "fit_weight_decay": 1e-3,
    }
    return gate, provenance


# ------------------------------------------------------------------------------ bank adapter
@dataclass(frozen=True)
class BankRows:
    """``SensorRows`` plus the descriptor de-duplication the gate is evaluated against."""

    rows: SensorRows
    unique_descriptor: torch.Tensor    # (U, 384)
    descriptor_id: torch.Tensor        # (R,)

    def to(self, device) -> "BankRows":
        return BankRows(
            rows=self.rows.to(device),
            unique_descriptor=self.unique_descriptor.to(device),
            descriptor_id=self.descriptor_id.to(device),
        )


@dataclass(frozen=True)
class SensorBankStore:
    """Compact GPU-resident source table used by Stage-2 episode assembly.

    The full bank keeps FP16 patch features and one descriptor id per row. Selected episode rows are
    promoted to FP32 only after gathering, avoiding thousands of tiny CPU-to-GPU transfers without
    changing any retrieval arithmetic.
    """

    feature: torch.Tensor
    bias: torch.Tensor
    modality: torch.Tensor
    gravity: torch.Tensor
    label: torch.Tensor
    dataset: torch.Tensor
    unique_descriptor: torch.Tensor
    descriptor_id: torch.Tensor
    source_window: torch.Tensor | None = None   # which recording each row came from

    @classmethod
    def from_bank(cls, bank: dict, device: torch.device | str) -> "SensorBankStore":
        if bank.get("schema_version", 0) < 5 or "sensor" not in bank:
            raise ValueError("a schema-5 per-sensor bank is required")
        table = bank["sensor"]
        cfg_names = bank["cfg_names"]
        metadata = bank.get("sensor_descriptions") or {}
        cfg_count = len(cfg_names)
        records: list[tuple[int, int, str]] = []
        for cfg_id in range(cfg_count):
            rows = metadata.get(cfg_id, metadata.get(str(cfg_id)))
            if not rows:
                raise ValueError(f"bank configuration {cfg_id} has no stored sensor descriptors")
            records.extend(
                (cfg_id, int(row["slot"]), str(row["text"])) for row in rows
            )
        unique_text = list(dict.fromkeys(text for _, _, text in records))
        unique_descriptor = _sensor_descriptor_embeddings(unique_text)
        text_id = {text: index for index, text in enumerate(unique_text)}
        max_slot = max(slot for _, slot, _ in records)
        descriptor_lookup = torch.full((cfg_count, max_slot + 1), -1, dtype=torch.long)
        for cfg_id, slot, text in records:
            descriptor_lookup[cfg_id, slot] = text_id[text]

        cfg = torch.as_tensor(table["cfg"], dtype=torch.long)
        slot = torch.as_tensor(table["slot"], dtype=torch.long)
        if int(slot.max()) > max_slot:
            raise ValueError("bank sensor rows reference a slot without descriptor metadata")
        descriptor_id = descriptor_lookup[cfg, slot]
        if bool(descriptor_id.lt(0).any()):
            raise ValueError("bank sensor rows reference an absent build-time descriptor")

        dataset_id_of: dict[str, int] = {}
        dataset_by_cfg = []
        for cfg_id in range(cfg_count):
            name = str(cfg_names[cfg_id]).partition("/")[0]
            dataset_by_cfg.append(dataset_id_of.setdefault(name, len(dataset_id_of)))
        dataset = torch.tensor(dataset_by_cfg, dtype=torch.long).index_select(0, cfg)

        device = torch.device(device)
        return cls(
            feature=table["Z"].to(device),
            bias=table["bias"].to(device),
            modality=table["modality"].to(device=device, dtype=torch.long),
            gravity=table["gravity"].to(device=device, dtype=torch.long),
            label=table["y"].to(device=device, dtype=torch.long),
            dataset=dataset.to(device),
            unique_descriptor=unique_descriptor.to(device),
            descriptor_id=descriptor_id.to(device),
            source_window=(table["window"].to(device=device, dtype=torch.long)
                           if "window" in table else None),
        )

    def select(
        self,
        row_indices: torch.Tensor | Sequence[int],
        enrolled_candidate: torch.Tensor | None = None,
    ) -> BankRows:
        selected = torch.as_tensor(row_indices, dtype=torch.long, device=self.feature.device)
        if selected.numel() == 0:
            raise ValueError("cannot construct an evidence table from zero sensor rows")
        descriptor_id = self.descriptor_id.index_select(0, selected)
        n_rows = len(selected)
        if enrolled_candidate is None:
            enrolled = torch.full(
                (n_rows,), -1, dtype=torch.long, device=self.feature.device
            )
        else:
            enrolled = torch.as_tensor(
                enrolled_candidate, dtype=torch.long, device=self.feature.device
            )
            if len(enrolled) != n_rows:
                raise ValueError(
                    f"enrolled_candidate has {len(enrolled)} rows, expected {n_rows}"
                )
        rows = SensorRows(
            feature=self.feature.index_select(0, selected).float(),
            descriptor=self.unique_descriptor.index_select(0, descriptor_id),
            bias=self.bias.index_select(0, selected).float(),
            modality=self.modality.index_select(0, selected),
            gravity=self.gravity.index_select(0, selected),
            label=self.label.index_select(0, selected),
            dataset=self.dataset.index_select(0, selected),
            enrolled_candidate=enrolled,
            source_window=(None if self.source_window is None
                           else self.source_window.index_select(0, selected)),
        )
        return BankRows(rows, self.unique_descriptor, descriptor_id)


def subset_bank_rows(bank_rows: BankRows, index: torch.Tensor) -> BankRows:
    """Select rows while retaining the exact descriptor de-duplication table."""
    index = torch.as_tensor(index, device=bank_rows.descriptor_id.device)
    if index.dtype == torch.bool:
        if len(index) != len(bank_rows.rows.feature):
            raise ValueError("bank-row mask length does not match the evidence table")
    else:
        index = index.long()
    return BankRows(
        rows=SensorRows(**{
            name: None if getattr(bank_rows.rows, name) is None
            else getattr(bank_rows.rows, name)[index]
            for name in bank_rows.rows.__dataclass_fields__
        }),
        unique_descriptor=bank_rows.unique_descriptor,
        descriptor_id=bank_rows.descriptor_id[index],
    )


def concatenate_bank_rows(left: BankRows, right: BankRows) -> BankRows:
    """Append enrolled rows without recomputing or duplicating text embeddings per evidence row."""
    if left.rows.feature.device != right.rows.feature.device:
        raise ValueError("both evidence tables must be on the same device")
    offset = len(left.unique_descriptor)
    fields = {
            # Provenance survives only if BOTH sides carry it; a half-populated column would let
            # enrolled rows claim co-membership with whichever corpus rows happened to share an id.
            name: None
            if getattr(left.rows, name) is None or getattr(right.rows, name) is None
            else torch.cat((getattr(left.rows, name), getattr(right.rows, name)), dim=0)
            for name in left.rows.__dataclass_fields__
        }
    if left.rows.source_window is not None and right.rows.source_window is not None:
        # Runtime ids are local to their encoded stream. Offset a compact copy so equality continues
        # to mean co-recording after it is appended to the archive's global namespace.
        _, right_local = torch.unique(right.rows.source_window, return_inverse=True)
        fields["source_window"] = torch.cat(
            (left.rows.source_window, right_local + int(left.rows.source_window.max()) + 1), dim=0,
        )
    return BankRows(
        rows=SensorRows(**fields),
        unique_descriptor=torch.cat((left.unique_descriptor, right.unique_descriptor), dim=0),
        descriptor_id=torch.cat((left.descriptor_id, right.descriptor_id + offset), dim=0),
    )


def _stream_sensor_descriptions(
    dataset: str, stream: str, modalities: Sequence[str],
) -> list[str]:
    from training.tokenizer.pretrain_data import stream_sensor_texts

    present = set(modalities)
    _, sensor_texts, _ = stream_sensor_texts(
        dataset,
        stream,
        has_accel="accel" in present,
        has_gyro="gyro" in present,
    )
    return list(sensor_texts)


@lru_cache(maxsize=None)
def _runtime_sensor_metadata(dataset: str, stream: str, channel_mask: tuple[bool, ...]):
    """Cache immutable stream metadata; SBERT text encoding must not run once per query window."""
    from eval.scoring import get_sbert_encoder
    from training.tokenizer.pretrain_data import (
        _stream_gravity_state, modalities_present, stream_sensor_bias, stream_sensor_texts,
    )

    modalities = modalities_present(channel_mask)
    gravity_removed = _stream_gravity_state(dataset, stream) == "removed"
    _, descriptions, _ = stream_sensor_texts(
        dataset, stream, gravity_removed=gravity_removed,
        has_accel="accel" in modalities, has_gyro="gyro" in modalities,
    )
    descriptors = torch.from_numpy(np.asarray(
        get_sbert_encoder()(descriptions), dtype=np.float32
    ))
    bias = stream_sensor_bias(dataset, stream, modalities)
    modality = torch.tensor(
        [0 if value == "accel" else 1 for value in modalities], dtype=torch.long
    )
    gravity = torch.tensor([
        1 if value == "accel" and gravity_removed else 0 for value in modalities
    ], dtype=torch.long)
    return descriptors, bias, modality, gravity


def sensor_rows_from_bank(
    bank: dict,
    enrolled_candidate: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
) -> BankRows:
    """Schema-5 bank -> ``SensorRows`` with a de-duplicated descriptor table."""
    if bank.get("schema_version", 0) < 5 or "sensor" not in bank:
        raise ValueError(
            "the admissibility predictor needs a schema-5 bank with per-sensor rows; build it with "
            "`build_memory --sensor-rows` against a sensor-granularity checkpoint. A schema-3 bank "
            "holds pooled session vectors, for which admissibility is undefined."
        )
    table = bank["sensor"]
    selected = (
        torch.arange(len(table["Z"]), dtype=torch.long)
        if row_indices is None else torch.as_tensor(row_indices, dtype=torch.long)
    )
    if selected.numel() == 0:
        raise ValueError("cannot construct an evidence table from zero sensor rows")
    cfg_names = bank["cfg_names"]
    cfg = table["cfg"].index_select(0, selected).to(torch.long)
    slot = table["slot"].index_select(0, selected).to(torch.long)
    metadata = bank.get("sensor_descriptions") or {}
    # One descriptor per (stream, slot), read from the immutable build-time map.
    pairs = sorted({(int(c), int(s)) for c, s in zip(cfg.tolist(), slot.tolist())})
    texts, pair_index = [], {}
    for c, s in pairs:
        key = cfg_names[c] if not isinstance(cfg_names, dict) else cfg_names[c]
        rows_for_config = metadata.get(c, metadata.get(str(c)))
        by_slot = {} if rows_for_config is None else {
            int(row["slot"]): row for row in rows_for_config
        }
        if s not in by_slot:
            raise ValueError(
                f"bank row references sensor slot {s} of {key}, but its stored descriptor is absent"
            )
        pair_index[(c, s)] = len(texts)
        texts.append(str(by_slot[s]["text"]))

    uniq_text = list(dict.fromkeys(texts))
    uniq_emb = _sensor_descriptor_embeddings(uniq_text)
    text_to_uniq = {t: i for i, t in enumerate(uniq_text)}
    pair_to_uniq = torch.tensor([text_to_uniq[texts[pair_index[p]]] for p in pairs],
                                dtype=torch.long)
    pair_lookup = {p: i for i, p in enumerate(pairs)}
    row_pair = torch.tensor([pair_lookup[(int(c), int(s))]
                             for c, s in zip(cfg.tolist(), slot.tolist())], dtype=torch.long)
    descriptor_id = pair_to_uniq[row_pair]

    gravity = table["gravity"].index_select(0, selected).to(torch.long)
    dataset_id_of: dict[str, int] = {}
    dataset_row = []
    for c in cfg.tolist():
        name = str(cfg_names[int(c)]).partition("/")[0]
        dataset_row.append(dataset_id_of.setdefault(name, len(dataset_id_of)))

    n_rows = int(len(selected))
    if enrolled_candidate is not None and len(enrolled_candidate) != n_rows:
        raise ValueError(
            f"enrolled_candidate has {len(enrolled_candidate)} rows, expected {n_rows}"
        )
    rows = SensorRows(
        feature=table["Z"].index_select(0, selected).float(),
        descriptor=uniq_emb.index_select(0, descriptor_id),
        bias=table["bias"].index_select(0, selected).float(),
        modality=table["modality"].index_select(0, selected).to(torch.long),
        gravity=gravity,
        label=table["y"].index_select(0, selected).to(torch.long),
        dataset=torch.tensor(dataset_row, dtype=torch.long),
        enrolled_candidate=(torch.full((n_rows,), -1, dtype=torch.long)
                            if enrolled_candidate is None else enrolled_candidate.to(torch.long)),
        # Which recording each row came from. The bank already tracks this per sensor row; it was
        # simply never surfaced, which is why the evidence mixer had no co-membership channel to
        # read at deployment.
        source_window=(table["window"].index_select(0, selected).to(torch.long)
                       if "window" in table else None),
    )
    return BankRows(rows=rows, unique_descriptor=uniq_emb, descriptor_id=descriptor_id)


def sensor_rows_from_encoded(
    encoded: dict,
    window_rows: torch.Tensor,
    dataset: str,
    stream: str,
    *,
    channel_mask: Sequence[bool],
    candidate_positions: torch.Tensor | None = None,
    sensor_row_indices: torch.Tensor | None = None,
) -> BankRows:
    """Turn selected external windows into per-sensor query or enrolled-evidence rows.

    ``candidate_positions`` binds each selected window to an episode candidate. Omit it for query
    rows. The signal vectors, slots and patch timing come directly from the same detailed export used
    by schema-5 bank construction; only query-side deterministic stream metadata is constructed here.
    """
    source_windows = torch.as_tensor(window_rows, dtype=torch.long).cpu()
    if source_windows.numel() == 0:
        raise ValueError("cannot construct runtime sensor rows from zero windows")
    if candidate_positions is not None and len(candidate_positions) != len(source_windows):
        raise ValueError("candidate_positions must have one entry per selected source window")
    all_source_rows = encoded["sensor_window"].long().cpu()
    if sensor_row_indices is None:
        selected = torch.isin(all_source_rows, source_windows)
        selected_index = selected.nonzero(as_tuple=False).flatten()
    else:
        selected_index = torch.as_tensor(sensor_row_indices, dtype=torch.long).cpu()
        if len(selected_index) and (
            int(selected_index.min()) < 0 or int(selected_index.max()) >= len(all_source_rows)
        ):
            raise IndexError("sensor_row_indices are outside the encoded sensor-row table")
    if not len(selected_index):
        raise ValueError("selected windows exported no per-sensor rows")
    source_row = all_source_rows.index_select(0, selected_index)
    if not bool(torch.isin(source_row, source_windows).all()):
        raise ValueError("sensor_row_indices include a window outside window_rows")
    slot = encoded["sensor_slot"].long().cpu().index_select(0, selected_index)

    descriptors, bias_by_slot, modality_by_slot, gravity_by_slot = _runtime_sensor_metadata(
        dataset, stream, tuple(bool(value) for value in channel_mask)
    )
    if int(slot.max()) >= len(descriptors):
        raise ValueError(
            f"sensor slot {int(slot.max())} exceeds {len(descriptors)} descriptions for "
            f"{dataset}/{stream}"
        )

    enrolled = torch.full((len(source_row),), -1, dtype=torch.long)
    if candidate_positions is not None:
        candidate_positions = torch.as_tensor(candidate_positions, dtype=torch.long).cpu()
        position_by_window = {
            int(window): int(position)
            for window, position in zip(source_windows.tolist(), candidate_positions.tolist())
        }
        enrolled = torch.tensor(
            [position_by_window[int(window)] for window in source_row.tolist()], dtype=torch.long
        )
    return BankRows(
        rows=SensorRows(
            feature=encoded["sensor_Z"].index_select(
                0, selected_index.to(encoded["sensor_Z"].device)
            ).float().cpu(),
            descriptor=descriptors.index_select(0, slot),
            bias=bias_by_slot.index_select(0, slot),
            modality=modality_by_slot.index_select(0, slot),
            gravity=gravity_by_slot.index_select(0, slot),
            label=torch.full((len(source_row),), -1, dtype=torch.long),
            dataset=torch.full((len(source_row),), -1, dtype=torch.long),
            enrolled_candidate=enrolled,
            source_window=source_row,
        ),
        unique_descriptor=descriptors,
        descriptor_id=slot,
    )


# ---------------------------------------------------------------------------------- prediction
def predict_bank_grouped(
    gate: AdmissibilityGate,
    bank_rows: BankRows,
    query_feature: torch.Tensor,      # (S, d)
    query_bias: torch.Tensor,         # (S, F)
    query_descriptor: torch.Tensor,   # (S, 384)
    query_modality: torch.Tensor,     # (S,)
    query_gravity: torch.Tensor,      # (S,)
    candidate_text: torch.Tensor,     # (C, 384)
    label_text: torch.Tensor,         # (V, 384)
    top_k: int = 64,
    bias_weight: float = BIAS_BLEND_WEIGHT,
    temperature: float = 0.07,
    sensor_weight: torch.Tensor | None = None,
    semantic_labels: bool = True,
    query_group: torch.Tensor | None = None,
    group_count: int | None = None,
    engine=None,                      # EvidenceEngine; None keeps the closed-form rule
    engine_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """(W,C) logits for grouped query sensors.

    With ``engine=None`` this is the closed-form rule and there is no learned component anywhere.
    With an engine it is the deployed form of the trained model — the same forward pass it was
    trained with. ``identity_logits`` stays closed-form either way, because it is the control the
    learned path is measured against.
    """
    rows = bank_rows.rows
    if query_group is None:
        query_group = torch.zeros(
            len(query_feature), dtype=torch.long, device=query_feature.device,
        )
        group_count = 1
    else:
        query_group = torch.as_tensor(
            query_group, dtype=torch.long, device=query_feature.device,
        )
        if len(query_group) != len(query_feature):
            raise ValueError("query_group must have one entry per query sensor")
        group_count = int(query_group.max()) + 1 if group_count is None else int(group_count)
        if group_count < 1 or bool(query_group.lt(0).any()) or \
                bool(query_group.ge(group_count).any()):
            raise ValueError("query_group indices must be within group_count")

    def grouped_sum(per_sensor: torch.Tensor) -> torch.Tensor:
        if sensor_weight is not None:
            per_sensor = per_sensor * sensor_weight.unsqueeze(-1)
        result = per_sensor.new_zeros((group_count, per_sensor.shape[-1]))
        return result.index_add(0, query_group, per_sensor)

    adm = admissibility_from_unique(gate, bank_rows.unique_descriptor, bank_rows.descriptor_id,
                                    candidate_text, query_descriptor,
                                    semantic_labels=semantic_labels)
    compatible = compatibility_mask(query_modality, query_gravity, rows)
    scores = rank_scores(query_feature, query_bias, rows, compatible, bias_weight=bias_weight)
    if engine is None:
        per_sensor = vote(scores, rows, candidate_text, label_text, adm,
                          top_k=top_k, temperature=temperature,
                          allow_corpus_text_vote=semantic_labels)
    else:
        query_rows = SensorRows(
            feature=query_feature, descriptor=query_descriptor, bias=query_bias,
            modality=query_modality, gravity=query_gravity,
            label=torch.full_like(query_modality, -1),
            dataset=torch.zeros_like(query_modality),
            enrolled_candidate=torch.full_like(query_modality, -1),
            source_window=query_group,
        )
        per_sensor = engine_vote(
            engine, query_rows, rows, candidate_text, label_text,
            top_k=top_k, generator=engine_generator,
        )
    logits = per_sensor if engine is not None else grouped_sum(per_sensor)
    # Arbitrary aliases intentionally disable semantic admissibility, so the learned and disabled
    # paths are identical by definition. Reusing the result avoids a redundant full retrieval pass
    # and makes that protocol invariant exact rather than merely expected.
    identity_per_sensor = vote(
        scores, rows, candidate_text, label_text, torch.ones_like(adm),
        top_k=top_k, temperature=temperature,
        allow_corpus_text_vote=semantic_labels,
    )
    aux = {
        "identity_logits": grouped_sum(identity_per_sensor),
        # Required-answer evaluation resolves a tie by candidate order. Expose the no-evidence
        # state so it cannot be mistaken for an evidence-supported prediction.
        "evidence/all_candidates_zero": float(
            logits.detach().eq(0).all(dim=1).float().mean()
        ),
        "evidence/zero_candidate_fraction": float(
            logits.detach().eq(0).float().mean()
        ),
        **AdmissibilityGate.spread(adm),
    }
    return logits, aux


def predict_bank(
    gate: AdmissibilityGate,
    bank_rows: BankRows,
    query_feature: torch.Tensor,
    query_bias: torch.Tensor,
    query_descriptor: torch.Tensor,
    query_modality: torch.Tensor,
    query_gravity: torch.Tensor,
    candidate_text: torch.Tensor,
    label_text: torch.Tensor,
    top_k: int = 64,
    bias_weight: float = BIAS_BLEND_WEIGHT,
    temperature: float = 0.07,
    sensor_weight: torch.Tensor | None = None,
    semantic_labels: bool = True,
) -> tuple[torch.Tensor, dict]:
    """(C,) logits plus telemetry for one query window."""
    logits, aux = predict_bank_grouped(
        gate, bank_rows,
        query_feature=query_feature,
        query_bias=query_bias,
        query_descriptor=query_descriptor,
        query_modality=query_modality,
        query_gravity=query_gravity,
        candidate_text=candidate_text,
        label_text=label_text,
        top_k=top_k,
        bias_weight=bias_weight,
        temperature=temperature,
        sensor_weight=sensor_weight,
        semantic_labels=semantic_labels,
    )
    aux["identity_logits"] = aux["identity_logits"][0]
    return logits[0], aux


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true", help="warm-start a gate on the measured table")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--mode", default="full", choices=("full", "pca"))
    ap.add_argument(
        "--text-views", type=int, default=4,
        help="equal-weight canonical/paraphrased views per measured physical cell",
    )
    ap.add_argument("--out", type=Path, default=GATE_PATH)
    ap.add_argument(
        "--bank", type=Path, default=None,
        help="schema-5 bank to bind into an evaluation-ready predictor artifact",
    )
    args = ap.parse_args()
    if not args.fit:
        ap.error("pass --fit")

    gate, provenance = fit_from_table(
        rank=args.rank, mode=args.mode, text_views=args.text_views,
    )
    bank = None
    if args.bank is not None:
        from training.evidence.bank_guard import assert_bank_current, assert_sensor_bank

        bank = torch.load(args.bank, map_location="cpu", weights_only=True)
        assert_bank_current(bank, context="gate_predictor")
        assert_sensor_bank(bank, context="gate_predictor")
    save_gate(gate, args.out, provenance, bank=bank)
    print(json.dumps(provenance, indent=2))
    print(f"\nneutral (unfamiliar-configuration fallback): {float(gate.neutral):.4f}")
    print(f"known sensor descriptions: {int(gate.known_sensors.shape[0])}")
    print(f"-> {args.out}")
    if bank is None:
        print("  diagnostic gate only: pass --bank to create an evaluation-ready bound artifact")
    print("\n  Held-out generalisation for this gate is measured by "
          "`python -m training.evidence.gate_extrapolation --split placement`.")
    print("  Read `skill_vs_concept_mean`, not `skill`: the latter is beaten by knowing which")
    print("  ACTIVITIES are easy, which requires no notion of configuration at all.")


if __name__ == "__main__":
    main()
