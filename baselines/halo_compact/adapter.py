"""HALO compact evidence engine — the CURRENT model, scored on the shared adaptation manifest.

Two faces, both from one checkpoint (default: the 2026-08-22 long-run best):

  * ``window_features`` — pooled frozen retrieval rows, consumed by the standard adaptation
    methods (nearest / prototype / ridge / linear_head). This row is apples-to-apples with the
    baselines' frozen-feature rows in ``eval/adaptation_results/v1_d85761d``.
  * native zero-shot and enrollment — one pooled row per six-second recording, raw cosine against
    every corpus/enrollment row, a bounded learned correction for each pair, then corrected 1NN.
    No head is fit at evaluation time; this is the mechanism the model ships with.

The engine, encoder, bank construction and window aggregation are the training-time code paths
(`load_evidence_engine`, the encoder's exact pooled output, and `SensorRows`), so the number scored
here is the deployed function rather than a re-implementation.

Checkpoint: env ``HALO_COMPACT_CKPT`` (default ``training/tokenizer/outputs/
e2e_compact_35k_20260823/best.pt``). Bank cache: ``baselines/halo_compact/bank_<fp>.pt`` (gitignored),
keyed by a content hash of the checkpoint, the bank seed and size.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaselineAdapter, InputContract, register
from eval import data as eval_data

_REPO = Path(__file__).resolve().parents[2]
_CKPT = Path(os.environ.get(
    "HALO_COMPACT_CKPT", _REPO / "training/tokenizer/outputs/e2e_compact_35k_20260823/best.pt"))
_HERE = Path(__file__).resolve().parent


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be a boolean flag, got {value!r}")
    return value in {"1", "true", "yes"}


_NEUTRAL_ACQUISITION_TEXT = _env_flag("HALO_NEUTRAL_ACQUISITION_TEXT")

BANK_WINDOWS = 512        # matches the training BankSpec deployment contract
BANK_SEED = 20260822
QUERY_CHUNK = 1024        # recording queries per forward (memory bank held once)


def _ckpt_fp() -> str:
    return _ckpt_sha256()[:16]


def _ckpt_sha256() -> str:
    return hashlib.sha256(_CKPT.read_bytes()).hexdigest() if _CKPT.exists() else ""


@register
class HALOCompactAdapter(BaselineAdapter):
    """Compact recording-level residual reranker, native input contract."""

    name = "halo_compact"
    tier = "evidence"
    contract = InputContract()          # native rate/channels; the tokenizer is the contract

    # ------------------------------------------------------------------ setup
    def setup(self, device):
        import torch as T
        from training.tokenizer.eval_transfer import build_encoder            # lazy: model pkg
        from training.evidence.gate_predictor import load_evidence_engine
        from eval.scoring import get_sbert_encoder

        if not _CKPT.exists():
            raise FileNotFoundError(f"HALO compact checkpoint missing at {_CKPT}")
        blob = T.load(str(_CKPT), map_location="cpu", weights_only=False)
        device = T.device(device)
        encoder = build_encoder(blob, device, training=False).eval()
        engine = load_evidence_engine(_CKPT, encoder=encoder, device=device)
        if engine is None:
            raise ValueError(f"{_CKPT} carries no evidence engine")
        engine = engine.to(device).eval()

        sbert = get_sbert_encoder()
        label_ids = blob["label_ids"]
        id_to_label = {v: k for k, v in label_ids.items()}
        vocab = [id_to_label[i] for i in range(len(id_to_label))]
        label_text = F.normalize(
            T.from_numpy(sbert(vocab)).float(), dim=-1).to(device)            # (V, 384)

        state = {
            "blob_config": blob["config"], "encoder": encoder, "engine": engine,
            "sbert": sbert, "vocab": vocab, "label_text": label_text,
            "device": device, "streams": {}, "feature_owner": {},
            "checkpoint_step": int(blob.get("step", 0)),
            "checkpoint_selection": float(blob.get("selection_value", float("nan"))),
            "checkpoint_corpus_fingerprint": blob.get("corpus_fingerprint"),
            "checkpoint_neutral_acquisition_text": bool(
                blob.get("config", {}).get("neutral_acquisition_text", False)
            ),
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
        }
        if state["checkpoint_neutral_acquisition_text"] and not _NEUTRAL_ACQUISITION_TEXT:
            raise ValueError(
                "this HALO checkpoint was trained without acquisition descriptions; evaluate it "
                "with HALO_NEUTRAL_ACQUISITION_TEXT=1"
            )
        state["bank"] = self._bank_rows(state)
        print(f"[halo_compact] {_CKPT.name} step {blob.get('step')} "
              f"sel {blob.get('selection_value', float('nan')):.4f} · "
              f"bank {int(state['bank'].feature.shape[0])} rows from {BANK_WINDOWS} windows · "
              f"acquisition_text={'neutral' if _NEUTRAL_ACQUISITION_TEXT else 'full'}",
              flush=True)
        return state

    # ------------------------------------------------------------- corpus bank
    def _bank_rows(self, state):
        """Stratified frozen memory bank with one pooled row per six-second window."""
        import torch as T
        from model.evidence.rows import SensorRows

        from training.tokenizer.pretrain import corpus_fingerprint
        from training.tokenizer.pretrain_data import CorpusIndex

        datasets = tuple(state["blob_config"]["train_datasets"])
        index = CorpusIndex(seed=20260818, datasets=datasets)
        current_corpus = corpus_fingerprint(index)
        if current_corpus != state["checkpoint_corpus_fingerprint"]:
            raise RuntimeError(
                "HALO checkpoint corpus does not match the current training grids: "
                f"checkpoint={state['checkpoint_corpus_fingerprint']} current={current_corpus}. "
                "A memory bank built from changed data would not be attributable to this model."
            )

        text_arm = "neutral" if _NEUTRAL_ACQUISITION_TEXT else "full"
        cache = _HERE / f"bank_{_ckpt_fp()}_{BANK_SEED}_{BANK_WINDOWS}_{text_arm}.pt"
        state["bank_cache"] = cache
        device = state["device"]
        if cache.exists():
            payload = T.load(cache, map_location=device, weights_only=True)
            expected = {
                "schema_version": 4,
                "checkpoint_sha256": _ckpt_sha256(),
                "corpus_fingerprint": current_corpus,
                "seed": BANK_SEED,
                "windows": BANK_WINDOWS,
                "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            }
            if all(payload.get(key) == value for key, value in expected.items()):
                return SensorRows(**payload["rows"])

        from training.tokenizer.pretrain_data import (MultiScaleCollate, PretrainDataset,
                                                      PATCH_SECONDS)
        from training.tokenizer.pretrain_episodic import encode_batch
        from training.tokenizer.episodic import EpisodicCollate, live_recording_rows

        # round-robin over labels, then streams within a label: every label present, no stream
        # dominating — the same stratification intent as the training BankSpec.
        rng = np.random.default_rng(BANK_SEED)
        by_label: dict[int, list[int]] = {}
        for position, key in enumerate(index.train):
            by_label.setdefault(int(key.label_id), []).append(position)
        for pool in by_label.values():
            rng.shuffle(pool)
        picks: list[int] = []
        labels_order = sorted(by_label)
        depth = 0
        while len(picks) < BANK_WINDOWS:
            advanced = False
            for label in labels_order:
                pool = by_label[label]
                if depth < len(pool):
                    picks.append(pool[depth])
                    advanced = True
                    if len(picks) >= BANK_WINDOWS:
                        break
            if not advanced:
                break
            depth += 1

        dataset = PretrainDataset(
            index, list(index.train), augment=False,
            neutral_acquisition_text=_NEUTRAL_ACQUISITION_TEXT,
        )
        collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
        parts = []
        with T.no_grad():
            for start in range(0, len(picks), 256):
                chunk = picks[start:start + 256]
                batch = collate([dataset[p] for p in chunk])
                out = encode_batch(state["encoder"], batch, device)
                labels = batch["labels"].to(device)
                live = live_recording_rows(
                    out, batch, labels=labels,
                    enrolled_candidate=T.full((len(chunk),), -1, dtype=T.long, device=device),
                )
                parts.append(live.rows)
        merged = {
            field: T.cat([getattr(rows, field) for rows in parts]).detach()
            for field in ("feature", "descriptor", "bias", "modality", "gravity",
                          "label", "dataset", "enrolled_candidate")
        }
        # source_window must stay distinct ACROSS chunks — reindex into one global window space.
        offset, source = 0, []
        for rows, chunk_len in zip(parts, [min(256, len(picks) - s)
                                           for s in range(0, len(picks), 256)]):
            source.append(rows.source_window + offset)
            offset += chunk_len
        merged["source_window"] = T.cat(source).detach()
        T.save({
            "schema_version": 4,
            "checkpoint_sha256": _ckpt_sha256(),
            "corpus_fingerprint": current_corpus,
            "seed": BANK_SEED,
            "windows": BANK_WINDOWS,
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            "rows": {k: v.cpu() for k, v in merged.items()},
        }, cache)
        from model.evidence.rows import SensorRows as SR
        return SR(**{k: v.to(device) for k, v in merged.items()})

    # ------------------------------------------------------- eval-stream rows
    def _stream_rows(self, stream, state):
        """One pooled recording row per eval window, plus the exact pooled 1-NN feature matrix."""
        import torch as T
        from model.evidence.rows import SensorRows
        from training.tokenizer.eval_transfer import encode_dataset_detailed
        from training.tokenizer.pretrain_data import stream_sensor_texts

        key = (stream.dataset, stream.stream)
        if key in state["streams"]:
            return state["streams"][key]
        device = state["device"]
        encoder = state["encoder"]
        from training.tokenizer.pretrain_data import (_stream_gravity_state,
                                                      stream_channel_descriptions)
        texts = (
            stream_channel_descriptions(stream.dataset, stream.stream, neutral=True)
            if _NEUTRAL_ACQUISITION_TEXT else
            (stream.channel_descriptions
             if getattr(stream, "channel_descriptions", None) is not None
             else stream_channel_descriptions(stream.dataset, stream.stream))
        )
        gravity_state = (stream.gravity_state
                         if getattr(stream, "gravity_state", None) is not None
                         else _stream_gravity_state(stream.dataset, stream.stream))
        detailed = encode_dataset_detailed(
            encoder, stream.windows, texts, device, stream.rate_hz,
            gravity_state=gravity_state, channel_mask=stream.mask,
            dataset=stream.dataset, stream=stream.stream,
            neutral_text=_NEUTRAL_ACQUISITION_TEXT,
            export_sensor_rows=False,
        )
        feature = detailed["pooled"].to(device)                       # (N, d)
        window = T.arange(len(feature), dtype=T.long, device=device)   # (N,)

        _, sensor_texts, sensor_id_list = stream_sensor_texts(
            stream.dataset, stream.stream,
            gravity_removed=(gravity_state == "removed"),
            has_accel=any(c.startswith("acc") for c, m in zip(stream.channels, stream.mask) if m),
            has_gyro=any(c.startswith("gyro") for c, m in zip(stream.channels, stream.mask) if m),
            neutral=_NEUTRAL_ACQUISITION_TEXT,
        )
        descriptors = encoder.text_encoder.encode_pooled(sensor_texts, device=device)
        descriptors = F.normalize(descriptors.float(), dim=-1)       # (S, 384)
        del sensor_id_list
        recording_descriptor = F.normalize(descriptors.mean(dim=0, keepdim=True), dim=-1)
        recording_descriptor = recording_descriptor.expand(len(feature), -1)
        zeros = T.zeros(len(feature), dtype=T.long, device=device)

        rows = SensorRows(
            feature=feature,
            descriptor=recording_descriptor,
            bias=feature.new_zeros((len(feature), 1)),
            modality=zeros,
            gravity=zeros,
            label=T.full((len(feature),), -1, dtype=T.long, device=device),
            dataset=zeros,
            enrolled_candidate=T.full((len(feature),), -1, dtype=T.long, device=device),
            source_window=window,
        )
        state["streams"][key] = (rows, window, int(stream.n_windows),
                                 detailed["pooled"].float().cpu().numpy())
        return state["streams"][key]

    # ------------------------------------------------ standard adaptation face
    def window_features(self, stream, state, device) -> np.ndarray:
        _, _, _, pooled = self._stream_rows(stream, state)
        state["feature_owner"][id(pooled)] = (stream.dataset, stream.stream)
        return pooled

    def supports_native_enrollment(self) -> bool:
        return True

    def evaluation_source_paths(self):
        return (
            _REPO / "model" / "evidence",
            _REPO / "model" / "tokenizer",
            _REPO / "training" / "tokenizer" / "episodic.py",
            _REPO / "training" / "tokenizer" / "pretrain_episodic.py",
        )

    @staticmethod
    def _take_rows(rows, indices):
        from model.evidence.rows import SensorRows
        return SensorRows(**{
            field: (None if getattr(rows, field) is None else getattr(rows, field)[indices])
            for field in rows.__dataclass_fields__
        })

    @staticmethod
    def _append_enrollment(base, support, support_window, plan, support_count):
        """Append one pooled memory row for every selected enrolled window."""
        import torch as T
        from model.evidence.rows import SensorRows

        pieces = []
        bound_parts = []
        support_windows = 0
        for candidate, executions in enumerate(plan["support_execution_rows"]):
            selected_windows = [
                int(window)
                for execution in executions[:support_count]
                for window in execution
            ]
            support_windows += len(selected_windows)
            wanted = T.as_tensor(selected_windows, dtype=T.long, device=support_window.device)
            row_index = T.nonzero(T.isin(support_window, wanted), as_tuple=True)[0]
            if not len(row_index):
                raise ValueError(f"candidate {candidate} enrollment selected no recording rows")
            pieces.append(HALOCompactAdapter._take_rows(support, row_index))
            bound_parts.append(T.full(
                (len(row_index),), candidate, dtype=T.long, device=support.feature.device,
            ))

        fields = {}
        enrollment_offset = int(base.source_window.max().item()) + 1
        for field in base.__dataclass_fields__:
            base_value = getattr(base, field)
            if field == "enrolled_candidate":
                enrolled_value = T.cat(bound_parts)
            elif field == "label":
                enrolled_value = T.full(
                    (sum(len(piece.feature) for piece in pieces),), -1,
                    dtype=base.label.dtype, device=base.label.device,
                )
            elif field == "source_window":
                raw = T.cat([piece.source_window for piece in pieces])
                unique, inverse = T.unique(raw, sorted=True, return_inverse=True)
                enrolled_value = inverse + enrollment_offset
                del unique
            else:
                enrolled_value = T.cat([getattr(piece, field) for piece in pieces])
            fields[field] = T.cat((base_value, enrolled_value))
        return SensorRows(**fields), support_windows, sum(len(piece.feature) for piece in pieces)

    def predict_enrollment(
        self, query_stream, support_stream, plan, support_count, candidate_texts,
        state, device, *, seed,
    ):
        import torch as T

        query_rows, query_window, _, _ = self._stream_rows(query_stream, state)
        support_rows, support_window, _, _ = self._stream_rows(support_stream, state)
        memory, support_windows, enrolled_rows = self._append_enrollment(
            state["bank"], support_rows, support_window, plan, support_count,
        )
        canonical_candidates = list(plan["candidate_names"])
        if len(candidate_texts) != len(canonical_candidates):
            raise ValueError("candidate text roster must align with the manifest candidate roster")
        candidate_text = F.normalize(
            T.from_numpy(state["sbert"]([text.replace("_", " ") for text in candidate_texts])).float(),
            dim=-1,
        ).to(state["device"])

        requested = [int(row) for row in plan["query_rows"]]
        prediction_by_window = {}
        generator = T.Generator().manual_seed(int(seed))
        with T.no_grad():
            for start in range(0, len(requested), QUERY_CHUNK):
                chunk_windows = requested[start:start + QUERY_CHUNK]
                wanted = T.as_tensor(chunk_windows, dtype=T.long, device=query_window.device)
                row_index = T.nonzero(T.isin(query_window, wanted), as_tuple=True)[0]
                if not len(row_index):
                    raise ValueError("native enrollment query selected no recording rows")
                chunk = self._take_rows(query_rows, row_index)
                result = state["engine"](
                    chunk, memory, candidate_text, state["label_text"], generator=generator,
                )
                positions = result["logits"].argmax(1).cpu().tolist()
                for window_id, position in zip(
                    result["query_window"].cpu().tolist(), positions, strict=True,
                ):
                    prediction_by_window[int(window_id)] = canonical_candidates[int(position)]
        missing = [window for window in requested if window not in prediction_by_window]
        if missing:
            raise ValueError(f"native enrollment omitted {len(missing)} query windows")
        return [prediction_by_window[window] for window in requested], {
            "mechanism": "engine_native_enrollment",
            "corpus_rows": int(len(state["bank"].feature)),
            "enrolled_executions": int(len(canonical_candidates) * support_count),
            "enrolled_windows": int(support_windows),
            "enrolled_recording_rows": int(enrolled_rows),
            "total_memory_rows": int(len(memory.feature)),
        }

    # ---------------------------------------------------- native zero-shot face
    def predict_candidates_from_features(self, features, candidates, state, device):
        import torch as T
        owner = state["feature_owner"].get(id(features))
        if owner is None:
            raise ValueError(
                "halo_compact scores zero-shot with its native engine over recording rows; the "
                "features passed here were not produced by this adapter's window_features")
        rows, window, n_windows, _ = state["streams"][owner]
        device = state["device"]
        engine, bank = state["engine"], state["bank"]
        candidates = list(candidates)
        candidate_text = F.normalize(
            T.from_numpy(state["sbert"]([c.replace("_", " ") for c in candidates])).float(),
            dim=-1).to(device)

        mass = T.zeros((n_windows, len(candidates)), device=device)
        generator = T.Generator().manual_seed(BANK_SEED)
        from model.evidence.rows import SensorRows
        with T.no_grad():
            # Query rows are complete six-second recordings; chunking only partitions queries.
            unique_windows = T.unique(window, sorted=True)
            for start in range(0, len(unique_windows), QUERY_CHUNK):
                chosen_windows = unique_windows[start:start + QUERY_CHUNK]
                row_index = T.nonzero(T.isin(window, chosen_windows), as_tuple=True)[0]
                chunk = SensorRows(**{
                    f: (getattr(rows, f)[row_index] if getattr(rows, f) is not None else None)
                    for f in ("feature", "descriptor", "bias", "modality", "gravity",
                              "label", "dataset", "enrolled_candidate", "source_window")
                })
                result = engine(chunk, bank, candidate_text, state["label_text"],
                                generator=generator)
                mass.index_copy_(0, result["query_window"], result["logits"].float())
        predictions = [candidates[i] for i in mass.argmax(dim=1).cpu().tolist()]
        return predictions, {"predicted_classes": sorted(set(predictions)),
                             "mechanism": "engine_native_zero_shot",
                             "bank_rows": int(bank.feature.shape[0])}

    def evaluation_artifacts(self, state):
        artifacts = {"checkpoint": _CKPT}
        if state.get("bank_cache") is not None:
            artifacts["corpus_bank_cache"] = state["bank_cache"]
        return artifacts

    def evaluation_config(self, state):
        return {
            "checkpoint_step": state["checkpoint_step"],
            "checkpoint_selection": state["checkpoint_selection"],
            "checkpoint_corpus_fingerprint": state["checkpoint_corpus_fingerprint"],
            "bank_windows": BANK_WINDOWS,
            "bank_seed": BANK_SEED,
            "native_positive_k": True,
            "neutral_acquisition_text": state["neutral_acquisition_text"],
            "checkpoint_neutral_acquisition_text": state[
                "checkpoint_neutral_acquisition_text"
            ],
        }

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        features = self.window_features(stream, state, device)
        return self.predict_candidates_from_features(
            features, stream.eval_labels, state, device)
