"""HALO compact evidence engine — the CURRENT model, scored on the shared adaptation manifest.

Two faces, both from one checkpoint (default: the 2026-08-22 long-run best):

  * ``window_features`` — pooled frozen retrieval rows, consumed by the standard adaptation
    methods (nearest / prototype / ridge / linear_head). This row is apples-to-apples with the
    baselines' frozen-feature rows in ``eval/adaptation_results/v1_d85761d``.
  * zero-shot — the engine's NATIVE deployment rule: per-(patch,sensor) rows -> cosine top-64
    over a frozen stratified training-corpus bank -> evidence mixer -> text vote over the
    candidate labels. No head is fit; this is the mechanism the model ships with.

The engine, encoder, bank construction and window aggregation are the training-time code paths
(`load_evidence_engine`, `encode_dataset_detailed(export_sensor_rows=True)`, `SensorRows`,
row-mass summation identical to ``run_episode``), so the number scored here is the deployed
function, not a re-implementation.

Checkpoint: env ``HALO_COMPACT_CKPT`` (default ``training/tokenizer/outputs/
long_4h_20260821/best.pt``). Bank cache: ``baselines/halo_compact/bank_<fp>.pt`` (gitignored),
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
    "HALO_COMPACT_CKPT", _REPO / "training/tokenizer/outputs/long_4h_20260821/best.pt"))
_HERE = Path(__file__).resolve().parent

BANK_WINDOWS = 512        # matches the training BankSpec deployment contract
BANK_SEED = 20260822
QUERY_CHUNK = 4096        # engine query rows per forward (memory bank held once)


def _ckpt_fp() -> str:
    return hashlib.sha256(_CKPT.read_bytes()).hexdigest()[:16] if _CKPT.exists() else ""


@register
class HALOCompactAdapter(BaselineAdapter):
    """Compact evidence engine (retrieve -> mix -> vote), native input contract."""

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
        }
        state["bank"] = self._bank_rows(state)
        print(f"[halo_compact] {_CKPT.name} step {blob.get('step')} "
              f"sel {blob.get('selection_value', float('nan')):.4f} · "
              f"bank {int(state['bank'].feature.shape[0])} rows from {BANK_WINDOWS} windows",
              flush=True)
        return state

    # ------------------------------------------------------------- corpus bank
    def _bank_rows(self, state):
        """Stratified frozen memory bank: BANK_WINDOWS training-corpus windows -> SensorRows."""
        import torch as T
        from model.evidence.rows import SensorRows

        cache = _HERE / f"bank_{_ckpt_fp()}_{BANK_SEED}_{BANK_WINDOWS}.pt"
        device = state["device"]
        if cache.exists():
            payload = T.load(cache, map_location=device, weights_only=True)
            return SensorRows(**payload)

        from training.tokenizer.pretrain_data import (CorpusIndex, MultiScaleCollate,
                                                      PretrainDataset, PATCH_SECONDS)
        from training.tokenizer.pretrain_episodic import encode_batch
        from training.tokenizer.episodic import EpisodicCollate, live_sensor_rows

        datasets = tuple(state["blob_config"]["train_datasets"])
        index = CorpusIndex(seed=20260818, datasets=datasets)
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

        dataset = PretrainDataset(index, list(index.train), augment=False)
        collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
        parts = []
        with T.no_grad():
            for start in range(0, len(picks), 256):
                chunk = picks[start:start + 256]
                batch = collate([dataset[p] for p in chunk])
                out = encode_batch(state["encoder"], batch, device)
                labels = batch["labels"].to(device)
                live = live_sensor_rows(
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
        T.save({k: v.cpu() for k, v in merged.items()}, cache)
        from model.evidence.rows import SensorRows as SR
        return SR(**{k: v.to(device) for k, v in merged.items()})

    # ------------------------------------------------------- eval-stream rows
    def _stream_rows(self, stream, state):
        """Per-(patch,sensor) SensorRows for one eval stream + row->window map."""
        import torch as T
        from model.evidence.rows import SensorRows
        from training.tokenizer.eval_transfer import encode_dataset_detailed
        from training.tokenizer.pretrain_data import stream_sensor_texts
        from training.tokenizer.episodic import sensor_modality_codes, gravity_codes

        key = (stream.dataset, stream.stream)
        if key in state["streams"]:
            return state["streams"][key]
        device = state["device"]
        encoder = state["encoder"]
        from training.tokenizer.pretrain_data import (_stream_gravity_state,
                                                      stream_channel_descriptions)
        texts = (stream.channel_descriptions
                 if getattr(stream, "channel_descriptions", None) is not None
                 else stream_channel_descriptions(stream.dataset, stream.stream))
        gravity_state = (stream.gravity_state
                         if getattr(stream, "gravity_state", None) is not None
                         else _stream_gravity_state(stream.dataset, stream.stream))
        detailed = encode_dataset_detailed(
            encoder, stream.windows, texts, device, stream.rate_hz,
            gravity_state=gravity_state, channel_mask=stream.mask,
            dataset=stream.dataset, stream=stream.stream,
            export_sensor_rows=True,
        )
        feature = detailed["sensor_Z"].to(device)                    # (R, d)
        window = detailed["sensor_window"].to(device).long()         # (R,)
        slot = detailed["sensor_slot"].to(device).long()             # (R,)

        _, sensor_texts, sensor_id_list = stream_sensor_texts(
            stream.dataset, stream.stream,
            gravity_removed=(gravity_state == "removed"),
            has_accel=any(c.startswith("acc") for c, m in zip(stream.channels, stream.mask) if m),
            has_gyro=any(c.startswith("gyro") for c, m in zip(stream.channels, stream.mask) if m),
        )
        descriptors = encoder.text_encoder.encode_pooled(sensor_texts, device=device)
        descriptors = F.normalize(descriptors.float(), dim=-1)       # (S, 384)
        sensor_id = T.tensor(sensor_id_list, dtype=T.long, device=device).unsqueeze(0)
        channel_mask = T.as_tensor(stream.mask, dtype=T.bool, device=device).unsqueeze(0)
        modality_by_slot = sensor_modality_codes(
            sensor_id, channel_mask, len(sensor_texts))[0]           # (S,)
        gravity_by_slot = gravity_codes(
            modality_by_slot.unsqueeze(0), [gravity_state])[0]

        rows = SensorRows(
            feature=feature,
            descriptor=descriptors[slot],
            bias=feature.new_zeros((len(feature), 1)),
            modality=modality_by_slot[slot],
            gravity=gravity_by_slot[slot],
            label=T.full((len(feature),), -1, dtype=T.long, device=device),
            dataset=T.zeros(len(feature), dtype=T.long, device=device),
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

    # ---------------------------------------------------- native zero-shot face
    def predict_candidates_from_features(self, features, candidates, state, device):
        import torch as T
        owner = state["feature_owner"].get(id(features))
        if owner is None:
            raise ValueError(
                "halo_compact scores zero-shot with its native engine over per-sensor rows; the "
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
            for start in range(0, len(rows.feature), QUERY_CHUNK):
                sl = slice(start, start + QUERY_CHUNK)
                chunk = SensorRows(**{
                    f: (getattr(rows, f)[sl] if getattr(rows, f) is not None else None)
                    for f in ("feature", "descriptor", "bias", "modality", "gravity",
                              "label", "dataset", "enrolled_candidate", "source_window")
                })
                result = engine(chunk, bank, candidate_text, state["label_text"],
                                generator=generator)
                # identical aggregation to run_episode: sum per-row vote mass per window
                mass.index_add_(0, window[sl], result["logits"].float())
        predictions = [candidates[i] for i in mass.argmax(dim=1).cpu().tolist()]
        return predictions, {"predicted_classes": sorted(set(predictions)),
                             "mechanism": "engine_native_zero_shot",
                             "bank_rows": int(bank.feature.shape[0])}

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        features = self.window_features(stream, state, device)
        return self.predict_candidates_from_features(
            features, stream.eval_labels, state, device)
