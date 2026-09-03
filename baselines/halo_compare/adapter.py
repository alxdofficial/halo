"""HALO comparison model, scored on the shared adaptation manifest.

Two faces from one checkpoint, exactly as the design specifies:

* ``window_features`` — the encoder's pooled row per evaluation window, consumed by the shared
  nearest / prototype / ridge / linear-head readouts. Apples-to-apples with every baseline's
  frozen-feature row.
* **native enrollment** — the deployed mechanism. The manifest's support rows *are* the support
  set: they are already execution-disjoint from the query and drawn from the same stream, hence
  config-compatible by construction. There is **no corpus bank** at k >= 1; that stage was removed
  from the model, not hidden here.

THE ZERO-SHOT ROW
-----------------
At k = 0 there is no support, so the deployed mechanism has nothing to compare against. Per the
2026-09-03 decision the k = 0 row keeps the same mechanism and draws its support from the
*training* corpus: recordings that are config-compatible with the query and whose labels exclude
every candidate. The model still cannot find the answer among them; it can only note which
training recordings the query resembles and let their label text bridge to the candidates.

Two evaluation streams have no compatible training partner at all (``upper_limb_use``, whose
wrist sensor is a research IMU rather than a watch, and ``usc_had/phone_hip``, which has near
misses only). For those the k = 0 row is reported as **unsupported**, with the reason recorded in
the artifact. It is not silently replaced by a different mechanism, and it is not padded with
incompatible rows.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch.nn.functional as F

from baselines.base import BaselineAdapter, InputContract, register

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

_CKPT = Path(os.environ.get(
    "HALO_COMPARE_CKPT", _REPO / "training/compare/outputs/arm_a/last.pt",
))
#: Arm A is the headline configuration: the encoder is never told the acquisition configuration.
_NEUTRAL_ACQUISITION_TEXT = os.environ.get("HALO_NEUTRAL_ACQUISITION_TEXT", "1") not in {"0", ""}
#: Windows drawn from the training corpus to support the k = 0 row.
ZERO_SHOT_SUPPORT = int(os.environ.get("HALO_COMPARE_ZERO_SHOT_SUPPORT", "64"))
ZERO_SHOT_SEED = 20260901
QUERY_CHUNK = 1024


def _ckpt_sha256() -> str:
    return hashlib.sha256(_CKPT.read_bytes()).hexdigest() if _CKPT.exists() else ""


@register
class HALOCompareAdapter(BaselineAdapter):
    """Recognise by comparing the query to the episode's own labelled support recordings."""

    name = "halo_compare"
    tier = "evidence"
    contract = InputContract()          # native rate and channels; the tokenizer is the contract

    # ------------------------------------------------------------------ setup
    def setup(self, device):
        import torch as T
        from eval.scoring import get_sbert_encoder
        from model.blocks import AttentionSpec
        from model.evidence.comparator import ComparatorConfig, SupportComparator
        from training.tokenizer.eval_transfer import build_encoder

        if not _CKPT.exists():
            raise FileNotFoundError(
                f"HALO compare checkpoint missing at {_CKPT}. Set HALO_COMPARE_CKPT or run "
                "training.compare.train."
            )
        blob = T.load(str(_CKPT), map_location="cpu", weights_only=False)
        device = T.device(device)
        encoder = build_encoder(blob, device, training=False).eval()

        spec = AttentionSpec(**blob["attention_spec"])
        comparator = SupportComparator(spec, ComparatorConfig(**blob["comparator_config"]))
        comparator.load_state_dict(blob["comparator"])
        comparator = comparator.to(device).eval()

        trained_neutral = bool(blob.get("args", {}).get("neutral_acquisition_text", False))
        if trained_neutral != _NEUTRAL_ACQUISITION_TEXT:
            raise ValueError(
                f"{_CKPT.name} was trained with neutral_acquisition_text={trained_neutral} but is "
                f"being evaluated with {_NEUTRAL_ACQUISITION_TEXT}. Arm A's claim is that the "
                "encoder is never told the acquisition configuration; evaluating the other way "
                "round would report a different model than the one that was trained."
            )
        return {
            "encoder": encoder, "comparator": comparator, "device": device,
            "sbert": get_sbert_encoder(), "streams": {}, "feature_owner": {},
            "checkpoint_step": int(blob.get("step", 0)),
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            "zero_shot_support": {},
        }

    # ------------------------------------------------------- stream encoding
    def _stream_rows(self, stream, state):
        """Pooled feature and acquisition descriptor for every window of one eval stream."""
        from training.tokenizer.eval_transfer import encode_dataset_detailed
        from training.tokenizer.pretrain_data import (
            _stream_gravity_state,
            stream_channel_descriptions,
            stream_sensor_texts,
        )

        key = (stream.dataset, stream.stream)
        if key in state["streams"]:
            return state["streams"][key]
        device, encoder = state["device"], state["encoder"]

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
            neutral_text=_NEUTRAL_ACQUISITION_TEXT, export_sensor_rows=False,
        )
        feature = detailed["pooled"].to(device)

        _, sensor_texts, _ = stream_sensor_texts(
            stream.dataset, stream.stream,
            gravity_removed=(gravity_state == "removed"),
            has_accel=any(c.startswith("acc") for c, m in zip(stream.channels, stream.mask) if m),
            has_gyro=any(c.startswith("gyro") for c, m in zip(stream.channels, stream.mask) if m),
            neutral=_NEUTRAL_ACQUISITION_TEXT,
        )
        descriptors = encoder.text_encoder.encode_pooled(sensor_texts, device=device)
        descriptor = F.normalize(
            F.normalize(descriptors.float(), dim=-1).mean(dim=0, keepdim=True), dim=-1,
        ).expand(len(feature), -1)

        payload = (feature, descriptor, detailed["pooled"].float().cpu().numpy())
        state["streams"][key] = payload
        return payload

    # ------------------------------------------------ standard adaptation face
    def window_features(self, stream, state, device) -> np.ndarray:
        _, _, pooled = self._stream_rows(stream, state)
        state["feature_owner"][id(pooled)] = (stream.dataset, stream.stream)
        return pooled

    def supports_native_enrollment(self) -> bool:
        return True

    # ---------------------------------------------------------- the mechanism
    def _score(self, state, query_feature, query_descriptor,
               support_feature, support_descriptor, support_label_text, support_bound,
               candidate_text):
        """Run the comparator over one padded episode and return per-query candidate logits."""
        import torch as T
        from model.evidence.comparator import comparator_logits

        device = state["device"]
        n_query = query_feature.shape[0]
        n_support = support_feature.shape[0]
        n_candidates = candidate_text.shape[0]

        out = comparator_logits(
            state["comparator"],
            candidate_text=candidate_text.unsqueeze(0).expand(n_query, -1, -1),
            query_feature=query_feature.unsqueeze(1),
            query_descriptor=query_descriptor.unsqueeze(1),
            query_mask=T.ones((n_query, 1), dtype=T.bool, device=device),
            support_feature=support_feature.unsqueeze(0).expand(n_query, -1, -1),
            support_descriptor=support_descriptor.unsqueeze(0).expand(n_query, -1, -1),
            support_label_text=support_label_text.unsqueeze(0).expand(n_query, -1, -1),
            support_bound=support_bound.unsqueeze(0).expand(n_query, -1),
            support_mask=T.ones((n_query, n_support), dtype=T.bool, device=device),
            candidate_slot=T.arange(n_candidates, device=device).unsqueeze(0).expand(n_query, -1),
        )
        return out["logits"]

    def predict_enrollment(
        self, query_stream, support_stream, plan, support_count, candidate_texts,
        state, device, *, seed,
    ) -> Tuple[List[str], dict]:
        import torch as T

        query_feature, query_descriptor, _ = self._stream_rows(query_stream, state)
        support_feature_all, support_descriptor_all, _ = self._stream_rows(support_stream, state)

        canonical = list(plan["candidate_names"])
        if len(candidate_texts) != len(canonical):
            raise ValueError("candidate text roster must align with the manifest roster")
        candidate_text = F.normalize(T.from_numpy(
            state["sbert"]([text.replace("_", " ") for text in candidate_texts])
        ).float(), dim=-1).to(state["device"])

        # The manifest already chose which executions are enrolled for each candidate; the support
        # set is exactly those rows. No corpus bank, no retrieval, no selection.
        rows: list[int] = []
        bound: list[int] = []
        for slot, executions in enumerate(plan["support_execution_rows"]):
            for execution in executions[:support_count]:
                for row in execution:
                    rows.append(int(row))
                    bound.append(slot)
        if not rows:
            raise ValueError("native enrollment received no support rows")
        index = T.as_tensor(rows, dtype=T.long, device=state["device"])
        support_bound = T.as_tensor(bound, dtype=T.long, device=state["device"])
        support_feature = support_feature_all.index_select(0, index)
        support_descriptor = support_descriptor_all.index_select(0, index)
        # An enrolled row's label IS its candidate's text, which is what binds it.
        support_label_text = candidate_text.index_select(0, support_bound)

        requested = [int(row) for row in plan["query_rows"]]
        predictions: list[str] = []
        with T.no_grad():
            for start in range(0, len(requested), QUERY_CHUNK):
                chunk = requested[start:start + QUERY_CHUNK]
                wanted = T.as_tensor(chunk, dtype=T.long, device=state["device"])
                logits = self._score(
                    state,
                    query_feature.index_select(0, wanted),
                    query_descriptor.index_select(0, wanted),
                    support_feature, support_descriptor, support_label_text, support_bound,
                    candidate_text,
                )
                predictions.extend(canonical[int(i)] for i in logits.argmax(1).cpu().tolist())
        return predictions, {
            "mechanism": "support_comparator",
            "support_rows": int(len(rows)),
            "corpus_rows": 0,
            "enrolled_executions": int(len(canonical) * support_count),
        }

    # ------------------------------------------------------------ zero shot
    def _zero_shot_support(self, stream, state, candidates: Sequence[str]):
        """Config-compatible training recordings whose labels exclude every candidate."""
        from data.scripts.curate.compatibility import are_compatible, stream_key
        from data.scripts.curate import deployment_policy
        from training.compare.sampling import build_support_corpus

        key = (stream.dataset, stream.stream)
        if key not in state["zero_shot_support"]:
            query_key = stream_key(stream.dataset, stream.stream)
            corpus = state.get("_corpus")
            if corpus is None:
                corpus = build_support_corpus(
                    deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS,
                    max_per_stream=4000, seed=ZERO_SHOT_SEED,
                )
                state["_corpus"] = corpus
            pool = [
                index for index, other in enumerate(corpus.keys)
                if are_compatible(query_key, other)
            ]
            state["zero_shot_support"][key] = (corpus, set(pool))
        corpus, streams = state["zero_shot_support"][key]
        if not streams:
            return None, "no config-compatible training stream exists for this configuration"

        banned = {str(text).lower() for text in candidates}
        rng = np.random.default_rng(ZERO_SHOT_SEED)
        eligible = [
            index for index, recording in enumerate(corpus.recordings)
            if recording.stream_index in streams and str(recording.label).lower() not in banned
        ]
        if not eligible:
            return None, "every compatible training recording carries a candidate label"
        chosen = rng.choice(
            eligible, size=min(ZERO_SHOT_SUPPORT, len(eligible)), replace=False,
        )
        return [corpus.recordings[int(i)] for i in chosen], None

    def evaluation_config(self, state) -> dict:
        return {
            "checkpoint": str(_CKPT),
            "checkpoint_sha256": _ckpt_sha256(),
            "checkpoint_step": state.get("checkpoint_step"),
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            "mechanism": "support_comparator",
            "corpus_bank_at_positive_k": False,
            "zero_shot_support_windows": ZERO_SHOT_SUPPORT,
            "zero_shot_support_policy": "config_compatible_training_rows_excluding_candidates",
        }

    def evaluation_artifacts(self, state):
        return {"checkpoint": _CKPT} if _CKPT.exists() else {}

    def evaluation_source_paths(self):
        return (
            _REPO / "model" / "evidence" / "comparator.py",
            _REPO / "training" / "compare",
            _REPO / "data" / "scripts" / "curate" / "compatibility.py",
        )
