"""HALO evidence engine — T2.0: the UNTRAINED retrieval + text-ensemble mechanism, wired
as a first-class baseline so its 47.5 macro-F1 sits in the official ZS-XD table beside
ConSE (42.7) and harnet (47.3).

NO learning. Raw frozen fixed+MR encoder features + frozen SBERT text bridge, using the
exact best config from the tier-1 sweep (`training.evidence.tier1_sweep`):

    raw features · full-soft retrieval (no top-k cutoff) · tau=0.03 · text-ensemble(E=8) · no CSLS

Per window: encode -> cosine to the frozen memory bank -> softmax(sim/tau) retrieval weights ->
each neighbour's (ensembled) label text votes for the candidate labels by cosine -> argmax.

This is the do-no-harm FLOOR that every Tier-2 learned component must beat
(docs/design/EVIDENCE_ENGINE_TIER2.md §3). It reuses the SAME encoder and the SAME cached
memory bank as the decoder trainer, so the harness number is apples-to-apples with the
learned decoder that will (must) replace it. Provenance-guarded: the encoder checkpoint's
content hash must match the bank's backbone fingerprint or setup fails loud.

Backbone: ``training/tokenizer/outputs/pretrain_fixed_mr/best.pt`` (gitignored).
Bank:     ``training/evidence/outputs/memory_bank.pt`` (gitignored; built by build_memory).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from baselines.base import BaselineAdapter, InputContract, register
from eval import data as eval_data
from eval import scoring

# Everything from ``training.tokenizer.*`` (which imports HALO's ``model`` package) is imported
# LAZILY inside methods — importing this adapter must not load ``model`` into sys.modules, or it
# would shadow other baselines' repo-local ``model`` module (same rule as baselines/halo).

_REPO = Path(__file__).resolve().parents[2]
_BACKBONE_CKPT = Path(os.environ.get(
    "HALO_CKPT", _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt"))
_BANK = Path(os.environ.get(
    "HALO_BANK", _REPO / "training/evidence/outputs/memory_bank.pt"))

# --- tier-1 winning config (frozen; see tier1_sweep.json) ---
TAU = 0.03
TOP_K = 0            # 0 = full-soft (no neighbour cutoff)
TEXT_ENSEMBLE = 8    # paraphrase variants averaged per label
EMBED_BATCH = 256


class HALOEvidenceAdapter(BaselineAdapter):
    """Retrieval-over-frozen-memory ZS-XD predictor. Native input contract (no resampling)."""

    name = "halo_evidence"
    tier = "retrieval"
    contract = InputContract()   # native: HALO ingests each stream as-is

    # ---- setup: frozen encoder + cached bank + ensembled memory-label text ----
    def setup(self, device):
        device = torch.device(device) if not isinstance(device, torch.device) else device
        if not _BACKBONE_CKPT.exists():
            raise FileNotFoundError(
                f"HALO checkpoint missing at {_BACKBONE_CKPT}. Point HALO_CKPT at the frozen "
                "fixed+MR Phase-A run (pretrain_fixed_mr/best.pt).")
        if not _BANK.exists():
            raise FileNotFoundError(
                f"memory bank missing at {_BANK}. Build it: "
                "python -m training.evidence.build_memory --device cuda")

        bank = torch.load(str(_BANK), map_location="cpu", weights_only=True)
        from training.evidence.bank_guard import assert_bank_current   # lazy
        assert_bank_current(bank, context="halo_evidence adapter")
        fp = hashlib.sha256(_BACKBONE_CKPT.read_bytes()).hexdigest()
        bank_fp = bank["backbone"].get("fingerprint")
        if bank_fp and fp != bank_fp:
            raise RuntimeError(
                "halo_evidence: encoder checkpoint != bank backbone fingerprint — the bank was "
                "built with a DIFFERENT encoder. Rebuild build_memory against this checkpoint.")

        from training.tokenizer.eval_transfer import build_encoder   # lazy (loads HALO model pkg)
        ckpt = torch.load(str(_BACKBONE_CKPT), map_location="cpu", weights_only=False)
        enc = build_encoder(ckpt, device)
        for p in enc.parameters():
            p.requires_grad_(False)

        Z = F.normalize(bank["Z"].float().to(device), dim=-1)   # (N, d) raw-feature retrieval space
        mem_y = bank["y"].to(device)                            # (N,) vocab index per memory entry
        vocab = list(bank["vocab"])

        sbert = scoring.get_sbert_encoder()
        from training.evidence.labeltext import ensemble_text
        ml_ens = ensemble_text(vocab, sbert, TEXT_ENSEMBLE).to(device)   # (L, 384) memory-label text

        print(f"[halo_evidence] encoder {_BACKBONE_CKPT.name} (val_ba {ckpt['val_ba']:.3f}) · "
              f"bank {Z.shape[0]} windows d={Z.shape[1]} · tau={TAU} ens={TEXT_ENSEMBLE}", flush=True)
        return {"enc": enc, "Z": Z, "mem_y": mem_y, "vocab": vocab,
                "ml_ens": ml_ens, "sbert": sbert}

    # ---- predict: per-window argmax over the stream's candidate labels ----
    def predict(self, stream: eval_data.EvalStream, state, device) -> Tuple[List[str], dict]:
        from training.tokenizer.eval_transfer import encode_dataset          # lazy
        from training.tokenizer.pretrain_data import (_stream_gravity_state,
                                                      stream_channel_descriptions)   # lazy
        from training.evidence.labeltext import ensemble_text

        device = torch.device(device) if not isinstance(device, torch.device) else device
        enc, Z, mem_y = state["enc"], state["Z"], state["mem_y"]
        ml_ens, sbert = state["ml_ens"], state["sbert"]
        labels = list(stream.eval_labels)

        texts = stream_channel_descriptions(stream.dataset, stream.stream)
        gs = _stream_gravity_state(stream.dataset, stream.stream)
        z = encode_dataset(enc, np.asarray(stream.windows), texts, device,
                           float(stream.rate_hz), gs).to(device)
        z = F.normalize(z, dim=-1)                                           # (Nq, d)

        cand = ensemble_text(labels, sbert, TEXT_ENSEMBLE).to(device)        # (C, 384)
        K = torch.relu(ml_ens[mem_y] @ cand.t())                            # (N, C) neighbour->cand votes

        preds = np.empty(len(z), dtype=object)
        with torch.no_grad():
            for s in range(0, len(z), EMBED_BATCH):
                sim = z[s:s + EMBED_BATCH] @ Z.t()                          # (b, N) cosine
                if TOP_K:
                    thr = sim.topk(TOP_K, dim=1).values[:, -1:]
                    sim = sim.masked_fill(sim < thr, float("-inf"))
                w = torch.softmax(sim / TAU, dim=1)                          # (b, N)
                e = w @ K                                                    # (b, C) per-candidate evidence
                idx = e.argmax(1).cpu().numpy()
                preds[s:s + EMBED_BATCH] = [labels[i] for i in idx]
        return list(preds), {"predicted_classes": sorted(set(preds.tolist())),
                             "mechanism": "untrained-retrieval+text-ensemble"}


register(HALOEvidenceAdapter)
