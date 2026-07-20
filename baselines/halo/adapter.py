"""HALO (our model) scored through the IDENTICAL ConSE ZS-XD path as the baselines.

HALO Phase-A (``best.pt``) is a configuration-conditional *representation* encoder — it
has no trained label-text head. To place HALO in the SAME table as the self-pretrained /
frozen baselines (crosshar, harnet, ...), we give it the SAME treatment those get: a
linear softmax head fit on FROZEN HALO features over our 59-way global training
vocabulary, then scored zero-shot through the ConSE bridge. This is methodologically
identical to the crosshar / harnet ConSE head-fit — SUBJECT-DISJOINT epoch selection +
temperature calibration (the #82 fix) — so HALO's row is apples-to-apples with the
baseline rows (same estimator, same bridge, same leakage-safe protocol).

INPUT CONTRACT: native. HALO's whole thesis is rate/channel invariance via the physical-Hz
filterbank tokenizer + gravity-aware DC feature, so — unlike every closed-vocab baseline,
which is resampled to a fixed rate/length — HALO takes each eval stream at its NATIVE rate
and length with NO resampling (``InputContract()`` = accepts native). The feature extractor
is the exact ``encode_dataset`` used by ``training.tokenizer.eval_transfer`` (the validated
frozen-encoder transfer path), so head-fit features and eval features are produced the same
way HALO's internal transfer probe produces them.

Backbone: ``training/tokenizer/outputs/pretrain_native/best.pt`` (gitignored; the real
30k-step d_model-256 run, val kNN-BA 0.659 — NOT the ``pretrain/`` smoke checkpoint).
Head cache: ``baselines/halo/halo_conse_head.pt`` (gitignored), stamped with the global
vocab + a content hash of the backbone so it re-fits if either changes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import ConSEAdapter, InputContract, global_labels, register
from data.scripts.eda.grid_io import discover_grids
from data.scripts.labels.canonical_labels import canonicalize
from eval import scoring

# NOTE: everything from ``training.tokenizer.*`` (which pulls in HALO's ``model`` package) is
# imported LAZILY inside the methods below — never at module top level. Importing this adapter must
# NOT load ``model`` into ``sys.modules``, or it would shadow other baselines' repo-local ``model``
# module (e.g. UniMTS's ``from model import ST_GCN_18``) whenever ``import baselines`` runs.

# --- backbone + head-fit hyperparameters (parity with the crosshar/harnet ConSE head-fit) ---
_REPO = Path(__file__).resolve().parents[2]
_BACKBONE_CKPT = Path(os.environ.get(
    "HALO_CKPT", _REPO / "training/tokenizer/outputs/pretrain_native/best.pt"))
_HEAD_CACHE = Path(__file__).resolve().parent / "halo_conse_head.pt"

FIT_EPOCHS = 100
FIT_BATCH = 512
FIT_LR = 1e-3
FIT_SEED = 3431
EMBED_BATCH = 256
HEAD_FIT_MAX_PER_STREAM = 20000   # cap per stream for a tractable head-fit (matches crosshar)


def _backbone_fp() -> str:
    """Content hash of the frozen encoder — stamped into the head cache so a re-trained
    checkpoint invalidates a head fit on the old feature space."""
    return hashlib.sha256(_BACKBONE_CKPT.read_bytes()).hexdigest() if _BACKBONE_CKPT.exists() else ""


def _encode(enc, data: np.ndarray, texts, rate: float, gravity_state, device) -> np.ndarray:
    """(N, T, 6) native-rate windows -> (N, d) pooled HALO embeddings (numpy float32).

    Thin wrapper over the validated ``eval_transfer.encode_dataset`` so head-fit and eval
    features are produced identically to HALO's internal frozen-encoder transfer probe."""
    from training.tokenizer.eval_transfer import encode_dataset   # lazy: loads HALO's model pkg
    z = encode_dataset(enc, data, texts, device, float(rate), gravity_state)  # (N, d) on cpu
    return z.numpy().astype(np.float32)


class HALOAdapter(ConSEAdapter):
    """HALO Phase-A frozen representation + fitted ConSE head. Native input contract."""

    name = "halo"
    # Native: no channel/rate/window constraint — HALO ingests each stream as-is.
    contract = InputContract()

    # ---- setup: load frozen encoder + fit (or load cached) ConSE head --------
    def setup(self, device):
        vocab = global_labels()
        enc, feat_dim = self._load_encoder(device)

        head = nn.Linear(feat_dim, len(vocab)).to(device)
        cached = self._load_cached_head(vocab, feat_dim, device)
        if cached is not None:
            head_sd, temperature = cached
            head.load_state_dict(head_sd)
        else:
            temperature = self._fit_head(enc, head, vocab, device)
        head.eval()
        return {"encoder": enc, "head": head, "temperature": temperature}

    def _load_encoder(self, device):
        if not _BACKBONE_CKPT.exists():
            raise FileNotFoundError(
                f"HALO checkpoint missing at {_BACKBONE_CKPT}. Point HALO_CKPT at the "
                "trained Phase-A run (training/tokenizer/outputs/pretrain_native/best.pt).")
        from training.tokenizer.eval_transfer import build_encoder   # lazy (loads HALO model pkg)
        ckpt = torch.load(str(_BACKBONE_CKPT), map_location="cpu", weights_only=False)
        enc = build_encoder(ckpt, device)   # already frozen (.eval()); build_encoder sets dropout=0
        for p in enc.parameters():
            p.requires_grad_(False)
        feat_dim = int(ckpt["config"]["d_model"])
        print(f"[halo] loaded {_BACKBONE_CKPT.name}: step {ckpt['step']}, "
              f"val_ba {ckpt['val_ba']:.3f}, git {ckpt['git']}, d_model {feat_dim}", flush=True)
        return enc, feat_dim

    def _load_cached_head(self, vocab, feat_dim, device):
        if not _HEAD_CACHE.exists():
            return None
        blob = torch.load(str(_HEAD_CACHE), map_location=device, weights_only=True)
        if list(blob.get("labels", [])) != list(vocab):
            return None
        if blob.get("backbone_fp") != _backbone_fp():
            return None                       # re-trained checkpoint -> old head is invalid
        if "temperature" not in blob:
            return None                       # pre-calibration cache -> re-fit (#82)
        return blob["head"], float(blob["temperature"])

    def _fit_head(self, enc, head, vocab, device):
        """Fit the softmax head on frozen HALO features over the training corpus, selecting
        the best epoch on a SUBJECT-DISJOINT held-out fold (no source subject in both folds)."""
        from training.tokenizer.pretrain_data import (TRAIN_DATASETS, _stream_gravity_state,
                                                      stream_channel_descriptions)  # lazy
        label_to_idx = {l: i for i, l in enumerate(vocab)}
        fit_device = device if isinstance(device, torch.device) else torch.device(device)
        rng = np.random.RandomState(FIT_SEED)

        feats, labs, subjs, used = [], [], [], []
        refs = sorted((r for r in discover_grids("native") if r.dataset in TRAIN_DATASETS),
                      key=lambda r: r.key)
        for ref in refs:
            gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in ref.labels])
            keep = np.where(gl >= 0)[0]
            if keep.size == 0:
                continue
            if keep.size > HEAD_FIT_MAX_PER_STREAM:
                keep = np.sort(rng.choice(keep, HEAD_FIT_MAX_PER_STREAM, replace=False))
            data = np.asarray(ref.load_data()[keep])          # (n, T, 6) into RAM
            texts = stream_channel_descriptions(ref.dataset, ref.stream)
            gs = _stream_gravity_state(ref.dataset, ref.stream)
            feats.append(_encode(enc, data, texts, ref.rate_hz, gs, fit_device))
            labs.append(gl[keep])
            subjs.append(np.array([f"{ref.dataset}:{s}" for s in np.asarray(ref.subjects)[keep]]))
            used.append(f"{ref.key}({keep.size})")
        print(f"[halo] head-fit corpus: {used}", flush=True)

        X = np.concatenate(feats, 0)
        Y = np.concatenate(labs, 0)
        S = np.concatenate(subjs, 0)

        ti, vi, _ = scoring.subject_disjoint_split(S, seed=FIT_SEED)
        Xt = torch.from_numpy(X[ti]).float()
        Yt = torch.from_numpy(Y[ti]).long()
        Xv = torch.from_numpy(X[vi]).float().to(fit_device)
        Yv = Y[vi]

        head.to(fit_device)
        opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
        crit = nn.CrossEntropyLoss()
        n = len(Xt)
        best_acc, best_sd = -1.0, None
        for _ in range(FIT_EPOCHS):
            head.train()
            perm = rng.permutation(n)
            for s in range(0, n, FIT_BATCH):
                bi = perm[s:s + FIT_BATCH]
                opt.zero_grad()
                loss = crit(head(Xt[bi].to(fit_device)), Yt[bi].to(fit_device))
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                va = float((head(Xv).argmax(1).cpu().numpy() == Yv).mean())
            if va > best_acc:
                best_acc = va
                best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if best_sd is not None:
            head.load_state_dict(best_sd)

        # Temperature-scale on the SAME subject-disjoint val fold so ConSE weights top-K by
        # calibrated confidences, not raw over-confident softmax (#82).
        head.eval()
        with torch.no_grad():
            temperature = scoring.fit_temperature(head(Xv).cpu().numpy(), Yv)
        n_val_subj = len(set(S[vi]))
        print(f"[halo] fitted head: val_acc={best_acc:.3f}, T={temperature:.3f} over "
              f"{n_val_subj} held-out subjects ({n} train / {len(Xv)} val windows)", flush=True)
        torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                    "labels": list(vocab), "backbone_fp": _backbone_fp(),
                    "temperature": float(temperature)}, str(_HEAD_CACHE))
        head.to(device)
        return float(temperature)

    # ---- window_probs: (N, 59) softmax over the global vocab ------------------
    def window_probs(self, stream, state, device) -> np.ndarray:
        from training.tokenizer.pretrain_data import (_stream_gravity_state,
                                                      stream_channel_descriptions)  # lazy
        enc, head = state["encoder"], state["head"]
        T = float(state.get("temperature", 1.0))     # calibrated temperature (#82)
        texts = stream_channel_descriptions(stream.dataset, stream.stream)
        gs = _stream_gravity_state(stream.dataset, stream.stream)
        feats = _encode(enc, np.asarray(stream.windows), texts, stream.rate_hz, gs, device)
        probs = []
        with torch.no_grad():
            for s in range(0, len(feats), EMBED_BATCH):
                b = torch.from_numpy(feats[s:s + EMBED_BATCH]).float().to(device)
                probs.append(F.softmax(head(b) / T, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)


register(HALOAdapter)
