"""Build the frozen archetypal memory bank for Phase-B (the evidence engine).

This is the ONE expensive pass in Phase B. We encode the training corpus a single
time with the frozen Pipeline-A encoder (the fixed+multiresolution default) and cache
the pooled window vectors + metadata to disk. Every downstream Phase-B training step
then operates purely on these cached vectors — the encoder never runs in the loop, so
episodic training is a batched matmul over a compact in-VRAM bank (see
docs/design/EVIDENCE_ENGINE.md §4.2).

Cached bank (``memory_bank.pt``):
    Z          (N, d)  float16   pooled frozen-encoder embeddings (L2-normalizable downstream)
    y          (N,)    int64     global-vocab label index (canonicalized; -1 dropped)
    subj       (N,)    int64     composite "dataset:subject" id (for subject-disjoint episodes)
    cfg        (N,)    int64     stream id "dataset/stream" (config bucket)
    vocab      list[str]         current global label vocabulary (row i == label index i)
    label_text (L, 384) float32  frozen-SBERT embedding of each vocab label (the ConSE text space)
    subj_names / cfg_names        int->string decoders for subj / cfg
    backbone   dict              provenance (ckpt path, step, val_ba, git, content fingerprint)
    patch      dict              schema-v2 valid patch vectors + parent window/event/config,
                                center time, duration, resolution, and verification metadata

The pooled keys are intentionally unchanged: they are the T2.0 control. ``schema_version=2`` adds
the patch table and a separate behavioral patch-path probe without reinterpreting pooled vectors.

Memory is built from CLEAN (un-augmented) encodings — a retrieval bank of jittered
vectors would be matching against noise. Label/query augmentation is a *training-loop*
concern (the learned t-kernel), not a memory concern.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      $PY -m training.evidence.build_memory --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from data.scripts.labels.canonical_labels import canonicalize
from eval.scoring import get_sbert_encoder
from training.tokenizer.eval_transfer import (
    build_encoder,
    embedding_fingerprint,
    encode_dataset_detailed,
    patch_embedding_fingerprint,
)
from training.tokenizer.pretrain_data import (TRAIN_DATASETS, _stream_gravity_state,
                                              stream_channel_descriptions)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt"
_DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "memory_bank.pt"
_GLOBAL_LABELS = _REPO / "data/labels/global_labels.json"


def _load_vocab() -> list[str]:
    return list(json.loads(_GLOBAL_LABELS.read_text())["labels"])


def _backbone_fp(ckpt_path: Path) -> str:
    return hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else ""


def label_config_balanced_keep(
    labels: torch.Tensor,
    configs: torch.Tensor,
    max_per_label: int | None,
    *,
    seed: int = 20260720,
) -> torch.Tensor:
    """Select whole windows with an equalized configuration budget inside every label."""
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    configs = torch.as_tensor(configs, dtype=torch.long).cpu()
    if labels.shape != configs.shape:
        raise ValueError("labels and configs must have the same shape")
    if max_per_label is None:
        return torch.ones(len(labels), dtype=torch.bool)
    if max_per_label < 1:
        raise ValueError("max_per_label must be positive or None")

    keep = torch.zeros(len(labels), dtype=torch.bool)
    for label in sorted(labels.unique().tolist()):
        label_rows = torch.nonzero(labels.eq(label), as_tuple=True)[0]
        if len(label_rows) <= max_per_label:
            keep[label_rows] = True
            continue
        groups = {
            config: label_rows[configs[label_rows].eq(config)]
            for config in sorted(configs[label_rows].unique().tolist())
        }

        # Water-fill the per-label budget: equal quota for well-populated configurations, while
        # preserving every row from configurations smaller than their fair share.
        quotas: dict[int, int] = {}
        active = dict(groups)
        remaining = max_per_label
        while active:
            share = remaining // len(active)
            if share == 0:
                for config in sorted(active)[:remaining]:
                    quotas[config] = 1
                break
            small = [
                config for config, rows in active.items()
                if len(rows) <= share
            ]
            if small:
                for config in small:
                    quotas[config] = len(active[config])
                    remaining -= len(active[config])
                    del active[config]
                continue
            base, extra = divmod(remaining, len(active))
            for position, config in enumerate(sorted(active)):
                quotas[config] = base + int(position < extra)
            break

        for config, quota in quotas.items():
            rows = groups[config]
            if quota >= len(rows):
                chosen = rows
            else:
                generator = torch.Generator().manual_seed(
                    int(seed + 1_000_003 * label + 9_973 * config)
                )
                chosen = rows[torch.randperm(len(rows), generator=generator)[:quota]]
            keep[chosen] = True
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT", _DEFAULT_CKPT)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-per-stream", type=int, default=50000,
                    help="cap windows per stream at ENCODE time (tractable; -1 = all)")
    ap.add_argument("--max-per-label", type=int, default=8000,
                    help="configuration-balanced cap per global label AFTER encoding (tames "
                         "head-class/config flooding; rare labels/configs kept; -1 = no cap)")
    ap.add_argument(
        "--label-cap-policy",
        choices=("configuration_balanced", "random"),
        default="configuration_balanced",
        help="how --max-per-label selects common-label windows; random is the historical control",
    )
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap = None if args.max_per_stream is not None and args.max_per_stream < 0 else args.max_per_stream
    label_cap = None if args.max_per_label is not None and args.max_per_label < 0 else args.max_per_label

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"encoder checkpoint missing at {args.checkpoint}. Point --checkpoint / HALO_CKPT "
            "at the frozen Phase-A run (default: the fixed+MR winner pretrain_fixed_mr/best.pt).")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    d_model = int(ckpt["config"]["d_model"])
    print(f"[memory] encoder {args.checkpoint.name}: step {ckpt['step']}, val_ba "
          f"{ckpt['val_ba']:.3f}, git {ckpt['git']}, frontend={ckpt['config'].get('frontend')}, "
          f"MR={ckpt['config'].get('multiresolution')}, d={d_model}", flush=True)

    vocab = _load_vocab()
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    rng = np.random.RandomState(20260720)

    # Build from the roster the ENCODER was trained on (F4), not the current TRAIN_DATASETS: a bank
    # encoded over a different roster than the checkpoint saw is unattributable. None config => full.
    roster = tuple(ckpt["config"].get("train_datasets") or TRAIN_DATASETS)
    print(f"[memory] roster from checkpoint config: {sorted(roster)}", flush=True)
    refs = sorted((r for r in discover_grids("native") if r.dataset in roster),
                  key=lambda r: r.key)
    Z_parts, y_parts, subj_parts, cfg_parts, event_parts, event_verified_parts = [], [], [], [], [], []
    patch_Z_parts, patch_y_parts, patch_subj_parts = [], [], []
    patch_cfg_parts, patch_sensor_parts = [], []
    patch_window_parts, patch_event_parts, patch_event_verified_parts = [], [], []
    patch_time_parts, patch_duration_parts, patch_resolution_parts = [], [], []
    subj_names: dict[str, int] = {}
    cfg_names: dict[str, int] = {}
    cfg_rates: dict[int, float] = {}
    event_names: dict[str, int] = {}
    bank_streams: dict[str, int] = {}       # ref.key -> encoded window count (corpus provenance, F3)
    bank_datasets: set[str] = set()
    next_window = 0

    # Quality screens. build_memory reads grids directly rather than through CorpusIndex, so it
    # must apply the same exclusions the pretraining corpus does — otherwise the bank holds windows
    # the encoder was never trained on, including ExtraSensory's stale-buffer repeats whose labels
    # contradict each other (see data/scripts/scan_duplicates).
    from data.scripts.scan_duplicates import load as _load_duplicates
    from data.scripts.scan_implausible import load as _load_implausible
    excluded_by_stream = _load_duplicates("native")
    for _key, _idx in _load_implausible("native").items():
        excluded_by_stream[_key] = excluded_by_stream.get(_key, set()) | _idx

    for ref in refs:
        gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in ref.labels])
        excluded = excluded_by_stream.get(ref.key)
        if excluded:
            idx = np.fromiter(excluded, dtype=np.int64, count=len(excluded))
            stale = idx[idx >= len(gl)]
            if stale.size:      # cache predates a grid rebuild — refuse to guess which windows moved
                raise RuntimeError(
                    f"{ref.key}: quality cache indexes window {int(stale.max())} but the grid has "
                    f"{len(gl)}. Re-run data.scripts.scan_implausible and scan_duplicates."
                )
            gl = gl.copy()
            gl[idx] = -1
            print(f"[memory]   {ref.key}: {len(excluded)} windows excluded by quality screens",
                  flush=True)
        keep = np.where(gl >= 0)[0]
        if keep.size == 0:
            print(f"[memory]   {ref.key}: 0 in-vocab windows, skipped", flush=True)
            continue
        if cap is not None and keep.size > cap:
            keep = np.sort(rng.choice(keep, cap, replace=False))
        data = np.asarray(ref.load_data()[keep])
        texts = stream_channel_descriptions(ref.dataset, ref.stream)
        gs = _stream_gravity_state(ref.dataset, ref.stream)
        encoded = encode_dataset_detailed(
            enc, data, texts, device, float(ref.rate_hz), gs,
            channel_mask=ref.mask, dataset=ref.dataset, stream=ref.stream,
        )
        z = encoded["pooled"]   # (n, d) cpu; unchanged pooled compatibility path
        cfg_id = cfg_names.setdefault(ref.key, len(cfg_names))
        cfg_rates[cfg_id] = float(ref.rate_hz)
        subj_arr = np.asarray(ref.subjects)[keep]
        s_ids = np.array([subj_names.setdefault(f"{ref.dataset}:{s}", len(subj_names))
                          for s in subj_arr], dtype=np.int64)
        event_arr = np.asarray(ref.event_ids, dtype=object)[keep]
        e_ids = np.array(
            [event_names.setdefault(str(event), len(event_names)) for event in event_arr],
            dtype=np.int64,
        )
        event_verified = torch.full(
            (keep.size,), bool(ref.event_ids_explicit), dtype=torch.bool
        )
        local_patch_window = encoded["patch_window"].long()
        patch_window_parts.append(local_patch_window + next_window)
        patch_Z_parts.append(encoded["patch_Z"].to(torch.float16))
        patch_y_parts.append(torch.from_numpy(gl[keep].astype(np.int64))[local_patch_window])
        patch_subj_parts.append(torch.from_numpy(s_ids)[local_patch_window])
        patch_cfg_parts.append(
            torch.full((len(local_patch_window),), cfg_id, dtype=torch.int64)
        )
        patch_sensor_parts.append(
            torch.full((len(local_patch_window),), cfg_id, dtype=torch.int64)
        )
        patch_event_parts.append(torch.from_numpy(e_ids)[local_patch_window])
        patch_event_verified_parts.append(event_verified[local_patch_window])
        patch_time_parts.append(encoded["patch_time"].float())
        patch_duration_parts.append(encoded["patch_duration"].float())
        patch_resolution_parts.append(encoded["patch_resolution"].long())
        Z_parts.append(z.to(torch.float16))
        y_parts.append(torch.from_numpy(gl[keep].astype(np.int64)))
        subj_parts.append(torch.from_numpy(s_ids))
        cfg_parts.append(torch.full((keep.size,), cfg_id, dtype=torch.int64))
        event_parts.append(torch.from_numpy(e_ids))
        event_verified_parts.append(event_verified)
        next_window += int(keep.size)
        bank_streams[ref.key] = int(keep.size)
        bank_datasets.add(ref.dataset)
        print(f"[memory]   {ref.key}: {keep.size} windows, {len(set(subj_arr))} subjects", flush=True)

    Z = torch.cat(Z_parts)
    y = torch.cat(y_parts)
    subj = torch.cat(subj_parts)
    cfg = torch.cat(cfg_parts)
    event = torch.cat(event_parts)
    event_verified = torch.cat(event_verified_parts)
    patch = {
        "Z": torch.cat(patch_Z_parts),
        "y": torch.cat(patch_y_parts),
        "subj": torch.cat(patch_subj_parts),
        "cfg": torch.cat(patch_cfg_parts),
        # Each row currently comes from one deployment stream, so sensor-group identity equals cfg.
        # A separate field keeps the structural contract explicit for future multi-sensor streams.
        "sensor": torch.cat(patch_sensor_parts),
        "window": torch.cat(patch_window_parts),
        "event": torch.cat(patch_event_parts),
        "event_verified": torch.cat(patch_event_verified_parts),
        "time": torch.cat(patch_time_parts),
        "duration": torch.cat(patch_duration_parts),
        "resolution": torch.cat(patch_resolution_parts),
    }

    # Per-label/config balance: cap each global label while dividing its budget across acquisition
    # configurations. A random label-only cap leaves common Capture-24 labels almost entirely
    # acc-only wrist; the evidence engine instead needs each available way of observing an activity.
    if label_cap is not None:
        if args.label_cap_policy == "configuration_balanced":
            keep_mask = label_config_balanced_keep(y, cfg, label_cap)
        else:
            keep_mask = torch.zeros(len(y), dtype=torch.bool)
            for label in y.unique().tolist():
                rows = torch.nonzero(y.eq(label), as_tuple=True)[0]
                if len(rows) > label_cap:
                    generator = torch.Generator().manual_seed(int(label))
                    rows = rows[torch.randperm(len(rows), generator=generator)[:label_cap]]
                keep_mask[rows] = True
        before = len(y)
        remap = torch.full((before,), -1, dtype=torch.long)
        remap[keep_mask] = torch.arange(int(keep_mask.sum()), dtype=torch.long)
        keep_patch = keep_mask[patch["window"]]
        patch = {name: values[keep_patch] for name, values in patch.items()}
        patch["window"] = remap[patch["window"]]
        Z, y, subj, cfg, event, event_verified = (
            Z[keep_mask], y[keep_mask], subj[keep_mask], cfg[keep_mask], event[keep_mask],
            event_verified[keep_mask],
        )
        print(f"[memory] per-label cap {label_cap}: {before} -> {len(y)} windows", flush=True)

    sbert = get_sbert_encoder()
    label_text = torch.from_numpy(sbert(vocab).astype(np.float32))   # (L, 384) L2-normalized

    n_per_label = np.bincount(y.numpy(), minlength=len(vocab))
    print(f"[memory] bank: {Z.shape[0]} windows · d={Z.shape[1]} · {len(subj_names)} subjects · "
          f"{len(cfg_names)} configs · {int((n_per_label > 0).sum())}/{len(vocab)} labels present",
          flush=True)
    print(f"[memory] size: Z={Z.numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)
    print(f"[memory] patch table: {len(patch['Z'])} valid patches · "
          f"{patch['Z'].numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)

    from training.evidence.bank_guard import bank_fingerprint, vocab_fingerprint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "Z": Z, "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": event_verified,
        # Versioned patch evidence. The legacy pooled keys above intentionally remain unchanged so
        # official pooled controls and old adapters keep their exact data contract.
        "patch": patch,
        "vocab": vocab, "vocab_fp": vocab_fingerprint(vocab), "label_text": label_text,
        "subj_names": {v: k for k, v in subj_names.items()},
        "cfg_names": {v: k for k, v in cfg_names.items()},
        "cfg_rate_hz": cfg_rates,
        "event_names": {v: k for k, v in event_names.items()},
        "d_model": d_model,
        "backbone": {"checkpoint": str(args.checkpoint), "step": int(ckpt["step"]),
                     "val_ba": float(ckpt["val_ba"]), "git": ckpt["git"],
                     "fingerprint": _backbone_fp(args.checkpoint),
                     "frontend": ckpt["config"].get("frontend"),
                     "multiresolution": ckpt["config"].get("multiresolution")},
        "max_per_stream": cap,
        "max_per_label": label_cap,
        "balance_policy": {
            "label_cap": f"{args.label_cap_policy}_v1",
        },
        # Bind the bank to the corpus it was built from AND to the Phase-A corpus the encoder was
        # trained on (F3): a matching vocabulary + backbone is not enough — a bank encoded over a
        # different dataset roster / cap than the encoder saw is unattributable.
        "corpus": {
            "datasets": sorted(bank_datasets),
            "n_streams": len(bank_streams),
            "streams": bank_streams,                       # ref.key -> encoded window count
            "n_encoded_windows": int(sum(bank_streams.values())),
            "phase_a_corpus_fp": ckpt.get("corpus_fingerprint"),   # None for pre-fingerprint ckpts
            "phase_a_corpus": ckpt.get("corpus"),
        },
        # BEHAVIOURAL fingerprint of the embedding path that produced Z. Corpus/weight/roster
        # fingerprints all stay identical when the encode CODE changes (e.g. the F1 pooling fix),
        # so without this a stale bank passes every guard while storing vectors from a different
        # function. Consumers re-run the probe and compare (see bank_guard).
        "embed_probe": embedding_fingerprint(enc, device),
        "patch_embed_probe": patch_embedding_fingerprint(enc, device),
    }
    payload["bank_fp"] = bank_fingerprint(payload)
    torch.save(payload, str(args.out))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
