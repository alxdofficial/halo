"""Held-out-config transfer probe for a frozen Phase-1 encoder.

The internal pretraining val-kNN is subject-disjoint but WITHIN the training datasets.
This measures the thing we actually care about: does the frozen representation cluster
activities on **held-out eval datasets** (unseen placements/devices/subjects)? For each
eval dataset we encode its windows with the frozen encoder and run a subject-disjoint
kNN balanced accuracy over that dataset's OWN labels — pure representation transfer, no
ConSE/text head involved (that's Pipeline B).

NOT the baseline-table number (which is ConSE macro-F1) — it's a fast, honest transfer
signal to decide whether Pipeline A cleared the bar before building Pipeline B.

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.eval_transfer \
        --checkpoint training/tokenizer/outputs/pretrain_native/best.pt
      # NB: pretrain_native/best.pt is the real trained model (val_ba 0.659). The default
      # outputs/pretrain/ dir holds only smoke/debug runs — do NOT evaluate that one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.pretrain_data import (DFT_SIZE, stream_channel_descriptions,
                                              stream_sensor_texts, _stream_gravity_state,
                                              MultiResolutionCollate, MultiScaleCollate,
                                              STREAM_SOURCE_RATE_HZ, VAL_RESOLUTION_PAIR)

# Held-out eval datasets (never in TRAIN_DATASETS). tnda_har/ut_complex excluded
# (degenerate subject ids -> can't do subject-disjoint kNN).
EVAL_STREAMS = (
    ("motionsense", "phone_front_pocket"),
    ("realworld", "phone_waist"),
    ("shoaib", "phone_right_pocket"),
    ("inclusivehar", "phone_waist"),
)
PATCH_SECONDS = 1.0
KNN_K = 5
SEED = 20260718


def build_encoder(ckpt: dict, device, *, training: bool = False) -> SetTokenizerEncoder:
    c = ckpt["config"]
    frontend = c.get("frontend", "fixed")
    kw = dict(
        d_model=c["d_model"], num_layers=c["num_layers"], num_heads=c["num_heads"],
        dim_feedforward=c["dim_feedforward"],
        dropout=float(c.get("dropout", 0.1)) if training else 0.0,
        dft_size=DFT_SIZE,
        frontend=frontend,                                  # reconstruct the ACTUAL arm (was: always filterbank)
        use_duration_embedding=c.get("multiresolution", False),
        duration_min_seconds=min(c.get("short_patch_choices", (0.4,))),
        duration_max_seconds=max(c.get("long_patch_choices", (1.5,))),
        duration_gate_init=c.get("duration_gate_init", 0.1),
        rope_min_period=0.4 if c.get("multiresolution", False) else 0.5,
        text_conditioning=c.get("text_conditioning", "per_channel"),  # reconstruct the ACTUAL arm
        gate_bias_init=c.get("gate_bias_init", -2.0),
    )
    kw.update(                                              # fixed / learnable filterbank hyperparams
        center_shift_fraction=c.get("center_shift_fraction", 0.45),
        bandwidth_factor_max=c.get("bandwidth_factor_max", 1.5),
        compression_gain_max=c.get("compression_gain_max", 2.0),
        filter_shape_min=c.get("filter_shape_min", 1.5),
        filter_shape_max=c.get("filter_shape_max", 2.5),
        adaptive_gate_init=c.get("adaptive_gate_init", 0.1),
    )
    enc = SetTokenizerEncoder(**kw)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval_resolution_pair = tuple(c.get("val_resolution_pair", VAL_RESOLUTION_PAIR))
    enc.min_resolution_ratio = float(c.get("min_resolution_ratio", 1.75))
    enc = enc.to(device)
    return enc.train() if training else enc.eval()


#: Fixed probe spec. A window length that does NOT divide evenly into the patch grid, so the probe
#: exercises partial-tail handling (duration-weighted pooling + the tail floor) — the exact code the
#: F1/F7 fixes changed. Never edit these numbers: doing so invalidates every stored fingerprint.
_PROBE = dict(n=2, samples=128, rate=50.0, dataset="wisdm", stream="phone_pocket")


@torch.no_grad()
def embedding_fingerprint(enc, device) -> torch.Tensor:
    """Deterministic BEHAVIOURAL fingerprint of the whole embedding path -> (n, d) float32 cpu.

    Weights + corpus + vocab fingerprints cannot detect a change to the *code* that turns them into
    an embedding (pooling, tail selection, text construction). A pre-F1-pooling memory bank paired
    with a post-F1 encoder passed every existing guard while storing vectors from a different
    function. This runs a FIXED synthetic window through the real ``encode_dataset`` path, so ANY
    behavioural change to that path shows up as a different fingerprint — no version bump to
    remember. Compared with a tolerance (not hashed) so CPU/GPU float noise can't false-alarm.
    """
    g = torch.Generator().manual_seed(20260724)
    data = torch.randn(_PROBE["n"], _PROBE["samples"], 6, generator=g).numpy()
    texts = [f"probe channel {i}" for i in range(6)]
    return encode_dataset(enc, data, texts, device, _PROBE["rate"], gravity_state="present",
                          channel_mask=[True] * 6, dataset=_PROBE["dataset"],
                          stream=_PROBE["stream"]).float().cpu()


@torch.no_grad()
def patch_embedding_fingerprint(enc, device) -> torch.Tensor:
    """Deterministic fingerprint of valid patch vectors and their structural metadata."""
    g = torch.Generator().manual_seed(20260724)
    data = torch.randn(_PROBE["n"], _PROBE["samples"], 6, generator=g).numpy()
    texts = [f"probe channel {i}" for i in range(6)]
    encoded = encode_dataset_detailed(
        enc, data, texts, device, _PROBE["rate"], gravity_state="present",
        channel_mask=[True] * 6, dataset=_PROBE["dataset"], stream=_PROBE["stream"],
    )
    pieces = [
        encoded["patch_Z"].float().flatten().cpu(),
        encoded["patch_window"].float().cpu(),
        encoded["patch_time"].float().cpu(),
        encoded["patch_duration"].float().cpu(),
        encoded["patch_resolution"].float().cpu(),
    ]
    return torch.cat(pieces)


def encode_dataset_detailed(enc, data, texts, device, rate: float, gravity_state=None,
                            channel_mask=None, dataset=None, stream=None,
                            source_rate: float | None = None,
                            _require_patches: bool = True,
                            requires_grad: bool = False,
                            neutral_text: bool = False,
                            export_sensor_rows: bool = False,
                            batch_size: int = 256) -> dict[str, torch.Tensor]:
    """Encode windows and retain both pooled and valid per-patch representations.

    Returns:
      pooled          (N, d)
      patch_Z         (M, d)
      patch_window    (M,) local input-window row
      patch_time      (M,) window-relative patch-center seconds
      patch_duration  (M,) represented physical seconds
      patch_resolution (M,) 0=single/short grid, 1=long grid

    ``dataset``/``stream`` are only needed for a FACTORED encoder (to build the role/sensor text);
    the default per_channel path uses the ``texts`` (per-channel descriptions) exactly as before.

    Only patches marked valid by ``patch_padding_mask`` are exported. This is important for the
    Phase-B memory bank: padding tokens must never become retrievable evidence. The historical
    :func:`encode_dataset` API below remains a pooled-only compatibility wrapper.
    """
    collate = (
        MultiResolutionCollate(fixed_patch_seconds=enc.eval_resolution_pair,
                               min_resolution_ratio=enc.min_resolution_ratio)
        if enc.use_duration_embedding else
        MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS)
    )
    if source_rate is None:
        source_rate = STREAM_SOURCE_RATE_HZ.get(f"{dataset}/{stream}", float(rate))
    source_rate = min(float(source_rate), float(rate))
    enc_texts = texts
    cmask = (torch.ones(6, dtype=torch.bool) if channel_mask is None
             else torch.as_tensor(channel_mask, dtype=torch.bool))
    if cmask.shape != (6,):
        raise ValueError(f"channel_mask must have shape (6,), got {tuple(cmask.shape)}")
    # Sensor granularity always needs the per-sensor text (role text is retired once xyz folds into
    # one token), so it takes the factored text path regardless of `text_conditioning`.
    factored = (getattr(enc, "text_conditioning", "per_channel") == "factored"
                or getattr(enc, "token_granularity", "channel") == "sensor")
    if factored:
        if dataset is None or stream is None:
            raise ValueError("a factored encoder needs dataset+stream to build the factored "
                             "(role/sensor) text conditioning; pass them to encode_dataset()")
        role_texts, sensor_texts, sensor_id_list = stream_sensor_texts(
            dataset,
            stream,
            gravity_removed=(None if gravity_state is None else gravity_state == "removed"),
            has_accel=bool(cmask[:3].any()),
            has_gyro=bool(cmask[3:].any()),
            neutral=neutral_text,
        )
        sensor_id_t = torch.tensor(sensor_id_list, dtype=torch.long)
    embs = []
    sensor_z_parts, sensor_window_parts, sensor_slot_parts = [], [], []
    sensor_time_parts, sensor_duration_parts, sensor_resolution_parts = [], [], []
    patch_z_parts = []
    patch_window_parts = []
    patch_time_parts = []
    patch_duration_parts = []
    patch_resolution_parts = []
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    for start in range(0, len(data), batch_size):
        with grad_context():
            block = torch.tensor(np.asarray(data[start:start + batch_size]), dtype=torch.float32)
            items = []
            for window in block:
                item = {"data": window, "rate": rate, "source_rate": source_rate,
                        "texts": enc_texts, "label_id": 0,
                        "channel_mask": cmask, "gravity_state": gravity_state, "source": "eval"}
                if factored:
                    item["role_texts"] = role_texts
                    item["sensor_texts"] = sensor_texts
                    item["sensor_id"] = sensor_id_t
                items.append(item)
            batch = collate(items)
            plen = batch["patch_len"]
            out = enc(
                batch["patches"].to(device), batch["rates"].to(device),
                plen.to(device),
                batch["role_texts"] if factored else batch["texts"],
                batch["positions"].to(device),
                patch_durations=(batch["patch_durations"].to(device)
                                 if "patch_durations" in batch else None),
                resolution_ids=(batch["resolution_ids"].to(device)
                                if "resolution_ids" in batch else None),
                channel_mask=batch["channel_mask"].to(device),
                patch_padding_mask=batch["patch_padding_mask"].to(device),
                sensor_texts=(batch["sensor_texts"] if factored else None),
                sensor_id=(batch["sensor_id"].to(device) if factored else None),
                source_rate_hz=batch["source_rates"].to(device),
            )
            embs.append(out["pooled"] if requires_grad else out["pooled"].cpu())
            if "per_patch" not in out:
                if _require_patches:
                    raise KeyError("encoder detailed export requires an out['per_patch'] tensor")
                continue

            valid = batch["patch_padding_mask"].bool()
            per_patch = out["per_patch"]
            valid_on_output = valid.to(per_patch.device)
            local_rows = (
                torch.arange(len(block), dtype=torch.long).unsqueeze(1)
                .expand_as(valid) + int(start)
            )
            if "patch_durations" in batch:
                durations = batch["patch_durations"].float()
                resolutions = batch["resolution_ids"].long()
            else:
                durations = (
                    batch["patch_len"].float() / batch["rates"].clamp_min(1e-8)
                ).unsqueeze(1).expand_as(batch["positions"])
                resolutions = torch.zeros_like(batch["positions"], dtype=torch.long)

            patch_z_parts.append(per_patch[valid_on_output])
            metadata_device = per_patch.device if requires_grad else torch.device("cpu")
            patch_window_parts.append(local_rows[valid].to(metadata_device))
            patch_time_parts.append(batch["positions"].float()[valid].to(metadata_device))
            patch_duration_parts.append(durations[valid].to(metadata_device))
            patch_resolution_parts.append(resolutions[valid].to(metadata_device))

            # PER-SENSOR rows (design of record bank layout). Only available at sensor granularity,
            # where out["tokens"] is (B,P,S,d) and each slot is one modality triad. Emitting these
            # is what lets an accel-only query match accel rows exactly instead of comparing against
            # an embedding with a channel-set mismatch smeared into it.
            if export_sensor_rows and "sensor_present" in out:
                tokens = out["tokens"]                                # (B,P,S,d)
                present = out["sensor_present"]                       # (B,S)
                keep = valid_on_output.unsqueeze(2) & present.unsqueeze(1)   # (B,P,S)
                b_idx, p_idx, s_idx = keep.nonzero(as_tuple=True)
                sensor_z_parts.append(tokens[b_idx, p_idx, s_idx])
                sensor_window_parts.append((b_idx.cpu() + int(start)).to(metadata_device))
                sensor_slot_parts.append(s_idx.to(metadata_device))
                sensor_time_parts.append(
                    batch["positions"].float().to(keep.device)[b_idx, p_idx].to(metadata_device))
                sensor_duration_parts.append(
                    durations.to(keep.device)[b_idx, p_idx].to(metadata_device))
                sensor_resolution_parts.append(
                    resolutions.to(keep.device)[b_idx, p_idx].to(metadata_device))

    pooled = torch.cat(embs)
    d_model = int(pooled.shape[-1])
    return {
        "pooled": pooled,
        "patch_Z": (
            torch.cat(patch_z_parts)
            if patch_z_parts else torch.empty((0, d_model), dtype=pooled.dtype)
        ),
        "patch_window": (
            torch.cat(patch_window_parts)
            if patch_window_parts else torch.empty(0, dtype=torch.long)
        ),
        "patch_time": (
            torch.cat(patch_time_parts)
            if patch_time_parts else torch.empty(0, dtype=torch.float32)
        ),
        "patch_duration": (
            torch.cat(patch_duration_parts)
            if patch_duration_parts else torch.empty(0, dtype=torch.float32)
        ),
        "patch_resolution": (
            torch.cat(patch_resolution_parts)
            if patch_resolution_parts else torch.empty(0, dtype=torch.long)
        ),
        # Per-(patch, sensor) rows — empty unless export_sensor_rows AND the encoder is at sensor
        # granularity. `sensor_slot` indexes into the stream's sensor_texts order, which is how a
        # consumer recovers modality, descriptor and sensor_bias for the row.
        "sensor_Z": (
            torch.cat(sensor_z_parts)
            if sensor_z_parts else torch.empty((0, d_model), dtype=pooled.dtype)
        ),
        "sensor_window": (
            torch.cat(sensor_window_parts)
            if sensor_window_parts else torch.empty(0, dtype=torch.long)
        ),
        "sensor_slot": (
            torch.cat(sensor_slot_parts)
            if sensor_slot_parts else torch.empty(0, dtype=torch.long)
        ),
        "sensor_time": (
            torch.cat(sensor_time_parts)
            if sensor_time_parts else torch.empty(0, dtype=torch.float32)
        ),
        "sensor_duration": (
            torch.cat(sensor_duration_parts)
            if sensor_duration_parts else torch.empty(0, dtype=torch.float32)
        ),
        "sensor_resolution": (
            torch.cat(sensor_resolution_parts)
            if sensor_resolution_parts else torch.empty(0, dtype=torch.long)
        ),
    }


@torch.no_grad()
def encode_dataset(enc, data, texts, device, rate: float, gravity_state=None,
                   channel_mask=None, dataset=None, stream=None,
                   source_rate: float | None = None,
                   neutral_text: bool = False) -> torch.Tensor:
    """Compatibility API: raw windows at native rate -> pooled embeddings only."""
    return encode_dataset_detailed(
        enc, data, texts, device, rate, gravity_state=gravity_state,
        channel_mask=channel_mask, dataset=dataset, stream=stream,
        source_rate=source_rate,
        neutral_text=neutral_text,
        _require_patches=False,
    )["pooled"]


def knn_balanced_acc(train_z, train_y, test_z, test_y, k=KNN_K) -> float:
    labels = sorted(set(train_y) & set(test_y))
    per_class = []
    for label in labels:
        idx = [i for i, y in enumerate(test_y) if y == label]
        hits = 0
        for i in idx:
            d = (train_z - test_z[i]).norm(dim=1)
            nn = [train_y[j] for j in d.argsort()[:k].tolist()]
            hits += max(set(nn), key=nn.count) == label
        per_class.append(hits / len(idx))
    return float(np.mean(per_class)) if per_class else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parity", action="store_true",
                    help="Also score the PARITY arm (neutral text: modality+axis only, no "
                         "placement/device/gravity) and report the gap. The gap IS the measured "
                         "value of acquisition-config conditioning — BASELINE_FAIRNESS_POLICY.md §5.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the JSON (default: <checkpoint dir>/transfer_eval.json)")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    print(f"loaded {args.checkpoint.name}: step {ckpt['step']}, "
          f"internal val_ba {ckpt['val_ba']:.3f}, git {ckpt['git']}", flush=True)

    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    arms = {"full": False, "parity": True} if args.parity else {"full": False}
    arm_results: dict[str, dict] = {}
    arm_means: dict[str, float] = {}

    for arm, neutral in arms.items():
        # Reseed per arm so BOTH arms hold out the SAME subjects. Without this the parity gap
        # would be confounded by a different train/test split, which on these cohort sizes is
        # comfortably larger than the effect under test.
        rng = np.random.default_rng(SEED)
        results = {}
        print(f"\n--- arm: {arm} ({'neutral text' if neutral else 'full conditioning'}) ---",
              flush=True)
        for dataset, stream in EVAL_STREAMS:
            ref = refs.get((dataset, stream))
            if ref is None:
                continue
            data = ref.load_data()
            labels = np.asarray(ref.labels)
            subjects = np.asarray(ref.subjects)
            texts = stream_channel_descriptions(dataset, stream, neutral=neutral)
            z = encode_dataset(enc, data, texts, device, ref.rate_hz,
                               _stream_gravity_state(dataset, stream),
                               channel_mask=ref.mask, dataset=dataset, stream=stream,
                               neutral_text=neutral)

            # subject-disjoint 50/50 split
            subj = sorted(set(subjects.tolist()))
            rng.shuffle(subj)
            hold = set(subj[: max(1, len(subj) // 2)])
            tr = [i for i in range(len(data)) if subjects[i] not in hold]
            te = [i for i in range(len(data)) if subjects[i] in hold]
            ba = knn_balanced_acc(z[tr], labels[tr].tolist(), z[te], labels[te].tolist())
            n_lab = len(set(labels.tolist()))
            results[dataset] = {"knn_ba": round(ba, 4), "windows": len(data),
                                "labels": n_lab, "chance": round(1 / n_lab, 3)}
            print(f"  {dataset:14s} kNN-BA={ba:.3f}  ({n_lab} labels, chance {1/n_lab:.3f}, "
                  f"{len(data)} windows)", flush=True)
        arm_results[arm] = results
        arm_means[arm] = float(np.mean([r["knn_ba"] for r in results.values()]))

    results, mean = arm_results["full"], arm_means["full"]
    print(f"\nHELD-OUT-CONFIG TRANSFER: mean kNN-BA = {mean:.3f} across {len(results)} datasets")
    payload = {"checkpoint": str(args.checkpoint), "step": ckpt["step"],
               "internal_val_ba": ckpt["val_ba"], "per_dataset": results,
               "mean_knn_ba": round(mean, 4)}
    if args.parity:
        gap = mean - arm_means["parity"]
        payload["parity"] = {
            "per_dataset": arm_results["parity"],
            "mean_knn_ba": round(arm_means["parity"], 4),
            "conditioning_gain": round(gap, 4),
            "per_dataset_gain": {d: round(results[d]["knn_ba"]
                                          - arm_results["parity"][d]["knn_ba"], 4)
                                 for d in results},
        }
        print(f"PARITY (neutral text):        mean kNN-BA = {arm_means['parity']:.3f}")
        print(f"CONDITIONING GAIN (full - parity) = {gap:+.4f}")
        print("  Interpretation: this gap is the measured value of acquisition-config "
              "conditioning.\n  A gain of ~0 means the placement/device/gravity text is inert.")
    out = args.out or (args.checkpoint.parent / "transfer_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
