"""Magnitude + gradient-flow diagnostic for the Phase-1 model (run before training).

Checks the two failure modes that silently wreck pretraining:
  * ACTIVATION MAGNITUDE imbalance — if the frozen text embeddings enter fusion at a
    wildly different scale than the sensor tokens, the gated sum is dominated by one
    side and the other's gradient starves. Reports RMS at every stage.
  * GRADIENT FLOW — per-module grad norms after one real backward; vanishing/exploding
    across transformer depth; dead parameters (grad exactly 0); the mask-token and
    frozen-text invariants (text LM must have NO grad; everything else must).

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.grad_check
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
from training.tokenizer.losses_repr import (
    GroundingTargets, elite3_loss, make_mask_plan,
)
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig, align_batch
from training.tokenizer.pretrain_data import (
    CorpusIndex, MultiScaleCollate, PretrainDataset,
)

OUT = Path(__file__).resolve().parent / "outputs" / "grad_check"
GYRO_IDX = [3, 4, 5]


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PretrainConfig(device=str(device))

    index = CorpusIndex(max_per_stream=400, seed=1)
    ds = PretrainDataset(index, index.train, augment=True)
    collate = MultiScaleCollate(fixed_patch_seconds=1.0, seed=1)
    keys = list(range(256))
    batch = align_batch(collate([ds[i] for i in keys]))

    target_tok = PhysicalFilterbankTokenizer(d_model=1, dft_size=256)
    target_tok.proj = nn.Identity()
    target_tok.reset_norm_accumulator()
    target_tok.accumulate_norm_stats(batch["patches"], batch["rates"], batch["patch_len"])
    target_tok.finalize_norm_stats()
    target_tok.eval().to(device)
    for p in target_tok.parameters():
        p.requires_grad_(False)

    model = PipelineAModel(cfg, a1_target_dim=target_tok.in_dim).to(device)
    for buf in ("norm_mu", "norm_sd", "dc_mu", "dc_sd"):
        getattr(model.encoder.filterbank, buf).copy_(getattr(target_tok, buf))

    patches = batch["patches"].to(device)
    rates = batch["rates"].to(device)
    plen = batch["patch_len"].to(device)
    pos = batch["positions"].to(device)
    cmask = batch["channel_mask"].to(device)
    labels = batch["labels"].to(device)
    B, P, _, C = patches.shape

    # ---------------------------------------------------------------- magnitudes
    mags: dict[str, float] = {}
    enc = model.encoder
    sensor = enc.tokenize(patches, rates, plen)
    mags["sensor_tokens(filterbank)"] = rms(sensor)
    text_embs, text_masks = enc.encode_texts(batch["texts"], device)
    mags["text_embeddings(raw MiniLM)"] = rms(text_embs)
    # fusion internals: projected text + channel embeddings vs sensor
    proj_text = enc.fusion.text_proj(text_embs.reshape(B * C, text_embs.shape[2], -1))
    mags["text_after_proj"] = rms(proj_text)
    fused = enc.fusion(sensor, text_embs, text_masks)
    mags["fused_tokens"] = rms(fused)
    mags["fusion_delta(fused-sensor)"] = rms(fused - sensor)
    h = enc.transformer(fused, channel_mask=cmask, positions=pos)
    mags["transformer_out"] = rms(h)
    out_full = enc.encode(sensor, text_embs, text_masks, pos, channel_mask=cmask)
    mags["pooled"] = rms(out_full["pooled"])
    mags["a2_proj_out"] = rms(model.a2_proj(out_full["pooled"]))

    fusion_ratio = mags["fusion_delta(fused-sensor)"] / (mags["sensor_tokens(filterbank)"] + 1e-9)

    # ---------------------------------------------------------------- gradient flow
    plan = make_mask_plan(B, P, C, GYRO_IDX, device=device)
    a1_mask = plan.token_mask & cmask.unsqueeze(1)
    with torch.no_grad():
        a1_target = target_tok(patches, rates, plen)
    st = enc.tokenize(patches, rates, plen)
    te, tm = enc.encode_texts(batch["texts"], device)
    masked = enc.encode(st, te, tm, pos, token_mask=plan.token_mask, channel_mask=cmask)
    clean = enc.encode(st, te, tm, pos, channel_mask=cmask)
    z = model.a2_proj(clean["pooled"])
    targets = GroundingTargets(
        batch["cadence_target"].to(device), batch["cadence_valid"].to(device),
        batch["eigen_target"].to(device), batch["eigen_valid"].to(device),
    )
    loss = elite3_loss(
        model.a1_head(masked["tokens"]), a1_target, a1_mask, z, labels,
        model.a3_cadence(clean["pooled"]).squeeze(1),
        model.a3_eigen(clean["pooled"]).view(B, 4, 3), targets,
    )
    model.zero_grad()
    loss.total.backward()

    # per-module grad norms
    def module_grad_norm(mod: nn.Module) -> float:
        gs = [p.grad.detach().float().norm() ** 2 for p in mod.parameters()
              if p.grad is not None]
        return float(torch.stack(gs).sum().sqrt()) if gs else 0.0

    grad = {
        "filterbank.proj": module_grad_norm(enc.filterbank.proj),
        "mask_token": float(enc.mask_token.grad.norm()) if enc.mask_token.grad is not None else 0.0,
        "fusion": module_grad_norm(enc.fusion),
        "a1_head": module_grad_norm(model.a1_head),
        "a2_proj": module_grad_norm(model.a2_proj),
        "a3_cadence": module_grad_norm(model.a3_cadence),
        "a3_eigen": module_grad_norm(model.a3_eigen),
    }
    # per transformer layer (vanishing/exploding across depth)
    layer_grads = [round(module_grad_norm(layer), 5)
                   for layer in enc.transformer.layers]
    grad["transformer_layers(shallow->deep)"] = layer_grads

    # invariants
    text_lm_grads = [p.grad is not None for p in enc.text_encoder.parameters()]
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.norm()) == 0.0)]

    report = {
        "device": str(device), "batch": [B, P, C],
        "loss_parts": loss.parts,
        "activation_rms": {k: round(v, 4) for k, v in mags.items()},
        "fusion_delta_ratio": round(fusion_ratio, 3),
        "grad_norms": {k: (v if isinstance(v, list) else round(v, 5))
                       for k, v in grad.items()},
        "checks": {
            "text_lm_has_no_grad": not any(text_lm_grads),
            "no_dead_trainable_params": len(dead) == 0,
            "dead_params": dead,
            "fusion_balanced(0.1<ratio<10)": 0.1 < fusion_ratio < 10,
            "layer_grads_monotone_healthy(no >20x jump)":
                all(layer_grads[i] == 0 or 0.05 < layer_grads[i + 1] / (layer_grads[i] + 1e-9) < 20
                    for i in range(len(layer_grads) - 1)),
            "all_activations_finite": all(v == v and abs(v) < 1e4 for v in mags.values()),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    verdict = all(report["checks"][k] for k in report["checks"]
                  if isinstance(report["checks"][k], bool))
    print(f"\nGRAD/MAGNITUDE CHECK: {'PASS' if verdict else 'ISSUES — see checks'}")


if __name__ == "__main__":
    main()
