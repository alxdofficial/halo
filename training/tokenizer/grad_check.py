"""Activation and gradient-flow diagnostic for the live Phase-A model.

Runs one real CPU batch through JEPA and the unified relation objective. It checks fusion scale,
per-module gradients, frozen text parameters, and dead trainable parameters.

Run: /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.grad_check
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.tokenizer.losses_repr import (
    make_mask_plan,
    masked_ema_latent_loss,
    phase_a_loss,
    relation_loss,
)
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig
from training.tokenizer.pretrain_data import CorpusIndex, MultiScaleCollate, PretrainDataset

OUT = Path(__file__).resolve().parent / "outputs" / "grad_check"
GYRO = [3, 4, 5]


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def module_grad_norm(module: nn.Module) -> float:
    squares = [parameter.grad.detach().float().square().sum()
               for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Diagnostics must never take a shared GPU implicitly.
    device = torch.device("cpu")
    cfg = PretrainConfig(
        d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
        device=str(device), text_conditioning="factored",
    )
    index = CorpusIndex(max_per_stream=200, seed=1)
    dataset = PretrainDataset(index, index.train, augment=True, two_view=True)
    collate = MultiScaleCollate(fixed_patch_seconds=1.0, seed=1, two_view=True)
    batch = collate([dataset[i] for i in range(64)])

    model = PipelineAModel(cfg).to(device)
    frontend = model.encoder.filterbank
    frontend.reset_norm_accumulator()
    frontend.accumulate_norm_stats(
        batch["patches"].to(device), batch["rates"].to(device),
        batch["patch_len"].to(device),
        patch_mask=batch["patch_padding_mask"].to(device),
        channel_mask=batch["channel_mask"].to(device),
        source_rate_hz=batch["source_rates"].to(device),
    )
    frontend.finalize_norm_stats()

    def encode(suffix: str = "", token_mask=None):
        patches = batch[f"patches{suffix}"].to(device)
        rates = batch[f"rates{suffix}"].to(device)
        lengths = batch[f"patch_len{suffix}"].to(device)
        positions = batch[f"positions{suffix}"].to(device)
        channel_mask = batch[f"channel_mask{suffix}"].to(device)
        patch_mask = batch[f"patch_padding_mask{suffix}"].to(device)
        tokens = model.encoder.tokenize(
            patches, rates, lengths,
            source_rate_hz=batch[f"source_rates{suffix}"].to(device),
        )
        text, text_mask, sensor_text, sensor_text_mask = model.encoder.encode_texts_factored(
            batch[f"role_texts{suffix}"], batch[f"sensor_texts{suffix}"], device,
        )
        sensor_id = batch[f"sensor_id{suffix}"].to(device)
        output = model.encoder.encode(
            tokens, text, text_mask, positions, token_mask=token_mask,
            channel_mask=channel_mask, patch_padding_mask=patch_mask,
            sensor_text_embs=sensor_text, sensor_text_masks=sensor_text_mask,
            sensor_id=sensor_id,
        )
        return tokens, text, text_mask, sensor_text, sensor_text_mask, sensor_id, output

    tokens, text, text_mask, sensor_text, sensor_text_mask, sensor_id, clean = encode()
    batch_size, patches, channels = clean["tokens"].shape[:3]
    plan = make_mask_plan(
        batch_size, patches, channels, GYRO, device=device,
        valid_patches=batch["patch_padding_mask"].to(device),
        channel_mask=batch["channel_mask"].to(device),
    )
    *_, masked = encode(token_mask=plan.token_mask)
    *_, view_b = encode("_b")
    jepa_prediction = model.jepa_predictor(masked["tokens"])
    jepa = masked_ema_latent_loss(
        jepa_prediction, clean["tokens"].detach(), plan.token_mask,
    )
    z_a = model.relation_projector(clean["pooled"])
    z_b = model.relation_projector(view_b["pooled"])
    relation = relation_loss(z_a, z_b)
    loss = phase_a_loss(jepa, relation.total)

    model.zero_grad(set_to_none=True)
    loss.total.backward()

    encoder = model.encoder
    projected_text = encoder.fusion.pool.text_proj(
        text.reshape(batch_size * channels, text.shape[2], -1)
    )
    fused = encoder.fusion(
        tokens, text, text_mask, sensor_text, sensor_text_mask, sensor_id,
    )
    magnitudes = {
        "sensor_tokens": rms(tokens),
        "text_embeddings": rms(text),
        "projected_text": rms(projected_text),
        "fused_tokens": rms(fused),
        "fusion_delta": rms(fused - tokens),
        "pooled": rms(clean["pooled"]),
        "relation_projection": rms(z_a),
    }
    fusion_ratio = magnitudes["fusion_delta"] / max(magnitudes["sensor_tokens"], 1e-9)
    gradients = {
        "encoder": module_grad_norm(encoder),
        "jepa_predictor": module_grad_norm(model.jepa_predictor),
        "relation_projector": module_grad_norm(model.relation_projector),
        "transformer_layers": [module_grad_norm(layer) for layer in encoder.transformer.layers],
    }
    dead = [name for name, parameter in model.named_parameters()
            if parameter.requires_grad and (parameter.grad is None or not bool(parameter.grad.any()))]
    checks = {
        "text_encoder_frozen": not any(
            parameter.grad is not None for parameter in encoder.text_encoder.parameters()
        ),
        "no_dead_trainable_parameters": not dead,
        # Factored conditioning intentionally starts as a light residual (gate bias -2).
        "fusion_scale_reasonable": 0.03 < fusion_ratio < 10.0,
        "finite_activations": all(value == value and abs(value) < 1e4
                                  for value in magnitudes.values()),
        "finite_loss": bool(torch.isfinite(loss.total)),
    }
    report = {
        "device": str(device),
        "batch": [batch_size, patches, channels],
        "loss": {"jepa": float(jepa.detach()), "relation": float(relation.total.detach())},
        "activation_rms": {key: round(value, 5) for key, value in magnitudes.items()},
        "fusion_delta_ratio": round(fusion_ratio, 5),
        "gradient_norms": gradients,
        "dead_parameters": dead,
        "checks": checks,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"GRAD CHECK: {'PASS' if all(checks.values()) else 'ISSUES'}")


if __name__ == "__main__":
    main()
