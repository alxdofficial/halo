"""Activation and gradient-flow diagnostic for the canonical Phase-A sensor model.

Runs one real CPU batch through fixed-one-second sensor-granularity JEPA and rotation VICReg. It
checks conditioning scale, per-module gradients, frozen text parameters, and dead trainable
parameters in the reference recipe.

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
    make_sensor_mask_plan,
    masked_ema_latent_loss,
    phase_a_loss,
    vicreg,
)
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig
from training.tokenizer.pretrain_data import SEED, CorpusIndex, MultiScaleCollate, PretrainDataset

OUT = Path(__file__).resolve().parent / "outputs" / "grad_check"


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
        device=str(device), text_conditioning="factored", token_granularity="sensor",
        multiresolution=False, descriptor_weight=0.0,
    )
    index = CorpusIndex(max_per_stream=200, seed=SEED)
    # CorpusIndex is stream-ordered. Taking rows 0..63 would inspect Capture-24 accelerometer only,
    # making cross-sensor attention and descriptor retrieval structurally inactive. Stratify across
    # the whole index so every canonical module receives a meaningful opportunity for gradient.
    diagnostic_keys = [index.train[i] for i in np.linspace(
        0, len(index.train) - 1, 64, dtype=np.int64,
    )]
    dataset = PretrainDataset(index, diagnostic_keys, augment=True, two_view=True)
    collate = MultiScaleCollate(fixed_patch_seconds=1.0, seed=SEED, two_view=True)
    batch = collate([dataset[i] for i in range(len(diagnostic_keys))])

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

    def encode(suffix: str = "", token_mask=None, descriptor_mask=None):
        patches = batch[f"patches{suffix}"].to(device)
        rates = batch[f"rates{suffix}"].to(device)
        lengths = batch[f"patch_len{suffix}"].to(device)
        positions = batch[f"positions{suffix}"].to(device)
        channel_mask = batch[f"channel_mask{suffix}"].to(device)
        patch_mask = batch[f"patch_padding_mask{suffix}"].to(device)
        patch_durations = batch[f"patch_durations{suffix}"].to(device)
        tokens = model.encoder.tokenize(
            patches, rates, lengths,
            channel_mask=channel_mask,
            source_rate_hz=batch[f"source_rates{suffix}"].to(device),
            sensor_id=batch[f"sensor_id{suffix}"].to(device),
            n_sensors=max(map(len, batch[f"sensor_texts{suffix}"])),
        )
        sensor_descriptors, sensor_text_ids = model.encoder.encode_sensor_descriptors_unique(
            batch[f"sensor_texts{suffix}"], device,
        )
        text = text_mask = role_ids = sensor_text = sensor_text_mask = None
        sensor_id = batch[f"sensor_id{suffix}"].to(device)
        output = model.encoder.encode(
            tokens, text, text_mask, positions,
            patch_durations=patch_durations,
            token_mask=token_mask,
            channel_mask=channel_mask, patch_padding_mask=patch_mask,
            sensor_text_embs=sensor_text, sensor_text_masks=sensor_text_mask,
            sensor_descriptors=sensor_descriptors,
            sensor_id=sensor_id, role_text_ids=role_ids, sensor_text_ids=sensor_text_ids,
            descriptor_mask=descriptor_mask,
        )
        return (tokens, text, text_mask, sensor_text, sensor_text_mask, sensor_id,
                sensor_text_ids, output)

    (tokens, text, text_mask, sensor_text, sensor_text_mask, sensor_id,
     sensor_text_ids, clean) = encode()
    batch_size, patches, sensors = clean["tokens"].shape[:3]
    plan = make_sensor_mask_plan(
        batch_size, patches, sensors, device=device,
        valid_patches=batch["patch_padding_mask"].to(device),
        sensor_present=clean["sensor_present"],
        sensor_placement=batch["sensor_placement"].to(device),
    )
    *_, masked = encode(token_mask=plan.token_mask)
    *_, view_b = encode("_b")
    jepa_prediction = model.jepa_predictor(masked["tokens"])
    jepa_mask = (plan.token_mask & masked["sensor_present"].unsqueeze(1)
                 & batch["patch_padding_mask"].to(device).unsqueeze(2))
    jepa = masked_ema_latent_loss(
        jepa_prediction, clean["tokens"].detach(), jepa_mask,
        token_durations=batch["patch_durations"].to(device),
    )
    z_a = model.vicreg_projector(clean["pooled"])
    z_b = model.vicreg_projector(view_b["pooled"])
    pooled_vicreg = vicreg(z_a, z_b)
    row_valid = (batch["patch_padding_mask"].to(device).unsqueeze(2)
                 & batch["patch_padding_mask_b"].to(device).unsqueeze(2)
                 & clean["sensor_present"].unsqueeze(1)
                 & view_b["sensor_present"].unsqueeze(1))
    row_vicreg = vicreg(clean["retrieval_tokens"][row_valid],
                        view_b["retrieval_tokens"][row_valid])
    vicreg_total = 0.5 * (pooled_vicreg.total + row_vicreg.total)
    loss = phase_a_loss(jepa, vicreg_total)

    model.zero_grad(set_to_none=True)
    loss.total.backward()

    encoder = model.encoder
    if encoder.sensor_fold is None:
        folded = tokens
        present = encoder.filterbank.sensor_presence(
            sensor_id, batch["channel_mask"].to(device), sensors,
        )
    else:
        folded, present = encoder.sensor_fold(
            tokens, sensor_id, batch["channel_mask"].to(device), n_sensors=sensors,
        )
    descriptor_conditioned = encoder.descriptor_proj(folded, clean["descriptor"], present)
    magnitudes = {
        "channel_tokens": rms(tokens),
        "sensor_tokens": rms(folded),
        "text_descriptors": rms(clean["descriptor"]),
        "descriptor_conditioned": rms(descriptor_conditioned),
        "fully_conditioned": rms(descriptor_conditioned),
        "conditioning_delta": rms(descriptor_conditioned - folded),
        "pooled": rms(clean["pooled"]),
        "vicreg_projection": rms(z_a),
    }
    conditioning_ratio = magnitudes["conditioning_delta"] / max(
        magnitudes["sensor_tokens"], 1e-9,
    )
    gradients = {
        "encoder": module_grad_norm(encoder),
        "jepa_predictor": module_grad_norm(model.jepa_predictor),
        "vicreg_projector": module_grad_norm(model.vicreg_projector),
        "descriptor_projection": module_grad_norm(encoder.descriptor_proj),
        "descriptor_head": module_grad_norm(encoder.descriptor_head),
        "transformer_layers": [module_grad_norm(layer) for layer in encoder.transformer.layers],
    }
    if encoder.sensor_fold is not None:
        gradients["sensor_fold"] = module_grad_norm(encoder.sensor_fold)
    dead = [name for name, parameter in model.named_parameters()
            if parameter.requires_grad and (parameter.grad is None or not bool(parameter.grad.any()))]
    checks = {
        "text_encoder_frozen": not any(
            parameter.grad is not None for parameter in encoder.text_encoder.parameters()
        ),
        "no_dead_trainable_parameters": not dead,
        # Factored conditioning intentionally starts as a light residual (gate bias -2).
        "conditioning_scale_reasonable": 0.03 < conditioning_ratio < 10.0,
        "finite_activations": all(value == value and abs(value) < 1e4
                                  for value in magnitudes.values()),
        "finite_loss": bool(torch.isfinite(loss.total)),
    }
    report = {
        "device": str(device),
        "batch": [batch_size, patches, sensors],
        "loss": {"jepa": float(jepa.detach()), "vicreg": float(vicreg_total.detach()),
                 "vicreg_pooled": float(pooled_vicreg.total.detach()),
                 "vicreg_retrieval": float(row_vicreg.total.detach())},
        "activation_rms": {key: round(value, 5) for key, value in magnitudes.items()},
        "conditioning_delta_ratio": round(conditioning_ratio, 5),
        "gradient_norms": gradients,
        "dead_parameters": dead,
        "checks": checks,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"GRAD CHECK: {'PASS' if all(checks.values()) else 'ISSUES'}")


if __name__ == "__main__":
    main()
