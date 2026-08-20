"""What the model costs, at several sizes, next to what the baselines cost.

The design constraint is parameter parity with the models HALO is compared against — ideally below
them. This script is the check. It builds the real modules rather than reasoning about them, so the
numbers cannot drift away from what actually gets trained.

Baseline counts are measured from the released checkpoints where those are resident on this
machine, and marked as unmeasured where they are not. Frozen text towers are reported in their own
column: a text-conditioned model carries one and it is not capacity the model learned.

    python -m eval.model_budget
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.evidence_mixer import EvidenceMixerConfig
from model.tokenizer.encoder import SetTokenizerEncoder

# (label, sensor-tower parameters, frozen text tower, how it was obtained)
BASELINES = [
    ("LIMU-BERT", 62_646, 0, "baselines/limubert/limubert_backbone.pt"),
    ("CrossHAR", 62_646, 0, "baselines/crosshar/crosshar_backbone.pt"),
    ("UniMTS", 5_189_956, 63_428_097, "UniMTS.pth (sensor tower = `acc`, text = CLIP)"),
    ("HARNet5", None, 0, "checkpoint not resident; torch.hub ResNet-V2 1D"),
    ("NormWear", None, None, "checkpoint not resident; carries a clinical TinyLlama"),
    ("ImageBind", 1_200_786_990, 0, "imagebind_huge.pth, all modalities"),
]


def build(d_model: int, trunk_layers: int, mixer_layers: int, top_k: int = 64,
          readout: str = "weights", n_heads: int = 4, ffn_mult: int = 2) -> EvidenceEngine:
    spec = AttentionSpec(d_model=d_model, n_heads=n_heads, ffn_mult=ffn_mult, dropout=0.1)
    encoder = SetTokenizerEncoder(
        d_model=spec.d_model, num_layers=trunk_layers, num_heads=spec.n_heads,
        dim_feedforward=spec.dim_feedforward, dropout=spec.dropout,
        token_granularity="sensor", text_conditioning="factored",
        trunk="temporal", descriptor_prediction=False,
    )
    cfg = EngineConfig(
        spec=spec, trunk_layers=trunk_layers, top_k=top_k,
        mixer=EvidenceMixerConfig(n_layers=mixer_layers, readout=readout),
    )
    return EvidenceEngine(encoder, cfg)


def main() -> int:
    grid = [
        ("compact", 96, 2, 2),
        ("small", 128, 3, 2),
        ("medium", 160, 3, 2),
        ("wide", 192, 4, 3),
    ]
    print("HALO evidence engine — learnable parameters by part\n")
    header = f"{'size':10s} {'d':>4s} {'trunk':>6s} {'mix':>4s} " + " ".join(
        f"{name:>12s}" for name in
        ("front end", "trunk", "scorer", "mixer", "TOTAL")
    )
    print(header)
    print("-" * len(header))
    for name, d_model, trunk_layers, mixer_layers in grid:
        engine = build(d_model, trunk_layers, mixer_layers)
        report = engine.parameter_report()
        front = sum(v for k, v in report.items()
                    if k.startswith("encoder.") and k != "encoder.transformer")
        row = (f"{name:10s} {d_model:>4d} {trunk_layers:>6d} {mixer_layers:>4d} "
               f"{front:>12,} {report['encoder.transformer']:>12,} "
               f"{report['scorer']:>12,} {report['mixer']:>12,} {report['TOTAL']:>12,}")
        print(row)

    weights = build(128, 3, 2, readout="weights").parameter_report()
    semantic = build(128, 3, 2, readout="semantic").parameter_report()
    print(f"\nreadout='weights'  mixer {weights['mixer']:,}  TOTAL {weights['TOTAL']:,}")
    print(f"readout='semantic' mixer {semantic['mixer']:,}  TOTAL {semantic['TOTAL']:,}")

    engine = build(128, 3, 2)
    print(f"frozen text encoder (all-MiniLM-L6-v2, shared, never trained): "
          f"{engine.frozen_text_parameters():,}")

    print("\nBaselines, for the parity check\n")
    print(f"{'model':14s} {'sensor tower':>14s} {'frozen text':>14s}  source")
    print("-" * 78)
    for label, sensor, text, source in BASELINES:
        sensor_s = f"{sensor:,}" if sensor is not None else "not measured"
        text_s = f"{text:,}" if text is not None else "not measured"
        print(f"{label:14s} {sensor_s:>14s} {text_s:>14s}  {source}")

    small = build(128, 3, 2).parameter_report()["TOTAL"]
    print(f"\nHALO 'small' sensor-side total: {small:,} "
          f"({small / 5_189_956:.2f}x UniMTS's sensor tower, "
          f"{small / 62_646:.0f}x LIMU-BERT)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
