"""The bank row type, shared by the model and by everything that builds a memory bank.

This lives under ``model/`` rather than ``training/`` because it is part of the model's input
contract: an :class:`~model.evidence.engine.EvidenceEngine` consumes ``SensorRows`` and nothing
else. The trainer, the offline bank builder and the deployment runtime each construct them their
own way, and the engine cannot tell which one it is looking at — which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SensorRows:
    """A bank slice, keyed per patch per sensor. All tensors share a leading row dimension R."""

    feature: torch.Tensor          # (R, d)   signal embedding
    descriptor: torch.Tensor       # (R, 384) frozen SBERT of the sensor text
    bias: torch.Tensor             # (R, F)   sensor_bias
    modality: torch.Tensor         # (R,)     0=accel, 1=gyro
    gravity: torch.Tensor          # (R,)     0=present, 1=removed
    label: torch.Tensor            # (R,)     vocab index, -1 = unlabelled corpus row
    dataset: torch.Tensor          # (R,)     provenance only, never scored
    enrolled_candidate: torch.Tensor  # (R,)  candidate slot this row is bound to, -1 = corpus row
    # Which RECORDING each row came from. Rows sharing a value are the same source window seen
    # through different patches or sensors. Only the evidence mixer reads it, as its co-membership
    # channel; every closed-form rule ignores it. None means the provenance is unknown, which is a
    # legitimate state for hand-built row sets and a hard error for the mixer, which would otherwise
    # silently treat every row as its own recording.
    source_window: torch.Tensor | None = None

    def __post_init__(self) -> None:
        R = self.feature.shape[0]
        for name in ("descriptor", "bias", "modality", "gravity", "label", "dataset",
                     "enrolled_candidate", "source_window"):
            field = getattr(self, name)
            if field is None:
                continue
            got = field.shape[0]
            if got != R:
                raise ValueError(f"{name} has {got} rows, expected {R}")

    def to(self, device) -> "SensorRows":
        return SensorRows(**{
            name: None if getattr(self, name) is None else getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        })


def evidence_label_tokens(
    rows: SensorRows,
    candidate_text: torch.Tensor,
    label_text: torch.Tensor,
) -> torch.Tensor:
    """Attach canonical text to corpus rows and episode-local text to enrolled rows."""
    bound = rows.enrolled_candidate.to(candidate_text.device)
    labels = rows.label.to(label_text.device)
    if bool(bound.ge(candidate_text.shape[0]).any()):
        raise ValueError("an enrolled evidence row is bound outside the candidate set")
    if bool((bound.lt(0) & labels.lt(0)).any()):
        raise ValueError("an unbound corpus evidence row has no canonical label")
    canonical = label_text[labels.clamp_min(0)]
    enrolled = candidate_text[bound.clamp_min(0)]
    return torch.where(bound.unsqueeze(1).ge(0), enrolled, canonical)
