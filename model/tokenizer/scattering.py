"""Frontend selection for the time->frequency stage (M1).

One flag chooses the frontend so ablations compare like-for-like
(EVIDENCE_ENGINE.md §7 / build plan M1):

  fixed      — PhysicalFilterbankTokenizer, frozen physical-Hz constant-Q centers.
               THE DEFAULT until an ablation earns a switch.
  learnable — the same physical filterbank with the bounded adaptive parameters documented
              in docs/design/LEARNABLE_TOKENIZER_ARM.md.
  sincnet   — backward-compatible alias for the learnable arm.
  scattering — fixed wavelet-scattering first order (deformation-stability north star).
               NOT YET IMPLEMENTED — deferred ablation.
  free_conv  — unconstrained conv frontend. Deliberately NOT implemented (M0/M1 design:
               free convs overfit acquisition configs); exists as a name so the ablation
               table has an explicit "we chose not to" row.
"""

from __future__ import annotations

from .filterbank import PhysicalFilterbankTokenizer

FRONTENDS = ("fixed", "learnable", "sincnet", "scattering", "free_conv")


def build_frontend(kind: str = "fixed", **filterbank_kwargs) -> PhysicalFilterbankTokenizer:
    """Build the time->frequency frontend. Default: the fixed physical filterbank."""
    if kind == "fixed":
        return PhysicalFilterbankTokenizer(learnable=False, **filterbank_kwargs)
    if kind in ("learnable", "sincnet"):
        return PhysicalFilterbankTokenizer(learnable=True, **filterbank_kwargs)
    if kind == "scattering":
        raise NotImplementedError(
            "scattering frontend is a deferred ablation (build plan M1) — "
            "implement against the M0 probe's time-warp stability check before enabling."
        )
    if kind == "free_conv":
        raise NotImplementedError(
            "free_conv is deliberately not implemented: unconstrained conv frontends "
            "overfit acquisition configs (EVIDENCE_ENGINE.md §7). Use 'fixed' or 'sincnet'."
        )
    raise ValueError(f"unknown frontend {kind!r}; choose one of {FRONTENDS}")
