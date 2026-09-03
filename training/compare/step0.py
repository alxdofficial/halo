"""Write the step-0 predictor: the trained system's own function before any update.

WHY THIS EXISTS
---------------
Every Phase-B comparison in this project's history that skipped this control was later revised.
Two measured reasons:

* the validation draw dominates the raw score — four replicates of one config that differed only
  by seed spread 0.068, five times the within-run scatter — so a raw number carries the draw, not
  the method. Paired against its own step 0, the same comparison tightens to 0.0069.
* "training helps" and "training hurts" were both concluded from unpaired numbers on this codebase,
  and both were wrong at least once.

So a run is reported as ``score(trained) - score(step 0 of that same run)``, and this module makes
the second term a real artifact rather than an assumption.

THE ONE PRECONDITION
--------------------
Paired gain is valid only when both arms share the same step-0 *function*. Changing the
architecture moves step 0, and then the two arms must be compared by raw score at matched seeds
instead. The identity assertion below is what lets us claim the shared function: at initialisation
the comparator IS the closed-form vote, exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits

IDENTITY_TOLERANCE = 1e-6


def assert_identity_at_init(
    comparator: SupportComparator,
    *,
    batch: int = 3,
    candidates: int = 5,
    support: int = 7,
    seed: int = 0,
) -> float:
    """Check the comparator equals the closed-form vote at init; return the largest gap."""

    generator = torch.Generator().manual_seed(seed)
    d = comparator.spec.d_model
    z = comparator.cfg.text_dim
    normal = lambda *shape: torch.randn(*shape, generator=generator)  # noqa: E731

    episode = {
        "candidate_text": torch.nn.functional.normalize(normal(batch, candidates, z), dim=-1),
        "query_feature": normal(batch, 2, d),
        "query_descriptor": torch.nn.functional.normalize(normal(batch, 2, z), dim=-1),
        "query_mask": torch.ones(batch, 2, dtype=torch.bool),
        "support_feature": normal(batch, support, d),
        "support_descriptor": torch.nn.functional.normalize(normal(batch, support, z), dim=-1),
        "support_label_text": torch.nn.functional.normalize(normal(batch, support, z), dim=-1),
        "support_bound": torch.randint(-1, candidates, (batch, support), generator=generator),
        "support_mask": torch.ones(batch, support, dtype=torch.bool),
        "candidate_slot": torch.arange(candidates).unsqueeze(0).expand(batch, candidates),
    }
    with torch.no_grad():
        learned = comparator_logits(comparator, **episode)["logits"]
        closed = comparator_logits(None, **episode)["logits"]
    gap = float((learned - closed).abs().max())
    if gap > IDENTITY_TOLERANCE:
        raise AssertionError(
            f"the comparator is not its own closed-form vote at initialisation (gap {gap:.3e}). "
            "The untrained floor and every paired step-0 comparison assume this equality; a "
            "non-zero residual head at init silently invalidates both."
        )
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, required=True,
                        help="the Phase-A checkpoint the paired run warm-starts from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # Written by our own Phase-A trainer; `config` holds plain Python values beside the tensors.
    checkpoint = torch.load(args.phase_a, map_location="cpu", weights_only=False)
    d_model = int(checkpoint["config"].get("d_model", 128))
    spec = AttentionSpec(d_model=d_model, n_heads=args.n_heads, ffn_mult=2, dropout=0.1)
    comparator = SupportComparator(spec, ComparatorConfig(n_layers=args.n_layers))

    gap = assert_identity_at_init(comparator)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": checkpoint["encoder"],
        "comparator": comparator.state_dict(),
        "comparator_config": dataclasses.asdict(comparator.cfg),
        "attention_spec": dataclasses.asdict(spec),
        "step": 0,
        "identity_gap": gap,
        "phase_a": str(args.phase_a),
    }, args.out / "step0.pt")
    (args.out / "step0.json").write_text(json.dumps({
        "step": 0,
        "identity_gap": gap,
        "tolerance": IDENTITY_TOLERANCE,
        "phase_a": str(args.phase_a),
        "d_model": d_model,
        "note": "the comparator at initialisation is exactly the closed-form support vote",
    }, indent=2) + "\n")
    print(f"[step0] identity gap {gap:.3e} (tolerance {IDENTITY_TOLERANCE:.0e}) -> {args.out}")


if __name__ == "__main__":
    main()
