"""Predict resolvability for an UNSEEN (placement, concept) pair, from text.

THE PIECE THAT MAKES THE GATE DEPLOYABLE. ``resolvability.py`` MEASURES, from signal, whether a
configuration can witness a concept. That measurement is ground truth, but it is a lookup: at
deployment, a novel label — the whole point of an open vocabulary — has no measured entry, and
``gate_tensor`` falls back to a neutral 0.5.

This module closes that gap the only way it can be closed: learn the map

    (placement description, concept description)  ->  resolvability

from text embeddings, supervised by the measured table. Measurement and prediction are complementary,
not alternatives — the measurement is what supervises the prediction, and the prediction is what
generalises the measurement.

WHY THIS IS A FAIRER TEST OF LANGUAGE THAN THE ONE THAT FAILED
--------------------------------------------------------------
The parity ablation showed language contributes ~nothing to *identifying* a concept from signal
(+0.0086 kNN-BA), and Haresamudram et al. AAAI 2025 report the same at larger scale. But "which of
these 30 activities is this window" and "can a wrist-worn sensor observe jumping jacks" are different
questions. The second is about body mechanics — which limb moves, whether the sensor is on it —
and that IS the kind of relation language encodes well. A model that fails the first may pass the
second, and this is where a language interface could finally earn its place.

THE ONLY EVALUATION THAT MATTERS IS LABEL-DISJOINT
--------------------------------------------------
Splitting rows at random would let the model memorise "bicep_curls scores 1.0 at the wrist" from one
dataset and recall it for another — measuring lookup, not generalisation. ``--split concept`` holds
out entire CONCEPTS, so every test pair involves a label whose resolvability was never seen at any
placement. That is the deployment condition. A ``placement`` split (unseen body locations) is
available for the same reason.

The baseline to beat is the training-set MEAN. A predictor that cannot beat a constant has learned
nothing, however low its MSE looks.

Run:
    python -m training.evidence.resolvability_predictor --fit --split concept
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_PATH = Path("training/evidence/outputs/resolvability_predictor.json")
SEED = 20260812
EPOCHS = 400
LR = 3e-3
HIDDEN = 128


def _placement_text(dataset: str, stream: str) -> str:
    """Natural-language description of where the sensor sits — the gate's generalisable key."""
    try:
        from data.scripts.curate.deployment_policy import get_stream_spec
        spec = get_stream_spec(dataset, stream)
        place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
            else f"the {spec.placement}"
        return f"a {spec.device_profile.replace('_', ' ')} on {place}"
    except Exception:                                          # noqa: BLE001
        return stream.replace("_", " ")


def load_rows(table: dict) -> list[dict]:
    """Flatten the measured table into (placement text, concept, score) training rows."""
    rows = []
    for key, payload in table["per_stream"].items():
        text = _placement_text(payload["dataset"], payload["stream"])
        for concept, score in payload["labels"].items():
            rows.append({"stream": key, "dataset": payload["dataset"],
                         "placement_text": text, "concept": concept, "score": float(score)})
    return rows


class _Predictor(nn.Module):
    """Small MLP over [placement_emb, concept_emb, elementwise product].

    The product term is what lets the model express INTERACTION — "this concept at this placement" —
    rather than only additive effects like "wrists are good" or "curls are easy". Without it the
    model cannot represent the inversions the measurement found (pocket wins squats, wrist wins
    curls), which are the entire reason the gate needs a per-concept argument.
    """

    def __init__(self, dim: int = 384, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, place: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
        x = torch.cat([place, concept, place * concept], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)          # resolvability lives in [0,1]


def _embed(texts: list[str]) -> torch.Tensor:
    from eval.scoring import get_sbert_encoder
    sbert = get_sbert_encoder()
    uniq = list(dict.fromkeys(texts))
    emb = F.normalize(torch.from_numpy(sbert(uniq).astype(np.float32)), dim=-1)
    index = {t: i for i, t in enumerate(uniq)}
    return emb[torch.tensor([index[t] for t in texts])]


def fit(table: dict, split: str = "concept", seed: int = SEED) -> dict:
    rows = load_rows(table)
    if len(rows) < 40:
        raise ValueError(f"only {len(rows)} measured rows — too few to fit and hold out")

    rng = np.random.default_rng(seed)
    key = "concept" if split == "concept" else "placement_text"
    groups = sorted({r[key] for r in rows})
    rng.shuffle(groups)
    held = set(groups[: max(1, len(groups) // 4)])             # 25% of concepts/placements held out
    tr = [r for r in rows if r[key] not in held]
    te = [r for r in rows if r[key] in held]
    if not te or not tr:
        raise ValueError("the split left one side empty")

    place_emb = _embed([r["placement_text"] for r in rows])
    concept_emb = _embed([r["concept"].replace("_", " ") for r in rows])
    scores = torch.tensor([r["score"] for r in rows], dtype=torch.float32)
    idx = {id(r): i for i, r in enumerate(rows)}
    tr_i = torch.tensor([idx[id(r)] for r in tr])
    te_i = torch.tensor([idx[id(r)] for r in te])

    torch.manual_seed(seed)
    model = _Predictor(place_emb.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(place_emb[tr_i], concept_emb[tr_i])
        loss = F.mse_loss(pred, scores[tr_i])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(place_emb[te_i], concept_emb[te_i])
        pred_tr = model(place_emb[tr_i], concept_emb[tr_i])
    truth_te = scores[te_i]

    # AFFINE RECALIBRATION, fitted on TRAIN ONLY. Separates two very different failures that raw MSE
    # conflates: "the model has no signal" and "the model ranks correctly but on the wrong scale".
    # For a gate the ordering is what matters — it down-weights evidence RELATIVELY — so a
    # rank-correct but miscalibrated predictor is usable and a rank-blind one is not. Fitting the
    # rescale on train keeps the held-out set honest.
    a_num = ((pred_tr - pred_tr.mean()) * (scores[tr_i] - scores[tr_i].mean())).sum()
    a_den = ((pred_tr - pred_tr.mean()) ** 2).sum().clamp(min=1e-8)
    slope = a_num / a_den
    intercept = scores[tr_i].mean() - slope * pred_tr.mean()
    pred_te_cal = (slope * pred_te + intercept).clamp(0.0, 1.0)
    # THE BASELINE: a constant, the training mean. Anything that cannot beat this has learned
    # nothing about the (placement, concept) relation, however small its absolute error.
    baseline = scores[tr_i].mean().expand_as(truth_te)
    mse = float(F.mse_loss(pred_te, truth_te))
    mse_cal = float(F.mse_loss(pred_te_cal, truth_te))
    mse_base = float(F.mse_loss(baseline, truth_te))
    corr = (float(np.corrcoef(pred_te.numpy(), truth_te.numpy())[0, 1])
            if truth_te.numel() > 2 and truth_te.std() > 0 else float("nan"))
    return {
        "split": split,
        "n_train_rows": len(tr), "n_test_rows": len(te),
        "n_held_out_groups": len(held), "n_groups": len(groups),
        "test_mse": round(mse, 5),
        "test_mse_calibrated": round(mse_cal, 5),
        "baseline_mse_train_mean": round(mse_base, 5),
        "skill_score": round(1.0 - mse / mse_base, 4) if mse_base > 0 else None,
        "skill_score_calibrated": round(1.0 - mse_cal / mse_base, 4) if mse_base > 0 else None,
        "test_pearson_r": round(corr, 4) if np.isfinite(corr) else None,
        "verdict": ("language predicts resolvability for unseen "
                    f"{'concepts' if split == 'concept' else 'placements'}"
                    if mse_cal < mse_base else
                    "NO SKILL — a constant predicts held-out resolvability as well or better"),
    }


def main() -> None:
    from training.evidence.resolvability import load as load_table
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--split", choices=("concept", "placement"), default="concept")
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat over this many held-out draws; one split is noise, not a result")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    if not args.fit:
        ap.error("pass --fit")

    table = load_table()
    runs = [fit(table, args.split, SEED + i) for i in range(args.seeds)]
    skills = [r["skill_score_calibrated"] for r in runs
              if r["skill_score_calibrated"] is not None]
    raw_skills = [r["skill_score"] for r in runs if r["skill_score"] is not None]
    corrs = [r["test_pearson_r"] for r in runs if r["test_pearson_r"] is not None]
    summary = {
        "split": args.split,
        "n_seeds": len(runs),
        "mean_skill_score_calibrated": round(float(np.mean(skills)), 4) if skills else None,
        "mean_skill_score_raw": round(float(np.mean(raw_skills)), 4) if raw_skills else None,
        "mean_pearson_r": round(float(np.mean(corrs)), 4) if corrs else None,
        "skill_scores_calibrated": skills,
        "positive_seeds": int(sum(s > 0 for s in skills)),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== LEARNED RESOLVABILITY, held-out {args.split}s, {len(runs)} seeds ===")
    for r in runs:
        print(f"  mse {r['test_mse']:.4f} -> calibrated {r['test_mse_calibrated']:.4f} "
              f"vs constant {r['baseline_mse_train_mean']:.4f} "
              f"| skill {r['skill_score']:+.3f} -> {r['skill_score_calibrated']:+.3f} "
              f"| r {r['test_pearson_r']:+.3f}")
    print(f"\nmean Pearson r      : {summary['mean_pearson_r']:+.4f}  (does text get the ORDER right?)")
    print(f"mean skill, raw     : {summary['mean_skill_score_raw']:+.4f}")
    print(f"mean skill, calibr. : {summary['mean_skill_score_calibrated']:+.4f} "
          f"({summary['positive_seeds']}/{len(runs)} seeds positive)")
    print("  skill > 0 means text beats a constant at predicting whether a configuration can")
    print("  witness a concept it has never been measured on. skill <= 0 means it cannot, and the")
    print("  gate must stay a lookup with a neutral default for unseen pairs.")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
