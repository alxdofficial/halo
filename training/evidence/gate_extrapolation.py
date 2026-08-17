"""Can the admissibility gate extrapolate to configurations and concepts it never saw?

THE EXPERIMENT THAT DECIDES THE DESIGN. ``admissibility_gate`` parameterises admissibility as a
low-rank bilinear form over frozen text so that an unseen sensor and an unseen label both produce a
value. Producing a value is trivial; producing a CORRECT one is the claim, and it is testable right
now against the measured table without training anything else.

FOUR SPLITS, IN INCREASING ORDER OF HONESTY
-------------------------------------------
  concept    hold out whole concepts, keep every stream    -> tests the concept projection B
  stream     hold out whole streams                        -> weak: a sibling stream from the same
                                                              study usually remains, so the model can
                                                              lean on a near-duplicate
  placement  hold out every stream at a body location      -> TESTS THE THESIS. "unseen acquisition
                                                              configuration" is precisely this, and it
                                                              is the split nobody runs
  dataset    hold out whole datasets                       -> hardest; also removes the shared
                                                              hardware and protocol, so a drop here
                                                              versus `placement` isolates provenance

``placement`` and ``dataset`` group streams using curated metadata. That is legitimate for an
EVALUATION protocol — it decides which rows are held out — and it is never an input to the gate,
which sees only the sensor's own text description.

THE BASELINE THAT COUNTS IS PER-CONCEPT
---------------------------------------
A global training mean is reported, but it is weak: a model beats it by learning only that some
activities are easier than others. The load-bearing comparison is the training-fold mean for each
concept. A gate must beat that baseline on held-out cells to show configuration-dependent skill.
Affine recalibration is fitted on the training fold only.

Run:
    python -m training.evidence.gate_extrapolation --split placement
    python -m training.evidence.gate_extrapolation --split all --ranks 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.evidence.admissibility_gate import (
    DEFAULT_RANK, TableObservations, fit_gate, observations_from_table, predict_cells,
)
from training.evidence.admissibility_text import (
    embed_text_views, flatten_training_views, observation_text_views,
)

OUT_PATH = Path("training/evidence/outputs/gate_extrapolation.json")
SEED = 20260812


# Body regions for the held-out split. Raw placement strings are far too fine-grained to hold out
# meaningfully: "left wrist", "right wrist", "the wrist", "wrist", "dominant wrist" and "the dominant
# wrist" are six distinct strings, so leaving one out keeps five near-duplicates in training and the
# split measures paraphrase robustness rather than configuration extrapolation. Eight pocket variants
# have the same problem. Checked in order, first match wins.
_BODY_REGIONS = (
    ("head", ("ear", "glasses", "head")),
    ("wrist", ("wrist",)),
    ("forearm", ("forearm",)),
    ("upper_arm", ("upper arm",)),
    ("hand", ("hand",)),
    ("torso", ("chest", "torso", "sternum")),
    ("back", ("back",)),
    ("waist", ("hip", "waist", "belt")),
    ("thigh", ("pocket", "thigh", "rectus femoris", "hamstring")),
    ("knee", ("knee",)),
    ("shank", ("shin", "calf", "gastrocnemius", "tibialis")),
    ("ankle", ("ankle", "foot")),
)


def _placement_of(key: str) -> str:
    """Raw placement string for a stream — EVALUATION GROUPING ONLY, never seen by the gate."""
    stream_key = key.partition("::")[0]
    dataset, _, stream = stream_key.partition("/")
    try:
        from data.scripts.curate.deployment_policy import get_stream_spec
        return get_stream_spec(dataset, stream).placement
    except Exception:                                              # noqa: BLE001
        return stream


def _body_region(key: str) -> str:
    """Coarse body region, so holding one out actually removes a body LOCATION from training."""
    text = _placement_of(key).lower()
    for region, keywords in _BODY_REGIONS:
        if any(word in text for word in keywords):
            return region
    return f"other:{text}"


def _group_key(obs: TableObservations, i: int, split: str) -> str:
    key = obs.stream_key[i]
    if split == "concept":
        return obs.concept[i]
    if split == "stream":
        return key
    if split == "placement":
        return _body_region(key)
    if split == "dataset":
        return key.partition("/")[0]
    raise ValueError(f"unknown split {split!r}")


def _embed(texts: list[str]) -> torch.Tensor:
    from eval.scoring import get_sbert_encoder
    sbert = get_sbert_encoder()
    uniq = list(dict.fromkeys(texts))
    emb = torch.from_numpy(np.asarray(sbert(uniq), dtype=np.float32))
    index = {t: i for i, t in enumerate(uniq)}
    return emb[torch.tensor([index[t] for t in texts])]


def evaluate(obs: TableObservations, sensor_emb: torch.Tensor, concept_emb: torch.Tensor,
             split: str, rank: int, mode: str, seed: int = SEED) -> dict:
    """Leave-one-group-out over every group, pooling held-out predictions before scoring.

    Pooling rather than averaging per-fold scores: some groups contribute two cells and some contribute
    thirty, and a per-fold mean would weight a two-cell placement as heavily as a thirty-cell one.
    """
    groups = [_group_key(obs, i, split) for i in range(len(obs))]
    unique = sorted(set(groups))
    if len(unique) < 3:
        return {"split": split, "rank": rank, "mode": mode, "n_groups": len(unique),
                "note": "fewer than 3 groups — leave-one-group-out is not meaningful"}
    group_arr = np.asarray(groups)
    if sensor_emb.ndim == 2:
        sensor_emb = sensor_emb.unsqueeze(1)
        concept_emb = concept_emb.unsqueeze(1)
    if sensor_emb.shape != concept_emb.shape or sensor_emb.ndim != 3:
        raise ValueError("text embeddings must have matching (cell, view, dim) shapes")
    target = torch.from_numpy(obs.value)

    concept_arr = np.asarray(obs.concept)
    pooled_pred, pooled_truth, pooled_baseline, pooled_concept_baseline = [], [], [], []
    pooled_view_std = []
    for held in unique:
        test = np.flatnonzero(group_arr == held)
        train = np.flatnonzero(group_arr != held)
        if test.size == 0 or train.size < 20:
            continue
        tr = torch.from_numpy(train)
        te = torch.from_numpy(test)
        try:
            fit_sensor, fit_concept, fit_target = flatten_training_views(
                sensor_emb[tr], concept_emb[tr], target[tr]
            )
            gate = fit_gate(
                fit_sensor, fit_concept, fit_target, rank=rank, mode=mode, seed=seed,
            )
        except ValueError:
            continue                                               # rank exceeded the fold's support
        # Abstention OFF for scoring. It would blend held-out configurations toward neutral, which is
        # the RIGHT deployment behaviour but would flatter this measurement — the question here is
        # whether the projection extrapolates at all, not whether the fallback is sane.
        test_sensor = sensor_emb[te]
        test_concept = concept_emb[te]
        pred_views = predict_cells(
            gate,
            test_sensor.reshape(-1, test_sensor.shape[-1]),
            test_concept.reshape(-1, test_concept.shape[-1]),
            abstain=False,
        ).reshape(len(te), sensor_emb.shape[1]).numpy()
        pred = pred_views.mean(axis=1)
        pooled_view_std.extend(pred_views.std(axis=1).tolist())
        # Affine recalibration fitted on TRAIN only, so the held-out set stays honest. Separates
        # "no signal" from "correct ordering on the wrong scale".
        train_sensor = sensor_emb[tr]
        train_concept = concept_emb[tr]
        tr_pred = predict_cells(
            gate,
            train_sensor.reshape(-1, train_sensor.shape[-1]),
            train_concept.reshape(-1, train_concept.shape[-1]),
            abstain=False,
        ).reshape(len(tr), sensor_emb.shape[1]).mean(dim=1).numpy()
        tr_truth = obs.value[train]
        var = float(((tr_pred - tr_pred.mean()) ** 2).sum())
        slope = (float(((tr_pred - tr_pred.mean()) * (tr_truth - tr_truth.mean())).sum()) / var
                 if var > 1e-8 else 0.0)
        intercept = float(tr_truth.mean() - slope * tr_pred.mean())
        pooled_pred.extend(np.clip(slope * pred + intercept, 0.0, 1.0).tolist())
        pooled_truth.extend(obs.value[test].tolist())
        pooled_baseline.extend([float(tr_truth.mean())] * test.size)
        # THE BASELINE THAT ACTUALLY CONTROLS FOR THE CONFOUND. A global constant is beaten by
        # anything that learns "some activities are simply easier to recognise than others" — which
        # requires no notion of configuration at all, and is not what the gate claims. The per-concept
        # training mean already knows that, so beating IT is the only evidence of configuration-
        # dependent structure. Concepts absent from the training fold fall back to the global mean.
        concept_mean = {c: float(tr_truth[concept_arr[train] == c].mean())
                        for c in set(concept_arr[train].tolist())}
        pooled_concept_baseline.extend(
            [concept_mean.get(c, float(tr_truth.mean())) for c in concept_arr[test]])

    if len(pooled_truth) < 10:
        return {"split": split, "rank": rank, "mode": mode,
                "note": "too few held-out cells scored"}
    pred = np.asarray(pooled_pred)
    truth = np.asarray(pooled_truth)
    base = np.asarray(pooled_baseline)
    concept_base = np.asarray(pooled_concept_baseline)
    mse = float(np.mean((pred - truth) ** 2))
    mse_base = float(np.mean((base - truth) ** 2))
    mse_concept = float(np.mean((concept_base - truth) ** 2))
    r = (float(np.corrcoef(pred, truth)[0, 1])
         if pred.std() > 0 and truth.std() > 0 else float("nan"))
    # Residual correlation after removing each concept's own difficulty from BOTH sides. This is the
    # configuration-dependent signal in isolation — the only part the contribution claims.
    resid_pred = pred - concept_base
    resid_truth = truth - concept_base
    r_resid = (float(np.corrcoef(resid_pred, resid_truth)[0, 1])
               if resid_pred.std() > 0 and resid_truth.std() > 0 else float("nan"))
    return {
        "split": split, "rank": rank, "mode": mode,
        "n_groups": len(unique), "n_held_out_cells": int(len(truth)),
        "n_text_views_per_cell": int(sensor_emb.shape[1]),
        "paraphrase_prediction_std_mean": round(float(np.mean(pooled_view_std)), 5),
        "mse": round(mse, 5),
        "baseline_mse_constant": round(mse_base, 5),
        "baseline_mse_per_concept_mean": round(mse_concept, 5),
        "skill": round(1.0 - mse / mse_base, 4) if mse_base > 0 else None,
        "skill_vs_concept_mean": round(1.0 - mse / mse_concept, 4) if mse_concept > 0 else None,
        "pearson_r": round(r, 4) if np.isfinite(r) else None,
        "residual_pearson_r": round(r_resid, 4) if np.isfinite(r_resid) else None,
        "verdict": ("beats the per-concept mean — configuration-dependent structure"
                    if mse < mse_concept else
                    "NO CONFIGURATION SKILL — per-concept difficulty explains it as well or better"),
    }


def recommended_rank(rows: list[dict]) -> int | None:
    """Select capacity from the two splits that remove acquisition configurations.

    This is a report recommendation, not an automatic mutation of the model configuration.  A rank
    must have valid full-projection results on both placement and dataset holdouts; ties prefer the
    smaller model.
    """
    by_rank: dict[int, list[float]] = {}
    for row in rows:
        if row.get("mode") != "full" or row.get("split") not in {"placement", "dataset"}:
            continue
        score = row.get("skill_vs_concept_mean")
        if score is not None:
            by_rank.setdefault(int(row["rank"]), []).append(float(score))
    eligible = {rank: values for rank, values in by_rank.items() if len(values) == 2}
    if not eligible:
        return None
    return max(eligible, key=lambda rank: (float(np.mean(eligible[rank])), -rank))


def main() -> None:
    from training.evidence.resolvability import load as load_table

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="all",
                    choices=("all", "concept", "stream", "placement", "dataset"))
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, DEFAULT_RANK])
    ap.add_argument("--modes", nargs="+", default=["pca", "full"], choices=["pca", "full"])
    ap.add_argument("--text-views", type=int, default=4,
                    help="equal-weight canonical/paraphrased views per physical cell")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    table = load_table()
    obs = observations_from_table(table)
    if len(obs) < 50:
        raise SystemExit(f"only {len(obs)} usable cells — nothing to extrapolate from")
    views = observation_text_views(obs, count=args.text_views, seed=SEED)
    sensor_emb, concept_emb = embed_text_views(views, _embed)
    print(f"{len(obs)} cells · {len(set(obs.stream_key))} sensors · "
          f"{len(set(obs.concept))} concepts", flush=True)

    splits = (["concept", "stream", "placement", "dataset"] if args.split == "all"
              else [args.split])
    rows = [evaluate(obs, sensor_emb, concept_emb, split, rank, mode)
            for split in splits for mode in args.modes for rank in args.ranks]
    selected_rank = recommended_rank(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_cells": len(obs), "text_views_per_cell": args.text_views,
        "recommended_rank": selected_rank, "rank_selection_splits": ["placement", "dataset"],
        "runs": rows,
    }, indent=2))

    print("\n=== GATE EXTRAPOLATION — held-out groups ===")
    print(f"{'split':<11}{'mode':<6}{'rank':>5}{'cells':>7}"
          f"{'skill/const':>12}{'skill/concept':>14}{'r':>8}{'r_resid':>9}{'text_sd':>9}")
    for row in rows:
        if "skill" not in row:
            print(f"{row['split']:<11}{row.get('mode',''):<6}{row.get('rank',''):>5}"
                  f"        {row.get('note','')}")
            continue
        print(f"{row['split']:<11}{row['mode']:<6}{row['rank']:>5}{row['n_held_out_cells']:>7}"
              f"{row['skill']:>+12.3f}{row['skill_vs_concept_mean']:>+14.3f}"
              f"{row['pearson_r']:>+8.3f}{row['residual_pearson_r']:>+9.3f}"
              f"{row['paraphrase_prediction_std_mean']:>9.3f}")
    print("\n  skill/const   : versus one global constant. Beaten by anything that learns which")
    print("                  ACTIVITIES are easy, which needs no notion of configuration at all.")
    print("  skill/concept : versus the per-concept training mean. THE ONE THAT COUNTS — only a")
    print("                  positive value here is evidence of configuration-dependent structure.")
    print("  r_resid       : correlation after removing each concept's own difficulty from both")
    print("                  sides, i.e. the inversion structure in isolation.")
    print("\n  `placement` is the row that matters: it is the unseen-acquisition-configuration claim.")
    print(f"  recommended rank from full-mode placement+dataset holdouts: {selected_rank}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
