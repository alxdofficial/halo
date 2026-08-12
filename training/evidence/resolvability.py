"""``resolvability`` — can a given acquisition configuration witness a given concept?

THE MEASUREMENT THE CONTRIBUTION RESTS ON (docs/design/DESIGN_OF_RECORD.md). The admissibility gate
consumes a (source config, target config, candidate concept) function; ``training/evidence/
admissible_retrieval.py`` accepts it as an argument and nothing produced it until this module.

THE CLAIM, STATED SO IT CAN FAIL
--------------------------------
A phone in a pocket is strong evidence about gait and none at all about what the arms are doing. If
that is true, then for a fixed concept the discriminability of that concept should vary SHARPLY and
SYSTEMATICALLY with sensor placement — and it should do so on the SAME physical events, not merely
across datasets with different subjects and protocols.

Two estimators, in increasing order of how much they prove:

1. ``per_stream_resolvability`` — for every (stream, label), how well does that stream separate that
   label from the rest of its own protocol? One-vs-rest, subject-disjoint kNN, rescaled so 0 = chance
   and 1 = perfect. This is "can this configuration witness this concept", measured directly.

2. ``paired_contrast`` — restricted to datasets where several streams record the SAME sessions
   simultaneously (sp_sw_har phone+watch, xrf_v2's six streams, opportunity, realdisp, mmfit). Same
   subjects, same events, same clock: the ONLY thing that differs is where the sensor sits. A large
   per-label spread across simultaneous streams is the jumping-jacks effect, measured. A small spread
   falsifies the premise, and would mean the admissibility gate has nothing to gate on.

Estimator 2 is the one that can kill the thesis, which is why it is reported separately rather than
averaged into estimator 1.

WHY ONE-VS-REST AND NOT ACCURACY
--------------------------------
A stream's overall accuracy conflates "this configuration is good" with "this protocol is easy".
Resolvability is a property of a (configuration, concept) PAIR, so each label is scored against its
own chance level within its own stream, and the score is the excess over chance.

Run:
    python -m training.evidence.resolvability --build --checkpoint <phase_a>/best.pt
    python -m training.evidence.resolvability --report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OUT_PATH = Path("training/evidence/outputs/resolvability.json")

# Datasets whose streams are simultaneous views of the same sessions. Only these support the
# paired contrast; everywhere else "different placement" is confounded with different subjects and
# protocols, which is exactly the confound the contrast exists to remove.
PAIRED_DATASETS = ("sp_sw_har", "xrf_v2", "opportunity", "realdisp", "mmfit")

MAX_WINDOWS_PER_STREAM = 3000
MIN_WINDOWS_PER_LABEL = 20
MIN_SUBJECTS = 2
KNN_K = 5
SEED = 20260812


def _one_vs_rest_resolvability(
    z: np.ndarray, labels: np.ndarray, subjects: np.ndarray, k: int = KNN_K,
) -> dict[str, float]:
    """Per-label excess-over-chance separability, subject-disjoint.

    For each label: hold out half the subjects, ask a kNN over the remaining half whether a query
    belongs to this label or not, and score balanced accuracy. Rescaled ``2*(ba - 0.5)`` so 0 means
    "this configuration cannot tell this concept from the rest of its protocol" and 1 means perfect.
    Clipped at 0: a below-chance estimate is noise, not negative information.
    """
    rng = np.random.default_rng(SEED)
    subject_list = sorted(set(subjects.tolist()))
    if len(subject_list) < MIN_SUBJECTS:
        return {}
    rng.shuffle(subject_list)
    held = set(subject_list[: max(1, len(subject_list) // 2)])
    train_idx = np.array([i for i in range(len(z)) if subjects[i] not in held])
    test_idx = np.array([i for i in range(len(z)) if subjects[i] in held])
    if train_idx.size == 0 or test_idx.size == 0:
        return {}

    zt = torch.from_numpy(z[train_idx]).float()
    zq = torch.from_numpy(z[test_idx]).float()
    zt = torch.nn.functional.normalize(zt, dim=1)
    zq = torch.nn.functional.normalize(zq, dim=1)
    sim = zq @ zt.t()                                        # (Q, T) cosine
    kk = min(k, zt.shape[0])
    nn_idx = sim.topk(kk, dim=1).indices.numpy()

    out: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        pos_train = (labels[train_idx] == label)
        pos_test = (labels[test_idx] == label)
        if pos_test.sum() < MIN_WINDOWS_PER_LABEL or pos_train.sum() < MIN_WINDOWS_PER_LABEL:
            continue
        if (~pos_test).sum() == 0:
            continue
        vote = pos_train[nn_idx].mean(axis=1) > 0.5          # kNN says "this label"
        tpr = float(vote[pos_test].mean())
        tnr = float((~vote[~pos_test]).mean())
        ba = 0.5 * (tpr + tnr)
        out[str(label)] = round(max(0.0, 2.0 * (ba - 0.5)), 4)
    return out


def build(checkpoint: Path, device: str = "cuda", limit_streams: int | None = None) -> dict:
    """Encode every stream with a frozen encoder and score per-(stream, label) resolvability."""
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.eval_transfer import build_encoder, encode_dataset
    from training.tokenizer.pretrain_data import (stream_channel_descriptions,
                                                  _stream_gravity_state)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, dev)
    print(f"encoder: {checkpoint} step {ckpt.get('step')} git {ckpt.get('git')}", flush=True)

    refs = sorted(discover_grids("native"), key=lambda r: (r.dataset, r.stream))
    if limit_streams:
        refs = refs[:limit_streams]

    per_stream: dict[str, dict] = {}
    for ref in refs:
        data = ref.load_data()
        labels = np.asarray(ref.labels)
        subjects = np.asarray(ref.subjects)
        if data.shape[0] > MAX_WINDOWS_PER_STREAM:
            step = data.shape[0] // MAX_WINDOWS_PER_STREAM
            data, labels, subjects = data[::step], labels[::step], subjects[::step]
        if len(set(subjects.tolist())) < MIN_SUBJECTS:
            print(f"  skip {ref.dataset}/{ref.stream}: <{MIN_SUBJECTS} subjects", flush=True)
            continue
        try:
            z = encode_dataset(enc, data, stream_channel_descriptions(ref.dataset, ref.stream),
                               dev, ref.rate_hz, _stream_gravity_state(ref.dataset, ref.stream),
                               channel_mask=ref.mask, dataset=ref.dataset, stream=ref.stream)
        except Exception as exc:                             # noqa: BLE001 - report, never fabricate
            print(f"  FAILED {ref.dataset}/{ref.stream}: {exc}", flush=True)
            continue
        scores = _one_vs_rest_resolvability(z.cpu().numpy(), labels, subjects)
        if not scores:
            continue
        per_stream[f"{ref.dataset}/{ref.stream}"] = {
            "dataset": ref.dataset, "stream": ref.stream,
            "n_windows": int(data.shape[0]), "labels": scores,
        }
        mean = float(np.mean(list(scores.values())))
        print(f"  {ref.dataset}/{ref.stream}: {len(scores)} labels, mean resolvability {mean:.3f}",
              flush=True)

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "per_stream": per_stream,
        "paired_contrast": paired_contrast(per_stream),
    }


def paired_contrast(per_stream: dict) -> dict:
    """Per-label spread across SIMULTANEOUS streams — the falsifiable half.

    Restricted to datasets whose streams record the same sessions at the same time, so subjects,
    protocol and clock are held fixed and placement is the only variable. Reports, per label, the
    best and worst configuration and the gap between them.

    A large mean gap is the jumping-jacks effect measured. A small one says placement barely matters
    for concept identifiability, which would leave the admissibility gate with nothing to gate on —
    a result that must be reported, not smoothed away.
    """
    out: dict[str, dict] = {}
    for dataset in PAIRED_DATASETS:
        streams = {k: v for k, v in per_stream.items() if v["dataset"] == dataset}
        if len(streams) < 2:
            continue
        by_label: dict[str, dict[str, float]] = defaultdict(dict)
        for key, payload in streams.items():
            for label, score in payload["labels"].items():
                by_label[label][payload["stream"]] = score
        rows = {}
        for label, scores in by_label.items():
            if len(scores) < 2:
                continue                                     # not observed by 2+ simultaneous streams
            best = max(scores, key=scores.get)
            worst = min(scores, key=scores.get)
            rows[label] = {
                "best_stream": best, "best": scores[best],
                "worst_stream": worst, "worst": scores[worst],
                "gap": round(scores[best] - scores[worst], 4),
                "n_streams": len(scores),
            }
        if rows:
            gaps = [r["gap"] for r in rows.values()]
            out[dataset] = {
                "n_labels": len(rows),
                "mean_gap": round(float(np.mean(gaps)), 4),
                "max_gap": round(float(np.max(gaps)), 4),
                "concept_dependence": _concept_dependence(by_label),
                "labels": rows,
            }
    return out


def _concept_dependence(by_label: dict[str, dict[str, float]]) -> dict:
    """Does the placement ranking DEPEND on the concept, or is it a fixed quality ordering?

    THE SHARPEST CLAIM IN THE DESIGN. "Some placements are just better sensors" would be a boring
    finding and would not need an admissibility gate — a single per-placement weight would do. The
    contribution requires that a placement good for one concept is bad for another: a pocket phone
    witnesses squats and not bicep curls, a wrist the reverse.

    Measured as the mean Pearson correlation between streams' resolvability profiles across labels.
      * correlation near +1 -> a fixed quality ordering; a scalar per placement would suffice, and
        the (config, config, CONCEPT) structure is unnecessary.
      * correlation near 0 or negative -> which placement wins depends on the concept, and the gate
        genuinely needs its third argument.
    """
    streams = sorted({s for scores in by_label.values() for s in scores})
    labels = [l for l, scores in by_label.items() if len(scores) >= 2]
    if len(streams) < 2 or len(labels) < 3:
        return {"n_pairs": 0, "mean_correlation": None,
                "note": "need >=2 streams and >=3 shared labels"}
    correlations, inversions, compared = [], 0, 0
    for i, a in enumerate(streams):
        for b in streams[i + 1:]:
            shared = [l for l in labels if a in by_label[l] and b in by_label[l]]
            if len(shared) < 3:
                continue
            va = np.array([by_label[l][a] for l in shared])
            vb = np.array([by_label[l][b] for l in shared])
            if va.std() == 0 or vb.std() == 0:
                continue
            correlations.append(float(np.corrcoef(va, vb)[0, 1]))
            # An INVERSION: a beats b on one label and b beats a on another, both decisively.
            diff = va - vb
            if (diff > 0.15).any() and (diff < -0.15).any():
                inversions += 1
            compared += 1
    if not correlations:
        return {"n_pairs": 0, "mean_correlation": None, "note": "no comparable stream pairs"}
    return {
        "n_pairs": compared,
        "mean_correlation": round(float(np.mean(correlations)), 4),
        "inverting_pairs": inversions,
        "inverting_fraction": round(inversions / compared, 4),
    }


# ----------------------------------------------------------------------------------------------
# Consumption by the admissibility gate
# ----------------------------------------------------------------------------------------------

def load(path: Path = OUT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — build it with "
            "`python -m training.evidence.resolvability --build --checkpoint <ckpt>`")
    return json.loads(path.read_text())


def gate_tensor(
    row_stream: list[str],           # (R,) "dataset/stream" of each retrieved row
    candidate_labels: list[str],     # (C,) candidate label strings
    table: dict,
    n_query_sensors: int = 1,
    default: float = 0.5,
) -> torch.Tensor:
    """(Q, R, C) resolvability for ``admissible_retrieval.admissibility``.

    Reads "can the ROW's configuration bear on this candidate concept" — the direction the thesis
    states ("however similar its signal, a pocket phone must not vote on an arm gesture"). Unmeasured
    (stream, label) pairs get ``default`` rather than 0 or 1: absent evidence must not silently
    become either a veto or a licence.
    """
    per_stream = table["per_stream"]
    values = np.full((len(row_stream), len(candidate_labels)), default, dtype=np.float32)
    for r, key in enumerate(row_stream):
        labels = per_stream.get(key, {}).get("labels", {})
        for c, label in enumerate(candidate_labels):
            if label in labels:
                values[r, c] = labels[label]
    return torch.from_numpy(values).unsqueeze(0).expand(n_query_sensors, -1, -1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/tokenizer/outputs/phase_a_headline/best.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-streams", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.build:
        payload = build(args.checkpoint, args.device, args.limit_streams)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n-> {args.out}  ({len(payload['per_stream'])} streams)")
        _print_report(payload)
    elif args.report:
        _print_report(load(args.out))
    else:
        ap.error("pass --build or --report")


def _print_report(payload: dict) -> None:
    contrast = payload.get("paired_contrast", {})
    print("\n=== PAIRED CONTRAST — simultaneous streams, same events, placement the only variable ===")
    if not contrast:
        print("  none: no paired dataset had 2+ scorable simultaneous streams")
        return
    for dataset, info in contrast.items():
        cd = info.get("concept_dependence", {})
        corr = cd.get("mean_correlation")
        print(f"\n{dataset}: {info['n_labels']} labels across simultaneous streams · "
              f"mean gap {info['mean_gap']:.3f} · max gap {info['max_gap']:.3f}")
        if corr is not None:
            print(f"    concept-dependence: stream-profile correlation {corr:+.3f} · "
                  f"{cd['inverting_pairs']}/{cd['n_pairs']} stream pairs INVERT "
                  f"({cd['inverting_fraction']:.0%})")
        worst = sorted(info["labels"].items(), key=lambda kv: -kv[1]["gap"])[:6]
        for label, row in worst:
            print(f"    {label:28s} best {row['best']:.2f} ({row['best_stream']})"
                  f"   worst {row['worst']:.2f} ({row['worst_stream']})   gap {row['gap']:.2f}")
    gaps = [i["mean_gap"] for i in contrast.values()]
    corrs = [i["concept_dependence"]["mean_correlation"] for i in contrast.values()
             if i.get("concept_dependence", {}).get("mean_correlation") is not None]
    print(f"\nOVERALL mean placement gap: {float(np.mean(gaps)):.3f}")
    if corrs:
        print(f"OVERALL stream-profile correlation: {float(np.mean(corrs)):+.3f}")
        print("  Near +1 would mean a fixed placement-quality ordering, and a scalar per placement")
        print("  would suffice. Near 0 means WHICH placement wins depends on the concept — which is")
        print("  what makes the gate's (config, config, concept) third argument necessary.")
    print("  A large gap is the premise of the admissibility gate, measured on identical events.")
    print("  A gap near zero would mean placement barely affects concept identifiability — which")
    print("  would leave the gate with nothing to gate on, and must be reported as such.")


if __name__ == "__main__":
    main()
