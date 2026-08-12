"""Encoder provenance probe — did the representation absorb dataset identity?

THE DECISIVE FINGERPRINT GUARD (docs/design/DESIGN_OF_RECORD.md). `sensor_bias` enters the trunk, and
it is close to a dataset fingerprint in this corpus: 34 datasets, one device model each, one
placement per stream. The offline guard in `data/scripts/curate/sensor_bias.py` reports INCONCLUSIVE
and cannot be made conclusive there, because dataset identity and acquisition hardware are confounded
by construction. This probe moves the question to where the confound does not apply: not "does the
descriptor cluster by dataset" (it should — same hardware really does look alike) but **"can dataset
identity be read out of the encoder's representation better than the activity content it is supposed
to encode?"**

THE COMPARISON IS THE POINT
---------------------------
A raw dataset-ID accuracy means little on its own: some provenance signal is unavoidable, because
different hardware genuinely produces different signals and that IS the physics the descriptor is
meant to capture. What would be damning is provenance being *easier to read than activity* — a
representation that has learned "which study is this" more sharply than "what is the person doing".

So the probe reports both, from the same embeddings under the same protocol, and the verdict is the
ratio. It also reports a WITHIN-PLACEMENT variant: restricted to streams sharing a placement,
dataset identity can no longer be inferred from body location, so what remains is closer to pure
provenance.

Both probes are linear (a single ridge-regularised readout, closed form) so the number measures what
is LINEARLY present in the representation rather than what a deep head could extract with enough
capacity — the standard, and the conservative, choice for this question.

Run:
    python -m training.tokenizer.probe_provenance --checkpoint <phase_a>/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

MAX_WINDOWS_PER_STREAM = 1500
MIN_PER_CLASS = 20
RIDGE_LAMBDA = 1.0
SEED = 20260812


def _linear_probe_ba(z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Group-disjoint ridge readout -> balanced accuracy and its chance level.

    Groups are SUBJECTS: a probe that trains and tests on the same person can read identity from
    gait idiosyncrasy rather than from the property under test, which would inflate both numbers and
    make the ratio meaningless.
    """
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return {"balanced_accuracy": None, "chance": None, "n_classes": len(classes),
                "note": "fewer than two classes"}
    keep = np.array([c for c in classes if (y == c).sum() >= MIN_PER_CLASS])
    if keep.size < 2:
        return {"balanced_accuracy": None, "chance": None, "n_classes": int(keep.size),
                "note": f"fewer than two classes with >= {MIN_PER_CLASS} rows"}
    mask = np.isin(y, keep)
    z, y, groups = z[mask], y[mask], groups[mask]

    rng = np.random.default_rng(SEED)
    unique_groups = sorted(set(groups.tolist()))
    rng.shuffle(unique_groups)
    held = set(unique_groups[: max(1, len(unique_groups) // 2)])
    tr = np.array([i for i in range(len(z)) if groups[i] not in held])
    te = np.array([i for i in range(len(z)) if groups[i] in held])
    # A class present only on one side of the split cannot be scored; drop it from BOTH so the
    # chance level and the accuracy describe the same label set.
    scorable = np.array(sorted(set(y[tr].tolist()) & set(y[te].tolist())))
    if scorable.size < 2:
        return {"balanced_accuracy": None, "chance": None, "n_classes": int(scorable.size),
                "note": "fewer than two classes survive the group-disjoint split"}
    tr = tr[np.isin(y[tr], scorable)]
    te = te[np.isin(y[te], scorable)]

    index = {c: i for i, c in enumerate(scorable)}
    X = torch.from_numpy(z[tr]).float()
    X = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    Y = torch.zeros(X.shape[0], len(scorable))
    Y[torch.arange(X.shape[0]), torch.tensor([index[c] for c in y[tr]])] = 1.0
    # PRIMAL-form ridge: (XᵀX + λI)⁻¹XᵀY, a (d+1)×(d+1) solve.
    # NOT the dual form used elsewhere in this repo. Phase-B's few-shot comparators solve in the
    # sample-space dual because there rows are few (k=8 support vectors) and dims many. Here it is
    # the opposite — ~60k windows against ~129 dims — so the dual would build a 60k×60k matrix
    # (~14 GB) to answer a 129×129 question. Same estimator, right factorisation.
    d = X.shape[1]
    A = X.t() @ X + RIDGE_LAMBDA * torch.eye(d)
    W = torch.linalg.solve(A, X.t() @ Y)
    Xq = torch.from_numpy(z[te]).float()
    Xq = torch.cat([Xq, torch.ones(Xq.shape[0], 1)], dim=1)
    pred = (Xq @ W).argmax(dim=1).numpy()
    truth = np.array([index[c] for c in y[te]])

    per_class = [float((pred[truth == i] == i).mean())
                 for i in range(len(scorable)) if (truth == i).any()]
    return {"balanced_accuracy": round(float(np.mean(per_class)), 4),
            "chance": round(1.0 / len(scorable), 4),
            "n_classes": len(scorable), "n_train": int(len(tr)), "n_test": int(len(te))}


def run(checkpoint: Path, device: str = "cuda", limit_streams: int | None = None) -> dict:
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.eval_transfer import build_encoder, encode_dataset
    from training.tokenizer.pretrain_data import (stream_channel_descriptions,
                                                  _stream_gravity_state)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, dev)

    refs = sorted(discover_grids("native"), key=lambda r: (r.dataset, r.stream))
    if limit_streams:
        refs = refs[:limit_streams]

    Z, ds, act, subj, place = [], [], [], [], []
    for ref in refs:
        data = ref.load_data()
        labels = np.asarray(ref.labels)
        subjects = np.asarray(ref.subjects)
        if data.shape[0] > MAX_WINDOWS_PER_STREAM:
            step = data.shape[0] // MAX_WINDOWS_PER_STREAM
            data, labels, subjects = data[::step], labels[::step], subjects[::step]
        try:
            z = encode_dataset(enc, data, stream_channel_descriptions(ref.dataset, ref.stream),
                               dev, ref.rate_hz, _stream_gravity_state(ref.dataset, ref.stream),
                               channel_mask=ref.mask, dataset=ref.dataset, stream=ref.stream)
        except Exception as exc:                              # noqa: BLE001
            print(f"  FAILED {ref.dataset}/{ref.stream}: {exc}", flush=True)
            continue
        try:
            from data.scripts.curate.deployment_policy import get_stream_spec
            placement = get_stream_spec(ref.dataset, ref.stream).placement
        except Exception:                                     # noqa: BLE001
            placement = ref.stream
        Z.append(z.cpu().numpy())
        ds.extend([ref.dataset] * len(z))
        act.extend(labels.tolist())
        subj.extend([f"{ref.dataset}:{s}" for s in subjects.tolist()])
        place.extend([placement] * len(z))
        print(f"  {ref.dataset}/{ref.stream}: {len(z)} windows", flush=True)

    z = np.concatenate(Z)
    ds, act, subj, place = map(np.asarray, (ds, act, subj, place))
    print(f"\nprobing {len(z)} windows · {len(set(ds))} datasets · {len(set(act))} activities",
          flush=True)

    provenance = _linear_probe_ba(z, ds, subj)
    activity = _linear_probe_ba(z, act, subj)

    # CLASS-COUNT-MATCHED comparison. Excess-over-chance is NOT comparable across a 32-class and a
    # 211-class problem: more classes lowers chance but also makes the task harder, so the two
    # excesses live on different scales and their ratio means little. Subsample the activity
    # vocabulary to the same number of classes as there are datasets, keeping the most populous, and
    # compare balanced accuracies directly on equal footing.
    n_ds = provenance.get("n_classes") or 0
    activity_matched = {"balanced_accuracy": None, "note": "provenance probe unscorable"}
    if n_ds >= 2:
        labels, counts = np.unique(act, return_counts=True)
        top = labels[np.argsort(-counts)][:n_ds]
        sel = np.isin(act, top)
        activity_matched = _linear_probe_ba(z[sel], act[sel], subj[sel])
        activity_matched["note"] = (
            f"activity restricted to the {n_ds} most populous labels, matching the dataset count")

    # Within-placement: streams sharing a body location, so dataset identity can no longer be read
    # off "where the sensor is". What survives here is closer to pure provenance.
    within: dict[str, dict] = {}
    for p in sorted(set(place.tolist())):
        sel = place == p
        if len(set(ds[sel].tolist())) < 2:
            continue                                          # only one dataset at this placement
        within[p] = _linear_probe_ba(z[sel], ds[sel], subj[sel])
    scored = [v for v in within.values() if v.get("balanced_accuracy") is not None]

    payload = {
        "checkpoint": str(checkpoint), "checkpoint_step": ckpt.get("step"),
        "n_windows": int(len(z)),
        "provenance_probe": provenance,
        "activity_probe": activity,
        "activity_probe_matched": activity_matched,
        "within_placement_provenance": within,
    }
    payload["verdict"] = _verdict(provenance, activity_matched, scored)
    return payload


def _verdict(provenance: dict, activity_matched: dict, within: list[dict]) -> dict:
    """Is provenance easier to read than activity, AT MATCHED CLASS COUNT?

    Compares balanced accuracies directly, on the same number of classes, so the comparison is
    apples-to-apples. The earlier excess-over-chance ratio across a 32-class and a 211-class problem
    was not: more classes lowers chance while making the task harder, so the two excesses are not on
    the same scale and their ratio overstated the gap.
    """
    p, a = provenance.get("balanced_accuracy"), activity_matched.get("balanced_accuracy")
    if p is None or a is None:
        return {"status": "INCONCLUSIVE", "note": "a probe could not be scored at matched classes"}
    status = ("FINGERPRINT RISK — at matched class count, provenance is read more sharply "
              "than activity" if p > a else "acceptable — activity dominates provenance")
    out = {
        "status": status,
        "provenance_ba": round(p, 4),
        "activity_ba_matched": round(a, 4),
        "n_classes_matched": activity_matched.get("n_classes"),
        "margin": round(p - a, 4),
    }
    if within:
        mean_within = float(np.mean([v["balanced_accuracy"] - v["chance"] for v in within]))
        out["within_placement_provenance_excess"] = round(mean_within, 4)
        out["n_placements_scored"] = len(within)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/tokenizer/outputs/phase_a_headline/best.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-streams", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("training/tokenizer/outputs/provenance_probe.json"))
    args = ap.parse_args()
    payload = run(args.checkpoint, args.device, args.limit_streams)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("\n=== PROVENANCE PROBE ===")
    print(f"  dataset ID : BA {payload['provenance_probe']['balanced_accuracy']} "
          f"(chance {payload['provenance_probe']['chance']}, "
          f"{payload['provenance_probe']['n_classes']} datasets)")
    print(f"  activity   : BA {payload['activity_probe']['balanced_accuracy']} "
          f"(chance {payload['activity_probe']['chance']}, "
          f"{payload['activity_probe']['n_classes']} activities)")
    am = payload["activity_probe_matched"]
    print(f"  activity@matched classes : BA {am.get('balanced_accuracy')} "
          f"({am.get('n_classes')} most populous activities) <- the fair comparison")
    print(json.dumps(payload["verdict"], indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
