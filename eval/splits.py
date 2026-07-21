"""Shared, dataset-stratified subject split manifest (REMEDIATION_PLAN Phase 1.1 — H7/H8).

**The problem this replaces.** Every model used to call `scoring.subject_disjoint_split(S, seed)`
on *its own* aggregate subject array. Two consequences, both measured:

  * **H8 — models disagreed on folds.** harnet excludes gravity-removed streams, which changes the
    subject universe *before* the shuffle, so **16.5% of the 363 shared subjects landed in a
    different train/val/test fold than HALO's.** The two models were therefore selected and
    calibrated on different data, which silently breaks the "same protocol" claim.
  * **H7 — the split was not stratified.** Shuffling the aggregate pool left **HHAR, MHEALTH and
    PAMAP2 with zero validation subjects**, and NFI-FARED with a single val subject carrying a
    single label. Epoch selection and temperature calibration were effectively blind to those
    datasets.

**The fix.** Split subjects **within each dataset**, once, deterministically, and cache the result.
Every model then *looks up* assignments instead of reshuffling — so folds are identical no matter
which streams a given model can consume. A model that must skip a stream simply contributes fewer
windows; it does not move anybody's subject.

Subject ids are ``"dataset:subject"``, matching what the adapters already build.

    python -m eval.splits              # build/refresh data/labels/subject_splits.json
    python -m eval.splits --describe   # show per-dataset fold sizes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "labels" / "subject_splits.json"
SPLIT_SEED = 3431                      # matches the historical FIT_SEED
FRACS = (0.8, 0.1, 0.1)                # train / val / test, by SUBJECT within each dataset


def _split_one_dataset(subjects: Sequence[str], seed: int) -> Dict[str, str]:
    """Assign each subject of ONE dataset to train/val/test. Deterministic given (subjects, seed).

    Guarantees val and test get >=1 subject whenever the cohort has >=3. Cohorts of 1-2 subjects
    cannot be split three ways: they go entirely to train (and are reported), because a 1-subject
    val fold produces a meaningless selection signal — which is exactly the NFI-FARED pathology.
    """
    uniq = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    perm = list(rng.permutation(uniq))
    n = len(perm)
    if n < 3:
        return {s: "train" for s in perm}
    n_val = max(1, int(n * FRACS[1]))
    n_test = max(1, int(n * FRACS[2]))
    n_train = max(1, n - n_val - n_test)
    out = {}
    for s in perm[:n_train]:
        out[s] = "train"
    for s in perm[n_train:n_train + n_val]:
        out[s] = "val"
    for s in perm[n_train + n_val:]:
        out[s] = "test"
    return out


def build_manifest(seed: int = SPLIT_SEED) -> dict:
    """Per-dataset stratified subject assignment over the whole training corpus."""
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.pretrain_data import TRAIN_DATASETS

    per_dataset: Dict[str, List[str]] = {}
    for ref in discover_grids("native"):
        if ref.dataset not in TRAIN_DATASETS:
            continue
        per_dataset.setdefault(ref.dataset, [])
        per_dataset[ref.dataset].extend(str(s) for s in ref.subjects)

    assignment: Dict[str, str] = {}
    summary: Dict[str, Dict[str, int]] = {}
    tiny: List[str] = []
    for ds in sorted(per_dataset):
        # seed per dataset so adding a dataset never reshuffles the others
        ds_seed = seed + int(hashlib.sha256(ds.encode()).hexdigest()[:8], 16) % 100000
        assign = _split_one_dataset(per_dataset[ds], ds_seed)
        counts = {"train": 0, "val": 0, "test": 0}
        for subj, fold in assign.items():
            assignment[f"{ds}:{subj}"] = fold
            counts[fold] += 1
        summary[ds] = counts
        if counts["val"] == 0:
            tiny.append(ds)
    return {"seed": seed, "fracs": list(FRACS), "assignment": assignment,
            "per_dataset": summary, "datasets_without_val": tiny,
            "n_subjects": len(assignment)}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"subject split manifest missing at {MANIFEST_PATH}. Build it:\n"
            f"    python -m eval.splits")
    return json.loads(MANIFEST_PATH.read_text())


def manifest_fingerprint(manifest: dict | None = None) -> str:
    m = manifest or load_manifest()
    return hashlib.sha256(json.dumps(m["assignment"], sort_keys=True).encode()).hexdigest()[:16]


def split_indices(subject_ids: Sequence[str],
                  manifest: dict | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an array of ``"dataset:subject"`` ids to (train_idx, val_idx, test_idx).

    Unknown subjects (e.g. a dataset outside the manifest) are assigned to TRAIN and reported by
    the caller if it cares — never silently placed in val/test, which would leak.
    """
    m = manifest or load_manifest()
    a = m["assignment"]
    folds = np.array([a.get(str(s), "train") for s in subject_ids])
    return (np.where(folds == "train")[0], np.where(folds == "val")[0],
            np.where(folds == "test")[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--describe", action="store_true", help="print the manifest, write nothing")
    args = ap.parse_args()

    m = build_manifest()
    print(f"[splits] {m['n_subjects']} subjects over {len(m['per_dataset'])} datasets "
          f"(seed {m['seed']}, per-dataset stratified {FRACS})")
    print(f"{'dataset':16}{'train':>7}{'val':>6}{'test':>6}")
    for ds, c in sorted(m["per_dataset"].items()):
        flag = "   <-- cohort too small to split" if c["val"] == 0 else ""
        print(f"{ds:16}{c['train']:>7}{c['val']:>6}{c['test']:>6}{flag}")
    if m["datasets_without_val"]:
        print(f"[splits] NOTE: no val subjects for {m['datasets_without_val']} "
              f"(cohort < 3 subjects; all windows go to train)")
    if args.describe:
        print("[splits] --describe: nothing written")
        return
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(f"[splits] wrote {MANIFEST_PATH}  fp={manifest_fingerprint(m)}")
    print("[splits] !! every cached ConSE head is now stale (split changed) and must be refit")


if __name__ == "__main__":
    main()
