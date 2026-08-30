"""Pull every number the weekly deck shows out of the sealed evaluation artifacts."""
import collections, json, statistics as st
from pathlib import Path

RES = Path("eval/adaptation_results")
KS = [1, 2, 4, 8, 16]
# Baselines re-run 2026-08-23 AFTER the LiMU-BERT g-convention fix and the UniMTS E=8 label
# ensemble. The 2026-08-17 v1_d85761d set is stale and understates both.
BASELINE_RUN = "e2e_compact_35k_20260823"
BASE = {"harnet": "harnet", "unimts": "unimts", "crosshar": "crosshar",
        "limubert": "limubert", "imagebind": "imagebind", "normwear": "normwear"}


def cells(path):
    return json.load(open(path))["results"]


def curve(path, method):
    """dataset-mean-then-regime-mean macro F1, per (regime, k)."""
    acc = collections.defaultdict(list)
    for key, c in cells(path).items():
        if not isinstance(c, dict) or c.get("kind") is None:
            continue
        if c.get("cohort") not in ("main", "secondary_high_support"):
            continue
        if c.get("label_mode") not in (None, "coherent"):
            continue
        m = c.get(method)
        if m is None:
            continue
        acc[(c["regime"], int(c["support_count"]), key.split("/")[0])].append(m["f1_macro"])
    per = collections.defaultdict(list)
    for (reg, k, _ds), v in acc.items():
        per[(reg, k)].append(st.mean(v))
    out = collections.defaultdict(dict)
    for (reg, k), v in per.items():
        out[reg][k] = round(st.mean(v), 2)
    return out


def zero_shot(path):
    acc = collections.defaultdict(list)
    for key, c in cells(path).items():
        if isinstance(c, dict) and c.get("kind") == "zero_shot":
            m = c.get("evidence_engine") or c.get("zero_shot") or c.get("nearest")
            if m:
                acc[(c["regime"], key.split("/")[0])].append(m["f1_macro"])
    per = collections.defaultdict(list)
    for (reg, _ds), v in acc.items():
        per[reg].append(st.mean(v))
    return {reg: round(st.mean(v), 2) for reg, v in per.items()}


out = {"headline": {}, "zero_shot": {}, "ladder": {}, "series": {}}

# --- headline: best HALO (long_4h) + every baseline, three no-fitting readouts
halo = RES / "halo_compact_20260822/halo_compact__adaptation_v1.json"
for method in ("nearest", "prototype", "ridge"):
    out["headline"].setdefault(method, {})["HALO"] = curve(halo, method)
    for name, stem in BASE.items():
        p = RES / f"{BASELINE_RUN}/{stem}__adaptation_v1.json"
        out["headline"][method][name] = curve(p, method)
out["zero_shot"]["HALO"] = zero_shot(halo)
for name, stem in BASE.items():
    out["zero_shot"][name] = zero_shot(RES / f"{BASELINE_RUN}/{stem}__adaptation_v1.json")

# --- ladder A: PB-03 mechanism decomposition, per k
# The decomposition artifact has no `kind`/`label_mode` fields; it carries `dataset` directly.
def decomposition_curve(path, method):
    acc = collections.defaultdict(list)
    for c in cells(path).values():
        m = c.get(method)
        if m is None:
            continue
        acc[(c["regime"], int(c["support_count"]), c["dataset"])].append(m["f1_macro"])
    per = collections.defaultdict(list)
    for (reg, k, _ds), v in acc.items():
        per[(reg, k)].append(st.mean(v))
    out_ = collections.defaultdict(dict)
    for (reg, k), v in per.items():
        out_[reg][k] = round(st.mean(v), 2)
    return out_

dec = RES / "e2e_recording_rerank_35k_v3_20260824/halo_engine_decomposition.json"
for method in json.load(open(dec))["methods"]:
    out["ladder"][method] = decomposition_curve(dec, method)
pb3 = RES / "e2e_recording_rerank_35k_v3_20260824/halo_compact__adaptation_v1.json"
out["ladder"]["pooled_execution_1nn"] = curve(pb3, "nearest")

# --- ladder B: across the three trained Phase-B architectures
for tag, run in [("PB-01", "e2e_compact_35k_20260823"),
                 ("PB-02", "e2e_compact_vector8_35k_20260824"),
                 ("PB-03", "e2e_recording_rerank_35k_v3_20260824")]:
    p = RES / run / "halo_compact__adaptation_v1.json"
    out["series"][tag] = {"nearest": curve(p, "nearest"),
                          "engine": curve(p, "evidence_engine"),
                          "zero_shot": zero_shot(p)}
out["series"]["best (long_4h)"] = {"nearest": curve(halo, "nearest"),
                                   "engine": None, "zero_shot": zero_shot(halo)}

Path("comms/2026-08-24-weekly/data/deck.json").write_text(json.dumps(out, indent=1))
print("wrote deck.json")
for reg in ("ordinary", "specialized_novel"):
    print(f"\n{reg} — 1-NN")
    for name, c in out["headline"]["nearest"].items():
        print(f"  {name:10s} " + " ".join(f"{c[reg].get(k, float('nan')):5.1f}" for k in KS))
