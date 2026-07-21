# HALO — canonical results (last updated 2026-07-21)

**This file is the single source of truth for numbers. If another doc disagrees, this one wins.**
Everything here is measured and reproducible; commands are in §6. Written to be slide-ready.

---

## 0. The one-paragraph summary

On a fixed frozen encoder and training corpus, replacing the standard **ConSE parametric bridge** with a
**non-parametric retrieval bridge** improves zero-shot cross-dataset transfer from **42.7 → 45.9 macro-F1
with no learning at all**. A trained evidence decoder reaches **46.1**. This is achieved with a **7.2 M**
parameter encoder trained on a **measured 290 h** of sensor data, versus harnet's 4.24 M parameters
pretrained on **~1.7 × 10⁷ h** (~58,000×). **We do not beat harnet (47.3) at baseline parity.**
The defensible claim is **mechanism + data-efficiency**, not a leaderboard win.

---

## 1. ⚠️ Read before quoting any number: two conventions that change results

**(a) Target-label text.** HALO's retrieval mechanism can embed either the bare eval label string or a
paraphrase ensemble. **Every ConSE baseline receives only the bare string** (`eval/scoring.py`:
`l.replace("_"," ")`). So:
- **"parity" / "raw labels"** = bare string, comparable to baselines. **Use this for any HALO-vs-baseline claim.**
- **"train-only ensemble"** = 8 paraphrase variants drawn *only* from training-dataset tables. Worth
  ~+0.2–0.3. A legitimate HALO-internal ablation, **not** for a baseline comparison unless every baseline
  gets the same treatment.
- **"contaminated ensemble"** = the pre-2026-07-20 behaviour, which merged paraphrase tables keyed to the
  **held-out eval datasets**. Worth **+1.4**. ⛔ **All numbers produced this way are RETRACTED.**

**(b) Two different HALO backbones exist. This resolves an apparent contradiction between docs:**

| checkpoint | used by | HALO+ConSE score |
|---|---|---|
| `pretrain_native/best.pt` | `baselines/halo` (the RESULTS_V2 table row) | **40.4** |
| `pretrain_fixed_mr/best.pt` | the evidence engine (bank, decoder, all §3 numbers) | **42.7** |

Both are "HALO + ConSE"; they differ only in backbone. `fixed_mr` is the better, current one.

---

## 2. Baseline table — 7 primary ZS-XD cells (from `docs/baselines/RESULTS_V2.md`)

| model | mean macro-F1 | notes |
|---|---|---|
| **harnet** (frozen, UK-Biobank) | **47.3** | strongest baseline; head fit on its legacy 9-dataset corpus |
| HALO Phase-A + ConSE (`pretrain_native`) | 40.4 | the published HALO row |
| crosshar / limubert / UniMTS / NormWear | lower | per-cell detail in `RESULTS_V2.md` |

harnet's per-cell CIs run ±2–5 points and one cell is CI-degenerate. **No HALO-vs-harnet result in this
document has a paired significance test.** Treat sub-2-point gaps as not demonstrated.

---

## 3. Evidence engine — current numbers (post-F1-fix)

Same frozen `pretrain_fixed_mr` encoder and same 164,516-window memory bank throughout.

### 3a. At baseline parity (raw labels) — **the row to use in comparisons**

| configuration | mean macro-F1 |
|---|---|
| ConSE bridge, same encoder | 42.7 |
| untrained retrieval, honest-selected config (top_k=200, τ=0.08) | 45.7 |
| untrained retrieval, eval-selected config (top_k=0, τ=0.03) | 45.9 |
| untrained retrieval @ top-k=48 (the decoder's identity control) | 45.1 |
| **trained decoder, no label augmentation @ top-k=48** | **46.1** |
| trained decoder, 16-variant label augmentation @ top-k=48 | 43.5 |
| *harnet* | *47.3* |

### 3b. With HALO's train-only text ensemble (internal ablation, **not** baseline-comparable)
untrained honest-selected **45.9** · untrained eval-selected **46.2** · trained decoder (aug) **44.2**

### 3c. ⛔ RETRACTED — contaminated-text era, do not cite
untrained full-soft **47.5** · untrained top-k=48 **46.7** · trained decoder **49.5**

These are the numbers behind the withdrawn "beats harnet" claim, retained only so the size of the
confound stays auditable. **§2 and §4 of `EVIDENCE_ENGINE_FINDINGS.md` were computed on this scale** —
their *directions* hold, their absolute values do not.

---

## 4. What is solid, and what is not

**Solid (internally controlled — same encoder, corpus, bank, text treatment):**
- **Retrieval bridge > ConSE bridge: 42.7 → 45.9, with zero learning.** +3.2, and the strongest result we have.
- **Efficiency**: ~10.0 M total params (7.17 M encoder + 2.81 M decoder) vs UniMTS 68.6 M and NormWear
  136 M (+1.1 B text tower) — both of which we beat. Corpus is a measured **290 h reachable / 547 h materialised**.
- **Decoder identity-at-init** is unit-tested (9 tests), which is what makes the `--untrained` control exact.
- **Bank size is an inverted U**: per-label cap 2000 beats 8000; cap 200 works at 6 % of the bank; below
  ~200 it collapses. (Measured on the contaminated-text scale — the *shape* holds, absolutes shift.)

**Not solid / open:**
- **We do not beat harnet at parity** (46.1 vs 47.3), and no significance test exists.
- Two of four fairness confounds remain unfixed (§5), and **both point toward the gap widening**.
- **The decoder's gains were concentrated on already-seen labels** (r = −0.973 vs unseen-label fraction).
  Measured on contaminated-text numbers, so the exact coefficient needs recomputing, but the direction was
  monotone across all 7 cells.
- **Label augmentation hurt** (−2.6) while *improving* the internal proxy — our selection metric is not
  aligned with the target.
- Single seed everywhere; no HP sweep except the retrieval config.

---

## 5. Fairness confounds — status

| id | issue | status |
|---|---|---|
| **F1** | Label ensemble pulled synonyms from **held-out eval datasets** (42/60 eval labels; e.g. `jogging → "mobile phone sensing light running"`, from motionsense's own table). Worth **+1.4**. | ✅ **FIXED** (`train_only=True` default) **and retrained twice** |
| **F2** | Retrieval config selected **on the eval cells**. | ✅ **FIXED** — re-selected on held-out training configs. **Optimism measured: only 0.2–0.3.** The eval-selected config ranked 3/36 under honest selection. |
| **F3** | harnet's head fit on 9 datasets vs HALO's 12; the missing four supply the **wrist** streams where we beat it. | ⚠️ **WIRED, NOT RUN** — `HARNET_CORPUS=matched` implemented + smoke-tested. Expect harnet to **improve**. |
| **F4** | UniMTS resampled with `np.interp` (no anti-aliasing) while others use `resample_poly`. | ❌ **NOT FIXED.** Fixing it should **raise** UniMTS. |
| **F5** | No subject-stratified CIs on evidence-engine numbers; never run through `eval.run_baselines`. | ❌ **NOT FIXED** |

Still-unrun decisive control: **retrieval-augmented harnet** (task #157) — build the same bank from frozen
harnet features. Since +3.2 of our improvement is *zero-learning* retrieval, if that also lifts harnet then
the contribution is **the mechanism, not our encoder**. That is a different paper, and it should be run
before any encoder-centric claim.

---

## 6. Reproducing

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m eval.run_baselines --device cuda                              # baseline table
$PY -m training.evidence.select_retrieval_config --device cuda       # F2 honest selection
$PY -m training.evidence.train_decoder  --device cuda --steps 3000   # 52 s
$PY -m training.evidence.eval_decoder   --device cuda --raw-labels             # parity -> 46.1
$PY -m training.evidence.eval_decoder   --device cuda --raw-labels --untrained # control -> 45.1
$PY -m training.evidence.bank_size_sweep --device cuda
HARNET_CORPUS=matched $PY -m eval.run_baselines --baselines harnet --device cuda   # F3, NOT YET RUN
```
All results are deterministic on re-run.

---

## 7. Model + data scale (measured from checkpoints on disk)

| model | params | pretraining data |
|---|---|---|
| LiMU-BERT / CrossHAR | 62.6 k | self-pretrained on **our** corpus |
| harnet5 trunk | 4.24 M | UK-Biobank ~700 k person-days (~1.7 × 10⁷ h) |
| **HALO encoder** | **7.17 M** | our corpus: **290 h reachable / 547 h materialised** |
| **HALO evidence decoder** | **2.81 M** | 3 k steps, **52 s**, on cached vectors |
| UniMTS | 68.6 M | HumanML3D mocap → simulated IMU |
| NormWear | 136 M (+~1.1 B text) | ~15 k signal-hours, mostly ECG/PPG/EEG |

**We are not under-parameterized; we are under-data'd.** See `docs/design/DATA_SCALING_PLAN.md` — headline
there: **we hold 100 % of Capture-24 on disk and train on ~1 % of it**, a ~10× corpus increase for zero cost.

---

## 8. Known open doc/code discrepancy

`docs/baselines/BASELINE_FAIRNESS_POLICY.md` (lines 44–45, 69–74, 224) states CrossHAR and LiMU-BERT are
retrained at **60 Hz / seq_len 360**. The code says otherwise: `baselines/crosshar/prep.py:34` and
`baselines/limubert/prep.py:31` both set `TARGET_HZ = 20`. **The code is what produced the reported
numbers, so the doc's 60 Hz claim does not describe what ran.** Unresolved — flagged rather than silently
edited, because it is not clear whether the intent or the implementation is the error. Easy reviewer catch.

---

## 9. Related docs
- `docs/design/EVIDENCE_ENGINE_FINDINGS.md` — analysis (leak audit, seen-vs-unseen, objective difficulty, bank sweep, fairness audit)
- `docs/design/EVIDENCE_ENGINE_TIER2.md` — design/plan and milestone status
- `docs/design/DATA_SCALING_PLAN.md` — corpus measurement and acquisition plan
- `docs/baselines/RESULTS_V2.md` — full per-cell baseline table with CIs
- `docs/baselines/RESULTS_PRELIMINARY.md` — ⛔ superseded (6-cell, older protocol)
