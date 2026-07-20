# Evidence engine — critical findings (2026-07-20)

Analysis run after the Tier-2 decoder cleared its gate (49.5 mean macro-F1 vs the 47.5 untrained floor).
Everything here is measured, with the scripts named. **The headline result survives, but its
interpretation changes substantially.** Read §2 first — it is the one that matters.

---

## 1. Leak audit — the protocol is clean, but "zero-shot" needs qualifying

**Dataset leakage: NONE.** Training corpus and eval datasets are fully disjoint, and the memory bank
contains only training datasets:

- train (12): capture24, harmes, hhar, kuhar, mhealth, nfi_fared, pamap2, sp_sw_har, uci_har,
  unimib_shar, wisdm, xrf_v2
- eval (7): inclusivehar, motionsense, realworld, shoaib, tnda_har, usc_had, ut_complex
- `TRAIN ∩ EVAL = ∅`, `BANK ∩ EVAL = ∅`. No subject overlap is possible (different cohorts).
- The decoder was selected on **held-out training configs**, never on eval data. No eval-set tuning.

**But label overlap is high.** Of 60 eval label slots, **44 (73%) already exist in the 59-label training
vocabulary**; only 16 (27%) are genuinely unseen strings:

| eval cell | #labels | in-train-vocab | unseen | unseen labels |
|---|---|---|---|---|
| motionsense/phone_front_pocket | 6 | 6 | 0 | — |
| realworld/phone_waist | 8 | 8 | 0 | — |
| shoaib/phone_right_pocket | 7 | 7 | 0 | — |
| tnda_har/watch_wrist | 8 | 6 | 2 | lying_down, riding |
| inclusivehar/phone_waist | 6 | 4 | 2 | ramp_ascent, ramp_descent |
| ut_complex/watch_wrist | 13 | 8 | 5 | biking, drinking_coffee, eating, smoking, talking |
| usc_had/phone_hip | 12 | 5 | 7 | elevator_up/down, jumping_up, running_forward, walking_forward/left/right |

This is **not a leak** — it is the normal cross-dataset HAR condition (activities recur across datasets),
and every ConSE baseline faces the identical candidate sets through the identical bridge, so it does not
advantage HALO over the baselines. **But it does mean the benchmark mostly measures cross-*config*
transfer (unseen device / placement / rate / subjects), not open-*vocabulary* transfer.** Our thesis
claims both axes; this eval set stresses the config axis far more than the label axis.

---

## 2. ⚠️ The decoder's gain is concentrated on SEEN labels — and it *hurts* on unseen ones

Cross-referencing per-cell unseen-label fraction against the decoder's gain over its own identity
control produces a near-perfect inverse relationship:

| eval cell | unseen-label % | control | trained | gain |
|---|---|---|---|---|
| motionsense/phone_front_pocket | 0% | 78.3 | 86.2 | **+7.9** |
| realworld/phone_waist | 0% | 43.8 | 51.4 | **+7.6** |
| shoaib/phone_right_pocket | 0% | 49.2 | 55.7 | **+6.5** |
| tnda_har/watch_wrist | 25% | 52.2 | 55.6 | +3.4 |
| inclusivehar/phone_waist | 33% | 28.6 | 29.1 | +0.5 |
| ut_complex/watch_wrist | 38% | 55.1 | 52.3 | **−2.8** |
| usc_had/phone_hip | 58% | 20.0 | 15.9 | **−4.1** |

**Pearson r = −0.973 (p = 0.0002); Spearman ρ = −0.964 (p = 0.0005), n = 7.**
Cells with 0% unseen labels: **mean +7.3**. Cells with ≥30% unseen: **mean −2.1**.

**Interpretation.** The decoder learned *seen-label discrimination*, not open-vocabulary transfer. This
is the M4a failure mode in a subtler form — the class-disjoint episodic loss was designed to prevent it
and only partially does, because episodes are still drawn from the 53-label training vocabulary, so
"held-out label" means "held-out *exemplars* of a label whose text the model has already calibrated to",
not "novel label semantics".

**Confound check.** Unseen-% is partly correlated with cell difficulty, but **ut_complex decouples it**:
it has the *second-highest* control score (55.1, i.e. not a hard cell) yet 38% unseen labels and a −2.8
regression. That points at the label axis, not difficulty. Caveat: n = 7 cells is small; this should be
re-tested as the eval set grows.

**Consequence for claims.** The honest statement is: *the evidence decoder improves cross-config transfer
for known activity vocabulary, but currently degrades open-vocabulary transfer relative to the untrained
retrieval mechanism.* Reporting only the +2.0/+2.8 mean would hide a regression on precisely the
capability the project claims.

---

## 3. Is the training objective too easy? — Yes, and diagnosably so

Measured on the trained decoder over 18-way class-disjoint episodes (chance = 0.056):

| query pool | episode balanced accuracy |
|---|---|
| train configs | **0.850** |
| held-out configs | **0.755** |

Three converging signals that the objective under-constrains the problem:

1. **It is nearly solved** — 0.85 on the training regime; the train→held-out-config gap is only 0.095,
   so this is not classic overfitting, the *task itself* is too easy.
2. **It converges in ~2200 steps / 52 s** and then drifts down (best 0.7039 at 2200, then 0.681–0.688).
3. **Its gains do not transfer to the axis we care about** (§2).

Root causes, in order of suspected impact:

- **Random distractors.** Episode candidate sets are sampled uniformly from the vocab, so most negatives
  are semantically trivial (walking vs brushing_teeth). Fine-grained confusions — the documented failure
  (stairs/ramp/elevator ≈ 0 F1) — almost never appear as negatives.
- **Small candidate sets** (12–24 of 59).
- **Fixed label text.** Candidate anchors are the same ensembled SBERT vectors every episode, so the
  model can memorize 53 fixed points instead of using text semantics.
- **No genuinely novel labels.** Every episode label is a training-vocab label.

---

## 4. Memory bank size — smaller *and* balanced helps, but "a few per label" does not

Sweep of the per-label cap on the untrained mechanism (no learning, so no confound),
`training/evidence/bank_size_sweep.py`, tier-1 config (full-soft, τ=0.03, ensemble 8):

| per-label cap | bank N | mean macro-F1 |
|---|---|---|
| 8000 (current) | 164,516 | 47.5 |
| **2000** | **75,958** | **48.5** ← best |
| 500 | 24,848 | 47.9 |
| **200** | **10,635** | **47.9** ← 15× smaller than current, still better |
| 50 | 2,830 | 46.7 |
| 20 | 1,140 | 45.9 |
| 5 | 285 | 42.8 |

**An inverted U.** The hubness hypothesis is confirmed at the top end: head-class flooding genuinely
hurts, and trimming 8000→2000 buys **+1.0 for free while halving the bank**. But the "tiny bank forces
discrimination" intuition **fails** below ~200/label — coverage of acquisition configs is what the bank
is really providing, and stripping it costs up to −4.7.

Two practically useful consequences: (a) **the default bank cap should move 8000 → 2000**; (b) **cap=200
gives 47.9 with only 10.6k windows (6% of the current bank)** — a strong efficiency/on-device story, and
notably usc_had (the hardest, most-unseen cell) *peaks* at cap 200–500 (21.7/21.3 vs 19.0), i.e.
balance helps exactly where open-vocab transfer is hardest.

---

## 5. What to do next — ranked by expected value

**Fix the objective (highest leverage; directly targets §2 and §3):**
1. **Hard-negative candidate sampling.** Choose distractors that are SBERT-near the true label
   (walking / walking_upstairs / jogging), not uniform. Directly manufactures the fine-grained
   discriminations the model currently never trains on.
2. **Reserve genuinely-unseen labels.** Hold a label subset out of *training candidate sets entirely*
   (not just out of memory), and select checkpoints on that — an internal open-vocabulary proxy that
   §2 shows we currently lack.
3. **Randomize label surface text per episode** (sample a different paraphrase/description each step)
   so candidate anchors cannot be memorized as 53 fixed vectors.
4. **Bigger candidate sets** (up to full vocab).

**Protect the regression (cheap, do alongside):**
5. **Stronger / tuned reg-to-identity**, plus a **confidence or density gate that falls back to the
   untrained mechanism** when evidence is out-of-distribution — would recover usc_had / ut_complex.
   A λ sweep is minutes of GPU and has never been run.

**Bank + eval hygiene:**
6. Move the default per-label cap to **2000**; report the **cap=200** efficiency point.
7. **Report unseen-label-stratified results as a first-class metric** (seen-label cells vs unseen-label
   cells), so a mean can never again hide a regression on the open-vocab axis.
8. Grow the eval set's genuinely-unseen label mass (n=7 cells is thin for the §2 correlation).

---

## 6. Reproducing

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m training.evidence.train_decoder --device cuda --steps 3000 --val-every 200  # 52 s
$PY -m training.evidence.eval_decoder  --device cuda                # trained -> 49.5
$PY -m training.evidence.eval_decoder  --device cuda --untrained    # identity control -> 46.7
$PY -m training.evidence.bank_size_sweep --device cuda              # bank-size inverted U
$PY -m eval.run_baselines --baselines halo_evidence --device cuda   # T2.0 floor -> 47.5
```
All results above are deterministic on re-run.
