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

## 6. ⚠️ Fairness audit — the "beats harnet" claim does NOT survive

An adversarial audit of the comparison found four confounds, **all independently verified by me against
the code**. Three of them are in code written during this Tier-2 work. The margin over harnet is +2.2
(49.5 − 47.3); each confound below is of comparable or larger size.

**F1 — Target-label text is hand-engineered for HALO only. CRITICAL. VERIFIED.**
`training/evidence/labeltext.py:global_label_paraphrases()` merges *every* entry of
`data/scripts/labels/label_augmentation.py:DATASET_CONFIGS` — and that table contains keys for the
**held-out eval datasets** `motionsense`, `realworld`, `shoaib`. Measured: **42 of 60 (70%) of eval
candidate labels receive hand-authored synonym lists** (motionsense 6/6, shoaib 7/7, realworld 6/8).
ConSE baselines get only `label.replace("_"," ")`; UniMTS gets the raw string; NormWear one fixed
template. This violates this project's own stated rule — *"Never fit anything on the candidate labels"*
(EVIDENCE_ENGINE_TIER2.md §1.2). We price the text ensemble at **+2.2**, i.e. **exactly the size of the
entire claimed margin**, and no baseline was given it.

**F2 — HALO's retrieval hyperparameters were selected ON the eval cells. CRITICAL. VERIFIED.**
`training/evidence/tier1_sweep.py:90` defaults `--datasets` to `policy.PRIMARY_EVAL_DATASETS`, sweeps 36
configs (top_k × tau × CSLS × text_ens), and ranks them by **mean macro-F1 on the eval cells**. The
winner (full-soft, τ=0.03, ensemble ON) was then frozen into `baselines/halo_evidence/adapter.py:50-52`.
That is textbook test-set model selection. No baseline received any eval-set tuning.

**F3 — harnet's ConSE head is fit on a different, smaller corpus than HALO's memory. HIGH. VERIFIED.**
`baselines/harnet/adapter.py:73` fits on 9 datasets; HALO's bank uses the 12-dataset
`pretrain_data.TRAIN_DATASETS`, which adds `sp_sw_har`, `nfi_fared`, `harmes`, `xrf_v2` — supplying
extra **wrist/forearm** streams. HALO's two clearest wins over harnet are the **wrist** cells
(tnda_har, ut_complex) — precisely the placements missing from harnet's head-fit corpus.
Note `crosshar`/`limubert` *are* corpus-matched; only harnet, the model we claim to beat, is not.

**F4 — UniMTS is resampled without anti-aliasing. MEDIUM-HIGH. VERIFIED.**
`baselines/unimts/adapter.py:110` uses `np.interp` for e.g. 100 Hz → 20 Hz, aliasing high-frequency
energy into UniMTS's band, while `harnet/adapter.py:129` uses `scipy.signal.resample_poly`. The fairness
policy promises *one* anti-aliased resampler for all. A real, fixable handicap on the strongest
text-native baseline.

**F5 — No confidence intervals on the headline. HIGH.** `eval_decoder.py` computes bare per-cell
macro-F1 with no subject-stratified CI, unlike `baselines/base.py:185-198`. harnet's per-cell CIs run
±2–5 points; one cell is CI-degenerate. A +2.2 unweighted 7-cell mean with no paired test is not a
demonstrated win. The 49.5 has also never been through the shared `eval.run_baselines` harness.

### What survives, and the framing to use

**RETRACTED:** "HALO beats frozen SOTA / beats harnet." The margin is the size of F1 alone, selected
under F2, against a baseline handicapped by F3, with no CIs (F5).

**SURVIVES (internally controlled — same encoder, same corpus, same bank, same bridge):**

> On a fixed frozen representation, a **non-parametric retrieval bridge** transfers to unseen datasets
> substantially better than the standard **ConSE parametric bridge**: 42.7 → 47.5 with *no learning*,
> → 49.5 with a transfer-aligned decoder. It reaches **parity** with a wrist-pretrained UK-Biobank model
> that used ~10⁴× more pretraining data, at ~2× its parameters.

That claim is unaffected by F1–F4 because they sit *inside* the comparison (ConSE vs retrieval on the
identical encoder/corpus/text treatment). Lead with **data-efficiency and the mechanism**, not a
leaderboard win.

### Required corrections, in order

1. **Purge eval-dataset synonyms** — restrict `global_label_paraphrases()` to `TRAIN_DATASETS` configs
   only (or use descriptions authored blind to eval label lists). Re-score. *Correctness fix, not an
   ablation.*
2. **Text-ensemble parity row** — apply the same ensemble to *every* baseline's target text and report
   the table with and without. Expect harnet to gain ~2 and erase the margin.
3. **Re-select HPs on held-out training configs**, never eval; report both numbers, the delta is the
   optimism.
4. **Retrieval-augmented harnet (the killer control)** — build an identical bank from frozen harnet
   features and run the identical mechanism. If it lands near 49.5, the win is the *mechanism*, not our
   encoder — still publishable, but a different paper. `build_memory.py` needs ~50 lines to become
   backbone-agnostic.
5. **Re-fit harnet's head on the 12-dataset corpus** so it is a same-data control.
6. **Route the decoder through `eval.run_baselines`** for subject-stratified CIs + a paired per-cell
   test. (Currently 4–3 vs harnet on cells.)
7. **Fix UniMTS to `resample_poly`** and re-score as a disclosed correction.
8. **Open-vocab-only slice** — macro-F1 over just the ~16 eval labels absent from the training vocab.
   That is the only genuinely zero-shot subset and the only place the language claim can be proven.

Also flagged and worth fixing: `BASELINE_FAIRNESS_POLICY.md` claims crosshar/limubert are retrained at
60 Hz / seq_len 360, but the code uses 20 Hz / seq_len 120 — the doc is wrong and is an easy reviewer catch.

---

## 7. CORRECTED RESULTS after fixing F1 — and a negative result on label augmentation

Two fixes applied: (i) `global_label_paraphrases(train_only=True)` is now the default, so paraphrases
come **only** from training-dataset tables (no eval-specific phrasing); (ii) `eval_decoder --raw-labels`
embeds the bare eval label string — exactly what `eval/scoring.py` gives every ConSE baseline.

**Measured cost of the F1 contamination** (untrained mechanism, full-soft, τ=0.03):
contaminated 8-variant ensemble **47.5** → raw eval labels **46.1**. The advantage was **+1.4**, and
**46.1 < harnet 47.3**. *The entire "we beat harnet" result was the contaminated text ensemble.*

**All numbers at baseline parity (raw eval labels), top-k=48:**

| configuration | mean macro-F1 |
|---|---|
| untrained control (decoder ≡ identity) | 45.1 |
| **trained decoder, no label augmentation** | **46.1** (+1.0 over control) |
| trained decoder, 16-variant label augmentation | 43.5 (−1.6 vs control) |
| *untrained, full-soft (best honest HALO config)* | *46.1* |
| harnet | **47.3** |

### Negative result: per-episode label augmentation hurt

Hypothesis (a good one): the decoder's gains were confined to seen labels because a single fixed anchor
per label lets it memorize 53 vectors instead of using semantics; re-drawing the surface form every
episode — independently on the evidence and candidate sides — should force semantic matching and
generalize to unseen label strings.

Implemented (`labeltext.build_label_variants`, `train_decoder.sample_text_tables`, `--label-variants`).
**It did not work at these settings: −2.6 macro-F1 vs no augmentation.**

The diagnostic is that **the internal proxy moved the opposite way from the target**:

| | internal class-disjoint proxy | ZS-XD @ parity | mean ‖Δ‖ |
|---|---|---|---|
| no augmentation | 0.694 | **46.1** | 0.65 |
| 16-variant augmentation | **0.733** | 43.5 | 1.16 |

Mechanism: augmentation nearly **doubled the refinement residual** (‖Δ‖ 0.65 → 1.16). A larger Δ means
more over-writing of the raw label text — precisely the failure mode that damages unseen labels — and
the reg-to-identity weight (λ=0.1, never tuned) did not compensate for the added text noise.

**This does not falsify the idea, but it does falsify this implementation.** Untested and plausible:
λ scaled up with augmentation strength; fewer, *curated* variants (some train-only paraphrases are poor,
e.g. "heterogeneous device seated"); same-variant rather than independent draws on the two sides;
LLM-authored descriptions instead of template×synonym products. Single seed, single λ, single K.

**The most consequential lesson is the proxy/target anti-correlation.** Our internal selection metric
(class-disjoint episodes drawn from the *training* vocabulary and configs) improved while real
cross-dataset transfer degraded. Any future decoder work needs a selection metric that tracks ZS-XD —
most likely the open-vocab-only slice (§6, correction 8) — or we will keep optimizing the wrong thing.

### Honest bottom line

At true baseline parity, HALO's best configuration is the **untrained retrieval mechanism at 46.1**,
which is **below harnet's 47.3**. The trained decoder recovers to 46.1 at top-k. What remains solid and
internally controlled: **retrieval bridge (46.1) > ConSE bridge (42.7) on the same frozen encoder and
corpus, with no learning** — a +3.4 mechanism effect, achieved with ~10⁴× less pretraining data than
harnet at ~2× its parameters. That is the claim to build slides on.

## 8. Reproducing

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m training.evidence.train_decoder --device cuda --steps 3000 --val-every 200  # 52 s
$PY -m training.evidence.eval_decoder  --device cuda                # trained -> 49.5
$PY -m training.evidence.eval_decoder  --device cuda --untrained    # identity control -> 46.7
$PY -m training.evidence.bank_size_sweep --device cuda              # bank-size inverted U
$PY -m eval.run_baselines --baselines halo_evidence --device cuda   # T2.0 floor -> 47.5
```
All results above are deterministic on re-run.
