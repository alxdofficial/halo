# Remediation plan — every issue found in the 2026-07-20/21 audits

Consolidates the fairness audit (F-series), the independent pre-run audit (B/H-series), and the
findings from our own diagnostics. Each item gives **exact location**, **what to do**, **effort**, and
a **gate**. Ordered so that nothing expensive is paid for twice.

Legend: ✅ done · ⚠️ partial · ❌ open

---

## Phase 0 — Protect what exists (do first; ~30 min, no GPU)

| # | issue | where | fix | status |
|---|---|---|---|---|
| 0.1 | Pre-change results will be overwritten by the refits | `eval/results/*.json` (56 files) | Copy to `eval/results_archive/2026-07-20_pre-vocab-fix/` + `git tag results-pre-vocab-fix`. Keeps the meeting numbers reproducible and enables a real before/after. | ❌ |
| 0.2 | **Mixed-protocol hazard** — `global_labels.json` says 93, bank says 59, nothing warns | `baselines/halo_evidence/adapter.py:90`, `training/evidence/train_decoder.py:140`, `training/evidence/eval_decoder.py` | Assert `bank["vocab"] == eval_data.load_global_labels()`; fail loud with "rebuild the bank" instead of silently blending protocols. | ❌ |
| 0.3 | Bank has no provenance vs the vocabulary | `training/evidence/build_memory.py` | Store `vocab_sha` in the bank; the 0.2 guard compares hashes, not just lengths. | ❌ |

**Gate:** a naive `run_baselines` invocation now either produces a single-protocol table or fails loudly.

---

## Phase 1 — Correctness prerequisites (MUST land before any refit; ~1 day, no GPU)

Everything here changes *how* the heads are fit. Doing it after the refits means paying the GPU twice.

### 1.1 ❌ H7/H8 — shared, dataset-stratified subject split
**Where:** `eval/scoring.py:91` (`subject_disjoint_split`), called from `baselines/{halo:158, harnet:337, crosshar:286, limubert:279}/adapter.py`.

**Problem:** each model shuffles *its own* aggregate subject universe, so harnet's gravity exclusions move **16.5% of shared subjects** into different folds than HALO's. Also not stratified — HHAR/MHEALTH/PAMAP2 contributed **zero** validation subjects, and NFI-FARED's val fold had 1 subject / 1 label.

**Fix:** new `eval/splits.py`:
- `build_subject_split_manifest()` — split subjects **within each dataset** (80/10/10, ≥1 each, deterministic seed), keyed `"dataset:subject"`, cached to `data/labels/subject_splits.json`.
- `split_indices(subject_ids)` — map any model's subject array onto the manifest.
- Every adapter looks up the manifest instead of reshuffling ⇒ **identical folds regardless of which streams a model can consume.**

**Gate:** unit test asserting HALO and harnet get identical fold assignments for every shared subject, and every dataset contributes ≥1 val subject.

### 1.2 ❌ H7b — epoch selection uses the wrong metric
**Where:** `baselines/halo/adapter.py:180`, `harnet:~345`, `crosshar`, `limubert` — all select on `(argmax == y).mean()` (window accuracy) while we **report macro-F1**, against ~530× class imbalance.

**Fix:** select on **balanced accuracy / macro-F1** on the val fold. One-line change per adapter, applied to all four identically.

### 1.3 ❌ #12 — temperature calibrated on the selection fold
**Where:** all four adapters call `scoring.fit_temperature` on the **same** `vi` used for epoch selection; the `_` (test) fold from `subject_disjoint_split` is **discarded**.

**Fix:** calibrate on the third fold. It already exists and is currently thrown away.

### 1.4 ❌ Probe capacity asymmetry (found while verifying paper fidelity)
**Where:** `harnet` uses the official 2-layer `EvaClassifier` (512→512→n, ~300k params); `halo:89`, `crosshar:224`, `limubert:218` use a single `nn.Linear(feat→n)` (~24k).

**Fix:** give **every** ConSE-tier model the identical 2-layer probe. This *strengthens* the baselines (and moves LIMU-BERT toward its paper's non-linear downstream classifier), leaves harnet unchanged, and yields one defensible sentence: *"identical 2-layer probe on frozen features for every representation model."*

### 1.5 ❌ Feature caching (makes Phase 2 and every future vocab change nearly free)
**Where:** the four `_fit_head` methods each re-extract frozen features over ~300k windows.

**Fix:** cache extracted features per (backbone-hash, preprocessing-version) to `baselines/<name>/_feat_cache.npz`. Features are **vocabulary-independent**, so refits become seconds instead of hours. **This is what turns the ~2–3 h Phase 2 into ~10 min.**

### 1.6 ⚠️ H6 — cache fingerprint too weak
**Where:** `baselines/harnet/adapter.py:259` and the other three.

**Currently checks:** labels, corpus_mode, dataset list. **Ignores:** backbone checkpoint hash, preprocessing/code version, stream list, per-stream cap, split-manifest hash, seed, head hyperparameters. A fabricated cache with `n_windows=1` was accepted.

**Fix:** one `fingerprint` dict stamped into every cache; mismatch ⇒ refit. Reuse `_backbone_fp()` (already exists in `baselines/halo`).

---

## Phase 2 — The rebuild cascade (~30 min GPU **if** 1.5 lands first, else ~3 h)

Strict order; every step invalidates the next.

1. **Rebuild the memory bank at 93 labels** — `python -m training.evidence.build_memory --device cuda`.
   Also switch it to store **label strings**, not vocabulary indices (see 5.3), so it can never be
   vocabulary-truncated again. *Expect ~+11.5% windows and 36 new label types, incl. `elevator_*`.*
2. **Refit all four ConSE heads together** (never partially — that silently breaks the comparison).
3. **Re-run everything**: `halo_evidence` T2.0 · `train_decoder` · `eval_decoder` (trained + `--untrained` + `--raw-labels`) · `bank_size_sweep` · `select_retrieval_config`.
4. **Report a clean before/after** against the Phase-0 archive.

**Pre-registered prediction (recorded before measuring):** retrieval should gain **more** than the
parametric heads from the recovered rare classes (~120 `elevator_up` windows is negligible for a 93-way
softmax, ample for kNN). **§2's r = −0.973 must be recomputed, not assumed** — usc_had is its extreme
anchor and is exactly the cell that gains elevator exemplars.

---

## Phase 3 — Remaining fairness work (~2–3 h GPU)

| # | issue | where | fix |
|---|---|---|---|
| 3.1 ❌ **F4** | UniMTS resampled with `np.interp` — **no anti-aliasing**, aliases 100→20 Hz straight into its band, while everyone else uses `resample_poly`. Violates our own stated one-resampler policy. | `baselines/unimts/adapter.py:110` | Switch to `scipy.signal.resample_poly`; re-score; report the delta as a disclosed correction. **Expect UniMTS to improve.** |
| 3.2 ❌ **F3** | harnet's head fit on 9 datasets vs HALO's 12 — the missing ones supply the **wrist** streams where we beat it | wired already | Run `HARNET_CORPUS=matched python -m eval.run_baselines --baselines harnet_matched`. Report **both** rows (off-the-shelf *and* corpus-matched). **Expect harnet to improve** ⇒ our deficit widens. |
| 3.3 ❌ **F5** | Headline has **no CIs** and never went through the shared harness | `training/evidence/eval_decoder.py` | Wire the decoder in as a proper adapter so it gets subject-stratified CIs; add a paired per-cell test. Currently 4–3 vs harnet on cells — well inside noise. |
| 3.4 ❌ **#157** | **The decisive control.** HALO gets non-parametric retrieval over 164k labeled windows; baselines get only a linear probe on the same corpus | `training/evidence/build_memory.py` (~50 lines to make backbone-agnostic; `harnet/adapter.py:151` `_extract_feats` already exists) | Build an identical bank from **frozen harnet features**, run the identical mechanism. If retrieval-harnet ≈ 46, the contribution is the **mechanism, not our encoder** — publishable, but a different paper. Also add a **prototype-only** variant (93 class means) to test whether 164k windows at inference are even needed. |

---

## Phase 4 — Disclosure / policy decisions (no compute; needs a human call)

| # | issue | where | options |
|---|---|---|---|
| 4.1 ❌ **H9** | Short windows **wrap-padded**: sp_sw_har 1.00 s duplicated **×5.00 exactly**, uci_har ×1.95, unimib ×1.65 — artificial periodicity into a frequency-sensitive CNN. 7.5% of all windows; **higher in matched mode** (sp_sw_har is matched-only), so it may *inflate* matched-harnet. | `baselines/harnet/adapter.py:153` | (a) build genuine contiguous 5 s inputs from source sessions; (b) exclude short sources + sensitivity analysis; (c) keep and disclose. **Pre-register the choice before running matched.** |
| 4.2 ❌ **#10/#11** | Fidelity wording | `docs/baselines/*` | Say "official released **HARNet-5** checkpoint" — not the paper's headline config (paper: 10 s, 1024-d, ~10M). Say our global-vocab probe + ConSE is a **standardized control**, not a reproduction of published downstream numbers. |
| 4.3 ❌ | LIMU-BERT probe deviates from its paper (theirs: **GRU** classifier; ours: linear) | `baselines/limubert/adapter.py:218` | Partly resolved by 1.4 (2-layer probe). Either go further to a GRU probe, or disclose — note the current setup **understates** LIMU-BERT, which flatters us. |

---

## Phase 5 — Science improvements (post-rebuild; the actual research)

| # | issue | where | fix |
|---|---|---|---|
| 5.1 ❌ | **Objective too easy** — 0.850 on 18-way episodes; uniform distractors mean fine-grained confusions never appear as negatives | `training/evidence/train_decoder.py` (`sample_H`, candidate sampling) | **Hard-negative sampling** (SBERT-near distractors); larger candidate sets; per-episode text resampling **with λ tuned** (naive version cost −2.6). |
| 5.2 ❌ | **Proxy/target anti-correlation** — internal metric rose while ZS-XD fell | selection metric in `train_decoder` | Select on an **open-vocab-only slice** (the ~16 eval labels absent from the training vocab) — the only genuinely zero-shot subset. |
| 5.3 ❌ | Bank is vocabulary-truncated by construction, and cap 8000 is suboptimal | `training/evidence/build_memory.py` | Store label **strings**; move default per-label cap **8000 → 2000** (measured +1.0, half the memory; cap 200 gives 47.9 at 6% of the bank). |
| 5.4 ❌ | Regressions on usc_had / ut_complex | task #154 | λ sweep (never run) + OOD confidence gate falling back to the untrained mechanism. |
| 5.5 ❌ | Results reported as a single mean, hiding the open-vocab regression | assembler | Report **unseen-label-stratified** metrics as first-class. |

---

## Phase 6 — Data (the highest-leverage item overall)

| # | issue | where | fix |
|---|---|---|---|
| 6.1 ❌ | **We own 100% of Capture-24 and train on ~1%** (13,120 sessions / 6.4 GB on disk; ~33 h reaches training out of 2,562 h) | `data/scripts/eda/build_grids.py:121` (`np.concatenate` OOMs), `data/datasets/capture24/metadata.json` (`max_hours_per_class: 25`), `training/tokenizer/pretrain_data.py:53` (`MAX_PER_STREAM`) | Chunked/appending grid writer; raise the caps (make `MAX_PER_STREAM` per-dataset + source-balance-aware). **~290 h → ~2,800 h for zero acquisition cost.** Then re-measure retrieval purity — *the* test of whether data moves the 0.68 ceiling. |
| 6.2 ❌ | Unlabeled data can't enter training at all | `training/tokenizer/pretrain_data.py:143,177-193` (`BalancedBatchSampler` keyed entirely on `label_id`) | Add an unlabeled branch + `labels=None` guard (A1/A3 are fully self-supervised; only A2 needs labels). Optionally SimCLR-mode A2. Unblocks NHANES (~2 M h). |

---

## Recommended sequence

**Before the meeting:** Phase 0 only (protects the numbers, ~30 min).

**Then:** Phase 1 → Phase 2 → Phase 3.1/3.2 → **Phase 3.4 (#157)**. 3.4 is the one that could change what
the paper *is*, so it should not wait behind the polish items.

**In parallel (no GPU contention):** Phase 4 decisions, Phase 6.1 (Capture-24 unlock) — 6.1 is the
highest-leverage single item on this list and is independent of everything above.

**Deferred until after a clean rebuild:** Phase 5.
