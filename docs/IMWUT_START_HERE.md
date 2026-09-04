# START HERE — branch `imwut/compare`

You are on the branch for the **IMWUT comparison-model line**. If you read only one document before
touching anything, read this one. It exists because this repository has carried three successive
research lines and most of the confusion available here comes from mistaking one for another.

Last updated 2026-09-04.

---

## 1. What we are building, and why

**Thesis: recognise by comparison, not classification.** Given a query recording and a handful of
labelled example recordings acquired under a compatible sensor configuration, a small comparator
decides which examples the query resembles and reads the answer off their labels in language space.

The model is *not* asked to learn a universal representation of all IMU data. It is asked to embed
motion well enough that attention over a compatible support set can make the call.

**What we claim as novel: the training curriculum** — how support sets are sampled. Nothing in the
architecture is claimed as new; the comparator is a matching network with a language readout, and
the related-work section says so.

**Target venue: IMWUT. Deadline: 1 November 2026.** Double-blind, 8–10k words, and a
major-revision-then-accept path is the norm rather than a failure. The venue read is in
[`docs/research/IMWUT_VENUE_READ.md`](research/IMWUT_VENUE_READ.md).

Why this line at all: the supervisor ruled out a clinical pivot on 2026-09-03 (clinical venues want
one condition with data gathered for it, not a generic method that touches several), and HALO began
as an ML paper. The evidence that shaped the design is a ledger of measured failures from the
earlier lines — summarised in §6.

---

## 2. Which code is live, and which is a previous life

| Line | Status | Where |
|---|---|---|
| **IMWUT comparison** (this branch) | **LIVE — this is the work** | `data/scripts/curate/compatibility.py`, `model/evidence/comparator.py`, `training/compare/`, `baselines/halo_compare/`, `eval/compare_cost.py`, `eval/enrollment_protocol.py` |
| Classification / evidence-engine | **superseded**, kept because it still produces the baseline table and the frozen results | `model/evidence/{engine,evidence_reranker,patch_retrieval,relational_decoder,confidence}.py`, `training/evidence/`, `training/tokenizer/`, `baselines/{halo,halo_compact,halo_evidence}/` |
| Application / motion-monitoring (Tasks 1–3) | **not the paper target**, but actively maintained by another agent on `main` | `applications/motion_monitoring/`, `docs/tasks/` |

**Do not delete the superseded code.** The classification stack still encodes evaluation streams,
still fits the ConSE bridge for closed-vocabulary baselines, and still reproduces the numbers quoted
in `docs/results/classification/`. It is superseded as a *design*, not as infrastructure.

**Do not touch `applications/`** unless asked. Another agent works there concurrently; unexpected
edits in this repository are usually that agent, not corruption.

**`docs/results/classification/` is frozen.** It is the provenance trail for numbers the paper
cites. Never edit those files; add new results elsewhere.

---

## 3. The model, end to end

1. **Draw an episode** (`training/compare/sampling.py`). Pick a query recording; look up its
   acquisition key; build the admissible pool — same key, different subject, different execution.
   With probability `p` the query's own label is among the candidates (few-shot); otherwise it is
   excluded (zero-shot). Pick 2–8 candidate labels verbatim, then fill K = 32 support recordings
   round-robin so labels are balanced to within one.
2. **Encode** the query and all support recordings together in one forward pass. Support is encoded
   **per query, with gradients** — nothing is cached, so the comparison itself trains the encoder.
3. **Centre** each episode on its own mean feature (default), so only how rows *differ* can drive
   anything downstream.
4. **Compare** (`model/evidence/comparator.py`). One set-attention sequence per query over
   `[candidates | query row | support rows + descriptors + labels]`. No retrieval, no top-k.
5. **Vote.** `score(c) = Σ_e weight(query, e, c) × vote(e, c)`, where an enrolled row votes 1 for
   its own candidate and any other row votes the rectified cosine between its label text and the
   candidate's. **No per-candidate parameters anywhere**, so an unseen candidate is scored by the
   same operation as a seen one.
6. **Loss.** Few-shot episodes get cross-entropy; zero-shot episodes get KL against the label-text
   similarity distribution at `tau_text = 0.1`, because with the answer absent a one-hot target is
   undefined.

At evaluation, k ≥ 1 uses the manifest's own enrolled support rows with **no corpus bank**. k = 0
is an **ensemble of 8 training-shaped draws** (4 seen labels × 8 recordings = K 32) with every
candidate label excluded, combined before the argmax.

The full design of record, including every ablation, is
[`docs/design/IMWUT_COMPARE_DESIGN.md`](design/IMWUT_COMPARE_DESIGN.md). **When this file and that
one disagree, that one wins** — it is the design of record and this is the orientation.

---

## 4. Standing conventions — violate these and the work is worthless

These are project rules, not preferences. Several exist because they were violated before and cost
a result.

1. **Never launch a training, control, or evaluation run without the user's explicit go.**
   "Implement" means build + tests + a short smoke on real data. Nothing longer.
2. **Two data roles only: training and evaluation. There is no development split.** Every constant,
   threshold and checkpoint choice is fixed *a priori*, by judgement, before the run. Never select a
   checkpoint or tune a hyper-parameter on evaluation data.
3. **Paired-gain rule.** Report a run as `score(trained) − score(its own step 0)`. The validation
   draw once contributed five times the run-to-run scatter of the thing being measured, so a raw
   number carries the draw, not the method. **Paired gain is only valid when both arms share the
   same step-0 function** — an architecture change (centering, front end) moves step 0, so those
   arms are compared by raw score at matched seeds instead.
4. **Every learned component is guilty until a control clears it.** Minimum: the untrained floor and
   the paired step-0 control; add a scrambled-vocabulary control wherever the language channel is
   involved. On this codebase almost every learned addition has been net-negative at first pass.
5. **Never fabricate a data channel.** Absent channels are exact zeros with a validity mask, never
   imputed, never randomly filled.
6. **"Unsupported" is a result.** When a dataset cannot answer a test, report that. Do not widen a
   relation or substitute a different mechanism to manufacture a number.
7. **The interpreter is `/home/alex/code/HALO/legacy_code/.venv/bin/python`**, run from the repo
   root. There is no venv in this repository.
8. **Another agent edits this repo concurrently.** Check `git status` before staging; never
   `checkout`/`reset` over someone else's work.
9. **Do not commit or push unless asked.**
10. **The GPU is shared and has 24 GB.** Estimate cost before proposing a run, and emit progress.

---

## 5. Datasets: what trains, what evaluates, and why

**Training — 18 datasets, 56 streams, 31 acquisition configs, 166 labels.**
`deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS`.

**Evaluation — 10 datasets, manifest `adaptation_v2`** (70 cells, 48 usable, fingerprint
`0f98ae25…`). `adaptation_v1` (7 datasets) is kept intact for comparability but is **not** the
current protocol; the two carry different protocol names so the assembler cannot mix their rows.

- 44 of the 55 evaluation labels are **never seen in training** — that is the open-vocabulary story.
- 4 of the 8 original evaluation acquisition configs are **unseen in training** — that is the
  heterogeneity story.
- `motionsense`, `realworld` and `shoaib` were restored on 2026-09-04. They were never unsuitable;
  they were the old *development* split, and that split no longer exists. They fix a confound —
  previously 8 of 11 streams were wrist and the hardest labels sat on exactly the streams with the
  narrowest support pools. **Disclose in the paper** that design decisions in the superseded
  Phase-B line were made while looking at them: no model was trained on them, so this is researcher
  exposure, not data leakage, and the a-priori-constants rule closes the channel by which it would
  otherwise inflate results.
- **`hapt` may never be evaluated.** It is the same 30 subjects as `uci_har`, which is in training.
  `build_manifest` raises if it is requested.
- `upper_limb_use` (4 streams) and `motionsense` have no config-compatible training stream, so their
  **k = 0 row is unsupported**. Their k ≥ 1 rows are unaffected. This is reported, not patched.

---

## 6. The evidence this design rests on

Every one of these is a measurement from the earlier lines, recorded in
`docs/results/classification/`. They are why the design looks like it does.

| Measurement | Consequence |
|---|---|
| Frozen features separate activities at 84 macro-F1; zero-shot through text sits near 40 | The encoder was never the bottleneck — the bridge was |
| Untrained retrieve-and-vote scored 47.5, edging harnet's 47.3 | Comparison against exemplars works *before* any training |
| Retrieval ranked by **acquisition configuration**, ×7.0 lift, not by activity | Delete the retrieval stage; and centre the episode, since configuration is a common mode |
| Learned retrieval was worth **+0.0000** paired | The stage is not worth its complexity |
| The attention mixer gained +0.12, and a scrambled-vocabulary control inverted it | The gain is *semantic*; keep language in the loop |
| Closed-vocabulary episodic CE pushed zero-shot **below chance** (16.18 → 9.44) | Zero-shot episodes need a soft text target, not one-hot |
| Every compact checkpoint that ever led the table was trained **from random init** | End-to-end from scratch is the default; warm start is a secondary arm |
| Signal-similarity vs label-name-similarity: r ≈ 0.11 | Zero-shot is structurally capped; headline on k ≥ 1 and disclose k = 0 |

---

## 7. Decided and closed

- Arm A (core) sees **no acquisition text at any stage**; Arm B (experiment) has text ON with
  unfiltered support. Separate pretrains per arm; the trainer refuses a mismatched checkpoint.
- Compatibility is **binary**: `are_compatible` (identical key) and `is_near_miss` (same device
  family, equivalent site). There is deliberately **no graded distance ladder**.
- Sampling rate is **not** part of the acquisition key — the filterbank is rate-invariant by
  construction, so rate is nuisance within a configuration.
- Laterality is preserved but wording is not: `"the left wrist"` and `"left wrist"` are one site,
  while left and right wrist are a near miss. Non-dominant, affected and unaffected wrists are their
  own sites.
- Constants, fixed a priori: `K = 32`, `p = 0.5`, label subset 2–8, 35k steps, final-step
  checkpoint, ≥ 3 seeds; zero-shot ensemble R = 8 draws × 4 labels × 8 rows, combiner
  `probability`.
- Centering is **ON by default**; `--no-center-features` is the ablation.
- Fixed filterbank is the design of record; the continuous kernel is the challenger, not the
  default.

## 8. Still open

- **No training run has been launched.** Nothing here has evidence of beating its untrained floor.
- The GPU budget across nine queued ablation arms does not fit the schedule; the cut has not been
  made.
- The clean baseline rerun on `adaptation_v2` has not happened.
- Baseline weight audit for the 2025–26 comparators (Wonderwall, IMUZero, LanHAR, GOAT, MOMENT).
- Whether to spend GPU settling fixed-vs-continuous, or report it open.

---

## 9. How to run it

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python   # from the repo root

# audit the acquisition-key table (read-only)
$PY -m data.scripts.curate.audit_compatibility

# train, end to end from scratch, Arm A. --smoke does ~50 steps on real data
$PY -m training.compare.train --out training/compare/outputs/arm_a \
    --device cuda --neutral-acquisition-text --smoke

# the paired step-0 control, in the same checkpoint format
$PY -m training.compare.step0 --phase-a <phase_a.pt> --out <dir>

# build an evaluation manifest
$PY -m eval.enrollment_protocol --out eval/manifests/adaptation_v2.json.gz

# cost table, on a named device
$PY -m eval.compare_cost --checkpoint <ckpt> --device cuda

# tests
$PY -m pytest tests -q
```

---

## 10. Document map for this line

| Document | What it owns | Status |
|---|---|---|
| **this file** | orientation, conventions, current state | live |
| [`design/IMWUT_COMPARE_DESIGN.md`](design/IMWUT_COMPARE_DESIGN.md) | **the design of record** — model, support contract, ablations, paper shape | live, authoritative |
| [`research/IMWUT_VENUE_READ.md`](research/IMWUT_VENUE_READ.md) | venue mechanics and the recalibration they forced | live |
| [`design/COMPATIBILITY_AUDIT.txt`](design/COMPATIBILITY_AUDIT.txt) | generated stream → acquisition-key table | regenerate, do not hand-edit |
| [`design/IMWUT_BUILD_PLAN.md`](design/IMWUT_BUILD_PLAN.md) | the 2026-09-03 sweep and schedule | **historical** — superseded in parts |
| [`design/IMWUT_HANDOFF.md`](design/IMWUT_HANDOFF.md) | implementation spec written for another agent | **historical** — the work is done |
| `results/classification/` | pre-pivot measurements | **frozen — never edit** |
| `tasks/`, `docs/data/APPLICATION_DATASETS.md` | the application line | not this paper |
