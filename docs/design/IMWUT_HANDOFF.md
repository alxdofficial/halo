# IMWUT implementation handoff — weeks 1-3

**Audience: the agent implementing this. Read this file and the two it points at before writing
code.** Design of record: `docs/design/IMWUT_COMPARE_DESIGN.md`. Schedule and sweep findings:
`docs/design/IMWUT_BUILD_PLAN.md`. Venue constraints: `docs/research/IMWUT_VENUE_READ.md`.

Branch: `imwut/compare`. Target: IMWUT Nov 1 2026.

---

## 0. Standing rules — these are not negotiable and are not in the code

1. **Never launch a training, control, or evaluation run without the user's explicit go.**
   "Implement" means build + unit tests + a short smoke on real data. Nothing longer.
2. **Never use the Workflow tool in this repo.** Use direct tools or one scoped subagent.
3. **Python interpreter is `/home/alex/code/HALO/legacy_code/.venv/bin/python`.** There is no venv
   in this repo. Run from the repo root.
4. **Do not commit unless asked.** Push only when asked.
5. **Another agent may be editing this repo concurrently.** Unexpected edits are not corruption.
   Check `git status` before staging; never `git checkout`/`reset` over someone else's work.
6. **Two data roles only: training and evaluation. There is no development split.** Every
   constant, threshold and checkpoint choice is fixed a priori, by judgement, before the run. Never
   select a checkpoint or a hyperparameter on evaluation data.
7. **The GPU is shared and has 24 GB.** Estimate cost before proposing a run; emit progress.
8. **Methodology rule:** any comparison between arms uses paired gain against each arm's *own*
   step-0 score, and paired gain is valid only when both arms share the same step-0 function. If an
   architecture change moves step 0, compare raw scores at matched seeds instead.
9. **Any new learned component is guilty until a control clears it** (untrained floor, and a
   scrambled-vocabulary control where the language channel is involved).
10. Scratch files go in the session scratchpad, never in the repo.

---

## 1. Decisions already made — do not re-open

| # | decision | resolution |
|---|---|---|
| 1 | Zero-shot (k=0) row mechanism | **Comparator over a candidate-excluded, config-compatible corpus draw.** Not ConSE. One mechanism across the whole k-curve; the k=0 support set is drawn from the *training* corpus, compatible with the query, with every candidate label excluded. Disclosed in the paper. |
| 2 | Does Arm A's encoder ever see acquisition text | **No — text is OFF at both stages.** Arm A gets its own Phase-A pretrain with neutral acquisition descriptions, then fine-tunes with them too. Arm B gets a separate text-ON Phase-A run. Two Phase-A runs, ~30 min each. |
| 3 | Arm B2 compatibility distance | **Binary, not a ladder. Distance 0 = identical key. Distance 1 = same device family + *equivalent* placement** (left wrist ↔ right wrist, left pocket ↔ right pocket ↔ front pocket). Anything further apart (wrist ↔ ankle, watch ↔ phone) is **out of scope**, not a further tier — we do not claim a graded degradation curve. |

Arm A = core (no acquisition text, compatibility enforced when building the support set).
Arm B = experiment (acquisition text ON, support unfiltered) — not a claim the paper rests on.

---

## 2. Blocking finding from the sweep — placement strings are prose

`sensor_compatibility_key` in `applications/motion_monitoring/data/compatibility.py` normalizes
placement by lowercasing and collapsing whitespace only. The `StreamSpec.placement` values are
free-text English, so today these are all **different keys for the same physical site**:

```
"the left wrist" (6 streams)   "left wrist" (1)     "the wrist" (2)      "wrist" (1)
"the right wrist" (8)          "right wrist" (3)    "dominant wrist" (1) "the non-dominant wrist" (1)
"the dominant wrist" (1)       "the wrist of the more-affected arm" (1)  ... etc
```

**This breaks Arm A, not only Arm B.** Arm A's support set must be config-*compatible*; if
`capture24` ("dominant wrist", watch) cannot match `harmes` ("the right wrist", watch), the
compatible pool fragments into near-singletons and cross-dataset support becomes impossible.

So **W1 (placement normalization) is the first work item and everything else depends on it.**

Other facts from the same enumeration (90 streams with specs):
- `device_profile` values are `phone` (21), `watch` (21), `device` (45), `watch_proxy` (1),
  `non_deployment` (2). `device` is a catch-all covering limb-mounted research IMUs, earbuds and
  smart glasses — it is too coarse on its own, but combined with a placement class it separates.
- `gravity_state` is `present` for 88 streams and `removed` for exactly 2 (`kuhar/phone_waist`,
  `xrf_v2/airpods_ear`).
- Beware `nfi_fared/wrist`: its `stream_id` says wrist but its `placement` is "the dominant
  forearm". **Always key off `placement`, never off `stream_id`.**

---

## 3. Work items

Each item states scope, definition of done, and the tests that must exist. Do them in order.

### W1 — Placement normalization and the compatibility key (blocking)

**Scope.** Add `data/scripts/curate/compatibility.py` (repo-level, not under `applications/`):

- `PLACEMENT_CLASS: dict[str, str]` — an explicit, checked-in map from every `StreamSpec.placement`
  string in the corpus to a canonical class. Enumerate the real values first (the loop in the sweep:
  iterate `deployment_policy`'s spec containers and collect `(dataset, stream_id) ->
  (device_profile, placement, gravity_state)`). Classes to create, from the observed vocabulary:
  `wrist`, `forearm`, `upper_arm`, `hand`, `waist_hip_belt`, `pocket`, `thigh`, `shin_calf`,
  `knee`, `ankle`, `back`, `chest_torso`, `ear`, `head`, plus one class per muscle-belly site for
  kneepad (it is not used for Nov 1; map it but do not rely on it).
- `EQUIVALENT_PLACEMENTS: frozenset[frozenset[str]]` — which classes are *equivalent* for
  decision 3. Per the user's rule, equivalence is **laterality and pocket variants only**, and the
  classes above already absorb laterality (left/right wrist both map to `wrist`), so in practice
  this set is small. Do NOT make `wrist` equivalent to `forearm`, or `pocket` to
  `waist_hip_belt` — a pocketed phone swings with the thigh, a belt-mounted one moves with the
  torso, and they are genuinely different signals.
- `compatibility_key(dataset, stream) -> SensorCompatibilityKey` using
  `(device_family, placement_class, channel_set, gravity_state)`. Sampling rate stays excluded.
- `are_compatible(a, b) -> bool` (identical key) and `is_near_miss(a, b) -> bool` (same family,
  equivalent placement class, anything else may differ) for Arm B2.

**Definition of done.**
- A script prints the full stream → key table; **the user reviews it by hand before W2 starts.**
- Every `StreamSpec.placement` in the corpus maps to a class. An unmapped string is a loud
  `KeyError`, never a silent fallback.
- Report: how many distinct keys the 18 training datasets collapse to, and the size of the largest
  compatible pool. If any evaluation stream's key has no compatible training partner, say so — that
  is a finding, not a bug to paper over.

**Tests.** `tests/test_compatibility_key.py`: every corpus placement maps; laterality collapses
(`dsads/left_wrist` and `dsads/right_wrist` share a key); `nfi_fared/wrist` keys as `forearm` not
`wrist`; `kuhar` is incompatible with every gravity-present phone stream; a near-miss pair is
`is_near_miss` but not `are_compatible`.

### W2 — Restore the classification results docs

Copy from `archive/pre-application-main-20260830` into `docs/results/classification/`:
`ADAPTATION_TABLE_20260822.md`, `ENCODER_COMPARISON_20260822.md`, `EVAL_HARNESS_AUDIT_20260822.md`,
`PHASE_B_STEP0_CONTROL.md`, `PHASE_B_DIAGNOSIS_20260820.md`, `PHASE_B_MIXER_20260819.md`, plus the
`figures/` they reference. Add a README saying these describe the pre-pivot classification line and
are the provenance trail for the paper's prior numbers. **Do not edit their contents.**

### W3 — Phase-A pretrain at the compact shape (two runs, needs the user's go)

The CLI already defaults to fixed frontend, single resolution, sensor granularity, factored text.
What changes is shape and the text arm:

```
Arm A:  --d-model 128 --trunk temporal --num-layers 3 --neutral-acquisition-text   (text OFF)
Arm B:  --d-model 128 --trunk temporal --num-layers 3                              (text ON)
```

Check first whether `--neutral-acquisition-text` exists on `pretrain.py` or only on
`pretrain_episodic.py` — the sweep found it on the episodic trainer. If Phase-A has no neutral
path, add one that routes `stream_channel_descriptions(..., neutral=True)` and
`stream_sensor_texts(..., neutral=True)`; that function already supports the parity arm.

**Definition of done.** Two checkpoints, each with `run_config.json` recording the text arm.
Encoder effective rank at the end of training is healthy (the last comparable run sat near 94 on a
256-d model; expect lower at 128-d but nowhere near single digits). ~26 min each at batch 1024.

### W4 — The support-only comparator

**Scope.** A new module (suggested `model/evidence/comparator.py`) reusing `EvidenceMixer` and
`mixed_vote`, with the retrieval machinery removed.

**Token layout, one sequence per query recording.** `EvidenceMixer.forward` currently takes
`retrieval_score` and builds six token groups. For the new path:

| group | count | content | role |
|---|---|---|---|
| candidates | C | `candidate_text` (frozen SBERT of each candidate label) | `ROLE_CANDIDATE` |
| query rows | Q | the encoder's pooled feature per (patch, sensor) of the query | `ROLE_QUERY` |
| query descriptors | Q | frozen SBERT of the query's sensor text | `ROLE_QUERY_DESC` |
| support rows | K | pooled feature per support recording | `ROLE_EVIDENCE` |
| support descriptors | K | sensor text of each support recording | `ROLE_EVIDENCE_DESC` |
| support labels | K | frozen SBERT of the support recording's **verbatim** label | `ROLE_EVIDENCE_LABEL` |

**Support rows are one pooled row per support recording**, not per patch×sensor — matching
`live_recording_rows`, which already builds exactly one row per six-second window from the
encoder's deployed pooled output. This keeps K small enough to attend over fully and makes the
support row identical to the feature the frozen-feature baselines are scored on.

Changes to `EvidenceMixer`: delete the `retrieval_score` argument and the `score_bias_gain`
parameter with it (with no retrieval there is no score to bias with, and feeding zeros leaves a
dead parameter). Everything else — roles, slots, groups, the zero-init `residual_head`, the
`ScaledSum` composition — stays.

**Readout.** `mixed_vote(log_weights, memory=support_rows, candidate_text, label_text,
top_k=None, allow_corpus_text_vote=True)` where `log_weights` is the mixer's per-(query,support,
candidate) output. Every support row is either enrolled (bound to a candidate slot, votes identity
1) or unbound (votes rectified `cos(label_text, candidate_text)`). Both paths already exist.

**Definition of done + tests** (`tests/test_comparator.py`):
- **Identity at init**: with `residual_head` zero-initialised the comparator's output equals the
  closed-form vote over the same support set, to 1e-6. This is the untrained floor and it must be
  exact, because the step-0 control depends on it.
- **Permutation invariance**: shuffling the support set does not change the logits.
- **Candidate permutation equivariance**: permuting candidates permutes the logits identically.
- **K = 0 works**: with an empty support set the forward runs and the vote is the text-only path.
- **No dead parameters**: every parameter receives gradient after the second optimizer step.

### W5 — The support-set sampler

**Scope.** `training/compare/sampling.py`. One sampler, two modes.

```
draw_episode(corpus, rng, *, K, p_gt_present, label_subset_range, mode):
    # mode: "compatible" (Arm A) | "unfiltered" (Arm B) | "near_miss" (Arm B2)
    1. query    <- draw one recording from the corpus
    2. key      <- compatibility_key(query.dataset, query.stream)
    3. pool     <- recordings that are:
                     - mode "compatible": are_compatible(key, their key)
                     - mode "near_miss":  is_near_miss(key, their key) AND NOT are_compatible(...)
                     - mode "unfiltered": everything
                   AND  subject != query.subject
                   AND  execution_id != query.execution_id
    4. labels_present <- labels available in pool
    5. gt_in    <- rng.random() < p_gt_present  AND query.label in labels_present
    6. n_labels <- rng.integers(label_subset_range)      # e.g. 2..8
       chosen   <- rng.choice(labels_present \ {query.label}, n_labels - gt_in)
       if gt_in: chosen <- chosen + [query.label]
    7. support  <- for each chosen label, draw round-robin from its recordings in pool
                   until K rows total; labels are balanced to within one row of each other
    8. candidates <- the chosen labels, verbatim, shuffled
    9. return Episode(query, support, candidates, gt_slot = index of query.label or None)
```

**Rules that must hold and must be tested.**
- The query's own recording and its own execution never appear in the support set.
- No support row shares a subject with the query.
- In `"compatible"` mode every support row's key equals the query's key exactly.
- Labels are used **verbatim**. No canonicalisation, no synonym merging, no deduplication — two
  candidates may legitimately have near-identical text and that is the point.
- When the pool cannot supply K rows: **shrink K for that episode and record it in telemetry.**
  Never pad with incompatible rows, and never silently skip the episode. If more than a small
  fraction of episodes shrink, that is a corpus finding to report, not a bug to hide.
- `gt_in` is a *sample*, so the realised rate is checked in telemetry against `p_gt_present`.

**Constants** (a priori, from the design doc; vary in training only as a deliberate experiment):
`K = 32`, `p_gt_present = 0.5`, `label_subset_range = (2, 8)`.

**Tests** (`tests/test_compare_sampling.py`): query never in support; subject-disjoint; compatible
mode gives identical keys; near-miss mode gives equivalent-but-not-identical keys; label balance
within one; realised `gt_in` rate over 10k draws within 3 sd of `p`; short-pool shrink is recorded
not padded; determinism under a fixed seed.

### W6 — `training/compare/train.py`

**Scope.** Warm-start from the W3 checkpoint, draw episodes with W5, run the W4 comparator, one
optimizer.

**Loss.** Per episode, logits `z ∈ R^C` over candidates from the vote.

- **Few-shot episode** (`gt_in` true): plain cross-entropy against the ground-truth candidate slot.
- **Zero-shot episode** (`gt_in` false): the correct answer is not among the candidates, so a
  one-hot target is undefined. Target is the **label-text similarity distribution**:

  ```
  t = softmax( cos(text(query_label), text(candidate_c)) / tau_text ),   tau_text = 0.1
  loss = KL(t || softmax(z))
  ```

  `tau_text = 0.1` is fixed a priori and matches the retrieval temperature already used elsewhere
  in the repo. This is the guard against the measured k=0-below-chance collapse.
- **Total** `L = mean over few-shot episodes CE + mean over zero-shot episodes KL`, each averaged
  within its own group then summed with weight 1.0 each. Do not weight by episode counts — the
  mix ratio is already `p`, and weighting twice would make `p` do two jobs.

**Schedule.** 35,000 steps, final-step checkpoint (no selection on held-out data), AdamW, cosine
schedule with the existing warmup, seed on the CLI. Encoder LR lower than the comparator LR (the
repo's `TOKENIZER_LR_SCALE = 0.05` is the precedent).

**Telemetry, logged every N steps** — these exist to catch the failure modes we have already hit:
- `encoder/effective_rank` — the collapse watchdog. It fell 24 → 9 in 300 steps on a from-scratch
  run. If it falls sharply, stop and report; do not train through it.
- `vote/enrolled_mass_share` and `vote/text_mass_share`.
- `sampler/realised_gt_rate`, `sampler/shrunk_episode_fraction`, `sampler/mean_support_size`.
- `loss/ce`, `loss/kl` separately.

**Definition of done.** A `--smoke` path that runs ~50 steps on real caches and prints the
telemetry. No full run without the user's go.

### W7 — Step-0 predictor

Same checkpoint format at initialisation, asserting identity-at-init against the closed-form vote
(the repo's `make_step0_predictor.py` is the precedent, including its 1.5e-8 gap assertion). Every
trained run is reported paired against its own step-0.

### W8 — `baselines/halo_compare/adapter.py`

Two faces, mirroring `baselines/halo_compact`:
- `window_features` — pooled rows, consumed by nearest/prototype/ridge/linear_head.
- native enrollment — the W4 comparator over the manifest's support rows. **No corpus bank** for
  k ≥ 1; the manifest's support rows *are* the support set, and they are already
  `same_configuration` and execution-disjoint.
- k = 0 — per decision 1, draw a candidate-excluded compatible support set from the training
  corpus. This is the one place a corpus draw survives. Cache it keyed by checkpoint hash + seed,
  and record the draw in `evaluation_config`.

### W9 — Clean baseline rerun

Re-run harnet, UniMTS, ImageBind, NormWear on `adaptation_v1` from a **clean, tagged** commit.
Drop CrossHAR and LiMU-BERT from the table (self-pretrained, not released checkpoints; keep the
code). Make `eval/assemble_adaptation.py` refuse any result row with `git_dirty: true`. Costs per
pass: harnet 31 s, UniMTS 50 s, ImageBind 48 s, NormWear 485 s, plus ConSE head fits.

---

## 4. What "done for week 3" looks like

- W1 table reviewed by the user; W2 docs restored; W4 + W5 + W7 built with their tests passing;
  W6 smoke runs on real caches and prints sane telemetry; W3 checkpoints exist (if the user gave
  the go); W9 either finished or scheduled.
- The full test suite passes (`tests/`, excluding `tests/applications` which belongs to the other
  line).
- No full training run has been launched without an explicit go.

## 5. Open questions to bring back rather than answer alone

1. If W1 shows an evaluation stream with no compatible training partner, the Arm A k=0 draw for
   that stream is undefined. Report it; do not invent a fallback.
2. If the compatible pool for a common key is smaller than K = 32 for a large share of episodes,
   K may need to drop. That is a user decision, not an implementation detail.
3. If encoder effective rank collapses in the W6 smoke despite warm-start, stop. The candidate
   fixes on record are a VICReg term on the query/support features, or an encoder LR reduction —
   but which one is a user decision.
