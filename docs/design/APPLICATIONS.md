# Downstream applications — what a HAR foundation model makes possible

> **Status: brainstorm, 2026-08-06. No decision taken.** Written to answer the supervisor's note
> that the paper needs something that excites MobiCom reviewers beyond a technique contribution.
> Nothing here is built or measured; the numbers cited are from
> `docs/archive/EVIDENCE_ENGINE_FINDINGS.md` and `eval/assemble_table`.

## 0. The constraint that should drive the choice

Current zero-shot standing: **harnet 45.7, HALO evidence engine 42.9, HALO+ConSE 34.4.** We are
behind the strongest baseline. Any application framed as *"more accurate activity recognition"*
puts the paper in a comparison it currently loses.

So the application must rest on **capabilities the baselines structurally cannot provide** — where
the comparison is not "better" but "possible at all."

| primitive | what it enables | baselines |
|---|---|---|
| append-only memory | add an activity from *k* examples, no gradient step | need a head refit |
| runtime label vocabulary | the deployment declares what to detect | only via a ConSE bolt-on |
| config-agnostic encoder | enroll on a watch, run on a pocket phone | placement-blind, or per-config models |
| evidence attribution | "this resembles these three exemplars" | none |
| calibrated abstention | "none of your labels" | must emit a class |

## 1. The metric that wins independently of accuracy

**Cost to add one activity.** HALO: encode *k* windows and append — milliseconds, no training.
Baselines: a gradient update, a fresh validation set, a redeploy. Orders of magnitude, and it is a
*systems* number, which is the currency of the venue.

Paired with it: **catastrophic forgetting.** Fine-tune a baseline on 5 new activities and measure
what happens to the original vocabulary. An append-only memory provably does not degrade. This is
a clean, favourable comparison that does not depend on who has the better encoder.

Both metrics hold even if accuracy ties. That property is why they should carry the section.

## 2. Candidates, ranked

**(1) Movement monitoring for people the training data forgot.** *Recommended.* Pretrained HAR
fails hardest on wheelchair users, post-stroke gait, tremor, and prosthesis users — and scale
cannot fix it, because those people are absent from Capture-24 and UK-Biobank entirely. Per-user
enrollment is the only available route, which is exactly the append-only primitive. We already
carry InclusiveHAR as a held-out set with ability-stratified reporting, so the measurement
apparatus partly exists. The "no labels exist by construction" argument is airtight, and the
framing is socially meaningful rather than merely convenient.

**(2) Personal activity vocabulary / enrollment by demonstration.** The developer-facing form:
detecting a new activity today means collecting a dataset and training a model; here it is five
demonstrations. Demo: a user teaches their phone *loading the dishwasher*, *using an inhaler*,
*shaking a medication bottle* — activities in no public dataset. Already sketched as the Q4 case
study in the paper.

**(3) Cross-device continuity.** Enroll on the watch, recognise on the pocket phone, no extra
data. Uniquely ours — it is the flat-memory property. Cheap to evaluate because paired-device
data already exists (xrf_v2, sp_sw_har, nfi_fared). **Caveat added 2026-08-06:** the
cross-placement objective that trained this property has since been removed from Phase A, so this
claim now rests on the tokenizer alone and must be re-measured before it is asserted.

**(4) Open-set hazard flagging with abstention.** "This is not any of the 20 normal activities"
for elder monitoring. Interesting because closed-vocabulary baselines *must* answer. Ranked below
the others because anomaly detection is a crowded field.

**(5) Sports / coaching.** Same primitive, weaker novelty — sports HAR is well trodden.

## 3. Experiment sketch

Small and feasible — an afternoon of collection, not a new corpus.

- 15–20 activities present in **no** public HAR dataset
- *k* ∈ {1, 2, 5, 10} demonstrations each
- enroll on device A, test on device B (the cross-device claim)
- baselines receive the **same** *k*, fine-tuned
- report: accuracy-vs-*k* curve; **seconds and joules to add one activity**; post-hoc accuracy on
  the original vocabulary (forgetting)

A 5–10 participant study in which people enroll their *own* activities is what would make this a
systems paper rather than another benchmark table.

## 4. Risks

**Novelty.** Enrollment by demonstration is not new as an idea; few-shot and prototype-based HAR
exist. The defensible claim is the *combination*: few-shot enrollment that survives a device and
placement change, which requires the config-agnostic encoder. If that transfer cannot be
demonstrated, the application degenerates into ordinary few-shot learning and the contribution
evaporates.

**Reads as a pivot.** Our *k*=0 numbers trail harnet, so a reviewer may read an application
section as retreat from a weak result. Better to state plainly that zero-shot accuracy is not the
contribution, and that the *k*-curve plus cost-to-enroll is.

**Unresolved control.** A retrieval-augmented harnet may do most of this too. That experiment is
already on the task list and should be settled *before* a demo is built on top of the assumption,
because if a frozen harnet with a memory attached matches us, the application loses its
foundation.

## 5. Open questions

- Which of (1)–(3) do we commit to? They share the primitive but differ in collection effort and
  in who the reviewer imagines using it.
- Does the iOS/Core ML path (paper §Q4) exist far enough to produce real latency and energy
  numbers, or is the cost argument a projection?
- Is cross-device transfer still real now that the objective which trained it has been removed?
