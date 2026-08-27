# Motivation: personalized movement monitoring

> **Design of record, 2026-08-27.** This document replaces the prior zero-shot/open-label
> motivation on the `application-motion-monitoring` branch.

## The problem

Generic human activity recognition asks a model to choose a label such as *walking*, *sitting*, or
*drinking* from a predefined vocabulary. This remains a difficult benchmark, but improving a
cross-dataset score from roughly 50% to a slightly higher value does not by itself make the result
useful. It also makes causal model comparisons difficult: published IMU encoders use different
corpora, sensor layouts, model sizes, and sometimes motion capture or proprietary data.

Rehabilitation and occupational monitoring pose a different and more useful question. The motion
of interest is often personal, task-specific, and already available as a recording:

- a clinician demonstrates or approves a rehabilitation exercise;
- a patient repeats that exercise at home over several weeks;
- an ergonomist wants to find a repetitive work motion without first naming every possible task;
- a worker's motion changes during a shift even though the nominal task stays the same.

These settings do not require a universal label vocabulary. They require a representation that can
match movement, align executions that differ in duration, distinguish stable variation from real
change, and remain useful after a watch is remounted or a compatible device changes.

## Research objective

HALO is now studied as a **movement representation and measurement system**, not primarily as an
open-vocabulary classifier.

> **Thesis:** pretrained IMU representations can support demonstration-conditioned detection,
> comparison, and discovery of personally or operationally meaningful movement using consumer
> wearables and little or no task-specific training.

The thesis is evaluated through three tasks:

1. **Arbitrary activity detection:** given one or a few reference recordings, find independent
   occurrences of that movement in later continuous data.
2. **Activity difference quantification:** align two executions of the same task and quantify how
   they differ in latent structure and interpretable physical measurements.
3. **Recurrent motion discovery:** find coherent motion motifs that recur frequently in unlabeled
   occupational recordings, then let a human identify which motifs correspond to meaningful work.

The tasks form one workflow: discover or demonstrate a movement, monitor its future occurrences,
and track how its execution changes.

## Why rehabilitation is the primary application

At-home rehabilitation needs more than exercise recognition. A clinician needs to know whether a
prescribed movement occurred, how many times it occurred, whether the repetitions were consistent,
and whether the execution changed over time. Patient-specific template comparison is already used
in wearable rehabilitation research, including comparison of home exercises with laboratory
references using dynamic time warping. The unresolved opportunity is to replace brittle raw-signal
templates with reusable pretrained representations while retaining inspectable physical outputs.

Relevant evidence includes:

- [patient-specific upper-limb quality monitoring at home](https://pmc.ncbi.nlm.nih.gov/articles/PMC10821060/);
- [MoniPar smartwatch measurements associated with MDS-UPDRS motor scores](https://pubmed.ncbi.nlm.nih.gov/38148984/);
- [PHYTMO repeated physical-therapy exercises with optical reference](https://pubmed.ncbi.nlm.nih.gov/35661743/); and
- [KneE-PAD patient exercise executions with clinically defined errors](https://pmc.ncbi.nlm.nih.gov/articles/PMC11992047/).

## Why occupational monitoring is related but distinct

The occupational task is not to infer injury risk directly from one wrist sensor. A phone or watch
usually cannot observe external load, joint torque, whole-body posture, or the worker's intent.
Instead, it can discover and quantify **repetitive motion exposure**: recurrence count, cumulative
duration, cadence, consistency, and within-shift drift. An ergonomist can then review and name the
recurrent motifs.

Claims about strain, fatigue, or musculoskeletal risk require additional evidence such as load,
force, posture, multiple sensors, or validated ergonomic instruments. Published occupational
systems commonly add biomechanical models, pressure insoles, or several body-worn IMUs; HALO must
not imply those quantities are observable from a single watch.

## What the system may and may not claim

The system may claim that two sensor recordings are similar, that a recurring temporal motif was
found, or that an execution changed beyond a measured personal baseline. It may not infer intent,
clinical improvement, exercise correctness, fatigue, or injury risk unless those outputs are
validated against appropriate human or physical ground truth.

The word **activity** is therefore reserved for a motion confirmed by a person or supplied as a
reference. Before confirmation, Task 3 returns **motion motifs**, not activities.

## Role of HALO and external encoders

The first experiments freeze HALO and author-released external encoders and apply the same matching,
alignment, and discovery procedures to every representation. This answers an application question:
which available representation is useful for these tasks? Different upstream pretraining corpora
are acceptable under that question, but they prevent a claim that one architecture is intrinsically
superior.

HALO earns a model contribution only if controlled ablations show that its physical-time frontend,
temporal patch structure, or heterogeneous-sensor handling improves these application outcomes.
Otherwise the contribution is the task formulation, protocol, system, and applied evidence.

## Non-goals

- Winning a generic zero-shot HAR leaderboard.
- Treating language similarity as motion similarity.
- Inferring intention from IMU signals alone.
- Diagnosing a disease or prescribing treatment.
- Estimating occupational injury risk from repetition count alone.
- Calling augmented copies of one excerpt independent real-world repetitions.

The previous zero-shot and enrollment work remains available in Git history at the branch point,
commit `32267b6`, but it is not the design of record on this branch.
