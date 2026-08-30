# Application results

> **Status: protocol and implementation phase, 2026-08-30.** No result for the four application
> tasks has been promoted yet.

## Active questions

0. Can an IMU timeline be segmented into coherent motion events at a practically useful boundary
   accuracy and false-proposal rate?
1. Can a frozen IMU representation find an independently repeated demonstrated movement in a
   continuous recording at a practically low false-alarm rate?
2. Can aligned latent and physical measurements distinguish real execution change from ordinary
   repetition, session, and device-remounting variability?
3. Can an unsupervised motif search recover frequently repeated occupational motions with acceptable
   review burden and false motif rate?

The metrics, controls, and dataset roles are fixed in
[`../design/EVALUATION_PROTOCOL.md`](../design/EVALUATION_PROTOCOL.md). Results will be added only
after the common `MotionSequence` interface and application manifest are implemented and audited.

## Prior evidence and why it is not an application result

The archived pre-application design contains the completed generic HAR comparison. Its main finding
was that released
foundation encoders and HALO remained far below useful supervised HAR accuracy in zero-shot settings,
while simple nearest-neighbor enrollment often matched or beat the learned HALO readout. Those
experiments established that temporal representations and retrieval are worth studying, but they did
not measure continuous event detection, movement change, or unsupervised recurrence.

The exact prior tables remain available with:

```bash
git show 32267b6:docs/results/RESULTS.md
```

The same commit is named by `archive/pre-application-main-20260830`.

They must not be copied into an application table or described as evidence for the new tasks.

## Promotion rule

A result becomes current only when it has:

- one versioned manifest and protocol fingerprint;
- independent reference and query executions;
- target-absent data;
- raw-signal and physical-feature controls;
- all eligible frozen released encoders;
- subject-level confidence intervals;
- per-dataset results and failure cases; and
- a generated artifact path linked from this document.

The optional motion-proposal baseline will report event recall, boundary quality, runtime, and false
proposals only where background annotation is exhaustive. Task 1 will report event AP, false alarms
per hour, count error, and boundary error. Task 2 will report
reliability, known-difference sensitivity, and longitudinal association. Task 3 will report motif
event recovery, fragmentation, false motifs, recurrence count error, and human review burden.
