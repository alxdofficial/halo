# Task 0 implementation

This package implements the statistical event-proposal design in
`docs/tasks/TASK0_EVENT_SEGMENTATION.md`. It operates on one explicitly selected native-time sensor
stream and produces class-agnostic motion intervals. It does not classify activities or infer
intent.

The active path is:

```text
canonical RawRecording
  -> dynamic-acceleration and angular-speed evidence
  -> development-fitted median/MAD scaling
  -> two-threshold hysteresis
  -> bounded PELT boundary refinement
  -> interval metrics
```

Missing samples remain masked, acceleration-only streams remain acceleration-only, and gravity is
removed only in the temporary dynamic-motion evidence. The original gravity-present signal is not
changed. Physical windows and all detector durations are expressed in seconds, so native 20, 27,
30, 50, and 100 Hz streams use the same contract without resampling.

Install the optional boundary-refinement dependency with `pip install -e '.[task0]'`. The code fails
explicitly when PELT is enabled without `ruptures`; it never silently falls back to another method.

## Commands

Fit robust feature scaling on declared training/development sources:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task0.cli fit \
  openpack recofit --output outputs/task0/scaler.json
```

Select a threshold operating point only on a source whose relevant background is exhaustively
annotated. Annotation level and sensor stream are always explicit:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task0.cli calibrate recofit \
  --model outputs/task0/scaler.json \
  --output outputs/task0/detector.json \
  --stream-id right_forearm_imu \
  --annotation-kind set \
  --confirm-exhaustive-background
```

Write proposals or evaluate one declared annotation level:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task0.cli detect c_mhad \
  --model outputs/task0/detector.json \
  --stream-id imu --output outputs/task0/c_mhad_proposals.jsonl

/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task0.cli evaluate c_mhad \
  --model outputs/task0/detector.json \
  --stream-id imu --annotation-kind event \
  --output outputs/task0/c_mhad_metrics.json
```

Do not pass `--exhaustive-background` for C-MHAD: its intervals identify target actions, not every
coherent movement. In that mode, average precision, event precision, event F1, and false proposals
per hour are intentionally omitted. WEAR and OCA may use exhaustive metrics only with the documented
activity/background label filter. Audited policies live in `policies.py`; the CLI rejects a wrong
stream, mixed annotation levels, C-MHAD exhaustive scoring, OCA exhaustive scoring without excluding
`Null`, and exhaustive scoring over only a selected subset of positive labels. An explicit
`--allow-exploratory-policy` override marks analyses that are not part of the frozen protocol.

Render a full raw/evidence/proposal/reference timeline before accepting an operating point:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task0.cli plot c_mhad \
  --model outputs/task0/detector.json \
  --recording-id tv_gestures_subject_01_run_01 \
  --stream-id imu --annotation-kind event \
  --output outputs/task0/c_mhad_timeline.png
```

## Stability gate

Task 0 is not considered experimentally frozen until the complete OpenPack/RecoFit development
calibration, held-out C-MHAD/WEAR/OCA evaluation, and proposal-timeline visual review are recorded.
No Task-1 implementation should consume Task-0 proposals before that gate is complete.
