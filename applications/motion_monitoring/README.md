# Movement-monitoring application code

This package implements the three application tasks defined in
[`docs/design/RESEARCH_TASKS.md`](../../docs/design/RESEARCH_TASKS.md). The authoritative algorithms,
data roles, and evaluation rules remain in the linked design documents; this file only maps those
contracts to code.

## Shared runtime

- `data/`: native-time dataset adapters, immutable caches, and validation tools.
- `sequence.py`: one timestamped `MotionSequence` contract for HALO and control encoders.
- `baseline_encoder.py`: released HARNet, UniMTS, and NormWear checkpoints adapted to the same
  temporal contract; ImageBind is an optional appendix control.
- `training.py`: low-cost optimizer and gradient-health checks shared by development smokes.
- `smoke.py`: one short real-cache training check spanning all three tasks.
- `evaluation.py`: strict task/dataset/split/provenance result records and comparable tables.

Phase A and Phase B train the upstream HALO representation. In the application study, released
external encoders are frozen and receive a separately trained small module for each task. HALO is
measured both frozen under that same protocol and with a task-specific end-to-end fine-tuning arm.
Before end-to-end fitting, every encoder must pass the paired frozen direct-versus-learned utility
gate in [`docs/design/EVALUATION_PROTOCOL.md`](../../docs/design/EVALUATION_PROTOCOL.md). This keeps
episode construction fixed and isolates whether a gain comes from the representation or the small
task head.

## Task packages

- `task1/`: bounded-reference matching against a complete query timeline.
- `task2/`: set-conditioned within-person execution comparison and known-change targets.
- `task3/`: dense multiscale candidates and arbitrary-identity recurrence learning.

Run the mechanical integration check with the required environment:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

Use `--encoder halo --checkpoint <phase-a.pt>` to exercise the real HALO encoder. Add
`--train-encoder` only for a short end-to-end gradient-path check. This command does not perform an
official training run and its tiny development-source metrics are not application results.

Smoke every primary released encoder against every task head with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.baseline_smoke \
  --baselines harnet unimts normwear --tasks task1 task2 task3 --device cuda
```

Probe the same encoders directly on real ALAMEDA and COPS Task-2 streams with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task2.baseline_data_probe --device cuda
```

Both commands are mechanical compatibility checks. Official comparisons require the frozen cohort
fingerprint, development-only model selection, and metrics in `docs/design/EVALUATION_PROTOCOL.md`.
