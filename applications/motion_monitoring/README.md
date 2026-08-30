# Movement-monitoring application code

This package implements the three application tasks defined in
[`docs/design/RESEARCH_TASKS.md`](../../docs/design/RESEARCH_TASKS.md). The authoritative algorithms,
data roles, and evaluation rules remain in the linked design documents; this file only maps those
contracts to code.

## Shared runtime

- `data/`: native-time dataset adapters, immutable caches, and validation tools.
- `sequence.py`: one timestamped `MotionSequence` contract for HALO and control encoders.
- `training.py`: low-cost optimizer and gradient-health checks shared by development smokes.
- `smoke.py`: one short real-cache training check spanning all three tasks.

## Task packages

- `task1/`: bounded-reference matching against a complete query timeline.
- `task2/`: aligned within-person execution comparison and known-change targets.
- `task3/`: dense multiscale candidates and arbitrary-identity recurrence learning.

Run the mechanical integration check with the required environment:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

Use `--encoder halo --checkpoint <phase-a.pt>` to exercise the real HALO encoder. Add
`--train-encoder` only for a short end-to-end gradient-path check. This command does not perform an
official training run and its tiny development-source metrics are not application results.
