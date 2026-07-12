# training

Training harness + scripts for HALO (symmetric InfoNCE alignment of IMU patches to SBERT label text,
with the channel-text + augmentation curriculum).

Two **training modes** share this harness:
- **harmonised** — fixed 6-channel input (predictable order/count),
- **normal (non-harmonised)** — native variable-channel input.

Any mode-specific data treatment is applied via [`../data/scripts`](../data/scripts) (assembly),
so the harness itself stays mode-agnostic. To be rebuilt — see the root README roadmap.
