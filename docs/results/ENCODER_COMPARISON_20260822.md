# Is our encoder the limiting factor? — 2026-08-22

**Question.** Our own encoder is trained from random init on a corpus of ~550 h. harnet5 is
pretrained self-supervised on UK Biobank; UniMTS on physics-simulated IMU from HumanML3D mocap. If
those representations are simply better, the honest move is to use one and drop ours. This is the
measurement that answers it.

Two independent tests: a frozen-feature probe over all six encoders, and a full **swap** of
harnet5 in place of our encoder inside the compact evidence engine.

## 1. Frozen-feature probe — identical windows, probe, and metrics

24,000 training / 6,000 validation windows drawn once (`numpy` seed 3) from the training corpus,
identical for every row. Each encoder's frozen features feed the same two-layer MLP probe (166-way,
6 epochs, AdamW 3e-4). Two metrics: **probe accuracy** (how much label information is linearly
available) and **cross-config lift** — same-label/different-stream retrieval within the top-32
neighbours, over the base rate. Cross-config lift is the axis retrieval actually needs.

| encoder | dim | probe acc | x-cfg lift (probe space) | **x-cfg lift (RAW features)** | same-config lift (raw) |
|---|---:|---:|---:|---:|---:|
| harnet5 — UK Biobank SSL, released | 512 | **0.576** | 2.48 | 2.36 | 1.30 |
| UniMTS — NeurIPS'24, released | 512 | 0.532 | 2.69 | 2.59 | 1.31 |
| **ours — compact encoder (90k)** | **128** | 0.534 | **3.11** | **2.82** | 1.44 |
| ours — fixed frontend (6k run) | 128 | 0.506 | 1.97 | 1.92 | 1.43 |
| crosshar — self-pretrained | 72 | 0.411 | 1.66 | 1.66 | 1.28 |
| limubert — self-pretrained | 72 | 0.394 | 1.51 | 1.37 | 1.19 |
| *reference: frozen filterbank + MLP* | 128 | 0.444 | — | 1.78 | — |
| *reference: raw-waveform CNN* | 128 | 0.397 | — | 2.01 | — |

**Read.** Our encoder matches UniMTS on label accuracy at a quarter of the feature dimension, and
has the **highest cross-configuration lift of anything measured** — 2.82 raw, against UniMTS 2.59
and harnet 2.36. harnet leads only on raw label accuracy (+0.042), which is attributable to
UK-Biobank-scale pretraining rather than architecture. The physical filterbank also beats a
raw-waveform CNN of the same budget (0.444 vs 0.397), so the front end is not discarding
label-relevant information.

Two caveats. Feature dimension is not matched (512 vs 128 mildly favours the larger rows in a
probe). The ours-fixed row is a *different, shorter run* (6k steps) than ours-learnable (90k), so
that pair does **not** isolate the front end — and it cannot, since the two frontends differ by 96
nearly-unmoved parameters.

## 2. Full engine swap — harnet in place of our encoder

The probe measures linear decodability, not usability in the retrieval role. So we also built
`model/tokenizer/baseline_backbone.py`, which presents a pretrained backbone under our encoder's
exact output contract — the memory bank, pair scoring, top-64, evidence mixer and text vote stay
byte-identical. Row population is verified identical (1,770 live rows from the same 256 windows,
same `sensor_present` mask), sensor isolation exact, row scale matched to our trunk's √d contract.

6,000 steps, bank 128, seed 20260830, pinned validation draw:

| arm | step-0 selection | best selection | paired gain |
|---|---:|---:|---:|
| ours (control) | 0.3336 | **0.4699** @ 3,500 | +0.1362 |
| harnet5, frozen | 0.3699 | 0.4662 @ 5,000 | +0.0964 |

**Read.** harnet's pretrained features start **higher** (0.3699 vs 0.3336 at step 0 — exactly what
the probe predicts) and finish **level** (0.4662 vs 0.4699, a 0.004 difference far inside noise).
Our randomly-initialised encoder closes the whole gap within 6k steps. Swapping in the strongest
baseline encoder buys nothing.

The UniMTS swap arms were cancelled before running. Cost is the reason and it is worth recording:
UniMTS is **36× slower per window than harnet** (127.9 ms vs 3.6 ms per 256 windows) because its
ST-GCN runs over a 22-node SMPL skeleton — one physical IMU occupies one joint and the other 21 are
zero-filled, so 95.5% of every forward pass computes on zeros, and its input BatchNorm hard-codes 66
channels so a smaller skeleton is not accepted. That is a real deployment cost of UniMTS's design
for single-wearable use, not merely an inconvenience here.

## 3. What this settles, and what it does not

**Settled: swapping encoders is not the lever.** Our encoder is competitive with released
foundation models on label information and leads all of them on cross-configuration structure, and
a direct swap of the strongest one produces no gain. Roughly 80% confidence.

**Not settled: whether encoder representation is the binding constraint in absolute terms.**
`PHASE_B_DIAGNOSIS_20260820.md` argues it is — retrieval ranks by acquisition configuration at ×7.0
lift while same-activity/different-device support rows sit at the 39th percentile. Both can be true
at once if **no available encoder solves cross-configuration matching**, which is what the ×1.4–2.8
range across every tested encoder suggests. If so the ceiling is shared, and the lever is the
objective or the retrieval metric, not the backbone.

**Relevance to the config thesis.** This is currently the *only* measurement that favours the
input-side/configuration story in `../design/MOTIVATION.md`: our encoder carries more
cross-configuration structure than encoders trained on far more data. It is a representation-level
result, not a demonstration that language conditioning caused it — the 2026-08-11 parity gate
(`../design/DESIGN_OF_RECORD.md`) found the conditioning itself inert (+0.0086, sign flipping on 2
of 4 datasets), and inference-time descriptor masking left cross-config lift unchanged. So: the
representation has the property; we have not shown the language interface is what produces it.

## Reproducing

Probe: `/tmp/.../scratchpad/unimts_probe.py` (scratch; the loaders it uses are
`baselines/{harnet,unimts,crosshar,limubert}/adapter.py` and
`training/tokenizer/eval_transfer.build_encoder`). Swap:

```
python -m training.tokenizer.pretrain_episodic --random-init --steps 6000 --warmup-steps 400 \
  --val-every 500 --val-episodes 32 --bank-windows 128 --num-workers 3 --seed 20260830 \
  --encoder-backbone harnet --freeze-backbone --out <dir>
```

Outputs: `training/tokenizer/outputs/encoder_swap_20260822/{ours,harnet_frozen}/`.
