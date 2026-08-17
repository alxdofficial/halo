# Results Index

> Project-wide index of measured results. Last updated 2026-08-17.
>
> Phase-B design status, run history, and adaptation tables live only in
> [`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md). The historical step-zero analysis is
> retained in [`PHASE_B_STEP0_CONTROL.md`](PHASE_B_STEP0_CONTROL.md). Neither file defines the current
> model; that contract is [`../design/PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md).

## Current Snapshot

| area | artifact/protocol | result | status |
|---|---|---|---|
| zero-shot baselines | v4, 93 labels, 7 datasets | HARNet 45.7 mean macro F1; CrossHAR 42.8; UniMTS 34.7 | historical completed table; predates the 18-source/166-label protocol |
| current Phase A | `phase_a_fixed_1s_rotation_20260817/best.pt` | selected step 27,000 | complete; sensor-granularity, fixed 1 s patches, rotation only |
| parked relational Phase B | v22 and checkpoint study | learned adaptation exists, but usually trails identity/prototype/ridge | historical evidence only |
| current admissibility Phase B | schema-5 sensor bank + rank-8 Stage-2 step 1,000 | coherent test 25.14 versus identity 25.04; random-label test 30.35, exactly identity | operational one-seed result; adaptation exists, but learned admissibility has no held-out advantage |

The current Phase-A checkpoint is
`training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt`. It records 18 training
datasets, `token_granularity='sensor'`, fixed one-second patches, and step 27,000. Its current
rank-8 enrollment run and controls are recorded in `PHASE_B_TRAINING_STATUS.md`. The corresponding
`memory_bank.pt`, `resolvability.json`, and Stage-2 artifacts are bound to that checkpoint; older
result JSON files remain historical and must not be mixed into the current table.

Under one matched fixed-one-second transfer protocol, the older sensor checkpoint scores 0.617 mean
kNN balanced accuracy and the current checkpoint scores 0.509 across the same seven held-out datasets
(-0.108). This removes evaluation patching as the explanation for the gap, but corpus and training
recipe changes remain confounded. See
`training/evidence/outputs/phase_a_checkpoint_selection_20260816/transfer_{old,new}_fixed1s.json`.

On 16 internal held-concept validation episodes, the rebuilt rank-8 warm-start admissibility gate
scores 0.380 mean macro F1 versus 0.592 for the same retrieval rule with admissibility set to one
(-0.211). Gradient and finiteness checks pass, so this is currently a model-quality deficit rather
than a dead training path. It is not an external benchmark result.

## Historical Zero-Shot Table

The last complete baseline table contains 56 cells (8 models by 7 datasets), protocol v4, generated
from `eval/results/` on 2026-08-06. It must not be mixed with current Phase-A or Phase-B results.

| model | mean macro F1 |
|---|---:|
| HARNet | **45.7** |
| HALO evidence, historical | 42.9 |
| CrossHAR | 42.8 |
| UniMTS | 34.7 |
| HALO ConSE, historical | 34.4 |
| LIMU-BERT | 32.2 |
| ImageBind | 11.4 |
| NormWear | 5.1 |

The two historical HALO rows do not identify the current sensor-granularity checkpoint and are not
current headline values. The 18-source expansion now has 166 canonical labels and 10 default
evaluation datasets. That complete table has not been rerun.

## Phase-B Design Ledger

| design | result | interpretation |
|---|---|---|
| v19 coherent relational decoder | learned output far below retrieval-only identity | memorized training-vocabulary query signatures; did not use enrollment effectively |
| v22 arbitrary-alias relational decoder | strong support-removal/shuffle effects; positive k-curve | learned support binding, but usually remained 3-5 F1 below prototype/ridge |
| Phase-A 4k vs 30k relational study | 4k representation usually stronger; decoder rarely beat identity | more Phase-A training did not repair evidence interpretation |
| frozen HARNet enrollment control | HALO identity retrieval led aggregate low-k cells | no evidence that HARNet alone removes the adaptation ceiling |
| current per-sensor admissibility design | valid one-seed external result; coherent test at identity parity, random-label path exactly identity | memory adaptation works, but learned admissibility has not improved the held-out result |

Full historical tables, artifact paths, and their scope limits are in
[`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md).

## Next Confirmatory Readout

The matched zero-shot, supervised-adaptation, and HALO-ablation protocol is defined in
`../baselines/BASELINE_FAIRNESS_POLICY.md` Section 6. The next confirmatory readout must record:

1. Exact Phase-A checkpoint, schema-5 bank fingerprint, active memory population, gate artifact, and
   evaluation source fingerprint, including modality/gravity partition coverage.
2. Gate extrapolation under held-out concept, stream, body-region, and dataset folds.
3. k = 0, 1, 2, 4, 8 and supported k=16, split into ordinary population activities, specialized
   novel activities, and a separate random-label binding control.
4. Same/cross-subject and same/cross-configuration cells without pooling unsupported cohorts.
5. Current-protocol coherent k=0 baselines; supervised head-only and end-to-end fine-tuning at
   positive k; and admissibility-disabled, support-removed, label-shuffled, nearest-support,
   prototype, and ridge controls on identical serialized manifests.
6. Subject-level paired bootstrap intervals and explicit candidate-roster coverage.

The current seven-dataset test roster has now been inspected under the current design and is
exploratory. Any subsequent design selected using those results requires confirmation on a newly
designated untouched holdout roster after the implementation and analysis are frozen.
