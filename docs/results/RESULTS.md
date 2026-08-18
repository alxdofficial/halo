# Results Index

> Project-wide index of measured results. Last updated 2026-08-18.
>
> Phase-B design status, run history, and adaptation tables live only in
> [`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md). The historical step-zero analysis is
> retained in [`PHASE_B_STEP0_CONTROL.md`](PHASE_B_STEP0_CONTROL.md). Neither file defines the current
> model; that contract is [`../design/PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md).

## Current Snapshot

| area | artifact/protocol | result | status |
|---|---|---|---|
| zero-shot baselines | v4, 93 labels, 7 datasets | HARNet 45.7 mean macro F1; CrossHAR 42.8; UniMTS 34.7 | historical completed table; predates the 18-source/166-label protocol |
| historical Phase A | `phase_a_fixed_1s_rotation_20260817/best.pt` | selected step 27,000; seven-dataset fixed-1s transfer 0.509 versus old 0.617 | completed but rejected for the next Phase-B bank |
| replacement Phase A | clean views, isolated retrieval rows, direct row VICReg, external-development selection | implementation complete; no trained result yet | pending training and controlled ablations |
| parked relational Phase B | v22 and checkpoint study | learned adaptation exists, but usually trails identity/prototype/ridge | historical evidence only |
| current admissibility Phase B | matched adaptation v1, rank-8 Stage-2 step 1,000, five seeds | ordinary coherent k=1: 44.79 versus identity 45.71; specialized k=1: 28.62 versus 28.58; arbitrary labels exactly identity | complete seven-dataset result; adaptation exists, but learned admissibility has no held-out advantage |

The Phase-B results below are bound to the historical Phase-A checkpoint
`training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt`. It records 18 training
datasets, `token_granularity='sensor'`, fixed one-second patches, and step 27,000. Its current
rank-8 matched enrollment suite and controls are recorded in `PHASE_B_TRAINING_STATUS.md`. The corresponding
`memory_bank.pt`, `resolvability.json`, and Stage-2 artifacts are bound to that checkpoint; older
result JSON files remain historical and must not be mixed into the current table.

The 2026-08-17 Phase-A regression is now diagnosed as a recipe and representation-path problem. Its
only active augmentation independently rotated the two VICReg views, forcing invariance to
gravity-frame orientation. Dataset regressions were largest on orientation-sensitive SPAR (-0.209),
Upper Limb Use (-0.178), and RealWorld (-0.142). The final logged JEPA/VICReg gradient cosine was
-0.901, but the complete 61-probe record is bimodal rather than uniformly conflicting (median
+0.563; 34.4% negative). Gradient clipping was nevertheless active on every logged probe (median
coefficient 0.091). Late encoder effective rank had median 39.1/256; the final 23.0 was unusually low
but not representative of every probe. These facts reject “one bad final batch” as the explanation
while avoiding the stronger unsupported claim that the objectives always cancel.

The replacement recipe removes source-fitted sensor statistics from the encoder trunk, exports
sensor-isolated temporal rows before descriptor and cross-sensor mixing, applies half of VICReg
directly to those rows, and starts with no augmentation. `best.pt` is selected every 2,000 steps by
dataset-macro subject-disjoint kNN over MotionSense, RealWorld, and Shoaib development data using the
actual rows Phase B stores. Rotation (shared and independent), rate, channel dropout,
multi-resolution, and descriptor reconstruction are separate ablations.

Bounded 1,000-step pilots use seed 20260718 and retain the full 7,500-step LR/EMA schedule. They are
screening evidence, not final comparisons:

| one-variable arm | development kNN BA | retrieval effective rank / 256 | JEPA/VICReg cosine at step 1,000 |
|---|---:|---:|---:|
| clean | 0.8244 | 87.4 | +0.455 |
| shared SO(3), p=1 | **0.8297** | 84.4 | +0.401 |
| independent SO(3), p=1 | **0.7392** | 57.7 | +0.564 |
| shared rate augmentation, p=0.5 | 0.8216 | 90.5 | +0.622 |
| shared channel dropout, p=0.3 | 0.8264 | 86.9 | +0.522 |
| descriptor prediction, weight 0.5 | 0.8181 | 88.2 | +0.652 |
| multi-resolution, batch 512 / step 2,000 | 0.8124 | **104.1** | +0.088 |

The clean, shared-rotation, rate, and dropout values are too close to rank from one seed. Independent
rotation is not: it loses 0.0852 BA against clean and 0.0905 against the matched shared-rotation arm.
This directly identifies invariance across rotations, rather than rotated input itself, as harmful.
The full next run should start clean; shared rotation is the first follow-up. Rate, dropout, and
descriptor prediction have not earned default complexity from this screen.
The multi-resolution arm uses 2,000 steps so its 1.024 million sampled windows match the other
arms' batch-1,024/step-1,000 exposure. It raises retrieval rank but lowers the development score by
0.012; this is a useful follow-up, not sufficient evidence to enable it by default.

The matched suite uses manifest fingerprint `1bd89d35f5ae`, five fixed episode seeds, seven held-out
datasets, and six external representations. HALO Stage 2 scores 23.75 ordinary and 9.81 specialized
macro F1 at semantic k=0. With enrollment, its ordinary curve is 44.79, 48.89, 51.36, 51.18, and
49.32 for k=1,2,4,8,16. This is below identity retrieval at every point and below the strongest
external frozen-feature linear head. Specialized activities are at identity parity through k=8 and
fall 1.39 points below it at k=16. Full generated tables are in
`eval/adaptation_tables/v1_d85761d_stage2/`.

“Ordinary” denotes the four held-out population locomotion/daily-activity datasets. “Specialized
novel” denotes the three held-out clinical, rehabilitation, and fine-grained upper-limb datasets; it
is a predefined evaluation proxy, not a claim that every underlying movement is physically new. The
complete model-by-k tables, including k=0, are in `PHASE_B_TRAINING_STATUS.md`.

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
current headline values. The separate legacy 10-dataset zero-shot table has not been rerun; the
current matched seven-dataset suite is reported above and in `PHASE_B_TRAINING_STATUS.md`.

## Phase-B Design Ledger

| design | result | interpretation |
|---|---|---|
| v19 coherent relational decoder | learned output far below retrieval-only identity | memorized training-vocabulary query signatures; did not use enrollment effectively |
| v22 arbitrary-alias relational decoder | strong support-removal/shuffle effects; positive k-curve | learned support binding, but usually remained 3-5 F1 below prototype/ridge |
| Phase-A 4k vs 30k relational study | 4k representation usually stronger; decoder rarely beat identity | more Phase-A training did not repair evidence interpretation |
| frozen HARNet enrollment control | HALO identity retrieval led aggregate low-k cells | no evidence that HARNet alone removes the adaptation ceiling |
| current per-sensor admissibility design | valid five-seed matched result; coherent test near or below identity, random-label path exactly identity | memory adaptation works, but learned admissibility has not improved the held-out result |

Full historical tables, artifact paths, and their scope limits are in
[`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md).

## Completed Matched Readout

The matched zero-shot, supervised-adaptation, and HALO-ablation protocol is defined in
`../baselines/BASELINE_FAIRNESS_POLICY.md` Section 6. The 2026-08-17 readout records:

1. Exact Phase-A checkpoint, schema-5 bank fingerprint, active memory population, gate artifact, and
   evaluation source fingerprint, including modality/gravity partition coverage.
2. Gate extrapolation under held-out concept, stream, body-region, and dataset folds.
3. k = 0, 1, 2, 4, 8 and supported k=16, split into ordinary population activities, specialized
   novel activities, and a separate random-label binding control.
4. Same/cross-subject and same/cross-configuration cells without pooling unsupported cohorts.
5. Current-protocol coherent k=0 baselines; common frozen-feature supervised heads at positive k;
   and admissibility-disabled, support-removed, label-shuffled, nearest-support, prototype, and ridge
   controls on identical serialized manifests. Model-native end-to-end fine-tuning remains a
   separate experiment and is not represented by the common head table.
6. Subject-level paired bootstrap intervals and explicit candidate-roster coverage.

The current seven-dataset test roster has now been inspected under the current design. Any subsequent
design selected using these results requires confirmation on a newly designated untouched holdout
roster after the implementation and analysis are frozen.
