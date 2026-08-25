# Baseline fairness and faithfulness policy

This is the treatment contract for the external models reported in the HALO paper. The active
roster is defined in [`BASELINES.md`](BASELINES.md).

## 1. Eligibility

An external model is eligible only when all of the following hold:

1. Its authors released a pretrained checkpoint.
2. The checkpoint can be used for IMU representation or open-vocabulary inference without
   reproducing upstream pretraining.
3. Its required preprocessing and input contract can be implemented from the publication and
   official code.
4. Its training data do not include the held-out test recordings used by our protocol, to the best
   of the information available from the publication.

The active baselines are HARNet, UniMTS, NormWear, and ImageBind. CrossHAR and LiMU-BERT are excluded
because the comparison available in this repository depends on backbones pretrained by us. Their
implementations are retained only as historical research artifacts.

## 2. Frozen-backbone rule

All active external backbones remain frozen. We do not update their encoders on HALO training data
or on target-dataset examples. This preserves the method represented by the author-released
checkpoint and avoids per-model optimizer, augmentation, and checkpoint-selection choices.

At enrollment time, the same non-gradient readouts are applied to each frozen representation:

- 1-NN over enrolled executions;
- normalized class prototypes; and
- closed-form ridge regression.

HALO is also reported with its own retrieve-mix-vote mechanism. That row is not treated as a matched
readout; it tests whether HALO's learned decision mechanism adds value beyond its representation.

## 3. Input faithfulness

Every model receives the maximum sensor input accepted by its released checkpoint.

| model | checkpoint status | sensor contract | label contract |
|---|---|---|---|
| **HARNet** | author released; frozen | 3-axis accelerometer, 30 Hz, 5 s crop | closed-set representation; enrollment readouts only in headline results |
| **UniMTS** | author released; frozen | 3-axis accelerometer through its released placement and resampling path | native text similarity at k=0 |
| **NormWear** | author released; frozen | variable real accelerometer and gyroscope channels at its expected rate | native signal/text distance at k=0 |
| **ImageBind** | author released; frozen | released 6-channel IMU preprocessing path | native IMU/text cosine similarity at k=0 |
| **HALO** | trained by us | available phone/watch accelerometer and gyroscope channels at native rate | HALO zero-shot path and matched enrollment controls |

We permit only input adaptation required by a checkpoint's documented contract: unit conversion,
resampling, crop or padding, channel selection and order, masks, and the checkpoint's own
normalization. We do not add new sensor-conditioning modules, train a new encoder layer, or expose a
baseline to information unavailable to other methods.

## 4. Zero-shot rule

The main `k=0` table contains only native open-vocabulary mechanisms. UniMTS, NormWear, and
ImageBind use their released text-aware decision paths; HALO uses its own. HARNet is omitted because
its released model has no native open-vocabulary classifier.

A fitted ConSE bridge for HARNet remains useful as an internal diagnostic, but it must not be labeled
as HARNet's native method or placed in the headline zero-shot comparison.

No target-dataset execution, label frequency, validation query, or test query is used to fit or tune
a zero-shot mechanism.

## 5. Shared enrollment protocol

- `k` is the number of independent labeled executions per candidate, not windows or gradient steps.
- Candidate labels, query executions, support prefixes, subject relations, and seeds come from one
  sealed manifest consumed by every model.
- Support and query executions are disjoint. Overlapping windows from one execution cannot cross the
  boundary.
- Support prefixes are nested across k. A larger k adds executions rather than replacing the smaller
  support set.
- Candidate rosters and query cohorts stay fixed across a reported curve. Unsupported larger-k cells
  are `n/a`; executions are never reused to manufacture support.
- Every selected support execution contributes one equally weighted pooled vector to the common
  external-model readouts, preventing longer recordings from receiving extra votes.
- Test queries are never used for fitting, early stopping, or hyperparameter selection.

## 6. Aggregation and reporting

Macro F1 is computed for each protocol cell. Seeds are averaged within each dataset, then datasets
are weighted equally. Report:

1. aggregate zero-shot performance for native open-vocabulary models;
2. aggregate k-curves for 1-NN, prototype, and ridge on every active released checkpoint;
3. HALO retrieve-mix-vote as a separate mechanism ablation;
4. per-dataset versions of the same results; and
5. exact checkpoint and manifest provenance.

Do not mix results from different manifests, candidate vocabularies, dataset rosters, or checkpoint
selection rules in one table. Historical results remain under dated paths and must be identified as
superseded.

## 7. Allowed and forbidden changes

Allowed:

- fixes that restore behavior documented by the publication or official implementation;
- deterministic conversion into the released checkpoint's expected units, rate, shape, and channel
  order;
- shared evaluation code and non-gradient readouts; and
- performance optimizations that leave outputs numerically equivalent within a declared tolerance.

Forbidden in the reported comparison:

- locally pretraining an external backbone;
- fine-tuning an external encoder on HALO or target data;
- tuning baseline-specific readout hyperparameters on test results;
- fitting a semantic bridge and presenting it as a native baseline capability;
- changing a checkpoint architecture to accept additional channels or metadata; and
- including CrossHAR or LiMU-BERT results from the locally pretrained checkpoints.

Any future roster change must update this policy, the table generator, tests, figures, and
`docs/results/RESULTS.md` in the same change.
