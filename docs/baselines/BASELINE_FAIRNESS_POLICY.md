# Representation comparison policy

## Eligibility

An external encoder is reportable only when its authors released a compatible pretrained checkpoint
and enough official preprocessing detail to run it faithfully. We do not reproduce external
pretraining for the main application comparison.

## Frozen representation rule

All external encoders remain frozen. HALO is frozen for the primary representation comparison. An
optional learned metric projection is trained separately for each frozen representation using the
same application training episodes and model budget; those rows are labeled as trained downstream
models and never mixed with the frozen result.

## Input faithfulness

- Preserve released units, sampling rate, channel order, crop/padding behavior, masks, and
  normalization.
- Supply every real sensor channel accepted by the checkpoint, but do not add channels or metadata
  the architecture was not trained to consume.
- Preserve physical timestamps in the common output even when a checkpoint resamples internally.
- Record the receptive field and output stride of every temporal embedding.
- For pooled-only interfaces, apply a common physical-time sliding window and avoid counting
  overlapping windows as independent evidence.

## Shared downstream rule

Raw methods, physical features, HALO, and external representations use the same task implementation:

- the same reference and query executions;
- the same monotonic sequence matcher and duration bounds;
- the same event-merging policy;
- thresholds chosen only on development subjects;
- the same target-absent recordings;
- the same motif duration search; and
- identical subject-level aggregation.

Task-specific exceptions must be declared before results are run. A method that cannot expose useful
temporal resolution is reported as unsupported rather than given privileged segmentation.

## Leakage and provenance

- References and evaluated events come from independent executions.
- Test subjects and sessions never select thresholds, checkpoints, or hyperparameters.
- A HALO checkpoint cannot claim unseen-dataset performance on a source consumed during its
  self-supervised pretraining.
- Known or possible overlap between an external checkpoint corpus and an application dataset is
  documented. If overlap cannot be ruled out, report the result as a deployment comparison rather
  than unseen-data generalization.
- Every artifact records checkpoint hashes, adapter version, manifest fingerprint, and data roles.

## Reporting

For each task report per-dataset and subject-aggregated results, model size, temporal granularity,
latency, peak memory, and accepted sensor modalities. Report failures and unsupported conditions.

The comparison supports statements such as “encoder A is more useful for cross-session motion
matching under this protocol.” It does not support “architecture A is better because of technique B”
without a matched within-model ablation.
