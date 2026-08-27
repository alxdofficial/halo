# HALO documentation

This branch contains one active research program: personalized movement monitoring from wearable
IMU representations.

## Reading order

1. [**MOTIVATION.md**](design/MOTIVATION.md) - why the project moved from generic open-label HAR to
   applied movement measurement.
2. [**RESEARCH_TASKS.md**](design/RESEARCH_TASKS.md) - the three agreed tasks and their boundaries.
3. [**DESIGN_OF_RECORD.md**](design/DESIGN_OF_RECORD.md) - the shared representation and algorithms.
4. [**EVALUATION_PROTOCOL.md**](design/EVALUATION_PROTOCOL.md) - leakage, metrics, controls, and data
   roles.
5. [**IMPLEMENTATION_PLAN.md**](design/IMPLEMENTATION_PLAN.md) - staged build order and exit criteria.
6. [**APPLICATION_DATASETS.md**](data/APPLICATION_DATASETS.md) - which local datasets can answer each
   task and which are contaminated by existing pretraining.
7. [**BASELINES.md**](baselines/BASELINES.md) - released representations and raw/physical controls.
8. [**RESULTS.md**](results/RESULTS.md) - promoted application results only; currently a pre-result
   protocol record.

## Implementation references

- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) - converter, session, unit, grid, and quality contracts.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) - verified per-source sensor semantics.
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) - available physical transformations; not automatically
  enabled for application tasks.
- [CONTINUOUS_KERNEL_FRONTEND.md](design/CONTINUOUS_KERNEL_FRONTEND.md) - continuous physical-time
  HALO frontend.
- [TEXT_CONDITIONING.md](design/TEXT_CONDITIONING.md) - implemented acquisition-description path;
  retained as an encoder reference, not an application claim.
- [`training/tokenizer/README.md`](../training/tokenizer/README.md) - optional representation
  pretraining and diagnostics.

## Documentation policy

- Active claims have exactly one owner document linked above.
- Generated tables do not become design documents.
- A promoted result is summarized only in `results/RESULTS.md` and points to its versioned artifact.
- Old zero-shot and evidence-engine documents are not duplicated in an active-looking archive. They
  remain available at commit `32267b6` and on the branches that use that design.
- Application terms are used precisely: an unconfirmed recurrence is a **motion motif**, latent
  distance is **difference**, and clinical or ergonomic interpretations require external validation.
