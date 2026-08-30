# External representation adapters

Each baseline directory contains its citation, local publication material, official repository
metadata, and a thin adapter around an author-released checkpoint.

The active application comparison treats these models as frozen movement representations. Their
native label heads and open-vocabulary mechanisms are not the common downstream interface. Every
eligible encoder instead exports a timestamped sequence consumed by the same detector, difference
measurement, and motif-discovery code.

Primary released-checkpoint roster:

- `harnet`
- `unimts`
- `normwear`

`imagebind` remains available as an optional generic multimodal appendix control. It is not part of
the routine application matrix because UniMTS is a more directly relevant motion-language control
and ImageBind requires the full, large multimodal checkpoint.

CrossHAR and LiMU-BERT are excluded from the main comparison because their usable local checkpoints
were pretrained by this project rather than released by the authors.

See:

- [`../docs/baselines/BASELINES.md`](../docs/baselines/BASELINES.md)
- [`../docs/baselines/BASELINE_FAIRNESS_POLICY.md`](../docs/baselines/BASELINE_FAIRNESS_POLICY.md)
- [`../docs/design/EVALUATION_PROTOCOL.md`](../docs/design/EVALUATION_PROTOCOL.md)
