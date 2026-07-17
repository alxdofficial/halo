# Experiments

Each experiment is a self-contained Python package with its own configuration,
training/evaluation entry points, documentation, and ignored outputs. Production
data preparation, model, and evaluation modules must not depend on experiment
code.

- `tokenizer_contrastive/`: supervised contrastive training of the physical-Hz
  tokenizer using paired geometric augmentations.
