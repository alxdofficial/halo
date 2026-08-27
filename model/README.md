# Model components

## `tokenizer/` - movement representation

The active model component converts native-rate IMU data into temporal patch embeddings.

| module | role |
|---|---|
| `filterbank.py` | fixed physical-Hz filterbank and signed low-frequency/gravity features |
| `continuous_kernel.py` | learned continuous physical-time convolutional frontend |
| `preprocess.py` | gravity alignment and per-window preparation |
| `sensor_tokens.py` | folds xyz channels into sensor-level tokens with validity masks |
| `transformer.py` | temporal and optional cross-sensor contextualization in physical time |
| `encoder.py` | representation interface returning tokens, per-patch vectors, and pooled vectors |
| `channel_text.py` | optional acquisition-description conditioning |
| `primitives.py` | interpretable physical diagnostics, not task labels |

The application design consumes timestamped **per-patch** vectors. Whole-recording pooling is a
control, not the default, because it removes movement phase and prevents subsequence alignment.

## `evidence/` - historical classification experiments

The evidence modules implement the prior candidate-label retrieval, reranking, and voting research.
They remain for reproducibility but are not used by the movement-monitoring design. Tasks 1-3 use a
shared sequence matcher and do not require candidate labels.

See [`../docs/design/DESIGN_OF_RECORD.md`](../docs/design/DESIGN_OF_RECORD.md) for the active model
boundary and [`../training/README.md`](../training/README.md) for training policy.
