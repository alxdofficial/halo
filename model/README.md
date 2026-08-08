# model

The HALO model. Two subpackages, matching the two training phases in [`../training`](../training).

## `tokenizer/` — the Phase-A representation (shared by everything)

| Module | Role |
|---|---|
| `filterbank.py` | fixed physical-Hz constant-Q filterbank + signed per-channel DC/gravity feature |
| `preprocess.py` | gravity alignment and per-window preparation |
| `transformer.py` | RoPE physical-time dual-branch transformer blocks |
| `encoder.py` | the config-conditional set encoder over channel/patch tokens |
| `channel_text.py` | channel-role + sensor-identity text conditioning (factored or per-channel) |
| `primitives.py` | named diagnostic primitives (probes, not training targets) |

Rate-invariant and channel-independent: filters are placed in physical Hz, so a 25 Hz and a 100 Hz
recording of the same motion land on the same features, and channels are encoded as a set rather
than a fixed-width vector.

## `evidence/` — the Phase-B evidence engine

| Module | Role |
|---|---|
| `patch_retrieval.py` | per-query-patch retrieval over the memory bank, with learned EMA subspaces |
| `decoder.py` | candidate-aware decoder (`QUERY` / `EVIDENCE` / `CANDIDATE` structural roles) |
| `confidence.py` | separate correct-and-answerable confidence calibration |

Design of record:
[`../docs/design/PHASE_B_TRAINING_INTENT.md`](../docs/design/PHASE_B_TRAINING_INTENT.md). Historical
research and retracted branches live in
[`../docs/archive/EVIDENCE_ENGINE.md`](../docs/archive/EVIDENCE_ENGINE.md) and
[`../docs/archive/EVIDENCE_ENGINE_FINDINGS.md`](../docs/archive/EVIDENCE_ENGINE_FINDINGS.md).

There is no InfoNCE label-alignment head. Activity-label language enters at Phase B through candidate
content (and through the ConSE comparison bridge). Phase A remains activity-label-free while using
acquisition-configuration language for channel roles and sensor identity.
