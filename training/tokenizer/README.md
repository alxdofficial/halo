# training/tokenizer/ — Pipeline A (Representation) harness

**Pipeline A** of the evidence engine: the *learnable representation pipeline* — everything that
turns raw signal into a heterogeneity-salient representation. Not a fixed transform; it is trained.

Covers: time-domain preprocessing, **cross-channel relational/causal learning**
(masked-channel set model → residuals), the **learnable frequency domain** (fixed physical
filterbank + constrained-learnable scattering/SincNet), and the **primitives**.

- **Design / plan:** [`docs/design/EVIDENCE_ENGINE.md`](../../docs/design/EVIDENCE_ENGINE.md) ·
  [`docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md`](../../docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md) ·
  [`docs/design/LEARNABLE_TOKENIZER_ARM.md`](../../docs/design/LEARNABLE_TOKENIZER_ARM.md).
- **Phase 1 training:** masked-channel SSL + config-conditional salient-contrastive +
  analysis-consistency (M0–M3). Validated by the robustness probe *before* Pipeline B exists.
- **Model components:** `model/tokenizer/`.
- **Seam to Pipeline B:** emits `{query representation + structured primitives + per-channel text id}`.

The fixed and constrained-learnable Phase-A arms are both wired end to end. The learnable preset
enables simultaneous short/long token grids; `--frontend` and `--[no-]multiresolution` expose the
two attribution diagnostics. Examples:

```bash
python -m training.tokenizer.pretrain --arm fixed --out training/tokenizer/outputs/fixed_run
python -m training.tokenizer.pretrain --arm learnable --out training/tokenizer/outputs/learnable_run
```

## Phase-A launch recipe — AUTHORITATIVE (current defaults; supersedes the preflight docs)

The other Phase-A docs (`docs/design/NATIVE_PRETRAIN_PREFLIGHT.md`, `PIPELINE_A_PREFLIGHT.md`,
`AUGMENTATIONS.md`) predate the SSL pivot and describe the OLD recipe (SupCon, A3 0.1, balanced
sampling, gravity-gated rotation, old batch/LR). The live source of truth is
`PretrainConfig` + `main()` in `training/tokenizer/pretrain.py`; this table mirrors them:

| knob | current default | note |
|---|---|---|
| objectives | **fixed A1 (1.0) + EMA-latent A1 (0.1) + A2 VICReg + TF-C VICReg (0.25) + placement VICReg (0.1) + A3 (0.05)** | all activity-label-free |
| A2 mode | `vicreg` (two augmented positives, no negatives) | `simclr` and label-based `supcon` are controls |
| TF-C | `tfc_weight=0.25`, VICReg | rate/position-aware time branch; `--tfc-loss nt_xent` is the historical control |
| placement | weight 0.1; 10% window-pair quota | only explicit simultaneous `event_ids`; a real run fails if grids predate them |
| EMA target | weight 0.1; decay 0.996 | clean stop-gradient teacher, masked student predictor; checkpointed/restored |
| sampler | **hierarchical temperature**, dataset α=0.25 with 25% ceiling, subject α=0.5, within-batch no-replacement | label-free; `--sampler balanced` needed only for SupCon |
| conditioning | **factored** (axis role + per-sensor identity) | CLI default (F8); `--text-conditioning per_channel` = ablation |
| frontend / MR | `fixed` / multiresolution ON | `--arm learnable`, `--no-multiresolution` to ablate |
| batch / lr / warmup / steps | 512 / 4.2e-4 / 1000 / 30000 | cosine LR; wd 0.05, grad-clip 1.0. **batch 512 does NOT fit a 24 GB card — see the memory note below before launching** |
| corpus | 12 datasets, **1,542,518/186,269** native train/val windows, 93 labels, 2,858.3 materialised hours | uncapped Capture-24; temperature-sampled, all placements in the encoder stream. Counts are post-quality-screen (95 implausible + 1,003 duplicate dropped) at the default `data_seed=20260718` |
| augmentations | `default_v2` | window-crop, channel-dropout, **SO(3) rotation (gravity-removed included)**, gravity p=0.15, rate, warps, jitter/scale, channel-text phrase/dropout, **sensor-text dropout**; label-text OFF in pretrain |
| resume | full-config validated; only device/num_workers/eval-cadence may differ | checkpoints missing new trajectory fields are rejected |

**Paper launch (all of the above are defaults):**
```bash
python -m data.scripts.scan_implausible          # quality screens; CorpusIndex + build_memory
python -m data.scripts.scan_duplicates           # both read data/quality/*.json
python -m data.scripts.build_grids --alignment native
python -m training.tokenizer.pretrain --device cuda --batch 16 \
    --out training/tokenizer/outputs/<run>
```
`--device cuda` is **required**: the parser defaults to CPU, so omitting it silently starts a
CPU run. The quality scans must be run before training — `scan_duplicates.load()` returns an
empty screen when its cache is absent, which is why the loaders now call it with `require=True`.

> **Memory (measured 2026-07-26, RTX 4090 / 24 GB).** The configured `batch_size=512` OOMs before
> step 1. Batch 512/256/128/64/48/32 all OOM; batch 16 runs at 12.7 GB and 6.5 steps/s. The
> `# peak 9.5 GB` note in `PretrainConfig` predates the TF-C rail, the EMA teacher, placement
> VICReg and multiresolution-by-default — the step now runs four gradient-carrying encoder passes
> plus two no-grad ones plus `TimeEncoder`. Peak is also a *random variable*: `patch_seconds` is
> drawn per batch, so token count P swings 12→22 and an unlucky draw OOMs a batch size that
> survived the previous 60 steps. Treat batch 16 as the only verified setting until the token
> budget is capped.
>
> `steps=30_000` is **~10 corpus passes at batch 512**, not the 51 the code comment claims (that
> figure assumed a ~300k-window corpus; it is now 1.54M). At batch 16 it is 0.31 passes / 1.3 h.
The grid rebuild is a required one-time migration for explicit simultaneous-event identities.
Dump the live config any time with:
`python -c "from dataclasses import asdict; from training.tokenizer.pretrain import PretrainConfig; import json; print(json.dumps(asdict(PretrainConfig()), indent=2))"`.

## Optional scale experiment

ExtraSensory and a bounded NHANES PAX80_G subset are fully wired but deliberately opt-in. They are not
silently mixed into the paper's matched 12-source technique comparison. The current local pilot adds
three ExtraSensory streams and one NHANES stream, bringing the expanded materialisation to 4,148.5
hours and the current subject split to 2,260,852 train / 243,054 validation windows. NHANES has no
activity annotations: its reserved `__unlabeled__` marker is excluded from semantic vocabulary,
validation probes, and Phase B.

```bash
# ExtraSensory: fetch/import -> convert -> native grids
python -m data.datasets.extrasensory.fetch
python -m data.datasets.extrasensory.convert
python -m data.scripts.build_grids --dataset extrasensory --alignment native

# NHANES: an explicit bounded subset is mandatory
python -m data.datasets.nhanes.fetch --subjects 8
python -m data.datasets.nhanes.convert --max-hours-per-subject 24
python -m data.scripts.build_grids --dataset nhanes --alignment native

# Expanded Phase-A run: the exact roster is persisted in run_config.json
python -m training.tokenizer.pretrain \
  --datasets uci_har hhar pamap2 wisdm kuhar unimib_shar mhealth capture24 \
             sp_sw_har nfi_fared harmes xrf_v2 extrasensory nhanes \
  --out training/tokenizer/outputs/expanded_scale
```

The local publications/protocol documents are under `references/datasets/{capture24,extrasensory,nhanes}`.
PAAWS was considered but is not integrated because its official release endpoint returns HTTP 403 from
this machine; its public sample/parser is sufficient to inspect format, but not to validate released
participant bytes.
