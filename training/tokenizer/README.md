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
| objectives | **fixed A1 (1.0) + EMA-latent A1 (0.1) + A2 VICReg (1.0) + placement VICReg (0.1, invariance-only) + A3 (0.05)** | all activity-label-free. TF-C **dropped** (`tfc_weight=0`) — see the memory note |
| A2 mode | `vicreg` (two augmented positives, no negatives) | `simclr` and label-based `supcon` are controls |
| TF-C | **off** (`tfc_weight=0`) | its two views are the same unaugmented window through two networks, and the time branch is unanchored + discarded; `--tfc-weight 0.25` restores it for the ablation |
| placement | weight 0.1; 10% window-pair quota | only explicit simultaneous `event_ids`; a real run fails if grids predate them |
| EMA target | weight 0.1; decay 0.996 | clean stop-gradient teacher, masked student predictor; checkpointed/restored |
| sampler | **hierarchical temperature**, dataset α=0.25 with 25% ceiling, subject α=0.5, within-batch no-replacement | label-free; `--sampler balanced` needed only for SupCon |
| conditioning | **factored** (axis role + per-sensor identity) | CLI default (F8); `--text-conditioning per_channel` = ablation |
| frontend / MR | `fixed` / multiresolution ON | `--arm learnable`, `--no-multiresolution` to ablate |
| batch / lr / warmup / steps | **256** / 4.2e-4 / 1000 / 30000 | cosine LR; wd 0.05, grad-clip 1.0; `MAX_BATCH_TOKENS=6144` bounds peak VRAM |
| corpus | 12 datasets, **1,542,518/186,269** native train/val windows, 93 labels, 2,858.3 materialised hours | uncapped Capture-24; temperature-sampled, all placements in the encoder stream. Counts are post-quality-screen (95 implausible + 1,003 duplicate dropped) at the default `data_seed=20260718` |
| augmentations | `default_v2` | window-crop, channel-dropout, **SO(3) rotation (gravity-removed included)**, gravity p=0.15, rate, warps, jitter/scale, channel-text phrase/dropout, **sensor-text dropout**; label-text OFF in pretrain |
| resume | full-config validated; only device/num_workers/eval-cadence may differ | checkpoints missing new trajectory fields are rejected |

**Paper launch (all of the above are defaults):**
```bash
python -m data.scripts.scan_implausible          # quality screens; CorpusIndex + build_memory
python -m data.scripts.scan_duplicates           # both read data/quality/*.json
python -m data.scripts.build_grids --alignment native
python -m training.tokenizer.pretrain --device cuda \
    --out training/tokenizer/outputs/<run>
```
`--device cuda` is **required**: the parser defaults to CPU, so omitting it silently starts a
CPU run. The quality scans must be run before training — `scan_duplicates.load()` returns an
empty screen when its cache is absent, which is why the loaders now call it with `require=True`.

> **Memory (re-measured 2026-07-26, RTX 4090 / 24 GB).** Two changes made the configured batch
> reachable again. (1) TF-C is off by default — the discarded 174k-param time branch cost 3.5x the
> step time and 22x the VRAM of the 7.17M-param encoder, and alone capped the batch at 16.
> (2) `MAX_BATCH_TOKENS` caps batch x patches, so `patch_seconds` can no longer draw a batch that
> OOMs one that survived the previous sixty steps.
>
> | batch | peak | placement pairs | note |
> |---|---|---|---|
> | 256 (default) | 12.0 GiB | 13 | recommended |
> | 512 | 13.6 GiB | 26 | runs, but the cap forces coarse patch pairs |
>
> The cap has a cost the new `a1_unmasked_frac_by_source` telemetry makes visible: coarser pairs
> mean fewer tokens per window, so short-window sources lose more A1 supervision. Measured
> sp_sw_har (1.00 s windows) 0.38-0.40 at batch 256 vs 0.60-0.65 at 512; uci_har (2.56 s) 0.00 at
> 256 vs 0.20-0.22 at 512. That is why 256 is the default rather than 512.
>
> `steps=30_000` is **~6 corpus passes at batch 256** on the 1.54M-window corpus. The old
> "51 passes" comment assumed a ~300k corpus and is gone.

The grid rebuild is a required one-time migration for explicit simultaneous-event identities.
Dump the live config any time with:
`python -c "from dataclasses import asdict; from training.tokenizer.pretrain import PretrainConfig; import json; print(json.dumps(asdict(PretrainConfig()), indent=2))"`.

## Optional scale experiment

ExtraSensory, a bounded NHANES PAX80_G subset, and H-MOG are fully wired but deliberately opt-in.
They are not silently mixed into the paper's matched 12-source technique comparison. The current
local pilot adds three ExtraSensory streams, one NHANES stream, and one H-MOG phone-in-hand stream,
bringing the expanded materialisation to 25 streams / 4,467.76 hours and the current subject split to
2,382,969 train / 277,724 validation windows after quality screens. NHANES has no activity
annotations: its reserved `__unlabeled__` marker is excluded from semantic vocabulary, validation
probes, and Phase B.

```bash
# ExtraSensory: fetch/import -> convert -> native grids
python -m data.datasets.extrasensory.fetch
python -m data.datasets.extrasensory.convert
python -m data.scripts.build_grids --dataset extrasensory --alignment native

# NHANES: an explicit bounded subset is mandatory
python -m data.datasets.nhanes.fetch --subjects 8
python -m data.datasets.nhanes.convert --max-hours-per-subject 24
python -m data.scripts.build_grids --dataset nhanes --alignment native

# H-MOG: accept the official non-commercial research terms -> verify -> convert -> native grid
python -m data.datasets.hmog.fetch --accept-license
python -m data.datasets.hmog.convert
python -m data.scripts.build_grids --dataset hmog --alignment native

# Expanded Phase-A run: the exact roster is persisted in run_config.json
python -m training.tokenizer.pretrain \
  --datasets uci_har hhar pamap2 wisdm kuhar unimib_shar mhealth capture24 \
             sp_sw_har nfi_fared harmes xrf_v2 extrasensory nhanes hmog \
  --out training/tokenizer/outputs/expanded_scale
```

The local publications/protocol documents are under
`references/datasets/{capture24,extrasensory,nhanes,hmog}`.
PAAWS was considered but is not integrated because its official release endpoint returns HTTP 403 from
this machine; its public sample/parser is sufficient to inspect format, but not to validate released
participant bytes.
