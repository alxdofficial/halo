# training/tokenizer/ — Pipeline A (Representation) harness

**Pipeline A** of the evidence engine: the *learnable representation pipeline* — everything that
turns raw signal into a heterogeneity-salient representation. Not a fixed transform; it is trained.

Covers: time-domain **preprocessing** (gravity-align), **cross-channel relational/causal learning**
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
| objectives | **A1 masked-recon (1.0) + A2 SimCLR (NT-Xent) + TF-C-inspired (0.25)** | A3 grounding OFF (`a3_weight=0.0`) |
| A2 mode | `simclr` (label-free two-view NT-Xent, τ=0.1) | `--a2-mode supcon` = legacy label ablation |
| TF-C | `tfc_weight=0.25`, τ=0.1 | rate/position-aware time branch; TF-C-*inspired*, not faithful; {0,0.1,0.25,1} ablation pending |
| sampler | **temperature**, α=0.5 (P∝n^0.5), within-batch no-replacement | `--sampler balanced` needed only for supcon |
| conditioning | **factored** (axis role + per-sensor identity) | CLI default (F8); `--text-conditioning per_channel` = ablation |
| frontend / MR | `fixed` / multiresolution ON | `--arm learnable`, `--no-multiresolution` to ablate |
| batch / lr / warmup / steps | 512 / 4.2e-4 / 1000 / 30000 | cosine LR; wd 0.05, grad-clip 1.0 |
| corpus | 12 datasets, native grids, ~305k/38k win, 93 labels | temperature-sampled, all placements in the encoder stream |
| augmentations | `default_v2` | window-crop, channel-dropout, **SO(3) rotation (gravity-removed included)**, gravity p=0.15, rate, warps, jitter/scale, channel-text phrase/dropout, **sensor-text dropout**; label-text OFF in pretrain |
| resume | full-config validated; only device/num_workers/eval-cadence may differ | fresh sampler epoch for remaining steps |

**Paper launch (all of the above are defaults):**
```bash
python -m training.tokenizer.pretrain --out training/tokenizer/outputs/<run>
```
Dump the live config any time with:
`python -c "from dataclasses import asdict; from training.tokenizer.pretrain import PretrainConfig; import json; print(json.dumps(asdict(PretrainConfig()), indent=2))"`.
