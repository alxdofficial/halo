# Phase-B Version Registry

> **Authoritative registry for Phase-B versions, checkpoints, and results.**
>
> Last updated: 2026-08-24. Architecture is defined by
> [`COMPACT_EVIDENCE_ENGINE.md`](../design/COMPACT_EVIDENCE_ENGINE.md). The latest completed numbers
> are in [`RESULTS.md`](RESULTS.md). Historical mechanisms are retained only through the source tags,
> checkpoint-local `run_config.json` / `source.patch`, and artifact paths listed here.

## Status Labels

- **active, untrained**: implemented and verified, but no full checkpoint or external result exists;
- **completed**: a full run and matched evaluation exist;
- **historical**: valid as a prior experiment, but not an active code path;
- **failed attempt**: incomplete or numerically invalid; never use its checkpoint in a result table.

## Compact Phase-B Versions

| ID | date | retrieval row | learned operation | final decision | training C | source | checkpoint/result status |
|---|---|---|---|---|---|---|---|
| `PB-01-FULL-MIX-VOTE` | 2026-08-23 | patch/sensor | two-layer set attention refines semantic evidence vectors | text-weighted evidence vote | 2/4/8/16 | dirty `5a23b44`, exact `source.patch` in run | historical, completed |
| `PB-02-SCALAR-MIX-VOTE` | 2026-08-24 | patch/sensor | one-layer set attention emits one scalar per top-64 evidence row | text-weighted evidence vote | 8/16/32/64 | tag `phaseb-vector8-vote-20260824` | historical, completed |
| `PB-03-PAIRWISE-1NN` | 2026-08-24 | one pooled six-second recording | independent pairwise MLP scalar for every memory row | corrected nearest neighbor | 8/16/32/64 | run source `8d60b86`; base tag `phaseb-recording-reranker-pretrain-20260824` | latest completed |
| `PB-04-SET-SCALAR-1NN` | 2026-08-24 | one pooled six-second recording | one-layer set attention emits one scalar per top-64 evidence row | corrected nearest neighbor | 2/4/8/16 | tag `phaseb-contextual-scalar-reranker-20260824` after release | **active, untrained** |

The architecture ID is stored in every new `run_config.json` and checkpoint under
`phase_b_version`. Do not identify a run only as “latest,” “native engine,” or “the mixer.” Use the
ID and checkpoint directory together.

## What Changed

### PB-01 to PB-02

PB-01 allowed attention to rewrite evidence vectors before a semantic vote. PB-02 retained the
candidate/query/evidence attention context but constrained its learned output to one scalar per
retrieved row. The semantic vote remained, so this experiment did not isolate scalar reranking from
voting.

### PB-02 to PB-03

PB-03 pooled each six-second recording to one row, removed set attention and semantic voting, and
applied an independent pairwise scalar correction to every one of the 512 memory rows. It then used
corrected nearest neighbor. This isolated a simple reranker, but removed evidence-set context.

### PB-03 to PB-04

PB-04 keeps PB-03's recording rows and corrected-nearest decision. It restores PB-02's unordered
candidate/query/evidence attention context, but the only learned output remains one bounded scalar
per retrieved row. It cannot refine vectors, emit candidate logits, or vote. Raw cosine selects 64
rows; non-selected rows still receive encoder signal through the smooth nearest-neighbor backward
surrogate. Candidate counts return to 2/4/8/16.

## Completed Checkpoints

### PB-01-FULL-MIX-VOTE

- checkpoint: `training/tokenizer/outputs/e2e_compact_35k_20260823/best.pt`
- checkpoint SHA-256: `d7e5690a1970fbd407cba6e5776a2d0cf2cd85d14f99eeb8d8be42f0a02d8754`
- selected step: 10,000 of 35,000
- wall time: 3,679.99 seconds
- exact source: `training/tokenizer/outputs/e2e_compact_35k_20260823/source.patch`
- raw evaluation: `eval/adaptation_results/e2e_compact_35k_20260823/`
- assembled evaluation: `eval/adaptation_tables/e2e_compact_35k_20260823/`
- status: historical; do not load with the active engine class

### PB-02-SCALAR-MIX-VOTE

- checkpoint: `training/tokenizer/outputs/e2e_compact_vector8_35k_20260824/best.pt`
- checkpoint SHA-256: `8330d868e1151f4137ca32ff6c4b06684c68c58ae17f4c48a56a347f94a0e4ca`
- selected step: 25,000 of 35,000
- wall time: 5,356.87 seconds
- source tag: `phaseb-vector8-vote-20260824`
- raw evaluation: `eval/adaptation_results/e2e_compact_vector8_35k_20260824/`
- status: historical; results were not promoted into the current headline table

### PB-03-PAIRWISE-1NN

- checkpoint: `training/tokenizer/outputs/e2e_recording_rerank_35k_v3_20260824/best.pt`
- checkpoint SHA-256: `b4ca46468e2708cfe52396a55cc98e343912b110f16075f057423f2beb1867da`
- selected step: 33,000 of 35,000
- wall time: 5,851.23 seconds, including the pre-cache episode-plan build
- exact clean source: commit `8d60b86`
- raw evaluation: `eval/adaptation_results/e2e_recording_rerank_35k_v3_20260824/`
- assembled evaluation: `eval/adaptation_tables/e2e_recording_rerank_35k_v3_20260824/`
- paper-facing summary: [`RESULTS.md`](RESULTS.md), explicitly labeled `PB-03-PAIRWISE-1NN`
- status: latest completed result; not the active architecture

## Failed Recording-Reranker Attempts

These directories are retained for diagnosis only and must never be used as completed checkpoints:

| directory | observed outcome | status |
|---|---|---|
| `e2e_recording_rerank_35k_20260824` | stopped around step 560; no summary | failed/incomplete |
| `e2e_recording_rerank_35k_v2_20260824` | became non-finite around step 2,800; no summary | failed/numerically invalid |
| `e2e_recording_rerank_35k_v3_20260824` | completed 35,000 finite steps | reportable PB-03 run |

The `v3` suffix belongs to the run repair history, not the architecture ID. In prose and tables call
it `PB-03-PAIRWISE-1NN`, not “v3.”

## Active Run Contract

The next full run must use:

```text
architecture:       PB-04-SET-SCALAR-1NN
output directory:   training/tokenizer/outputs/e2e_set_scalar_1nn_35k_20260824
steps:              35,000
episodes per step:  8
candidate counts:   2, 4, 8, 16
query labels:       up to 4 per episode
queries per label:  4
memory rows:        512 per episode
retrieval shortlist:64 per query
support k:          0, 1, 2, 4, 8, 16
aliases/augmentation: disabled
```

Expected result directories:

- raw: `eval/adaptation_results/e2e_set_scalar_1nn_35k_20260824/`
- assembled: `eval/adaptation_tables/e2e_set_scalar_1nn_35k_20260824/`

Do not overwrite PB-03 outputs. After PB-04 completes, add its checkpoint hash, selected step, wall
time, and exact result paths here before changing [`RESULTS.md`](RESULTS.md).

## Legacy Pre-Compact Line

The 2026-08-17 rank-8 per-sensor admissibility-gate/Stage-2 system predates the compact series. Its
matched artifacts remain under `eval/adaptation_results/v1_d85761d_stage2/` and
`eval/adaptation_tables/v1_d85761d_stage2/`. Call it `PB-L0-ADMISSIBILITY-GATE`; do not describe it
as PB-01 or as the current model. The prior long-form ledger remains recoverable from Git history,
while this file is intentionally limited to the version map needed for current work.
