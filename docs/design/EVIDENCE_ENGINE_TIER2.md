# Evidence Engine — Tier-2 improvement plan

Status: plan, 2026-07-20. Supersedes nothing; extends `EVIDENCE_ENGINE.md` (§4.2 head, §5 objectives)
and `EVIDENCE_ENGINE_BUILD_PLAN.md` (M4–M6) with the concrete Tier-2 architecture + loss, informed by
the M4a diagnostic ([[halo-phaseB-m4a-results]]) and a literature pass.

## 0. Where we are (the facts this plan is built on)

M4a is built and diagnosed. On the 7-cell ZS-XD gate (same frozen fixed+MR encoder):

| configuration | mean macro-F1 |
|---|---|
| ConSE (fitted softmax head) | 42.7 |
| **UNtrained retrieval + text-ensemble** (raw features, `g=t=identity`) | **47.5** |
| harnet (external UK-Biobank baseline) | 47.3 |
| M4a **trained** head (learned `g,t` on closed-vocab CE) | 40.9 |

Diagnostic verdict (`diagnose.py`): **credit assignment is healthy** (grad on `g`,`t` both flow; purity
0.36→0.68) — the failure was the **loss**. Closed-vocab cross-entropy overfits the seen-label geometry
and *destroys* open-vocab transfer, so the untrained mechanism wins. Retrieval telemetry: purity flat
~0.68 across k (encoder ceiling), effective-k ~2200 (diffuse), hubness Gini 0.81. Worst classes
(stairs/ramp/elevator/smoking ≈ 0 F1) are fine-grained locomotion the *encoder* can't separate.

**The floor to beat is 47.5**, not 42.7. Every Tier-2 change must clear it on held-out configs or be
dropped ("do no harm").

## 1. Design principles

1. **Keep the sensor→sensor retrieval bridge.** Query key = frozen encoder vector; retrieve
   nearest *sensor* neighbors (`⟨g(z), g(z_i)⟩`); transfer to unseen labels *label-text→target-text*
   (`⟨t(label_i), t(c)⟩`). This IS a sensor↔text alignment — an indirect, retrieval-mediated one
   (kNN-LM/RETRO) — and it is the *safe* one: the sensor geometry stays frozen-encoder-owned and
   non-parametric, so it can't collapse the way a direct CLIP projection can. It already beats harnet.
2. **Frozen-target asymmetry.** Learn transforms on the **evidence side** (our known training labels —
   safe) but keep the **target-set** label text **frozen SBERT** (unseen/arbitrary — must stay
   open-vocab). Never fit anything on the candidate labels.
3. **Do no harm.** Every learned component is a **residual init at identity** (starts *as* the untrained
   mechanism) and is gated on beating 47.5 on held-out configs. Select checkpoints on **held-out-config
   transfer**, never closed-vocab val (which anti-correlates — the M4a trap).
4. **The loss must be transfer-aligned, not closed-vocab.** Episodic class-holdout + retrieval/contrastive
   objectives; closed-vocab CE is banned (it's what broke M4a). Confirmed by the zero-shot HAR SOTA
   (IMU2CLIP, UniMTS, ZeroHAR all train contrastive-to-text, freeze the text encoder, augment label text)
   and by episodic-ZSL theory (metric overfits seen classes unless trained on held-out episodes).

## 2. Target architecture

```
session ─▶ frozen encoder ─▶ per-PATCH vectors {z_p}  (and pooled z)
                                     │
                    per-patch sensor→sensor retrieval over frozen memory bank
                                     │  top-k neighbors per patch (each carries a known label)
                                     ▼
        EVIDENCE DECODER (transformer):
          • self-attention MIXES information across the retrieved evidence set + across patches
          • query-conditioned REFINEMENT of each neighbor's label text:
                t'(label_i) = t(label_i) + Δ(t(label_i), z_query, evidence_set)      (residual, reg→0)
          • vote:  e_c = Σ_i a_i · relu(⟨ t'(label_i), t_frozen(c) ⟩)                 (target text FROZEN)
          • attention-weighted accumulation across patches (MIL: signature is sparse → weight, don't average)
                                     ▼
        EVIDENTIAL HEAD: density gate ρ (FIX-B) → Dirichlet α=ρ·e+1 → expected prob (predict)
                          + belief/vacuity (FIX-A) → candidate-set-invariant reject score (FIX-C) → ABSTAIN
```

Deltas from M4a: (a) **per-patch** retrieval + attention accumulation instead of one pooled query;
(b) an **evidence decoder** (attention among evidence) replacing the independent weighted sum;
(c) **query-conditioned label-text refinement** (residual, evidence-side only); (d) the **Dirichlet/
abstain** head (M5). The sensor→sensor + text→text skeleton is unchanged.

## 3. Loss / training recipe (the core fix)

Train **episodically** with the encoder frozen (Lever A), the decoder/refiner as residuals-from-identity:

- **Episodic class-holdout.** Each step: sample a candidate label subset; with prob `p_holdout`, remove
  the query's true label from the candidate set AND its exemplars from the retrieval memory → the query
  must transfer to a *held-out* label via text (or, if truly unsupported, abstain). Trains the deployment
  condition, not seen-label memorization. (Prototypical/Matching-Net episodic training; L2G; MetaZero.)
- **Retrieval / contrastive objective, not closed-vocab CE.** Reward "right neighbors retrieved + right
  text-transfer to held-out label"; margin/InfoNCE over the episode's candidate set. Closed-vocab softmax
  CE is banned.
- **Reg-to-identity.** Penalize the decoder/refiner residual norm so it can only *improve* on the 47.5
  untrained mechanism, never destroy it.
- **Text-ensemble (already +2.2) + fine-grained descriptions.** Keep the label-paraphrase ensemble; add
  LLM biomechanical **descriptions** (ZeroHAR's 262% lever) for evidence + target labels — directly
  targets the stairs/ramp/elevator ≈0 F1 failures.
- **Two-regime EDL loss for M5** (KNOWN → grow evidence for y + KL wrong→uniform; NOVEL/held-out → drive
  vacuous → abstain), θ calibrated on val. Apply FIX-A/B/C.
- **Checkpoint selection = held-out-config transfer.**

## 4. Milestones + gates (each must beat the prior / the 47.5 floor on held-out configs)

- **T2.0 — Lock the baseline into the harness.** Wire the untrained retrieval + text-ensemble mechanism
  as the `halo_evidence` adapter in `eval/run_baselines` so 47.5 sits in the official table beside ConSE
  and harnet. Cheap, no training. *Gate: reproduces 47.5 through the harness.*
- **T2.1 — Fine-grained label descriptions.** Inference-only text upgrade (bare names → descriptions +
  ensemble), both evidence + target side. *Gate: lifts the fine-grained cells without hurting the mean.*
- **T2.2 — Loss redesign (frozen encoder).** Reinstate `g`/`t` as residuals; train episodic class-holdout
  + retrieval objective + reg-to-identity. *Gate: beats 47.5. If not, keep untrained and skip to T2.4.*
- **T2.3 — Evidence decoder.** Attention among evidence + query-conditioned label refinement (frozen
  target). *Gate: beats T2.2.*
- **T2.4 — Per-patch evidence + attention accumulation (MIL).** Patch-level memory + patch retrieval;
  decoder attends over patches×evidence; soft accumulation (not majority vote). *Gate: beats T2.3, esp.
  on fine-grained/free-living cells. Open risk: per-patch encoder features may be weaker than pooled —
  test first.*
- **T2.5 — Evidential head + abstention (M5).** Density gate + Dirichlet + candidate-set-invariant reject
  + class-holdout novelty. *Gate: ECE/OSCR sane; macro-F1 ≥ T2.4; abstains on novelty, answers on conflict.*
- **T2.6 — End-to-end tokenizer refinement (Lever B).** Unfreeze the tokenizer under the transfer-aligned
  loss with a momentum/slow memory encoder (avoid drift/collapse); warm-start from T2.3–T2.5. The only
  lever that lifts purity above 0.68. *Gate: purity ↑, beats T2.5. Kill if it degrades transfer.*
- **T2.7 — Continual-learning / few-shot memory augmentation** ([[task #146]]). Append seen labeled target
  exemplars to memory; sweep shots. *Gate: monotone few-shot curve; passes harnet on few-shot — the
  capability axis ConSE/harnet structurally lack.*

## 5. Risks & kill criteria

- **Learned text/decoder overfits again** → reg-to-identity + episodic holdout + do-no-harm gate; if any
  stage can't beat the untrained floor, ship untrained for that piece.
- **Per-patch features too weak** (encoder pooled for session) → fall back to pooled query; keep the
  decoder/refinement (they don't depend on patch granularity).
- **End-to-end (T2.6) drift/collapse** → momentum memory + low encoder LR + warm start; if unstable, keep
  encoder frozen — Tier-2 already beats harnet without it.
- **Strict zero-shot ceiling is the encoder** (purity 0.68). If T2.2–T2.5 plateau below a decisive harnet
  win, that's expected — the decisive win needs T2.6 and/or more Phase-A data; report honestly and lean on
  the capability axis (T2.7).

## 6. Sequencing

Frozen-encoder stages first (T2.0–T2.5) — cheap, safe, and the fastest path to a defensible "beats
harnet + calibrated abstention" result. Open the encoder (T2.6) only after the transfer-aligned loss is
proven, or it will distort the encoder the way closed-vocab CE distorted the bolt-on metric. T2.7 runs
any time after T2.0 (it only needs the retrieval mechanism).
