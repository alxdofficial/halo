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
| untrained @ top-k=48 (identity CONTROL for the decoder) | 46.7 |
| **T2.2+T2.3 trained evidence DECODER** (episodic class-disjoint loss) | **49.5** |

**RESULT (2026-07-20): the decoder clears the gate.** 49.5 > 47.5 floor, > harnet 47.3. Against its own
identity control at the same retrieval config (46.7 — top-k costs 0.8 vs full-soft), the decoder is worth
**+2.8**, so the gain is attributable to the decoder, not the retrieval change. Per-cell vs control:
motionsense 78.3→86.2, realworld 43.8→51.4, shoaib 49.2→55.7, tnda 52.2→55.6, inclusivehar 28.6→29.1,
**usc_had 20.0→15.9 and ut_complex 55.1→52.3 REGRESS** (ut_complex was the untrained standout). Internal
proxy: held-out-config × class-disjoint transfer bAcc 0.204→0.704. Training is cheap: 3000 steps ≈ 52 s
on the cached bank. **This is the first time learning has HELPED** — M4a's trained head was net-negative
(40.9), confirming the M4a diagnosis that the loss, not credit assignment, was the bug.

Diagnostic verdict (`diagnose.py`): **credit assignment is healthy** (grad on `g`,`t` both flow; purity
0.36→0.68) — the failure was the **loss**. Closed-vocab cross-entropy overfits the seen-label geometry
and *destroys* open-vocab transfer, so the untrained mechanism wins. Retrieval telemetry: purity flat
~0.68 across k (encoder ceiling), effective-k ~2200 (diffuse), hubness Gini 0.81. Worst classes
(stairs/ramp/elevator/smoking ≈ 0 F1) are fine-grained locomotion the *encoder* can't separate.

**The floor to beat is 47.5**, not 42.7. Every Tier-2 change must clear it on held-out configs or be
dropped ("do no harm").

### 0.1 Scale context — are we sized adequately? (measured 2026-07-20)

Parameter counts measured directly from the checkpoints on disk; pretraining scale per
`docs/baselines/BASELINES.md`.

| model | params | pretraining data |
|---|---|---|
| LiMU-BERT | 62.6 k | self-pretrained on **our** corpus |
| CrossHAR | 62.6 k | self-pretrained on **our** corpus |
| harnet5 (ssl-wearables trunk) | 4.24 M | UK-Biobank **~700 k person-days** |
| **HALO Phase-A encoder** | **7.17 M** | our corpus: 305,049 train windows / 20 streams / 93 labels; 25 k of 30 k steps |
| **HALO evidence decoder** | **2.81 M** | 3 k steps × 256 over 149,774 cached vectors — **52 s** |
| **HALO total** | **≈ 9.98 M** | |
| UniMTS | 68.6 M | HumanML3D mocap → simulated IMU + GPT-3.5 text |
| NormWear | 136 M (+ ~1.1 B TinyLlama text tower) | ~15 k signal-hours, mostly ECG/PPG/EEG |

**Read:** we are **not under-parameterized — we are under-*data*'d.**

1. *vs the model we actually beat.* HALO totals ~2.4× harnet's parameters, but harnet was pretrained on
   roughly **four orders of magnitude** more sensor data (700 k person-days ≈ 1.7×10⁷ h; our 305 k
   windows ≈ 10³ h assuming ~10 s windows — an order-of-magnitude estimate, not a measured duration).
   So the 49.5-vs-47.3 win is a **data-efficiency** result, not a scale win, and it is a *narrow* margin
   against a model with an enormous data advantage.
2. *vs the largest frozen SOTA.* We beat UniMTS and NormWear while being **7×–14× smaller** on the
   backbone (~130× counting NormWear's text tower). That efficiency gap is the more defensible claim.
3. *Is the ENCODER undersized?* It is the documented bottleneck (retrieval purity flat at 0.68;
   stairs/ramp/elevator ≈ 0 F1). But with only 305 k windows, that ceiling is far more likely
   **data-limited than parameter-limited** — widening a 7.2 M encoder on this corpus would mostly
   overfit. This is the quantitative case for the data-broadening plan and for T2.6, over making the
   encoder wider.
4. *Is the DECODER undersized?* Probably not — arguably already at the edge. It converged in 52 s
   (best at step 2200, drifting down after), it is a *residual* on a strong non-parametric mechanism,
   and its episodic supervision draws on only 53 training labels. The two regressions (usc_had,
   ut_complex) look like over-writing/over-fitting rather than insufficient capacity — testable via the
   λ sweep in the regression diagnosis.

**Consequence for how we report this:** lead with efficiency ("comparable-or-better zero-shot transfer at
7–50× fewer parameters and ~10⁴× less pretraining data"), not with an absolute-SOTA claim. Scaling *data*
is higher-leverage than scaling *parameters*.

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

## 2. Target architecture (locked decoder spec)

One **set-transformer** over a heterogeneous token set of query + evidence patches (across all sensors),
each token carrying: **config text** (which sensor/placement — text-keyed), **physical time** (window-
relative continuous-time features — see §2.2, *not* a global sequence index), a **role** (QUERY |
EVIDENCE), and for evidence tokens its **known label text**.

```
session ─▶ frozen encoder ─▶ per-PATCH vectors {z_p} per sensor  (encoder is already a text-keyed channel SET)
                                     │
              per-sensor sensor→sensor retrieval over frozen memory ─▶ top-k evidence patches (each: label + config)
                                     ▼
   token set = { QUERY patches (all sensors) ; EVIDENCE patches (all sensors, each w/ label) }
                                     │
                    ▼  SELF-ATTENTION (L layers): everything attends to everything
                    │     — mixes across sensors, patches, and evidence
                    │
   ┌────────────────┴────────────────┐
   ▼ (a) REFINE evidence label text   ▼ (b) importance weight a_i per evidence token
        (NOT just reweight — this is        (attention/MIL pooling weight)
         what disambiguates fine-grained:
         "walking" → "walking-upstairs")
        t'_i = t(label_i) + Δ(context_i)     residual, reg→0, EVIDENCE side only
   └────────────────┬────────────────┘
                    ▼  VOTE (target text FROZEN):  s_{i,c} = relu( ⟨ t'_i , t_frozen(c) ⟩ )
                    ▼  attention-pool over evidence: e_c = Σ_i a_i · s_{i,c}
                    ▼  EVIDENTIAL HEAD: density gate ρ (FIX-B) → Dirichlet α=ρ·e+1 → expected prob (predict)
                       + belief/vacuity (FIX-A) → candidate-set-invariant reject (FIX-C) → ABSTAIN
```

**Roles (do not blur):** QUERY patches have no label → they don't vote; they are **context** that
conditions the evidence refinement (a) and the pooling weights (b). **EVIDENCE** patches are the voters.
**Target** label text is **frozen SBERT** (unseen/arbitrary — never fit). We pool the *evidence* then run
the Dirichlet head — never pool argmax predictions (so calibration/abstention see accumulated evidence).

**The precision that matters:** the self-attention output must **refine the evidence label semantics**
(step a), not only reweight it (step b). If mixing only changes `a_i` and we vote with the raw label,
fine-grained classes (stairs/ramp/elevator ≈ 0 F1) can't be disambiguated. Refine-then-vote is the fix.

Deltas from M4a: (a) **per-patch/per-sensor** retrieval + attention accumulation instead of one pooled
query; (b) an **evidence decoder** (attention among evidence) replacing the independent weighted sum;
(c) **query-conditioned label-text refinement** (residual, evidence-side only); (d) the **Dirichlet/
abstain** head (M5). The sensor→sensor + (refined) text→text skeleton is unchanged. Full self-attention
(query+evidence) is safe — under class-holdout the query's label isn't in the evidence, so no leak.

### 2.1 Multi-sensor fusion (free, and a capability axis)

Because the encoder is a text-keyed channel **set**, a multi-sensor session (phone + watch, or xrf_v2's
6 placements incl. glasses/earbuds) is just a **bigger set** — nothing special. In the decoder it's the
same token set with tokens from multiple sensors. Retrieval is done **per sensor / per config-group**
(so a phone+watch query can pull phone-only *and* watch-only evidence from memory), then the union feeds
one self-attention. **Graceful degradation** is automatic: a missing device just shrinks the set.

This is a differentiator baselines structurally lack (harnet/UniMTS/ConSE are fixed-input): **inference-
time multi-device fusion, no retraining.** Requires a **data-pipeline change** — currently each placement
is a separate stream; build **joint sessions** from the paired datasets (SP-SW-HAR phone+watch, xrf_v2
6-placement). Controlled experiment: does phone+watch fusion beat phone-alone and watch-alone? (See T2.4b.)

### 2.2 Transformer hygiene (locked — get these right or the decoder silently underperforms)

The decoder is small (**L=2–4 layers, d=256** to match the encoder so patch vectors need no up-projection,
**4–8 heads**, head-dim 32–64). The mechanics below are not defaults-by-omission; each is chosen for the
fact that this is a **permutation-invariant set spanning multiple windows and sensors**, not a sequence.

**Normalization — pre-LN, always.** Each sublayer is `x = x + gate · sublayer(LN(x))` (pre-norm), with a
**final LN** before the refiner/pooling heads read the token states. Pre-LN because we train from scratch,
small, without the long warmup post-LN needs to stay stable; it also makes the identity-init below exact.
LN is per-token over the feature dim (standard). No BatchNorm anywhere (set sizes vary per session →
batch stats are meaningless).

**FFN — position-wise, every token, GELU, 4× expansion** (`d→4d→d`). Applied identically to query and
evidence tokens (it's per-token; roles are handled by the input embedding + attention, not by separate
FFNs). Dropout 0.1 on attn weights, FFN hidden, and residual.

**Attention — multi-head, `1/√d_head` scaling, full self-attention + key-padding mask.** Everything
attends to everything (the design), but sessions have variable #sensors / #evidence / #patches → a
**key-padding mask** (`-inf` on padded keys, masked-softmax) is mandatory. Guard the degenerate case: a
row must never be fully masked (every token attends to at least itself). Optional **QK-norm** (L2 on q,k
before the dot) if we see attention-logit blow-up; cheap insurance for a from-scratch small transformer.

**Positional encoding — THE trap here. No global sequence index.** The token set has no canonical order
across sensors or across evidence windows, and evidence windows come from *other sessions* whose absolute
clock is meaningless relative to the query. So:
- **Time = additive continuous-time Fourier features, measured relative to each token's own window start**
  (sinusoidal over the patch's physical center-time in seconds). This preserves intra-window temporal
  structure, is permutation-invariant across the set, and — because the query session's sensors share one
  origin — makes co-temporal cross-sensor patches (phone@2.0s ↔ watch@2.0s) line up. Each evidence window
  keeps its **own** origin (we match internal temporal *shape*, not the query's clock).
- **Explicitly do NOT use a single global RoPE / absolute index across the set.** Shared RoPE makes the
  q·k phase depend on `(t_a − t_b)`; across two different windows that offset is noise, so it would inject
  a spurious relative geometry between query and evidence. RoPE is fine *only* if scoped strictly
  intra-window — additive window-relative features are simpler and avoid the footgun, so that's the pick.
- Compose with (don't duplicate) Phase-A's duration embedding: since tokens are **multi-resolution**
  patches (short + long grids), each token also carries a **patch-duration** feature so the decoder can
  tell a 0.5 s patch from a 4 s one.

**Structural (non-positional) token features — how heterogeneity enters:**
- **Role** (QUERY | EVIDENCE): a 2-way *learned* embedding (only 2 values → parametric is safe).
- **Config / placement**: **text-keyed**, not a lookup — frozen SBERT of the config text, projected in.
  Keeps unseen placements open-vocab (a lookup table would break the whole thesis).
- **Label text** (evidence only): the frozen SBERT label vector, projected in (query tokens get a
  learned no-label token here so shapes are uniform).
- **Same-window co-membership**: a single **learned scalar relative bias** added to the attention logit
  when two tokens share a window. Permutation-invariant (unlike a per-window-index embedding, which would
  break set-invariance) and lets the model reason about "these patches are one session."

**Identity-at-init = the do-no-harm gate, made concrete at the transformer level.** The whole decoder must
boot up *equal to the untrained 47.5 mechanism*, then only improve. Three coordinated zero-inits:
1. **LayerScale gate γ per sublayer, init ≈ 0** (or equivalently zero-init each sublayer's output
   projection) → at step 0 every `x = x + γ·sublayer(...) ≈ x`, so token states pass through unchanged.
2. **Refiner Δ head zero-init** → `t'_i = t(label_i) + Δ(context_i) = t(label_i)` at init (no refinement
   until it earns it; §3 reg-to-identity keeps it honest).
3. **Pooling as a residual on retrieval weights**: `a_i = softmax(log w_retr,i + φ(token_i))` with **φ
   zero-init** → at init `a_i = w_retr,i`, exactly reproducing the untrained weighted-sum. Do *not* learn
   `a_i` from scratch (that would discard the retrieval prior and re-open the M4a overfit).
With all three, `decoder@init ≡ untrained mechanism`, so any measured gain is strictly additive and the
"beat 47.5" gate is a true ratchet.

**Optimization**: AdamW, weight decay on linears only (exclude LN/bias/γ/embeddings), linear warmup →
cosine, grad-clip 1.0. Everything trains under the episodic class-holdout loss of §3, encoder frozen.

## 3. Loss / training recipe (the core fix)

Train **episodically** with the encoder frozen (Lever A), the decoder/refiner as residuals-from-identity:

- **Episodic class-holdout.** Two ingredients, both needed:
  - **Varying candidate set per episode** — each step scores against a *different, restricted* label
    subset, not the full 59-way vocab. This alone stops the fixed-boundary closed-vocab overfit (the
    model must score against *whatever* set it's handed = the deployment condition).
  - **Hold the query's true label out of the retrieval memory**, in two regimes:
    - *keep it in the candidate set* → no same-label neighbors, correct answer still available → forces
      **transfer** via semantically-related neighbors + text (retrieve "walking" → text-transfer to
      "walking-upstairs").
    - *drop it from the candidate set* → class isn't an option → forces **abstain** (the M5 novelty
      signal). Which regime fires is the supervision for transfer-vs-abstain (semantic reachability).
  - **Why this is the fix:** M4a training is subject-disjoint but **label-PRESENT** — a "walking" query
    still retrieves "walking" from other subjects, so it only ever learns "retrieve same label" =
    closed-vocab discrimination = the overfit. Class-holdout removes that crutch. (Prototypical/Matching
    Nets; episodic ZSL [1]; L2G [11]; MetaZero [9].)
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

- **T2.0 — Lock the baseline into the harness.** ✅ DONE. `baselines/halo_evidence/adapter.py` wires the
  untrained retrieval + text-ensemble mechanism; reproduces **47.5** exactly through `eval.run_baselines`
  over the 7 primary cells (motionsense 80.8, realworld 44.2, shoaib 51.5, inclusivehar 29.0, usc_had
  19.0, tnda 53.4, ut_complex 54.9). *Gate met.*
- **T2.1 — Fine-grained label descriptions.** Hook built: `training/evidence/labeltext.py` appends a
  `data/labels/label_descriptions.json` anchor to the ensemble when present (absent ⇒ no-op, still 47.5).
  *Remaining: author the descriptions. Gate: lifts the fine-grained cells without hurting the mean.*
- **T2.2 — Loss redesign (frozen encoder).** ✅ **GATE PASSED (49.5 > 47.5).**
  `training/evidence/train_decoder.py`: class-disjoint episodes (hold out a label SET; memory excludes
  it; candidates = it) + reg-to-identity (Δ-norm + KL-to-retrieval-prior) + held-out-config ×
  class-disjoint checkpoint selection on fixed val episodes. Internal transfer bAcc 0.204→0.704;
  3000 steps ≈ 52 s. The transfer-aligned loss is what turned learning from net-negative to net-positive.
- **T2.3 — Evidence decoder.** ✅ **GATE PASSED (+2.8 over its identity control).**
  `model/evidence/decoder.py` implements the §2.2 spec (pre-LN + LayerScale, window-relative Fourier
  time, role/text-config/label embeddings, same-window bias, zero-init Δ + pooling-as-residual). 9 unit
  tests incl. **identity-at-init ≡ untrained mechanism** and set-permutation invariance — which is what
  makes `--untrained` in `eval_decoder.py` an exact control. *Open: usc_had (−4.1) and ut_complex (−2.8)
  regress; diagnose before T2.4 (both are the cells where untrained retrieval was strongest).*
- **T2.4 — Per-patch evidence + attention accumulation (MIL).** Patch-level memory + patch retrieval;
  decoder attends over patches×evidence; soft accumulation (not majority vote). *Gate: beats T2.3, esp.
  on fine-grained/free-living cells. Open risk: per-patch encoder features may be weaker than pooled —
  test first.*
- **T2.4b — Multi-sensor fusion (capability axis).** Build joint multi-sensor sessions from paired
  datasets (SP-SW-HAR phone+watch; xrf_v2 6-placement); per-sensor retrieval → union → decoder. *Gate:
  phone+watch fusion beats phone-alone and watch-alone on shared activities; graceful degradation when a
  device is dropped. A capability harnet/ConSE structurally lack (fixed-input).*
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
