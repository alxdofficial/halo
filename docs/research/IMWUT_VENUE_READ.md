# IMWUT venue read and recalibration (2026-09-03)

Brief read of the target venue, then a recalibration of the paper structure, the model design and
the experiment plan in `docs/design/IMWUT_COMPARE_DESIGN.md`. Thinking stage; nothing built.

Sources: IMWUT author guidelines (dl.acm.org/journal/imwut/author-guidelines), the UbiComp/ISWC
2025-26 author guides, the founding editorial (Abowd, Kostakos, Santini, Scott, Yatani 2017), and the
2024-26 IMWUT HAR-foundation-model papers already in our reference set.

---

## 1. What the venue is, mechanically

| item | fact | consequence for us |
|---|---|---|
| Cadence | **Three deadlines a year: Feb 1, May 1, Nov 1** (was four; changed ~2024). Issues Mar/Jun/Sep/Dec. | Next: **Nov 1 2026** (~8 weeks). Then **Feb 1 2027**. |
| Review | ~6-8 weeks to a decision: accept-with-minor, **major revision (resubmit within 6 months)**, or reject. | Major revision is the normal path (first cycle ever: 22 of 41 went to major revision). Plan for one round of revision, not for a one-shot accept. |
| Length | No hard limit; **8,000-10,000 words (13-16 pages) excluding refs/figures is "appropriate"**; longer gets extra scrutiny. | Our experiment list is bigger than a 10k-word paper. Prioritise (Section 3). |
| Blinding | **Double-blind**, desk reject for violations. | The HALO arXiv preprint (2608.27233) exists under our name. The submission must not self-identify: no "our prior HALO", anonymised repo, cite the preprint in third person or not at all. |
| Resubmission | A previously rejected IMWUT paper must attach the old reviews and a change summary. | Not our case (MobiCom rejection is a different venue). |
| Presentation | Accepted papers present at UbiComp/ISWC the following year. | Nice-to-have, not planning-relevant. |
| Datasets | Dataset-as-contribution has its own guidelines. | We are not a dataset paper; the manifest is infrastructure, not a claimed contribution. |

## 2. What the venue rewards (from its own author guidance)

The ISWC/IMWUT author guide is unusually explicit. The questions it says reviewers ask, and how we
stand:

1. **"What is the problem?"** — must be crisp in the abstract and hinted in the title. The senior
   reviewer is chosen from this alone. *Ours*: "heterogeneous sensor setups break HAR models; recognise
   by comparing to a few compatible labelled recordings instead of classifying in isolation." Crisp.
2. **"What has been done before, and how is this different?"** — *specific* comparisons, not a list of
   vaguely related work. *Ours*: we must compare against, by name, (a) matching/prototypical networks
   (Vinyals 2016, Snell 2017) — the comparator is a matching network with a language readout, say so;
   (b) ZARA — same "features + reference recordings" structure, LLM replaced by a 1M comparator;
   (c) UniMTS/Wonderwall — invariance/simulation approaches to the same heterogeneity; (d) GOAT/LanHAR/
   IMUZero — language-aligned open-set; (e) CrossHAR — the IMWUT cross-dataset baseline everyone cites.
3. **"What did you accomplish, and can you prove it?"** — "if you find this part hard to write the work
   is not finished." *Ours*: the enrollment k-curve is the proof; the untrained floor and step-0 control
   are what make it a proof rather than a number.
4. **"Enough detail to replicate?"** — the question that "generates the most discussion". *Ours*:
   release code, checkpoints, the frozen manifest with fingerprints, and every baseline adapter. Our
   fairness/faithfulness contract is a strength here; state it in the paper, not only in the repo.
5. **"Do the numbers lie?"** — say whether measured/simulated/derived, on what hardware. *Ours*: seeds,
   CIs, the chance and constant-predictor floors, and the cost table with the device named.
6. **Scope discipline** — "if you solved a single practical problem, don't generalise it for
   publication." *Ours*: do not call this a foundation model or in-context learning in the abstract.
   Call it what it is: comparison-based recognition with a small learned comparator.

What the recent IMWUT HAR-FM papers actually do (CrossHAR, oneHAR, IMUZero, Wonderwall, Customizable
FM, MASTER): 4-10 public datasets, zero-shot + few-shot + (often) full fine-tune columns, an ablation
table, released code, and almost always a deployment or efficiency element. Our 7 held-out datasets,
k-curve, ablations and cost table are squarely in the pattern.

## 3. Recalibration

### 3.1 Paper structure — keep, with three changes

- **Name the contribution type in the introduction**: a method paper with a rigorous cross-dataset
  evaluation. Not a system, not a benchmark, not a dataset.
- **Lead with the deployment story, not the ML story.** IMWUT is a ubicomp venue. The k-curve *is* an
  enrollment story: a user records a few labelled examples on their own device and the recogniser
  compares against them. Write the introduction from that side — "enrollment on the user's own
  sensor" — and let few-shot/zero-shot be the technical names in the method section.
- **Budget the experiments to ~10k words.** Main paper: main k-curve table, per-dataset breakdown,
  heterogeneity-axis figure, one ablation table with the four load-bearing rows (untrained floor /
  step-0, comparator vs 1-NN vs prototype, compatibility filter on/off, p sweep), Arm B as one
  subsection, cost table. Supplement: filterbank variants, warm-start vs scratch, text-vote vs
  one-hot + scrambled vocabulary, seeds detail, per-baseline contract notes.

### 3.2 Model design — unchanged, one framing note

Nothing in the venue read argues against the Arm A design. One addition to how it is *described*:
position the comparator explicitly as a matching network (attention over support, label read from
support) so the reviewer who thinks "this is just matching networks" finds that sentence in our
related work with the delta stated: language-space readout over verbatim labels, config-compatible
support, and the joint ZS/FS curriculum with soft zero-shot targets.

### 3.3 Experiments — two additions, one demotion

- **Add**: a *deployment enrollment* experiment that mirrors real use — support drawn from one
  held-out subject's own recordings (same-subject enrollment) alongside the cross-subject curve we
  already plan. Our old step-0 control already showed same-subject k=2 at 62 vs cross-subject 49; that
  gap is the ubicomp headline figure, not a footnote.
- **Add**: name the device in the cost table (a phone CPU and, if feasible, a watch-class core), report
  measured latency for encode-K-support + query, and memory. "CPU time measurements are meaningless
  unless the reader is told the machine."
- **Demote**: the acquisition-text-ON ablation on Arm A is redundant with Arm B; drop it from the
  ablation table and let Arm B carry the question.

### 3.4 Baselines — the stance needs one sentence in the paper

"Released checkpoints only; training regimen and data are part of each method" is defensible and has
precedent in CrossHAR and Wonderwall, but reviewers will ask "did you train on the same data?". Put the
answer in the evaluation-protocol section: we compare deployed systems as released, on data none of
them saw, with one shared enrollment protocol and ConSE as the open-set bridge for closed-set models.

### 3.5 Timeline — the honest read

Nov 1 2026 is eight weeks away and requires: Phase-A single-res pretrain, the episode sampler and
comparator fine-tune, the baseline weight audit plus any new adapters, all evaluation runs with seeds,
and the writing. That is feasible only if nothing surprises us, which the ledger says is unlikely.
**Plan for Feb 1 2027 as the target, and treat Nov 1 as a stretch that we take only if the k-curve
result is in hand by early October.** Either way, the major-revision round is expected; budget the
following cycle for it.

## 4. Decisions this read changes

1. Target deadline: Feb 1 2027 (Nov 1 2026 opportunistic).
2. Introduction is written from the enrollment/deployment side.
3. Same-subject enrollment added as a first-class experiment; acquisition-text-ON ablation dropped
   from Arm A.
4. Related work must name matching/prototypical networks and ZARA as the nearest neighbours, with
   the delta stated.
5. Submission must be blind with respect to the HALO preprint.

None of these change the model or the support-set contract in the design doc.
