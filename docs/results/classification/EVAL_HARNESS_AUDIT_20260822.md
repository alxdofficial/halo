# Eval-harness audit — 2026-08-22

Requested sweep of the code behind the RESULTS.md zero-shot table (`eval/run_baselines.py` →
`baselines/*/adapter.py` → `eval/scoring.py` → `eval/assemble_table.py`), checking every model is
used according to its developers' original intent. Verified against the upstream repos on disk
(`legacy_code/auxiliary_repos/{LIMU-BERT-Public,CrossHAR,UniMTS,NormWear}`, the pinned
ssl-wearables hub cache) plus functional/numeric spot checks.

## Verified sound

| layer | check | result |
|---|---|---|
| runner | atomic cell writes, stale-delete before run, disclosed N/A, failures recorded not skipped, protocol stamped into every cell, cell locks | ✅ by construction, code read |
| assembler | rejects STALE (protocol mismatch), MISSING and BAD cells loudly | ✅ |
| ground truth | offset-free, canonical-synonym translation, OOV drop; functional test passes | ✅ |
| ConSE bridge | one-hot probs → nearest target label (functional); target labels enter only as projection space (no leakage); ONE frozen SBERT for all bridged models; ONE 2-layer probe for every ConSE baseline | ✅ |
| CIs | subject bootstrap / small-cohort jackknife, frozen class set per replicate, degenerate-CI guard | ✅ |
| eval data | all 14 primary streams measured: acc in g with gravity (DC 0.95–1.05), gyro rad/s (med 0.04–0.59) | ✅ |
| harnet | hubconf-verified (N,3,150)=5s@30Hz; resample timing vs analytic sine err 9e-4; subject-disjoint head-fit; cache fingerprint covers vocab/split/probe/cap/corpus/backbone; hub tag pinned both paths | ✅ |
| crosshar | per-window InstanceNorm == upstream IMUDataset exactly (scale-invariant ⇒ unit convention moot) | ✅ |
| unimts | 20 Hz / wrap-pad-200 / m/s²+gravity / single-SMPL-joint == upstream data.py+contrastive.py; encode_image path identical | ✅ |
| imagebind | (6,2000)@200 Hz architecture-derived; zero-pad == IMU2CLIP padIMU; both towers L2-normed | ✅ |
| normwear | native L1 matching, 65 Hz, channel-independent, real channels only via stream.mask | ✅ |
| halo | identical ConSE treatment (same probe/seed/split); native input per contract; head cache stamped with backbone content hash | ✅ |

## Findings

**F1 — LiMU-BERT input scale is 9.8× below the upstream design point (moderate; limubert only).**
Upstream `Preprocess4Normalization` divides acc by 9.8 assuming m/s² input, so their model sees
acc ≈ 1 (g units) vs gyro O(0.1–0.5) rad/s. Our grids already store g; both `train.py:96` and the
adapter's `_normalize` divide by 9.8 **again**, so our backbone sees acc ≈ 0.10 — the acc:gyro
relative scale is ~10× off the design the architecture was tuned for. It is SELF-CONSISTENT
(pretrain and eval share the convention), so the published number is internally valid, but the
convention plausibly depresses limubert's ceiling (it is the lowest self-pretrained row, 32.2).
Fix if desired: drop the division (grids are already g) and re-pretrain backbone + head — a
retrain, not an eval patch. Not applied in this sweep.

**F2 — UniMTS gets weaker label text than its own evaluation used (minor).**
Upstream joins each class's full `label_dictionary` synonym list into one string
(`' '.join(labels)`); our adapter passes the single canonical label. Defensible (symmetric with
what ConSE models see) and comparable with our published row, but a paraphrase-ensemble variant
would be closer to upstream intent and likely help UniMTS. Not applied; would need a disclosed
protocol change.

**F3 — noted only:** `predict_from_similarity` argmax breaks ties toward the first candidate
(measure-zero for real cosine scores); monipar has no gyroscope, so 6-channel baselines get
zero-filled gyro there (their documented "gyro absent" encoding — benign, disclosed).

No correctness bug was found in the scoring path itself.
