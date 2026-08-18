#!/usr/bin/env bash
# Phase-A wave 2 (2026-08-18): objective vs the diagnosed defects, as a 2x2.
#
#                     | fixes OFF (depth 6, no aug)   | fixes ON (depth 3, jitter+scale)
#   JEPA  (EMA latent)| = wave-1 a_clean, 0.8284      | f_jepa_fixes
#   MAE   (fixed phys)| g_mae_plain                   | h_mae_fixes
#
# "fixes" are the two defects measured on 2026-08-18 (docs/results/PHASE_A_RECOVERY_20260818.md):
#   depth  - activity information peaks at depth 1-3 and decays 4-8 points by depth 6 in every
#            trained arm, while a random-init trunk is flat.
#   aug    - with all augmentation off, the subject probe rises monotonically with training
#            (0.342 -> 0.525 -> 0.584). Jitter and scale suppress that WITHOUT touching
#            gravity-frame orientation, so they do not repeat the independent-SO(3) failure.
#
# CAVEAT, stated up front: objective calibration balances JEPA against VICReg by gradient norm and
# is undefined at --jepa-weight 0, so the MAE arms disable it and inherit wave-1's calibrated
# vicreg_weight with mae_weight 1.0. The MAE/VICReg balance is therefore UNTUNED. If an MAE arm
# looks promising, that balance is the first thing to sweep before believing the margin.
set -uo pipefail
cd /home/alex/code/HALO/halo
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
OUT=training/tokenizer/outputs
S=20260818

COMMON="--device cuda --token-granularity sensor --text-conditioning factored \
--retrieval-vicreg-fraction 0.5 --lr 6e-4 --warmup-steps 250 --val-every 500 \
--selection-every 500 --num-workers 12 --seed 20260718 --data-seed 20260718 \
--steps 7500 --batch 1024 --patch-seconds 1.0 --no-multiresolution --rotation-p 0.0"
CALIB="--calibrate-objectives-at 500 --objective-calibration-mode apply"
NOCAL="--calibrate-objectives-at 0 --vicreg-weight 0.6532320126237195"
FIXES="--num-layers 3 --jitter-p 0.5 --scale-p 0.5"

run () { local n=$1; shift
  echo "=== [$(date -u +%H:%M:%S)] $n"
  $PY -m training.tokenizer.pretrain $COMMON "$@" --out "$OUT/phase_a_${n}_${S}" --force \
      > "$OUT/phase_a_${n}_${S}.launch.log" 2>&1
  echo "=== [$(date -u +%H:%M:%S)] $n exit=$?"; return 0; }

run f_jepa_fixes  $CALIB $FIXES --jepa-weight 1.0
run g_mae_plain   $NOCAL --jepa-weight 0 --mae-weight 1.0
run h_mae_fixes   $NOCAL $FIXES --jepa-weight 0 --mae-weight 1.0
echo "=== [$(date -u +%H:%M:%S)] WAVE2 COMPLETE"
