#!/usr/bin/env bash
# Phase-A wave 3 (2026-08-18): does the REST of old-good's augmentation stack close the last gap?
#
# Wave 2 established: depth 3 is the real gain (+0.0157 main effect, above the 0.012 noise floor);
# MAE vs JEPA is +0.0027, undetectable. Best arm h_mae_fixes 0.8468, old-good 0.8577 (-0.011).
#
# BUT jitter+scale did NOT fix subject leakage -- still ~0.55 against old-good's 0.3146. So the
# transfer gain came from DEPTH, not nuisance suppression, and old-good's remaining advantage is
# still unexplained. The untested part of its stack is gravity / rate / channel-dropout / text.
# All arms keep JEPA (MAE earned nothing) and depth 3 (which did).
set -uo pipefail
cd /home/alex/code/HALO/halo
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
OUT=training/tokenizer/outputs; S=20260818
COMMON="--device cuda --token-granularity sensor --text-conditioning factored \
--retrieval-vicreg-fraction 0.5 --calibrate-objectives-at 500 --objective-calibration-mode apply \
--lr 6e-4 --warmup-steps 250 --val-every 500 --selection-every 500 --num-workers 12 \
--seed 20260718 --data-seed 20260718 --steps 7500 --batch 1024 --patch-seconds 1.0 \
--no-multiresolution --rotation-p 0.0 --jepa-weight 1.0 --num-layers 3"
run () { local n=$1; shift; echo "=== [$(date -u +%H:%M:%S)] $n"
  $PY -m training.tokenizer.pretrain $COMMON "$@" --out "$OUT/phase_a_${n}_${S}" --force \
      > "$OUT/phase_a_${n}_${S}.launch.log" 2>&1; echo "=== [$(date -u +%H:%M:%S)] $n exit=$?"; return 0; }

# i: physical acquisition stack (gravity + rate + channel dropout) on top of wave-2's jitter+scale.
run i_aug_physical --jitter-p 0.5 --scale-p 0.5 --gravity-p 0.5 --rate-augmentation-p 0.5 \
                   --channel-dropout-p 0.3
# j: i + text augmentation -- the complete old-good stack.
run j_aug_full     --jitter-p 0.5 --scale-p 0.5 --gravity-p 0.5 --rate-augmentation-p 0.5 \
                   --channel-dropout-p 0.3 --channel-text-phrase-p 0.5 --channel-text-dropout-p 0.15
echo "=== [$(date -u +%H:%M:%S)] WAVE3 COMPLETE"
