#!/usr/bin/env bash
# Phase-A recovery fleet (2026-08-18). One GPU, arms run SERIALLY.
#
# Reference points measured 2026-08-18 by `development_transfer_score` through the isolated
# retrieval rows, scored on motionsense/realworld/shoaib (ExtraSensory reported but unscored):
#
#   random init (floor)  0.8012      an arm below this has learned nothing usable for retrieval
#   old-good 4k          0.8577      the best checkpoint ever trained -- THE BAR
#   rejected 27k         0.7161      the arm this fleet replaces
#
# The old-good recipe carried multiresolution AND descriptor prediction, both of which the clean
# reference drops. Arms C/D exist to test whether those two are what the +0.057 was made of.
#
# Batch/steps differ between the 1s arms (1024 x 7500) and the multiresolution arms (512 x 15000)
# so that TOTAL WINDOW EXPOSURE matches at 7.68M. Compare within a pair first.
set -uo pipefail
cd /home/alex/code/HALO/halo
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
OUT=training/tokenizer/outputs
STAMP=20260818

COMMON="--device cuda --token-granularity sensor --text-conditioning factored \
--retrieval-vicreg-fraction 0.5 --calibrate-objectives-at 500 --objective-calibration-mode apply \
--lr 6e-4 --warmup-steps 250 --val-every 500 --selection-every 500 --num-workers 12 \
--seed 20260718 --data-seed 20260718"

FIXED1S="--steps 7500 --batch 1024 --patch-seconds 1.0 --no-multiresolution"
MULTIRES="--steps 15000 --batch 512 --multiresolution"

run () {                      # run <name> <extra args...>
  local name=$1; shift
  local dir="$OUT/phase_a_${name}_${STAMP}"
  echo "=== [$(date -u +%H:%M:%S)] ARM ${name} -> ${dir}"
  $PY -m training.tokenizer.pretrain $COMMON "$@" --out "$dir" --force \
      > "${dir}.launch.log" 2>&1
  local rc=$?
  echo "=== [$(date -u +%H:%M:%S)] ARM ${name} exit=${rc}"
  return 0                    # never let one failed arm abort the fleet
}

run a_clean              $FIXED1S --rotation-p 0.0
run b_sharedrot          $FIXED1S --rotation-p 1.0 --rotation-pairing shared
run c_multires_desc      $MULTIRES --rotation-p 0.0 --descriptor-weight 0.5
run d_sharedrot_mr_desc  $MULTIRES --rotation-p 1.0 --rotation-pairing shared --descriptor-weight 0.5
run e_clean_long         --steps 15000 --batch 1024 --patch-seconds 1.0 --no-multiresolution --rotation-p 0.0

echo "=== [$(date -u +%H:%M:%S)] FLEET COMPLETE"
