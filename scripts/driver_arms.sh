#!/usr/bin/env bash
# Per-GPU chain for the lean re-run: 600 first (fast proof) then 3600. Same ARMS both bins.
# Caller sets ARMS (comma list), TAG (dir label suffix), GPU.
set -u
ARMS=${ARMS:?}; TAG=${TAG:?}; GPU=${GPU:?}
SLM=/workspace/slm-lab
for BIN in 600 3600; do
  echo "=== CHAIN $TAG GPU=$GPU BIN=$BIN start $(date -u) ==="
  BIN=$BIN ARMS="$ARMS" LABEL="${BIN}_${TAG}" CUDA_VISIBLE_DEVICES=$GPU bash "$SLM/scripts/run_valid_arms.sh"
done
echo "=== CHAIN $TAG GPU=$GPU COMPLETE $(date -u) ==="
