#!/usr/bin/env bash
# Per-GPU chain for the bulletproof re-run. Runs 600 FIRST (fast, ~20min -> proves the
# coverage gate green cheaply) THEN 3600. Caller sets G (A|B), GPU.
set -u
G=${G:?A|B}; GPU=${GPU:?}
SLM=/workspace/slm-lab
for BIN in 600 3600; do
  echo "=== CHAIN G=$G GPU=$GPU BIN=$BIN start $(date -u) ==="
  BIN=$BIN GROUP=$G CUDA_VISIBLE_DEVICES=$GPU bash "$SLM/scripts/run_valid_group.sh"
done
echo "=== CHAIN G=$G GPU=$GPU COMPLETE $(date -u) ==="
