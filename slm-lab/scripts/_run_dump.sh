#!/usr/bin/env bash
# Phase 2 (retrieve-then-ground): SigLIP score-cache dump on a GPU pod.
# One-time GPU cost -> makes ALL union experiments (scripts/union_ceiling.py) free thereafter.
# scp results/scores/scores.jsonl back to local, then run union_ceiling.py with NO GPU.
#
#   BINS="600 3600" bash scripts/_run_dump.sh      # decision bins (default)
#   BINS="60 600 3600" bash scripts/_run_dump.sh   # + 60s sanity
set -e
cd /workspace/slm-lab
export HF_HOME=/workspace/hf PYTHONPATH=.
export PIP_BREAK_SYSTEM_PACKAGES=1          # PEP-668 pods: bare pip installs silently no-op without this
BINS="${BINS:-600 3600}"

echo "=== dump_scores bins=[$BINS] $(date) ==="
python3 -u scripts/dump_scores.py \
  --manifest data/manifest.lvb.frames.100.json \
  --bins $BINS \
  --out results/scores/scores.jsonl

echo "=== DONE $(date) -> results/scores/scores.jsonl ==="
wc -l results/scores/scores.jsonl
