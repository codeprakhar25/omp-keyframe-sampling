#!/usr/bin/env bash
# Fork B decisive run: grounding-dino-base + owlv2-large, both strong families, one pass.
# Sequential (compute-bound, not VRAM-bound -> parallel gives no speedup, messier logs).
set -e
cd /workspace/slm-lab
export HF_HOME=/workspace/hf PYTHONPATH=.
set -a; . ./.env; set +a

COMMON="--scorer ground --manifest data/manifest.lvb.frames.100.json --bins 15 60 600 3600 --M 4 --agg max --probe all --ms 1 2 3"

echo "=== [1/2] grounding-dino-base $(date) ==="
GROUND_BATCH=48 python3 -u scripts/region_recall.py $COMMON \
  --detector IDEA-Research/grounding-dino-base \
  --out results/rr_ground_base_n100.json

echo "=== [2/2] owlv2-large $(date) ==="
GROUND_BATCH=24 python3 -u scripts/region_recall.py $COMMON \
  --detector google/owlv2-large-patch14-ensemble \
  --out results/rr_ground_owlv2_n100.json

echo "=== BOTH_DONE $(date) ==="
