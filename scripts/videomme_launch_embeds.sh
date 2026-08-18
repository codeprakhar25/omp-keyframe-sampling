#!/usr/bin/env bash
# 2-GPU LongCLIP image embeds for Video-MME short || medium.
# ONE video encode per videoID (dedup hardlink across 3 question_ids).
# Does NOT touch SigLIP/DINO. Writes results/embeds_vmm/<question_id>.npz
set -euo pipefail
cd /workspace/slm-lab
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/slm-lab
bash scripts/videomme_stage1_verify.sh

OUT=results/embeds_vmm
mkdir -p "$OUT" results/videomme_logs
MAN=data/manifest.videomme.json
PY=/workspace/lmmsenv/bin/python
COMMON=(scripts/dump_embeds.py --manifest "$MAN" --out-dir "$OUT"
  --longclip --no-use-siglip --no-use-dino --dedup-by-video
  --no-gold-reliable-only --dump-fps 1.0 --max-frames 3600
  --video-root /workspace/hf/videomme/data --batch-size 64)

echo "[$(date -u +%H:%M:%S)] GPU0 short"
CUDA_VISIBLE_DEVICES=0 nohup $PY "${COMMON[@]}" --bins short \
  > results/videomme_logs/embeds_short.log 2>&1 &
echo "  pid $! log results/videomme_logs/embeds_short.log"

echo "[$(date -u +%H:%M:%S)] GPU1 medium"
CUDA_VISIBLE_DEVICES=1 nohup $PY "${COMMON[@]}" --bins medium \
  > results/videomme_logs/embeds_medium.log 2>&1 &
echo "  pid $! log results/videomme_logs/embeds_medium.log"

echo "Expect ~300 video encodes/bin (900 qids via hardlink). Monitor:"
echo "  tail -f results/videomme_logs/embeds_{short,medium}.log"
