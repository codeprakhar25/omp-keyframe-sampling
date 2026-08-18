#!/bin/bash
# Video-MME GPT-5-mini T1: 5 methods × 2700 Q (short+med+long), LDDR F.3.
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
VID_ROOT=${VID_ROOT:-/workspace/hf/videomme/data}
MANI=${MANI:-$SLM/data/manifest.videomme.json}
MODEL=${MODEL:-gpt-5-mini}
EFFORT=${EFFORT:-low}
MAX_TOKENS=${MAX_TOKENS:-1024}
WORKERS=${WORKERS:-3}
PYTHON=${PYTHON:-python3.11}
OUTDIR=$SLM/results/gpt_mini
mkdir -p "$OUTDIR"

declare -A PICKS=(
  [omp]=$SLM/data/picks_omp_lc_vmm_all_k8.json
  [topk]=$SLM/data/picks_topk_lc_vmm_all_k8.json
  [aks]=$SLM/data/picks_aks_lc_vmm_all_k8.json
  [focus]=$SLM/data/picks_focus_lc_vmm_all_k8.json
  [dppmm]=$SLM/data/picks_dppmm_lc_vmm_all_k8.json
)

METHODS=${METHODS:-omp topk aks focus dppmm}
nv=$($PYTHON -c "from pathlib import Path; print(len(list(Path('$VID_ROOT').glob('*.mp4'))))")
echo "videos=$nv (expect ~900) workers=$WORKERS methods=$METHODS"
test "$nv" -ge 850

$PYTHON - <<'PY'
import sys
sys.path.insert(0, "/workspace/slm-lab")
from scripts.gpt_mini_lvbench import format_question, PRE_PROMPTS, POST_PROMPTS
print("PRE", PRE_PROMPTS["videomme"][:70]+"...")
print("POST", repr(POST_PROMPTS["videomme"]))
print(format_question("Q?\nA. a\nB. b\nC. c\nD. d", "videomme")[:200])
PY

pids=()
for m in $METHODS; do
  out=$OUTDIR/vmm_${m}_k8_gpt5mini.json
  ckpt=${out}.ckpt.jsonl
  log=$OUTDIR/vmm_${m}_k8_gpt5mini.log
  if [ -f "${out}.done" ]; then
    echo "skip $m (.done)"; continue
  fi
  echo "LAUNCH $m -> $log"
  nohup $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "${PICKS[$m]}" \
    --video-root "$VID_ROOT" \
    --bench videomme \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    --n 99999 \
    --env-file "$SLM/.env" \
    --out "$out" \
    --ckpt "$ckpt" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "  pid=${pids[-1]}"
  sleep 2
done
printf "%s\n" "${pids[@]}" > "$OUTDIR/vmm_parallel.pids"
echo "PIDs: ${pids[*]}"
