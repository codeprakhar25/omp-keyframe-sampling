#!/bin/bash
# P0 — LVBench GPT-5-mini Claims A/B: OMP-8 full · D@53 OMP-8 · D@50 OMP-16.
# Same LDDR F.3 harness; resprop picks carry [[sec,frac]].
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
VID_ROOT=${VID_ROOT:-/workspace/hf/lvbench}
MANI=${MANI:-$SLM/data/manifest.lvbench.json}
MODEL=${MODEL:-gpt-5-mini}
EFFORT=${EFFORT:-low}
MAX_TOKENS=${MAX_TOKENS:-1024}
WORKERS=${WORKERS:-3}
PYTHON=${PYTHON:-python3.11}
OUTDIR=$SLM/results/gpt_mini
mkdir -p "$OUTDIR"

declare -A PICKS=(
  [omp8]=$SLM/data/picks_omp_lc_lvbench_k8.json
  [d53]=$SLM/data/picks_resprop_omp_lc_lvbench_k8_b53.json
  [d50k16]=$SLM/data/picks_resprop_omp_lc_lvbench_k16_b50.json
)
declare -A REQUIRE_COMP=(
  [omp8]=0
  [d53]=1
  [d50k16]=1
)
declare -A OUTNAME=(
  [omp8]=lvbench_omp_k8_gpt5mini.json
  [d53]=lvbench_resprop_k8_b53_gpt5mini.json
  [d50k16]=lvbench_resprop_k16_b50_gpt5mini.json
)

ARMS=${ARMS:-omp8 d53 d50k16}
nv=$($PYTHON -c "from pathlib import Path; print(len(list(Path('$VID_ROOT').rglob('*.mp4'))))")
echo "videos=$nv (expect ~100+) root=$VID_ROOT arms=$ARMS"
test "$nv" -ge 80

# Preflight all arms (prompt + picks) before any launch
for a in $ARMS; do
  echo "=== PREFLIGHT $a ==="
  extra=()
  if [[ "${REQUIRE_COMP[$a]}" == "1" ]]; then
    extra+=(--require-compression)
  fi
  $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "${PICKS[$a]}" \
    --video-root "$VID_ROOT" \
    --bench lvbench \
    --n 99999 \
    --preflight-only \
    --out "$OUTDIR/_preflight_${a}.json" \
    "${extra[@]}"
done

pids=()
for a in $ARMS; do
  out=$OUTDIR/${OUTNAME[$a]}
  ckpt=${out}.ckpt.jsonl
  log=$OUTDIR/${OUTNAME[$a]%.json}.log
  if [[ -f "${out}.done" ]]; then
    echo "skip $a (.done)"; continue
  fi
  extra=()
  if [[ "${REQUIRE_COMP[$a]}" == "1" ]]; then
    extra+=(--require-compression)
  fi
  echo "LAUNCH $a -> $log"
  nohup $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "${PICKS[$a]}" \
    --video-root "$VID_ROOT" \
    --bench lvbench \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    --n 99999 \
    --env-file "$SLM/.env" \
    --out "$out" \
    --ckpt "$ckpt" \
    "${extra[@]}" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "  pid=${pids[-1]}"
  sleep 2
done
printf "%s\n" "${pids[@]}" > "$OUTDIR/lvbench_ab.pids"
echo "PIDs: ${pids[*]}"
echo "tail -f $OUTDIR/lvbench_*gpt5mini.log"
