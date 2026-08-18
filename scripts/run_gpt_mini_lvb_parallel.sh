#!/bin/bash
# LongVideoBench GPT-5-mini T1: all bins (15/60/600/3600), LDDR F.3.
# topk k8 picks missing on Azure — default methods exclude it.
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
VID_ROOT=${VID_ROOT:-/workspace/slm-lab/data/videos}
MANI=${MANI:-$SLM/data/manifest.lvb.allbins.json}
MODEL=${MODEL:-gpt-5-mini}
EFFORT=${EFFORT:-low}
MAX_TOKENS=${MAX_TOKENS:-1024}
WORKERS=${WORKERS:-3}
PYTHON=${PYTHON:-python3.11}
OUTDIR=$SLM/results/gpt_mini
mkdir -p "$OUTDIR"

declare -A PICKS=(
  [omp]=$SLM/data/picks_omp_lc_lvb_all_k8.json
  [aks]=$SLM/data/picks_aks_lc_lvb_all_k8.json
  [focus]=$SLM/data/picks_focus_lc_lvb_all_k8.json
  [dppmm]=$SLM/data/picks_dppmm_lc_lvb_all_k8.json
)

METHODS=${METHODS:-omp aks focus dppmm}
nv=$($PYTHON -c "from pathlib import Path; print(len(list(Path('$VID_ROOT').glob('*.mp4'))))")
echo "videos=$nv (expect ~753) workers=$WORKERS methods=$METHODS"
test "$nv" -ge 700

# Hard preflight: every queued prompt must contain A/B option lines.
$PYTHON - <<PY
import json, sys
sys.path.insert(0, "$SLM")
from scripts.gpt_mini_lvbench import (
    format_question, item_option_lines, prompt_has_options, valid_letters_for,
    PRE_PROMPTS, POST_PROMPTS,
)
mani = json.load(open("$MANI"))
assert len(mani) == 1337, len(mani)
bad = []
e_count = 0
for it in mani:
    opts = item_option_lines(it, bench="longvideobench")
    q = it.get("question_stem") or it["question"]
    text = format_question(q, "longvideobench", options=opts or None)
    if not prompt_has_options(text):
        bad.append(it["id"])
    if "E" in valid_letters_for(it, "longvideobench"):
        e_count += 1
if bad:
    print("PREFLIGHT_FAIL missing options:", bad[:10], "n=", len(bad))
    sys.exit(2)
print("PREFLIGHT_OK n=1337 with_options=1337 with_E=", e_count)
print("PRE", PRE_PROMPTS["longvideobench"][:70] + "...")
print("POST", repr(POST_PROMPTS["longvideobench"]))
# show one formatted prompt tail
it = mani[0]
opts = item_option_lines(it, bench="longvideobench")
text = format_question(it.get("question_stem") or it["question"], "longvideobench", options=opts or None)
print("--- SAMPLE PROMPT TAIL ---")
print(text[-400:])
print("--- END ---")
PY

pids=()
for m in $METHODS; do
  picks="${PICKS[$m]:-}"
  if [[ -z "$picks" || ! -f "$picks" ]]; then
    echo "SKIP $m — picks missing: $picks"
    continue
  fi
  out=$OUTDIR/lvb_${m}_k8_gpt5mini.json
  ckpt=${out}.ckpt.jsonl
  log=$OUTDIR/lvb_${m}_k8_gpt5mini.log
  if [[ -f "${out}.done" ]]; then
    echo "skip $m (.done)"; continue
  fi
  echo "LAUNCH $m -> $log"
  nohup $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "$picks" \
    --video-root "$VID_ROOT" \
    --bench longvideobench \
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
printf "%s\n" "${pids[@]}" > "$OUTDIR/lvb_parallel.pids"
echo "PIDs: ${pids[*]}"
echo "tail -f $OUTDIR/lvb_*_k8_gpt5mini.log"
