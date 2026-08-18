#!/bin/bash
# Full LVBench GPT frontier T1: 5 methods × 1549 Q (LDDR F.3 protocol).
# Prereq: /workspace/hf/lvbench/ has all videos; picks+manifest in data/.
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
VID_ROOT=${VID_ROOT:-/workspace/hf/lvbench}
MANI=${MANI:-$SLM/data/manifest.lvbench.json}
MODEL=${MODEL:-gpt-5-mini}
EFFORT=${EFFORT:-low}
MAX_TOKENS=${MAX_TOKENS:-1024}
WORKERS=${WORKERS:-6}
PYTHON=${PYTHON:-python3.11}
BENCH=lvbench
OUTDIR=$SLM/results/gpt_mini
mkdir -p "$OUTDIR"

# method -> picks file
declare -A PICKS=(
  [omp]=$SLM/data/picks_omp_lc_lvbench_k8.json
  [topk]=$SLM/data/picks_topk_lc_lvbench_k8.json
  [aks]=$SLM/data/picks_aks_lc_lvbench_k8.json
  [focus]=$SLM/data/picks_focus_lc_lvbench_k8.json
  [dppmm]=$SLM/data/picks_dppmm_lc_lvbench_k8.json
)

METHODS=${METHODS:-omp topk aks focus dppmm}

# gate: video count
nv=$($PYTHON -c "from pathlib import Path; print(len(list(Path('$VID_ROOT').glob('*.mp4'))))")
echo "videos_present=$nv (expect ~103)"
if [ "$nv" -lt 100 ]; then
  echo "REFUSE: need full lvbench pull first" >&2
  exit 1
fi

# protocol smoke print
$PYTHON - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/slm-lab")
from scripts.gpt_mini_lvbench import format_question, POST_PROMPTS, PRE_PROMPTS
q = "What year?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"
print("PRE", repr(PRE_PROMPTS["lvbench"]))
print("POST", repr(POST_PROMPTS["lvbench"]))
print("SAMPLE:\n"+format_question(q, "lvbench"))
PY

for m in $METHODS; do
  picks=${PICKS[$m]}
  out=$OUTDIR/lvbench_${m}_k8_gpt5mini.json
  ckpt=${out}.ckpt.jsonl
  log=$OUTDIR/lvbench_${m}_k8_gpt5mini.log
  echo "===== METHOD=$m picks=$picks out=$out ====="
  if [ -f "${out}.done" ]; then
    echo "skip $m (.done present)"
    continue
  fi
  set +e
  $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "$picks" \
    --video-root "$VID_ROOT" \
    --bench "$BENCH" \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    --n 99999 \
    --env-file "$SLM/.env" \
    --out "$out" \
    --ckpt "$ckpt" \
    2>&1 | tee "$log"
  set -e
  # mark done if coverage complete
  if $PYTHON - "$out" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(f"DONE {sys.argv[1]} acc={d.get('accuracy')} n={d.get('n')}/{d.get('n_requested')} cov={d.get('coverage')}")
ok = d.get("n_requested") and d.get("n") == d.get("n_requested") and d.get("coverage", 0) >= 0.99
sys.exit(0 if ok else 2)
PY
  then
    date -u +"done %Y-%m-%dT%H:%M:%SZ" > "${out}.done"
  else
    echo "WARN: $m incomplete — not marking .done" >&2
  fi
done

echo "ALL METHODS FINISHED"
for m in $METHODS; do
  f=$OUTDIR/lvbench_${m}_k8_gpt5mini.json
  [ -f "$f" ] && $PYTHON -c "import json;d=json.load(open('$f'));print(f\"$m acc={d.get('accuracy')} n={d.get('n')}/{d.get('n_requested')}\")"
done
