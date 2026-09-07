#!/bin/bash
# lmms-eval driver (RUN ON POD w/ GPU + volume).
# Default = SHORT bins only (15s+60s) — thesis-first validation.
# 600s / 3600s ONLY when explicitly requested.
# See HANDOFF_LMMSEVAL.md.
set -euo pipefail

LMMS=${LMMS:-/workspace/lmms-eval}
SLM=${SLM:-/workspace/slm-lab}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
NFRAMES=${NFRAMES:-8}          # CRITICAL: adapter default is 32
OUT=${OUT:-$SLM/results/lmmseval}
mkdir -p "$OUT" "$SLM/results/picks_lmmseval"

# --- install injection + short-bin patches into task package ---------------
TASKDIR=$LMMS/lmms_eval/tasks/longvideobench
cp -f "$SLM/harness/lmmseval_patch/picks_utils.py" "$TASKDIR/"
cp -f "$SLM/harness/lmmseval_patch/longvideobench_val_picks.yaml" "$TASKDIR/"
cp -f "$SLM/harness/lmmseval_patch/longvideobench_val_picks_short.yaml" "$TASKDIR/"
cp -f "$SLM/harness/lmmseval_patch/longvideobench_val_v_short.yaml" "$TASKDIR/"
cp -f "$SLM/harness/lmmseval_patch/longvideobench_val_i_short.yaml" "$TASKDIR/"
# _i pathway reads max_num_frames from YAML dataset_kwargs (not model_args)
sed -i -E "s/max_num_frames:[[:space:]]*[0-9]+/max_num_frames: $NFRAMES/" \
  "$TASKDIR/longvideobench_val_i_short.yaml"

MARGS="pretrained=$MODEL,max_num_frames=$NFRAMES,device_map=auto"

# ONE env only: lmmsenv (+ its system-site torch/tv). NEVER prepend slmenv —
# newer slmenv torchvision shadows lmmsenv's working read_video and breaks imports.
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="/workspace/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
# scrub any leaked slmenv from caller env
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/slmenv/' | paste -sd: -)
if [ -x /workspace/lmmsenv/bin/python ]; then
  PY=/workspace/lmmsenv/bin/python
else
  PY=python3
fi

run () {
  local task=$1 tag=$2
  echo "=== $tag : $task (nframes=$NFRAMES) $(date -u) ==="
  # --force_simple: HANDOFF verified max_num_frames→nframes on models/simple/qwen3_vl.py
  "$PY" -m lmms_eval --model qwen3_vl --force_simple --model_args "$MARGS" \
    --tasks "$task" --batch_size 1 \
    --output_path "$OUT/$tag" \
    --log_samples 2>&1 | tee "$OUT/$tag.log"
}

export_short_topk () {
  # SigLIP top-k picks for 15+60 (from full1560 scores)
  local k=${1:-8}
  local scores=${2:-$SLM/results/scores/scores_1560_full.jsonl}
  local out=$SLM/results/picks_lmmseval/sig_topk${k}_short1560.json
  "$PY" "$SLM/scripts/export_picks_lmmseval.py" \
    --from-scores "$scores" --method topk --k "$k" \
    --out "$out"
  # exporter filters by --bin if set; without --bin keeps all rows in file (=15+60)
  echo "$out"
}

case "${1:-short}" in
  short|calib-short)
    # GOAL: validate harness on 15s+60s ONLY (n≈361).
    # Target LDDR Uniform@8: 15s=70.9, 60s=66.9.
    # Then topk@NFRAMES inject vs uniform_i@NFRAMES (same frames pathway).
    run longvideobench_val_v_short "short_uniform${NFRAMES}_v"
    run longvideobench_val_i_short "short_uniform${NFRAMES}_i"
    PICKS=$(export_short_topk "$NFRAMES")
    export LVB_PICKS="$PICKS"
    run longvideobench_val_picks_short "short_topk${NFRAMES}_sig"
    ;;
  short-topk6)
    # tight-budget topk@6 on short bins (our hunt k); needs uniform_i@6 control too
    NFRAMES=6
    MARGS="pretrained=$MODEL,max_num_frames=$NFRAMES,device_map=auto"
    run longvideobench_val_i_short "short_uniform${NFRAMES}_i"
    PICKS=$(export_short_topk 6)
    export LVB_PICKS="$PICKS"
    run longvideobench_val_picks_short "short_topk${NFRAMES}_sig"
    ;;
  calib)
    # FULL val (all bins) — DO NOT run unless explicitly asked
    echo "REFUSING full calib by default. Pass: $0 calib-full  (includes 600+3600)" >&2
    exit 2
    ;;
  calib-full)
    echo "WARN: full LVB val incl 600s+3600s — long run" >&2
    run longvideobench_val_v "uniform${NFRAMES}_v"
    run longvideobench_val_i "uniform${NFRAMES}_i"
    ;;
  inject)
    : "${LVB_PICKS:?export LVB_PICKS=/path/to/picks.json first}"
    export LVB_PICKS
    run longvideobench_val_picks "inject_$(basename "${LVB_PICKS%.json}")"
    ;;
  inject-short)
    : "${LVB_PICKS:?export LVB_PICKS=/path/to/picks.json first}"
    export LVB_PICKS
    run longvideobench_val_picks_short "inject_short_$(basename "${LVB_PICKS%.json}")"
    ;;
  *) echo "usage: $0 {short|short-topk6|calib-full|inject|inject-short}"; exit 1;;
esac
echo "DONE $(date -u)"
