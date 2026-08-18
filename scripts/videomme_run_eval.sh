#!/usr/bin/env bash
# args: GPU TASK OUTTAG ENVNAME PICKS_PATH
# Example:
#  bash scripts/videomme_run_eval.sh 0 videomme_picks_omp_lc_short_k8 omp_short_k8 VMM_PICKS_OMP_LC_K8 results/picks_lmmseval/picks_omp_lc_vmm_short_k8.json
set -uo pipefail
GPU=$1; TASK=$2; OUT=$3; ENVN=$4; PICKS=$5; LIMIT=${6:-0}
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf FORCE_QWENVL_VIDEO_READER=decord HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
export "$ENVN=$PICKS"
[ -s "$PICKS" ] || { echo "!! missing picks $PICKS"; exit 1; }
M=$SLM/results/videomme_eval; mkdir -p "$M"
LIM=""; [ "$LIMIT" -gt 0 ] && LIM="--limit $LIMIT"
echo "[$(date -u +%H:%M:%S)] GPU$GPU RUN $OUT task=$TASK env=$ENVN"
/workspace/lmmsenv/bin/python -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$TASK" --batch_size 1 $LIM \
  --output_path "$M/$OUT" --log_samples > "$M/$OUT.log" 2>&1
rc=$?
echo "[$(date -u +%H:%M:%S)] $OUT rc=$rc"
exit $rc
