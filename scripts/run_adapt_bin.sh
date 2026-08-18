#!/usr/bin/env bash
# ADAPTIVE-k budget run: k decided per-question by residual/gain vs tau.
# SAME wiring as the OMP grid (LongCLIP, fixed tokens/frame, Qwen3-VL-8B, bs=1) -- only the pick
# file (hence frame COUNT) changes. Variable frame count is native to the picks path (task reads
# len(list)); k4/5/6 task-name slot labels are cosmetic. See results/adapt_ablation/ARM_MAP.json.
#   slot LC   ($LVB_PICKS_LC_K8)     = OMP k=4
#   slot SIG  ($LVB_PICKS_SIG_K8)    = OMP k=5
#   slot OMPLC($LVB_PICKS_OMP_LC_K8) = OMP k=6
# Compare vs banked k8 OMP in lmmseval_matrix_clean. Env LOAD-BEARING on Blackwell sm_120.
# Caller sets BIN, SHARD, CUDA_VISIBLE_DEVICES, NSHARD.
set -uo pipefail
: "${BIN:?600|3600}"; : "${SHARD:?0|1}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD="${NSHARD:-1}" LVB_DOC_SHARD="$SHARD"
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K8="$P/picks_adaptk005_lc_${BIN}_k8.json"
export LVB_PICKS_SIG_K8="$P/picks_adaptk01_lc_${BIN}_k8.json"
export LVB_PICKS_OMP_LC_K8="$P/picks_adaptk02_lc_${BIN}_k8.json"
for v in LVB_PICKS_LC_K8 LVB_PICKS_SIG_K8 LVB_PICKS_OMP_LC_K8; do
  f=${!v}; [ -s "$f" ] || { echo "!! missing $v -> $f"; exit 1; }
done
TAG="${BIN}_sh${SHARD}"
OUT=$SLM/results/adapt_ablation; mkdir -p "$OUT"
tasks="longvideobench_val_picks_lc_${BIN}s_k8,longvideobench_val_picks_sig_${BIN}s_k8,longvideobench_val_picks_omp_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES nshard=${NSHARD:-1} shard=$SHARD $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
echo "[$TAG] rc=$rc sample_files=$got (expect 3) $(date -u)"
[ "$got" -ge 3 ] && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
