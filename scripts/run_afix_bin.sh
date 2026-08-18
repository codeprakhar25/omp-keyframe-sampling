#!/usr/bin/env bash
# Anti-drift ablation (2026-07-18 visual analysis: OMP #1 failure = drift to irrelevant frames).
# ONE bin, ONE doc-shard, ONE model load. 3 arms via the 3 honest picks slots (task names cosmetic):
#   slot LC   ($LVB_PICKS_LC_K8)     = rfloor frac=0.15 (tight relevance floor)
#   slot SIG  ($LVB_PICKS_SIG_K8)    = rfloor frac=0.33 (mild relevance floor)
#   slot OMPLC($LVB_PICKS_OMP_LC_K8) = iteralpha (gain-proportional deflation)
# See results/afix_ablation/ARM_MAP.json. Baselines (uniform/topk/OMP a1) already banked in
# lmmseval_matrix_clean — do NOT rerun. Env LOAD-BEARING on Blackwell sm_120 (see run_matrix_k8.sh).
# Caller sets BIN, SHARD, CUDA_VISIBLE_DEVICES.
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
export LVB_PICKS_LC_K8="$P/picks_rf15_lc_${BIN}_k8.json"
export LVB_PICKS_SIG_K8="$P/picks_rfloor_lc_${BIN}_k8.json"
export LVB_PICKS_OMP_LC_K8="$P/picks_iteralpha_lc_${BIN}_k8.json"
for v in LVB_PICKS_LC_K8 LVB_PICKS_SIG_K8 LVB_PICKS_OMP_LC_K8; do
  f=${!v}; [ -s "$f" ] || { echo "!! missing $v -> $f"; exit 1; }
done
TAG="${BIN}_sh${SHARD}"
OUT=$SLM/results/afix_ablation; mkdir -p "$OUT"
tasks="longvideobench_val_picks_lc_${BIN}s_k8,longvideobench_val_picks_sig_${BIN}s_k8,longvideobench_val_picks_omp_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES nshard=${NSHARD:-1} shard=$SHARD $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
err=$(grep -c "Error during evaluation" "$OUT/$TAG.log" 2>/dev/null | head -1)
echo "[$TAG] rc=$rc sample_files=$got (expect 3) eval_errors=$err $(date -u)"
[ "$got" -ge 3 ] && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
