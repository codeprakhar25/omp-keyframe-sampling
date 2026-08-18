#!/usr/bin/env bash
# Idea-1 (alpha) + Idea-B (MMR) ablation, ONE bin, ONE doc-shard, ONE model load.
# 3 arms bundled via the 3 honest picks slots (task names are cosmetic labels here):
#   slot LC   ($LVB_PICKS_LC_K8)     = MMR hybrid (3 OMP anchors + diversity fill, lam=0.5)
#   slot SIG  ($LVB_PICKS_SIG_K8)    = alpha=0.5 partial-deflation OMP
#   slot OMPLC($LVB_PICKS_OMP_LC_K8) = alpha=0.75 partial-deflation OMP
# See results/ab_ablation/ARM_MAP.json. Baselines (uniform@8, topk@8, OMP@8=alpha1) already
# in results/lmmseval_matrix_clean — do NOT rerun them.
# Env is LOAD-BEARING on Blackwell sm_120 (see run_matrix_k8.sh). Caller sets BIN, SHARD, CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${BIN:?set BIN 600|3600}"; : "${SHARD:?set SHARD 0|1}"; : "${CUDA_VISIBLE_DEVICES:?set GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=2 LVB_DOC_SHARD="$SHARD"
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K8="$P/picks_mmr_lc_${BIN}_k8.json"
export LVB_PICKS_SIG_K8="$P/picks_alpha0.5_lc_${BIN}_k8.json"
export LVB_PICKS_OMP_LC_K8="$P/picks_alpha0.75_lc_${BIN}_k8.json"
for v in LVB_PICKS_LC_K8 LVB_PICKS_SIG_K8 LVB_PICKS_OMP_LC_K8; do
  f=${!v}; [ -s "$f" ] || { echo "!! missing $v -> $f"; exit 1; }
done
TAG="${BIN}_sh${SHARD}"
OUT=$SLM/results/ab_ablation; mkdir -p "$OUT"
tasks="longvideobench_val_picks_lc_${BIN}s_k8,longvideobench_val_picks_sig_${BIN}s_k8,longvideobench_val_picks_omp_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES shard=$SHARD/2 $(date -u)"
echo "[$TAG] tasks=$tasks"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
err=$(grep -c "Error during evaluation" "$OUT/$TAG.log" 2>/dev/null || echo 0)
echo "[$TAG] rc=$rc sample_files=$got (expect 3) eval_errors=$err $(date -u)"
[ "$got" -ge 3 ] && [ "$err" -eq 0 ] && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
