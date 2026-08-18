#!/usr/bin/env bash
# Lever-1 Arm A (2026-07-19). DPP with LDDR MinMax quality-norm vs banked OMP/z-DPP.
# ONE arm/bin (MinMax picks routed through the LC pick-slot; task name cosmetic). Same wiring
# as the OMP grid: LongCLIP scorer, 1fps pool, FIXED tokens/frame, Qwen3-VL-8B, bs=1, k8.
# Only diff from banked DPP = quality r_i = MinMax(cos) instead of exp(beta*z(cos)).
# Compare vs banked OMP k8 in lmmseval_matrix_clean/k8_${BIN} and banked DPP in dpp_ablation.
# Env LOAD-BEARING on Blackwell sm_120. Caller sets BIN, CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${BIN:?600|3600}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K8="$P/picks_dppmm_lc_${BIN}_k8.json"
[ -s "$LVB_PICKS_LC_K8" ] || { echo "!! missing $LVB_PICKS_LC_K8"; exit 1; }
TAG="${BIN}_sh0"
OUT=$SLM/results/dppmm_ablation; mkdir -p "$OUT"
task="longvideobench_val_picks_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$task" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
echo "[$TAG] rc=$rc sample_files=$got (expect 1) $(date -u)"
[ "$got" -ge 1 ] && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
