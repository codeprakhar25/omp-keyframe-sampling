#!/usr/bin/env bash
# One-off matched-k16 head-to-head, 3600s only (2026-07-19). Fair budget comparison:
#   OMP-lc @ k16   vs   LDDR-selection (MinMax linear-DPP) @ k16.
# Both fresh THIS session so the comparison is drift-free (no cross-run/temp-0 concern).
# Same wiring: LongCLIP, 1fps pool, FIXED tokens/frame, Qwen3-VL-8B, bs=1. NSHARD=1 full n=564.
# Pick file routed through the LC slot; task ..._k8 is COSMETIC (reads len(list)=16). Caller sets
# ARM (omp|dppmm), CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${ARM:?omp|dppmm}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
BIN=3600
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K8="$P/picks_${ARM}_lc_${BIN}_k16.json"
[ -s "$LVB_PICKS_LC_K8" ] || { echo "!! missing $LVB_PICKS_LC_K8"; exit 1; }
TAG="${ARM}_${BIN}"
OUT=$SLM/results/k16_cmp; mkdir -p "$OUT"
task="longvideobench_val_picks_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES pickfile=$LVB_PICKS_LC_K8 $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$task" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
n=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" -exec sh -c 'grep -c . "$1"' _ {} \; 2>/dev/null | head -1)
echo "[$TAG] rc=$rc files=$got n=$n (expect 1 file, 564 lines) $(date -u)"
{ [ "$got" -ge 1 ] && [ "${n:-0}" -eq 564 ]; } && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
