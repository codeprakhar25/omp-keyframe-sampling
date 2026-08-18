#!/usr/bin/env bash
# ONE alpha arm on ONE GPU. Idea-1 partial-deflation ablation.
# Reuses the OMP-lc-k8 task/env var, pointed at an alpha picks file.
# Caller sets: CUDA_VISIBLE_DEVICES, PICKS (alpha picks json), BIN (600|3600), TAG.
# Env below is LOAD-BEARING on Blackwell sm_120 (verified 2026-07-15, see run_matrix_k8.sh).
set -uo pipefail
: "${CUDA_VISIBLE_DEVICES:?set GPU}"; : "${PICKS:?set picks}"; : "${BIN:?set bin}"; : "${TAG:?set tag}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
PYBIN=/workspace/lmmsenv/bin/python
SLMPY=$SLM/slmenv/bin/python
OUT=$SLM/results/alpha_ablation
mkdir -p "$OUT"
task="longvideobench_val_picks_omp_lc_${BIN}s_k8"
export LVB_PICKS_OMP_LC_K8="$PICKS"

[ -s "$PICKS" ] || { echo "!! [$TAG] missing/empty picks $PICKS"; exit 1; }
# preflight: picks must cover the whole bin BEFORE the 16GB load
PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages" \
  "$SLMPY" scripts/preflight_picks.py "$BIN" "$PICKS" >/dev/null 2>&1 \
  || { echo "!! [$TAG] picks do not cover ${BIN}s — ABORT"; exit 1; }

echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES task=$task picks=$(basename "$PICKS") $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$task" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_longvideobench_*${BIN}s_k8.jsonl" 2>/dev/null | wc -l)
err=$(grep -c "Error during evaluation" "$OUT/$TAG.log" 2>/dev/null || echo 0)
echo "[$TAG] done rc=$rc sample_files=$got eval_errors=$err $(date -u)"
[ "$got" -ge 1 ] && [ "$err" -eq 0 ] && touch "$OUT/$TAG.done" || echo "!! [$TAG] INCOMPLETE"
