#!/usr/bin/env bash
# Rebuild the VOID k16/3600 budget row (uniform + topk), NSHARD=1 full, coverage-gated.
# OMP@k16 and LDDR-MinMax@k16 come separately from run_k16cmp.sh (the paired run).
# Real k16 tasks: val_i_3600s_k16 = uniform@16 (k baked in fn); picks_lc_3600s_k16 = topk-lc,
# reads LVB_PICKS_LC_K16. Same wiring: LongCLIP, fixed tokens, Qwen3-VL-8B, bs=1.
# Caller sets CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${CUDA_VISIBLE_DEVICES:?GPU}"
BIN=3600; EXP=564
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0        # NO SHARDING
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K16="$P/picks_topk_lc_${BIN}_k16.json"   # topk arm -> LC slot, K16 env
[ -s "$LVB_PICKS_LC_K16" ] || { echo "!! missing topk picks $LVB_PICKS_LC_K16"; exit 1; }
m=$($PYBIN -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$LVB_PICKS_LC_K16")
[ "$m" = "$EXP" ] || { echo "!! PREGATE topk qids=$m exp=$EXP"; exit 1; }
OUT=$SLM/results/k16_rebuild/${BIN}; mkdir -p "$OUT"
tasks="longvideobench_val_i_${BIN}s_k16,longvideobench_val_picks_lc_${BIN}s_k16"
echo "[k16rebuild $BIN] GPU=$CUDA_VISIBLE_DEVICES arms=uniform,topk exp=$EXP nshard=1 $(date -u)"
"$PYBIN" -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT" --log_samples > "$OUT/run.log" 2>&1
rc=$?
echo "[k16rebuild] rc=$rc — coverage gate $(date -u)"
if "$PYBIN" "$SLM/scripts/cov_gate.py" "$OUT" "$EXP"; then
  touch "$OUT.done"; echo "[k16rebuild] COVERAGE GATE PASSED -> done"
else
  echo "!! [k16rebuild] COVERAGE GATE FAILED — NOT marking done."
fi
