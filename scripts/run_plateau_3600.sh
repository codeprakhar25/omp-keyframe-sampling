#!/usr/bin/env bash
# Residual-plateau early-stop OMP arms on 3600s (NSHARD=1, coverage-gated).
# Variable-k picks injected via the k-agnostic task longvideobench_val_picks_omp_lc_3600s
# (filter_3600s + doc_to_visual reads $LVB_PICKS_OMP_LC). k baked into no name — frames
# come straight from the picks json. bs=1, LongCLIP-scored, Qwen3-VL-8B.
# Usage: CUDA_VISIBLE_DEVICES=0 run_plateau_3600.sh <picks.json> <tag>
set -uo pipefail
: "${CUDA_VISIBLE_DEVICES:?GPU}"
PICKS="${1:?picks json}"; TAG="${2:?tag}"
BIN=3600; EXP=564
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0          # NO SHARDING (corruption guard)
PYBIN=/workspace/lmmsenv/bin/python

[ -s "$PICKS" ] || { echo "!! missing picks $PICKS"; exit 1; }
m=$($PYBIN -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$PICKS")
[ "$m" = "$EXP" ] || { echo "!! PREGATE $TAG qids=$m exp=$EXP"; exit 1; }
export LVB_PICKS_OMP_LC="$PICKS"

OUT=$SLM/results/plateau_3600/$TAG; mkdir -p "$OUT"
echo "[plateau $TAG] GPU=$CUDA_VISIBLE_DEVICES picks=$PICKS qids=$m nshard=1 $(date -u)"
"$PYBIN" -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks longvideobench_val_picks_omp_lc_3600s --batch_size 1 \
  --output_path "$OUT" --log_samples > "$OUT/run.log" 2>&1
rc=$?
echo "[plateau $TAG] rc=$rc — coverage gate $(date -u)"
if "$PYBIN" "$SLM/scripts/cov_gate.py" "$OUT" "$EXP"; then
  touch "$OUT.done"; echo "[plateau $TAG] COVERAGE GATE PASSED -> done"
else
  echo "!! [plateau $TAG] COVERAGE GATE FAILED — NOT marking done."
fi
