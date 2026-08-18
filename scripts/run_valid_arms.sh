#!/usr/bin/env bash
# General bulletproof re-run: arbitrary arm list (1-3) in one model-load. NSHARD=1, pre-gate
# (pick files full qid coverage) + post-gate (cov_gate.py: EXP unique doc_ids per arm) or NO .done.
# Arms map in order to LC / SIG / OMP_LC pick slots + picks_lc/sig/omp_lc tasks (cosmetic labels).
# Caller sets BIN (600|3600), ARMS (comma list, e.g. alpha0.5,alpha0.75), LABEL, CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${BIN:?600|3600}"; : "${ARMS:?comma list}"; : "${LABEL:?}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
EXP=$([ "$BIN" = 600 ] && echo 412 || echo 564)
IFS=',' read -ra AR <<< "$ARMS"
slots=(LC SIG OMP_LC); tnames=(picks_lc picks_sig picks_omp_lc)
[ "${#AR[@]}" -le 3 ] || { echo "!! max 3 arms/load"; exit 1; }
tasks=""
for i in "${!AR[@]}"; do
  a=${AR[$i]}; s=${slots[$i]}; tn=${tnames[$i]}
  f="$P/picks_${a}_lc_${BIN}_k8.json"
  [ -s "$f" ] || { echo "!! missing pick file $f"; exit 1; }
  m=$($PYBIN -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f")
  [ "$m" = "$EXP" ] || { echo "!! PREGATE FAIL $a qids=$m exp=$EXP"; exit 1; }
  export LVB_PICKS_${s}_K8="$f"
  tasks="${tasks:+$tasks,}longvideobench_val_${tn}_${BIN}s_k8"
done
OUT=$SLM/results/valid_rerun/${LABEL}; mkdir -p "$OUT"
printf '{"bin":"%s","arms":"%s","nshard":1,"slot_order":"LC,SIG,OMP_LC"}\n' "$BIN" "$ARMS" > "$OUT/ARM_MAP.json"
echo "[$LABEL] GPU=$CUDA_VISIBLE_DEVICES arms=$ARMS exp=$EXP nshard=1 $(date -u)"
"$PYBIN" -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT" --log_samples > "$OUT/run.log" 2>&1
rc=$?
echo "[$LABEL] rc=$rc — coverage gate $(date -u)"
if "$PYBIN" "$SLM/scripts/cov_gate.py" "$OUT" "$EXP"; then
  touch "$OUT.done"; echo "[$LABEL] COVERAGE GATE PASSED -> done"
else
  echo "!! [$LABEL] COVERAGE GATE FAILED — NOT marking done."
fi
