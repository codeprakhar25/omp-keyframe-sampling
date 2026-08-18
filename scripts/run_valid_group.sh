#!/usr/bin/env bash
# BULLETPROOF re-run of the sharding-corrupted selection variants (2026-07-19).
# NSHARD=1 ALWAYS (no doc-sharding = the bug is structurally impossible). One model-load,
# 3 arms via the 3 pick slots (task names cosmetic). PRE-gate: every pick file has EXP
# unique qids. POST-gate (cov_gate.py): every arm output has EXP unique doc_ids AND EXP
# lines, else NO .done + loud fail. Same wiring: LongCLIP, fixed tokens, Qwen3-VL-8B, bs=1, k8.
# Caller sets BIN (600|3600), GROUP (A|B), CUDA_VISIBLE_DEVICES.
set -uo pipefail
: "${BIN:?600|3600}"; : "${GROUP:?A|B}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
export LVB_DOC_NSHARD=1 LVB_DOC_SHARD=0        # <<< NO SHARDING. full pool per GPU.
PYBIN=/workspace/lmmsenv/bin/python
P=$SLM/results/picks_lmmseval
case "$GROUP" in
  A) A1=mmr;  A2=alpha0.5; A3=alpha0.75;;
  B) A1=rf15; A2=rfloor;   A3=iteralpha;;
  *) echo "!! bad GROUP $GROUP"; exit 1;;
esac
export LVB_PICKS_LC_K8="$P/picks_${A1}_lc_${BIN}_k8.json"
export LVB_PICKS_SIG_K8="$P/picks_${A2}_lc_${BIN}_k8.json"
export LVB_PICKS_OMP_LC_K8="$P/picks_${A3}_lc_${BIN}_k8.json"
EXP=$([ "$BIN" = 600 ] && echo 412 || echo 564)
# PRE-GATE: pick files exist with full qid coverage
for v in LVB_PICKS_LC_K8 LVB_PICKS_SIG_K8 LVB_PICKS_OMP_LC_K8; do
  f=${!v}; [ -s "$f" ] || { echo "!! missing $v -> $f"; exit 1; }
  m=$($PYBIN -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f")
  [ "$m" = "$EXP" ] || { echo "!! PREGATE FAIL $v qids=$m exp=$EXP"; exit 1; }
done
TAG="${BIN}_${GROUP}"
OUT=$SLM/results/valid_rerun/$TAG; mkdir -p "$OUT"
cat > "$OUT/ARM_MAP.json" <<J
{"bin":"$BIN","group":"$GROUP","nshard":1,"k":8,"exp_unique":$EXP,"slots":{"picks_lc":"$A1","picks_sig":"$A2","picks_omp_lc":"$A3"}}
J
tasks="longvideobench_val_picks_lc_${BIN}s_k8,longvideobench_val_picks_sig_${BIN}s_k8,longvideobench_val_picks_omp_lc_${BIN}s_k8"
echo "[$TAG] GPU=$CUDA_VISIBLE_DEVICES arms=$A1/$A2/$A3 exp=$EXP nshard=1 $(date -u)"
"$PYBIN" -c "import torch;print('[$TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$tasks" --batch_size 1 \
  --output_path "$OUT" --log_samples > "$OUT/run.log" 2>&1
rc=$?
echo "[$TAG] rc=$rc — running coverage gate $(date -u)"
if "$PYBIN" "$SLM/scripts/cov_gate.py" "$OUT" "$EXP"; then
  touch "$OUT.done"; echo "[$TAG] COVERAGE GATE PASSED -> done"
else
  echo "!! [$TAG] COVERAGE GATE FAILED — NOT marking done. Investigate before trusting."
fi
