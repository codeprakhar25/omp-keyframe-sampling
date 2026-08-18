#!/usr/bin/env bash
set -uo pipefail
: "${HALF:?hA|hB|hC}"; : "${CUDA_VISIBLE_DEVICES:?GPU}"
SLM=/workspace/slm-lab; cd "$SLM"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 FORCE_QWENVL_VIDEO_READER=decord PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
PYBIN=/workspace/lmmsenv/bin/python
export VMM_PICKS_OMP_LC_K8="$SLM/results/picks_lmmseval/picks_focus_lc_vmm_long_k8_${HALF}.json"
[ -s "$VMM_PICKS_OMP_LC_K8" ] || { echo "!! missing $VMM_PICKS_OMP_LC_K8"; exit 1; }
TAG="long_${HALF}"
OUT=$SLM/results/focus_vmm; mkdir -p "$OUT"
task="videomme_picks_omp_lc_long_k8"
n=$("$PYBIN" -c "import json;print(len(json.load(open('$VMM_PICKS_OMP_LC_K8'))))")
echo "[focus-vmm $TAG] GPU=$CUDA_VISIBLE_DEVICES n=$n $(date -u)"
"$PYBIN" -c "import torch;print('[focus-vmm $TAG] torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))"
"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$task" --batch_size 1 \
  --output_path "$OUT/$TAG" --log_samples > "$OUT/$TAG.log" 2>&1
rc=$?
got=$(find "$OUT/$TAG" -name "*samples_videomme*.jsonl" 2>/dev/null | wc -l)
echo "[focus-vmm $TAG] rc=$rc sample_files=$got $(date -u)"
if [ "$got" -lt 1 ]; then echo "!! INCOMPLETE"; exit 1; fi
# coverage by line count (doc_id is local index)
"$PYBIN" - <<PY
import json
from pathlib import Path
picks=json.load(open("$VMM_PICKS_OMP_LC_K8"))
fs=list(Path("$OUT/$TAG").rglob("*samples_videomme*.jsonl"))
n=sum(1 for f in fs for line in open(f) if line.strip())
print(f"[focus-vmm $TAG] coverage lines={n} picks={len(picks)}")
if n!=len(picks): raise SystemExit(2)
Path("$OUT/$TAG.done").write_text(f"n={n}\n")
PY
