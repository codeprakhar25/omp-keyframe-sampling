#!/usr/bin/env bash
# PRIMARY RUN: the LDDR-comparable #F=8 grid, clean STEM-ONLY picks.
#
# Target row (LDDR Table 1, Qwen3-VL-8B*, #F=8, subtitles OFF, LongCLIP for all
# baselines -- verified from the PDF 2026-07-15):
#     bin    Uniform  AKS   Q-frame  FOCUS  MDP3   LD    LDDR
#     15s    70.9     68.8  75.1     65.6   73.5   70.9  72.0
#     60s    66.9     72.1  72.7     75.0   70.9   73.3  77.3
#     600s   54.6     61.2  59.5     63.1   60.9   66.3  67.5
# Our uniform@8 must land near their Uniform row or nothing downstream is comparable.
#
# ALL ARMS IN ONE PROCESS = one model load. lmms-eval loads Qwen3-VL-8B once and
# sweeps --tasks; each picks arm reads its own $LVB_PICKS_* env var, which is the
# entire reason the per-(method,scorer,k) env var split exists.
#
# OMP is included at k=8 now rather than as a second trip: the picks already exist,
# and a separate run would pay the model load again for nothing.
#
# ENV IS LOAD-BEARING (Blackwell RTX PRO 4500, sm_120) -- verified 2026-07-15:
#   1. slmenv site-packages on PYTHONPATH: torch 2.11.0+cu130 has sm_120. lmmsenv's
#      own torch 2.4.1+cu124 stops at sm_90 -> weights load, then the first kernel in
#      generate() dies "no kernel image is available for execution on the device".
#   2. FORCE_QWENVL_VIDEO_READER=decord: slmenv torchvision 0.26 dropped io.read_video,
#      which qwen_vl_utils calls by default.
#   3. LD_LIBRARY_PATH -> slmenv nvidia/*/lib: torch dlopen()s native libs
#      (libnvrtc-builtins.so.13.0) that resolve via the loader path only.
set -uo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
OUT=$SLM/results/lmmseval_matrix_clean
mkdir -p "$OUT"

export HF_HOME=/workspace/hf
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
export FORCE_QWENVL_VIDEO_READER=decord
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
PYBIN=/workspace/lmmsenv/bin/python

P=$SLM/results/picks_lmmseval
export LVB_PICKS_LC_K8="$P/picks_lc_k8.json"
export LVB_PICKS_SIG_K8="$P/picks_sig_k8.json"
export LVB_PICKS_OMP_LC_K8="$P/picks_omp_lc_k8.json"

# preflight: every picks file must exist and be non-empty BEFORE a 16GB model loads
for v in LVB_PICKS_LC_K8 LVB_PICKS_SIG_K8 LVB_PICKS_OMP_LC_K8; do
  f=${!v}
  [ -s "$f" ] || { echo "REFUSING: $v -> $f missing/empty"; exit 1; }
  n=$($PYBIN -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f")
  echo "  $v = $(basename "$f")  ($n qids)"
done

TASKS=$(cat <<'EOF' | tr '\n' ',' | sed 's/,$//'
longvideobench_val_i_15s_k8
longvideobench_val_picks_lc_15s_k8
longvideobench_val_picks_omp_lc_15s_k8
longvideobench_val_i_60s_k8
longvideobench_val_picks_lc_60s_k8
longvideobench_val_picks_sig_60s_k8
longvideobench_val_picks_omp_lc_60s_k8
longvideobench_val_i_600s_k8
longvideobench_val_picks_lc_600s_k8
longvideobench_val_picks_omp_lc_600s_k8
EOF
)

echo "=== k=8 clean matrix  $(date -u) ==="
echo "tasks: $TASKS"
$PYBIN -c "import torch;print('torch',torch.__version__,'arch',torch.cuda.get_arch_list()[-2:],'cap',torch.cuda.get_device_capability(0))"

"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks "$TASKS" --batch_size 1 \
  --output_path "$OUT/k8" --log_samples 2>&1 | tee "$OUT/k8.log"

# DONE only when real accuracies parse. lmms-eval exits 0 on a failed eval, so exit
# status is not evidence -- a watcher trusting it reported a false DONE on 2026-07-15.
n=$(grep -c "Error during evaluation" "$OUT/k8.log" 2>/dev/null || echo 0)
got=$(find "$OUT" -name "*results*.json" | wc -l)
if [ "$n" -gt 0 ] || [ "$got" -eq 0 ]; then
  echo "!! FAILED: eval errors=$n results_json=$got"
  exit 1
fi
touch "$OUT/k8.done"
echo "=== k8.done written $(date -u) ==="
