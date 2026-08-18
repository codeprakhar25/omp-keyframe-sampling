#!/usr/bin/env bash
# 15s uniform@8 native-video gate vs LDDR Table 1 (Qwen3-VL-8B, #F=8) = 70.9.
# PASS = within ~1.5pt. Validates the harness before any injection run.
#
# ENV IS LOAD-BEARING — three things, all verified 2026-07-15, cost ~2.5h to learn:
#
# 1. slmenv site-packages on PYTHONPATH. This box is RTX PRO 4500 Blackwell
#    (sm_120, cap 12.0). system/lmmsenv torch 2.4.1+cu124 arch list STOPS at
#    sm_90 -> weights load 750/750, "Model Responding: 0/189" prints, then the
#    first kernel in generate() dies "no kernel image is available for execution
#    on the device". slmenv torch 2.11.0+cu130 has sm_120. Do NOT remove slmenv
#    to get "one clean env" — that is the broken config.
# 2. FORCE_QWENVL_VIDEO_READER=decord. slmenv torchvision 0.26 dropped
#    io.read_video, which qwen_vl_utils calls by default. decord makes it
#    unreachable. REQUIRED, not belt-and-braces.
# 3. LD_LIBRARY_PATH -> slmenv nvidia/*/lib. PYTHONPATH gets torch's python
#    modules; torch also dlopen()s NATIVE libs (libnvrtc-builtins.so.13.0) that
#    resolve via the LOADER path only. Without it:
#      nvrtc: error: failed to open libnvrtc-builtins.so.13.0
#    at the first sample. Same trick as scripts/hunt_topk_arms.sh.
set -euo pipefail
SLM=/workspace/slm-lab
OUT=$SLM/results/lmmseval
mkdir -p "$OUT"
export HF_HOME=/workspace/hf
export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
export FORCE_QWENVL_VIDEO_READER=decord
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
PYBIN=/workspace/lmmsenv/bin/python

echo "=== gate15 uniform@8 val_v (target 0.709) $(date -u) ==="
"$PYBIN" -c "import torch;print('torch',torch.__version__,'arch',torch.cuda.get_arch_list()[-2:],'cap',torch.cuda.get_device_capability(0))" 2>/dev/null || true

"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=8,device_map=auto \
  --tasks longvideobench_val_v_15s --batch_size 1 \
  --output_path "$OUT/gate15_v2" --log_samples 2>&1 | tee "$OUT/gate15_v2.log"

# Write the accuracy ONLY when a real number parses -> a DONE that cannot lie.
acc=$("$PYBIN" - << 'PYEOF'
import glob, json
# lmms-eval names it "<timestamp>_results.json", so the glob MUST be *results*.json
# -- "results*.json" silently matches nothing and makes a PASSING run look FAILED.
fs = sorted(glob.glob("/workspace/slm-lab/results/lmmseval/gate15_v2/**/*results*.json", recursive=True))
if not fs:
    print(""); raise SystemExit
d = json.load(open(fs[-1]))
for task, m in d.get("results", {}).items():
    for k, v in m.items():
        if "lvb_acc" in k and isinstance(v, (int, float)):
            print(f"{v:.4f}"); raise SystemExit
print("")
PYEOF
)
if [ -n "$acc" ]; then
  echo "$acc" > "$OUT/GATE15_ACC.txt"
  echo "GATE15 acc=$acc  LDDR=0.709  delta=$("$PYBIN" -c "print(f'{float('"$acc"')-0.709:+.4f}')")"
else
  echo "GATE15: no accuracy parsed — run FAILED, not writing GATE15_ACC.txt" >&2
  exit 1
fi
