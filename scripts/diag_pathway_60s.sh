#!/usr/bin/env bash
# Which pathway does LDDR's Table 1 actually correspond to?
#
# At 15s the gate could not tell: val_v=69.3 (-1.6 vs their 70.9) and val_i=72.5 (+1.6).
# Both "pass" a +-2pt gate, in opposite directions. At 60s they diverge (val_i=72.67 vs
# their 66.9, +5.8), so 60s is where the pathways are separable -- run the SAME uniform@8
# control through the native-video reader and see which side lands on 66.9.
#
# ENV IS A VERBATIM COPY OF scripts/gate15.sh (see its header for why each line exists;
# it cost ~2.5h to learn). The ONLY intended difference from the gate is the task name.
# In particular PYBIN is lmmsenv, NOT slmenv: slmenv has no decord. Keep max_num_frames=8
# -- val_v reads the video itself, so unlike the val_i/picks arms the budget is not baked
# into doc_to_visual.
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

# Refuse to contend with a live lmms-eval run: two model loads do not fit in 31.37 GiB and
# the loser is whichever job is 40 minutes in. Bracket trick -- a bare `pgrep -f lmms_eval`
# matches this script's own ssh cmdline and always self-reports RUNNING.
if pgrep -f "[l]mms_eval" > /dev/null; then
  echo "REFUSING: an lmms_eval run is live. Wait for it, then re-run."; exit 1
fi

echo "=== diag: uniform@8 val_v 60s (LDDR 60s uniform@8 = 0.669) $(date -u) ==="
"$PYBIN" -c "import torch;print('torch',torch.__version__,'arch',torch.cuda.get_arch_list()[-2:],'cap',torch.cuda.get_device_capability(0))" 2>/dev/null || true

"$PYBIN" -m lmms_eval --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=8,device_map=auto \
  --tasks longvideobench_val_v_60s --batch_size 1 \
  --output_path "$OUT/diag_v_60s" --log_samples 2>&1 | tee "$OUT/diag_v_60s.log"

# Parse only a real number -> a DONE that cannot lie. Glob is *results*.json: lmms-eval
# names it "<timestamp>_results.json", so "results*.json" silently matches nothing and
# makes a passing run look failed.
"$PYBIN" - << 'PYEOF'
import glob, json
fs = sorted(glob.glob("/workspace/slm-lab/results/lmmseval/diag_v_60s/**/*results*.json",
                      recursive=True))
if not fs:
    raise SystemExit("REFUSING: no results json -- the run did not produce a number.")
r = json.load(open(fs[-1]))
for task, m in r.get("results", {}).items():
    acc = m.get("lvb_acc,none", m.get("lvb_acc"))
    print(f"\n{task}: {acc}")
print("""
  LDDR 60s uniform@8 = 0.669 | our val_i_60s_k8 = 0.7267
  -> near 0.669 means LDDR's table is the NATIVE-VIDEO pathway (val_v), and val_i's
     +5.8pt is a pathway/token-budget artifact, not a win.
  -> near 0.7267 means both our pathways sit ~6pt above their uniform at 60s, which is a
     protocol gap we have NOT explained. Do not publish either absolute until it resolves.
  Either way the paired picks-vs-uniform tests are unaffected: every arm is val_i.
""")
PYEOF
