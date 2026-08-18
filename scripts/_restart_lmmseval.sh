#!/bin/bash
# One-shot: sync-safe restart of fast+watch under lmmsenv-only PYTHONPATH.
set -euo pipefail
cd /workspace/slm-lab
chmod +x scripts/lmmseval_*.sh
[ -f scripts/HANDOFF_LMMSEVAL.md ] && mv -f scripts/HANDOFF_LMMSEVAL.md ./ || true

# kill old
for pidfile in results/lmmseval/fast_driver.pid results/lmmseval/watch.pid; do
  if [ -f "$pidfile" ]; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
  fi
done
sleep 1
ps -eo pid,cmd | awk '/lmmseval_fast\.sh|lmmseval_watch\.sh|lmms_eval --model qwen3_vl/ && !/awk|restart_lmmseval/ {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

# wipe failed/partial gate from shadowed run
rm -rf results/lmmseval/gate15_uniform8_v
rm -f results/lmmseval/PIPELINE_COMPLETE results/lmmseval/GATE15_ACC.txt results/lmmseval/GATE15_PASS

export HF_HOME=/workspace/hf
export BATCH=1
export NFRAMES=8
unset FORCE_QWENVL_VIDEO_READER || true
export PYTHONPATH=/workspace/lmms-eval

# preflight with the REAL interpreter used by pipeline
/workspace/lmmsenv/bin/python - <<'PY'
import sys
sys.path = [p for p in sys.path if "slmenv" not in p]
import torchvision
assert hasattr(torchvision.io, "read_video"), torchvision.__version__
print("preflight_tv_ok", torchvision.__version__, torchvision.__file__)
print("sys.executable", sys.executable)
# prove slmenv tv (no read_video) is NOT winning if that env exists
try:
    import importlib.util
    p = "/workspace/slm-lab/slmenv/lib/python3.11/site-packages/torchvision/__init__.py"
    print("slmenv_tv_exists", __import__("os").path.exists(p))
except Exception as e:
    print("slmenv_check", e)
PY

: > results/lmmseval/fast_driver.log
nohup env BATCH=1 NFRAMES=8 HF_HOME=/workspace/hf PYTHONPATH=/workspace/lmms-eval \
  bash scripts/lmmseval_fast.sh >> results/lmmseval/fast_driver.log 2>&1 &
echo $! > results/lmmseval/fast_driver.pid

nohup env PYTHONPATH=/workspace/lmms-eval bash scripts/lmmseval_watch.sh \
  > results/lmmseval/watch_stdout.log 2>&1 &
echo $! > results/lmmseval/watch.pid

sleep 12
echo "fast=$(cat results/lmmseval/fast_driver.pid) watch=$(cat results/lmmseval/watch.pid)"
echo "=== fast_driver.log ==="
tail -50 results/lmmseval/fast_driver.log
echo "=== watch_status ==="
cat results/lmmseval/watch_status.txt 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
pgrep -af "lmmseval_fast|lmms_eval --model" | head -8
