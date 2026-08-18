#!/bin/bash
# Auto-fix watcher for lmmseval_fast.sh.
# DONE only when GATE15_ACC.txt (parsed number) + PIPELINE_COMPLETE exist.
# Never green-light on process exit / log "DONE tag" alone.
set -uo pipefail

SLM=${SLM:-/workspace/slm-lab}
OUT=${OUT:-$SLM/results/lmmseval}
PY=${PY:-/workspace/lmmsenv/bin/python}
PIP=${PIP:-/workspace/lmmsenv/bin/pip}
FAST=${FAST:-$SLM/scripts/lmmseval_fast.sh}
POLL=${POLL:-20}
MAX_FIX=${MAX_FIX:-8}
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pipcache}"
# never inherit slmenv shadowing
export PYTHONPATH="/workspace/lmms-eval"
mkdir -p "$OUT/log"

status() {
  local msg="[$(date -u +%H:%M:%S)] $*"
  echo "$msg" | tee -a "$OUT/watch.log"
  echo "$msg" > "$OUT/watch_status.txt"
}

driver_alive() {
  local pid
  pid=$(cat "$OUT/fast_driver.pid" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

lmms_alive() {
  pgrep -f "lmms_eval --model qwen3_vl" >/dev/null 2>&1
}

# HARD gate: must have parsed 15s accuracy artifact + pipeline complete marker
finished() {
  [ -s "$OUT/GATE15_ACC.txt" ] || return 1
  [ -f "$OUT/PIPELINE_COMPLETE" ] || return 1
  # sanity: acc looks like a float in (0,1]
  "$PY" - "$OUT/GATE15_ACC.txt" <<'PY' 2>/dev/null
import sys
v=float(open(sys.argv[1]).read().strip())
assert 0.0 < v <= 1.0
PY
}

last_missing_mod() {
  local hit
  hit=$(grep -h "ModuleNotFoundError: No module named" "$OUT"/log/*.log "$OUT"/fast_driver.log 2>/dev/null | tail -1 || true)
  [ -z "$hit" ] && { echo ""; return; }
  echo "$hit" | sed -n "s/.*No module named ['\"]\\([^'\"]*\\)['\"].*/\\1/p"
}

pip_fix() {
  local mod=$1 pkg=$mod
  case "$mod" in
    cv2) pkg=opencv-python-headless ;;
    PIL) pkg=pillow ;;
    sklearn) pkg=scikit-learn ;;
    yaml) pkg=pyyaml ;;
    av) pkg="av<16" ;;
  esac
  status "FIX pip install $pkg (from missing import $mod)"
  $PIP install -q $pkg 2>&1 | tee -a "$OUT/watch.log" | tail -5 || true
}

last_shadow_bug() {
  # slmenv torchvision shadowing symptom
  grep -h "read_video\|slmenv.*torchvision\|No module named 'tenacity'" \
    "$OUT"/log/*.log 2>/dev/null | tail -1 | grep -q .
}

restart_fast() {
  status "RESTART lmmseval_fast (resume skips completed arms w/ real results)"
  if [ -f "$OUT/fast_driver.pid" ]; then
    kill "$(cat "$OUT/fast_driver.pid")" 2>/dev/null || true
  fi
  pkill -f "lmms_eval --model qwen3_vl" 2>/dev/null || true
  sleep 2
  cd "$SLM"
  export BATCH="${BATCH:-1}" NFRAMES="${NFRAMES:-8}"
  unset FORCE_QWENVL_VIDEO_READER  # clean lmmsenv tv has read_video; don't paper over
  {
    echo ""
    echo "[watch $(date -u +%H:%M:%S)] ---- RESTART ----"
    env BATCH="$BATCH" NFRAMES="$NFRAMES" HF_HOME="$HF_HOME" \
      PYTHONPATH="/workspace/lmms-eval" \
      bash "$FAST"
  } >> "$OUT/fast_driver.log" 2>&1 &
  echo $! > "$OUT/fast_driver.pid"
  status "restarted pid=$(cat "$OUT/fast_driver.pid")"
}

FIXES=0
status "watcher up poll=${POLL}s max_fix=$MAX_FIX (DONE requires GATE15_ACC+PIPELINE_COMPLETE)"

while true; do
  if finished; then
    status "OK pipeline DONE gate15_acc=$(cat "$OUT/GATE15_ACC.txt") — watcher exit"
    exit 0
  fi

  if driver_alive || lmms_alive; then
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "?")
    g15="noacc"; [ -s "$OUT/GATE15_ACC.txt" ] && g15=$(cat "$OUT/GATE15_ACC.txt")
    status "RUNNING driver=$(driver_alive && echo y || echo n) lmms=$(lmms_alive && echo y || echo n) gpu=$gpu gate15=$g15 fixes=$FIXES"
  else
    mod=$(last_missing_mod)
    if [ -n "$mod" ] && [ "$FIXES" -lt "$MAX_FIX" ]; then
      FIXES=$((FIXES + 1))
      pip_fix "$mod"
      # wipe failed arm dirs so skip logic doesn't false-positive
      rm -rf "$OUT"/gate15_uniform8_v
      rm -f "$OUT/PIPELINE_COMPLETE" "$OUT/GATE15_PASS"
      restart_fast
    elif last_shadow_bug && [ "$FIXES" -lt "$MAX_FIX" ]; then
      FIXES=$((FIXES + 1))
      status "FIX PYTHONPATH shadowing symptom — restart w/ lmmsenv-only path"
      rm -rf "$OUT"/gate15_uniform8_v
      rm -f "$OUT/PIPELINE_COMPLETE" "$OUT/GATE15_PASS"
      restart_fast
    elif [ "$FIXES" -lt "$MAX_FIX" ]; then
      FIXES=$((FIXES + 1))
      status "DEAD — restart ($FIXES/$MAX_FIX)"
      $PIP install -q tenacity dill tabulate pyyaml 2>/dev/null || true
      rm -rf "$OUT"/gate15_uniform8_v
      rm -f "$OUT/PIPELINE_COMPLETE" "$OUT/GATE15_PASS"
      restart_fast
    else
      status "FAIL too many fixes ($FIXES) — manual intervene"
      exit 1
    fi
  fi
  sleep "$POLL"
done
