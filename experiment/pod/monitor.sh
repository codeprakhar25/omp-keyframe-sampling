#!/usr/bin/env bash
# =============================================================================
# MONITOR — glanceable status of the running pilot. Safe to run any time.
#   no args  -> one snapshot
#   -w       -> refresh every 20s (watch mode)
# =============================================================================
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"

snapshot() {
  echo "================ PILOT STATUS  $(date '+%H:%M:%S') ================"
  echo "tmux procs alive: $(pgrep -fc run_pilot.py 2>/dev/null || echo 0)/3"
  echo
  echo "---- per-repo progress (last lines) ----"
  for r in firebase pdm opshin; do
    f="results/pilot_$r.log"
    [ -f "$f" ] || { echo "[$r] (no log yet)"; continue; }
    echo "[$r] $(grep -E 'RUN|SKIP|passed=|complete:' "$f" 2>/dev/null | grep -vE 'Deprec|warn' | tail -1)"
  done
  echo
  echo "---- 🚨 PUSH/PR ALARMS (must be ZERO) ----"
  alarms=$(grep -rl 'PUSH/PR ATTEMPT' results/*.log 2>/dev/null | wc -l)
  if [ "$alarms" -eq 0 ]; then echo "  none ✓"; else echo "  !!! $alarms log(s) with alarms — inspect immediately"; grep -h 'PUSH/PR ATTEMPT' results/*.log | tail -5; fi
  echo
  echo "---- DB tally ----"
  python3 - <<'PY' 2>/dev/null
import sqlite3
c=sqlite3.connect('results/experiment.db')
done=c.execute("select count(*) from runs where agent='claude_code' and eval_method is not null and (error is null or error='')").fetchone()[0]
err=c.execute("select count(*) from runs where error is not null and error!=''").fetchone()[0]
byrep=list(c.execute("select repeat_index,count(*) from runs where eval_method is not null group by repeat_index"))
print(f"  clean cells: {done}   errored rows: {err}")
print(f"  by repeat_index: {dict(byrep)}   (target: 90 clean = 30 x 3 repeats)")
PY
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; snapshot; sleep 20; done
else
  snapshot
fi
