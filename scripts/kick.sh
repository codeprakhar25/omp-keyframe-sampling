#!/usr/bin/env bash
# Wait for the in-flight picks rebuild, preflight it, then hand off to overnight.sh.
# Detached + resumable: safe to lose the laptop the moment this is launched.
set -uo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
PY=$SLM/slmenv/bin/python
log () { echo "[$(date -u +%H:%M:%S)] $*"; }

log "waiting for the picks rebuild to finish ..."
while pgrep -f "[m]ake_picks_all|[g]en_omp_picks|[e]xport_picks_lmmseval|[s]core_from_embeds" >/dev/null; do
  sleep 20
done
log "picks rebuild done"

P=$SLM/results/picks_lmmseval
ok=1
for b in 15 60 600; do
  log "preflight bin ${b}s:"
  $PY scripts/preflight_picks.py "$b" "$P/picks_lc_k8.json" "$P/picks_omp_lc_k8.json" \
    2>&1 | while read -r l; do log "  $l"; done
  $PY scripts/preflight_picks.py "$b" "$P/picks_lc_k8.json" "$P/picks_omp_lc_k8.json" \
    >/dev/null 2>&1 || ok=0
done
if [ "$ok" -ne 1 ]; then
  log "!! PREFLIGHT FAILED — picks still do not cover a bin. NOT launching."
  log "!! (this is the check that was missing when k=8 died at 60% on a single qid)"
  exit 1
fi
log "preflight PASSED for 15/60/600 — handing off to overnight.sh"
touch "$SLM/results/stages/picks_all.done"
exec bash scripts/overnight.sh
