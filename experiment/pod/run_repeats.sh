#!/usr/bin/env bash
# =============================================================================
# RUN REPEATS — launch the repeats=3 pilot, 3 processes by repo, in tmux.
#
# Pre-req: setup.sh OK, firewall.sh applied, verify_lock.sh GREEN, .env sourced.
#
# Runs run_pilot.py --repeats 3 per repo (separate workspaces -> no conflict).
# repeat-0 cells already in the DB are SKIPPED by the resumable driver, so this
# only fills repeats 1 and 2 = 60 new cells (10 tasks x 3 strat x 2).
#
# Each repo logs to results/pilot_<repo>.log. A db-sync sidecar snapshots the
# DB to the volume every 60s. Watch with: bash pod/monitor.sh
# =============================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "FATAL: ANTHROPIC_API_KEY not set. Run: set -a; source .env; set +a"; exit 1
fi
# Hard refuse to run if the lock isn't verified.
if ! bash pod/verify_lock.sh >/tmp/verify_lock.out 2>&1; then
  echo "FATAL: verify_lock.sh failed. See /tmp/verify_lock.out. NOT launching."; exit 1
fi
echo "==> lock verified, launching."

# Caps already defaulted high in the harness; re-export here so they're explicit
# in the run environment and visible in logs.
export EXP_MAX_COST_USD="${EXP_MAX_COST_USD:-6.0}"
export EXP_ABSOLUTE_TIMEOUT="${EXP_ABSOLUTE_TIMEOUT:-7200}"
export EXP_INACTIVITY_TIMEOUT="${EXP_INACTIVITY_TIMEOUT:-1800}"
export EXP_LIVE="${EXP_LIVE:-1}"           # per-turn one-liners in the logs
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agents-experiment-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EXP_DIR/.uv-cache}"
echo "    caps: cost=\$$EXP_MAX_COST_USD abs=${EXP_ABSOLUTE_TIMEOUT}s inact=${EXP_INACTIVITY_TIMEOUT}s live=$EXP_LIVE"

mkdir -p results
SESS="pilot"
tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" -n firebase

launch() {  # repo  window
  local repo="$1" win="$2"
  local cmd="cd '$EXP_DIR'; set -a; source .env; set +a; \
export EXP_LIVE=$EXP_LIVE EXP_MAX_COST_USD=$EXP_MAX_COST_USD \
EXP_ABSOLUTE_TIMEOUT=$EXP_ABSOLUTE_TIMEOUT EXP_INACTIVITY_TIMEOUT=$EXP_INACTIVITY_TIMEOUT \
XDG_CACHE_HOME='$XDG_CACHE_HOME' UV_CACHE_DIR='$UV_CACHE_DIR'; \
python3 run_pilot.py --repeats 3 --repo $repo 2>&1 | tee results/pilot_$repo.log; \
echo '=== $repo DONE ==='; exec bash"
  if [ "$win" = "0" ]; then
    tmux send-keys -t "$SESS:0" "$cmd" C-m
  else
    tmux new-window -t "$SESS" -n "$repo"
    tmux send-keys -t "$SESS:$repo" "$cmd" C-m
  fi
}

launch firebase 0
launch pdm      1
launch opshin   2

# db-sync sidecar: snapshot the DB to the volume every 60s (cheap insurance on
# top of the live writes). The laptop-pull is separate (see pod/pull_results.sh).
tmux new-window -t "$SESS" -n sync
tmux send-keys -t "$SESS:sync" \
  "cd '$EXP_DIR'; while true; do cp -f results/experiment.db results/experiment.db.snapshot 2>/dev/null; sleep 60; done" C-m

echo
echo "==> LAUNCHED in tmux session '$SESS' (windows: firebase, pdm, opshin, sync)"
echo "    attach : tmux attach -t $SESS    (switch windows: Ctrl-b <n>; detach: Ctrl-b d)"
echo "    monitor: bash pod/monitor.sh"
