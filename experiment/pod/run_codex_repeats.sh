#!/usr/bin/env bash
# =============================================================================
# RUN CODEX REPEATS — full Codex arm: 11 tasks x 3 strategies x 3 repeats = 99
# cells, 3 processes by repo, in tmux. Writes into results/experiment.db with
# agent='codex' (the claude_code rows are untouched → both arms coexist).
#
# Pre-req: codex_setup.sh done, firewall.sh applied, verify_lock.sh codex GREEN,
#          codex authed (ChatGPT plan), smoke passed.
#
# Resumable: clean codex cells already in the DB are SKIPPED. Each repo logs to
# results/codex_<repo>.log; a db-sync sidecar snapshots every 60s.
# =============================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"
export CODEX_HOME="${CODEX_HOME:-$EXP_DIR/pod/codex_home}"

if ! bash pod/verify_lock.sh codex >/tmp/verify_lock_codex.out 2>&1; then
  echo "FATAL: verify_lock.sh codex failed. See /tmp/verify_lock_codex.out. NOT launching."; exit 1
fi
echo "==> lock verified (codex), launching."

export EXP_LIVE="${EXP_LIVE:-1}"
export EXP_CODEX_MODEL="${EXP_CODEX_MODEL:-gpt-5.5}"
export EXP_ABSOLUTE_TIMEOUT="${EXP_ABSOLUTE_TIMEOUT:-7200}"
export EXP_INACTIVITY_TIMEOUT="${EXP_INACTIVITY_TIMEOUT:-1800}"
export EXP_MAX_TURNS="${EXP_MAX_TURNS:-120}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agents-experiment-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EXP_DIR/.uv-cache}"
echo "    model=$EXP_CODEX_MODEL abs=${EXP_ABSOLUTE_TIMEOUT}s inact=${EXP_INACTIVITY_TIMEOUT}s max_turns=$EXP_MAX_TURNS live=$EXP_LIVE"

mkdir -p results
SESS="codex"
tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" -n firebase

launch() {  # repo  window
  local repo="$1" win="$2"
  local cmd="cd '$EXP_DIR'; export PATH=\$HOME/.local/bin:\$PATH CODEX_HOME='$CODEX_HOME' \
EXP_LIVE=$EXP_LIVE EXP_CODEX_MODEL=$EXP_CODEX_MODEL \
EXP_ABSOLUTE_TIMEOUT=$EXP_ABSOLUTE_TIMEOUT EXP_INACTIVITY_TIMEOUT=$EXP_INACTIVITY_TIMEOUT \
EXP_MAX_TURNS=$EXP_MAX_TURNS XDG_CACHE_HOME='$XDG_CACHE_HOME' UV_CACHE_DIR='$UV_CACHE_DIR'; \
python3 run_pilot.py --agent codex --repeats 3 --repo $repo 2>&1 | tee results/codex_$repo.log; \
echo '=== codex $repo DONE ==='; exec bash"
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

tmux new-window -t "$SESS" -n sync
tmux send-keys -t "$SESS:sync" \
  "cd '$EXP_DIR'; while true; do cp -f results/experiment.db results/experiment.db.snapshot 2>/dev/null; sleep 60; done" C-m

echo
echo "==> LAUNCHED codex arm in tmux session '$SESS' (windows: firebase, pdm, opshin, sync)"
echo "    attach : tmux attach -t $SESS"
echo "    monitor: bash pod/monitor.sh        # (agent-agnostic cell counts)"
