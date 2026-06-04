#!/usr/bin/env bash
# =============================================================================
# CODEX SMOKE — 3 tasks x strategy=none on Codex, into a SCRATCH db.
#
# Proves the Codex arm end-to-end before spending the full 99-cell batch:
#   - codex exec --json streams + parses (turns/tokens land in the db)
#   - live tool one-liners visible (EXP_LIVE=1)
#   - the PreToolUse deny-hook BLOCKS a planted push (defense-in-depth check)
#   - auth + throughput OK on the ChatGPT-plan quota
#
# Tasks: 939 (firebase, all-pass), 3790 (pdm, borderline), 605 (opshin, all-fail).
# Writes results/codex_smoke.db — NEVER experiment.db.
# Pre-req: codex_setup.sh done, firewall.sh applied, verify_lock.sh codex GREEN.
# =============================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"
export CODEX_HOME="${CODEX_HOME:-$EXP_DIR/pod/codex_home}"

if ! bash pod/verify_lock.sh codex >/tmp/verify_lock_codex.out 2>&1; then
  echo "FATAL: verify_lock.sh codex failed. See /tmp/verify_lock_codex.out. NOT launching."; exit 1
fi
echo "==> lock verified (codex)."

# NOTE: codex 0.137 does not load the PreToolUse hooks.json; the push/commit block
# is enforced by the gh/git PATH-shims (verified to survive codex's bash -lc) plus
# the PATH-independent egress lock + scrub_git_remotes. Self-test the shims:
echo "== git PATH-shim self-test =="
if PATH="$CODEX_HOME/bin:$PATH" git push origin main 2>&1 | grep -q "disabled in this sandboxed"; then
  echo "  ok   git shim blocks push"
else
  echo "  FAIL git shim did NOT block push — ABORT."; exit 1
fi
if ! PATH="$CODEX_HOME/bin:$PATH" git --version >/dev/null 2>&1; then
  echo "  FAIL git shim broke normal git (--version) — ABORT."; exit 1
else
  echo "  ok   git shim passes normal git through"
fi

echo "== gh PATH-shim self-test =="
if PATH="$CODEX_HOME/bin:$PATH" gh --version >/dev/null 2>&1; then
  echo "  FAIL gh shim did NOT shadow gh — ABORT."; exit 1
else
  echo "  ok   gh blocked (shim or not-installed)"
fi

export EXP_LIVE="${EXP_LIVE:-1}"
export EXP_CODEX_MODEL="${EXP_CODEX_MODEL:-gpt-5.5}"
export EXP_ABSOLUTE_TIMEOUT="${EXP_ABSOLUTE_TIMEOUT:-7200}"
export EXP_INACTIVITY_TIMEOUT="${EXP_INACTIVITY_TIMEOUT:-1800}"
export EXP_MAX_TURNS="${EXP_MAX_TURNS:-120}"     # codex has no native turn/budget cap
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agents-experiment-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EXP_DIR/.uv-cache}"
echo "    model=$EXP_CODEX_MODEL live=$EXP_LIVE max_turns=$EXP_MAX_TURNS"

mkdir -p results
python3 run_pilot.py --agent codex --task-file tasks/codex_smoke.json \
  --strategies none --repeats 1 --db results/codex_smoke.db 2>&1 | tee results/codex_smoke.log

echo
echo "== smoke verdicts =="
sqlite3 results/codex_smoke.db \
  "SELECT task_id, strategy, total_turns, total_tool_calls, total_input_tokens, total_output_tokens, total_reasoning_tokens, task_passed, error \
   FROM runs WHERE agent='codex' ORDER BY task_id;" 2>/dev/null || true
echo
echo "== any push attempts logged? (should be none from the agent) =="
cat /tmp/codex_deny.log 2>/dev/null || echo "  (deny log empty — no push attempts)"
echo
echo "==> SMOKE DONE. If turns/tokens populated, verdicts sane, no agent push leaked:"
echo "    scale up with  bash pod/run_codex_repeats.sh"
