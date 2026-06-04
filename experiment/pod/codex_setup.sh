#!/usr/bin/env bash
# =============================================================================
# CODEX SETUP — install the Codex CLI and prepare the locked CODEX_HOME.
#
# Run AFTER setup.sh (repos cloned) and firewall.sh (egress locked). Idempotent.
#
# Auth: ChatGPT-plan login (free quota), NO API key. `codex login` is browser/
# device-code, so on a headless pod use ONE of:
#   (a) device-code flow:   codex login         (follow the printed URL/code)
#   (b) copy from laptop:    codex login on your laptop, then scp
#       ~/.codex/auth.json  ->  $CODEX_HOME/auth.json on the pod
# GPT-5.5 requires ChatGPT-account auth (not API-key) — matches this path.
# =============================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME="${CODEX_HOME:-$EXP_DIR/pod/codex_home}"
mkdir -p "$CODEX_HOME"

echo "== install codex CLI =="
if command -v codex >/dev/null 2>&1; then
  echo "  codex already installed: $(codex --version 2>/dev/null | head -1)"
else
  if command -v npm >/dev/null 2>&1; then
    npm install -g @openai/codex
  else
    echo "  npm not found — install Node first (pod/setup.sh installs node), or use the official installer."
    exit 1
  fi
  echo "  installed: $(codex --version 2>/dev/null | head -1)"
fi

echo "== render deny-hook =="
# hooks.json is also rendered by the harness at run-time, but render here too so a
# manual `codex` invocation under this CODEX_HOME is already guarded.
sed "s#__CODEX_HOME__#$CODEX_HOME#g" "$CODEX_HOME/hooks.json.template" > "$CODEX_HOME/hooks.json"
chmod +x "$CODEX_HOME/deny_push.sh"
echo "  wrote $CODEX_HOME/hooks.json"

echo "== auth status =="
if [ -f "$CODEX_HOME/auth.json" ] || [ -f "$HOME/.codex/auth.json" ]; then
  echo "  auth.json present."
else
  echo "  NOT authed yet. Do ONE of:"
  echo "    CODEX_HOME=$CODEX_HOME codex login        # device-code flow"
  echo "    # or scp your laptop's ~/.codex/auth.json -> $CODEX_HOME/auth.json"
fi

echo
echo "==> codex setup done. Next: bash pod/verify_lock.sh codex"
