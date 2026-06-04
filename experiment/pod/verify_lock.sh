#!/usr/bin/env bash
# =============================================================================
# VERIFY LOCK — HARD GATE. Run after firewall.sh, before launching agents.
#
# Asserts the egress lock is correct in BOTH directions:
#   GitHub      -> must be UNREACHABLE (push impossible)
#   PyPI        -> must be REACHABLE   (per-cell eval installs deps)
#   Anthropic   -> must be REACHABLE   (claude agent LLM calls)
#   OpenAI/ChatGPT -> must be REACHABLE (codex agent, when AGENT=codex)
# Also confirms repos are cloned full and the agent's auth + safety hook present.
#
# Usage: bash verify_lock.sh [claude|codex]   (default claude)
# Exits NON-ZERO on any failure. Do NOT run the pilot if this fails.
# =============================================================================
set -uo pipefail

AGENT="${1:-claude}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
ok()   { echo "  ok   $*"; }
bad()  { echo "  FAIL $*"; fail=1; }

curl_unreachable() {  # host -> expect failure
  if curl -sS --max-time 8 "https://$1" -o /dev/null 2>/dev/null; then
    bad "$1 is REACHABLE (should be blocked)"
  else
    ok "$1 blocked"
  fi
}
curl_reachable() {    # host -> expect success
  if curl -sS --max-time 15 "https://$1" -o /dev/null 2>/dev/null; then
    ok "$1 reachable"
  else
    bad "$1 UNREACHABLE (eval/agent will break)"
  fi
}

echo "== egress (agent=$AGENT) =="
curl_unreachable github.com
curl_unreachable codeload.github.com
curl_unreachable raw.githubusercontent.com
curl_reachable   pypi.org
curl_reachable   files.pythonhosted.org
if [ "$AGENT" = "codex" ]; then
  curl_reachable chatgpt.com
  curl_reachable api.openai.com
else
  curl_reachable api.anthropic.com
fi

echo "== git push path =="
# A dummy push must NOT resolve to a real GitHub endpoint.
if git ls-remote https://github.com/firebase/firebase-admin-python.git >/dev/null 2>&1; then
  bad "git can still reach github (ls-remote succeeded)"
else
  ok "git ls-remote to github fails"
fi

echo "== repos cloned full =="
for slug in firebase__firebase-admin-python pdm-project__pdm OpShin__opshin; do
  if [ -d "$EXP_DIR/repos/$slug/.git" ]; then
    # full clone => not shallow
    if [ -f "$EXP_DIR/repos/$slug/.git/shallow" ]; then
      bad "$slug is SHALLOW (need full clone for offline checkout)"
    else
      ok "$slug cloned (full)"
    fi
  else
    bad "$slug NOT cloned (run setup.sh)"
  fi
done

echo "== auth =="
if [ "$AGENT" = "codex" ]; then
  CH="${CODEX_HOME:-$EXP_DIR/pod/codex_home}"
  if [ -f "$CH/auth.json" ] || [ -f "$HOME/.codex/auth.json" ]; then
    ok "codex auth.json present (CODEX_HOME=$CH)"
  else
    bad "codex NOT authed — run 'codex login' or copy auth.json into $CH (see codex_setup.sh)"
  fi
  command -v codex >/dev/null 2>&1 && ok "codex CLI installed ($(codex --version 2>/dev/null | head -1))" \
    || bad "codex CLI not installed (run pod/codex_setup.sh)"
else
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then ok "ANTHROPIC_API_KEY set"; else bad "ANTHROPIC_API_KEY missing (source .env)"; fi
fi

echo "== codex safety layers =="
# Primary push/commit block (codex 0.137 doesn't load the hooks.json): PATH-shims.
if [ -x "$EXP_DIR/pod/codex_home/bin/git" ] && [ -x "$EXP_DIR/pod/codex_home/bin/gh" ]; then
  ok "git + gh PATH-shims present + executable"
  # capture-then-grep: under `set -o pipefail`, `git push(exit1) | grep` returns 1
  # even on a match, so a piped `if` would misread. Grab output first.
  shim_out="$(PATH="$EXP_DIR/pod/codex_home/bin:$PATH" git push origin x 2>&1 || true)"
  if printf '%s' "$shim_out" | grep -q "disabled in this sandboxed"; then
    ok "git shim blocks push (live check)"
  else
    [ "$AGENT" = "codex" ] && bad "git shim did NOT block a push — fix before running" \
      || ok "git shim live-check n/a for claude arm"
  fi
else
  [ "$AGENT" = "codex" ] && bad "git/gh PATH-shims missing/not-exec at pod/codex_home/bin/ (chmod +x)" \
    || ok "PATH-shims n/a for claude arm"
fi

echo "== github creds =="
if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then bad "GH_TOKEN/GITHUB_TOKEN present in env — UNSET it (no GitHub creds on pod)"; else ok "no GitHub token in env"; fi

echo
if [ "$fail" -eq 0 ]; then
  echo "==> LOCK VERIFIED. Safe to run the pilot."
  exit 0
else
  echo "==> LOCK VERIFICATION FAILED. Do NOT run the pilot until green."
  exit 1
fi
