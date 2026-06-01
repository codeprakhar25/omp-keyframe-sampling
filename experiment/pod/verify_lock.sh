#!/usr/bin/env bash
# =============================================================================
# VERIFY LOCK — HARD GATE. Run after firewall.sh, before launching agents.
#
# Asserts the egress lock is correct in BOTH directions:
#   GitHub      -> must be UNREACHABLE (push impossible)
#   PyPI        -> must be REACHABLE   (per-cell eval installs deps)
#   Anthropic   -> must be REACHABLE   (agent LLM calls)
# Also confirms repos are cloned full and the API key is present.
#
# Exits NON-ZERO on any failure. Do NOT run the pilot if this fails.
# =============================================================================
set -uo pipefail

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

echo "== egress =="
curl_unreachable github.com
curl_unreachable codeload.github.com
curl_unreachable raw.githubusercontent.com
curl_reachable   pypi.org
curl_reachable   files.pythonhosted.org
curl_reachable   api.anthropic.com

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

echo "== api key =="
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then ok "ANTHROPIC_API_KEY set"; else bad "ANTHROPIC_API_KEY missing (source .env)"; fi
if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then bad "GH_TOKEN/GITHUB_TOKEN present in env — UNSET it (no GitHub creds on pod)"; else ok "no GitHub token in env"; fi

echo
if [ "$fail" -eq 0 ]; then
  echo "==> LOCK VERIFIED. Safe to run the pilot."
  exit 0
else
  echo "==> LOCK VERIFICATION FAILED. Do NOT run the pilot until green."
  exit 1
fi
