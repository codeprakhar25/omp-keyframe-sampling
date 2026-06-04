#!/usr/bin/env bash
# Codex PreToolUse deny-hook — the deny-side analog of claude `--disallowedTools`
# (which codex lacks). Reads the pending tool-call JSON on stdin; if the bash
# command is a git push/commit/remote, gh, or request-pull, it emits a
# permissionDecision=deny response (and logs a loud alarm). Mirrors _DENY_TOOLS /
# _PUSH_ALARM in harness/agent.py. Layer with egress lock + scrub_git_remotes.
set -euo pipefail

payload="$(cat)"

cmd="$(printf '%s' "$payload" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
ti = d.get("tool_input", {}) or {}
# Bash uses tool_input.command; apply_patch/others carry no shell command.
print(ti.get("command", "") or "")
')"

# Blocked patterns. `gh` anchored so it does not match "github"/"high"/etc.
if printf '%s' "$cmd" | grep -Eiq \
  '(^|[^a-z])git[[:space:]]+push|(^|[^a-z])git[[:space:]]+commit|(^|[^a-z])git[[:space:]]+remote|(^|[^a-z])git[[:space:]]+request-pull|(^|[^a-z])gh([[:space:]]|$)'; then
  ts="$(date -u +%FT%TZ)"
  echo "[$ts] 🚨 PUSH/PR ATTEMPT BLOCKED-BY-HOOK: ${cmd}" >> "${CODEX_DENY_LOG:-/tmp/codex_deny.log}"
  echo "🚨 PUSH/PR ATTEMPT BLOCKED-BY-HOOK: ${cmd}" >&2
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"git push/commit/remote and gh are blocked in this sandboxed experiment (no external pushes)."}}'
  exit 0
fi

# Allow: empty stdout + exit 0 = proceed normally.
exit 0
