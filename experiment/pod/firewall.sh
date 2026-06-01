#!/usr/bin/env bash
# =============================================================================
# FIREWALL / EGRESS LOCK — run AFTER setup.sh, BEFORE any agent run.
#
# Goal: make a push/PR to GitHub PHYSICALLY impossible while keeping PyPI +
# Anthropic reachable (the per-cell eval installs deps from PyPI; the agent
# talks to api.anthropic.com). This is the network layer on top of the harness
# software layers (scrub remotes, --disallowedTools, blank GH_TOKEN).
#
# PRIMARY method: /etc/hosts blackhole of every GitHub host -> 127.0.0.1.
#   Works in ANY container with no special caps. Blocks git/gh/curl by name.
# OPTIONAL hardening: iptables DROP to github (only if NET_ADMIN is available).
#
# Idempotent. Re-running is safe.
# =============================================================================
set -euo pipefail

MARKER="# === EXP-EGRESS-LOCK (github blackhole) ==="
GH_HOSTS=(
  github.com www.github.com api.github.com codeload.github.com
  ssh.github.com uploads.github.com objects.githubusercontent.com
  raw.githubusercontent.com camo.githubusercontent.com
  github.io githubusercontent.com
)

echo "==> blackholing GitHub via /etc/hosts"
# Remove any prior block first (idempotent), then append a fresh one.
if grep -qF "$MARKER" /etc/hosts; then
  sed -i "/$(printf '%s' "$MARKER" | sed 's/[][\/.*]/\\&/g')/,/# === END-EXP-EGRESS-LOCK ===/d" /etc/hosts
fi
{
  echo "$MARKER"
  for h in "${GH_HOSTS[@]}"; do
    echo "127.0.0.1 $h"
    echo "::1 $h"
  done
  echo "# === END-EXP-EGRESS-LOCK ==="
} >> /etc/hosts
echo "    added ${#GH_HOSTS[@]} GitHub hosts -> 127.0.0.1 / ::1"

# ---- OPTIONAL: iptables hardening (best-effort, needs NET_ADMIN) ----------
# Blackhole protects by hostname; a process using a raw GitHub IP would slip it.
# If we can use iptables, also DROP outbound to GitHub's published CIDRs.
if iptables -L >/dev/null 2>&1; then
  echo "==> iptables available — adding GitHub CIDR DROP rules (defense in depth)"
  # GitHub Meta CIDRs (git+web). Static enough for a short run; refresh if stale.
  GH_CIDRS=(140.82.112.0/20 143.55.64.0/20 192.30.252.0/22 185.199.108.0/22 20.205.243.0/24 20.200.245.0/24)
  for c in "${GH_CIDRS[@]}"; do
    iptables -C OUTPUT -d "$c" -j DROP 2>/dev/null || iptables -A OUTPUT -d "$c" -j DROP
  done
  echo "    added ${#GH_CIDRS[@]} CIDR DROP rules"
else
  echo "==> iptables NOT available (no NET_ADMIN) — relying on /etc/hosts blackhole"
  echo "    (acceptable: remotes are also scrubbed + push tools denied + token blanked)"
fi

echo
echo "==> LOCK APPLIED. Run: bash pod/verify_lock.sh   (must pass before agents)"
