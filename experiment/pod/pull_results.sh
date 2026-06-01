#!/usr/bin/env bash
# =============================================================================
# PULL RESULTS — run on YOUR LAPTOP, not the pod.
#
# Laptop pulls the DB + result JSON from the pod on a loop (pod can't push to a
# laptop behind NAT, so the laptop initiates). This is the off-pod copy that
# survives even if the RunPod volume is deleted.
#
# Usage:
#   POD="root@<pod-ip> -p <ssh-port>"  bash pod/pull_results.sh
#   (or hardcode POD below). Ctrl-C to stop.
# =============================================================================
set -uo pipefail

# RunPod gives an ssh string like: ssh root@194.x.x.x -p 12345 -i ~/.ssh/id_ed25519
POD="${POD:?set POD='root@<ip> -p <port>' (the RunPod SSH target)}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/experiment/results}"
LOCAL_DIR="${LOCAL_DIR:-./results_from_pod}"
INTERVAL="${INTERVAL:-120}"

mkdir -p "$LOCAL_DIR"
echo "==> pulling $POD:$REMOTE_DIR -> $LOCAL_DIR every ${INTERVAL}s (Ctrl-C to stop)"

# Build rsync -e ssh string from POD (host + port/flags).
host="${POD%% *}"; sshflags="${POD#"$host"}"
while true; do
  rsync -az --partial -e "ssh $sshflags" \
    --include='*.db' --include='*.db.snapshot' --include='*.json' \
    --include='*.log' --include='*.stream.jsonl' \
    "$host:$REMOTE_DIR/" "$LOCAL_DIR/" 2>/dev/null \
    && echo "  $(date '+%H:%M:%S') synced ($(ls "$LOCAL_DIR" | wc -l) files)" \
    || echo "  $(date '+%H:%M:%S') sync failed (pod down? retrying)"
  sleep "$INTERVAL"
done
