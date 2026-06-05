#!/usr/bin/env bash
# Laptop-side: pull the codex run's db snapshot + logs from the pod.
set -e
SSHO="-o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 -i $HOME/.ssh/runpod -p 24783"
HOST="root@213.173.107.19"
cd "$(dirname "$0")"; mkdir -p results_from_pod_codex
# Pull the LIVE db + WAL + SHM (sqlite replays the WAL on open). The sidecar's
# experiment.db.snapshot misses rows still buffered in -wal, so grab all three.
ssh $SSHO $HOST 'cd /workspace/experiment && tar czf - results/experiment.db results/experiment.db-wal results/experiment.db-shm results/codex_*.log 2>/dev/null' \
  | tar xzf - -C results_from_pod_codex --strip-components=1 2>/dev/null
python3 - <<'PY'
import sqlite3,os
p='results_from_pod_codex/experiment.db'
c=sqlite3.connect(p)
n=c.execute("SELECT COUNT(*) FROM runs WHERE agent='codex'").fetchone()[0]
ok=c.execute("SELECT COUNT(*) FROM runs WHERE agent='codex' AND (error IS NULL OR error='')").fetchone()[0]
print(f"codex cells: {n}/99  (clean: {ok})")
for r in c.execute("SELECT task_id,strategy,eval_method,total_tool_calls,task_passed FROM runs WHERE agent='codex' ORDER BY rowid DESC LIMIT 8"):
    print("  ",r[0].split('__')[-1],r[1],r[2],"tools=%s"%r[3],"pass=%s"%r[4])
PY
