#!/usr/bin/env bash
# Laptop-side: pull the task-expansion 24-cell codex run (always_on+selective for
# the 4 new borderline tasks) from pod 110.36 -> results_from_pod_screen/.
set -e
SSHO="-o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 -i $HOME/.ssh/runpod -p 16439"
HOST="root@213.173.110.36"
cd "$(dirname "$0")"; mkdir -p results_from_pod_screen
ssh $SSHO $HOST 'cd /workspace/experiment && tar czf - results/expansion_codex.db results/expansion_codex.db-wal results/expansion_codex.db-shm results/expansion.log 2>/dev/null' \
  | tar xzf - -C results_from_pod_screen --strip-components=1 2>/dev/null
python3 - <<'PY'
import sqlite3, os
p='results_from_pod_screen/expansion_codex.db'
if not os.path.exists(p):
    print("no expansion_codex.db yet"); raise SystemExit
c=sqlite3.connect(p)
n=c.execute("SELECT COUNT(*) FROM runs WHERE agent='codex'").fetchone()[0]
print(f"expansion cells: {n}/24")
from collections import defaultdict
s=defaultdict(lambda:[0,0])
for tid,strat,p_ in c.execute("SELECT task_id,strategy,task_passed FROM runs WHERE agent='codex' AND eval_method IS NOT NULL AND (error IS NULL OR error='')"):
    k=(tid.split('__')[-1],strat); s[k][0]+=1; s[k][1]+=(p_ or 0)
for (t,strat),(r,pp) in sorted(s.items()):
    print(f"   {t:6s} {strat:10s} {pp}/{r}")
PY
