#!/usr/bin/env bash
# Laptop-side: pull the parallel 12-cell codex run (609/614 always_on+selective)
# from pod 108.47 -> results_from_pod_screen/.
set -e
SSHO="-o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 -i $HOME/.ssh/runpod -p 13109"
HOST="root@213.173.108.47"
cd "$(dirname "$0")"; mkdir -p results_from_pod_screen
ssh $SSHO $HOST 'cd /workspace/experiment && tar czf - results/expansion2_codex.db results/expansion2_codex.db-wal results/expansion2_codex.db-shm results/expansion2.log 2>/dev/null' \
  | tar xzf - -C results_from_pod_screen --strip-components=1 2>/dev/null
python3 - <<'PY'
import sqlite3, os
p='results_from_pod_screen/expansion2_codex.db'
if not os.path.exists(p):
    print("no expansion2_codex.db yet"); raise SystemExit
c=sqlite3.connect(p)
n=c.execute("SELECT COUNT(*) FROM runs WHERE agent='codex'").fetchone()[0]
print(f"expansion2 cells: {n}/12")
from collections import defaultdict
s=defaultdict(lambda:[0,0])
for tid,strat,p_ in c.execute("SELECT task_id,strategy,task_passed FROM runs WHERE agent='codex' AND eval_method IS NOT NULL AND (error IS NULL OR error='')"):
    k=(tid.split('__')[-1],strat); s[k][0]+=1; s[k][1]+=(p_ or 0)
for (t,strat),(r,pp) in sorted(s.items()):
    print(f"   {t:6s} {strat:10s} {pp}/{r}")
PY
