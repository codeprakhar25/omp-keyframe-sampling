#!/usr/bin/env bash
# Laptop-side: pull the Claude borderline run (4 tasks x 3 strat x 3 rep) from
# pod 108.47:35061 -> results_from_pod_screen/.
set -e
SSHO="-o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 -i $HOME/.ssh/runpod -p 35061"
HOST="root@213.173.108.47"
cd "$(dirname "$0")"; mkdir -p results_from_pod_screen
ssh $SSHO $HOST 'cd /workspace/experiment && tar czf - results/claude_borderline.db results/claude_borderline.db-wal results/claude_borderline.db-shm results/claude_borderline.log 2>/dev/null' \
  | tar xzf - -C results_from_pod_screen --strip-components=1 2>/dev/null
python3 - <<'PY'
import sqlite3, os
p='results_from_pod_screen/claude_borderline.db'
if not os.path.exists(p): print("no claude_borderline.db yet"); raise SystemExit
c=sqlite3.connect(p)
n=c.execute("SELECT COUNT(*) FROM runs WHERE agent='claude_code' AND eval_method IS NOT NULL").fetchone()[0]
print(f"claude cells: {n}/36")
from collections import defaultdict
s=defaultdict(lambda:[0,0])
for tid,strat,p_ in c.execute("SELECT task_id,strategy,task_passed FROM runs WHERE agent='claude_code' AND eval_method IS NOT NULL AND (error IS NULL OR error='')"):
    k=(tid.split('__')[-1],strat); s[k][0]+=1; s[k][1]+=(p_ or 0)
for (t,strat),(r,pp) in sorted(s.items()):
    print(f"   {t:6s} {strat:10s} {pp}/{r}")
PY
