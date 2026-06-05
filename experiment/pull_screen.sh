#!/usr/bin/env bash
# Laptop-side: pull the task-expansion screener's db + log from the 2nd pod.
set -e
SSHO="-o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 -i $HOME/.ssh/runpod -p 16439"
HOST="root@213.173.110.36"
cd "$(dirname "$0")"; mkdir -p results_from_pod_screen
# Pull LIVE db + WAL + SHM (sqlite replays WAL on open) + the screen log + report.
# Screener now runs on CODEX (free pool) -> results/screen_codex.db, agent='codex'.
ssh $SSHO $HOST 'cd /workspace/experiment && tar czf - results/screen_codex.db results/screen_codex.db-wal results/screen_codex.db-shm results/screen.log tasks/screen_report.md tasks/screen_keepers.json 2>/dev/null' \
  | tar xzf - -C results_from_pod_screen --strip-components=1 2>/dev/null
python3 - <<'PY'
import sqlite3, os
p='results_from_pod_screen/screen_codex.db'
if not os.path.exists(p):
    print("no screen_codex.db yet"); raise SystemExit
c=sqlite3.connect(p)
n=c.execute("SELECT COUNT(*) FROM runs WHERE strategy='none' AND agent='codex'").fetchone()[0]
print(f"screen cells: {n}/84")
# pass-rate per candidate -> borderline classification preview
from collections import defaultdict
s=defaultdict(lambda:[0,0])
for tid,p_ in c.execute("SELECT task_id,task_passed FROM runs WHERE strategy='none' AND agent='codex' AND eval_method IS NOT NULL AND (error IS NULL OR error='')"):
    s[tid][0]+=1; s[tid][1]+=(p_ or 0)
border=[t for t,(r,pp) in s.items() if 0<pp<r]
print(f"  borderline so far: {len(border)}")
for t,(r,pp) in sorted(s.items()):
    cls="BORDERLINE" if 0<pp<r else ("hard" if pp==0 else "easy")
    print(f"   {t.split('__')[-1]:8s} {pp}/{r} {cls}")
PY
