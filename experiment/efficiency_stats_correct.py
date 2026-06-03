"""Correct efficiency stats: per-task means (n=11 tasks), paired Wilcoxon, Holm.

Fixes the pseudoreplication in the earlier analysis (which treated 3 repeats of a
task as independent -> n=15). The repeats of one task are correlated; the unit of
analysis is the TASK. So we average repeats within (task, strategy) -> one value
per task per strategy -> paired across the 11 tasks (n=11).

Primary comparison: none vs selective (the non-mechanical one; always_on vs
selective is partly tautological since always_on re-injects AGENTS.md every turn).

Holm-Bonferroni correction across the family of reported tests.
"""
import sqlite3, sys, statistics
from scipy.stats import wilcoxon

DB = sys.argv[1] if len(sys.argv) > 1 else "results/experiment.db"
c = sqlite3.connect(DB)
CL = "eval_method is not null and (error is null or error='')"

rows = c.execute(f"""select task_id, strategy, repeat_index,
                      total_turns, total_cache_read_tokens,
                      total_cache_creation_tokens, total_tool_calls, total_duration_s
                     from runs where {CL}""").fetchall()

# per-task-per-strategy means over repeats
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))  # metric -> (task,strat) -> [vals]
metrics = ["turns","cache_read","cache_create","tools","duration"]
idx = {"turns":3,"cache_read":4,"cache_create":5,"tools":6,"duration":7}
tasks=set()
for r in rows:
    t,s = r[0], r[1]; tasks.add(t)
    for m in metrics:
        agg[m][(t,s)].append(r[idx[m]])
tasks=sorted(tasks)
print(f"DB: {DB}")
print(f"tasks (n) = {len(tasks)}  | cells = {len(rows)}\n")

def paired(metric, a, b):
    xa, xb = [], []
    for t in tasks:
        va = agg[metric].get((t,a)); vb = agg[metric].get((t,b))
        if not va or not vb: continue
        xa.append(statistics.mean(va)); xb.append(statistics.mean(vb))
    # direction: how many tasks have b < a (b cheaper)
    bcheaper = sum(1 for i in range(len(xa)) if xb[i] < xa[i])
    try:
        stat,p = wilcoxon(xa, xb)  # two-sided, exact for small n
    except ValueError:
        p = float("nan")
    return len(xa), statistics.median(xa), statistics.median(xb), bcheaper, p

pairs = [("none","selective"), ("none","always_on"), ("always_on","selective")]
results=[]
print("="*78)
print("PER-TASK PAIRED WILCOXON (n = tasks, repeats averaged)")
print("="*78)
for metric in ["turns","cache_read","cache_create","duration"]:
    print(f"\n{metric}:")
    for a,b in pairs:
        n,ma,mb,bch,p = paired(metric,a,b)
        results.append((metric,a,b,p))
        print(f"   {a:9} vs {b:9}  n={n}  median {ma:>10.0f} vs {mb:>10.0f}  "
              f"({b} cheaper in {bch}/{n})  p={p:.4f}")

# Holm-Bonferroni across all tests
print("\n" + "="*78)
print("HOLM-BONFERRONI CORRECTION (family = all tests above)")
print("="*78)
valid=[(m,a,b,p) for (m,a,b,p) in results if p==p]
valid.sort(key=lambda x: x[3])
k=len(valid)
print(f"family size m = {k}\n")
prev=0
for i,(m,a,b,p) in enumerate(valid):
    adj = min(1.0, p*(k-i))
    adj = max(adj, prev); prev=adj
    sig = "***" if adj<0.001 else "**" if adj<0.01 else "*" if adj<0.05 else "ns"
    print(f"  {m:12} {a:9} vs {b:9}  raw p={p:.4f}  Holm p={adj:.4f}  {sig}")

print("\nNote: min possible two-sided exact p at n=11 ~ 0.001; at n=5 ~ 0.0625.")
print("Primary scientific test = none vs selective (non-mechanical).")
