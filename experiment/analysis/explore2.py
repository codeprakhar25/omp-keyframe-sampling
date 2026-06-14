#!/usr/bin/env python3
"""Deeper pass: (1) verify effort->fail survives CLEAN (tests-only) eval;
(2) decompose tool-mix in doomed vs winning runs (thrashing).

Run: /tmp/edaenv/bin/python analysis/explore2.py
"""
import sqlite3, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB="results_from_pod_codex/experiment_full.db"; OUT="analysis"
AGENTS=["claude_code","codex"]; COL={"claude_code":"#d97706","codex":"#2563eb"}
c=sqlite3.connect(DB); S=[]
def say(x): print(x); S.append(x)

# per-run: outcome, total tools, eval_method, + tool-name counts from turns
def run_tools(eval_filter):
    q="SELECT run_id,agent,task_passed,total_tool_calls,eval_method FROM runs"
    runs={r[0]:dict(agent=r[1],passed=r[2],tools=r[3] or 0,eval=r[4]) for r in c.execute(q)}
    if eval_filter:
        runs={k:v for k,v in runs.items() if v["eval"]==eval_filter}
    tc=collections.defaultdict(collections.Counter)
    for rid,j in c.execute("SELECT run_id,tool_calls_json FROM turns"):
        if rid not in runs: continue
        for call in json.loads(j or "[]"): tc[rid][call.get("name","?")]+=1
    return runs, tc

say("# Deeper Pass — effort/outcome on CLEAN eval + thrashing decomposition\n")

# ---- 1. effort vs pass: tests-only vs all ----
say("## 1. Effort -> outcome, CLEAN (eval_method='tests') vs all")
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(lab,filt) in zip(axes,[("tests-only","tests"),("all eval methods",None)]):
    for ag in AGENTS:
        runs,_=run_tools(filt)
        a=sorted([v for v in runs.values() if v["agent"]==ag],key=lambda r:r["tools"])
        if len(a)<4: continue
        n=len(a); bins=[a[i*n//4:(i+1)*n//4] for i in range(4)]
        xs=[sum(r["tools"] for r in b)/len(b) for b in bins if b]
        ys=[100*sum(r["passed"] for r in b)/len(b) for b in bins if b]
        ax.plot(xs,ys,"o-",color=COL[ag],label=ag)
        if filt=="tests":
            say(f"- **{ag} (tests-only)**: pass% by tool-call quartile = "+
                " -> ".join(f"{y:.0f}%" for y in ys)+f"  (n={n})")
    ax.set_title(f"effort vs pass — {lab}"); ax.set_xlabel("avg tool calls (quartile)")
    ax.set_ylabel("pass %"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/07_effort_clean.png",dpi=110); plt.close()
say("")

# ---- 2. tool-mix: doomed (top-quartile tools) vs winning ----
say("## 2. Tool-mix — what dominates the long doomed runs? (tests-only)")
runs,tc=run_tools("tests")
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,ag in zip(axes,AGENTS):
    a=[(rid,v) for rid,v in runs.items() if v["agent"]==ag]
    # split by outcome
    names=set()
    for rid,_ in a: names|=set(tc[rid])
    names=sorted(names, key=lambda nm:-sum(tc[rid][nm] for rid,_ in a))[:5]
    groups={"pass":[r for r in a if r[1]["passed"]==1],"fail":[r for r in a if r[1]["passed"]==0]}
    width=0.35; x=range(len(names))
    for off,(grp,rows),hatch in [(-width/2,("pass",groups["pass"]),""),(width/2,("fail",groups["fail"]),"//")]:
        if not rows: continue
        avg=[sum(tc[rid][nm] for rid,_ in rows)/len(rows) for nm in names]
        ax.bar([i+off for i in x],avg,width,label=grp,hatch=hatch,
               color="#16a34a" if grp=="pass" else "#dc2626",alpha=0.75)
    ax.set_xticks(list(x)); ax.set_xticklabels(names,rotation=20,fontsize=8)
    ax.set_ylabel("avg calls/run"); ax.set_title(f"{ag} — tool mix pass vs fail"); ax.legend()
    # numbers
    for grp,rows in groups.items():
        if rows:
            tot=sum(sum(tc[rid].values()) for rid,_ in rows)/len(rows)
            say(f"- {ag} {grp}: {tot:.0f} tools/run avg (n={len(rows)})")
plt.tight_layout(); plt.savefig(f"{OUT}/08_toolmix_outcome.png",dpi=110); plt.close()
say("")

# ---- 3. how lopsided is the work? top tool share in fails ----
say("## 3. Concentration — share of the single most-used tool (tests-only)")
for ag in AGENTS:
    for grp in ("pass","fail"):
        rows=[rid for rid,v in runs.items() if v["agent"]==ag and v["passed"]==(1 if grp=="pass" else 0)]
        if not rows: continue
        shares=[]
        for rid in rows:
            tot=sum(tc[rid].values())
            if tot: shares.append(max(tc[rid].values())/tot)
        say(f"- {ag} {grp}: top tool = {100*sum(shares)/len(shares):.0f}% of all calls (avg)")
say("")

open(f"{OUT}/SUMMARY2.md","w").write("\n".join(S))
print(f"\nwrote {OUT}/SUMMARY2.md + 2 PNGs (07,08)")
