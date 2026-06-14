#!/usr/bin/env python3
"""Exploratory analysis of the context-files study runs (experiment_full.db).

For our own understanding — tool use, traces, efficiency, outcomes. Not for a paper.
Generates PNGs + SUMMARY.md into analysis/.

Run:  /tmp/edaenv/bin/python analysis/explore.py
"""
import sqlite3, json, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = "results_from_pod_codex/experiment_full.db"
OUT = "analysis"
AGENTS = ["claude_code", "codex"]
COL = {"claude_code": "#d97706", "codex": "#2563eb"}
os.makedirs(OUT, exist_ok=True)
c = sqlite3.connect(DB)
S = []  # summary lines
def say(x): print(x); S.append(x)


def runs_df():
    cols = ["run_id","task_id","repo","strategy","agent","repeat_index","total_turns",
            "total_duration_s","total_input_tokens","total_output_tokens",
            "total_cache_read_tokens","total_cache_creation_tokens","total_reasoning_tokens",
            "total_tool_calls","unique_files_read","unique_files_written","task_passed","eval_method"]
    rows = c.execute(f"SELECT {','.join(cols)} FROM runs").fetchall()
    numeric = set(cols) - {"run_id","task_id","repo","strategy","agent","eval_method"}
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in numeric:
            if d[k] is None: d[k] = 0
        out.append(d)
    return out


R = runs_df()
def byagent(rows, ag): return [r for r in rows if r["agent"] == ag]

say("# Context-Files Study — Exploratory Analysis\n")
say(f"Source: `{DB}` — {len(R)} runs, {c.execute('SELECT COUNT(*) FROM turns').fetchone()[0]} turns.\n")

# ---------------------------------------------------------------- 1. tool use
say("## 1. Tool-use profile (parsed from turns.tool_calls_json)")
tool_freq = {ag: collections.Counter() for ag in AGENTS}
for ag in AGENTS:
    rows = c.execute("""SELECT t.tool_calls_json FROM turns t JOIN runs r ON t.run_id=r.run_id
                        WHERE r.agent=?""", (ag,)).fetchall()
    for (j,) in rows:
        for call in json.loads(j or "[]"):
            tool_freq[ag][call.get("name", "?")] += 1
    top = tool_freq[ag].most_common(8)
    total = sum(tool_freq[ag].values())
    say(f"- **{ag}**: {total} tool calls — " + ", ".join(f"{n}={k}" for n, k in top))
say("")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ag in zip(axes, AGENTS):
    top = tool_freq[ag].most_common(8)[::-1]
    ax.barh([n for n, _ in top], [k for _, k in top], color=COL[ag])
    ax.set_title(f"{ag} — top tools"); ax.set_xlabel("calls")
plt.tight_layout(); plt.savefig(f"{OUT}/01_tool_use.png", dpi=110); plt.close()

# ---------------------------------------------------- 2. tool calls per run / outcome
say("## 2. Tool calls per run, by outcome")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ag in zip(axes, AGENTS):
    a = byagent(R, ag)
    for passed, lab, col in [(1, "pass", "#16a34a"), (0, "fail", "#dc2626")]:
        v = [r["total_tool_calls"] for r in a if r["task_passed"] == passed]
        if v: ax.hist(v, bins=15, alpha=0.6, label=f"{lab} (n={len(v)}, med {sorted(v)[len(v)//2]})", color=col)
    ax.set_title(f"{ag} — tool calls/run"); ax.set_xlabel("tool calls"); ax.legend()
    p = [r["total_tool_calls"] for r in a if r["task_passed"]==1]
    f = [r["total_tool_calls"] for r in a if r["task_passed"]==0]
    if p and f:
        say(f"- **{ag}**: median tool calls — pass {sorted(p)[len(p)//2]} vs fail {sorted(f)[len(f)//2]}")
plt.tight_layout(); plt.savefig(f"{OUT}/02_toolcalls_outcome.png", dpi=110); plt.close()
say("")

# ----------------------------------------------------------- 3. pass-rate heatmap
say("## 3. Pass rate by repo × strategy")
strategies = ["none", "always_on", "selective"]
repos = sorted(set(r["repo"] for r in R))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ag in zip(axes, AGENTS):
    a = byagent(R, ag)
    M = [[None]*len(strategies) for _ in repos]
    for i, rp in enumerate(repos):
        for j, st in enumerate(strategies):
            v = [r["task_passed"] for r in a if r["repo"]==rp and r["strategy"]==st]
            M[i][j] = sum(v)/len(v) if v else float("nan")
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(strategies))); ax.set_xticklabels(strategies, rotation=20)
    ax.set_yticks(range(len(repos))); ax.set_yticklabels([r.split("/")[-1] for r in repos], fontsize=8)
    for i in range(len(repos)):
        for j in range(len(strategies)):
            if M[i][j]==M[i][j]: ax.text(j,i,f"{M[i][j]:.2f}",ha="center",va="center",fontsize=8)
    ax.set_title(f"{ag} — pass rate")
fig.colorbar(im, ax=axes, shrink=0.7)
plt.savefig(f"{OUT}/03_passrate_heatmap.png", dpi=110); plt.close()
for ag in AGENTS:
    a = byagent(R, ag)
    for st in strategies:
        v=[r["task_passed"] for r in a if r["strategy"]==st]
        if v: say(f"- {ag} {st}: {100*sum(v)/len(v):.1f}% (n={len(v)})")
say("")

# --------------------------------------------------------- 4. duration & tokens
say("## 4. Wall-time & token composition")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# duration by outcome
ax = axes[0]
data, labs, cols = [], [], []
for ag in AGENTS:
    a = byagent(R, ag)
    for passed, tag, col in [(1,"pass","#16a34a"),(0,"fail","#dc2626")]:
        v=[r["total_duration_s"]/60 for r in a if r["task_passed"]==passed]
        if v: data.append(v); labs.append(f"{ag.split('_')[0]}\n{tag}"); cols.append(col)
bp=ax.boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True)
for p,cl in zip(bp["boxes"],cols): p.set_facecolor(cl); p.set_alpha(0.6)
ax.set_ylabel("minutes"); ax.set_title("wall-time per run by outcome")
# token composition stacked
ax = axes[1]
comps = ["total_input_tokens","total_output_tokens","total_reasoning_tokens","total_cache_read_tokens"]
labels_c = ["input","output","reasoning","cache_read"]
bottoms=[0,0]
for comp,lab in zip(comps,labels_c):
    vals=[sum(r[comp] for r in byagent(R,ag))/max(1,len(byagent(R,ag))) for ag in AGENTS]
    ax.bar(AGENTS, vals, bottom=bottoms, label=lab)
    bottoms=[b+v for b,v in zip(bottoms,vals)]
ax.set_ylabel("avg tokens/run"); ax.set_title("token composition"); ax.legend(fontsize=8); ax.set_yscale("log")
plt.tight_layout(); plt.savefig(f"{OUT}/04_time_tokens.png", dpi=110); plt.close()
for ag in AGENTS:
    a=byagent(R,ag)
    say(f"- **{ag}**: avg {sum(r['total_duration_s'] for r in a)/len(a)/60:.1f} min/run | "
        f"in {sum(r['total_input_tokens'] for r in a)//len(a)} / out {sum(r['total_output_tokens'] for r in a)//len(a)} / "
        f"reasoning {sum(r['total_reasoning_tokens'] for r in a)//len(a)} / cache_read {sum(r['total_cache_read_tokens'] for r in a)//len(a)} tok")
say("")

# --------------------------------------------------------- 5. files touched
say("## 5. Files read / written per run")
fig, ax = plt.subplots(figsize=(8,5))
x=range(len(AGENTS)); w=0.35
for off,key,lab,col in [(-w/2,"unique_files_read","read","#0891b2"),(w/2,"unique_files_written","written","#ea580c")]:
    vals=[sum(r[key] for r in byagent(R,ag))/len(byagent(R,ag)) for ag in AGENTS]
    ax.bar([i+off for i in x], vals, w, label=lab, color=col)
ax.set_xticks(list(x)); ax.set_xticklabels(AGENTS); ax.set_ylabel("avg unique files/run"); ax.legend()
ax.set_title("files touched per run")
plt.tight_layout(); plt.savefig(f"{OUT}/05_files.png", dpi=110); plt.close()
for ag in AGENTS:
    a=byagent(R,ag)
    say(f"- **{ag}**: read {sum(r['unique_files_read'] for r in a)/len(a):.1f} / wrote {sum(r['unique_files_written'] for r in a)/len(a):.1f} files/run avg")
say("")

# --------------------------------------------------- 6. effort vs outcome
say("## 6. Does more effort → success? (tool calls binned vs pass rate)")
fig, ax = plt.subplots(figsize=(8,5))
for ag in AGENTS:
    a=sorted(byagent(R,ag), key=lambda r:r["total_tool_calls"])
    # quartile bins
    n=len(a); bins=[a[i*n//4:(i+1)*n//4] for i in range(4)]
    xs=[sum(r["total_tool_calls"] for r in b)/len(b) for b in bins if b]
    ys=[100*sum(r["task_passed"] for r in b)/len(b) for b in bins if b]
    ax.plot(xs, ys, "o-", color=COL[ag], label=ag)
ax.set_xlabel("avg tool calls (quartile bin)"); ax.set_ylabel("pass rate %"); ax.legend()
ax.set_title("effort vs outcome")
plt.tight_layout(); plt.savefig(f"{OUT}/06_effort_vs_pass.png", dpi=110); plt.close()
say("(see 06_effort_vs_pass.png — is the relationship positive, flat, or inverted?)\n")

open(f"{OUT}/SUMMARY.md","w").write("\n".join(S))
print(f"\nwrote {OUT}/SUMMARY.md + 6 PNGs")
