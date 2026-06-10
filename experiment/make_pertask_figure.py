#!/usr/bin/env python3
"""Per-task pass-rate dot plot (Fig. 1 in the paper).

Two panels (Claude, Codex). Tasks on y-axis, ordered by mean pass-rate so the
floor/ceiling structure (and the rare borderline tasks) is visible at a glance.
Each task shows three markers: none / always_on / selective. Identical-rate
markers are nudged horizontally so overlaps stay readable.

Usage: python make_pertask_figure.py [experiment_full.db] [out.pdf]
"""
import sqlite3
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DB = sys.argv[1] if len(sys.argv) > 1 else "results_from_pod_codex/experiment_full.db"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/figures/pertask.pdf"

STRATS = ["none", "always_on", "selective"]
LABELS = {"none": "none", "always_on": "always_on", "selective": "selective"}
COLORS = {"none": "#4C4C4C", "always_on": "#D1495B", "selective": "#2E86AB"}
MARKERS = {"none": "o", "always_on": "s", "selective": "^"}

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT agent, task_id, repo, strategy, COUNT(*) n, SUM(task_passed) p "
    "FROM runs GROUP BY agent, task_id, strategy"
).fetchall()


def short(task_id, repo):
    num = task_id.split("__")[-1]
    rep = repo.split("/")[-1].replace("firebase-admin-python", "firebase")
    return f"{rep}#{num}"


data = {}  # agent -> {task_label -> {strat -> rate}}
for agent, task_id, repo, strat, n, p in rows:
    lab = short(task_id, repo)
    data.setdefault(agent, {}).setdefault(lab, {})[strat] = p / n

AGENTS = [("claude_code", "Claude Code (sonnet-4-6)"), ("codex", "Codex (gpt-5.5)")]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.6), sharex=True)

for ax, (agent, title) in zip(axes, AGENTS):
    tasks = data[agent]
    # order: lowest mean pass-rate at bottom, highest at top
    order = sorted(tasks, key=lambda t: (sum(tasks[t].values()) / len(tasks[t]), t))
    ypos = {t: i for i, t in enumerate(order)}
    for t in order:
        y = ypos[t]
        rates = tasks[t]
        # group strategies by identical rate to nudge overlaps apart
        by_rate = {}
        for s in STRATS:
            if s in rates:
                by_rate.setdefault(round(rates[s], 6), []).append(s)
        for rate, ss in by_rate.items():
            offs = [(k - (len(ss) - 1) / 2) * 0.16 for k in range(len(ss))]
            for s, dy in zip(ss, offs):
                ax.scatter(
                    rate, y + dy, s=42, color=COLORS[s], marker=MARKERS[s],
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlim(-0.08, 1.08)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(["0", "1/3", "2/3", "1"], fontsize=8)
    ax.set_xlabel("pass-rate (3 repeats)", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.axvspan(0.0, 0.001, color="#cccccc", alpha=0.0)  # noop, keep spacing
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

handles = [
    Line2D([0], [0], marker=MARKERS[s], color="w", markerfacecolor=COLORS[s],
           markeredgecolor="white", markersize=8, label=LABELS[s])
    for s in STRATS
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.95])
import os

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
