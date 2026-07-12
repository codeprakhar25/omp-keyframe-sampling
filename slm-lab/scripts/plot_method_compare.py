#!/usr/bin/env python3
"""Frame-retrieval comparison across every selector WE ran + the recall-vs-k recovery.

All arms scored on the SAME task: does a gold frame land in top-k, echo answerer, n=25/bin,
k=6 unless noted. Numbers read from results/ (scorer_swap archived) — no literature here because
the timeline papers report MCQA accuracy, NOT frame hit@k (see keyframe_sampling_timeline.md:127).

Left panel: hit@6 vs video length, one line per method (the honest apples-to-apples).
Right panel: SigLIP so400m recall vs k (budget) — shows the wall is tight-k, loosens at k=40.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BINS = ["15s", "60s", "600s", "3600s"]
X = np.arange(len(BINS))

# hit@6, k=6, echo, n=25/bin
METHODS = {
    "SigLIP so400m top-6 (OURS)": ([0.96, 0.72, 0.36, 0.24], "#CC3311", "o", 2.4),
    "SigLIP2 top-6":              ([0.88, 0.76, 0.24, 0.08], "#EE7733", "s", 1.4),
    "Uniform (no selection)":     ([1.00, 0.56, 0.04, 0.00], "#999999", "D", 1.4),
    "Grounding-DINO region":      ([0.94, 0.61, 0.13, 0.04], "#009988", "^", 1.4),
    "X-CLIP videoret (clip-pool)":([0.44, 0.16, 0.12, 0.04], "#AA3377", "v", 1.4),
}
# Marengo 3.0 commercial: only 3600s ran (n=10), lenient .30 / strict .10
MARENGO = (3, [0.10, 0.30])

# recall vs k, SigLIP so400m
KS = [6, 12, 20, 40]
RECALL = {"15s": [0.96, 1.0, 1.0, 1.0], "60s": [0.72, 0.72, 0.92, 1.0],
          "600s": [0.36, 0.36, 0.44, 0.6], "3600s": [0.24, 0.24, 0.24, 0.32]}
RCOL = {"15s": "#4477AA", "60s": "#66CCEE", "600s": "#CCBB44", "3600s": "#EE6677"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

for name, (y, c, mk, lw) in METHODS.items():
    axL.plot(X, y, marker=mk, color=c, lw=lw, ms=7,
             mew=(1.8 if "OURS" in name else 1.0), label=name,
             zorder=(5 if "OURS" in name else 3))
# Marengo point + strict/lenient band at 3600s
axL.plot([MARENGO[0]], [MARENGO[1][1]], marker="*", ms=15, color="#333333",
         label="Marengo 3.0 (paid, 3600s only)", zorder=6)
axL.vlines(MARENGO[0], MARENGO[1][0], MARENGO[1][1], color="#333333", lw=1.2, zorder=6)
axL.axhline(0.25, ls=":", color="#bbbbbb", lw=1)
axL.text(0.02, 0.255, "≈ chance (top-6 of many)", fontsize=7.5, color="#888888",
         transform=axL.get_yaxis_transform())
axL.set_xticks(X); axL.set_xticklabels(BINS)
axL.set_xlabel("video length bin"); axL.set_ylabel("frame hit@6  (gold frame in top-6)")
axL.set_ylim(-0.03, 1.05); axL.grid(alpha=0.18)
axL.set_title("Every selector we ran — same task, same metric, k=6", fontsize=11)
axL.legend(fontsize=7.8, loc="upper right", framealpha=0.9)

for b in BINS:
    axR.plot(KS, RECALL[b], marker="o", color=RCOL[b], lw=1.8, ms=6, label=b)
axR.set_xticks(KS); axR.set_xlabel("frame budget k"); axR.set_ylabel("hit@k (SigLIP so400m)")
axR.set_ylim(-0.03, 1.05); axR.grid(alpha=0.18)
axR.set_title("Why ours looks 'low': it is at k=6.\nLoosen budget → recall recovers (60s→1.0, 600s→.60)", fontsize=10.5)
axR.legend(fontsize=9, title="length", loc="lower right")

fig.suptitle("Frame-retrieval comparison — LongVideoBench fine-needle subset  (literature reports MCQA accuracy, not this)",
             y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig("results/method_compare.png", dpi=150, bbox_inches="tight")
print("wrote results/method_compare.png")
