#!/usr/bin/env python3
"""Curve panels for the option-posterior kill-test: WHERE did greedy-KL drift from top-k?

Joins results/scores/scores.jsonl (cached SigLIP question-cosine curves + gold spans, 1:1
frame alignment verified) with results/posterior_killtest.json (greedy-KL vs top-k indices)
and the 600-bin MCQA arms (topk vs pins predictions). Default panel set = questions where
top-k frames let GPT answer right but greedy-KL (pins) frames did not — the drift cases —
plus a couple where both were right, for contrast.

Style follows scripts/lvb_curve.py: one neutral curve, gold spans shaded, marker series for
the two selectors (color+shape both differ — not color-alone).
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVE = "#555555"      # neutral ink, single series
GOLD = "#E8B339"       # amber shading for gold spans
TOPK = "#4477AA"       # blue circles
GREEDY = "#CC3311"     # vermillion X — CVD-safe against the blue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--killtest", default="results/posterior_killtest.json")
    ap.add_argument("--pins", default="results/mcqa_600_pins.json")
    ap.add_argument("--topk", default="results/mcqa_600_topk.json")
    ap.add_argument("--n-drift", type=int, default=6)
    ap.add_argument("--n-agree", type=int, default=2)
    ap.add_argument("--out", default="results/killtest_curves.png")
    args = ap.parse_args()

    curves = {json.loads(l)["id"]: json.loads(l) for l in open(args.scores)}
    kt = {r["id"]: r for r in json.load(open(args.killtest))["records"]}
    pins = {r["id"]: r for r in json.load(open(args.pins))["records"]}
    topk = {r["id"]: r for r in json.load(open(args.topk))["records"]}

    both = [i for i in pins if i in topk and i in kt and i in curves]
    drift = [i for i in both if topk[i]["ok"] and not pins[i]["ok"]][: args.n_drift]
    agree = [i for i in both if topk[i]["ok"] and pins[i]["ok"]][: args.n_agree]
    ids = drift + agree
    if not ids:
        raise SystemExit("no joinable cases")

    ncol = 2
    nrow = (len(ids) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow), squeeze=False)

    for ax, iid in zip(axes.flat, ids):
        c, k = curves[iid], kt[iid]
        t = np.asarray(c["times"])
        s = np.asarray(c["scores"])
        for lo, hi in c.get("gold_evidence_seconds") or []:
            ax.axvspan(lo, hi, color=GOLD, alpha=0.30, lw=0)
        ax.plot(t, s, "-", color=CURVE, lw=1.0)
        ti, gi = k["topk_indices"], k["greedy_indices"]
        ax.plot(t[ti], s[ti], "o", ms=6, mfc="none", mec=TOPK, mew=1.6)
        ax.plot(t[gi], s[gi], "x", ms=7, color=GREEDY, mew=1.8)
        verdict = f"topk {topk[iid]['pred']}{'✓' if topk[iid]['ok'] else '✗'}  " \
                  f"greedy {pins[iid]['pred']}{'✓' if pins[iid]['ok'] else '✗'}  gold {k['gold_letter']}"
        ax.set_title(f"{iid}   {verdict}   overlap {k['overlap']:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15)
    for ax in axes.flat[len(ids):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color=CURVE, lw=1.0, label="SigLIP question cosine"),
        plt.Rectangle((0, 0), 1, 1, color=GOLD, alpha=0.30, label="gold span"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=TOPK, mew=1.6, ms=6, label="top-k @8"),
        plt.Line2D([], [], marker="x", ls="", color=GREEDY, mew=1.8, ms=7, label="greedy-KL @8"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("600-bin drift cases: top-k right, greedy-KL (pins) wrong — plus two agree rows",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} ({len(drift)} drift + {len(agree)} agree panels)")


if __name__ == "__main__":
    main()
