#!/usr/bin/env python3
"""LVB cosine curves — the SAME plot as QVH, from data we already have (scores.jsonl + gold).

Shows WHY LVB failed, visually: SigLIP cosine per frame vs the narrow gold needle. Overlays
top-k, our peak-NMS picks, and the gold frame(s). Contrast to QVH: QVH curve plateaus on the
moment; LVB curve is flat noise with the needle buried mid-pack. No pod, no download.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from harness.union_retrieval import build_union_indices

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gold_idx(times, spans, tol=0.5):
    return [i for i, t in enumerate(times) if any(b - tol <= t <= e + tol for b, e in spans)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--bin", default="600")
    ap.add_argument("--ids", default="", help="comma ids; else auto-pick spread by gold-rank")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--outdir", default="results/lvb_curves")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.scores) if l.strip()]
    rows = [r for r in rows if str(r["length_bin"]) == args.bin and r.get("gold_evidence_seconds")]

    # gold cosine rank per item (0 = top) to pick a spread: some hits, some deeply buried
    def grank(r):
        sc = np.array(r["scores"]); gi = gold_idx(r["times"], r["gold_evidence_seconds"])
        if not gi:
            return 10**9
        order = list(np.argsort(-sc)); return min(order.index(i) for i in gi)
    for r in rows:
        r["_grank"] = grank(r)
    if args.ids:
        want = set(args.ids.split(",")); sel = [r for r in rows if r["id"] in want]
    else:
        rows.sort(key=lambda r: r["_grank"])
        # spread: best-ranked, median, worst-ranked
        idxs = np.linspace(0, len(rows) - 1, args.n).astype(int)
        sel = [rows[i] for i in idxs]

    summ = []
    for r in sel:
        sc = np.array(r["scores"], float); tm = np.array(r["times"], float)
        spans = r["gold_evidence_seconds"]
        gi = gold_idx(r["times"], spans)
        topk = sorted(np.argsort(-sc)[: args.topk].tolist())
        peak = build_union_indices(sc.tolist(), tm.tolist(), budget=args.budget)
        grank = r["_grank"]
        gap = float(sc[gi].mean() - np.delete(sc, gi).mean()) if gi and len(gi) < len(sc) else None

        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(tm, sc, "-", color="#999", lw=0.8, zorder=1)
        ax.plot(tm, sc, ".", color="#555", ms=2.5, zorder=1, label="SigLIP cosine")
        for b, e in spans:
            ax.axvspan(b, e, color="#4caf50", alpha=0.30, zorder=0, label="gold needle")
        ax.scatter(tm[topk], sc[topk], color="#1e88e5", s=45, zorder=4, label=f"top-{args.topk}")
        ax.scatter(tm[peak], sc[peak], facecolors="none", edgecolors="#e53935", s=120, lw=1.8,
                   zorder=5, label="peak-NMS")
        if gi:
            ax.scatter(tm[gi], sc[gi], marker="*", color="#ff9800", s=180, edgecolors="k",
                       lw=0.5, zorder=6, label="gold frame")
        ax.set_xlabel("time (s)"); ax.set_ylabel("cosine")
        ax.set_title(f"{r['id']}  gold-rank {grank}/{len(sc)}  plateau_gap "
                     f"{gap:.2f}" if gap is not None else r["id"], fontsize=10)
        h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout(); out = os.path.join(args.outdir, f"{r['id']}.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        hit = "HIT" if grank < args.topk else ("union" if grank < 50 else "LOST")
        summ.append({"id": r["id"], "gold_rank": grank, "n": len(sc), "plateau_gap":
                     round(gap, 3) if gap is not None else None, "class": hit})
        print(f"{r['id']:18} gold-rank {grank:>3}/{len(sc)}  gap {gap if gap is None else round(gap,2)}  {hit}")
    json.dump(summ, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {len(sel)} curves -> {args.outdir}/")


if __name__ == "__main__":
    main()
