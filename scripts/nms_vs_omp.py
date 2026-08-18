#!/usr/bin/env python3
"""Is OMP just implicit temporal NMS on top-k?

Measured motivation (2026-07-20, 3600s n=564): top-k@8 puts 3.13 of its 7 inter-pick
gaps UNDER 2 SECONDS -- it spends ~3 of 8 frames on adjacent-second near-duplicates.
OMP does this 0.30/7 times. Meanwhile the residual trace says cos_resid is dead by pick 5
while cos_orig stays flat ~.20, i.e. OMP's later picks are NOT explaining new query mass.

Hypothesis: in embedding space adjacent frames are near-identical, so orthogonalizing
against a picked frame implicitly suppresses its temporal neighbours. If so, greedy cosine
top-k with a temporal NMS gap reproduces OMP -- and the Gram-Schmidt machinery (ours, LDDR,
DPP) is an expensive way to say "don't pick the same moment twice".

Tests, free CPU:
  1. pick overlap  topk+NMS(G) vs OMP, swept over G
  2. gold-evidence recall@k for topk / topk+NMS(G) / OMP  (hostile proxy, direction only)
  3. redundancy signature (mean gap, sub-2s pairs) per method

A high overlap AND matched recall at some G is strong evidence OMP == NMS, which would be
a real negative about the whole selection literature -- better found by us than by a
reviewer. A large residual difference means OMP does something NMS cannot.

Usage: PYTHONPATH=. python3 scripts/nms_vs_omp.py --bin 3600
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_rank(query, emb, k):
    """OMP picks in RANK order (not time order)."""
    E = emb.astype(np.float32)
    q = query.astype(np.float32).copy()
    basis, chosen, sel = [], np.zeros(E.shape[0], dtype=bool), []
    for _ in range(min(k, E.shape[0])):
        s = E @ q
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        chosen[b] = True
        sel.append(b)
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            q = q - (q @ v) * v
    return sel


def topk_nms(sim, times, k, gap):
    """Greedy cosine top-k; a candidate must be >= gap seconds from every accepted pick."""
    order = np.argsort(-sim)
    sel = []
    for i in order:
        t = float(times[i])
        if all(abs(t - float(times[j])) >= gap for j in sel):
            sel.append(int(i))
            if len(sel) == k:
                break
    if len(sel) < k:                      # pool too tight for this gap -- backfill by score
        for i in order:
            if int(i) not in sel:
                sel.append(int(i))
                if len(sel) == k:
                    break
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gaps", type=float, nargs="+", default=[2, 5, 10, 30, 60])
    ap.add_argument("--tol", type=float, default=0.6, help="sec tolerance for pick matching")
    ap.add_argument("--manifest", default="data/manifest.lvb.long976.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(f"results/embeds_text/text_lc_{args.bin}.npz")
    qmap = dict(zip(z["ids"].tolist(), z["text"]))

    gold = {}
    for it in json.load(open(args.manifest, encoding="utf-8")):
        g = it.get("gold_evidence_seconds")
        if isinstance(g, str):
            g = json.loads(g)
        if g:
            gold[it["id"]] = g

    def match(a, b):
        b = list(b); n = 0
        for x in a:
            for i, y in enumerate(b):
                if abs(x - y) <= args.tol:
                    n += 1; b.pop(i); break
        return n

    ov = {g: [] for g in args.gaps}
    rec = {"topk": 0, "omp": 0, **{f"nms{g:g}": 0 for g in args.gaps}}
    gapstat = {"topk": [], "omp": [], **{f"nms{g:g}": [] for g in args.gaps}}
    adj = {kk: [] for kk in gapstat}
    ngold = used = 0

    for qid, qv in qmap.items():
        times, emb = load_image_embed(qid, "lc", "results/embeds", "results/embeds_lc")
        if times is None or emb.shape[0] < args.k:
            continue
        E = l2(emb)
        q = l2(np.asarray(qv, dtype=np.float32))
        sim = E @ q
        used += 1

        idx_omp = omp_rank(q, E, args.k)
        idx_top = np.argsort(-sim)[:args.k].tolist()
        t_omp = sorted(float(times[i]) for i in idx_omp)
        t_top = sorted(float(times[i]) for i in idx_top)

        sets = {"omp": t_omp, "topk": t_top}
        for g in args.gaps:
            sets[f"nms{g:g}"] = sorted(float(times[i]) for i in topk_nms(sim, times, args.k, g))
            ov[g].append(match(sets[f"nms{g:g}"], t_omp) / args.k)

        for kk, ts in sets.items():
            d = np.diff(ts)
            gapstat[kk].append(float(np.mean(d)) if len(d) else 0.0)
            adj[kk].append(int(np.sum(d < 2.0)))

        if qid in gold:
            ngold += 1
            win = gold[qid]
            for kk, ts in sets.items():
                if any(any(w[0] - 1.0 <= t <= w[1] + 1.0 for w in win) for t in ts):
                    rec[kk] += 1

    print(f"\n=== NMS-vs-OMP  bin={args.bin}s  n={used}  k={args.k} ===")
    print("\n1. PICK OVERLAP: topk+NMS(G) vs OMP")
    for g in args.gaps:
        print(f"   G={g:>4g}s : {100*np.mean(ov[g]):5.1f}%  "
              f"({np.mean(ov[g])*args.k:.2f}/{args.k} frames)")
    print(f"   (plain topk vs OMP baseline for reference: see redundancy table)")

    print(f"\n2. GOLD-EVIDENCE RECALL@{args.k} (hostile proxy, direction only, n={ngold})")
    order = ["topk"] + [f"nms{g:g}" for g in args.gaps] + ["omp"]
    for kk in order:
        print(f"   {kk:>8} : {100*rec[kk]/max(ngold,1):5.1f}%")

    print("\n3. REDUNDANCY SIGNATURE")
    print(f"   {'method':>8}  {'mean gap':>9}  {'sub-2s pairs':>13}")
    for kk in order:
        print(f"   {kk:>8}  {np.mean(gapstat[kk]):8.1f}s  {np.mean(adj[kk]):12.2f}/{args.k-1}")

    if args.out:
        json.dump({"bin": args.bin, "n": used, "k": args.k, "n_gold": ngold,
                   "overlap_vs_omp": {str(g): float(np.mean(ov[g])) for g in args.gaps},
                   "recall": {kk: rec[kk] / max(ngold, 1) for kk in order},
                   "mean_gap": {kk: float(np.mean(gapstat[kk])) for kk in order},
                   "sub2s": {kk: float(np.mean(adj[kk])) for kk in order}},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
