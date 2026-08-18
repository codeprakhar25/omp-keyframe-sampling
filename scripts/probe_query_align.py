#!/usr/bin/env python3
"""Feasibility probe: can a LEARNED map close the CLIP modality gap for evidence selection?

Question (2026-07-21): every axis we closed points at the CLIP dual-encoder gap -- text
query ~97% orthogonal to the image subspace, cos_orig maxes ~.233, OMP reaches ~3.3% of
query norm. Before spending a GPU or rewriting the train-free thesis, settle ONE thing on
data we already have: does a shallow learned map g(q_text) land the query CLOSER to the
gold-evidence frames on HELD-OUT videos?

If a cross-validated linear map cannot lift held-out cosine above the raw gap, training is
dead -- a clean negative ("the gap is not closable by a shallow map on this supervision").
If it lifts recall meaningfully, the scope question reopens with eyes open.

Design (no leakage):
  x_qid = L2 LongCLIP text-query embed
  y_qid = L2( mean of gold-window frame image embeds )        (the target the query should reach)
  Ridge  W = (X'X + lambda I)^-1 X'Y   fit on TRAIN qids only, per fold
  5-fold CV over qids. Long bins only (600 + 3600) -- the real game.

Held-out metrics, per fold, averaged:
  cos_raw   = cos(x, y)                         <- the modality gap, our measured ~.20
  cos_proj  = cos(Wx, y)                        <- does the map close it?
  recall@k  raw : rank pool by cos(x, frame)    -> top-k hit a gold window?
  recall@k  proj: rank pool by cos(Wx, frame)   <- does the map improve FRAME RANKING?
  recall@k  omp : OMP(x) reference (train-free, what we already ship)

lambda swept; the curve is reported so a single lucky value can't masquerade as signal.

CPU, minutes. Usage:
  PYTHONPATH=. python3 scripts/probe_query_align.py --bins 600 3600 --k 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_rank(query, E, k):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", nargs="+", default=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--manifest", default="data/manifest.lvb.long976.json")
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--out", default="results/facets/probe_query_align.json")
    args = ap.parse_args()

    gold = {}
    for it in json.load(open(args.manifest, encoding="utf-8")):
        g = it.get("gold_evidence_seconds")
        if isinstance(g, str):
            g = json.loads(g)
        if g:
            gold[it["id"]] = g

    # Build the supervised set: qid needs a text embed, a frame pool, and >=1 gold frame.
    X, Y, qids, pools = [], [], [], {}
    for b in args.bins:
        z = np.load(f"{args.text_dir}/text_lc_{b}.npz")
        qmap = dict(zip(z["ids"].tolist(), z["text"]))
        for qid, qv in qmap.items():
            if qid not in gold:
                continue
            times, emb = load_image_embed(qid, "lc", args.embeds_dir, args.embeds_lc_dir)
            if times is None or emb.shape[0] < args.k:
                continue
            E = l2(emb)
            win = gold[qid]
            mask = np.array([any(w[0] - args.tol <= float(t) <= w[1] + args.tol for w in win)
                             for t in times])
            if not mask.any():
                continue
            y = E[mask].mean(0)
            ny = float(np.linalg.norm(y))
            if ny < 1e-6:
                continue
            x = l2(np.asarray(qv, dtype=np.float32))
            X.append(x); Y.append(y / ny); qids.append(qid)
            pools[qid] = (np.asarray(times, dtype=np.float32), E, win)

    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
    n, d = X.shape
    print(f"\nsupervised set: n={n} qids  d={d}  bins={args.bins}  k={args.k}")
    if n < 2 * args.folds:
        print("too few gold videos to cross-validate."); return

    def inwin(qid, idx):
        times, _, win = pools[qid]
        return any(any(w[0] - args.tol <= float(times[i]) <= w[1] + args.tol for w in win)
                   for i in idx)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, args.folds)

    # OMP + raw recall don't depend on lambda -- compute once.
    rec_raw = rec_omp = cos_raw = 0.0
    for i in range(n):
        qid = qids[i]
        times, E, win = pools[qid]
        x = X[i]
        cos_raw += float(x @ Y[i])
        sim = E @ x
        rec_raw += inwin(qid, np.argsort(-sim)[:args.k])
        rec_omp += inwin(qid, omp_rank(x, E, args.k))
    cos_raw /= n; rec_raw /= n; rec_omp /= n

    print(f"\n{'lambda':>8} {'cos_raw':>8} {'cos_proj':>9} {'d_cos':>7} "
          f"{'rec_raw':>8} {'rec_proj':>9} {'rec_omp':>8}")
    results = []
    for lam in args.lambdas:
        cos_proj = rec_proj = 0.0
        for f in range(args.folds):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
            Xtr, Ytr = X[tr], Y[tr]
            W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32),
                                Xtr.T @ Ytr)                       # (d,d)
            for i in te:
                qid = qids[i]
                times, E, win = pools[qid]
                xp = X[i] @ W
                npn = float(np.linalg.norm(xp))
                if npn < 1e-8:
                    continue
                xp = xp / npn
                cos_proj += float(xp @ Y[i])
                rec_proj += inwin(qid, np.argsort(-(E @ xp))[:args.k])
        cos_proj /= n; rec_proj /= n
        print(f"{lam:>8g} {cos_raw:>8.4f} {cos_proj:>9.4f} {cos_proj-cos_raw:>+7.4f} "
              f"{rec_raw:>8.4f} {rec_proj:>9.4f} {rec_omp:>8.4f}")
        results.append({"lambda": lam, "cos_raw": cos_raw, "cos_proj": cos_proj,
                        "rec_raw": rec_raw, "rec_proj": rec_proj, "rec_omp": rec_omp})

    best = max(results, key=lambda r: r["rec_proj"])
    print(f"\nbest held-out: lambda={best['lambda']:g}  "
          f"cos {cos_raw:.4f}->{best['cos_proj']:.4f} ({best['cos_proj']-cos_raw:+.4f})  "
          f"recall {rec_raw:.4f}->{best['rec_proj']:.4f} ({best['rec_proj']-rec_raw:+.4f})  "
          f"[OMP {rec_omp:.4f}]")
    verdict = ("LIFTS -- reopen scope" if best["rec_proj"] > rec_raw + 0.02
               and best["cos_proj"] > cos_raw + 0.02
               else "FLAT/DEAD -- gap not closable by a shallow map on this supervision")
    print(f"VERDICT: {verdict}")

    json.dump({"n": n, "d": d, "bins": args.bins, "k": args.k,
               "cos_raw": cos_raw, "rec_raw": rec_raw, "rec_omp": rec_omp,
               "curve": results, "best": best, "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
