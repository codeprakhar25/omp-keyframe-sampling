#!/usr/bin/env python3
"""ST-OMP (Simultaneous Temporal OMP) — free CPU pre-registration tests.

Proposal: replace the pick math with S-OMP over a query-gated velocity matrix.
  v_t = f_{t+1} - f_t                      (temporal velocity)
  w_t = max(0, v_t . q)                    (query gate)
  Y_t = w_t * v_t                          (target trajectory matrix)
  S_i = sum_t (R_t . f_i)^2                (score), Gram-Schmidt out of ALL rows

Three tests, all free, run BEFORE any GPU:

(a) GATE SIGNAL-vs-NOISE. v_t.q = (f_{t+1}.q) - (f_t.q) is a difference of two small
    similar image-text cosines (measured: those max at .233). If that difference is
    noise, the query enters the algorithm only through noise and ST-OMP degenerates
    into a query-independent motion detector. Compared against a null: the same
    statistic computed with a SHUFFLED (mismatched) query.

(b) QUERY-SWAP. Run ST-OMP on a video with its own query vs another video's query.
    A genuinely query-conditioned selector must move its picks. If overlap is high,
    ST-OMP is a shot-boundary detector wearing OMP notation. Plain cosine top-k is
    included as the positive control (it MUST move).

(c) GOLD-EVIDENCE RECALL vs stem-OMP, the same hostile-but-symmetric proxy that
    killed Path 3. Direction only, never a paper number.

Efficiency note: the proposed `R @ E.T` is O(T^2 d) (the source claimed O(KTd),
which is wrong). Identical scores come from S_i = f_i^T (R^T R) f_i, i.e. O(T d^2)
and no T x T intermediate -- that is what is implemented here.

Usage: PYTHONPATH=. python3 scripts/stomp_diagnostic.py --bin 3600
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def st_omp(query, emb, k):
    """ST-OMP picks. Scores via Gram matrix (O(T d^2)), not the O(T^2 d) form."""
    T = emb.shape[0]
    if T == 0:
        return []
    if k >= T:
        return list(range(T))
    E = emb.astype(np.float32)
    vel = E[1:] - E[:-1]
    w = np.maximum(0.0, vel @ query)
    Y = vel * w[:, None]
    if not np.any(w > 0) or np.allclose(Y, 0):
        return [int(x) for x in np.linspace(0, T - 1, k, dtype=int)]   # spec's fallback
    R = Y.copy()
    basis, chosen, sel = [], np.zeros(T, dtype=bool), []
    for _ in range(k):
        G = R.T @ R                              # (d,d)
        scores = np.einsum("ij,jk,ik->i", E, G, E)   # f_i^T G f_i  == sum_t (R_t.f_i)^2
        scores[chosen] = -np.inf
        b = int(np.argmax(scores))
        if not np.isfinite(scores[b]):
            break
        sel.append(b)
        chosen[b] = True
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv <= 1e-6:
            break
        v /= nv
        basis.append(v)
        R -= (R @ v)[:, None] * v
    return sorted(sel)


def omp(query, emb, k):
    """Stem OMP baseline (harness/replay_selectors.py math)."""
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
    return sorted(sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="data/manifest.lvb.long976.json")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(f"results/embeds_text/text_lc_{args.bin}.npz")
    qmap = dict(zip(z["ids"].tolist(), z["text"]))
    qids = list(qmap)
    if args.limit:
        qids = qids[:args.limit]

    gold = {}
    for it in json.load(open(args.manifest, encoding="utf-8")):
        g = it.get("gold_evidence_seconds")
        if isinstance(g, str):
            g = json.loads(g)
        if g:
            gold[it["id"]] = g

    gate_true, gate_null, cos_frame = [], [], []
    ov_st, ov_cos = [], []
    hit_st = hit_omp = ngold = 0
    used = 0
    rng = np.random.default_rng(0)

    for n, qid in enumerate(qids):
        times, emb = load_image_embed(qid, "lc", args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < args.k + 2:
            continue
        E = l2(emb)
        q = l2(np.asarray(qmap[qid], dtype=np.float32))
        other = qids[(n + len(qids) // 2) % len(qids)]          # mismatched query
        qo = l2(np.asarray(qmap[other], dtype=np.float32))
        used += 1

        # (a) gate statistic vs its mismatched-query null
        vel = E[1:] - E[:-1]
        gate_true.append(np.abs(vel @ q).mean())
        gate_null.append(np.abs(vel @ qo).mean())
        cos_frame.append(np.abs(E @ q).mean())

        # (b) query-swap: ST-OMP vs cosine top-k positive control
        a, b = st_omp(q, E, args.k), st_omp(qo, E, args.k)
        ov_st.append(len(set(a) & set(b)) / args.k)
        ca = set(np.argsort(-(E @ q))[:args.k].tolist())
        cb = set(np.argsort(-(E @ qo))[:args.k].tolist())
        ov_cos.append(len(ca & cb) / args.k)

        # (c) gold-evidence recall vs stem OMP
        if qid in gold:
            ngold += 1
            win = gold[qid]
            def hit(idx):
                return any(any(w[0] - 1.0 <= float(times[i]) <= w[1] + 1.0 for w in win)
                           for i in idx)
            hit_st += hit(a)
            hit_omp += hit(omp(q, E, args.k))

        if used % 50 == 0:
            print(f"  {used} videos...")

    gt, gn, cf = np.mean(gate_true), np.mean(gate_null), np.mean(cos_frame)
    print(f"\n=== ST-OMP diagnostics  bin={args.bin}s  n={used}  k={args.k} ===")
    print("\n(a) GATE SIGNAL vs NOISE   mean |v_t . q|")
    print(f"    true query      : {gt:.5f}")
    print(f"    mismatched query: {gn:.5f}   (null)")
    print(f"    ratio true/null : {gt/gn:.3f}   (1.00 = pure noise, query irrelevant)")
    print(f"    per-frame |f.q| : {cf:.5f}   (gate is {100*gt/cf:.1f}% the size of the "
          "raw frame-query signal it is built from)")

    print("\n(b) QUERY-SWAP  pick overlap own-query vs another video's query")
    print(f"    ST-OMP        : {100*np.mean(ov_st):.1f}%  "
          f"(identical picks on {100*np.mean([o == 1.0 for o in ov_st]):.1f}% of videos)")
    print(f"    cosine top-k  : {100*np.mean(ov_cos):.1f}%   <- positive control, must be LOW")

    if ngold:
        print(f"\n(c) GOLD-EVIDENCE RECALL@{args.k} (hostile proxy, direction only, n={ngold})")
        print(f"    stem-OMP : {100*hit_omp/ngold:.1f}%")
        print(f"    ST-OMP   : {100*hit_st/ngold:.1f}%   delta "
              f"{100*(hit_st-hit_omp)/ngold:+.1f}pt")

    if args.out:
        json.dump({"bin": args.bin, "n": used, "k": args.k,
                   "gate_true": float(gt), "gate_null": float(gn),
                   "gate_ratio": float(gt / gn), "cos_frame": float(cf),
                   "swap_overlap_stomp": float(np.mean(ov_st)),
                   "swap_overlap_cosine": float(np.mean(ov_cos)),
                   "recall_stomp": hit_st / max(ngold, 1),
                   "recall_omp": hit_omp / max(ngold, 1), "n_gold": ngold},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
