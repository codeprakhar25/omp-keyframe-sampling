#!/usr/bin/env python3
"""OMP residual-query trace, k=8 -> k=16 transition (CPU, cached embeds only).

For each video in a bin, replay textbook OMP (harness.replay_selectors.omp_indices
math, inlined so we can instrument every step) out to k=16 and record, per pick t:

  resid_frac[t] = ||q_t|| / ||q_0||      residual query norm AFTER removing t comps
  drained[t]    = q_{t-1}.v_t            query mass this pick explains (v orthonormal)
  cos_resid[t]  = E[best].q_{t-1}/||q_{t-1}||   relevance to the RESIDUAL (what OMP maxes)
  cos_orig[t]   = E[best].q_0            relevance of pick t to the ORIGINAL query

Question the trace answers: as OMP goes 8->16, are picks 9-16 still explaining
real query signal, or is the residual already drained (picks become near-orthogonal
diversity)?  That is the mechanism behind the k16 long-bin saturation.

Usage:
  PYTHONPATH=. python3 scripts/omp_residual_trace.py --bin 3600
  PYTHONPATH=. python3 scripts/omp_residual_trace.py --bin 600 --examples 5
"""
import argparse
import json

import numpy as np

from harness.embeds import l2, load_image_embed


def omp_trace(query, emb, k):
    """OMP to k, returning per-step (idx, resid_frac, drained, cos_resid, cos_orig)."""
    q0 = query.astype(np.float32).copy()
    q0n = float(np.linalg.norm(q0))
    q = q0.copy()
    E = emb.astype(np.float32)
    n = E.shape[0]
    basis, chosen = [], np.zeros(n, dtype=bool)
    steps = []
    for _ in range(min(k, n)):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        qn = float(np.linalg.norm(q))
        cos_resid = float(s[best] / qn) if qn > 1e-9 else 0.0
        cos_orig = float(E[best] @ q0 / q0n) if q0n > 1e-9 else 0.0
        chosen[best] = True
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        norm = float(np.linalg.norm(v))
        drained = 0.0
        if norm > 1e-6:
            v /= norm
            drained = float(q @ v)          # component of residual removed this step
            basis.append(v)
            q = q - (q @ v) * v
        resid_frac = float(np.linalg.norm(q) / q0n) if q0n > 1e-9 else 0.0
        steps.append((best, resid_frac, abs(drained), cos_resid, cos_orig))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--scorer", default="lc")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t = np.load(f"results/embeds_text/text_lc_{args.bin}.npz")
    qmap = dict(zip(t["ids"].tolist(), t["text"]))

    K = args.k
    # per-step accumulators
    RF = [[] for _ in range(K)]   # resid_frac
    DR = [[] for _ in range(K)]   # drained
    CR = [[] for _ in range(K)]   # cos_resid
    CO = [[] for _ in range(K)]   # cos_orig
    resid_after8, resid_after16 = [], []
    used, skipped = 0, 0
    examples = []

    for qid, qv in qmap.items():
        times, emb = load_image_embed(qid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < K:
            skipped += 1
            continue
        E = l2(emb)
        q = l2(np.asarray(qv, dtype=np.float32))
        steps = omp_trace(q, E, K)
        if len(steps) < K:
            skipped += 1
            continue
        used += 1
        for i, (_, rf, dr, cr, co) in enumerate(steps):
            RF[i].append(rf); DR[i].append(dr); CR[i].append(cr); CO[i].append(co)
        resid_after8.append(steps[7][1])
        resid_after16.append(steps[15][1])
        if len(examples) < args.examples:
            examples.append((qid, [(round(s[1], 3), round(s[2], 3), round(s[3], 3))
                                   for s in steps]))

    def col(a, i):
        v = np.array(a[i])
        return float(v.mean()), float(np.median(v))

    print(f"\nbin={args.bin}s  scorer={args.scorer}  n_used={used}  skipped={skipped}\n")
    print("step  resid_frac(mean/med)   drained(mean)  cos_resid(mean)  cos_orig(mean)")
    for i in range(K):
        rfm, rfd = col(RF, i)
        print(f"{i+1:>4}   {rfm:6.3f} / {rfd:6.3f}        {np.mean(DR[i]):6.3f}"
              f"         {np.mean(CR[i]):6.3f}          {np.mean(CO[i]):6.3f}")

    r8 = np.array(resid_after8); r16 = np.array(resid_after16)
    print(f"\nresidual query norm remaining:")
    print(f"  after  8 picks: mean {r8.mean():.3f}  median {np.median(r8):.3f}  "
          f"(explained {100*(1-r8.mean()):.1f}% of query mass)")
    print(f"  after 16 picks: mean {r16.mean():.3f}  median {np.median(r16):.3f}  "
          f"(explained {100*(1-r16.mean()):.1f}% of query mass)")
    print(f"  picks 9-16 drain an extra {100*(r8.mean()-r16.mean()):.1f}pt of query mass")

    # relevance decay: how much does per-pick relevance-to-residual fall 8->16
    cr8 = np.mean(CR[7]); cr16 = np.mean(CR[15]); cr1 = np.mean(CR[0])
    print(f"\ncos_resid decay: pick1 {cr1:.3f} -> pick8 {cr8:.3f} -> pick16 {cr16:.3f}")
    co8 = np.mean(CO[7]); co16 = np.mean(CO[15])
    print(f"cos_orig  decay: pick8 {co8:.3f} -> pick16 {co16:.3f}  "
          f"(picks 9-16 relevance to ORIGINAL query)")

    print("\nexample per-video traces (resid_frac, drained, cos_resid) steps 1..K:")
    for qid, tr in examples:
        print(f"  {qid}")
        print("    " + "  ".join(f"{a}" for a in tr))

    if args.out:
        json.dump({"bin": args.bin, "k": K, "n": used,
                   "resid_frac_mean": [float(np.mean(RF[i])) for i in range(K)],
                   "drained_mean": [float(np.mean(DR[i])) for i in range(K)],
                   "cos_resid_mean": [float(np.mean(CR[i])) for i in range(K)],
                   "cos_orig_mean": [float(np.mean(CO[i])) for i in range(K)],
                   "resid_after8_mean": float(r8.mean()),
                   "resid_after16_mean": float(r16.mean())},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
