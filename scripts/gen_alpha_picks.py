#!/usr/bin/env python3
"""Partial-deflation OMP picks (Idea-1): q <- q - alpha*(q.v)v per pick.

alpha=1.0 == shipped harness.replay_selectors.omp_indices (verified byte-faithful in the
2026-07-17 M0 triage, 412/412 @600s + 564/564 @3600s). alpha=0.0 == flat top-k. Interior
alpha (0.5, 0.75) is the ablation arm: does softening the residual deflation recover the
top-k rescue set without giving up OMP-only wins?

Same inputs/format as scripts/gen_omp_picks.py: STEM-ONLY text embeds, cached LongCLIP image
embeds via harness.embeds, scores.jsonl for the frame-count check + id filter. Emits the
lmms-eval injection format {qid: sorted([seconds])} so it drops straight into
$LVB_PICKS_OMP_LC_K8. CPU, seconds. Refuses on any unusable item unless --allow-skips.
"""
import argparse
import json
import os

import numpy as np

from harness.embeds import l2, load_image_embed
from harness.replay_selectors import omp_indices


def omp_indices_alpha(query: np.ndarray, emb: np.ndarray, k: int, alpha: float):
    """Faithful copy of replay_selectors.omp_indices with the deflation scaled by alpha.
    Only line changed vs shipped: q = q - alpha*(q@v)*v  (shipped alpha=1)."""
    n = emb.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    q = query.astype(np.float32).copy()
    E = emb.astype(np.float32)
    basis = []
    selected = []
    chosen = np.zeros(n, dtype=bool)
    for _ in range(k):
        s = E @ q
        s[chosen] = -np.inf
        best = int(np.argmax(s))
        selected.append(best)
        chosen[best] = True
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v /= norm
            basis.append(v)
            q = q - alpha * (q @ v) * v
    return sorted(selected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scorer", required=True, choices=["sig", "lc"])
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--text-embeds", required=True, help="STEM-ONLY text embeds npz (ids,text)")
    ap.add_argument("--bin", default=None)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-skips", action="store_true")
    args = ap.parse_args()

    t = np.load(args.text_embeds)
    qmap = dict(zip(t["ids"].tolist(), t["text"]))

    rows = {}
    with open(args.scores) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.bin and str(r["length_bin"]).rstrip("s") != str(args.bin).rstrip("s"):
                continue
            rows[r["id"]] = r
    print(f"scores rows in scope: {len(rows)}  (bin={args.bin or 'ALL'})  alpha={args.alpha}")

    picks, reasons = {}, {"no_text_embed": [], "no_image_embed": [], "frame_mismatch": []}
    faithful_hits = faithful_tot = 0
    for rid, r in rows.items():
        if rid not in qmap:
            reasons["no_text_embed"].append(rid)
            continue
        times, emb = load_image_embed(rid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None:
            reasons["no_image_embed"].append(rid)
            continue
        E = l2(emb)
        if len(r["scores"]) != E.shape[0]:
            reasons["frame_mismatch"].append(rid)
            continue
        q0 = l2(qmap[rid].astype(np.float32))
        idx = omp_indices_alpha(q0, E, args.k, args.alpha)
        # in-process faithfulness gate: alpha=1 MUST equal shipped omp_indices
        if abs(args.alpha - 1.0) < 1e-12:
            faithful_tot += 1
            if idx == omp_indices(q0, E, args.k):
                faithful_hits += 1
        picks[rid] = {"idx": [int(i) for i in idx],
                      "secs": [round(float(times[i]), 1) for i in idx]}

    if faithful_tot:
        print(f"alpha=1 faithfulness vs shipped omp_indices: {faithful_hits}/{faithful_tot}")
        if faithful_hits != faithful_tot:
            raise SystemExit("REFUSING: alpha=1 diverged from shipped omp_indices — bug in copy.")

    skipped = sum(len(v) for v in reasons.values())
    if skipped:
        print(f"!! {skipped} items unusable:")
        for why, ids in reasons.items():
            if ids:
                print(f"   {why:16s} {len(ids):4d}  e.g. {ids[:3]}")
        if not args.allow_skips:
            raise SystemExit("REFUSING incomplete picks (fails inside lmms-eval AFTER model load).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({q: sorted(v["secs"]) for q, v in picks.items()}, open(args.out, "w"))
    side = args.out.replace(".json", ".meta.json")
    json.dump({"method": f"omp_alpha{args.alpha}_k{args.k}", "scorer": args.scorer,
               "bin": args.bin, "k": args.k, "alpha": args.alpha, "n": len(picks),
               "picks": picks}, open(side, "w"))
    print(f"wrote {args.out}: {len(picks)} items ({skipped} skipped)  [+ {side}]")


if __name__ == "__main__":
    main()
