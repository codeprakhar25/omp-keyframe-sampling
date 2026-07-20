#!/usr/bin/env python3
"""Path 3 stage D: multi-facet matching-pursuit picks + the pick-overlap kill-switch.

Selection: round-robin OMP over the facet set. At each step the facet with the largest
remaining relative residual gets to choose the next frame; the chosen direction is then
orthogonalized out of EVERY facet residual (one shared basis), so a frame that satisfies
two facets at once is not paid for twice. Falls back to plain stem OMP when a question has
no usable facet -- so the arm is never advantaged by silently dropping hard questions.

Emits lmms-eval picks ({qid: [secs]}) and reports overlap against the existing stem-OMP
picks. Overlap > --kill-overlap means the picks did not move, so the answerer cannot move
either -> stop, spend no GPU. Reference scale: the 2026-07-15 query-bug fix moved 42% of
picked frames and did move accuracy.

Grounding filter mirrors stage C: an ungrounded facet moves picks (and so would PASS this
kill-switch) while pointing at random frames. Dropped before selection, counted in output.

Optional --gold-recall scores hit@k against manifest gold_evidence_seconds.
That metric is a HOSTILE proxy -- a near-duplicate frame just outside the window counts as
a miss though it carries the same evidence -- so it is reported as a direction only. It is
never the accept gate and must not appear as a paper number.

CPU, seconds. Usage:
  PYTHONPATH=. python3 scripts/gen_facet_picks.py --bin 3600 --k 8 \
      --facet-embeds results/facets/facet_embeds.npz \
      --ref-picks results/picks_lmmseval/picks_omp_lc_k8_3600.json \
      --manifest data/manifest.lvb.long976.json \
      --out results/picks_lmmseval/picks_facet_omp_lc_3600_k8.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from harness.embeds import l2, load_image_embed


def facet_omp(qs, E, k):
    """Round-robin OMP over m facet queries against a shared orthonormal basis."""
    R = [q.astype(np.float32).copy() for q in qs]
    n0 = [float(np.linalg.norm(q)) or 1.0 for q in qs]
    basis, chosen = [], np.zeros(E.shape[0], dtype=bool)
    order, owner = [], []
    for _ in range(min(k, E.shape[0])):
        j = int(np.argmax([np.linalg.norm(r) / n for r, n in zip(R, n0)]))
        s = E @ R[j]
        s[chosen] = -np.inf
        b = int(np.argmax(s))
        if not np.isfinite(s[b]):
            break
        chosen[b] = True
        order.append(b)
        owner.append(j)
        v = E[b].copy()
        for u in basis:
            v -= (v @ u) * u
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            v /= nv
            basis.append(v)
            R = [r - (r @ v) * v for r in R]
    return order, owner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, choices=["600", "3600"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--facet-embeds", default="results/facets/facet_embeds.npz")
    ap.add_argument("--min-grounding", type=float, default=0.5)
    ap.add_argument("--ref-picks", default=None, help="stem-OMP picks for the overlap gate")
    ap.add_argument("--kill-overlap", type=float, default=0.90)
    ap.add_argument("--manifest", default=None, help="enables the gold-evidence recall proxy")
    ap.add_argument("--tol", type=float, default=1.0, help="sec tolerance for the recall proxy")
    ap.add_argument("--scorer", default="lc")
    ap.add_argument("--embeds-dir", default="results/embeds")
    ap.add_argument("--embeds-lc-dir", default="results/embeds_lc")
    ap.add_argument("--text-dir", default="results/embeds_text")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # facet_embeds.npz spans BOTH bins -- without this filter every --bin emitted all 976
    # qids (same bug as facet_diagnostic.py, caught 2026-07-20). The lmms-eval pregate
    # expects exactly 412 / 564, so an unfiltered picks file would fail there anyway.
    binids = set(np.load(f"{args.text_dir}/text_lc_{args.bin}.npz")["ids"].tolist())

    z = np.load(args.facet_embeds, allow_pickle=True)
    ids, slots, vec, gnd = z["ids"], z["slots"], z["vec"].astype(np.float32), z["grounding"]
    stem_of, facets_of = {}, defaultdict(list)
    for i in range(len(ids)):
        if ids[i] not in binids:
            continue
        if slots[i] == -1:
            stem_of[ids[i]] = vec[i]
        elif gnd[i] >= args.min_grounding:
            facets_of[ids[i]].append(vec[i])

    gold = {}
    if args.manifest:
        for it in json.load(open(args.manifest, encoding="utf-8")):
            g = it.get("gold_evidence_seconds")
            if isinstance(g, str):
                g = json.loads(g)
            if g:
                gold[it["id"]] = g

    ref = json.load(open(args.ref_picks)) if args.ref_picks else {}

    picks, kdist = {}, defaultdict(int)
    ov, nfallback, used = [], 0, 0
    hit_f, hit_r, ngold = 0, 0, 0
    owner_share = defaultdict(int)

    for qid, q in stem_of.items():
        times, emb = load_image_embed(qid, args.scorer, args.embeds_dir, args.embeds_lc_dir)
        if times is None or emb.shape[0] < 1:
            continue
        E = l2(emb)
        F = [l2(np.asarray(f, dtype=np.float32)) for f in facets_of.get(qid, [])]
        if not F:
            F = [l2(np.asarray(q, dtype=np.float32))]
            nfallback += 1
        used += 1
        kdist[len(F)] += 1
        order, owner = facet_omp(F, E, args.k)
        for o in owner:
            owner_share[o] += 1
        secs = sorted(round(float(times[i]), 1) for i in order)
        picks[qid] = secs

        if qid in ref:
            a, b = set(secs), set(round(float(x), 1) for x in ref[qid])
            ov.append(len(a & b) / max(len(b), 1))

        if qid in gold:
            ngold += 1
            win = gold[qid]
            def hit(ss):
                return any(any(w[0] - args.tol <= s <= w[1] + args.tol for w in win) for s in ss)
            hit_f += hit(secs)
            if qid in ref:
                hit_r += hit([round(float(x), 1) for x in ref[qid]])

    print(f"\nbin={args.bin}s  k={args.k}  n={used}  "
          f"stem-fallback (no grounded facet): {nfallback}")
    print("facet-count dist: " + "  ".join(f"m{k}:{kdist[k]}" for k in sorted(kdist)))
    tot = sum(owner_share.values()) or 1
    print("budget share by facet slot: " +
          "  ".join(f"f{j}:{100*owner_share[j]/tot:.0f}%" for j in sorted(owner_share)))

    if ov:
        mo = float(np.mean(ov))
        print(f"\nPICK OVERLAP vs stem-OMP: {100*mo:.1f}%  (moved {100*(1-mo):.1f}% of frames)")
        print(f"  identical pick sets: {100*np.mean([o == 1.0 for o in ov]):.1f}% of videos")
        if mo > args.kill_overlap:
            print(f"  !! KILL-SWITCH: overlap > {100*args.kill_overlap:.0f}% -- picks did not "
                  "move, answerer cannot move. Do NOT spend GPU.")
        else:
            print(f"  picks moved enough to be worth testing "
                  f"(query-bug fix reference: 42% moved).")

    if ngold and ref:
        print(f"\ngold-evidence recall@{args.k} (HOSTILE proxy, direction only, n={ngold}):")
        print(f"  stem-OMP  {100*hit_r/ngold:.1f}%   facet-OMP {100*hit_f/ngold:.1f}%   "
              f"delta {100*(hit_f-hit_r)/ngold:+.1f}pt")

    if args.out:
        json.dump(picks, open(args.out, "w"))
        print(f"\nwrote {args.out}  ({len(picks)} qids)")


if __name__ == "__main__":
    main()
