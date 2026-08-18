#!/usr/bin/env python3
"""Is gold-evidence recall@k predictive of ANSWER ACCURACY? Validate before trusting it.

Motivation (2026-07-20): recall@8 was used as the direction signal that killed NEG#10
(facets) and NEG#11 (ST-OMP). Then nms_vs_omp.py showed OMP has WORSE recall than top-k
(16.5% vs 20.4%) but BETTER accuracy (.5461 vs .5106) at 3600s k8 -- the proxy pointing
the wrong way on the one pair with ground truth.

Suspected mechanism: recall@k is binary "did ANY pick land in the gold window", and top-k
concentrates ~3 picks within 2s of the score peak = more shots at the same moment. OMP
spreads out. So the proxy rewards CONCENTRATION while accuracy rewards COVERAGE, biasing
it against exactly the diversity-seeking selectors it was used to judge.

Test: 6 cells (2 bins x k in {8,16,32}) x 3 arms (uniform/topk/OMP) with known lmms-eval
accuracy. Within each cell, does recall rank the arms the way accuracy does?

Uniform caveat: uniform picks are regenerated here as linspace over the frame pool, which
approximates but may not byte-match what the eval task sampled. topk/OMP use the exact
picks files the eval consumed, so the topk-vs-OMP comparison is exact.

Usage: PYTHONPATH=. python3 scripts/validate_recall_proxy.py
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from harness.embeds import load_image_embed

# lmms-eval accuracy, CORRECT_FINDINGS "Budget curve" (bs=1, NSHARD=1, cov-gated)
ACC = {
    ("3600", 8):  {"uniform": .4716, "topk": .5106, "omp": .5461},
    ("3600", 16): {"uniform": .4770, "topk": .5461, "omp": .5798},
    ("3600", 32): {"uniform": .5142, "topk": .5727, "omp": .5851},
    ("600", 8):   {"uniform": .5534, "topk": .6141, "omp": .6311},
    ("600", 16):  {"uniform": .5850, "topk": .6505, "omp": .6578},
    ("600", 32):  {"uniform": .6044, "topk": .6481, "omp": .6650},
}
PICKS = {"topk": "results/picks_lmmseval/picks_lc_k{k}.json",
         "omp": "results/picks_lmmseval/picks_omp_lc_k{k}.json"}


def main() -> None:
    gold = {}
    for it in json.load(open("data/manifest.lvb.long976.json", encoding="utf-8")):
        g = it.get("gold_evidence_seconds")
        if isinstance(g, str):
            g = json.loads(g)
        if g:
            gold[it["id"]] = g

    pools = {}          # qid -> times array, loaded once
    rows = []
    for (b, k), accs in sorted(ACC.items()):
        ids = [i for i in np.load(f"results/embeds_text/text_lc_{b}.npz")["ids"].tolist()
               if i in gold]
        loaded = {}
        for arm in ("uniform", "topk", "omp"):
            if arm != "uniform":
                loaded[arm] = json.load(open(PICKS[arm].format(k=k)))
        hit = {a: 0 for a in accs}
        n = 0
        for qid in ids:
            if qid not in pools:
                t, _ = load_image_embed(qid, "lc", "results/embeds", "results/embeds_lc")
                pools[qid] = t
            times = pools[qid]
            if times is None or len(times) < k:
                continue
            n += 1
            win = gold[qid]
            def inwin(ts):
                return any(any(w[0] - 1.0 <= float(t) <= w[1] + 1.0 for w in win) for t in ts)
            hit["uniform"] += inwin(times[np.linspace(0, len(times) - 1, k, dtype=int)])
            for arm in ("topk", "omp"):
                p = loaded[arm].get(qid)
                if p:
                    hit[arm] += inwin(p)
        for arm in ("uniform", "topk", "omp"):
            rows.append({"bin": b, "k": k, "arm": arm,
                         "recall": hit[arm] / max(n, 1), "acc": accs[arm], "n": n})

    print(f"\n{'bin':>5} {'k':>3} {'arm':>8} {'recall@k':>9} {'accuracy':>9}")
    for r in rows:
        print(f"{r['bin']:>5} {r['k']:>3} {r['arm']:>8} {100*r['recall']:8.1f}% {r['acc']:9.4f}")

    print("\n=== WITHIN-CELL ORDERING (the decisive test) ===")
    ok = bad = 0
    for (b, k) in sorted(ACC):
        cell = [r for r in rows if r["bin"] == b and r["k"] == k]
        line = []
        for x, y in combinations(cell, 2):
            agree = (x["acc"] - y["acc"]) * (x["recall"] - y["recall"]) > 0
            ok, bad = ok + agree, bad + (not agree)
            line.append(f"{x['arm']}-v-{y['arm']}:{'OK' if agree else 'INVERTED'}")
        print(f"  {b}s k={k}: " + "  ".join(line))
    print(f"\n  pairs agreeing with accuracy: {ok}/{ok+bad}")

    a = np.array([r["acc"] for r in rows]); c = np.array([r["recall"] for r in rows])
    def spearman(u, v):
        ru = np.argsort(np.argsort(u)); rv = np.argsort(np.argsort(v))
        return float(np.corrcoef(ru, rv)[0, 1])
    print(f"  Spearman(recall, accuracy) over all {len(rows)} arms: {spearman(c, a):+.3f}")
    print(f"  Pearson  : {float(np.corrcoef(c, a)[0,1]):+.3f}")

    # within-cell only, removing the budget/bin trend that inflates any global correlation
    ds, da = [], []
    for (b, k) in sorted(ACC):
        cell = [r for r in rows if r["bin"] == b and r["k"] == k]
        for x, y in combinations(cell, 2):
            ds.append(x["recall"] - y["recall"]); da.append(x["acc"] - y["acc"])
    print(f"  Pearson on WITHIN-CELL deltas (trend removed): "
          f"{float(np.corrcoef(ds, da)[0,1]):+.3f}   <- this is what a gate relies on")
    json.dump(rows, open("results/facets/recall_proxy_validation.json", "w"), indent=2)
    print("\nwrote results/facets/recall_proxy_validation.json")


if __name__ == "__main__":
    main()
