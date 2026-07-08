#!/usr/bin/env python3
"""Offline gold-in-union analyzer -- the free decision gate for Stage 1
(research/retrieve_then_ground_spec.md §4/§7), BEFORE any GPT-5.5 spend.

Reads a scripts/dump_scores.py score cache (no GPU, no torch import anywhere in this file). For
each item, builds a union of candidate frames at EQUAL frame budget B three ways:

  - peak_nms   -- ours: harness.union_retrieval.build_union_indices
  - quadrant   -- split into M contiguous chunks (same chunking region_recall.py uses), rank
                  chunks by max frame score, take frames from the best chunks until B
  - flat_top_b -- top-B frames by raw score, no temporal structure at all

...then asks: does the gold evidence land in that union? Two reads:
  - any-hit (PRIMARY):  >=1 union frame inside ANY gold span (does the VLM stage even get a shot?)
  - full-recall (secondary): EVERY gold span has >=1 union frame (matters for multi-needle items)

Per-bin rates + Wilson 95% CI (harness.metrics.wilson_ci). If peak_nms doesn't clear
quadrant/flat_top_b here, Stage 2 (the VLM) can't possibly fix it -- gate closes before any
API spend.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

from harness.metrics import wilson_ci
from harness.union_retrieval import any_hit, build_union_indices, full_recall

METHODS = ["peak_nms", "quadrant", "flat_top_b"]


def quadrant_union(scores: List[float], budget: int, M: int = 4) -> List[int]:
    """Baseline: contiguous M-chunk split (mirrors region_recall.py's `_chunk_bounds`), chunk
    score = max frame score, take frames from best-ranked chunks until we have >= budget
    candidates, then UNIFORM-subsample that pool to `budget`.

    Uniform (not chunk-prefix) is the fair form: a single chunk can exceed the budget (150-frame
    chunk vs budget 100 at 600s/M=4); taking the prefix would make gold in the chunk's tail
    structurally unreachable and cripple the baseline we're trying to beat. Uniform sampling
    spreads coverage across the whole selected region -- same convention region_recall uses."""
    n = len(scores)
    bounds = [(i * n // M, (i + 1) * n // M) for i in range(M)]
    chunk_score = [max(scores[lo:hi]) if hi > lo else float("-inf") for (lo, hi) in bounds]
    order = sorted(range(M), key=lambda c: chunk_score[c], reverse=True)
    pool: List[int] = []
    for c in order:  # add whole chunks in rank order until the pool can cover the budget
        lo, hi = bounds[c]
        pool.extend(range(lo, hi))
        if len(pool) >= budget:
            break
    pool = sorted(set(pool))
    if len(pool) <= budget:
        return pool
    # uniform-subsample the temporally-sorted pool down to exactly `budget`
    picked = {pool[round(i * (len(pool) - 1) / (budget - 1))] for i in range(budget)}
    return sorted(picked)


def flat_top_union(scores: List[float], budget: int) -> List[int]:
    """Baseline: top-B frames by raw score. No spatial/temporal structure -- the pure ranking floor."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:budget])


def rate(vals: List[float]) -> dict:
    n = len(vals)
    k = sum(vals)
    lo, hi = wilson_ci(k, n)
    return {"rate": (k / n if n else None), "n": n, "ci": [round(lo, 3), round(hi, 3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--budget", type=int, default=100, help="equal frame budget B for all 3 methods")
    ap.add_argument("--gmin", type=float, default=10.0)
    ap.add_argument("--pad", type=float, default=2.0)
    ap.add_argument("--k-std", type=float, default=1.0)
    ap.add_argument("--quadrant-m", type=int, default=4, help="M for the quadrant baseline")
    ap.add_argument("--tol", type=float, default=0.5, help="seconds tolerance for gold-in-union match (1fps quantization)")
    ap.add_argument("--out", default="results/union_ceiling.json")
    args = ap.parse_args()

    any_acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    full_acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    size_acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))  # frames used (cost)
    n_items: Dict[str, int] = defaultdict(int)

    used = 0
    with open(args.scores, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            times, scores = row["times"], row["scores"]
            spans = row.get("gold_evidence_seconds")
            b = row.get("length_bin", "all")
            if not spans or len(scores) != len(times):
                continue
            n_items[b] += 1
            used += 1

            unions = {
                "peak_nms": build_union_indices(scores, times, budget=args.budget,
                                                gmin=args.gmin, pad=args.pad, k_std=args.k_std),
                "quadrant": quadrant_union(scores, args.budget, M=args.quadrant_m),
                "flat_top_b": flat_top_union(scores, args.budget),
            }
            for method, idx in unions.items():
                u_times = [times[i] for i in idx]
                any_acc[b][method].append(1.0 if any_hit(u_times, spans, tol=args.tol) else 0.0)
                full_acc[b][method].append(1.0 if full_recall(u_times, spans, tol=args.tol) else 0.0)
                size_acc[b][method].append(len(idx))  # frames actually sent (cost story)

            print(f"[{used}] {row.get('id')} bin={b} nf={len(scores)} "
                  f"peak={len(unions['peak_nms'])} quad={len(unions['quadrant'])} flat={len(unions['flat_top_b'])}")

    result = {
        "args": vars(args),
        "n_items": dict(n_items),
        "any_hit": {b: {m: rate(v) for m, v in md.items()} for b, md in any_acc.items()},
        "full_recall": {b: {m: rate(v) for m, v in md.items()} for b, md in full_acc.items()},
        "mean_union_size": {b: {m: round(sum(v) / len(v), 1) for m, v in md.items()}
                            for b, md in size_acc.items()},
    }
    json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out} ({used} items)")

    print(f"\n== gold-in-union @ budget={args.budget} -- any-hit (PRIMARY) ==")
    for b in sorted(result["any_hit"]):
        row = " | ".join(f"{m}={result['any_hit'][b][m]['rate']:.3f} ci={result['any_hit'][b][m]['ci']}"
                          for m in METHODS if m in result["any_hit"][b])
        print(f"  bin {b} (n={n_items[b]}): {row}")

    print(f"\n== gold-in-union @ budget={args.budget} -- full-recall (secondary, multi-needle) ==")
    for b in sorted(result["full_recall"]):
        row = " | ".join(f"{m}={result['full_recall'][b][m]['rate']:.3f} ci={result['full_recall'][b][m]['ci']}"
                          for m in METHODS if m in result["full_recall"][b])
        print(f"  bin {b} (n={n_items[b]}): {row}")

    print(f"\n== mean union size (frames sent = cost; lower at equal recall = the thesis) ==")
    for b in sorted(result["mean_union_size"]):
        row = " | ".join(f"{m}={result['mean_union_size'][b][m]}"
                          for m in METHODS if m in result["mean_union_size"][b])
        print(f"  bin {b}: {row}")


if __name__ == "__main__":
    main()
