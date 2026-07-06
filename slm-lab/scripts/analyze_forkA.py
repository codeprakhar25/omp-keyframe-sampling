#!/usr/bin/env python3
"""Fork A analysis: full-dump vs top-k across video length.

Reads one or more `runs.jsonl` files (each row carries arm / length_bin /
video_seconds / accuracy / hit_at_k / input_tokens), aggregates per
(length_bin, arm), reports Wilson 95% CIs (n is small), and flags the
crossover bin — the first length bin where top-k accuracy >= full-dump.

Usage:
    python scripts/analyze_forkA.py results/p0_*/runs.jsonl
    python scripts/analyze_forkA.py --arm-full A --arm-topk C-so400m results/**/runs.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion (k successes of n)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(p, 3), round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


# length bins in seconds -> label; matches LongVideoBench duration groups but works for any
BIN_EDGES = [(0, 60, "0-1m"), (60, 300, "1-5m"), (300, 900, "5-15m"),
             (900, 1800, "15-30m"), (1800, 10 ** 9, "30m+")]


def bin_label(row: dict) -> str:
    lb = row.get("length_bin")
    if lb is not None:
        return str(lb)
    s = row.get("video_seconds")
    if s is None:
        return "all"
    for lo, hi, name in BIN_EDGES:
        if lo <= s < hi:
            return name
    return "all"


def load_rows(patterns: List[str]) -> List[dict]:
    rows: List[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _bin_sort_key(b: str):
    order = ["0-1m", "1-5m", "5-15m", "15-30m", "30m+"]
    return (order.index(b) if b in order else 99, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="runs.jsonl path(s) / globs")
    ap.add_argument("--arm-full", default="A", help="arm label for full-dump")
    ap.add_argument("--arm-topk", default="C", help="arm label for the top-k selector")
    ap.add_argument("--metric", default="hit_at_k", choices=["hit_at_k", "accuracy"],
                    help="primary metric (hit@k is the honest default; accuracy secondary)")
    ap.add_argument("--out", default=None, help="optional json dump of the table")
    args = ap.parse_args()

    rows = load_rows(args.runs)
    for r in rows:
        r["_bin"] = bin_label(r)
        r["_arm"] = r.get("arm") or r.get("condition")

    # group: bin -> arm -> list of metric values
    grp: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp[r["_bin"]][r["_arm"]].append(r)

    arms = sorted({r["_arm"] for r in rows})
    print(f"arms present: {arms}")
    print(f"comparing full='{args.arm_full}' vs top-k='{args.arm_topk}' on '{args.metric}'\n")

    header = f"{'bin':>7} {'arm':>12} {'n':>3} {args.metric:>9} {'95% CI':>15} {'mean_tok':>10} {'tok_red':>8}"
    print(header)
    print("-" * len(header))

    table: Dict[str, dict] = {}
    crossover: Optional[str] = None
    for b in sorted(grp, key=_bin_sort_key):
        table[b] = {}
        full_p = topk_p = None
        full_tok = None
        for arm in sorted(grp[b], key=lambda a: (a != args.arm_full, a)):
            rs = grp[b][arm]
            vals = [x[args.metric] for x in rs if x.get(args.metric) is not None]
            n = len(vals)
            k = int(round(sum(vals)))
            p, lo, hi = wilson(k, n)
            toks = [x["input_tokens"] for x in rs if x.get("input_tokens") is not None]
            mtok = round(mean(toks)) if toks else None
            table[b][arm] = {"n": n, "metric": p, "ci": [lo, hi], "mean_tokens": mtok}
            if arm == args.arm_full:
                full_p, full_tok = p, mtok
            if arm == args.arm_topk:
                topk_p = p
            red = ""
            if arm == args.arm_full:
                full_tok = mtok
            print(f"{b:>7} {arm:>12} {n:>3} {p:>9.3f} {f'[{lo:.2f},{hi:.2f}]':>15} {str(mtok):>10} {red:>8}")
        # token reduction + crossover
        if full_tok:
            for arm in grp[b]:
                mt = table[b][arm]["mean_tokens"]
                table[b][arm]["tok_reduction_vs_full"] = round(full_tok / mt, 2) if mt else None
        if full_p is not None and topk_p is not None and topk_p >= full_p and crossover is None:
            crossover = b
        print()

    print("=" * 60)
    if crossover:
        print(f"CROSSOVER: top-k '{args.arm_topk}' first >= full-dump '{args.arm_full}' "
              f"on {args.metric} at bin '{crossover}' (and cheaper).")
    else:
        print(f"NO CROSSOVER: full-dump '{args.arm_full}' stays >= top-k on {args.metric} "
              f"in all bins. (Valid negative: compression buys cost, not accuracy.)")
    print("Reminder: n is small -> read the Wilson CIs, not point estimates.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"table": table, "crossover": crossover, "metric": args.metric}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
