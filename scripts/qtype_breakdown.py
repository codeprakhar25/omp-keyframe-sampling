#!/usr/bin/env python3
"""Does frame selection help temporal questions? (LongVideoBench, per question_category)

Recomputes the qtype interaction from the banked per-item files rather than trusting
the 2026-07-21 log-analysis note. Splits LongVideoBench's 17 question categories into
temporal-referred (T*) and the rest, and runs McNemar OMP-vs-uniform inside each.

Reads the *_i / *_picks_lc / *_picks_omp_lc sample jsonl for a bin.
Keyed on lvb_acc.id; correct = (answer == parsed_pred).
"""
import argparse
import json
import os
from collections import defaultdict
from math import comb


def load(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        a = json.loads(line)["lvb_acc"]
        d[a["id"]] = (int(a["answer"] == a["parsed_pred"]), a["question_category"])
    return d


def mcnemar(a, b, ids):
    b01 = sum(1 for i in ids if a[i][0] == 0 and b[i][0] == 1)
    b10 = sum(1 for i in ids if a[i][0] == 1 and b[i][0] == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def acc(d, ids):
    return sum(d[i][0] for i in ids) / len(ids) if ids else float("nan")


def report(bindir, binname):
    arms = {}
    for name, tag in [("uniform", "i"), ("topk", "picks_lc"), ("omp", "picks_omp_lc")]:
        arms[name] = load(os.path.join(bindir, f"{binname}_{tag}.jsonl"))

    ids = sorted(set(arms["uniform"]) & set(arms["omp"]) & set(arms["topk"]))
    cats = defaultdict(list)
    for i in ids:
        cats[arms["uniform"][i][1]].append(i)

    # LongVideoBench's first letter is the referred-context type; T* = temporal-referred.
    temporal = sorted(c for c in cats if c.startswith("T"))
    other = sorted(c for c in cats if not c.startswith("T"))
    t_ids = [i for c in temporal for i in cats[c]]
    o_ids = [i for c in other for i in cats[c]]

    print(f"\n{'='*74}\n{binname}s bin — n={len(ids)}\n{'='*74}")
    print(f"temporal-referred (T*): {len(temporal)} categories, {len(t_ids)} questions "
          f"({100*len(t_ids)/len(ids):.1f}%)  {temporal}")
    print(f"other:                  {len(other)} categories, {len(o_ids)} questions "
          f"({100*len(o_ids)/len(ids):.1f}%)")

    for label, sub in [("ALL", ids), ("TEMPORAL (T*)", t_ids), ("NON-TEMPORAL", o_ids)]:
        print(f"\n  {label}  n={len(sub)}")
        print(f"    uniform {acc(arms['uniform'], sub):.4f}   "
              f"top-k {acc(arms['topk'], sub):.4f}   omp {acc(arms['omp'], sub):.4f}")
        for m in ("topk", "omp"):
            b01, b10, p = mcnemar(arms["uniform"], arms[m], sub)
            d = 100 * (acc(arms[m], sub) - acc(arms["uniform"], sub))
            star = "SIG" if p < .05 else ("~" if p < .10 else "ns")
            print(f"    {m:5s} vs uniform  d={d:+6.2f} pp  disc {b10}v{b01}  p={p:.4g}  {star}")

    print("\n  per-category, OMP vs uniform:")
    rows = []
    for c in sorted(cats):
        sub = cats[c]
        b01, b10, p = mcnemar(arms["uniform"], arms["omp"], sub)
        rows.append((100 * (acc(arms["omp"], sub) - acc(arms["uniform"], sub)),
                     c, len(sub), b10, b01, p))
    for d, c, n, b10, b01, p in sorted(rows, reverse=True):
        mark = "T" if c.startswith("T") else " "
        print(f"    {mark} {c:5s} n={n:3d}  uni {acc(arms['uniform'], cats[c]):.3f} "
              f"omp {acc(arms['omp'], cats[c]):.3f}  d={d:+6.2f}  {b10}v{b01}  p={p:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding {bin}_{i,picks_lc,picks_omp_lc}.jsonl")
    ap.add_argument("--bins", nargs="+", default=["600", "3600"])
    a = ap.parse_args()
    for b in a.bins:
        report(a.dir, b)
