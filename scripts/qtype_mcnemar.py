#!/usr/bin/env python3
"""McNemar significance for the question-type split: is OMP's gain really absent on temporal?

Paired per-item (same videos, two arms). McNemar exact two-sided on discordant pairs:
  b = arm-A correct & arm-B wrong,  c = A wrong & B correct.
Reports, per (bin,k) and per group, omp-vs-uniform and topk-vs-uniform.
Zero GPU, existing logs. Usage: python3 scripts/qtype_mcnemar.py --root results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from math import comb


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(min(b, c) + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def parse_name(fn):
    m = re.search(r"_val_(picks_omp_lc|picks_lc|i)_(\d+)s_k(\d+)\.jsonl", os.path.basename(fn))
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    args = ap.parse_args()
    armname = {"picks_omp_lc": "omp", "picks_lc": "topk", "i": "uniform"}

    cell = defaultdict(lambda: defaultdict(dict))
    cat_of = {}
    for fn in glob.glob(f"{args.root}/**/*samples_longvideobench_val_*.jsonl", recursive=True):
        p = parse_name(fn)
        if not p:
            continue
        arm = armname[p[0]]
        for line in open(fn, encoding="utf-8"):
            try:
                la = json.loads(line).get("lvb_acc")
            except Exception:
                la = None
            if not la:
                continue
            cat_of[la["id"]] = la["question_category"]
            cell[(p[1], p[2])][arm][la["id"]] = int(
                str(la["answer"]).strip() == str(la["parsed_pred"]).strip())

    def test(ids, A, B):
        b = sum(1 for q in ids if A[q] and not B[q])
        c = sum(1 for q in ids if not A[q] and B[q])
        return b, c, mcnemar_p(b, c)

    for (bn, k) in sorted(cell, key=lambda x: (int(x[0]), x[1])):
        arms = cell[(bn, k)]
        if not all(a in arms for a in ("uniform", "topk", "omp")):
            continue
        ids = set(arms["uniform"]) & set(arms["topk"]) & set(arms["omp"])
        if len(ids) < 20:
            continue
        print(f"\n=== BIN {bn}s k={k}  n={len(ids)} ===")
        for gname, pred in (("ALL ", lambda c: True),
                            ("T*  ", lambda c: c[0] == "T"),
                            ("rest", lambda c: c[0] != "T")):
            gids = [q for q in ids if pred(cat_of[q])]
            n = len(gids)
            du = sum(arms["uniform"][q] for q in gids) / n
            do = sum(arms["omp"][q] for q in gids) / n
            dt = sum(arms["topk"][q] for q in gids) / n
            bo, co, po = test(gids, arms["uniform"], arms["omp"])       # +co favors omp
            bt, ct, pt = test(gids, arms["uniform"], arms["topk"])
            star = "*" if po < 0.05 else ("." if po < 0.10 else " ")
            print(f"  {gname} n={n:>3}  unif {du:.3f} topk {dt:.3f} omp {do:.3f} | "
                  f"omp-unif {do-du:+.3f} (disc {bo}/{co}, p={po:.3f}){star}  "
                  f"topk-unif {dt-du:+.3f} (p={pt:.3f})")


if __name__ == "__main__":
    main()
