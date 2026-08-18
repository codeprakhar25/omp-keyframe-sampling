#!/usr/bin/env python3
"""Pooled paired McNemar across bins.

Each --pair gives BIN:TEST.jsonl:BASE.jsonl. Pairing is by lvb_acc.id WITHIN a bin
(ids are disjoint across bins, but we key on (bin, id) to be safe). Discordant
counts are summed across bins and one exact two-sided binomial test is run on the
pooled counts -- this is the standard stratified/pooled McNemar and is valid
because the strata (duration bins) are disjoint sets of videos.

Usage: pooled_mcnemar.py --pair 600:test.jsonl:base.jsonl --pair 3600:t.jsonl:b.jsonl
"""
import argparse, json, sys
from math import comb


def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        try:
            la = json.loads(line).get("lvb_acc")
        except Exception:
            la = None
        if la:
            d[la["id"]] = int(str(la["answer"]).strip() == str(la["parsed_pred"]).strip())
    return d


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n * 2)


ap = argparse.ArgumentParser()
ap.add_argument("--pair", action="append", required=True)
ap.add_argument("--label", default="test")
a = ap.parse_args()

tot_b = tot_c = tot_n = 0
tot_test = tot_base = 0
print(f"\n=== pooled McNemar: {a.label} vs baseline ===")
for spec in a.pair:
    binname, tf, bf = spec.split(":", 2)
    T, B = load(tf), load(bf)
    ids = sorted(set(T) & set(B))
    n = len(ids)
    if n == 0:
        sys.exit(f"FATAL: no paired ids for bin {binname}")
    at = sum(T[i] for i in ids) / n
    ab = sum(B[i] for i in ids) / n
    b = sum(1 for i in ids if B[i] and not T[i])
    c = sum(1 for i in ids if T[i] and not B[i])
    print(f"  bin {binname:>5}: n={n:>4}  base {ab:.4f}  {a.label} {at:.4f}  "
          f"delta {at-ab:+.4f}  disc {b}/{c}  p={mcnemar_p(b,c):.4f}"
          f"   (test file n={len(T)}, base n={len(B)})")
    tot_b += b
    tot_c += c
    tot_n += n
    tot_test += sum(T[i] for i in ids)
    tot_base += sum(B[i] for i in ids)

p = mcnemar_p(tot_b, tot_c)
print(f"\n  POOLED n={tot_n}  base {tot_base/tot_n:.4f}  {a.label} {tot_test/tot_n:.4f}  "
      f"delta {(tot_test-tot_base)/tot_n:+.4f}")
print(f"  pooled discordant base+/test+ = {tot_b}/{tot_c}   exact two-sided p = {p:.4f}")
print(f"  {'SIGNIFICANT (p<0.05)' if p < 0.05 else 'not significant'}\n")
