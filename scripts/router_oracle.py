"""Offline router/oracle analysis over existing k=8 lmms-eval per-item outputs.

Zero GPU, zero re-scoring: joins the already-run topk-lc and omp-lc (and uniform)
per-item predictions and asks (a) what an oracle that picks the better arm per item
would score (union ceiling), and (b) what a *pre-registered* text-rule router scores.

PRE-REGISTERED ROUTER RULE (fixed 2026-07-19 before computing any outcome numbers,
from the sonnet visual-analysis mechanism read -- T3-family/temporal-anchor questions
favor topk's dense-local sampling, everything else favors OMP's spread):
    route to TOPK if the question STEM (text before the first "\nA. " option line,
    lowercased) matches any of:
        \bbefore\b | \bafter\b | \bfirst\b | \bimmediately\b
        | when-...-say (regex: \bwhen\b.*\b(say|says|said|mention|mentions|mentioned)\b)
    else route to OMP.
Options text is EXCLUDED from the rule input (selector phase is question-only; and
option strings contain spurious "first/before" tokens). Gold question_category is
NEVER used for routing (leakage) -- it appears only in clearly-labeled diagnostic
tables (in-sample category ceiling).

Verification gates (abort if any fails):
  - each arm file has the exact official bin count (15s:189, 60s:172, 600s:412, 3600s:564)
  - id sets identical across arms within a bin, no duplicate ids
  - recomputed arm accuracies match the known mega-table numbers to 4 decimals
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

BASE = Path("/workspace/slm-lab/results/lmmseval_matrix_clean")

BINS = {
    "15": ("k8_15", "20260716_124827", "15s", 189),
    "60": ("k8_60", "20260716_131701", "60s", 172),
    "600": ("k8_600", "20260716_134540", "600s", 412),
    "3600": ("k8_3600", "20260716_193815", "3600s", 564),
}
ARMS = {"uniform": "i", "topk": "picks_lc", "omp": "picks_omp_lc"}

# mega-table cross-check (accuracy to 4dp) -- verification gate
EXPECTED_ACC = {
    ("15", "uniform"): 0.7249, ("15", "topk"): 0.7249, ("15", "omp"): 0.7249,
    ("60", "uniform"): 0.7267, ("60", "topk"): 0.7442, ("60", "omp"): 0.7384,
    ("600", "uniform"): 0.5534, ("600", "topk"): 0.6141, ("600", "omp"): 0.6311,
    ("3600", "uniform"): 0.4716, ("3600", "topk"): 0.5106, ("3600", "omp"): 0.5461,
}

TEMPORAL_RE = re.compile(
    r"\bbefore\b|\bafter\b|\bfirst\b|\bimmediately\b"
    r"|\bwhen\b.*\b(say|says|said|mention|mentions|mentioned)\b"
)


def load_arm(bin_key: str, arm: str) -> dict:
    d, ts, suffix, _ = BINS[bin_key]
    f = (BASE / d / "Qwen__Qwen3-VL-8B-Instruct"
         / f"{ts}_samples_longvideobench_val_{ARMS[arm]}_{suffix}_k8.jsonl")
    out = {}
    for line in open(f):
        r = json.loads(line)
        acc = r["lvb_acc"]
        qid = acc["id"]
        if qid in out:
            sys.exit(f"ABORT dup id {qid} in {f}")
        stem = r["input"].split("\nA. ")[0].strip()
        out[qid] = {
            "correct": acc["answer"] == acc["parsed_pred"],
            "cat": acc["question_category"],
            "stem": stem,
        }
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    data = {}   # bin -> qid -> {uniform,topk,omp,cat,stem}
    for bin_key, (_, _, _, n_expected) in BINS.items():
        arms = {a: load_arm(bin_key, a) for a in ARMS}
        ids = set(arms["omp"])
        for a, recs in arms.items():
            if len(recs) != n_expected:
                sys.exit(f"ABORT {bin_key}/{a}: n={len(recs)} != {n_expected}")
            if set(recs) != ids:
                sys.exit(f"ABORT {bin_key}/{a}: id set mismatch vs omp arm")
        # verification gate: recomputed accuracy vs mega table
        for a in ARMS:
            acc = sum(r["correct"] for r in arms[a].values()) / n_expected
            exp = EXPECTED_ACC[(bin_key, a)]
            if abs(acc - exp) > 5e-5:
                sys.exit(f"ABORT {bin_key}/{a}: acc {acc:.4f} != expected {exp:.4f}")
        # stems must agree across arms (same question, different frames)
        merged = {}
        for qid in ids:
            stems = {arms[a][qid]["stem"] for a in ARMS}
            if len(stems) != 1:
                sys.exit(f"ABORT {bin_key}/{qid}: stem differs across arms")
            merged[qid] = {
                "cat": arms["omp"][qid]["cat"],
                "stem": arms["omp"][qid]["stem"],
                **{a: arms[a][qid]["correct"] for a in ARMS},
            }
        data[bin_key] = merged
        print(f"VERIFIED bin {bin_key}: n={n_expected}, ids consistent, "
              f"accuracies match mega table")

    print("\n=== per-bin results (k=8) ===")
    header = (f"{'bin':>5} {'n':>4} {'unif':>6} {'topk':>6} {'omp':>6} "
              f"{'oracle2':>7} {'oracle3':>7} {'router':>6} {'nT':>4} "
              f"{'p_vs_omp':>9} {'p_vs_topk':>9}")
    print(header)
    tot = {k: 0 for k in ("n", "uniform", "topk", "omp", "or2", "or3",
                          "router", "routed_topk", "rb", "rc", "tb", "tc")}
    per_bin = {}
    for bin_key, merged in data.items():
        n = len(merged)
        acc = {a: sum(r[a] for r in merged.values()) / n for a in ARMS}
        or2 = sum(r["topk"] or r["omp"] for r in merged.values()) / n
        or3 = sum(r["uniform"] or r["topk"] or r["omp"]
                  for r in merged.values()) / n
        routed, rb, rc, tb, tc, n_topk_routed = 0, 0, 0, 0, 0, 0
        for r in merged.values():
            to_topk = bool(TEMPORAL_RE.search(r["stem"].lower()))
            n_topk_routed += to_topk
            hit = r["topk"] if to_topk else r["omp"]
            routed += hit
            # discordant pairs router vs omp / router vs topk
            if hit and not r["omp"]:
                rb += 1
            if r["omp"] and not hit:
                rc += 1
            if hit and not r["topk"]:
                tb += 1
            if r["topk"] and not hit:
                tc += 1
        p_omp = mcnemar_exact(rb, rc)
        p_topk = mcnemar_exact(tb, tc)
        print(f"{bin_key:>5} {n:>4} {acc['uniform']:.4f} {acc['topk']:.4f} "
              f"{acc['omp']:.4f} {or2:>7.4f} {or3:>7.4f} {routed/n:>6.4f} "
              f"{n_topk_routed:>4} {p_omp:>9.4f} {p_topk:>9.4f}")
        per_bin[bin_key] = {
            "n": n, **{a: acc[a] for a in ARMS}, "oracle2": or2, "oracle3": or3,
            "router": routed / n, "routed_to_topk": n_topk_routed,
            "router_vs_omp": {"b": rb, "c": rc, "p": p_omp},
            "router_vs_topk": {"b": tb, "c": tc, "p": p_topk},
        }
        tot["n"] += n
        for a in ARMS:
            tot[a] += sum(r[a] for r in merged.values())
        tot["or2"] += sum(r["topk"] or r["omp"] for r in merged.values())
        tot["or3"] += sum(r["uniform"] or r["topk"] or r["omp"]
                          for r in merged.values())
        tot["router"] += routed
        tot["routed_topk"] += n_topk_routed
        tot["rb"] += rb
        tot["rc"] += rc
        tot["tb"] += tb
        tot["tc"] += tc

    n = tot["n"]
    p_omp_all = mcnemar_exact(tot["rb"], tot["rc"])
    p_topk_all = mcnemar_exact(tot["tb"], tot["tc"])
    print(f"{'ALL':>5} {n:>4} {tot['uniform']/n:.4f} {tot['topk']/n:.4f} "
          f"{tot['omp']/n:.4f} {tot['or2']/n:>7.4f} {tot['or3']/n:>7.4f} "
          f"{tot['router']/n:>6.4f} {tot['routed_topk']:>4} "
          f"{p_omp_all:>9.4f} {p_topk_all:>9.4f}")

    print("\n=== rescue structure (long bins) ===")
    for bin_key in ("600", "3600"):
        merged = data[bin_key]
        omp_f = [r for r in merged.values() if not r["omp"]]
        topk_f = [r for r in merged.values() if not r["topk"]]
        t_rescues = sum(r["topk"] for r in omp_f)
        o_rescues = sum(r["omp"] for r in topk_f)
        both_f = sum(1 for r in merged.values() if not r["omp"] and not r["topk"])
        print(f"{bin_key}s: omp fails {len(omp_f)} (topk rescues {t_rescues}), "
              f"topk fails {len(topk_f)} (omp rescues {o_rescues}), "
              f"both fail {both_f}")

    print("\n=== DIAGNOSTIC ONLY (gold-category routing = leakage; in-sample "
          "ceiling for any category router) ===")
    for bin_key in ("600", "3600"):
        merged = data[bin_key]
        cats = sorted({r["cat"] for r in merged.values()})
        cat_ceiling = 0
        rows = []
        for cat in cats:
            rs = [r for r in merged.values() if r["cat"] == cat]
            ta = sum(r["topk"] for r in rs)
            oa = sum(r["omp"] for r in rs)
            cat_ceiling += max(ta, oa)
            rows.append((cat, len(rs), ta, oa))
        print(f"\n{bin_key}s per-category (n, topk✓, omp✓):")
        for cat, cn, ta, oa in rows:
            better = "topk" if ta > oa else ("omp" if oa > ta else "tie")
            print(f"  {cat:>4} n={cn:>3} topk={ta:>3} omp={oa:>3}  -> {better}")
        n_bin = len(merged)
        print(f"  gold-cat ceiling: {cat_ceiling}/{n_bin} = {cat_ceiling/n_bin:.4f}"
              f"  (vs omp {per_bin[bin_key]['omp']:.4f})")

    out = Path("/workspace/slm-lab/results/router_oracle_k8.json")
    out.write_text(json.dumps({"per_bin": per_bin,
                               "rule": TEMPORAL_RE.pattern}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
