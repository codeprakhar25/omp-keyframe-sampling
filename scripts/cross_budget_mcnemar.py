"""Cross-budget paired significance: does OMP at k=8 match/beat uniform at 2x-4x tokens?

Zero GPU: joins existing per-item lmms-eval outputs across budgets (same official LVB
val items per bin, so pairing is exact) and runs exact binomial McNemar.

Claim under test (budget/token framing): selection (OMP-LongCLIP) buys a 2-4x visual-token
reduction at equal-or-better accuracy vs the field-standard uniform baseline.

Verification gates (abort if any fails):
  - each file has the exact official bin count AND no duplicate ids
    (dup/unique check also screens out the sharded-corruption failure mode)
  - id sets identical across every compared pair within a bin
  - recomputed accuracies match the mega-table numbers to 4 decimals
"""
from __future__ import annotations

import os
import json
import math
import sys
from pathlib import Path

BASE = Path(os.environ.get("SLM_RESULTS", "results"))
MC = BASE / "lmmseval_matrix_clean"
Q = "Qwen__Qwen3-VL-8B-Instruct"

# (label, path, expected_n, expected_acc from mega table)
FILES = {
    "omp_k8_60":   (MC / "k8_60" / Q / "20260716_131701_samples_longvideobench_val_picks_omp_lc_60s_k8.jsonl", 172, 0.7384),
    "unif_k16_60": (MC / "k16_60" / Q / "20260716_061556_samples_longvideobench_val_i_60s_k16.jsonl", 172, 0.7093),
    "omp_k8_600":   (MC / "k8_600" / Q / "20260716_134540_samples_longvideobench_val_picks_omp_lc_600s_k8.jsonl", 412, 0.6311),
    "unif_k16_600": (MC / "k16_600" / Q / "20260717_144556_samples_longvideobench_val_i_600s_k16.jsonl", 412, 0.5850),
    "unif_k32_600": (BASE / "k32_600_new" / Q / "20260716_204121_samples_longvideobench_val_i_600s_k32.jsonl", 412, 0.6044),
    "omp_k16_600":  (MC / "k16_600" / Q / "20260717_144556_samples_longvideobench_val_picks_omp_lc_600s_k16.jsonl", 412, 0.6578),
    "omp_k8_3600":   (MC / "k8_3600" / Q / "20260716_193815_samples_longvideobench_val_picks_omp_lc_3600s_k8.jsonl", 564, 0.5461),
    "unif_k16_3600": (MC / "k16_3600_merged" / "merged_samples_longvideobench_val_i_3600s_k16.jsonl", 564, 0.4770),
    # topk_k16_3600 EXCLUDED: local merged file scores .5461 != mega-table .5532
    # (merged dir holds a stale/void topk arm; valid re-run not local) -- gate caught it.
}

# (a, b, note) -- H: a >= b with a at fewer visual tokens
PAIRS = [
    ("omp_k8_60", "unif_k16_60", "60s: OMP 8f vs uniform 16f (1/2 tokens)"),
    ("omp_k8_600", "unif_k16_600", "600s: OMP 8f vs uniform 16f (1/2 tokens)"),
    ("omp_k8_600", "unif_k32_600", "600s: OMP 8f vs uniform 32f (1/4 tokens)"),
    ("omp_k16_600", "unif_k32_600", "600s: OMP 16f vs uniform 32f (1/2 tokens)"),
    ("omp_k8_3600", "unif_k16_3600", "3600s: OMP 8f vs uniform 16f (1/2 tokens)"),
]


def load(label: str) -> dict:
    f, n_exp, acc_exp = FILES[label]
    out = {}
    for line in open(f):
        r = json.loads(line)
        acc = r["lvb_acc"]
        qid = acc["id"]
        if qid in out:
            sys.exit(f"ABORT dup id {qid} in {label} ({f})")
        out[qid] = acc["answer"] == acc["parsed_pred"]
    if len(out) != n_exp:
        sys.exit(f"ABORT {label}: n={len(out)} != {n_exp}")
    acc_got = sum(out.values()) / n_exp
    if abs(acc_got - acc_exp) > 5e-5:
        sys.exit(f"ABORT {label}: acc {acc_got:.4f} != mega-table {acc_exp:.4f}")
    print(f"VERIFIED {label}: n={n_exp}, unique ids, acc {acc_got:.4f} == mega table")
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    data = {label: load(label) for label in FILES}
    print("\n=== cross-budget paired McNemar ===")
    results = []
    for a, b, note in PAIRS:
        ra, rb = data[a], data[b]
        if set(ra) != set(rb):
            sys.exit(f"ABORT id set mismatch {a} vs {b}")
        n = len(ra)
        win_a = sum(1 for q in ra if ra[q] and not rb[q])
        win_b = sum(1 for q in ra if rb[q] and not ra[q])
        p = mcnemar_exact(win_a, win_b)
        acc_a = sum(ra.values()) / n
        acc_b = sum(rb.values()) / n
        print(f"{note}\n  {a} {acc_a:.4f} vs {b} {acc_b:.4f}  "
              f"delta {100*(acc_a-acc_b):+.2f}pt  discordant {win_a}/{win_b}  p={p:.4g}")
        results.append({"a": a, "b": b, "note": note, "acc_a": acc_a, "acc_b": acc_b,
                        "win_a": win_a, "win_b": win_b, "p": p, "n": n})
    out = BASE / "cross_budget_mcnemar_k8.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
