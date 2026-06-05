"""Efficiency analysis — sliced by outcome group.

Reads the experiment DB, computes per-strategy efficiency metrics for:
  - always-pass tasks (clean signal, outcome controlled)
  - always-fail tasks (exploration behavior)
  - boundary tasks (3790 — reported separately)

Runs paired Wilcoxon signed-rank tests on the always-pass group.
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "results_from_pod_v2/experiment.db"

STRATEGIES = ["none", "always_on", "selective"]

ALWAYS_PASS = {"939", "940", "3769", "3781", "616"}
NEAR_PASS   = {"595"}       # none r2 flakes
BOUNDARY    = {"3790"}
ALWAYS_FAIL = {"920", "3797", "605", "610"}

GHOST_RUNS = {
    ("605", "none", 0, 35),      # duplicate laptop run
    ("3797", "always_on", 1, 1), # 1-turn error ghost
}


def load_data(db_path: str):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT task_id, strategy, repeat_index, total_turns,
               total_cache_read_tokens, total_cache_creation_tokens,
               total_input_tokens, total_output_tokens,
               total_tool_calls, total_duration_s, task_passed
        FROM runs
        WHERE agent='claude_code' AND eval_method IS NOT NULL
        ORDER BY task_id, strategy, repeat_index
    """).fetchall()
    conn.close()

    data = []
    for r in rows:
        tid = r[0].split("__")[-1]
        strat, rep, turns = r[1], r[2], r[3]
        if (tid, strat, rep, turns) in GHOST_RUNS:
            continue
        data.append({
            "task": tid, "strategy": strat, "repeat": rep,
            "turns": turns,
            "cache_read": r[4] or 0, "cache_creation": r[5] or 0,
            "input_tokens": r[6] or 0, "output_tokens": r[7] or 0,
            "tool_calls": r[8] or 0, "duration_s": r[9] or 0,
            "passed": r[10],
        })
    return data


def group_by(data, task_set):
    return [d for d in data if d["task"] in task_set]


def median(values):
    return float(np.median(values)) if values else 0.0


def fmt_k(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}k"
    return str(int(v))


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def per_task_strategy_medians(data):
    """Returns {task: {strategy: {metric: median_value}}}"""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in data:
        for metric in ["turns", "cache_read", "cache_creation", "duration_s", "tool_calls"]:
            grouped[d["task"]][d["strategy"]][metric].append(d[metric])

    result = {}
    for task in grouped:
        result[task] = {}
        for strat in grouped[task]:
            result[task][strat] = {
                m: median(grouped[task][strat][m])
                for m in grouped[task][strat]
            }
    return result


def strategy_aggregates(data):
    """Returns {strategy: {metric: [values]}}"""
    agg = defaultdict(lambda: defaultdict(list))
    for d in data:
        for metric in ["turns", "cache_read", "cache_creation", "duration_s", "tool_calls"]:
            agg[d["strategy"]][metric].append(d[metric])
    return agg


def paired_wilcoxon(data, metric, strat_a, strat_b):
    """Paired Wilcoxon on (task, repeat) pairs."""
    pairs_a, pairs_b = {}, {}
    for d in data:
        key = (d["task"], d["repeat"])
        if d["strategy"] == strat_a:
            pairs_a[key] = d[metric]
        elif d["strategy"] == strat_b:
            pairs_b[key] = d[metric]

    common = sorted(set(pairs_a) & set(pairs_b))
    if len(common) < 6:
        return None, None, len(common)

    a = np.array([pairs_a[k] for k in common])
    b = np.array([pairs_b[k] for k in common])

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        return stat, p, len(common)
    except ValueError:
        return None, None, len(common)


def main():
    data = load_data(DB_PATH)
    print(f"Loaded {len(data)} clean rows from {DB_PATH}")

    # ── 1. ALWAYS-PASS TASKS ──
    ap_data = group_by(data, ALWAYS_PASS)
    print_section("ALWAYS-PASS TASKS (939, 940, 3769, 3781, 616)")
    print(f"  {len(ap_data)} cells, all pass — efficiency comparison is clean\n")

    medians = per_task_strategy_medians(ap_data)
    print(f"  {'Task':<6} {'Metric':<12} {'none':>10} {'always_on':>10} {'selective':>10}  {'winner'}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}  {'-'*10}")

    for task in sorted(medians):
        for metric in ["turns", "cache_read", "duration_s"]:
            vals = {s: medians[task].get(s, {}).get(metric, 0) for s in STRATEGIES}
            winner = min(vals, key=vals.get)
            label = metric.replace("_", " ")
            if metric == "cache_read":
                print(f"  {task:<6} {label:<12} {fmt_k(vals['none']):>10} {fmt_k(vals['always_on']):>10} {fmt_k(vals['selective']):>10}  ← {winner}")
            else:
                print(f"  {task:<6} {label:<12} {vals['none']:>10.0f} {vals['always_on']:>10.0f} {vals['selective']:>10.0f}  ← {winner}")
        print()

    # Aggregate medians across all always-pass
    agg = strategy_aggregates(ap_data)
    print(f"\n  AGGREGATE (median across all always-pass cells):")
    print(f"  {'Metric':<18} {'none':>10} {'always_on':>10} {'selective':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["turns", "cache_read", "cache_creation", "duration_s", "tool_calls"]:
        vals = {s: median(agg[s][metric]) for s in STRATEGIES}
        label = metric.replace("_", " ")
        if "cache" in metric or "token" in metric:
            print(f"  {label:<18} {fmt_k(vals['none']):>10} {fmt_k(vals['always_on']):>10} {fmt_k(vals['selective']):>10}")
        else:
            print(f"  {label:<18} {vals['none']:>10.0f} {vals['always_on']:>10.0f} {vals['selective']:>10.0f}")

    # ── 2. PAIRED WILCOXON ──
    print_section("PAIRED WILCOXON — ALWAYS-PASS (n=15 pairs per comparison)")
    comparisons = [
        ("none", "always_on"),
        ("none", "selective"),
        ("always_on", "selective"),
    ]
    for metric in ["turns", "cache_read", "cache_creation", "duration_s"]:
        label = metric.replace("_", " ")
        print(f"\n  {label}:")
        for sa, sb in comparisons:
            stat, p, n = paired_wilcoxon(ap_data, metric, sa, sb)
            med_a = median([d[metric] for d in ap_data if d["strategy"] == sa])
            med_b = median([d[metric] for d in ap_data if d["strategy"] == sb])
            if p is not None:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"    {sa:>10} vs {sb:<10}  p={p:.4f} {sig:>3}  (n={n}, median {fmt_k(med_a)} vs {fmt_k(med_b)})")
            else:
                print(f"    {sa:>10} vs {sb:<10}  n={n} — insufficient pairs")

    # ── 3. ALWAYS-FAIL TASKS ──
    af_data = group_by(data, ALWAYS_FAIL)
    print_section("ALWAYS-FAIL TASKS (920, 3797, 605, 610)")
    print(f"  {len(af_data)} cells — exploration behavior comparison\n")

    af_agg = strategy_aggregates(af_data)
    print(f"  {'Metric':<18} {'none':>10} {'always_on':>10} {'selective':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["turns", "cache_read", "cache_creation", "duration_s", "tool_calls"]:
        vals = {s: median(af_agg[s][metric]) for s in STRATEGIES}
        label = metric.replace("_", " ")
        if "cache" in metric:
            print(f"  {label:<18} {fmt_k(vals['none']):>10} {fmt_k(vals['always_on']):>10} {fmt_k(vals['selective']):>10}")
        else:
            print(f"  {label:<18} {vals['none']:>10.0f} {vals['always_on']:>10.0f} {vals['selective']:>10.0f}")

    # ── 4. BOUNDARY TASK 3790 ──
    b_data = group_by(data, BOUNDARY)
    print_section("BOUNDARY TASK — 3790 (pdm, medium)")
    print(f"  {'repeat':<8} {'none':>20} {'always_on':>20} {'selective':>20}")
    for rep in range(3):
        cells = {s: None for s in STRATEGIES}
        for d in b_data:
            if d["repeat"] == rep:
                cells[d["strategy"]] = d
        parts = []
        for s in STRATEGIES:
            c = cells[s]
            if c:
                p = "PASS" if c["passed"] else "FAIL"
                parts.append(f"{p} {c['turns']:>2}t {fmt_k(c['cache_read']):>6}")
            else:
                parts.append("—")
        print(f"  r{rep:<7} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")

    # ── 5. 595 FLAKE ──
    f_data = group_by(data, NEAR_PASS)
    print_section("NEAR-PASS TASK — 595 (opshin, simple)")
    for rep in range(3):
        cells = {s: None for s in STRATEGIES}
        for d in f_data:
            if d["repeat"] == rep:
                cells[d["strategy"]] = d
        parts = []
        for s in STRATEGIES:
            c = cells[s]
            if c:
                p = "PASS" if c["passed"] else "FAIL"
                parts.append(f"{p} {c['turns']:>2}t {fmt_k(c['cache_read']):>6}")
            else:
                parts.append("—")
        print(f"  r{rep:<7} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")

    # ── 6. COST ESTIMATE ──
    print_section("COST SUMMARY (cache-based estimate)")
    total_cache = sum(d["cache_read"] + d["cache_creation"] for d in data)
    total_output = sum(d["output_tokens"] for d in data)
    est_cost = (total_cache * 0.30 / 1_000_000) + (total_output * 15.0 / 1_000_000)
    print(f"  Total cache tokens: {fmt_k(total_cache)}")
    print(f"  Total output tokens: {fmt_k(total_output)}")
    print(f"  Estimated cost (Sonnet 4 pricing): ~${est_cost:.0f}")
    print(f"  Per cell average: ~${est_cost/len(data):.2f}")


if __name__ == "__main__":
    main()
