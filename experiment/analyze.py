"""Quick analysis of experiment results from SQLite DB.

Usage:
    python analyze.py [--agent claude_code] [--repo firebase]
"""

import argparse
import sqlite3
import sys
from pathlib import Path


DB_PATH = Path(__file__).parent / "results" / "experiment.db"


def run_query(con, sql, params=()):
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def print_table(rows, cols=None):
    if not rows:
        print("  (no data)")
        return
    if cols is None:
        cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="claude_code")
    parser.add_argument("--repo", default=None, help="Filter by repo substring")
    args = parser.parse_args()

    con = sqlite3.connect(str(DB_PATH))

    # Only trustworthy runs: eval_method populated (bare + Tier C pipeline).
    # Old contaminated/raw-API runs have eval_method NULL and are excluded.
    where = "agent = ? AND eval_method IS NOT NULL AND eval_method != ''"
    params = [args.agent]
    if args.repo:
        where += " AND repo LIKE ?"
        params.append(f"%{args.repo}%")

    print(f"\n=== Runs by agent={args.agent} ===")
    rows = run_query(con, f"""
        SELECT strategy, COUNT(*) as n_runs,
               SUM(CASE WHEN task_passed=1 THEN 1 ELSE 0 END) as n_pass,
               ROUND(100.0*SUM(CASE WHEN task_passed=1 THEN 1 ELSE 0 END)/COUNT(*), 1) as pass_pct,
               ROUND(AVG(total_turns), 1) as avg_turns,
               ROUND(AVG(total_cache_read_tokens)/1000, 1) as avg_cache_read_k,
               ROUND(AVG(total_cache_creation_tokens)/1000, 1) as avg_cache_create_k,
               ROUND(AVG(total_tool_calls), 1) as avg_tools
        FROM runs WHERE {where}
        GROUP BY strategy ORDER BY strategy
    """, params)
    print_table(rows)

    print(f"\n=== Per-task results (agent={args.agent}) ===")
    rows = run_query(con, f"""
        SELECT task_id, strategy,
               COUNT(*) as n,
               SUM(CASE WHEN task_passed=1 THEN 1 ELSE 0 END) as pass,
               ROUND(AVG(total_turns),1) as turns,
               ROUND(AVG(total_cache_read_tokens)/1000,1) as cache_r_k,
               ROUND(AVG(total_cache_creation_tokens)/1000,1) as cache_c_k,
               ROUND(AVG(total_tool_calls),1) as tools,
               MAX(eval_method) as eval
        FROM runs WHERE {where}
        GROUP BY task_id, strategy
        ORDER BY task_id, strategy
    """, params)
    for r in rows:
        r["task_id"] = r["task_id"].split("__")[-1]
    print_table(rows, cols=["task_id", "strategy", "n", "pass", "turns", "cache_r_k", "cache_c_k", "tools", "eval"])

    print(f"\n=== Test summaries (Tier C) ===")
    rows = run_query(con, f"""
        SELECT task_id, strategy, task_passed, test_summary FROM runs
        WHERE {where} AND eval_method='tests'
        ORDER BY task_id, strategy
    """, params)
    for r in rows:
        t = r["task_id"].split("__")[-1]
        print(f"  {t} | {r['strategy']:<11} pass={bool(r['task_passed'])} | {r['test_summary']}")

    print(f"\n=== Errors ===")
    rows = run_query(con, f"""
        SELECT task_id, strategy, error FROM runs
        WHERE {where} AND error IS NOT NULL
        ORDER BY task_id
    """, params)
    for r in rows:
        print(f"  {r['task_id']} | {r['strategy']}: {(r['error'] or '')[:80]}")

    con.close()


if __name__ == "__main__":
    main()
