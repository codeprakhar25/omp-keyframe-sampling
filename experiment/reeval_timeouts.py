"""Re-evaluate cells whose Tier C verdict was a false 'test run timed out'.

Does NOT re-run the agent. Rebuilds each cell's workspace from its STORED
final_diff (base_sha-anchored agent source diff), then re-runs Tier C eval with
the raised EXP_EVAL_TIMEOUT, and updates the DB row in place. Preserves the exact
agent behavior (turns/cache/diff unchanged); only the pass/fail verdict is fixed.

Usage (on the pod, deps already built):
    set -a; source .env; set +a            # not strictly needed (no agent/API)
    EXP_EVAL_TIMEOUT=3600 python3 reeval_timeouts.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from harness.config import ExperimentConfig
from harness.runner import (
    checkout_sha, scrub_git_remotes, strip_future_history, load_tasks,
)
from harness.evaluate import evaluate_with_tests
from harness.db import ResultsDB


def main() -> None:
    cfg = ExperimentConfig()
    db = ResultsDB(cfg.db_path)
    tasks = {t.task_id: t for t in load_tasks("tasks/pilot.json")}

    rows = db.conn.execute(
        "SELECT run_id, task_id, strategy, repeat_index, final_diff "
        "FROM runs WHERE test_summary LIKE '%timed out%'"
    ).fetchall()
    print(f"timed-out cells to re-eval: {len(rows)}")

    for run_id, task_id, strategy, repeat, final_diff in rows:
        short = task_id.split("__")[-1]
        task = tasks.get(task_id)
        if task is None:
            print(f"  SKIP {short} {strategy} r{repeat}: task not in pilot.json")
            continue
        if not final_diff or not final_diff.strip():
            print(f"  SKIP {short} {strategy} r{repeat}: empty stored diff")
            continue

        repo_dir = str(cfg.repos_dir / task.repo_slug)
        ws = str(cfg.repos_dir / f"{task.repo_slug}__reeval")

        # fresh workspace at base_sha
        if os.path.isdir(ws):
            shutil.rmtree(ws)
        shutil.copytree(repo_dir, ws)
        if task.base_sha and not checkout_sha(ws, task.base_sha):
            print(f"  FAIL {short} {strategy} r{repeat}: checkout failed")
            continue
        scrub_git_remotes(ws)
        if task.base_sha:
            strip_future_history(ws, task.base_sha)

        # apply the agent's stored source diff
        ap = subprocess.run(["git", "-C", ws, "apply", "--whitespace=nowarn"],
                            input=final_diff, capture_output=True, text=True)
        if ap.returncode != 0:
            # retry 3-way (needs blobs; base is present)
            ap = subprocess.run(["git", "-C", ws, "apply", "--3way", "--whitespace=nowarn"],
                                input=final_diff, capture_output=True, text=True)
        if ap.returncode != 0:
            print(f"  FAIL {short} {strategy} r{repeat}: git apply failed: {ap.stderr[:120]}")
            continue

        passed, tr = evaluate_with_tests(ws, task.repo_slug, task.gold_diff, task.base_sha)
        if tr is None:
            print(f"  WARN {short} {strategy} r{repeat}: no tests -> skipped (keep old)")
            continue
        summary = (f"pass={tr.n_passed} fail={tr.n_failed} err={tr.n_errors} "
                   f"{tr.error or ''}".strip())
        db.conn.execute(
            "UPDATE runs SET task_passed=?, eval_method='tests', test_summary=? WHERE run_id=?",
            (1 if passed else 0, summary, run_id),
        )
        db.conn.commit()
        print(f"  OK   {short} {strategy} r{repeat}: passed={passed} | {summary}")

    db.close()
    print("done.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
