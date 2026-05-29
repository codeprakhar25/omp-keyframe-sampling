"""Resumable full-pilot driver.

Runs every (task, strategy, repeat) cell. Skips cells already present in the DB
as a clean run (eval_method populated, no error), so it survives interruptions
(rate limits, crashes) — just re-launch and it continues where it left off.

Requires ANTHROPIC_API_KEY in env (bare-mode billing, no session limit).

Usage:
    set -a; source .env; set +a
    python3 run_pilot.py --repeats 1
    python3 run_pilot.py --repeats 3 --repo firebase
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from harness.config import AgentBackend, ContextStrategy, ExperimentConfig, RunConfig
from harness.db import ResultsDB
from harness.runner import load_tasks, run_single_task

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pilot")

STRATEGIES = [ContextStrategy.NONE, ContextStrategy.ALWAYS_ON, ContextStrategy.SELECTIVE]


def already_done(db: ResultsDB, task_id: str, strategy: str, repeat: int) -> bool:
    """A clean run exists: eval_method populated and no fatal error."""
    cur = db.conn.execute(
        """SELECT COUNT(*) FROM runs
           WHERE task_id=? AND strategy=? AND repeat_index=? AND agent='claude_code'
                 AND eval_method IS NOT NULL AND eval_method != ''
                 AND (error IS NULL OR error='')
                 AND (test_summary IS NULL OR test_summary NOT LIKE '%failed%')""",
        (task_id, strategy, repeat),
    )
    return cur.fetchone()[0] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--repo", default=None, help="filter tasks by repo substring")
    ap.add_argument("--task-file", default="tasks/pilot.json")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Run: set -a; source .env; set +a")
        sys.exit(1)

    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    db = ResultsDB(cfg.db_path)

    tasks = load_tasks(args.task_file)
    if args.repo:
        tasks = [t for t in tasks if args.repo.lower() in t.repo_full_name.lower()]

    total = len(tasks) * len(STRATEGIES) * args.repeats
    done = skipped = ran = failed = 0

    log.info("Pilot: %d tasks x %d strategies x %d repeats = %d cells",
             len(tasks), len(STRATEGIES), args.repeats, total)

    for repeat in range(args.repeats):
        for task in tasks:
            for strategy in STRATEGIES:
                done += 1
                tag = f"[{done}/{total}] {task.task_id.split('__')[-1]} {strategy.value} r{repeat}"
                if already_done(db, task.task_id, strategy.value, repeat):
                    skipped += 1
                    log.info("%s SKIP (done)", tag)
                    continue
                log.info("%s RUN", tag)
                try:
                    run_log = run_single_task(task, strategy, AgentBackend.CLAUDE, cfg, repeat)
                    rc = RunConfig(task=task, strategy=strategy, agent=AgentBackend.CLAUDE).to_dict()
                    db.save_run(run_log, rc)
                    if run_log.error:
                        failed += 1
                        log.warning("%s ERROR: %s", tag, run_log.error[:120])
                    else:
                        ran += 1
                        log.info("%s -> passed=%s (%s) %s", tag, run_log.task_passed,
                                 run_log.eval_method, run_log.test_summary)
                except Exception as e:
                    failed += 1
                    log.error("%s EXCEPTION: %s", tag, e)

    log.info("Pilot complete: ran=%d skipped=%d failed=%d / %d", ran, skipped, failed, total)
    db.close()


if __name__ == "__main__":
    main()
