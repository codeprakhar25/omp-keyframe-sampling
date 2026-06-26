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
# Reward-hacking arms selectable via --strategies (not in the default set).
ALL_STRATEGIES = STRATEGIES + [ContextStrategy.NEUTRAL, ContextStrategy.PRESSURE]


# db agent column = RunLog.agent (claude arm = "claude_code", codex arm = "codex").
# Distinct strings → codex rows never collide with the existing claude dataset.
_DB_AGENT = {"claude": "claude_code", "codex": "codex"}
_BACKEND = {"claude": AgentBackend.CLAUDE, "codex": AgentBackend.CODEX}


def already_done(db: ResultsDB, task_id: str, strategy: str, repeat: int, db_agent: str) -> bool:
    """A clean run exists: eval_method populated and no fatal error."""
    cur = db.conn.execute(
        """SELECT COUNT(*) FROM runs
           WHERE task_id=? AND strategy=? AND repeat_index=? AND agent=?
                 AND eval_method IS NOT NULL AND eval_method != ''
                 AND (error IS NULL OR error='')
                 AND (test_summary IS NULL OR test_summary NOT LIKE '%failed%')""",
        (task_id, strategy, repeat, db_agent),
    )
    return cur.fetchone()[0] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--repo", default=None, help="filter tasks by repo substring")
    ap.add_argument("--task-file", default="tasks/pilot.json")
    ap.add_argument("--agent", choices=["claude", "codex"], default="claude",
                    help="agent backend: claude (ANTHROPIC_API_KEY) or codex (codex login)")
    ap.add_argument("--db", default=None, help="DB path (default results/experiment.db)")
    ap.add_argument("--strategies", default=None,
                    help="comma list subset of none,always_on,selective,neutral,pressure "
                         "(default = first 3)")
    ap.add_argument("--gameability", action="store_true",
                    help="reward-hacking: expose writable visible test subset, grade on "
                         "held-out (sets EXP_GAMEABILITY=1). Git guards stay ON.")
    args = ap.parse_args()

    if args.gameability:
        os.environ["EXP_GAMEABILITY"] = "1"

    backend = _BACKEND[args.agent]
    db_agent = _DB_AGENT[args.agent]

    strategies = STRATEGIES
    if args.strategies:
        want = {s.strip() for s in args.strategies.split(",") if s.strip()}
        strategies = [s for s in ALL_STRATEGIES if s.value in want]
        if not strategies:
            log.error("no valid strategies in %r (pick from none,always_on,selective,neutral,pressure)", args.strategies)
            sys.exit(1)

    # claude arm bills via ANTHROPIC_API_KEY; codex arm authenticates via `codex
    # login` (ChatGPT plan) → no API key required here.
    if args.agent == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Run: set -a; source .env; set +a")
        sys.exit(1)

    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    db = ResultsDB(args.db or str(cfg.db_path))

    tasks = load_tasks(args.task_file)
    if args.repo:
        tasks = [t for t in tasks if args.repo.lower() in t.repo_full_name.lower()]

    total = len(tasks) * len(strategies) * args.repeats
    done = skipped = ran = failed = 0

    log.info("Pilot [%s]: %d tasks x %d strategies x %d repeats = %d cells -> %s",
             args.agent, len(tasks), len(strategies), args.repeats, total, args.db or str(cfg.db_path))

    for repeat in range(args.repeats):
        for task in tasks:
            for strategy in strategies:
                done += 1
                tag = f"[{done}/{total}] {args.agent} {task.task_id.split('__')[-1]} {strategy.value} r{repeat}"
                if already_done(db, task.task_id, strategy.value, repeat, db_agent):
                    skipped += 1
                    log.info("%s SKIP (done)", tag)
                    continue
                log.info("%s RUN", tag)
                try:
                    run_log = run_single_task(task, strategy, backend, cfg, repeat)
                    rc = RunConfig(task=task, strategy=strategy, agent=backend).to_dict()
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
