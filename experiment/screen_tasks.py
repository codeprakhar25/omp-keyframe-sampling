"""Borderline-task screener (Option A: same 3 Python repos).

Purpose: find tasks worth ADDING to the experiment. Runs candidate tasks at the
`none` strategy only (baseline difficulty), N times each, and classifies by
pass-rate:

    too_easy   : passes every time  (all-pass -> no dynamic range -> DROP)
    too_hard   : fails every time   (all-fail -> no dynamic range -> DROP)
    BORDERLINE : sometimes passes   (~30-70% -> outcome can move -> KEEP)

Borderline tasks are the only ones that carry signal for the *correctness*
question (like 3790). For the *efficiency* question any non-trivial task adds
power, so the report also flags `medium_effort` keepers (enough turns/tool calls
to be non-trivial even if all-pass).

Why `none` only: we just need each candidate's baseline difficulty, not the full
3-strategy comparison. Screening at one strategy is 1/3 the cost.

Workflow:
  1. (laptop / network-open, needs gh) generate candidates, NOT the existing 11:
       python3 -m harness.task_generator --repos pdm-project/pdm \
         firebase/firebase-admin-python OpShin/opshin --per-repo 12 \
         --output tasks/candidates.json
  2. (locked pod) screen them:
       set -a; source .env; set +a
       EXP_LIVE=1 python3 screen_tasks.py --task-file tasks/candidates.json --repeats 3
  3. Review tasks/screen_report.md; merge tasks/screen_keepers.json into pilot.json.

Resumable: re-running skips (task, repeat) cells already in screen.db.
Writes to results/screen.db (NOT experiment.db — keeps the real dataset clean).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from harness.config import AgentBackend, ContextStrategy, ExperimentConfig, RunConfig
from harness.db import ResultsDB
from harness.runner import load_tasks, run_single_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("screen")


# DB agent column = the agent's own RunLog.agent string (claude arm = "claude_code",
# codex arm = "codex"). Keep in sync with run_pilot.py._DB_AGENT.
_DB_AGENT = {"claude": "claude_code", "codex": "codex"}


def already_done(db: ResultsDB, task_id: str, repeat: int, db_agent: str) -> bool:
    cur = db.conn.execute(
        """SELECT COUNT(*) FROM runs
           WHERE task_id=? AND strategy='none' AND repeat_index=? AND agent=?
                 AND eval_method IS NOT NULL AND eval_method != ''
                 AND (error IS NULL OR error='')""",
        (task_id, repeat, db_agent),
    )
    return cur.fetchone()[0] > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen candidate tasks for borderline difficulty")
    ap.add_argument("--task-file", default="tasks/candidates.json")
    ap.add_argument("--repeats", type=int, default=3, help="runs per candidate at none (3 -> 33/67%% bins)")
    ap.add_argument("--exclude", default="tasks/pilot.json", help="task file whose IDs to skip (already used)")
    ap.add_argument("--max-tasks", type=int, default=0, help="cap candidates screened (0=all)")
    ap.add_argument("--db", default=None, help="screening DB (default results/screen.db)")
    ap.add_argument("--agent", choices=["claude", "codex"], default="claude",
                    help="which agent screens (codex = free/cheap pool, preferred for triage)")
    args = ap.parse_args()

    backend = AgentBackend.CLAUDE if args.agent == "claude" else AgentBackend.CODEX
    db_agent = _DB_AGENT[args.agent]

    # claude needs the API key in env; codex auths via CODEX_HOME/auth.json (verify_lock checks).
    if args.agent == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Run: set -a; source .env; set +a")
        sys.exit(1)

    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    screen_db_path = args.db or str(cfg.results_dir / "screen.db")
    db = ResultsDB(screen_db_path)

    candidates = load_tasks(args.task_file)
    used = set()
    if os.path.isfile(args.exclude):
        used = {t.task_id for t in load_tasks(args.exclude)}
    candidates = [t for t in candidates if t.task_id not in used]
    if args.max_tasks:
        candidates = candidates[: args.max_tasks]

    n = args.repeats
    log.info("Screening %d candidates x %d (none) on agent=%s = %d runs -> %s",
             len(candidates), n, args.agent, len(candidates) * n, screen_db_path)
    log.info("(excluded %d already-used task IDs)", len(used))

    for ti, task in enumerate(candidates, 1):
        for rep in range(n):
            tag = f"[{ti}/{len(candidates)}] {task.task_id.split('__')[-1]} r{rep}"
            if already_done(db, task.task_id, rep, db_agent):
                log.info("%s SKIP", tag); continue
            log.info("%s RUN", tag)
            try:
                rl = run_single_task(task, ContextStrategy.NONE, backend, cfg, rep)
                rc = RunConfig(task=task, strategy=ContextStrategy.NONE, agent=backend).to_dict()
                db.save_run(rl, rc)
                log.info("%s -> passed=%s %s", tag, rl.task_passed, rl.test_summary or rl.error or "")
            except Exception as e:  # noqa: BLE001
                log.error("%s EXCEPTION %s", tag, e)

    classify_and_report(db, candidates, n, cfg, db_agent)
    db.close()


def classify_and_report(db: ResultsDB, candidates, n: int, cfg: ExperimentConfig,
                        db_agent: str = "claude_code") -> None:
    """Tally pass-rate per candidate, write report + keepers task file."""
    stats: dict[str, dict] = defaultdict(lambda: {"pass": 0, "runs": 0, "turns": [], "tools": []})
    for r in db.conn.execute(
        """SELECT task_id, task_passed, total_turns, total_tool_calls
           FROM runs WHERE strategy='none' AND agent=?
                 AND eval_method IS NOT NULL AND (error IS NULL OR error='')""", (db_agent,)):
        s = stats[r[0]]
        s["runs"] += 1; s["pass"] += (r[1] or 0)
        s["turns"].append(r[2] or 0); s["tools"].append(r[3] or 0)

    # Effort gate for the medium_effort (all-pass but non-trivial) class.
    # Codex emits exactly ONE turn.completed per session, so total_turns collapses
    # to 1 and a turns-based threshold can NEVER fire -> every codex all-pass would
    # be mislabelled too_easy. Use the pre-registered portable metric (tool_calls)
    # for codex; keep the turns convention for claude (where turns are meaningful).
    if db_agent == "codex":
        effort_key, effort_thresh, effort_label = "tools", 15, "tools"
    else:
        effort_key, effort_thresh, effort_label = "turns", 30, "turns"

    cand_by_id = {t.task_id: t for t in candidates}
    keepers, report_rows = [], []
    for tid, s in sorted(stats.items()):
        runs, p = s["runs"], s["pass"]
        rate = p / runs if runs else 0
        avg_turns = sum(s["turns"]) / len(s["turns"]) if s["turns"] else 0
        avg_tools = sum(s["tools"]) / len(s["tools"]) if s["tools"] else 0
        avg_effort = avg_tools if effort_key == "tools" else avg_turns
        if runs == 0:
            cls = "no_data"
        elif p == 0:
            cls = "too_hard"
        elif p == runs:
            cls = "medium_effort" if avg_effort >= effort_thresh else "too_easy"
        else:
            cls = "BORDERLINE"
        report_rows.append((tid.split("__")[-1], f"{p}/{runs}", f"{100*rate:.0f}%", f"{avg_effort:.0f}", cls))
        # KEEP borderline (correctness signal) + medium_effort all-pass (efficiency power)
        if cls in ("BORDERLINE", "medium_effort") and tid in cand_by_id:
            t = cand_by_id[tid]
            keepers.append({
                "task_id": t.task_id, "repo_full_name": t.repo_full_name,
                "repo_url": t.repo_url, "base_sha": t.base_sha, "prompt": t.prompt,
                "gold_diff": t.gold_diff, "complexity": str(getattr(t.complexity, "value", t.complexity)),
                "pr_number": t.pr_number, "_screen": {"pass": p, "runs": runs, "rate": rate, "class": cls},
            })

    keepers_path = str(cfg.results_dir.parent / "tasks" / "screen_keepers.json")
    with open(keepers_path, "w") as f:
        json.dump(keepers, f, indent=2, ensure_ascii=False)

    report_path = str(cfg.results_dir.parent / "tasks" / "screen_report.md")
    with open(report_path, "w") as f:
        f.write("# Screening report (none strategy)\n\n")
        f.write(f"Candidates with data: {len(report_rows)}  | keepers: {len(keepers)} "
                f"(BORDERLINE + medium_effort)\n\n")
        f.write(f"| task | pass | rate | {effort_label} | class |\n|---|---|---|---|---|\n")
        for row in sorted(report_rows, key=lambda x: x[4]):
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n**KEEP** = BORDERLINE (correctness signal) or medium_effort all-pass "
                "(efficiency power). Merge `tasks/screen_keepers.json` into `pilot.json`.\n")
        f.write("DROP = too_easy / too_hard (no dynamic range).\n")

    bl = sum(1 for r in report_rows if r[4] == "BORDERLINE")
    log.info("=== SCREEN DONE: %d candidates | %d borderline | %d keepers ===",
             len(report_rows), bl, len(keepers))
    log.info("report: %s", report_path)
    log.info("keepers: %s", keepers_path)


if __name__ == "__main__":
    main()
