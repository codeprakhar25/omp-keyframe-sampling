"""Experiment runner — orchestrates clone, setup, agent run, and teardown.

Usage:
    python -m harness.runner --task-file tasks/pilot.json --strategy none --agent claude
    python -m harness.runner --task-file tasks/pilot.json --strategy always_on --agent claude
    python -m harness.runner --task-file tasks/pilot.json --strategy selective --agent claude
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .agent import ClaudeCodeAgent
from .config import (
    AgentBackend,
    ContextStrategy,
    ExperimentConfig,
    RunConfig,
    TaskConfig,
)
from .context import split_agents_md_to_wiki
from .evaluate import evaluate_with_tests
from .db import ResultsDB
from .logger import RunLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def clone_repo(repo_url: str, target_dir: str) -> bool:
    if os.path.isdir(target_dir):
        log.info("Repo already cloned at %s", target_dir)
        return True

    log.info("Cloning %s → %s", repo_url, target_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, target_dir],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("Clone failed: %s", e.stderr)
        return False


def _normalize_diff_lines(diff_text: str) -> set[tuple[str, str]]:
    """Extract (file, changed_line) pairs from a diff, ignoring context and headers."""
    changes: set[tuple[str, str]] = set()
    current_file = ""
    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            current_file = match.group(1) if match else ""
        elif line.startswith("+") and not line.startswith("+++"):
            changes.add((current_file, "+" + line[1:].strip()))
        elif line.startswith("-") and not line.startswith("---"):
            changes.add((current_file, "-" + line[1:].strip()))
    return changes


def evaluate_diff(actual_diff: str, gold_diff: str) -> bool:
    """Check whether the agent's diff covers the same substantive changes as the gold diff.

    Uses a relaxed comparison: extracts the set of (file, changed_line) pairs
    from both diffs and checks overlap. Ignores whitespace-only differences,
    context lines, and AGENTS.md removal (which is a harness artifact).
    """
    actual = _normalize_diff_lines(actual_diff)
    gold = _normalize_diff_lines(gold_diff)

    actual = {(f, l) for f, l in actual if not f.endswith("AGENTS.md")}
    gold = {(f, l) for f, l in gold if not f.endswith("AGENTS.md")}

    if not gold:
        return len(actual) == 0

    overlap = actual & gold
    recall = len(overlap) / len(gold) if gold else 0

    return recall >= 0.5


def checkout_sha(repo_dir: str, sha: str) -> bool:
    log.info("Checking out %s in %s", sha, repo_dir)
    try:
        subprocess.run(["git", "-C", repo_dir, "fetch", "--depth=1", "origin", sha],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["git", "-C", repo_dir, "fetch", "--unshallow"],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            pass

    try:
        subprocess.run(
            ["git", "-C", repo_dir, "checkout", sha],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", repo_dir, "clean", "-fdx"],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("Checkout failed: %s", e.stderr)
        return False


def get_diff(repo_dir: str, base_sha: str = "") -> str:
    # Exclude context files (AGENTS.md/CLAUDE.md) — their removal is a harness
    # artifact that pollutes the diff and crowds out real code changes.
    # Anchor on base_sha (not HEAD) so agent `git commit`s are still captured.
    ref = base_sha if base_sha else "HEAD"
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "-P", "diff", ref, "--",
             ".", ":(exclude)AGENTS.md", ":(exclude)CLAUDE.md", ":(exclude)**/CLAUDE.md"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def load_agents_md(repo_dir: str) -> str:
    for name in ["AGENTS.md", "CLAUDE.md"]:
        path = os.path.join(repo_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def run_single_task(
    task: TaskConfig,
    strategy: ContextStrategy,
    agent_backend: AgentBackend,
    exp_config: ExperimentConfig,
    repeat_index: int = 0,
) -> RunLog:
    """Set up workspace, run agent, capture results."""
    run_id = f"{task.task_id}__{strategy.value}__{repeat_index}__{uuid.uuid4().hex[:8]}"
    log.info("=== Starting run: %s ===", run_id)

    repo_dir = str(exp_config.repos_dir / task.repo_slug)
    workspace_dir = str(exp_config.repos_dir / f"{task.repo_slug}__work")

    if not clone_repo(task.repo_url, repo_dir):
        run_log = RunLog(run_id=run_id, task_id=task.task_id, strategy=strategy.value, agent=agent_backend.value)
        run_log.error = "Clone failed"
        return run_log

    if os.path.isdir(workspace_dir):
        subprocess.run(
            ["git", "-C", workspace_dir, "reset", "--hard", "HEAD"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", workspace_dir, "clean", "-fdx"],
            capture_output=True, text=True,
        )
        shutil.rmtree(workspace_dir)
    shutil.copytree(repo_dir, workspace_dir)

    if task.base_sha:
        if not checkout_sha(workspace_dir, task.base_sha):
            run_log = RunLog(run_id=run_id, task_id=task.task_id, strategy=strategy.value, agent=agent_backend.value)
            run_log.error = "Checkout failed"
            return run_log

    agents_md_content = load_agents_md(workspace_dir)

    # Under --bare isolation, the agent never auto-reads context files. We remove
    # AGENTS.md/CLAUDE.md from the workspace in ALL strategies so the ONLY context
    # the agent receives is what we explicitly inject via --append-system-prompt.
    # This makes injection strategy the sole independent variable.
    def _strip_context_files() -> None:
        for name in ["AGENTS.md", "CLAUDE.md"]:
            p = os.path.join(workspace_dir, name)
            if os.path.isfile(p):
                os.remove(p)

    wiki_path = None
    append_system: str | None = None

    if strategy == ContextStrategy.NONE:
        _strip_context_files()

    elif strategy == ContextStrategy.ALWAYS_ON:
        _strip_context_files()
        if agents_md_content:
            append_system = (
                "The following is the repository's AGENTS.md guide. Follow its "
                "conventions and architectural guidance when making changes.\n\n"
                "--- AGENTS.md ---\n"
                f"{agents_md_content}\n"
                "--- END AGENTS.md ---"
            )

    elif strategy == ContextStrategy.SELECTIVE:
        # Generate wiki files from AGENTS.md
        source_wiki_path = str(exp_config.wiki_dir / task.repo_slug)
        if agents_md_content:
            split_agents_md_to_wiki(agents_md_content, source_wiki_path)

        # Copy wiki files into workspace so the agent can Read them directly
        workspace_wiki = os.path.join(workspace_dir, "wiki")
        if os.path.isdir(source_wiki_path):
            if os.path.isdir(workspace_wiki):
                shutil.rmtree(workspace_wiki)
            shutil.copytree(source_wiki_path, workspace_wiki)

        wiki_path = workspace_wiki

        # Remove AGENTS.md — agent must seek context in wiki/
        _strip_context_files()

        append_system = (
            "Repository context is available in the wiki/ subdirectory of the workspace. "
            "Use the Read tool on wiki/*.md files or Bash to grep the wiki/ directory "
            "for relevant context before making changes."
        )

    run_config = RunConfig(
        task=task,
        strategy=strategy,
        agent=agent_backend,
        run_id=run_id,
        repeat_index=repeat_index,
        workspace_dir=workspace_dir,
        agents_md_path=os.path.join(workspace_dir, "AGENTS.md") if strategy == ContextStrategy.ALWAYS_ON else None,
        wiki_dir=wiki_path,
    )

    if agent_backend == AgentBackend.CLAUDE:
        agent = ClaudeCodeAgent(run_config, workspace_dir)
        run_log = agent.run(task.prompt, append_system=append_system)
    else:
        log.error("Codex backend not yet wired — use Paper 1 replication package")
        run_log = RunLog(run_id=run_id, task_id=task.task_id, strategy=strategy.value, agent=agent_backend.value)
        run_log.error = "Codex backend not implemented in this harness"

    # Capture the agent's source diff BEFORE injecting any gold tests.
    # Anchor on base_sha so agent commits (which move HEAD) are still captured.
    run_log.final_diff = get_diff(workspace_dir, task.base_sha)

    if task.gold_diff:
        # Tier C: test-based eval. Falls back to line-overlap if gold has no tests.
        passed, test_result = evaluate_with_tests(
            workspace_dir, task.repo_slug, task.gold_diff, task.base_sha
        )
        if test_result is None:
            # No tests in gold PR — use relaxed line-overlap fallback.
            run_log.task_passed = evaluate_diff(run_log.final_diff, task.gold_diff)
            run_log.eval_method = "line_overlap"
        else:
            run_log.task_passed = passed
            run_log.eval_method = "tests"
            run_log.test_summary = (
                f"pass={test_result.n_passed} fail={test_result.n_failed} "
                f"err={test_result.n_errors} {test_result.error or ''}".strip()
            )
            if test_result.error:
                log.warning("Test eval issue: %s", test_result.error)

    log.info("Run complete: %d turns, %d tool calls, passed=%s (%s) %s",
             len(run_log.turns), run_log.total_tool_calls,
             run_log.task_passed, getattr(run_log, "eval_method", "n/a"),
             getattr(run_log, "test_summary", ""))

    return run_log


def load_tasks(task_file: str) -> list[TaskConfig]:
    with open(task_file, "r") as f:
        data = json.load(f)

    tasks = []
    for item in data:
        tasks.append(TaskConfig(
            task_id=item["task_id"],
            repo_full_name=item["repo_full_name"],
            repo_url=item.get("repo_url", f"https://github.com/{item['repo_full_name']}.git"),
            base_sha=item.get("base_sha", ""),
            prompt=item["prompt"],
            gold_diff=item.get("gold_diff"),
            complexity=item.get("complexity", "medium"),
            pr_number=item.get("pr_number"),
        ))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment tasks")
    parser.add_argument("--task-file", required=True, help="JSON file with task definitions")
    parser.add_argument("--strategy", required=True, choices=[s.value for s in ContextStrategy])
    parser.add_argument("--agent", default="claude", choices=[a.value for a in AgentBackend])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--task-id", help="Run only a specific task ID")
    args = parser.parse_args()

    exp_config = ExperimentConfig()
    exp_config.ensure_dirs()

    db = ResultsDB(exp_config.db_path)

    tasks = load_tasks(args.task_file)
    strategy = ContextStrategy(args.strategy)
    agent_backend = AgentBackend(args.agent)

    if args.task_id:
        tasks = [t for t in tasks if t.task_id == args.task_id]

    log.info("Loaded %d tasks, strategy=%s, agent=%s, repeats=%d",
             len(tasks), strategy.value, agent_backend.value, args.repeats)

    for task in tasks:
        for repeat in range(args.repeats):
            run_log = run_single_task(task, strategy, agent_backend, exp_config, repeat)

            run_config_dict = RunConfig(task=task, strategy=strategy, agent=agent_backend).to_dict()
            db.save_run(run_log, run_config_dict)

            results_json = exp_config.results_dir / f"{run_log.run_id}.json"
            with open(results_json, "w") as f:
                json.dump(run_log.to_dict(), f, indent=2)
            log.info("Saved results to %s", results_json)

    comparison = db.get_comparison_table()
    if comparison:
        log.info("=== Results Summary ===")
        for row in comparison:
            log.info(
                "  %s | %s | turns=%.1f | tokens=%d/%d | tools=%.1f | pass=%d/%d",
                row["task_id"][:30], row["strategy"],
                row["avg_turns"], row["avg_input_tokens"], row["avg_output_tokens"],
                row["avg_tool_calls"], row["pass_count"], row["n_runs"],
            )

    db.close()


if __name__ == "__main__":
    main()
