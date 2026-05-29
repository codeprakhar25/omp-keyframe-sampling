"""Generate task definitions from a list of specific, pre-validated PR numbers.

Unlike task_generator.py (which discovers PRs automatically), this script
takes PRs you've already validated and fetches all the metadata needed
for the harness: base SHA, diff, linked issue text, patch stats.

Usage:
    python -m harness.task_generator_from_prs --output tasks/pilot.json
"""

from __future__ import annotations

import json
import logging
import os
import sys

from .task_generator import (
    PRRecord,
    PatchStats,
    classify_complexity,
    extract_linked_issue,
    fetch_issue_body,
    fetch_pr_diff,
    generate_prompt_from_diff,
    gh_api,
    parse_patch,
    records_to_task_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-validated PR list from phase1-repo-validation.md
# ---------------------------------------------------------------------------

VALIDATED_PRS: list[dict] = [
    # --- pdm-project/pdm ---
    {"repo": "pdm-project/pdm", "pr": 3769, "tier": "simple",
     "note": "Accept build system extensions from template on matching backend"},
    {"repo": "pdm-project/pdm", "pr": 3781, "tier": "simple",
     "note": "update pdm_scheme to set pep582_base"},
    {"repo": "pdm-project/pdm", "pr": 3790, "tier": "medium",
     "note": "update plugin installation path to use project_plugins_dir"},
    {"repo": "pdm-project/pdm", "pr": 3797, "tier": "medium",
     "note": "Use an existing pyproject.toml with PDM"},

    # --- firebase/firebase-admin-python ---
    {"repo": "firebase/firebase-admin-python", "pr": 939, "tier": "simple",
     "note": "Remove debug print for HTTP status error"},
    {"repo": "firebase/firebase-admin-python", "pr": 940, "tier": "simple",
     "note": "Add FCM satellite/bandwidth flags"},
    {"repo": "firebase/firebase-admin-python", "pr": 920, "tier": "medium",
     "note": "Enable Cloud Task Queue Emulator support"},
    {"repo": "firebase/firebase-admin-python", "pr": 942, "tier": "medium",
     "note": "Add Firebase Phone Number Verification"},

    # --- OpShin/opshin ---
    {"repo": "OpShin/opshin", "pr": 595, "tier": "simple",
     "note": "align int(str) with Python whitespace handling"},
    {"repo": "OpShin/opshin", "pr": 616, "tier": "simple",
     "note": "Loop optimizer steps to fixed-point"},
    {"repo": "OpShin/opshin", "pr": 605, "tier": "medium",
     "note": "Correctly handle fallthrough"},
    {"repo": "OpShin/opshin", "pr": 610, "tier": "medium",
     "note": "Fix nested Any and Union list casts"},
]


def fetch_pr_record(repo: str, pr_number: int, tier_hint: str, generate: bool = True) -> PRRecord | None:
    """Fetch full PR data from GitHub and build a PRRecord."""
    log.info("Fetching %s#%d ...", repo, pr_number)

    try:
        pr_data = gh_api(f"/repos/{repo}/pulls/{pr_number}")
    except Exception as e:
        log.error("Failed to fetch PR %s#%d: %s", repo, pr_number, e)
        return None

    pr_title = pr_data.get("title", "")
    pr_body = pr_data.get("body", "") or ""
    base_sha = pr_data.get("base", {}).get("sha", "")
    head_sha = pr_data.get("head", {}).get("sha", "")
    merge_sha = pr_data.get("merge_commit_sha", "")
    labels = [l.get("name", "") for l in pr_data.get("labels", [])]

    diff_text = fetch_pr_diff(repo, pr_number)
    if not diff_text:
        log.error("Empty diff for %s#%d", repo, pr_number)
        return None

    stats = parse_patch(diff_text)
    complexity = classify_complexity(stats)

    linked_issue = extract_linked_issue(pr_body, repo)
    linked_issue_body = None
    if linked_issue:
        linked_issue_body = fetch_issue_body(repo, linked_issue)

    prompt = ""
    prompt_source = ""
    if linked_issue_body and len(linked_issue_body.strip()) > 50:
        prompt = linked_issue_body.strip()
        prompt_source = "issue"
    elif pr_body and len(pr_body.strip()) > 80:
        prompt = pr_body.strip()
        prompt_source = "pr_body"
    elif generate:
        prompt = generate_prompt_from_diff(diff_text, repo, pr_title)
        prompt_source = "generated"
    else:
        prompt = f"# {pr_title}\n\n[Prompt generation skipped — use --generate flag]"
        prompt_source = "skipped"

    return PRRecord(
        repo=repo,
        number=pr_number,
        title=pr_title,
        body=pr_body,
        base_sha=base_sha,
        head_sha=head_sha,
        merge_commit_sha=merge_sha,
        diff_text=diff_text,
        linked_issue_number=linked_issue,
        linked_issue_body=linked_issue_body,
        labels=labels,
        patch_stats=stats,
        complexity=complexity,
        generated_prompt=prompt,
        prompt_source=prompt_source,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from pre-validated PRs")
    parser.add_argument("--output", default="tasks/pilot.json", help="Output task file")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip Claude API prompt generation")
    parser.add_argument("--raw-output", help="Optional path for full metadata")
    parser.add_argument("--repos", nargs="*",
                        help="Only process these repos (default: all)")
    args = parser.parse_args()

    records: list[PRRecord] = []
    targets = VALIDATED_PRS

    if args.repos:
        targets = [p for p in targets if p["repo"] in args.repos]

    for entry in targets:
        record = fetch_pr_record(
            entry["repo"], entry["pr"], entry["tier"],
            generate=not args.no_generate,
        )
        if record:
            records.append(record)
            log.info(
                "  OK: %s#%d [%s] — %d files, %d lines, prompt=%s",
                record.repo, record.number, record.complexity,
                record.patch_stats.files_changed if record.patch_stats else 0,
                record.patch_stats.total_lines if record.patch_stats else 0,
                record.prompt_source,
            )
        else:
            log.warning("  SKIP: %s#%d", entry["repo"], entry["pr"])

    tasks = records_to_task_json(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d tasks to %s", len(tasks), args.output)

    if args.raw_output:
        raw = []
        for r in records:
            raw.append({
                "repo": r.repo, "number": r.number, "title": r.title,
                "complexity": r.complexity, "prompt_source": r.prompt_source,
                "linked_issue": r.linked_issue_number, "base_sha": r.base_sha,
                "files_changed": r.patch_stats.files_changed if r.patch_stats else 0,
                "total_lines": r.patch_stats.total_lines if r.patch_stats else 0,
            })
        with open(args.raw_output, "w") as f:
            json.dump(raw, f, indent=2)

    print(f"\n=== Generated {len(tasks)} task definitions ===")
    for repo in sorted(set(r.repo for r in records)):
        repo_records = [r for r in records if r.repo == repo]
        print(f"\n  {repo}: {len(repo_records)} tasks")
        for r in repo_records:
            print(f"    #{r.number} [{r.complexity}] {r.prompt_source} — {r.title[:60]}")


if __name__ == "__main__":
    main()
