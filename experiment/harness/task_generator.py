"""Independent task generator for the Context File Paradox experiment.

Fetches merged PRs from any GitHub repo, classifies complexity,
extracts real issue descriptions (or generates them), and outputs
task definitions for the harness.

Design principles:
  - No dependency on Paper 1 or Paper 2 data
  - Prefer real issue text over synthetic — more ecologically valid
  - Transparent complexity classification from patch stats
  - Reproducible: all metadata preserved for audit

Usage:
    python -m harness.task_generator \
        --repos firebase/firebase-admin-python pdm-project/pdm OpShin/opshin \
        --per-repo 8 \
        --output tasks/pilot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchStats:
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    hunks: int = 0
    test_files_changed: int = 0
    non_test_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    non_test_lines_added: int = 0
    non_test_lines_removed: int = 0

    @property
    def total_lines(self) -> int:
        return self.lines_added + self.lines_removed

    @property
    def non_test_total_lines(self) -> int:
        return self.non_test_lines_added + self.non_test_lines_removed

    @property
    def non_test_files_changed(self) -> int:
        return len(self.non_test_files)


@dataclass
class PRRecord:
    """Raw PR data fetched from GitHub."""
    repo: str
    number: int
    title: str
    body: str
    base_sha: str
    head_sha: str
    merge_commit_sha: str
    diff_text: str
    linked_issue_number: int | None = None
    linked_issue_body: str | None = None
    labels: list[str] = field(default_factory=list)
    patch_stats: PatchStats | None = None
    complexity: str = ""
    generated_prompt: str = ""
    prompt_source: str = ""  # "issue", "pr_body", "generated"


# ---------------------------------------------------------------------------
# GitHub data fetching via `gh` CLI
# ---------------------------------------------------------------------------

def gh_api(endpoint: str, params: dict[str, str] | None = None) -> Any:
    """Call GitHub API via gh CLI and return parsed JSON."""
    cmd = ["gh", "api", endpoint, "--header", "Accept: application/vnd.github+json"]
    if params:
        for k, v in params.items():
            cmd.extend(["-f", f"{k}={v}"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed: {result.stderr}")
    return json.loads(result.stdout)


def gh_api_paginated(endpoint: str, per_page: int = 100, max_pages: int = 5) -> list[dict]:
    """Fetch paginated results from GitHub API."""
    all_results = []
    for page in range(1, max_pages + 1):
        url = f"{endpoint}?per_page={per_page}&page={page}&state=closed&sort=updated&direction=desc"
        data = gh_api(url)
        if not data:
            break
        all_results.extend(data)
        if len(data) < per_page:
            break
    return all_results


def fetch_merged_prs(repo: str, limit: int = 50) -> list[dict]:
    """Fetch recent merged PRs for a repo."""
    log.info("Fetching merged PRs for %s (limit=%d)", repo, limit)

    pages_needed = (limit // 30) + 2
    raw_prs = gh_api_paginated(f"/repos/{repo}/pulls", per_page=30, max_pages=pages_needed)

    merged = [pr for pr in raw_prs if pr.get("merged_at") is not None]
    log.info("Found %d merged PRs out of %d closed PRs for %s", len(merged), len(raw_prs), repo)
    return merged[:limit]


def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch the diff for a specific PR."""
    cmd = ["gh", "api", f"/repos/{repo}/pulls/{pr_number}",
           "--header", "Accept: application/vnd.github.v3.diff"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning("Failed to fetch diff for %s#%d: %s", repo, pr_number, result.stderr)
        return ""
    return result.stdout


def fetch_issue_body(repo: str, issue_number: int) -> str:
    """Fetch the body of a GitHub issue."""
    try:
        data = gh_api(f"/repos/{repo}/issues/{issue_number}")
        return data.get("body", "") or ""
    except Exception as e:
        log.warning("Failed to fetch issue %s#%d: %s", repo, issue_number, e)
        return ""


def extract_linked_issue(pr_body: str, repo: str) -> int | None:
    """Extract linked issue number from PR body.

    Looks for patterns like:
      - Fixes #123, Closes #456, Resolves #789
      - https://github.com/owner/repo/issues/123
    """
    if not pr_body:
        return None

    patterns = [
        r'(?:fix(?:es|ed)?|clos(?:es|ed)?|resolv(?:es|ed)?)\s+#(\d+)',
        rf'https?://github\.com/{re.escape(repo)}/issues/(\d+)',
        r'#(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, pr_body, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


# ---------------------------------------------------------------------------
# Patch analysis and complexity classification
# ---------------------------------------------------------------------------

TEST_PATTERNS = re.compile(
    r'(test[_/s]|_test\.|spec[_/s]|_spec\.|__tests__|fixtures|conftest|mock)',
    re.IGNORECASE,
)


def parse_patch(diff_text: str) -> PatchStats:
    """Parse a unified diff and extract statistics."""
    stats = PatchStats()

    current_file = None
    current_is_test = False
    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                current_file = match.group(1)
                stats.files_changed += 1
                current_is_test = bool(TEST_PATTERNS.search(current_file))
                if current_is_test:
                    stats.test_files_changed += 1
                    stats.test_files.append(current_file)
                else:
                    stats.non_test_files.append(current_file)
        elif line.startswith("@@"):
            stats.hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            stats.lines_added += 1
            if not current_is_test:
                stats.non_test_lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats.lines_removed += 1
            if not current_is_test:
                stats.non_test_lines_removed += 1

    return stats


def classify_complexity(stats: PatchStats) -> str:
    """Classify PR complexity into simple/medium/complex.

    Uses non-test files and non-test lines only — test code is output
    the agent must produce but the implementation difficulty is driven
    by the production code changes.
    """
    nf = stats.non_test_files_changed
    tl = stats.non_test_total_lines

    if nf <= 1 and tl <= 50:
        return "simple"
    elif nf <= 4 and tl <= 200:
        return "medium"
    else:
        return "complex"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

PROMPT_GENERATION_SYSTEM = """You are an expert software engineer acting as a "Bug/Feature Issue Writer."

Given a git diff, you must produce a GitHub Issue that describes the underlying 
problem or feature request — as if the fix does NOT exist yet.

Rules:
- Write from the perspective of someone who discovered the problem, NOT someone who already has the fix.
- NEVER mention "diff", "patch", "PR", "pull request", or commit hashes.
- NEVER describe the solution or say "this change does X."
- Ground all statements in the code evidence from the diff.
- If the change is a bug fix, describe the bug. If it's a feature, describe the need.
- Keep it concise and actionable, like a real GitHub issue.

Use this exact format:

# Title
One concise sentence.

# Background
- Bullets on affected area(s), current behavior, and why it matters.

# Problem Statement
Short paragraph describing the issue in plain language.

# Acceptance Criteria
- [ ] Verifiable outcomes as checkboxes.
- [ ] Include "does not break X" constraints when relevant.

# Notes / Pointers
- List relevant files/modules.
- Mention constraints from the codebase."""


def generate_prompt_from_diff(diff_text: str, repo: str, pr_title: str) -> str:
    """Use Claude API to generate an issue description from a diff.

    Falls back to a template if API is unavailable.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        user_msg = (
            f"Repository: {repo}\n"
            f"PR Title (for context only, do NOT reference it): {pr_title}\n\n"
            f"Diff:\n```\n{diff_text[:12000]}\n```"
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=PROMPT_GENERATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        return response.content[0].text

    except Exception as e:
        log.warning("Claude API unavailable for prompt generation: %s. Using template.", e)
        return _template_prompt(repo, pr_title, diff_text)


def _template_prompt(repo: str, title: str, diff_text: str) -> str:
    """Fallback template when API is unavailable."""
    stats = parse_patch(diff_text)
    files = ", ".join(f"`{f}`" for f in stats.non_test_files[:5])
    return (
        f"# Title\n{title}\n\n"
        f"# Background\n"
        f"- Affected files: {files}\n"
        f"- {stats.lines_added} lines added, {stats.lines_removed} lines removed\n\n"
        f"# Problem Statement\n"
        f"[Issue description needed — review the affected files and determine the underlying problem.]\n\n"
        f"# Acceptance Criteria\n"
        f"- [ ] The issue described above is resolved.\n"
        f"- [ ] All existing tests continue to pass.\n"
    )


# ---------------------------------------------------------------------------
# PR filtering criteria
# ---------------------------------------------------------------------------

def should_include_pr(stats: PatchStats, labels: list[str]) -> tuple[bool, str]:
    """Decide whether a PR is suitable for the experiment.

    Returns (include, reason).
    """
    skip_labels = {"dependencies", "dependabot", "renovate", "release", "ci", "docs"}
    label_overlap = set(l.lower() for l in labels) & skip_labels
    if label_overlap:
        return False, f"skip label: {label_overlap}"

    if stats.files_changed == 0:
        return False, "empty diff"

    if stats.non_test_files_changed == 0 and stats.test_files_changed > 0:
        return False, "test-only change"

    if stats.total_lines > 800:
        return False, f"too large ({stats.total_lines} lines)"

    if stats.total_lines < 3:
        return False, f"too trivial ({stats.total_lines} lines)"

    if stats.files_changed > 20:
        return False, f"too many files ({stats.files_changed})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_repo(
    repo: str,
    target_count: int = 8,
    generate_prompts: bool = True,
) -> list[PRRecord]:
    """Fetch, filter, classify, and generate tasks for a repo."""
    log.info("=== Processing %s ===", repo)

    raw_prs = fetch_merged_prs(repo, limit=target_count * 5)
    records: list[PRRecord] = []

    complexity_buckets: dict[str, int] = {"simple": 0, "medium": 0, "complex": 0}
    target_per_bucket = max(target_count // 3, 1)

    for pr_data in raw_prs:
        if len(records) >= target_count:
            break

        pr_number = pr_data["number"]
        pr_title = pr_data.get("title", "")
        pr_body = pr_data.get("body", "") or ""
        labels = [l.get("name", "") for l in pr_data.get("labels", [])]
        base_sha = pr_data.get("base", {}).get("sha", "")
        head_sha = pr_data.get("head", {}).get("sha", "")
        merge_sha = pr_data.get("merge_commit_sha", "")

        diff_text = fetch_pr_diff(repo, pr_number)
        if not diff_text:
            continue

        stats = parse_patch(diff_text)
        include, reason = should_include_pr(stats, labels)
        if not include:
            log.debug("Skipping %s#%d: %s", repo, pr_number, reason)
            continue

        complexity = classify_complexity(stats)

        if complexity_buckets[complexity] >= target_per_bucket + 1:
            log.debug("Skipping %s#%d: bucket %s full", repo, pr_number, complexity)
            continue

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
        elif generate_prompts:
            prompt = generate_prompt_from_diff(diff_text, repo, pr_title)
            prompt_source = "generated"
        else:
            prompt = _template_prompt(repo, pr_title, diff_text)
            prompt_source = "template"

        record = PRRecord(
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

        records.append(record)
        complexity_buckets[complexity] += 1
        log.info(
            "  #%d [%s] %s — %d files, %d lines, prompt=%s",
            pr_number, complexity, pr_title[:60], stats.non_test_files_changed,
            stats.total_lines, prompt_source,
        )

    log.info(
        "Selected %d tasks for %s: simple=%d medium=%d complex=%d",
        len(records), repo,
        complexity_buckets["simple"], complexity_buckets["medium"], complexity_buckets["complex"],
    )
    return records


def records_to_task_json(records: list[PRRecord]) -> list[dict[str, Any]]:
    """Convert PRRecords to the harness task JSON format."""
    tasks = []
    for r in records:
        task = {
            "task_id": f"{r.repo.replace('/', '__')}__{r.number}",
            "repo_full_name": r.repo,
            "repo_url": f"https://github.com/{r.repo}.git",
            "base_sha": r.base_sha,
            "pr_number": r.number,
            "complexity": r.complexity,
            "prompt": r.generated_prompt,
            "prompt_source": r.prompt_source,
            "gold_diff": r.diff_text,
            "pr_title": r.title,
            "linked_issue_number": r.linked_issue_number,
            "metadata": {
                "head_sha": r.head_sha,
                "merge_commit_sha": r.merge_commit_sha,
                "labels": r.labels,
                "files_changed": r.patch_stats.files_changed if r.patch_stats else 0,
                "non_test_files_changed": r.patch_stats.non_test_files_changed if r.patch_stats else 0,
                "lines_added": r.patch_stats.lines_added if r.patch_stats else 0,
                "lines_removed": r.patch_stats.lines_removed if r.patch_stats else 0,
                "total_lines": r.patch_stats.total_lines if r.patch_stats else 0,
                "non_test_lines_added": r.patch_stats.non_test_lines_added if r.patch_stats else 0,
                "non_test_lines_removed": r.patch_stats.non_test_lines_removed if r.patch_stats else 0,
                "non_test_total_lines": r.patch_stats.non_test_total_lines if r.patch_stats else 0,
                "non_test_files": r.patch_stats.non_test_files if r.patch_stats else [],
                "test_files": r.patch_stats.test_files if r.patch_stats else [],
            },
        }
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment task definitions from GitHub PRs"
    )
    parser.add_argument(
        "--repos", nargs="+", required=True,
        help="GitHub repos in owner/name format",
    )
    parser.add_argument(
        "--per-repo", type=int, default=8,
        help="Target number of tasks per repo (default 8)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip Claude API prompt generation (use templates instead)",
    )
    parser.add_argument(
        "--raw-output",
        help="Optional path to save full PR records with all metadata",
    )
    args = parser.parse_args()

    all_records: list[PRRecord] = []
    for repo in args.repos:
        records = process_repo(repo, args.per_repo, generate_prompts=not args.no_generate)
        all_records.extend(records)

    tasks = records_to_task_json(all_records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d tasks to %s", len(tasks), args.output)

    if args.raw_output:
        raw_data = []
        for r in all_records:
            d = {
                "repo": r.repo, "number": r.number, "title": r.title,
                "complexity": r.complexity, "prompt_source": r.prompt_source,
                "linked_issue_number": r.linked_issue_number,
                "base_sha": r.base_sha,
                "files_changed": r.patch_stats.files_changed if r.patch_stats else 0,
                "total_lines": r.patch_stats.total_lines if r.patch_stats else 0,
                "non_test_files": r.patch_stats.non_test_files if r.patch_stats else [],
            }
            raw_data.append(d)
        with open(args.raw_output, "w") as f:
            json.dump(raw_data, f, indent=2)
        log.info("Wrote raw metadata to %s", args.raw_output)

    print("\n=== Task Summary ===")
    print(f"Total tasks: {len(tasks)}")
    for repo in args.repos:
        repo_tasks = [t for t in tasks if t["repo_full_name"] == repo]
        sources = {}
        complexities = {}
        for t in repo_tasks:
            src = t["prompt_source"]
            sources[src] = sources.get(src, 0) + 1
            cx = t["complexity"]
            complexities[cx] = complexities.get(cx, 0) + 1
        print(f"\n  {repo}: {len(repo_tasks)} tasks")
        print(f"    Complexity: {complexities}")
        print(f"    Prompt source: {sources}")


if __name__ == "__main__":
    main()
