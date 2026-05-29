"""Tier C evaluation — test-based outcome scoring (SWE-bench style).

The agent edits source code but never sees the gold PR's test files. After the
agent finishes, we extract the test-file portion of the gold diff, apply it onto
the agent's workspace, and run those tests. The task passes iff the gold tests
pass against the agent's code.

Falls back to line-overlap (evaluate_diff) for tasks whose gold PR has no tests.

Per-repo environment setup is config-driven (REPO_TEST_CONFIG). Commands mirror
the validated commands in harness/repo-validation/phase1-repo-validation.md.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-repo test environment configuration
# ---------------------------------------------------------------------------

@dataclass
class RepoTestConfig:
    """How to set up deps and run pytest for a given repo."""
    # Shell to prepare the env once per workspace. Runs with cwd=workspace.
    # Should be idempotent / cheap on re-run. May be empty (uv resolves on demand).
    setup_cmd: list[str] = field(default_factory=list)
    # Template to run pytest. {python} and {tests} are substituted.
    # {tests} expands to space-joined "file::nodeid" or just file paths.
    test_cmd: list[str] = field(default_factory=list)
    # Extra env vars merged into os.environ for setup + test.
    env: dict[str, str] = field(default_factory=dict)
    # If True, create a venv at <workspace>/.venv and use its python.
    needs_venv: bool = False


_CACHE_ENV = {"XDG_CACHE_HOME": "/tmp/agents-experiment-cache"}

REPO_TEST_CONFIG: dict[str, RepoTestConfig] = {
    "firebase__firebase-admin-python": RepoTestConfig(
        needs_venv=True,
        setup_cmd=[
            "uv", "pip", "install", "--python", ".venv/bin/python",
            "-r", "requirements.txt", "-e", ".",
        ],
        test_cmd=[".venv/bin/python", "-m", "pytest", "{tests}", "-q",
                  "--no-header", "-p", "no:cacheprovider"],
        env=_CACHE_ENV,
    ),
    "pdm-project__pdm": RepoTestConfig(
        test_cmd=["uv", "run", "--group", "test", "pytest", "{tests}", "-q",
                  "--no-header", "-p", "no:cacheprovider"],
        env=_CACHE_ENV,
    ),
    "OpShin__opshin": RepoTestConfig(
        test_cmd=["uv", "run", "--group", "dev", "pytest", "{tests}", "-q",
                  "--no-header", "-p", "no:cacheprovider"],
        env=_CACHE_ENV,
    ),
}


# ---------------------------------------------------------------------------
# Gold-diff test extraction
# ---------------------------------------------------------------------------

_TEST_PATH_RE = re.compile(r"(^|/)(tests?|test)([/_]|\.py$)", re.IGNORECASE)


def _is_test_file(path: str) -> bool:
    """Heuristic: path is a test file. Excludes integration tests (need creds)."""
    if "integration" in path.lower():
        return False
    base = os.path.basename(path)
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or "/tests/" in path
        or path.startswith("tests/")
        or "/test/" in path
    )


def split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified diff into (file_path, file_diff_block) pairs."""
    blocks: list[tuple[str, str]] = []
    current_file = ""
    current_lines: list[str] = []
    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file and current_lines:
                blocks.append((current_file, "\n".join(current_lines)))
            m = re.search(r" b/(.+)$", line)
            current_file = m.group(1) if m else ""
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_file and current_lines:
        blocks.append((current_file, "\n".join(current_lines)))
    return blocks


def extract_test_patch(gold_diff: str) -> tuple[str, list[str]]:
    """Return (test_only_patch_text, list_of_test_file_paths) from gold diff.

    Integration tests are excluded (they need real credentials / emulators).
    """
    test_blocks = []
    test_files = []
    for path, block in split_diff_by_file(gold_diff):
        if _is_test_file(path):
            test_blocks.append(block)
            test_files.append(path)
    patch = "\n".join(test_blocks)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch, test_files


# ---------------------------------------------------------------------------
# Environment prep + test execution
# ---------------------------------------------------------------------------

def _make_env(cfg: RepoTestConfig) -> dict[str, str]:
    env = dict(os.environ)
    env.update(cfg.env)
    # Use API key auth path, never interactive
    return env


def prepare_env(workspace: str, repo_slug: str) -> bool:
    """Set up the test environment for a workspace. Returns True on success."""
    cfg = REPO_TEST_CONFIG.get(repo_slug)
    if cfg is None:
        log.warning("No test config for repo %s", repo_slug)
        return False

    env = _make_env(cfg)

    if cfg.needs_venv:
        venv_python = os.path.join(workspace, ".venv", "bin", "python")
        if not os.path.isfile(venv_python):
            try:
                subprocess.run(
                    ["uv", "venv", "--python", "3.12", ".venv"],
                    cwd=workspace, env=env, check=True,
                    capture_output=True, text=True, timeout=300,
                )
            except subprocess.CalledProcessError as e:
                log.error("venv creation failed: %s", e.stderr[:300])
                return False

    if cfg.setup_cmd:
        try:
            r = subprocess.run(
                cfg.setup_cmd, cwd=workspace, env=env,
                capture_output=True, text=True, timeout=900,
            )
            if r.returncode != 0:
                log.error("setup failed (rc=%d): %s", r.returncode, r.stderr[:500])
                return False
        except subprocess.TimeoutExpired:
            log.error("setup timed out")
            return False
    return True


def apply_test_patch(workspace: str, test_patch: str) -> bool:
    """Apply the gold test-file patch onto the workspace. Returns True on success."""
    if not test_patch.strip():
        return True
    # Try a sequence of increasingly lenient git-apply strategies.
    for extra in (["--3way"], ["--reject"], ["-C1"], []):
        proc = subprocess.run(
            ["git", "-C", workspace, "apply", "--whitespace=nowarn", *extra, "-"],
            input=test_patch, capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return True
    log.error("git apply test patch failed: %s", proc.stderr[:400])
    return False


@dataclass
class TestResult:
    passed: bool
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = 0
    raw_output: str = ""
    error: str | None = None


def _parse_pytest(output: str) -> tuple[int, int, int]:
    """Parse pytest summary line for passed/failed/error counts."""
    n_pass = n_fail = n_err = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        n_pass = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        n_fail = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        n_err = int(m.group(1))
    return n_pass, n_fail, n_err


def run_tests(workspace: str, repo_slug: str, test_files: list[str]) -> TestResult:
    """Run the given test files in the workspace. Pass iff no failures/errors."""
    cfg = REPO_TEST_CONFIG.get(repo_slug)
    if cfg is None:
        return TestResult(passed=False, error=f"no test config for {repo_slug}")
    if not test_files:
        return TestResult(passed=False, error="no test files")

    env = _make_env(cfg)
    tests_arg = test_files  # run whole test files (all nodes)

    cmd: list[str] = []
    for tok in cfg.test_cmd:
        if tok == "{tests}":
            cmd.extend(tests_arg)
        else:
            cmd.append(tok)

    try:
        r = subprocess.run(
            cmd, cwd=workspace, env=env,
            capture_output=True, text=True, timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return TestResult(passed=False, error="test run timed out (1200s)")

    out = (r.stdout or "") + "\n" + (r.stderr or "")
    n_pass, n_fail, n_err = _parse_pytest(out)
    passed = (r.returncode == 0) and n_fail == 0 and n_err == 0 and n_pass > 0
    return TestResult(
        passed=passed, n_passed=n_pass, n_failed=n_fail, n_errors=n_err,
        raw_output=out[-3000:],
    )


def evaluate_with_tests(
    workspace: str, repo_slug: str, gold_diff: str, base_sha: str = ""
) -> tuple[bool, TestResult | None]:
    """Full Tier C eval. Returns (passed, TestResult).

    Returns (False, None) with TestResult.error if no tests in gold (caller
    should fall back to line-overlap).
    """
    test_patch, test_files = extract_test_patch(gold_diff)
    if not test_files:
        return False, None  # signal: no tests, use fallback

    if not prepare_env(workspace, repo_slug):
        return False, TestResult(passed=False, error="env prep failed")

    # Revert the agent's edits to the gold test files back to base_sha. The gold
    # tests are the oracle — the agent doesn't get to write/modify them. Source
    # changes are preserved. (Standard SWE-bench: checkout base test files, then
    # apply the gold test patch onto a clean base.)
    #
    # Anchor to base_sha, NOT HEAD: agents sometimes `git commit`, which moves
    # HEAD so "checkout HEAD -- file" would restore the agent's version, not base.
    ref = base_sha if base_sha else "HEAD"
    for tf in test_files:
        # If the test file existed at base_sha, restore the base version (discard
        # agent edits). If it's a NEW file (gold adds it), the agent may have created
        # its own version at that path — delete it so the gold patch creates it fresh.
        existed = subprocess.run(
            ["git", "-C", workspace, "cat-file", "-e", f"{ref}:{tf}"],
            capture_output=True, text=True,
        ).returncode == 0
        if existed:
            r = subprocess.run(
                ["git", "-C", workspace, "checkout", ref, "--", tf],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                log.warning("reset of %s failed: %s", tf, r.stderr[:150])
        else:
            fp = os.path.join(workspace, tf)
            if os.path.isfile(fp):
                os.remove(fp)

    if not apply_test_patch(workspace, test_patch):
        return False, TestResult(passed=False, error="test patch apply failed")

    result = run_tests(workspace, repo_slug, test_files)
    return result.passed, result
