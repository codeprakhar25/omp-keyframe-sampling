"""Calibrate Tier C eval WITHOUT calling any agent / API.

For a task:
  POSITIVE control: apply gold SOURCE changes (perfect agent) -> tests must PASS
  NEGATIVE control: apply nothing (no fix)               -> tests must FAIL

If both hold, the eval discriminates correctly.
"""
import sys, os, shutil, subprocess, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from harness.config import ExperimentConfig
from harness.evaluate import (
    extract_test_patch, split_diff_by_file, _is_test_file,
    prepare_env, apply_test_patch, run_tests,
)
from harness.runner import clone_repo, checkout_sha


def source_patch(gold_diff: str) -> str:
    blocks = [b for p, b in split_diff_by_file(gold_diff) if not _is_test_file(p)]
    patch = "\n".join(blocks)
    return patch + "\n" if patch and not patch.endswith("\n") else patch


def apply_patch(workspace, patch, label):
    if not patch.strip():
        return True
    for extra in (["--3way"], ["--reject"], []):
        p = subprocess.run(["git", "-C", workspace, "apply", "--whitespace=nowarn", *extra, "-"],
                           input=patch, capture_output=True, text=True)
        if p.returncode == 0:
            return True
    print(f"  [{label}] git apply FAILED: {p.stderr[:300]}")
    return False


def fresh_workspace(repo_dir, workspace, sha):
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    shutil.copytree(repo_dir, workspace)
    return checkout_sha(workspace, sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    tasks = json.load(open("tasks/pilot.json"))
    task = next(t for t in tasks if t["task_id"] == args.task_id)
    slug = task["repo_full_name"].replace("/", "__")
    repo_dir = str(cfg.repos_dir / slug)
    workspace = str(cfg.repos_dir / f"{slug}__calib")
    gold = task["gold_diff"]

    clone_repo(task.get("repo_url", f"https://github.com/{task['repo_full_name']}.git"), repo_dir)

    test_patch, test_files = extract_test_patch(gold)
    print(f"Task {args.task_id}: test_files={test_files}")
    if not test_files:
        print("  NO TESTS — line-overlap fallback task, skipping calibration")
        return

    # ---- POSITIVE control: gold source -> tests pass ----
    print("\n[POSITIVE] gold source changes + gold tests:")
    if not fresh_workspace(repo_dir, workspace, task["base_sha"]):
        print("  checkout failed"); return
    if not apply_patch(workspace, source_patch(gold), "src"):
        return
    if not prepare_env(workspace, slug):
        print("  env prep FAILED"); return
    apply_test_patch(workspace, test_patch)
    pos = run_tests(workspace, slug, test_files)
    print(f"  -> passed={pos.passed} (pass={pos.n_passed} fail={pos.n_failed} err={pos.n_errors}) {pos.error or ''}")
    if not pos.passed:
        print("  TAIL:", pos.raw_output[-800:])

    # ---- NEGATIVE control: no source change -> tests fail ----
    print("\n[NEGATIVE] no fix + gold tests:")
    if not fresh_workspace(repo_dir, workspace, task["base_sha"]):
        return
    prepare_env(workspace, slug)
    apply_test_patch(workspace, test_patch)
    neg = run_tests(workspace, slug, test_files)
    print(f"  -> passed={neg.passed} (pass={neg.n_passed} fail={neg.n_failed} err={neg.n_errors}) {neg.error or ''}")

    print("\n=== VERDICT ===")
    ok = pos.passed and not neg.passed
    print(f"  POSITIVE passes: {pos.passed}")
    print(f"  NEGATIVE fails:  {not neg.passed}")
    print(f"  EVAL DISCRIMINATES: {ok}")

    if os.path.isdir(workspace):
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
