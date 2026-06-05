#!/usr/bin/env python3
"""
Download SWE-Bench-Verified and SWE-Bench-Lite, compute complexity proxy
from gold patches, and output a stratified CSV for experiment task selection.

Complexity proxy:
  - files_changed: number of files in gold patch
  - lines_changed: total added + removed lines in gold patch
  - hunks: number of diff hunks

Tiers:
  Simple:  1 file AND < 30 lines changed
  Medium:  2-4 files OR 30-150 lines changed (whichever triggers first)
  Complex: 5+ files OR > 150 lines changed

Outputs:
  data/swe_bench_verified.csv   — full verified set with complexity
  data/swe_bench_lite.csv       — full lite set with complexity
  data/swe_bench_stratified.csv — merged + deduped, ready for task selection
"""

import csv
import os
import re
import sys
from collections import Counter

from datasets import load_dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_patch_stats(patch: str) -> dict:
    """Extract files changed, lines added/removed, and hunk count from a unified diff."""
    if not patch:
        return {"files_changed": 0, "lines_added": 0, "lines_removed": 0, "hunks": 0}

    files = set()
    added = 0
    removed = 0
    hunks = 0

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                files.add(match.group(1))
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    return {
        "files_changed": len(files),
        "lines_added": added,
        "lines_removed": removed,
        "hunks": hunks,
    }


def classify_complexity(files_changed: int, total_lines: int) -> str:
    if files_changed >= 5 or total_lines > 150:
        return "complex"
    if files_changed >= 2 or total_lines >= 30:
        return "medium"
    return "simple"


def process_dataset(name: str, dataset_path: str, split: str = "test") -> list[dict]:
    print(f"Loading {name} from {dataset_path} (split={split})...")
    ds = load_dataset(dataset_path, split=split)
    print(f"  Loaded {len(ds)} instances")

    rows = []
    for item in ds:
        instance_id = item.get("instance_id", "")
        repo = instance_id.rsplit("-", 1)[0] if "-" in instance_id else instance_id
        patch = item.get("patch", "")

        stats = parse_patch_stats(patch)
        total_lines = stats["lines_added"] + stats["lines_removed"]
        tier = classify_complexity(stats["files_changed"], total_lines)

        rows.append({
            "instance_id": instance_id,
            "repo": repo,
            "source": name,
            "files_changed": stats["files_changed"],
            "lines_added": stats["lines_added"],
            "lines_removed": stats["lines_removed"],
            "total_lines_changed": total_lines,
            "hunks": stats["hunks"],
            "complexity_tier": tier,
        })

    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"  No rows to write for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def print_summary(rows: list[dict], label: str) -> None:
    tier_counts = Counter(r["complexity_tier"] for r in rows)
    repo_counts = Counter(r["repo"] for r in rows)
    print(f"\n{'='*50}")
    print(f"  {label}: {len(rows)} total instances")
    print(f"  Tiers: simple={tier_counts.get('simple',0)}  medium={tier_counts.get('medium',0)}  complex={tier_counts.get('complex',0)}")
    print(f"  Repos: {len(repo_counts)} unique")
    top5 = repo_counts.most_common(5)
    for repo, count in top5:
        print(f"    {repo}: {count}")
    print(f"{'='*50}\n")


def main():
    verified_rows = process_dataset("verified", "princeton-nlp/SWE-bench_Verified")
    write_csv(verified_rows, os.path.join(DATA_DIR, "swe_bench_verified.csv"))
    print_summary(verified_rows, "SWE-Bench Verified")

    lite_rows = process_dataset("lite", "princeton-nlp/SWE-bench_Lite")
    write_csv(lite_rows, os.path.join(DATA_DIR, "swe_bench_lite.csv"))
    print_summary(lite_rows, "SWE-Bench Lite")

    seen_ids = set()
    merged = []
    for r in verified_rows + lite_rows:
        if r["instance_id"] not in seen_ids:
            seen_ids.add(r["instance_id"])
            merged.append(r)

    merged.sort(key=lambda r: ({"simple": 0, "medium": 1, "complex": 2}[r["complexity_tier"]], r["repo"], r["instance_id"]))

    write_csv(merged, os.path.join(DATA_DIR, "swe_bench_stratified.csv"))
    print_summary(merged, "Merged (deduped)")

    print("Done. Next step: open data/swe_bench_stratified.csv and pick 15-20 tasks per tier.")


if __name__ == "__main__":
    main()
