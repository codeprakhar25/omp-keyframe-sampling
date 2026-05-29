"""Standalone CLI to generate wiki topic files from an AGENTS.md.

Usage:
    python -m harness.generate_wiki --agents-md /path/to/AGENTS.md --output-dir wiki/repo_name/
    python -m harness.generate_wiki --repo-url https://github.com/firebase/firebase-admin-python --output-dir wiki/firebase__firebase-admin-python/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from .context import split_agents_md_to_wiki


def fetch_agents_md_from_repo(repo_url: str) -> str:
    """Fetch AGENTS.md content from a repo using GitHub API or shallow clone."""
    repo_path = repo_url.replace("https://github.com/", "").replace(".git", "")

    for filename in ["AGENTS.md", "CLAUDE.md"]:
        try:
            result = subprocess.run(
                ["gh", "api", f"/repos/{repo_path}/contents/{filename}",
                 "--header", "Accept: application/vnd.github.raw+json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, tmpdir],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Error cloning: {e}", file=sys.stderr)
            return ""

        for name in ["AGENTS.md", "CLAUDE.md"]:
            path = os.path.join(tmpdir, name)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wiki topic files from AGENTS.md")
    parser.add_argument("--agents-md", help="Path to local AGENTS.md file")
    parser.add_argument("--repo-url", help="GitHub repo URL to fetch AGENTS.md from")
    parser.add_argument("--output-dir", required=True, help="Directory to write topic files into")
    args = parser.parse_args()

    if args.agents_md:
        with open(args.agents_md, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.repo_url:
        content = fetch_agents_md_from_repo(args.repo_url)
    else:
        print("Provide either --agents-md or --repo-url", file=sys.stderr)
        sys.exit(1)

    if not content.strip():
        print("No AGENTS.md content found.", file=sys.stderr)
        sys.exit(1)

    created = split_agents_md_to_wiki(content, args.output_dir)
    print(f"Created {len(created)} topic files in {args.output_dir}:")
    for path in created:
        size = os.path.getsize(path)
        print(f"  {os.path.basename(path)} ({size} bytes)")


if __name__ == "__main__":
    main()
