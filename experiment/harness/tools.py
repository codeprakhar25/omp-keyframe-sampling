"""Tool definitions and execution for the Claude API agent.

Provides the core tools a coding agent needs:
  - read_file: read a file from the workspace
  - str_replace: targeted edit — replace old_str with new_str in a file
  - create_file: create a new file (not overwrite)
  - run_bash: execute a shell command
  - search_files: grep/search for patterns
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path relative to the workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from workspace root",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "str_replace",
        "description": "Replace a specific string in a file with a new string. The old_str must match EXACTLY (including whitespace and indentation). Use this for all code edits — it is much more efficient than rewriting entire files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from workspace root",
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find in the file (must match exactly, including whitespace)",
                },
                "new_str": {
                    "type": "string",
                    "description": "The string to replace old_str with",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "create_file",
        "description": "Create a new file with the given content. Fails if the file already exists — use str_replace to edit existing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from workspace root",
                },
                "content": {
                    "type": "string",
                    "description": "The full content for the new file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_bash",
        "description": "Execute a bash command in the workspace directory. Use for running tests, build commands, git operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60, max 300)",
                    "default": 60,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for a pattern in files using grep. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (relative to workspace root, default '.')",
                    "default": ".",
                },
                "include": {
                    "type": "string",
                    "description": "File glob pattern to include (e.g. '*.py', '*.js')",
                },
            },
            "required": ["pattern"],
        },
    },
]

MAX_OUTPUT_CHARS = 15_000
BASH_TIMEOUT_MAX = 300


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    workspace_dir: str,
) -> tuple[str, set[str], set[str]]:
    """Execute a tool and return (result_text, files_read, files_written)."""

    files_read: set[str] = set()
    files_written: set[str] = set()

    if name == "read_file":
        result = _read_file(arguments, workspace_dir)
        path = arguments.get("path", "")
        if path:
            files_read.add(path)

    elif name == "str_replace":
        result = _str_replace(arguments, workspace_dir)
        path = arguments.get("path", "")
        if path:
            files_read.add(path)
            files_written.add(path)

    elif name == "create_file":
        result = _create_file(arguments, workspace_dir)
        path = arguments.get("path", "")
        if path:
            files_written.add(path)

    elif name == "write_file":
        result = _write_file(arguments, workspace_dir)
        path = arguments.get("path", "")
        if path:
            files_written.add(path)

    elif name == "run_bash":
        result = _run_bash(arguments, workspace_dir)

    elif name == "search_files":
        result = _search_files(arguments, workspace_dir)

    else:
        result = f"Unknown tool: {name}"

    return result, files_read, files_written


def _read_file(args: dict[str, Any], workspace: str) -> str:
    path = args.get("path", "")
    full_path = os.path.join(workspace, path)

    if not os.path.isfile(full_path):
        return f"Error: File not found: {path}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    if len(content) > MAX_OUTPUT_CHARS:
        return content[:MAX_OUTPUT_CHARS] + f"\n\n[Truncated — file is {len(content)} chars total]"
    return content


def _str_replace(args: dict[str, Any], workspace: str) -> str:
    path = args.get("path", "")
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")
    full_path = os.path.join(workspace, path)

    if not os.path.isfile(full_path):
        return f"Error: File not found: {path}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    count = content.count(old_str)
    if count == 0:
        snippet = content[:2000] if len(content) > 2000 else content
        return f"Error: old_str not found in {path}. File content preview:\n{snippet}"
    if count > 1:
        return f"Error: old_str found {count} times in {path}. Make it more specific to match exactly once."

    new_content = content.replace(old_str, new_str, 1)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully replaced text in {path} ({len(old_str)} chars → {len(new_str)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def _create_file(args: dict[str, Any], workspace: str) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    full_path = os.path.join(workspace, path)

    if ".." in path or path.startswith("/"):
        return f"Error: Path must be relative and within workspace: {path}"

    if os.path.exists(full_path):
        return f"Error: File already exists: {path}. Use str_replace to edit existing files."

    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created {path} ({len(content)} chars)"
    except Exception as e:
        return f"Error creating file: {e}"


def _write_file(args: dict[str, Any], workspace: str) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    full_path = os.path.join(workspace, path)

    if ".." in path or path.startswith("/"):
        return f"Error: Path must be relative and within workspace: {path}"

    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _run_bash(args: dict[str, Any], workspace: str) -> str:
    command = args.get("command", "")
    timeout = min(args.get("timeout", 60), BASH_TIMEOUT_MAX)

    blocked = ["rm -rf /", "rm -rf /*", "mkfs", ":(){", "dd if=/dev"]
    if any(b in command for b in blocked):
        return "Error: Command blocked for safety."

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr

        if not output:
            output = "(no output)"

        output = f"Exit code: {result.returncode}\n{output}"

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n[Truncated — {len(output)} chars total]"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"


def _search_files(args: dict[str, Any], workspace: str) -> str:
    pattern = args.get("pattern", "")
    search_path = args.get("path", ".")
    include = args.get("include", "")

    full_path = os.path.join(workspace, search_path)
    if not os.path.exists(full_path):
        return f"Error: Path not found: {search_path}"

    cmd = ["grep", "-rn", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.extend([pattern, full_path])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        output = result.stdout or "(no matches)"
        output = output.replace(workspace + "/", "")

        if len(output) > MAX_OUTPUT_CHARS:
            lines = output.split("\n")
            output = "\n".join(lines[:200]) + f"\n[Truncated — {len(lines)} total matches]"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s"
    except Exception as e:
        return f"Error searching: {e}"
