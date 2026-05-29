"""Agent runners — wrappers around production CLI agents.

ClaudeCodeAgent: shells out to `claude --print --output-format stream-json`
parse_codex_output: parses Codex CLI JSONL for future CodexCLIAgent support

Raw API ClaudeAgent has been removed — it could not solve real coding tasks
(hit 30-turn cap on all but trivial 1-liner tasks, pass rate ~5%).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

from .config import AgentBackend, ContextStrategy, RunConfig
from .logger import RunLog, ToolCallRecord, TurnRecord

log = logging.getLogger(__name__)

_FILE_READ_TOOLS = {"Read"}
_FILE_WRITE_TOOLS = {"Edit", "Write", "MultiEdit"}


class ClaudeCodeAgent:
    """Runs Claude Code CLI and parses stream-json output into a RunLog."""

    def __init__(self, run_config: RunConfig, workspace_dir: str):
        self.config = run_config
        self.workspace_dir = workspace_dir
        self.run_log = RunLog(
            run_id=run_config.run_id,
            task_id=run_config.task.task_id,
            strategy=run_config.strategy.value,
            agent="claude_code",
            repeat_index=run_config.repeat_index,
        )

    def run(self, task_prompt: str, append_system: str | None = None) -> RunLog:
        # --bare isolates the agent from the user's global config (no ~/.claude/CLAUDE.md,
        # no SessionStart hooks, no auto-memory, no CLAUDE.md auto-discovery) and forces
        # strict ANTHROPIC_API_KEY auth. This removes contamination and bypasses the OAuth
        # session rate limit. Context is injected ONLY via --append-system-prompt, so the
        # injection strategy is the sole independent variable.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.run_log.error = "ANTHROPIC_API_KEY not set (required for --bare experiment mode)"
            self.run_log.run_start_time = time.time()
            self.run_log.run_end_time = time.time()
            return self.run_log

        cmd = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--bare",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--model", self.config.claude_model,
        ]

        if append_system:
            cmd += ["--append-system-prompt", append_system]

        if self.config.max_cost_usd:
            cmd += ["--max-budget-usd", str(self.config.max_cost_usd)]

        cmd.append(task_prompt)

        log.info("Running ClaudeCodeAgent: cwd=%s model=%s strategy=%s",
                 self.workspace_dir, self.config.claude_model, self.config.strategy.value)

        self.run_log.run_start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.workspace_dir,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            self.run_log.error = "Timeout (3600s)"
            self.run_log.run_end_time = time.time()
            return self.run_log
        except Exception as e:
            self.run_log.error = str(e)
            self.run_log.run_end_time = time.time()
            return self.run_log

        self.run_log.run_end_time = time.time()

        if result.returncode != 0 and not result.stdout.strip():
            self.run_log.error = (result.stderr or "Non-zero exit with no output")[:500]
            log.error("claude CLI failed (rc=%d): %s", result.returncode, self.run_log.error)
            return self.run_log

        self._parse_stream_json(result.stdout)
        return self.run_log

    def _parse_stream_json(self, output: str) -> None:
        # Each API turn emits multiple `assistant` events (one per content block)
        # with identical usage. USER events mark turn boundaries.
        # RESULT event has accurate aggregate totals.
        #
        # Strategy: buffer assistant events per turn, flush on USER or result.
        # Use per-turn input_tokens from streaming; output_tokens from result.

        buf_usage: dict[str, Any] | None = None
        buf_tools: list[ToolCallRecord] = []
        buf_text: list[str] = []
        buf_files_r: set[str] = set()
        buf_files_w: set[str] = set()
        turn_index = 0

        def flush_turn() -> None:
            nonlocal turn_index, buf_usage, buf_tools, buf_text, buf_files_r, buf_files_w
            if buf_usage is None and not buf_tools and not buf_text:
                return
            t = TurnRecord(turn_index=turn_index)
            if buf_usage:
                t.input_tokens = buf_usage.get("input_tokens", 0)
                # output_tokens streaming values are unreliable; will be set from result
                t.cache_read_tokens = buf_usage.get("cache_read_input_tokens", 0)
                t.cache_creation_tokens = buf_usage.get("cache_creation_input_tokens", 0)
            t.tool_calls = buf_tools[:]
            t.agent_text = "\n".join(buf_text)
            t.files_read = buf_files_r.copy()
            t.files_written = buf_files_w.copy()
            t.stop_reason = "end_turn"
            self.run_log.turns.append(t)
            turn_index += 1
            buf_usage = None
            buf_tools = []
            buf_text = []
            buf_files_r = set()
            buf_files_w = set()

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "assistant":
                msg = event.get("message", {})
                usage = msg.get("usage", {})
                # All events in one turn share the same usage — capture once
                if usage and buf_usage is None:
                    buf_usage = usage

                for block in msg.get("content", []):
                    btype = block.get("type")
                    if btype == "text":
                        buf_text.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        buf_tools.append(ToolCallRecord(
                            name=tool_name,
                            arguments=tool_input,
                        ))
                        fpath = tool_input.get("file_path", "")
                        if fpath:
                            bname = os.path.basename(fpath)
                            if tool_name in _FILE_READ_TOOLS:
                                buf_files_r.add(bname)
                            elif tool_name in _FILE_WRITE_TOOLS:
                                buf_files_w.add(bname)

            elif etype == "user":
                # Tool results returned → end of one API turn
                flush_turn()

            elif etype == "result":
                flush_turn()  # flush final assistant turn
                if event.get("is_error") or event.get("subtype") == "error":
                    err = event.get("result", "")
                    if err and not self.run_log.error:
                        self.run_log.error = err[:500]
                    log.warning("claude CLI result error: %s", err[:200])
                # Distribute output tokens from result across turns proportionally
                result_usage = event.get("usage", {})
                total_out = result_usage.get("output_tokens", 0)
                if self.run_log.turns and total_out:
                    per_turn = total_out // len(self.run_log.turns)
                    for t in self.run_log.turns:
                        t.output_tokens = per_turn


def parse_codex_output(raw_output: str) -> RunLog:
    """Parse Codex CLI JSONL output into a RunLog.

    Codex emits one JSON per line. turn.completed events carry usage data.
    """
    run_log = RunLog(agent="codex")

    lines = raw_output.strip().splitlines()
    turn_index = 0
    current_turn: TurnRecord | None = None

    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "turn.started":
            current_turn = TurnRecord(turn_index=turn_index)
            current_turn.start_time = time.time()

        elif etype == "item.completed" and current_turn is not None:
            item = event.get("item", {})
            item_type = item.get("type", "")

            if item_type == "command_execution":
                current_turn.tool_calls.append(ToolCallRecord(
                    name="run_bash",
                    arguments={"command": item.get("command", "")},
                    result_preview=item.get("aggregated_output", "")[:200],
                ))
            elif item_type == "file_change":
                for change in item.get("changes", []):
                    path = change.get("path", "")
                    if path:
                        current_turn.files_written.add(os.path.basename(path))
            elif item_type == "file_read":
                path = item.get("path", "")
                if path:
                    current_turn.files_read.add(os.path.basename(path))
            elif item_type == "agent_message":
                current_turn.agent_text = item.get("text", "")

        elif etype == "turn.completed":
            if current_turn is not None:
                current_turn.end_time = time.time()
                usage = event.get("usage", {})
                current_turn.input_tokens = usage.get("input_tokens", 0)
                current_turn.output_tokens = usage.get("output_tokens", 0)
                current_turn.cache_read_tokens = usage.get("cached_input_tokens", 0)
                current_turn.stop_reason = "turn_completed"
                run_log.turns.append(current_turn)
                turn_index += 1
                current_turn = None

    return run_log
