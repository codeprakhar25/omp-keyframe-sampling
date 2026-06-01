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

# SAFETY: tool-pattern denials passed to `claude --disallowedTools`. Deny-side
# patterns are honored even under --dangerously-skip-permissions (allow-side has
# a known bypass bug; deny-side is safe). Layer 2 of the push/PR block.
# See PILOT_HANDOFF incident #12 (agent opened real PR #953).
_DENY_TOOLS = [
    "Bash(git push:*)",
    "Bash(git commit:*)",
    "Bash(git remote:*)",
    "Bash(gh:*)",
    "Bash(git request-pull:*)",
]

# Substrings that, if seen in a Bash tool_use, mean the agent is TRYING to push /
# PR despite the block — log loudly so it's visible during manual runs.
_PUSH_ALARM = ("git push", "gh pr", "gh repo", "gh api", "git commit", "request-pull")


def _live_enabled() -> bool:
    return os.environ.get("EXP_LIVE", "").lower() in ("1", "true", "yes", "on")


def _pretty_event(event: dict[str, Any]) -> list[str]:
    """One-or-more short human lines for a stream-json event (live manual view)."""
    out: list[str] = []
    etype = event.get("type")
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                txt = " ".join(block.get("text", "").split())
                if txt:
                    out.append(f"  💬 {txt[:140]}")
            elif bt == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if name == "Bash":
                    arg = inp.get("command", "")
                elif name in _FILE_READ_TOOLS or name in _FILE_WRITE_TOOLS:
                    arg = inp.get("file_path", "")
                else:
                    arg = json.dumps(inp)[:80]
                arg = " ".join(str(arg).split())
                alarm = any(s in arg for s in _PUSH_ALARM)
                icon = "🚨" if alarm else "🔧"
                out.append(f"  {icon} {name}: {arg[:140]}")
                if alarm:
                    out.append(f"  🚨🚨 PUSH/PR ATTEMPT BLOCKED-BY-DENY: {name}: {arg[:200]}")
    elif etype == "result":
        sub = event.get("subtype", "")
        cost = event.get("total_cost_usd", "?")
        nturns = event.get("num_turns", "?")
        out.append(f"  ✅ result: {sub} cost=${cost} turns={nturns}")
    return out


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
            # SAFETY layer 2: deny push/PR tooling at the agent level.
            "--disallowedTools", ",".join(_DENY_TOOLS),
        ]

        if append_system:
            cmd += ["--append-system-prompt", append_system]

        if self.config.max_cost_usd:
            cmd += ["--max-budget-usd", str(self.config.max_cost_usd)]

        cmd.append(task_prompt)

        # SAFETY layer 3: strip GitHub auth from the agent's env so even a push
        # that slipped through has no credentials. GIT_TERMINAL_PROMPT=0 stops
        # git from interactively asking for creds (which would hang the run).
        env = os.environ.copy()
        env["GH_TOKEN"] = ""
        env["GITHUB_TOKEN"] = ""
        env["GIT_TERMINAL_PROMPT"] = "0"

        live = _live_enabled()
        log.info("Running ClaudeCodeAgent: cwd=%s model=%s strategy=%s live=%s deny=%s",
                 self.workspace_dir, self.config.claude_model, self.config.strategy.value,
                 live, _DENY_TOOLS)

        # Stream raw stream-json to a per-run file for post-hoc inspection.
        stream_log_path = self._stream_log_path()

        self.run_log.run_start_time = time.time()
        try:
            stdout_text = self._run_streaming(cmd, env, stream_log_path, live)
        except subprocess.TimeoutExpired:
            self.run_log.error = "Timeout (3600s)"
            self.run_log.run_end_time = time.time()
            return self.run_log
        except Exception as e:  # noqa: BLE001
            self.run_log.error = str(e)
            self.run_log.run_end_time = time.time()
            return self.run_log

        self.run_log.run_end_time = time.time()

        if not stdout_text.strip():
            self.run_log.error = "No output from claude CLI"
            log.error("claude CLI produced no output (see %s)", stream_log_path)
            return self.run_log

        self._parse_stream_json(stdout_text)
        return self.run_log

    def _stream_log_path(self) -> str:
        # results/ lives at repo root; workspace is repos/<slug>__work
        root = os.path.dirname(os.path.dirname(self.workspace_dir))
        results_dir = os.path.join(root, "results")
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"{self.run_log.run_id}.stream.jsonl")

    def _run_streaming(self, cmd: list[str], env: dict[str, str],
                       stream_log_path: str, live: bool) -> str:
        """Run claude, tee stdout to a file, optionally print live one-liners.

        Returns full accumulated stdout for parsing. Replaces the old blocking
        subprocess.run so manual runs can watch tool calls in real time.

        WATCHDOG: a blocking read on a stuck stream (e.g. claude's upstream
        socket dies mid-turn) would hang forever — `proc.wait(timeout)` is never
        reached because the for-loop never yields. So we arm an inactivity timer
        that kills the process if NO new output arrives for INACTIVITY_TIMEOUT,
        and an absolute-cap timer. Resets per line. (Regression guard: the old
        subprocess.run(timeout=) is gone.)
        """
        import threading

        # 1800s (was 900): opshin agents run slow compiler tests via Bash that
        # stream nothing for long stretches; 900s risked a false "stuck" kill
        # during a legit long test run. Env-overridable.
        INACTIVITY_TIMEOUT = int(os.environ.get("EXP_INACTIVITY_TIMEOUT", "1800"))
        # 7200s (was 3600): opshin compiler tests are slow; 3600 killed
        # opshin 610 always_on/selective + 605 none mid-run. Pod is dedicated
        # (no laptop to free), so a longer wall-clock cap is safe.
        ABSOLUTE_TIMEOUT = int(os.environ.get("EXP_ABSOLUTE_TIMEOUT", "7200"))

        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=self.workspace_dir, env=env, bufsize=1,
        )
        killed = {"reason": ""}

        def _kill(reason: str) -> None:
            killed["reason"] = reason
            proc.kill()

        abs_timer = threading.Timer(ABSOLUTE_TIMEOUT, _kill, args=(f"absolute {ABSOLUTE_TIMEOUT}s",))
        abs_timer.daemon = True
        abs_timer.start()
        inactivity = threading.Timer(INACTIVITY_TIMEOUT, _kill, args=(f"inactivity {INACTIVITY_TIMEOUT}s",))
        inactivity.daemon = True
        inactivity.start()

        try:
            with open(stream_log_path, "w", encoding="utf-8") as sf:
                for raw in proc.stdout:  # type: ignore[union-attr]
                    inactivity.cancel()
                    inactivity = threading.Timer(INACTIVITY_TIMEOUT, _kill, args=(f"inactivity {INACTIVITY_TIMEOUT}s",))
                    inactivity.daemon = True
                    inactivity.start()
                    sf.write(raw)
                    sf.flush()
                    lines.append(raw)
                    if live:
                        stripped = raw.strip()
                        if not stripped:
                            continue
                        try:
                            event = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        for pl in _pretty_event(event):
                            log.info(pl)
                proc.wait()
        finally:
            abs_timer.cancel()
            inactivity.cancel()

        if killed["reason"]:
            self.run_log.error = f"Killed by watchdog: {killed['reason']} (stuck/over-budget stream)"
            log.error("claude CLI killed by watchdog: %s (parsed %d lines so far)",
                      killed["reason"], len(lines))
            # Return what we got — parser will salvage completed turns.
            return "".join(lines)

        if proc.returncode and not "".join(lines).strip():
            err = (proc.stderr.read() if proc.stderr else "") or "Non-zero exit, no output"
            self.run_log.error = err[:500]
            log.error("claude CLI failed (rc=%s): %s", proc.returncode, err[:300])
        return "".join(lines)

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
