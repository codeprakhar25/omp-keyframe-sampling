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


def stream_subprocess(cmd, env, cwd, stream_log_path, live, pretty_fn, run_log,
                      max_turns=0, turn_event_type=None):
    """Run a CLI agent, tee stdout JSONL to a file, optionally print live lines.

    Shared by ClaudeCodeAgent (claude stream-json) and CodexCLIAgent (codex
    --json). Returns full accumulated stdout for parsing.

    WATCHDOG: a blocking read on a stuck stream would hang forever (the for-loop
    never yields, so `proc.wait(timeout)` is never reached). So we arm an
    inactivity timer that kills the process if NO new output arrives for
    INACTIVITY_TIMEOUT, plus an absolute-cap timer. Resets per line.

    Optional `max_turns`: count parsed events whose `type == turn_event_type` and
    kill once the count exceeds max_turns — replaces the per-agent budget flag for
    agents (Codex) that have none.
    """
    import threading

    INACTIVITY_TIMEOUT = int(os.environ.get("EXP_INACTIVITY_TIMEOUT", "1800"))
    ABSOLUTE_TIMEOUT = int(os.environ.get("EXP_ABSOLUTE_TIMEOUT", "7200"))

    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env, bufsize=1,
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

    parse_each = live or (max_turns and turn_event_type)
    turns_seen = 0
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
                if parse_each:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if live:
                        for pl in pretty_fn(event):
                            log.info(pl)
                    if max_turns and turn_event_type and event.get("type") == turn_event_type:
                        turns_seen += 1
                        if turns_seen > max_turns:
                            _kill(f"max_turns {max_turns}")
                            break
            proc.wait()
    finally:
        abs_timer.cancel()
        inactivity.cancel()

    if killed["reason"]:
        run_log.error = f"Killed by watchdog: {killed['reason']} (stuck/over-budget stream)"
        log.error("CLI killed by watchdog: %s (parsed %d lines so far)",
                  killed["reason"], len(lines))
        return "".join(lines)

    if proc.returncode and not "".join(lines).strip():
        err = (proc.stderr.read() if proc.stderr else "") or "Non-zero exit, no output"
        run_log.error = err[:500]
        log.error("CLI failed (rc=%s): %s", proc.returncode, err[:300])
    return "".join(lines)


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
        # Claude Code refuses --dangerously-skip-permissions when running as root
        # (rc=1, "cannot be used with root/sudo privileges"). On the locked pod we
        # ARE root inside a sandbox (egress-locked, no creds, scrubbed remotes), so
        # mark it a sandbox to allow the flag. Without this the agent never starts.
        env["IS_SANDBOX"] = "1"

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
        """Run claude via the shared watchdog streamer (see stream_subprocess)."""
        return stream_subprocess(
            cmd, env, self.workspace_dir, stream_log_path, live,
            _pretty_event, self.run_log,
        )

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


_CODEX_TURN_EVENT = "turn.completed"


def _pretty_codex_event(event: dict[str, Any]) -> list[str]:
    """One-or-more short human lines for a codex --json event (live manual view).

    Mirrors _pretty_event (claude). Alarms 🚨 on push/PR command substrings — the
    PreToolUse deny-hook blocks them, this just makes an attempt visible in logs.
    """
    out: list[str] = []
    etype = event.get("type", "")
    if etype in ("item.completed", "item.started"):
        item = event.get("item", {})
        it = item.get("type", "")
        if it == "command_execution":
            cmd = " ".join(str(item.get("command", "")).split())
            alarm = any(s in cmd for s in _PUSH_ALARM)
            out.append(f"  {'🚨' if alarm else '🔧'} run_bash: {cmd[:140]}")
            if alarm:
                out.append(f"  🚨🚨 PUSH/PR ATTEMPT (deny-hook should block): {cmd[:200]}")
        elif it == "file_change" and etype == "item.completed":
            paths = [c.get("path", "") for c in item.get("changes", [])]
            out.append(f"  ✏️  file_change: {', '.join(p for p in paths if p)[:140]}")
        elif it == "agent_message" and etype == "item.completed":
            txt = " ".join(str(item.get("text", "")).split())
            if txt:
                out.append(f"  💬 {txt[:140]}")
    elif etype == "turn.completed":
        u = event.get("usage", {})
        out.append(f"  ✅ turn done: in={u.get('input_tokens',0)} "
                   f"out={u.get('output_tokens',0)} cached={u.get('cached_input_tokens',0)} "
                   f"reasoning={u.get('reasoning_output_tokens',0)}")
    elif etype in ("turn.failed", "error"):
        msg = event.get("error", {}).get("message", "") or event.get("message", "")
        out.append(f"  ⚠️  {etype}: {str(msg)[:160]}")
    return out


def parse_codex_output(raw_output: str) -> RunLog:
    """Parse Codex CLI `exec --json` JSONL output into a RunLog.

    Codex emits one JSON per line. A turn opens at `turn.started`, accrues
    `item.completed` events, and closes at `turn.completed` (which carries usage).
    Item types we record: command_execution (→tool), file_change (→files_written),
    agent_message (→text), mcp_tool_call (→tool). reasoning is internal-only.

    Usage maps: cached_input_tokens→cache_read; there is NO cache-creation concept
    in Codex (cache_creation_tokens stays 0). reasoning_output_tokens is folded
    into output_tokens so the portable total (input+output) counts generated work.
    """
    run_log = RunLog(agent="codex")
    turn_index = 0
    current_turn: TurnRecord | None = None

    for line in raw_output.strip().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "turn.started":
            current_turn = TurnRecord(turn_index=turn_index)

        elif etype == "item.completed":
            if current_turn is None:  # tolerate a missing turn.started
                current_turn = TurnRecord(turn_index=turn_index)
            item = event.get("item", {})
            item_type = item.get("type", "")
            if item_type == "command_execution":
                current_turn.tool_calls.append(ToolCallRecord(
                    name="run_bash",
                    arguments={"command": item.get("command", "")},
                    result_preview=str(item.get("aggregated_output", ""))[:200],
                ))
            elif item_type == "mcp_tool_call":
                current_turn.tool_calls.append(ToolCallRecord(
                    name=item.get("tool", "mcp"),
                    arguments=item.get("arguments", {}) or {},
                ))
            elif item_type == "file_change":
                for change in item.get("changes", []):
                    path = change.get("path", "")
                    if path:
                        current_turn.files_written.add(os.path.basename(path))
            elif item_type == "agent_message":
                current_turn.agent_text = item.get("text", "")

        elif etype == "turn.completed":
            if current_turn is None:
                continue
            usage = event.get("usage", {})
            current_turn.input_tokens = usage.get("input_tokens", 0)
            current_turn.output_tokens = (usage.get("output_tokens", 0)
                                          + usage.get("reasoning_output_tokens", 0))
            current_turn.cache_read_tokens = usage.get("cached_input_tokens", 0)
            current_turn.stop_reason = "turn_completed"
            run_log.turns.append(current_turn)
            turn_index += 1
            current_turn = None

    return run_log


class CodexCLIAgent:
    """Runs OpenAI Codex CLI (`codex exec --json`) and parses JSONL into a RunLog.

    Safety parity with ClaudeCodeAgent (and more):
      - CODEX_HOME points to a clean dir carrying our PreToolUse deny-push hook
        (the deny-side analog of claude's --disallowedTools, which codex lacks)
        and NO global AGENTS.md/config (= --bare analog: zero context leak).
      - --sandbox workspace-write confines writes to the workspace AND disables
        agent-command network (extra isolation the claude arm did not have).
      - --ask-for-approval never = autonomous; --ephemeral = no session files.
      - env credential scrub (GH_TOKEN/GITHUB_TOKEN/GIT_TERMINAL_PROMPT) reused.
    Layered with the pod-wide egress lock + scrub_git_remotes (runner) for the
    same defense-in-depth as the claude arm.
    """

    def __init__(self, run_config: RunConfig, workspace_dir: str):
        self.config = run_config
        self.workspace_dir = workspace_dir
        self.run_log = RunLog(
            run_id=run_config.run_id,
            task_id=run_config.task.task_id,
            strategy=run_config.strategy.value,
            agent="codex",
            repeat_index=run_config.repeat_index,
        )

    def _stream_log_path(self) -> str:
        root = os.path.dirname(os.path.dirname(self.workspace_dir))
        results_dir = os.path.join(root, "results")
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"{self.run_log.run_id}.codex.jsonl")

    def _codex_home(self) -> str:
        """Clean CODEX_HOME carrying our deny hook. Defaults to pod/codex_home in
        the repo if present, else a config-supplied path, else a scratch dir.
        Renders hooks.json from hooks.json.template with the absolute home path so
        the PreToolUse deny-hook resolves wherever this runs (laptop or pod)."""
        if getattr(self.config, "codex_home", None):
            home = str(self.config.codex_home)
        else:
            root = os.path.dirname(os.path.dirname(self.workspace_dir))
            repo_default = os.path.join(root, "pod", "codex_home")
            home = repo_default if os.path.isdir(repo_default) \
                else os.path.join(root, "results", "_codex_home")
        os.makedirs(home, exist_ok=True)
        self._render_hooks(home)
        return home

    def _render_hooks(self, home: str) -> None:
        """Materialize hooks.json (absolute deny-script path) from the template."""
        tmpl = os.path.join(home, "hooks.json.template")
        if not os.path.isfile(tmpl):
            log.warning("codex deny-hook template missing at %s — running WITHOUT "
                        "the PreToolUse push-block (egress lock still applies)", tmpl)
            return
        with open(tmpl, encoding="utf-8") as f:
            rendered = f.read().replace("__CODEX_HOME__", home)
        with open(os.path.join(home, "hooks.json"), "w", encoding="utf-8") as f:
            f.write(rendered)

    def run(self, task_prompt: str, append_system: str | None = None) -> RunLog:
        model = getattr(self.config, "codex_model", "gpt-5.5")
        sandbox = getattr(self.config, "codex_sandbox", "workspace-write")
        approval = getattr(self.config, "codex_approval", "never")

        # Codex has no --append-system-prompt; prepend injected context to the
        # prompt (closest parity to the claude arm's system-prompt injection).
        prompt = f"{append_system}\n\n{task_prompt}" if append_system else task_prompt

        cmd = [
            "codex", "exec", "--json",
            "--model", model,
            "--sandbox", sandbox,
            "--ask-for-approval", approval,
            "--ephemeral",
            "--skip-git-repo-check",
            "-C", self.workspace_dir,
            prompt,
        ]

        env = os.environ.copy()
        # SAFETY: strip GitHub creds so any push that slipped through has none,
        # and stop git from interactively prompting (which would hang the run).
        env["GH_TOKEN"] = ""
        env["GITHUB_TOKEN"] = ""
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Clean CODEX_HOME = --bare analog (no global AGENTS.md/config) + carries
        # the PreToolUse deny-push hook.
        env["CODEX_HOME"] = self._codex_home()

        live = _live_enabled()
        log.info("Running CodexCLIAgent: cwd=%s model=%s sandbox=%s approval=%s strategy=%s live=%s home=%s",
                 self.workspace_dir, model, sandbox, approval, self.config.strategy.value,
                 live, env["CODEX_HOME"])

        stream_log_path = self._stream_log_path()
        # No native budget/turn flag in codex → cap turns via the watchdog.
        max_turns = int(os.environ.get("EXP_MAX_TURNS", "0"))

        self.run_log.run_start_time = time.time()
        try:
            out = stream_subprocess(
                cmd, env, self.workspace_dir, stream_log_path, live,
                _pretty_codex_event, self.run_log,
                max_turns=max_turns, turn_event_type=_CODEX_TURN_EVENT,
            )
        except FileNotFoundError:
            self.run_log.error = "codex CLI not found (install: npm i -g @openai/codex)"
            self.run_log.run_end_time = time.time()
            return self.run_log
        except Exception as e:  # noqa: BLE001
            self.run_log.error = str(e)
            self.run_log.run_end_time = time.time()
            return self.run_log
        self.run_log.run_end_time = time.time()

        if not out.strip():
            if not self.run_log.error:
                self.run_log.error = "No output from codex CLI"
            log.error("codex CLI produced no output (see %s)", stream_log_path)
            return self.run_log

        parsed = parse_codex_output(out)
        self.run_log.turns = parsed.turns
        return self.run_log
