"""Per-turn instrumentation logger.

Captures for every agent turn:
  - input_tokens, output_tokens, cache_read_tokens
  - tool calls (name, arguments summary)
  - files read and written
  - wall-clock duration
  - the agent's text response
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result_preview: str = ""    # first 200 chars of result


@dataclass
class TurnRecord:
    """Metrics for a single agent turn (one API call / one CLI invocation)."""
    turn_index: int
    start_time: float = 0.0
    end_time: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    files_written: set[str] = field(default_factory=set)

    agent_text: str = ""
    stop_reason: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "duration_s": round(self.duration_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "tool_call_count": self.tool_call_count,
            "tool_calls": [
                {"name": tc.name, "args_keys": list(tc.arguments.keys())}
                for tc in self.tool_calls
            ],
            "files_read": sorted(self.files_read),
            "files_written": sorted(self.files_written),
            "stop_reason": self.stop_reason,
            "agent_text_preview": self.agent_text[:300] if self.agent_text else "",
        }


@dataclass
class RunLog:
    """Aggregated log for an entire experiment run (all turns of one task)."""
    run_id: str = ""
    task_id: str = ""
    strategy: str = ""
    agent: str = ""
    repeat_index: int = 0

    turns: list[TurnRecord] = field(default_factory=list)

    run_start_time: float = 0.0
    run_end_time: float = 0.0

    final_diff: str = ""
    task_passed: bool | None = None
    error: str | None = None
    eval_method: str = ""      # "tests" | "line_overlap"
    test_summary: str = ""     # pass/fail/err counts when eval_method == tests

    @property
    def total_duration_s(self) -> float:
        return self.run_end_time - self.run_start_time

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(t.cache_read_tokens for t in self.turns)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(t.cache_creation_tokens for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        return sum(t.tool_call_count for t in self.turns)

    @property
    def unique_files_read(self) -> set[str]:
        result: set[str] = set()
        for t in self.turns:
            result |= t.files_read
        return result

    @property
    def unique_files_written(self) -> set[str]:
        result: set[str] = set()
        for t in self.turns:
            result |= t.files_written
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "strategy": self.strategy,
            "agent": self.agent,
            "repeat_index": self.repeat_index,
            "total_turns": len(self.turns),
            "total_duration_s": round(self.total_duration_s, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_tool_calls": self.total_tool_calls,
            "unique_files_read": len(self.unique_files_read),
            "unique_files_written": len(self.unique_files_written),
            "task_passed": self.task_passed,
            "error": self.error,
            "eval_method": self.eval_method,
            "test_summary": self.test_summary,
        }

    def to_dict(self) -> dict[str, Any]:
        s = self.summary()
        s["turns"] = [t.to_dict() for t in self.turns]
        s["final_diff_preview"] = self.final_diff[:500] if self.final_diff else ""
        s["files_read_list"] = sorted(self.unique_files_read)
        s["files_written_list"] = sorted(self.unique_files_written)
        return s


class TurnTimer:
    """Context manager to time a turn and populate a TurnRecord."""
    def __init__(self, record: TurnRecord):
        self.record = record

    def __enter__(self) -> TurnRecord:
        self.record.start_time = time.time()
        return self.record

    def __exit__(self, *exc) -> None:
        self.record.end_time = time.time()
