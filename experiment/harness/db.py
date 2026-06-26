"""SQLite results storage.

Stores per-run summaries and per-turn details for analysis.
Schema designed for easy pandas queries: each row is one turn,
with run-level metadata denormalized for convenience.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .logger import RunLog


def _b(v: bool | None) -> int | None:
    """bool -> 0/1, None stays None (for nullable INTEGER columns)."""
    return None if v is None else (1 if v else 0)


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    repo            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    agent           TEXT NOT NULL,
    model           TEXT NOT NULL,
    repeat_index    INTEGER NOT NULL,
    total_turns     INTEGER,
    total_duration_s REAL,
    total_input_tokens  INTEGER,
    total_output_tokens INTEGER,
    total_cache_read_tokens INTEGER,
    total_cache_creation_tokens INTEGER,
    total_tool_calls    INTEGER,
    unique_files_read   INTEGER,
    unique_files_written INTEGER,
    task_passed     INTEGER,
    error           TEXT,
    final_diff      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    duration_s      REAL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    tool_call_count INTEGER,
    tool_calls_json TEXT,
    files_read_json TEXT,
    files_written_json TEXT,
    stop_reason     TEXT,
    agent_text_preview TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy);
"""


class ResultsDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")  # 60s — tolerate concurrent writers
        self.conn.executescript(SCHEMA)
        # Migrate: add columns added after initial schema
        self._ensure_columns("runs", {
            "total_cache_creation_tokens": "INTEGER",
            "eval_method": "TEXT",
            "test_summary": "TEXT",
            "total_reasoning_tokens": "INTEGER",
            # Reward-hacking study
            "visible_passed": "INTEGER",
            "heldout_passed": "INTEGER",
            "delta": "REAL",
            "integrity_hacked": "INTEGER",
            "hack_types": "TEXT",
            "opportunity_tags": "TEXT",
        })
        self._ensure_columns("turns", {
            "reasoning_tokens": "INTEGER",
        })

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        for col, coltype in columns.items():
            if col not in existing:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                except sqlite3.OperationalError as e:
                    # Concurrent writers (3 repo procs open the same DB at once) can
                    # race the check-then-ALTER: another proc added the column between
                    # our PRAGMA and ALTER. "duplicate column name" → already done, fine.
                    if "duplicate column name" not in str(e):
                        raise
        self.conn.commit()

    def save_run(self, run_log: RunLog, run_config_dict: dict[str, Any] | None = None) -> None:
        model = ""
        repo = ""
        if run_config_dict:
            model = run_config_dict.get("model", "")
            repo = run_config_dict.get("repo", "")

        summary = run_log.summary()

        self.conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, task_id, repo, strategy, agent, model, repeat_index,
                total_turns, total_duration_s, total_input_tokens, total_output_tokens,
                total_cache_read_tokens, total_cache_creation_tokens, total_tool_calls,
                unique_files_read, unique_files_written, task_passed, error, final_diff,
                eval_method, test_summary, total_reasoning_tokens,
                visible_passed, heldout_passed, delta, integrity_hacked,
                hack_types, opportunity_tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
            (
                run_log.run_id,
                run_log.task_id,
                repo,
                run_log.strategy,
                run_log.agent,
                model,
                run_log.repeat_index,
                summary["total_turns"],
                summary["total_duration_s"],
                summary["total_input_tokens"],
                summary["total_output_tokens"],
                summary["total_cache_read_tokens"],
                summary["total_cache_creation_tokens"],
                summary["total_tool_calls"],
                summary["unique_files_read"],
                summary["unique_files_written"],
                1 if run_log.task_passed else (0 if run_log.task_passed is False else None),
                run_log.error,
                run_log.final_diff[:20000] if run_log.final_diff else None,
                getattr(run_log, "eval_method", ""),
                getattr(run_log, "test_summary", ""),
                summary["total_reasoning_tokens"],
                _b(run_log.visible_passed),
                _b(run_log.heldout_passed),
                run_log.delta,
                _b(run_log.integrity_hacked),
                json.dumps(run_log.hack_types) if run_log.hack_types else None,
                json.dumps(run_log.opportunity_tags) if run_log.opportunity_tags else None,
            ),
        )

        self.conn.execute("DELETE FROM turns WHERE run_id = ?", (run_log.run_id,))

        for turn in run_log.turns:
            td = turn.to_dict()
            self.conn.execute(
                """INSERT INTO turns
                   (run_id, turn_index, duration_s, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens, tool_call_count,
                    tool_calls_json, files_read_json, files_written_json,
                    stop_reason, agent_text_preview, reasoning_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_log.run_id,
                    td["turn_index"],
                    td["duration_s"],
                    td["input_tokens"],
                    td["output_tokens"],
                    td["cache_read_tokens"],
                    td["cache_creation_tokens"],
                    td["tool_call_count"],
                    json.dumps(td["tool_calls"]),
                    json.dumps(td["files_read"]),
                    json.dumps(td["files_written"]),
                    td["stop_reason"],
                    td["agent_text_preview"],
                    td["reasoning_tokens"],
                ),
            )

        self.conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def get_runs_for_task(self, task_id: str) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE task_id = ? ORDER BY strategy, repeat_index",
            (task_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_comparison_table(self) -> list[dict[str, Any]]:
        """Get aggregated metrics grouped by task_id + strategy for quick comparison."""
        cur = self.conn.execute("""
            SELECT
                task_id, repo, strategy, agent, model,
                COUNT(*) as n_runs,
                AVG(total_turns) as avg_turns,
                AVG(total_duration_s) as avg_duration_s,
                AVG(total_input_tokens) as avg_input_tokens,
                AVG(total_output_tokens) as avg_output_tokens,
                AVG(total_cache_read_tokens) as avg_cache_tokens,
                AVG(total_tool_calls) as avg_tool_calls,
                AVG(unique_files_read) as avg_files_read,
                AVG(unique_files_written) as avg_files_written,
                SUM(CASE WHEN task_passed = 1 THEN 1 ELSE 0 END) as pass_count
            FROM runs
            GROUP BY task_id, strategy, agent
            ORDER BY task_id, strategy
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
