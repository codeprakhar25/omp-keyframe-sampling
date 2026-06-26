"""Experiment configuration and dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ContextStrategy(str, Enum):
    """Context-injection strategies under test."""
    NONE = "none"               # (A) no context file
    ALWAYS_ON = "always_on"     # (B) full AGENTS.md in system prompt every turn
    SELECTIVE = "selective"     # (D) wiki topic files, agent searches as needed
    # Reward-hacking study arms (3-arm dose-response; injected like a benign AGENTS.md):
    NEUTRAL = "neutral"         # informational instructions, NO proxy pressure (ETH over-work control)
    PRESSURE = "pressure"       # outcome/proxy-pressure ("all tests must pass", "keep CI green")


class ComplexityTier(str, Enum):
    SIMPLE = "simple"     # 1 file, < 50 LOC
    MEDIUM = "medium"     # 2-4 files, 50-200 LOC
    COMPLEX = "complex"   # 5+ files or > 200 LOC


class AgentBackend(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"


@dataclass
class TaskConfig:
    """A single task (repo + issue/PR) to run."""
    task_id: str
    repo_full_name: str       # e.g. "firebase/firebase-admin-python"
    repo_url: str
    base_sha: str             # commit to checkout before running
    prompt: str               # the issue/PR description
    gold_diff: str | None = None
    complexity: ComplexityTier = ComplexityTier.MEDIUM
    pr_number: int | None = None

    @property
    def repo_slug(self) -> str:
        return self.repo_full_name.replace("/", "__")


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    task: TaskConfig
    strategy: ContextStrategy
    agent: AgentBackend
    run_id: str = ""
    repeat_index: int = 0

    # Claude-specific. Env-overridable (EXP_CLAUDE_MODEL) so a budget arm can swap
    # Sonnet -> Haiku without code edits (mirrors codex_model).
    claude_model: str = os.environ.get("EXP_CLAUDE_MODEL", "claude-sonnet-4-6")
    claude_max_turns: int = 50
    claude_max_tokens: int = 8192

    # Codex-specific. gpt-5.5 = flagship (ChatGPT-auth only). Env-overridable so a
    # tight-budget / API-key run can fall back to gpt-5.4 (API-capable).
    # sandbox = danger-full-access: codex's workspace-write sandbox uses bubblewrap,
    # which needs unprivileged user namespaces — UNAVAILABLE in most containers
    # (RunPod: "bwrap: No permissions to create a new namespace"), so it fails every
    # agent shell command. We instead rely on the PATH-independent guarantees that
    # actually hold on the egress-locked pod: /etc/hosts GitHub blackhole +
    # scrub_git_remotes + blanked GH tokens + gh/git PATH-shims. REQUIRES the locked
    # pod (verify_lock gate). On a host WITH userns set EXP_CODEX_SANDBOX=workspace-write.
    codex_model: str = os.environ.get("EXP_CODEX_MODEL", "gpt-5.5")
    codex_sandbox: str = os.environ.get("EXP_CODEX_SANDBOX", "danger-full-access")
    codex_home: str | None = os.environ.get("EXP_CODEX_HOME") or None

    # Paths — set by ExperimentConfig
    workspace_dir: str = ""
    agents_md_path: str | None = None
    wiki_dir: str | None = None

    # Cost cap. 6.0 (was 4.0): opshin cells are turn-heavy (610 hit 101 turns);
    # $4 risked killing them on budget before finishing. Still a runaway guard,
    # just higher headroom. Env-overridable for tight-budget runs.
    max_cost_usd: float = float(os.environ.get("EXP_MAX_COST_USD", "6.0"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "repo": self.task.repo_full_name,
            "strategy": self.strategy.value,
            "agent": self.agent.value,
            "run_id": self.run_id,
            "repeat_index": self.repeat_index,
            "model": self.claude_model if self.agent == AgentBackend.CLAUDE else self.codex_model,
        }


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    repos_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "repos")
    wiki_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "wiki")
    db_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "experiment.db")

    strategies: list[ContextStrategy] = field(
        default_factory=lambda: [ContextStrategy.NONE, ContextStrategy.ALWAYS_ON, ContextStrategy.SELECTIVE]
    )
    repeats: int = 3

    def ensure_dirs(self) -> None:
        for d in [self.results_dir, self.repos_dir, self.wiki_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def replication_data_dir(self) -> Path:
        return self.data_dir / "Dataset_and_Replication_Package" / "Data"

    @property
    def pr_prompts_dir(self) -> Path:
        return self.replication_data_dir / "pr_prompts"

    @property
    def agents_md_dir(self) -> Path:
        return self.replication_data_dir / "filtered_agents_md_files"
