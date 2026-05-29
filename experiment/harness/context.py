"""Context injection strategies.

Three modes:
  (A) NONE       — no context file, system prompt only
  (B) ALWAYS_ON  — full AGENTS.md content prepended to system prompt every turn
  (D) SELECTIVE  — wiki topic files available; agent searches with a tool
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .config import ContextStrategy


SYSTEM_BASE = (
    "You are a software engineer tasked with making changes to a codebase. "
    "You have access to tools to read files, write files, run shell commands, "
    "and search the codebase. The user message describes a bug or feature request. "
    "Your job is to make the necessary code changes to resolve the issue. "
    "You MUST actually edit files using the write_file tool to implement the fix. "
    "Do not just analyze or describe what should be done — make the changes. "
    "When you have made all necessary changes, say DONE and stop."
)


class ContextInjector:
    """Manages context injection for a given strategy."""

    def __init__(
        self,
        strategy: ContextStrategy,
        agents_md_content: str = "",
        wiki_dir: str | None = None,
    ):
        self.strategy = strategy
        self.agents_md_content = agents_md_content
        self.wiki_dir = wiki_dir
        self._wiki_index: dict[str, str] | None = None

    def get_system_prompt(self) -> str:
        if self.strategy == ContextStrategy.NONE:
            return SYSTEM_BASE

        if self.strategy == ContextStrategy.ALWAYS_ON:
            return (
                f"{SYSTEM_BASE}\n\n"
                f"--- REPOSITORY CONTEXT (AGENTS.md) ---\n"
                f"{self.agents_md_content}\n"
                f"--- END REPOSITORY CONTEXT ---"
            )

        if self.strategy == ContextStrategy.SELECTIVE:
            topics = self._list_wiki_topics()
            topic_list = "\n".join(f"  - {t}" for t in topics) if topics else "  (none available)"
            return (
                f"{SYSTEM_BASE}\n\n"
                f"Repository context is available as searchable topic files. "
                f"Use the `search_wiki` tool to retrieve relevant context before making changes.\n"
                f"Available topics:\n{topic_list}"
            )

        return SYSTEM_BASE

    def get_extra_tools(self) -> list[dict[str, Any]]:
        """Return additional tool definitions for the selective strategy."""
        if self.strategy != ContextStrategy.SELECTIVE:
            return []

        return [
            {
                "name": "search_wiki",
                "description": (
                    "Search the repository wiki for context on a topic. "
                    "Returns the content of the matching topic file. "
                    "Use this before making changes to understand repo conventions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Topic name or keyword to search for (e.g. 'testing', 'architecture', 'build commands')",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

    def handle_wiki_search(self, query: str) -> str:
        """Execute a wiki search and return matching content."""
        if self.strategy != ContextStrategy.SELECTIVE or not self.wiki_dir:
            return "Wiki not available in this context mode."

        index = self._get_wiki_index()
        query_lower = query.lower()

        exact = index.get(query_lower)
        if exact:
            return exact

        matches = []
        for topic, content in index.items():
            if query_lower in topic or any(w in topic for w in query_lower.split()):
                matches.append((topic, content))

        if not matches:
            for topic, content in index.items():
                if query_lower in content.lower():
                    matches.append((topic, content))

        if not matches:
            return f"No wiki entries found for '{query}'. Available topics: {', '.join(sorted(index.keys()))}"

        results = []
        for topic, content in matches[:3]:
            results.append(f"## {topic}\n{content}")
        return "\n\n---\n\n".join(results)

    def _list_wiki_topics(self) -> list[str]:
        return sorted(self._get_wiki_index().keys())

    def _get_wiki_index(self) -> dict[str, str]:
        if self._wiki_index is not None:
            return self._wiki_index

        self._wiki_index = {}
        if not self.wiki_dir or not os.path.isdir(self.wiki_dir):
            return self._wiki_index

        for fname in os.listdir(self.wiki_dir):
            if not fname.endswith(".md"):
                continue
            topic = fname.replace(".md", "").replace("_", " ").replace("-", " ").lower()
            filepath = os.path.join(self.wiki_dir, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                self._wiki_index[topic] = f.read()

        return self._wiki_index


def split_agents_md_to_wiki(agents_md_content: str, output_dir: str) -> list[str]:
    """Split an AGENTS.md file into per-topic wiki files.

    Splits on markdown H2 (##) headers. Each section becomes a separate
    .md file named after the header, stored in output_dir.

    Returns list of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    sections = re.split(r"(?=^## )", agents_md_content, flags=re.MULTILINE)
    created: list[str] = []

    preamble = sections[0].strip() if sections else ""
    if preamble and not preamble.startswith("## "):
        fname = os.path.join(output_dir, "overview.md")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(preamble)
        created.append(fname)

    for section in sections:
        section = section.strip()
        if not section.startswith("## "):
            continue

        header_match = re.match(r"^## (.+)$", section, re.MULTILINE)
        if not header_match:
            continue

        title = header_match.group(1).strip()
        safe_name = re.sub(r"[^\w\s-]", "", title).strip().lower()
        safe_name = re.sub(r"[\s]+", "_", safe_name)
        if not safe_name:
            safe_name = "section"

        fname = os.path.join(output_dir, f"{safe_name}.md")
        counter = 1
        while os.path.exists(fname):
            fname = os.path.join(output_dir, f"{safe_name}_{counter}.md")
            counter += 1

        with open(fname, "w", encoding="utf-8") as f:
            f.write(section)
        created.append(fname)

    if not created and agents_md_content.strip():
        fname = os.path.join(output_dir, "overview.md")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(agents_md_content)
        created.append(fname)

    return created
