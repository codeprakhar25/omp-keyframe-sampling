# tests/test_select_tasks.py
import json
from pathlib import Path
from src.select_tasks import filter_tasks

def test_filter_tasks_returns_n_with_required_fields():
    fake_tasks = [
        {"task_id": f"BCB/{i}", "instruct_prompt": "x " * 100, "canonical_solution": "def f(): pass", "test": "assert True"}
        for i in range(50)
    ]
    selected = filter_tasks(fake_tasks, n=30, min_words=80)
    assert len(selected) == 30
    for t in selected:
        assert "task_id" in t
        assert "instruct_prompt" in t
        assert "canonical_solution" in t
        assert "test" in t
        assert len(t["instruct_prompt"].split()) >= 80

def test_filter_tasks_skips_short_prompts():
    fake_tasks = [
        {"task_id": "BCB/0", "instruct_prompt": "short", "canonical_solution": "def f(): pass", "test": "assert True"},
        {"task_id": "BCB/1", "instruct_prompt": "word " * 100, "canonical_solution": "def f(): pass", "test": "assert True"},
    ]
    selected = filter_tasks(fake_tasks, n=2, min_words=80)
    assert len(selected) == 1
    assert selected[0]["task_id"] == "BCB/1"
