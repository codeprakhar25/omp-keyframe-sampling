"""Fetch BigCodeBench-Hard, filter, save 30 tasks to data/tasks.jsonl."""
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from src.config import CFG

DATA_DIR = Path(__file__).parent.parent / "data"


def filter_tasks(tasks: list[dict], n: int, min_words: int) -> list[dict]:
    """Filter tasks by minimum prompt word count and required fields.

    Args:
        tasks: List of task dicts from the dataset.
        n: Maximum number of tasks to return.
        min_words: Minimum number of words required in instruct_prompt.

    Returns:
        Filtered list of up to n tasks that meet all criteria.
    """
    out = []
    for t in tasks:
        if len(t["instruct_prompt"].split()) < min_words:
            continue
        if not t.get("canonical_solution") or not t.get("test"):
            continue
        out.append(t)
        if len(out) == n:
            break
    return out


def main() -> None:
    """Load BigCodeBench-Hard, filter tasks, and write to data/tasks.jsonl."""
    ds = load_dataset(CFG.dataset_name, split="v0.1.4")
    tasks = [dict(t) for t in ds]
    selected = filter_tasks(tasks, n=CFG.n_tasks_primary, min_words=CFG.min_prompt_words)
    if len(selected) < CFG.n_tasks_primary:
        raise RuntimeError(
            f"Only {len(selected)} tasks met filter; need {CFG.n_tasks_primary}"
        )
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "tasks.jsonl"
    with out_path.open("w") as f:
        for t in selected:
            record = {
                "task_id": t["task_id"],
                "instruct_prompt": t["instruct_prompt"],
                "canonical_solution": t["canonical_solution"],
                "test": t["test"],
                "entry_point": t.get("entry_point"),
                "content_hash": hashlib.sha256(
                    t["instruct_prompt"].encode()
                ).hexdigest()[:16],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(selected)} tasks to {out_path}")


if __name__ == "__main__":
    main()
