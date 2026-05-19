"""Run the messifier on each task in tasks.jsonl. Writes data/messy.jsonl.
Idempotent: skips tasks whose task_id already appears in messy.jsonl."""
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.config import CFG
from src.llm import call_anthropic, get_anthropic_client

DATA_DIR = Path(__file__).parent.parent / "data"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: Path, record: dict) -> None:
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    load_dotenv()
    client = get_anthropic_client()
    system = (PROMPTS_DIR / "messifier.md").read_text()
    tasks = load_jsonl(DATA_DIR / "tasks.jsonl")
    existing = {r["task_id"] for r in load_jsonl(DATA_DIR / "messy.jsonl")}
    out_path = DATA_DIR / "messy.jsonl"
    for t in tqdm(tasks, desc="messify"):
        if t["task_id"] in existing:
            continue
        result = call_anthropic(
            client=client,
            model=CFG.messifier,
            system=system,
            user=t["instruct_prompt"],
            temperature=CFG.temperature,
            max_tokens=CFG.max_output_tokens,
        )
        append_jsonl(out_path, {
            "task_id": t["task_id"],
            "original_prompt": t["instruct_prompt"],
            "messy_prompt": result["text"],
            "messifier_input_tokens": result["input_tokens"],
            "messifier_output_tokens": result["output_tokens"],
        })


if __name__ == "__main__":
    main()
