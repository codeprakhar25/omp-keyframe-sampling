"""Run the reformulator on each messy prompt. Writes data/reformulated.jsonl.
Idempotent: skips task_ids already present in output."""
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
    system = (PROMPTS_DIR / "reformulator.md").read_text()
    messy = load_jsonl(DATA_DIR / "messy.jsonl")
    existing = {r["task_id"] for r in load_jsonl(DATA_DIR / "reformulated.jsonl")}
    out_path = DATA_DIR / "reformulated.jsonl"
    for m in tqdm(messy, desc="reformulate"):
        if m["task_id"] in existing:
            continue
        result = call_anthropic(
            client=client,
            model=CFG.reformulator,
            system=system,
            user=m["messy_prompt"],
            temperature=CFG.temperature,
            max_tokens=CFG.max_output_tokens,
        )
        append_jsonl(out_path, {
            "task_id": m["task_id"],
            "messy_prompt": m["messy_prompt"],
            "reformulated_prompt": result["text"],
            "reformulator_input_tokens": result["input_tokens"],
            "reformulator_output_tokens": result["output_tokens"],
        })


if __name__ == "__main__":
    main()
