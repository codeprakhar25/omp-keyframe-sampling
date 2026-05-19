"""Run the target model for each (task, arm, seed). Append-only writes to results/runs.jsonl.
Idempotent: skips (task_id, arm, seed, model) tuples already present."""
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.config import CFG
from src.llm import call_anthropic, call_openai, get_anthropic_client, get_openai_client

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET_SYSTEM_PROMPT = (
    "You are a precise Python code generator. Read the task and return a single "
    "Python code block containing the complete implementation. Do not include "
    "explanations outside the code block. Do not include test code unless asked."
)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: Path, record: dict) -> None:
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_one(client, provider: str, model: str, task_id: str, arm: str,
            seed: int, prompt: str) -> dict:
    if provider == "anthropic":
        result = call_anthropic(client, model, TARGET_SYSTEM_PROMPT, prompt,
                                CFG.temperature, CFG.max_output_tokens)
    else:
        result = call_openai(client, model, TARGET_SYSTEM_PROMPT, prompt,
                             CFG.temperature, CFG.max_output_tokens, seed=seed)
    return {
        "task_id": task_id,
        "arm": arm,
        "seed": seed,
        "model": model,
        "provider": provider,
        "output": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "timestamp": time.time(),
    }


def build_arm_prompts(tasks, messy, reformulated):
    by_id = {t["task_id"]: t for t in tasks}
    messy_by_id = {m["task_id"]: m for m in messy}
    ref_by_id = {r["task_id"]: r for r in reformulated}
    out = {}
    for tid in by_id:
        out[tid] = {
            "A": by_id[tid]["instruct_prompt"],
            "B": messy_by_id[tid]["messy_prompt"],
            "C": ref_by_id[tid]["reformulated_prompt"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["primary", "cross"], default="primary")
    args = parser.parse_args()
    load_dotenv()

    tasks = load_jsonl(DATA_DIR / "tasks.jsonl")
    messy = load_jsonl(DATA_DIR / "messy.jsonl")
    reformulated = load_jsonl(DATA_DIR / "reformulated.jsonl")
    prompts = build_arm_prompts(tasks, messy, reformulated)

    if args.target == "primary":
        provider, model = "anthropic", CFG.target_primary
        client = get_anthropic_client()
        task_ids = [t["task_id"] for t in tasks[:CFG.n_tasks_primary]]
        seeds = CFG.seed_list_primary
    else:
        provider, model = "openai", CFG.target_cross
        client = get_openai_client()
        task_ids = [t["task_id"] for t in tasks[:CFG.n_tasks_cross]]
        seeds = CFG.seed_list_cross

    out_path = RESULTS_DIR / "runs.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    done = {(r["task_id"], r["arm"], r["seed"], r["model"])
            for r in load_jsonl(out_path)}

    plan = [(tid, arm, seed) for tid in task_ids for arm in ("A", "B", "C") for seed in seeds]
    for tid, arm, seed in tqdm(plan, desc=f"target={args.target}"):
        if (tid, arm, seed, model) in done:
            continue
        record = run_one(client, provider, model, tid, arm, seed, prompts[tid][arm])
        append_jsonl(out_path, record)


if __name__ == "__main__":
    main()
