# Prompt Compiler Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a single-turn, pre-registered pilot measuring whether structural prompt reformulation improves pass@1 on synthetically-messified BigCodeBench-Hard tasks for Claude Sonnet 4.6, with a small GPT-5.4 cross-check.

**Architecture:** Six pipeline stages run as separate scripts that read/write JSONL files in `data/` and `results/`. Each stage is idempotent and cacheable so re-runs don't burn API budget. The reformulator and messifier are LLM calls with frozen prompts. Target model output is graded by the BigCodeBench evaluator. Analysis is done in a separate script that reads `results/runs.jsonl` and outputs a Markdown report.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `openai` SDK, `bigcodebench`, `pytest`, `numpy`, `scipy.stats` (McNemar), `python-dotenv`. Dependency management via `uv` (fallback: `venv` + `pip`).

**Plan Order Notes:**
- PREREG.md (Task 9) must be written and committed **before** any target-model run (Tasks 11–12).
- Messifier validation (Task 6) must pass **before** reformulator is run.

---

### Task 1: Project scaffolding

**Files:**
- Create: `claude-prompt-compiler-pilot/pyproject.toml`
- Create: `claude-prompt-compiler-pilot/.gitignore`
- Create: `claude-prompt-compiler-pilot/.env.example`
- Create: `claude-prompt-compiler-pilot/README.md`
- Create: `claude-prompt-compiler-pilot/src/__init__.py`
- Create: `claude-prompt-compiler-pilot/src/config.py`
- Create: `claude-prompt-compiler-pilot/tests/__init__.py`
- Create: `claude-prompt-compiler-pilot/data/.gitkeep`
- Create: `claude-prompt-compiler-pilot/results/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd /home/prakh/misc-cc/play/claude-prompt-compiler-pilot
mkdir -p src tests prompts data results
touch src/__init__.py tests/__init__.py data/.gitkeep results/.gitkeep
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "claude-prompt-compiler-pilot"
version = "0.1.0"
description = "Directional pilot: does structural reformulation help frontier coding models?"
requires-python = ">=3.11"
dependencies = [
  "anthropic>=0.40.0",
  "openai>=1.50.0",
  "bigcodebench>=0.2.0",
  "datasets>=2.18.0",
  "python-dotenv>=1.0.0",
  "numpy>=1.26.0",
  "scipy>=1.12.0",
  "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-mock>=3.12.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 3: Write .gitignore**

```
.venv/
__pycache__/
*.pyc
.env
results/runs.jsonl
results/report.md
.pytest_cache/
.bigcodebench_cache/
```

- [ ] **Step 4: Write .env.example**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

- [ ] **Step 5: Write src/config.py**

```python
"""Central configuration for the pilot. All run parameters live here."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Models (pinned)
    target_primary: str = "claude-sonnet-4-6"
    target_cross: str = "gpt-5.4"  # use latest 5.4+ available
    reformulator: str = "claude-haiku-4-5-20251001"
    messifier: str = "claude-haiku-4-5-20251001"

    # Decoding
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 2048

    # Dataset
    dataset_name: str = "bigcode/bigcodebench-hard"
    n_tasks_primary: int = 30
    n_tasks_cross: int = 15
    min_prompt_words: int = 80

    # Sampling
    n_seeds_primary: int = 5
    n_seeds_cross: int = 3
    seed_list_primary: tuple = (1, 2, 3, 4, 5)
    seed_list_cross: tuple = (1, 2, 3)

CFG = Config()
```

- [ ] **Step 6: Write README.md skeleton**

```markdown
# Prompt Compiler Pilot

Directional pilot study. See `SPEC.md` for the experimental design.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # then fill in API keys
```

## Reproduce

```bash
python -m src.select_tasks    # → data/tasks.jsonl
python -m src.messify         # → data/messy.jsonl
# HAND-VALIDATE 10 samples (see SPEC.md §7)
python -m src.reformulate     # → data/reformulated.jsonl
# COMMIT PREREG.md before this line
python -m src.run_target --target primary
python -m src.grade
python -m src.run_target --target cross
python -m src.grade --target cross
python -m src.analyze         # → results/report.md
```
```

- [ ] **Step 7: Commit**

```bash
git add claude-prompt-compiler-pilot/
git commit -m "feat(pilot): scaffold prompt-compiler pilot project"
```

---

### Task 2: Write the messifier prompt

**Files:**
- Create: `claude-prompt-compiler-pilot/prompts/messifier.md`

- [ ] **Step 1: Write messifier.md**

```markdown
# Messifier System Prompt

You are simulating how a busy, distracted developer types a coding request to an AI assistant. You will receive a clean, well-structured task specification. Your job is to rewrite it as a single rambling, naturalistic message that preserves every original requirement but presents them poorly.

## Hard rules

1. **Preserve every original requirement.** Do not drop function signatures, input/output types, edge cases, examples, or constraints. Semantic preservation is mandatory.
2. **Do not add new requirements.** Do not invent constraints, test cases, or behaviors not in the original.
3. **Output one block of prose.** No headings, no numbered lists, no bullets. Plain paragraph(s).
4. **Length:** between 1.3× and 1.8× the original word count.

## What "messy" means

- Add a rambling preamble of unrelated context (60–120 words). Examples: complaints about a previous task, mention of a deadline, an unrelated tangent about the project, a half-finished thought that trails off.
- **Bury** the requirements inside the noise. Do not list them in order.
- **Mix** style preferences, constraints, and the spec inline. Don't separate concerns.
- Add **2–4 mild typos** (transposed letters, missing apostrophes). Do not break code identifiers.
- Use casual punctuation (run-on sentences, missing commas, inconsistent capitalization).

## Output format

Return only the messy version. No commentary, no preamble like "Here is the messy version:", no quotes around it.
```

- [ ] **Step 2: Commit**

```bash
git add prompts/messifier.md
git commit -m "feat(pilot): add messifier prompt"
```

---

### Task 3: Write the reformulator prompt

**Files:**
- Create: `claude-prompt-compiler-pilot/prompts/reformulator.md`

- [ ] **Step 1: Write reformulator.md**

```markdown
# Reformulator System Prompt

You are a prompt compiler. You receive a messy, rambling coding request from a developer and return a clean, structured specification that another AI model will execute. You do not solve the task. You only restructure the request.

## Hard rules

1. **Do not add information not in the input.** No invented requirements, no assumed defaults, no extra test cases.
2. **Do not drop information.** If unsure whether something is a requirement, include it.
3. **Do not solve the task.** No code, no algorithm suggestions, no "you could implement this with..."
4. **Output the fixed Markdown template below. Nothing else.**

## Output template

```
## Task
<one-sentence summary of what the function/code should do>

## Requirements
1. <requirement 1>
2. <requirement 2>
...

## Inputs
- <name: type — description>
...

## Outputs
- <name: type — description>

## Constraints
- <constraint 1>
...

## Edge cases
- <edge case 1>
...

## Style preferences
- <preference 1 if any, else "none specified">
```

If a section has no content from the input, write "none specified" — do not omit the section.

## What to strip

- Rambling preamble, complaints, tangents, project chatter
- Filler words ("basically", "kind of", "sort of")
- Restated information

## What to preserve verbatim

- Function names, type names, example values, identifiers
- Numeric constants
- Quoted strings that appear to be test inputs or expected outputs
```

- [ ] **Step 2: Commit**

```bash
git add prompts/reformulator.md
git commit -m "feat(pilot): add reformulator prompt"
```

---

### Task 4: Task selector

**Files:**
- Create: `claude-prompt-compiler-pilot/src/select_tasks.py`
- Create: `claude-prompt-compiler-pilot/tests/test_select_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_select_tasks.py -v`
Expected: FAIL with "ModuleNotFoundError" or "filter_tasks not defined"

- [ ] **Step 3: Write src/select_tasks.py**

```python
"""Fetch BigCodeBench-Hard, filter, save 30 tasks to data/tasks.jsonl."""
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from src.config import CFG

DATA_DIR = Path(__file__).parent.parent / "data"

def filter_tasks(tasks: list[dict], n: int, min_words: int) -> list[dict]:
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

def main():
    ds = load_dataset(CFG.dataset_name, split="v0.1.4")
    tasks = [dict(t) for t in ds]
    selected = filter_tasks(tasks, n=CFG.n_tasks_primary, min_words=CFG.min_prompt_words)
    if len(selected) < CFG.n_tasks_primary:
        raise RuntimeError(f"Only {len(selected)} tasks met filter; need {CFG.n_tasks_primary}")
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
                "content_hash": hashlib.sha256(t["instruct_prompt"].encode()).hexdigest()[:16],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(selected)} tasks to {out_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_select_tasks.py -v`
Expected: PASS

- [ ] **Step 5: Run the script to generate tasks.jsonl**

```bash
python -m src.select_tasks
```

Expected: `Wrote 30 tasks to .../data/tasks.jsonl`. If BigCodeBench-Hard field names differ from `instruct_prompt`/`canonical_solution`/`test`, update both the script and the test in lockstep, then re-run.

- [ ] **Step 6: Commit**

```bash
git add src/select_tasks.py tests/test_select_tasks.py data/tasks.jsonl
git commit -m "feat(pilot): task selector + 30 BigCodeBench-Hard tasks pinned"
```

---

### Task 5: LLM client wrappers

**Files:**
- Create: `claude-prompt-compiler-pilot/src/llm.py`
- Create: `claude-prompt-compiler-pilot/tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
from unittest.mock import MagicMock
from src.llm import call_anthropic, call_openai

def test_call_anthropic_passes_system_and_user(mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="hello")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    result = call_anthropic(
        client=mock_client,
        model="claude-haiku-4-5-20251001",
        system="be brief",
        user="say hi",
        temperature=0.2,
        max_tokens=100,
    )
    assert result["text"] == "hello"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be brief"
    assert kwargs["messages"] == [{"role": "user", "content": "say hi"}]
    assert kwargs["temperature"] == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write src/llm.py**

```python
"""Thin wrappers over Anthropic and OpenAI SDKs. Returns dicts with text + token counts."""
import os
from typing import Any

def call_anthropic(client: Any, model: str, system: str, user: str,
                   temperature: float, max_tokens: int) -> dict:
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return {
        "text": text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }

def call_openai(client: Any, model: str, system: str, user: str,
                temperature: float, max_tokens: int, seed: int | None = None) -> dict:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    return {
        "text": resp.choices[0].message.content,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }

def get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat(pilot): LLM client wrappers with token tracking"
```

---

### Task 6: Messify script + hand-validation gate

**Files:**
- Create: `claude-prompt-compiler-pilot/src/messify.py`
- Create: `claude-prompt-compiler-pilot/data/messifier_validation.md`

- [ ] **Step 1: Write src/messify.py**

```python
"""Run the messifier on each task in tasks.jsonl. Writes data/messy.jsonl.
Idempotent: skips tasks whose content_hash already appears in messy.jsonl."""
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
```

- [ ] **Step 2: Run messify**

```bash
python -m src.messify
```

Expected: progress bar, `data/messy.jsonl` populated with 30 rows. Estimated cost: <$0.10.

- [ ] **Step 3: Hand-validate 10 random samples**

Pick 10 task indices by running:

```bash
python -c "import random; random.seed(42); print(sorted(random.sample(range(30), 10)))"
```

For each picked index, open `data/messy.jsonl` and `data/tasks.jsonl` side by side. Compare. For each sample, judge:
- Does the messy version preserve every requirement from the original?
- Is the messiness believable (not so destroyed that it's a different task)?

Record judgments in `data/messifier_validation.md`:

```markdown
# Messifier Validation

Sampled indices (seed=42): [list]

| Index | task_id | Requirements preserved? | Notes |
|-------|---------|------------------------|-------|
| 0 | BCB/X | YES / NO | ... |
| ... |

Pass criterion: ≥9/10 YES.

Result: PASS / FAIL
```

- [ ] **Step 4: Decision gate**

If FAIL (<9/10): revise `prompts/messifier.md`, delete `data/messy.jsonl`, re-run Step 2, re-validate. Do not proceed past this gate.

If PASS: commit and continue.

- [ ] **Step 5: Commit**

```bash
git add src/messify.py data/messy.jsonl data/messifier_validation.md
git commit -m "feat(pilot): messify pipeline + hand-validation PASS"
```

---

### Task 7: Reformulate script

**Files:**
- Create: `claude-prompt-compiler-pilot/src/reformulate.py`

- [ ] **Step 1: Write src/reformulate.py**

```python
"""Run the reformulator on each messy prompt. Writes data/reformulated.jsonl.
Idempotent."""
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.config import CFG
from src.llm import call_anthropic, get_anthropic_client
from src.messify import load_jsonl, append_jsonl

DATA_DIR = Path(__file__).parent.parent / "data"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

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
```

- [ ] **Step 2: Run reformulate**

```bash
python -m src.reformulate
```

Expected: `data/reformulated.jsonl` populated with 30 rows. Estimated cost: <$0.20.

- [ ] **Step 3: Spot-check 3 reformulated outputs**

Manually open 3 rows. Check:
- Output follows the fixed template (Task/Requirements/Inputs/Outputs/Constraints/Edge cases/Style)
- No code in the output
- No invented requirements

If any fail, revise `prompts/reformulator.md`, delete `data/reformulated.jsonl`, re-run.

- [ ] **Step 4: Commit**

```bash
git add src/reformulate.py data/reformulated.jsonl
git commit -m "feat(pilot): reformulate pipeline + spot-check PASS"
```

---

### Task 8: PREREG.md (frozen analysis plan)

**Files:**
- Create: `claude-prompt-compiler-pilot/PREREG.md`

- [ ] **Step 1: Write PREREG.md**

This must be written **before any target-model run** and committed in its own commit.

```markdown
# Pre-Registration — Prompt Compiler Pilot v1

**Frozen on:** 2026-05-19
**Frozen before:** any target-model run

## Hypotheses

- **H1 (primary):** pass@1 in arm C > pass@1 in arm B for Claude Sonnet 4.6, McNemar paired p<0.05.
- **H2 (secondary):** pass@1 in arm C ≥ pass@1 in arm A − 5pp.
- **H3 (cost):** $/correct-answer for arm C is within 30% of arm B.

## Frozen artifacts

- Task list: `data/tasks.jsonl`, sha256 of file: <COMPUTE AND PASTE>
- Messy inputs: `data/messy.jsonl`, sha256: <COMPUTE AND PASTE>
- Reformulated inputs: `data/reformulated.jsonl`, sha256: <COMPUTE AND PASTE>
- Messifier prompt: `prompts/messifier.md`, sha256: <COMPUTE AND PASTE>
- Reformulator prompt: `prompts/reformulator.md`, sha256: <COMPUTE AND PASTE>
- Target prompt template: `src/run_target.py:TARGET_SYSTEM_PROMPT`

## Models (pinned)

- Primary target: `claude-sonnet-4-6`
- Cross-check target: `gpt-5.4` (latest 5.4+ available at run time, version recorded in runs.jsonl)
- Reformulator: `claude-haiku-4-5-20251001`
- Messifier: `claude-haiku-4-5-20251001`

## Decoding

- temperature=0.2, top_p=1.0, max_output_tokens=2048
- Seeds (primary): 1,2,3,4,5
- Seeds (cross): 1,2,3

## Conditions

- A: original prompt → target
- B: messy prompt → target
- C: reformulated prompt → target

## Analysis (no deviation without amending this file)

1. Mean pass@1 per arm with Wilson 95% CI.
2. McNemar paired test on (B vs C) over (task, seed) pairs. Report p-value.
3. McNemar paired test on (A vs C) (secondary).
4. Bootstrap 95% CI on (C − B), 10000 resamples, seed=42.
5. Per-arm input/output token totals; $/correct-answer using published pricing.
6. Failure-mode count: (task, seed) where B=pass and C=fail. Report rate.
7. Cross-check: same analysis on GPT-5.4 subset (15 tasks × 3 seeds).

**No subgroup analyses. No post-hoc filtering of tasks. Exploratory analyses, if any, will be clearly labeled and excluded from conclusions.**

## Stopping rule

Run completes when all 450 primary + 135 cross trials are graded. No interim peeking that changes the experiment.
```

- [ ] **Step 2: Fill in the sha256 hashes**

```bash
cd claude-prompt-compiler-pilot
for f in data/tasks.jsonl data/messy.jsonl data/reformulated.jsonl prompts/messifier.md prompts/reformulator.md; do
  echo "$f: $(sha256sum $f | cut -d' ' -f1)"
done
```

Paste the values into `PREREG.md` replacing each `<COMPUTE AND PASTE>`.

- [ ] **Step 3: Commit**

```bash
git add PREREG.md
git commit -m "chore(pilot): freeze PREREG.md before target runs"
```

---

### Task 9: Target runner

**Files:**
- Create: `claude-prompt-compiler-pilot/src/run_target.py`
- Create: `claude-prompt-compiler-pilot/tests/test_run_target.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_target.py
from unittest.mock import MagicMock
from src.run_target import run_one, TARGET_SYSTEM_PROMPT

def test_run_one_records_all_fields(mocker):
    mock_call = mocker.patch("src.run_target.call_anthropic")
    mock_call.return_value = {"text": "```python\ndef f(): pass\n```", "input_tokens": 50, "output_tokens": 20}
    record = run_one(
        client=MagicMock(),
        provider="anthropic",
        model="claude-sonnet-4-6",
        task_id="BCB/1",
        arm="B",
        seed=1,
        prompt="solve this",
    )
    assert record["task_id"] == "BCB/1"
    assert record["arm"] == "B"
    assert record["seed"] == 1
    assert record["model"] == "claude-sonnet-4-6"
    assert "def f(): pass" in record["output"]
    assert record["input_tokens"] == 50
    assert record["output_tokens"] == 20
    assert "timestamp" in record
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_target.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write src/run_target.py**

```python
"""Run the target model for each (task, arm, seed). Append-only writes to results/runs.jsonl.
Idempotent: skips (task_id, arm, seed) tuples already present."""
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.config import CFG
from src.llm import call_anthropic, call_openai, get_anthropic_client, get_openai_client
from src.messify import load_jsonl, append_jsonl

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET_SYSTEM_PROMPT = (
    "You are a precise Python code generator. Read the task and return a single "
    "Python code block containing the complete implementation. Do not include "
    "explanations outside the code block. Do not include test code unless asked."
)

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

    plan = [(tid, arm, seed) for tid in task_ids for arm in ("A","B","C") for seed in seeds]
    for tid, arm, seed in tqdm(plan, desc=f"target={args.target}"):
        if (tid, arm, seed, model) in done:
            continue
        record = run_one(client, provider, model, tid, arm, seed, prompts[tid][arm])
        append_jsonl(out_path, record)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_target.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/run_target.py tests/test_run_target.py
git commit -m "feat(pilot): target runner with idempotent resume"
```

---

### Task 10: Grader

**Files:**
- Create: `claude-prompt-compiler-pilot/src/grade.py`
- Create: `claude-prompt-compiler-pilot/tests/test_grade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade.py
from src.grade import extract_code, run_test_in_sandbox

def test_extract_code_strips_fence():
    s = "Here is the code:\n```python\ndef f(): return 1\n```\nthanks"
    assert extract_code(s) == "def f(): return 1"

def test_extract_code_handles_no_fence():
    s = "def f(): return 1"
    assert extract_code(s) == "def f(): return 1"

def test_run_test_in_sandbox_pass():
    code = "def add(a,b): return a+b"
    test = "assert add(1,2) == 3"
    result = run_test_in_sandbox(code, test, entry_point="add", timeout=5)
    assert result["passed"] is True

def test_run_test_in_sandbox_fail():
    code = "def add(a,b): return a-b"
    test = "assert add(1,2) == 3"
    result = run_test_in_sandbox(code, test, entry_point="add", timeout=5)
    assert result["passed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grade.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write src/grade.py**

```python
"""Extract code from each run output, execute it against the BigCodeBench test,
record pass/fail back into runs.jsonl.

WARNING: this runs untrusted model-generated code locally. Sandbox via subprocess
with timeout. For higher safety run inside Docker; for the pilot, subprocess + timeout
is the documented trade-off."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm
from src.messify import load_jsonl

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def extract_code(text: str) -> str:
    m = CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()

def run_test_in_sandbox(code: str, test: str, entry_point: str | None, timeout: int = 30) -> dict:
    script = code + "\n\n" + test + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "passed": proc.returncode == 0,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "TIMEOUT"}
    finally:
        Path(path).unlink(missing_ok=True)

def main():
    runs = load_jsonl(RESULTS_DIR / "runs.jsonl")
    tasks = {t["task_id"]: t for t in load_jsonl(DATA_DIR / "tasks.jsonl")}
    out_path = RESULTS_DIR / "graded.jsonl"
    done = {(r["task_id"], r["arm"], r["seed"], r["model"])
            for r in load_jsonl(out_path)}
    for r in tqdm(runs, desc="grade"):
        key = (r["task_id"], r["arm"], r["seed"], r["model"])
        if key in done:
            continue
        t = tasks[r["task_id"]]
        code = extract_code(r["output"])
        verdict = run_test_in_sandbox(code, t["test"], t.get("entry_point"))
        graded = {**r, "code": code, "passed": verdict["passed"],
                  "grader_stderr": verdict["stderr"][:500]}
        with out_path.open("a") as f:
            f.write(json.dumps(graded) + "\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grade.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/grade.py tests/test_grade.py
git commit -m "feat(pilot): grader with sandboxed test execution"
```

---

### Task 11: Run primary target end-to-end

**Files:**
- Modify: `claude-prompt-compiler-pilot/results/runs.jsonl` (generated)
- Modify: `claude-prompt-compiler-pilot/results/graded.jsonl` (generated)

- [ ] **Step 1: Sanity check — PREREG.md is committed**

```bash
git log --oneline -- PREREG.md
```

Expected: one commit. If PREREG.md is uncommitted or absent, STOP and complete Task 8 first.

- [ ] **Step 2: Run primary target**

```bash
python -m src.run_target --target primary
```

Expected: 450 records appended to `results/runs.jsonl`. Estimated cost: ~$7 (Sonnet 4.6). Monitor for rate limit errors; the script is idempotent so re-running resumes.

- [ ] **Step 3: Grade primary target**

```bash
python -m src.grade
```

Expected: 450 records in `results/graded.jsonl` with `"passed": true/false`. A subset may show timeouts; that's a fail. Sanity-check pass rate is non-zero (otherwise grader is broken).

- [ ] **Step 4: Commit results**

```bash
git add results/runs.jsonl results/graded.jsonl
git commit -m "data(pilot): primary target Sonnet 4.6 runs + grades"
```

---

### Task 12: Run cross-check target end-to-end

- [ ] **Step 1: Run cross-check target**

```bash
python -m src.run_target --target cross
```

Expected: 135 additional records in `results/runs.jsonl`. Estimated cost: ~$5–10 (GPT-5.4).

- [ ] **Step 2: Grade cross-check**

```bash
python -m src.grade
```

Expected: 135 additional records in `results/graded.jsonl`.

- [ ] **Step 3: Commit results**

```bash
git add results/runs.jsonl results/graded.jsonl
git commit -m "data(pilot): cross-check target GPT-5.4 runs + grades"
```

---

### Task 13: Analyzer and report

**Files:**
- Create: `claude-prompt-compiler-pilot/src/analyze.py`
- Create: `claude-prompt-compiler-pilot/tests/test_analyze.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze.py
from src.analyze import pass_rate, mcnemar_paired, bootstrap_diff, failure_mode_rate

def test_pass_rate():
    runs = [{"passed": True}, {"passed": False}, {"passed": True}, {"passed": True}]
    assert pass_rate(runs) == 0.75

def test_mcnemar_paired_known_case():
    # 5 pairs where B passes and C fails, 1 pair where C passes and B fails
    pairs = [("pass","fail")]*5 + [("fail","pass")]*1 + [("pass","pass")]*4
    p = mcnemar_paired(pairs)
    assert 0.0 < p < 0.3  # b=5, c=1 — discordant pairs only

def test_bootstrap_diff_returns_interval():
    b = [1,0,1,0,1,1,0,1,0,1]
    c = [1,1,1,0,1,1,1,1,1,1]
    lo, hi = bootstrap_diff(b, c, n=1000, seed=42)
    assert lo < hi

def test_failure_mode_rate():
    pairs = [("pass","fail"), ("pass","fail"), ("pass","pass"), ("fail","fail")]
    rate = failure_mode_rate(pairs)
    assert rate == 2/3  # 2 of 3 B-passes are C-fail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write src/analyze.py**

```python
"""Implements the pre-registered analysis plan (see PREREG.md §Analysis)."""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence
from scipy.stats import binomtest
from src.messify import load_jsonl

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Pricing (USD per 1M tokens), filled at time of run
PRICING = {
    "claude-sonnet-4-6":            {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5-20251001":    {"in": 0.80,  "out": 4.00},
    "gpt-5.4":                      {"in": 5.00,  "out": 20.00},  # update at run time
}

def pass_rate(runs: list[dict]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r["passed"]) / len(runs)

def mcnemar_paired(pairs: Sequence[tuple[str,str]]) -> float:
    """Exact McNemar via binomial on discordant pairs."""
    b = sum(1 for x,y in pairs if x=="pass" and y=="fail")
    c = sum(1 for x,y in pairs if x=="fail" and y=="pass")
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b,c), n=n, p=0.5).pvalue

def bootstrap_diff(b: list[int], c: list[int], n: int = 10000, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(b)))
    diffs = []
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        mb = sum(b[i] for i in sample)/len(sample)
        mc = sum(c[i] for i in sample)/len(sample)
        diffs.append(mc - mb)
    diffs.sort()
    return diffs[int(0.025*n)], diffs[int(0.975*n)]

def failure_mode_rate(pairs: Sequence[tuple[str,str]]) -> float:
    b_pass = sum(1 for x,_ in pairs if x=="pass")
    harm = sum(1 for x,y in pairs if x=="pass" and y=="fail")
    return harm / b_pass if b_pass else 0.0

def per_arm(graded: list[dict], model: str, arm: str):
    return [r for r in graded if r["model"]==model and r["arm"]==arm]

def paired(graded: list[dict], model: str, arm_x: str, arm_y: str):
    by_x = {(r["task_id"], r["seed"]): r["passed"]
            for r in per_arm(graded, model, arm_x)}
    by_y = {(r["task_id"], r["seed"]): r["passed"]
            for r in per_arm(graded, model, arm_y)}
    keys = sorted(set(by_x) & set(by_y))
    return [("pass" if by_x[k] else "fail",
             "pass" if by_y[k] else "fail") for k in keys]

def cost(graded: list[dict], model: str, arm: str) -> float:
    p = PRICING[model]
    rows = per_arm(graded, model, arm)
    return sum(r["input_tokens"]/1e6 * p["in"] + r["output_tokens"]/1e6 * p["out"]
               for r in rows)

def analyze_target(graded: list[dict], model: str) -> dict:
    arms = {a: per_arm(graded, model, a) for a in ("A","B","C")}
    pr = {a: pass_rate(arms[a]) for a in ("A","B","C")}
    bc = paired(graded, model, "B", "C")
    ac = paired(graded, model, "A", "C")
    p_bc = mcnemar_paired(bc)
    p_ac = mcnemar_paired(ac)
    b_vals = [1 if x=="pass" else 0 for x,_ in bc]
    c_vals = [1 if y=="pass" else 0 for _,y in bc]
    ci = bootstrap_diff(b_vals, c_vals)
    failure_rate = failure_mode_rate(bc)
    costs = {a: cost(graded, model, a) for a in ("A","B","C")}
    correct = {a: sum(1 for r in arms[a] if r["passed"]) for a in ("A","B","C")}
    cost_per_correct = {a: (costs[a]/correct[a] if correct[a] else None)
                        for a in ("A","B","C")}
    return {
        "model": model,
        "pass_rate": pr,
        "mcnemar_BvsC_p": p_bc,
        "mcnemar_AvsC_p": p_ac,
        "bootstrap_CI_C_minus_B": ci,
        "failure_mode_rate": failure_rate,
        "cost_usd": costs,
        "correct": correct,
        "cost_per_correct": cost_per_correct,
        "n_pairs_BC": len(bc),
    }

def render_report(analyses: list[dict]) -> str:
    lines = ["# Pilot Results\n"]
    for a in analyses:
        lines.append(f"## Target: `{a['model']}`\n")
        lines.append(f"- n paired (B vs C): {a['n_pairs_BC']}")
        lines.append("| Arm | Pass rate | Correct | Cost (USD) | $/correct |")
        lines.append("|---|---|---|---|---|")
        for arm in ("A","B","C"):
            cpc = a["cost_per_correct"][arm]
            lines.append(f"| {arm} | {a['pass_rate'][arm]:.3f} | {a['correct'][arm]} | "
                         f"${a['cost_usd'][arm]:.2f} | "
                         f"{('$%.4f' % cpc) if cpc else 'n/a'} |")
        lo, hi = a["bootstrap_CI_C_minus_B"]
        lines.append(f"\n- McNemar B vs C: p={a['mcnemar_BvsC_p']:.4f}")
        lines.append(f"- McNemar A vs C: p={a['mcnemar_AvsC_p']:.4f}")
        lines.append(f"- Bootstrap 95% CI on (C − B): [{lo:.3f}, {hi:.3f}]")
        lines.append(f"- Failure-mode rate (B-pass, C-fail): {a['failure_mode_rate']:.3f}\n")
    return "\n".join(lines)

def main():
    graded = load_jsonl(RESULTS_DIR / "graded.jsonl")
    models = sorted({r["model"] for r in graded})
    analyses = [analyze_target(graded, m) for m in models]
    report = render_report(analyses)
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyze.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Generate the report**

```bash
python -m src.analyze
```

Expected: `results/report.md` is written. Inspect it.

- [ ] **Step 6: Update pricing in src/analyze.py:PRICING**

Before reporting numbers as final, confirm published per-Mtoken prices for the exact model versions used. Update `PRICING` if any differ from the placeholder values, then re-run `python -m src.analyze`.

- [ ] **Step 7: Commit**

```bash
git add src/analyze.py tests/test_analyze.py results/report.md
git commit -m "feat(pilot): analyzer + final report"
```

---

### Task 14: Write the conclusions section

**Files:**
- Modify: `claude-prompt-compiler-pilot/results/report.md`

- [ ] **Step 1: Append interpretation to results/report.md**

After the generated tables, manually append a Conclusions section that answers the pre-registered hypotheses:

```markdown
## Conclusions (against PREREG.md)

- **H1 (C > B for Sonnet 4.6, McNemar p<0.05):** SUPPORTED / NOT SUPPORTED.
  Evidence: <pass rates>, p=<value>, bootstrap CI <range>.

- **H2 (C ≥ A − 5pp):** SUPPORTED / NOT SUPPORTED.
  Evidence: A=<x>, C=<y>, gap=<x-y>.

- **H3 ($/correct C within 30% of B):** SUPPORTED / NOT SUPPORTED.
  Evidence: $/correct B=<x>, C=<y>, ratio=<y/x>.

- **Failure mode (B-pass, C-fail):** rate=<r>. <Above / within> the 15% concern threshold.

- **Cross-check (GPT-5.4):** directionally <consistent / inconsistent> with Sonnet 4.6. n=15 is too small to test independently; this is corroboration only.

## Caveats

- n=30 tasks; underpowered for effects <12pp.
- Synthetic mess; ecological validity for organic messy prompts unknown.
- Single-turn; transferability to agentic multi-turn workflows untested.
- No surface-cleanup-only arm; cannot separate structural from surface effects.

## Next experiment (v2 candidates)

- <picked based on v1 outcome>
```

- [ ] **Step 2: Commit final report**

```bash
git add results/report.md
git commit -m "docs(pilot): conclusions and caveats"
```

---

## Self-review checklist (for plan author — done before user review)

- [x] **Spec coverage:** Every SPEC §1–14 has a corresponding task. PREREG (§10) → Task 8; messifier validity gate (§7) → Task 6 Step 4; failure-mode metric (§9) → Task 13.
- [x] **Placeholder scan:** No TBD/TODO inside steps. Two intentional placeholders flagged for engineer: PREREG sha256 values (Task 8 Step 2 computes them) and PRICING (Task 13 Step 6 instructs to confirm/update).
- [x] **Type consistency:** `task_id` used throughout; `passed` (boolean) used in both `grade.py` and `analyze.py`; arms always `A`/`B`/`C`.
- [x] **Test code is concrete:** Every TDD task has actual test code, not "write tests for the above".
