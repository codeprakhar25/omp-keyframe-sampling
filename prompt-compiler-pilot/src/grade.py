"""Extract code from each run output, execute it against the BigCodeBench test,
record pass/fail into results/graded.jsonl.

WARNING: runs untrusted model-generated code locally via subprocess + timeout.
For higher safety, run inside Docker. Pilot trade-off: subprocess is sufficient."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: Path, record: dict) -> None:
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def extract_code(text: str) -> str:
    m = CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_test_in_sandbox(code: str, test: str, entry_point: str | None,
                        timeout: int = 30) -> dict:
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
        append_jsonl(out_path, graded)


if __name__ == "__main__":
    main()
