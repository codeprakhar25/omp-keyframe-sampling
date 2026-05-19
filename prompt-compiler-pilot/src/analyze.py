"""Implements the pre-registered analysis plan (see PREREG.md).
Reads results/graded.jsonl, outputs results/report.md."""
import json
import random
from pathlib import Path
from typing import Sequence
from scipy.stats import binomtest

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Pricing USD per 1M tokens — update at run time to match actual published prices
PRICING = {
    "claude-sonnet-4-6":         {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5-20251001": {"in": 0.80,  "out": 4.00},
    "gpt-4.1":                   {"in": 2.00,  "out": 8.00},
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pass_rate(runs: list[dict]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r["passed"]) / len(runs)


def mcnemar_paired(pairs: Sequence[tuple[str, str]]) -> float:
    """Exact McNemar via binomial on discordant pairs."""
    b = sum(1 for x, y in pairs if x == "pass" and y == "fail")
    c = sum(1 for x, y in pairs if x == "fail" and y == "pass")
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n=n, p=0.5).pvalue


def bootstrap_diff(b: list[int], c: list[int], n: int = 10000,
                   seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    idx = list(range(len(b)))
    diffs = []
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        mb = sum(b[i] for i in sample) / len(sample)
        mc = sum(c[i] for i in sample) / len(sample)
        diffs.append(mc - mb)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def failure_mode_rate(pairs: Sequence[tuple[str, str]]) -> float:
    b_pass = sum(1 for x, _ in pairs if x == "pass")
    harm = sum(1 for x, y in pairs if x == "pass" and y == "fail")
    return harm / b_pass if b_pass else 0.0


def per_arm(graded: list[dict], model: str, arm: str) -> list[dict]:
    return [r for r in graded if r["model"] == model and r["arm"] == arm]


def make_pairs(graded: list[dict], model: str,
               arm_x: str, arm_y: str) -> list[tuple[str, str]]:
    by_x = {(r["task_id"], r["seed"]): r["passed"]
            for r in per_arm(graded, model, arm_x)}
    by_y = {(r["task_id"], r["seed"]): r["passed"]
            for r in per_arm(graded, model, arm_y)}
    keys = sorted(set(by_x) & set(by_y))
    return [("pass" if by_x[k] else "fail",
             "pass" if by_y[k] else "fail") for k in keys]


def arm_cost(graded: list[dict], model: str, arm: str) -> float:
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    rows = per_arm(graded, model, arm)
    return sum(r["input_tokens"] / 1e6 * p["in"] +
               r["output_tokens"] / 1e6 * p["out"] for r in rows)


def analyze_target(graded: list[dict], model: str) -> dict:
    arms = {a: per_arm(graded, model, a) for a in ("A", "B", "C")}
    pr = {a: pass_rate(arms[a]) for a in ("A", "B", "C")}
    bc = make_pairs(graded, model, "B", "C")
    ac = make_pairs(graded, model, "A", "C")
    p_bc = mcnemar_paired(bc)
    p_ac = mcnemar_paired(ac)
    b_vals = [1 if x == "pass" else 0 for x, _ in bc]
    c_vals = [1 if y == "pass" else 0 for _, y in bc]
    ci = bootstrap_diff(b_vals, c_vals) if b_vals else (0.0, 0.0)
    failure_rate = failure_mode_rate(bc)
    costs = {a: arm_cost(graded, model, a) for a in ("A", "B", "C")}
    correct = {a: sum(1 for r in arms[a] if r["passed"]) for a in ("A", "B", "C")}
    cost_per_correct = {
        a: (costs[a] / correct[a] if correct[a] else None)
        for a in ("A", "B", "C")
    }
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
        lines.append("\n| Arm | Pass rate | Correct | Cost (USD) | $/correct |")
        lines.append("|---|---|---|---|---|")
        for arm in ("A", "B", "C"):
            cpc = a["cost_per_correct"][arm]
            cpc_str = f"${cpc:.4f}" if cpc is not None else "n/a"
            lines.append(
                f"| {arm} | {a['pass_rate'][arm]:.3f} | {a['correct'][arm]} | "
                f"${a['cost_usd'][arm]:.2f} | {cpc_str} |"
            )
        lo, hi = a["bootstrap_CI_C_minus_B"]
        lines.append(f"\n- McNemar B vs C: p={a['mcnemar_BvsC_p']:.4f}")
        lines.append(f"- McNemar A vs C: p={a['mcnemar_AvsC_p']:.4f}")
        lines.append(f"- Bootstrap 95% CI on (C − B): [{lo:.3f}, {hi:.3f}]")
        lines.append(f"- Failure-mode rate (B-pass, C-fail): {a['failure_mode_rate']:.3f}\n")
    return "\n".join(lines)


def main():
    graded = load_jsonl(RESULTS_DIR / "graded.jsonl")
    if not graded:
        print("No graded results found. Run grade.py first.")
        return
    models = sorted({r["model"] for r in graded})
    analyses = [analyze_target(graded, m) for m in models]
    report = render_report(analyses)
    out = RESULTS_DIR / "report.md"
    out.write_text(report)
    print(f"Report written to {out}")
    print(report)


if __name__ == "__main__":
    main()
