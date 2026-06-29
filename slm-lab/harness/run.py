"""S0 orchestrator: run conditions over a manifest and report accuracy-vs-tokens.

Example (offline smoke test, no API key, no GPU):
    python scripts/make_synthetic.py
    python -m harness.run --manifest data/manifest.synthetic.json \
        --conditions A C --answerer echo --selector uniform --k 4

Example (real run):
    python -m harness.run --manifest data/manifest.json \
        --conditions A C --answerer anthropic --model claude-sonnet-4-5 \
        --selector embedding --k 6 --judge
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

from .answerers import build_answerer, make_text_judge
from .media import load_frames
from .metrics import exact_match, llm_judge, recall_at_k
from .selectors import EmbeddingSelector, FullDumpSelector, Selector, UniformSelector


def build_selector(name: str) -> Selector:
    if name == "uniform":
        return UniformSelector()
    if name == "embedding":
        return EmbeddingSelector()
    raise ValueError(f"unknown selector: {name!r}")


def load_manifest(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # allow {"items": [...]}
        data = data["items"]
    return data


def run_condition(
    cond: str,
    selector: Selector,
    answerer,
    items: List[dict],
    k: int,
    dump_fps: float,
    max_dump_frames: int,
    judge_fn,
) -> List[dict]:
    rows: List[dict] = []
    for item in items:
        frames = load_frames(item, dump_fps=dump_fps, max_frames=max_dump_frames)
        selected = selector.select(frames, item["question"], k)

        t0 = time.time()
        ans = answerer.answer(item["question"], selected)
        latency = time.time() - t0

        gold = item.get("gold_answer", "")
        acc_exact = exact_match(ans.text, gold)
        acc_judge = llm_judge(item["question"], ans.text, gold, judge_fn) if judge_fn else None
        accuracy = acc_judge if acc_judge is not None else acc_exact

        rows.append(
            {
                "condition": cond,
                "item_id": item.get("id"),
                "selector": selector.name,
                "n_candidate_frames": len(frames),
                "n_selected_frames": len(selected),
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
                "latency_s": round(latency, 3),
                "recall_at_k": recall_at_k(selected, item),
                "accuracy": accuracy,
                "accuracy_exact": acc_exact,
                "accuracy_judge": acc_judge,
                "pred": ans.text,
                "gold": gold,
            }
        )
    return rows


def _mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if v is not None]
    return round(mean(nums), 4) if nums else None


def summarize(rows: List[dict]) -> Dict[str, dict]:
    by_cond: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    summary: Dict[str, dict] = {}
    for cond, rs in by_cond.items():
        summary[cond] = {
            "n": len(rs),
            "accuracy": _mean_or_none([r["accuracy"] for r in rs]),
            "mean_input_tokens": _mean_or_none([r["input_tokens"] for r in rs]),
            "mean_recall_at_k": _mean_or_none([r["recall_at_k"] for r in rs]),
            "mean_selected_frames": _mean_or_none([r["n_selected_frames"] for r in rs]),
            "mean_latency_s": _mean_or_none([r["latency_s"] for r in rs]),
        }

    # token reduction vs full-dump condition A (the SPEC's headline ratio)
    base = summary.get("A", {}).get("mean_input_tokens")
    for cond, s in summary.items():
        mit = s.get("mean_input_tokens")
        s["token_reduction_vs_A"] = round(base / mit, 2) if base and mit else None
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="S0 visual-evidence-compression harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conditions", nargs="+", default=["A", "C"], choices=["A", "C"])
    ap.add_argument("--answerer", default="echo", choices=["echo", "anthropic", "openai"])
    ap.add_argument("--model", default=None, help="answerer model id (provider-specific)")
    ap.add_argument("--selector", default="uniform", choices=["uniform", "embedding"],
                    help="selector used for condition C")
    ap.add_argument("--k", type=int, default=6, help="frames to select in condition C")
    ap.add_argument("--dump-fps", type=float, default=1.0, help="candidate-frame sampling rate for video")
    ap.add_argument("--max-dump-frames", type=int, default=64, help="cap on condition A frames")
    ap.add_argument("--max-side", type=int, default=768, help="downscale long image side before sending")
    ap.add_argument("--judge", action="store_true", help="use LLM judge for accuracy (else exact match)")
    ap.add_argument("--judge-provider", default=None, help="defaults to --answerer provider")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    answerer = build_answerer(args.answerer, args.model, max_side=args.max_side)
    judge_fn = None
    if args.judge:
        judge_fn = make_text_judge(args.judge_provider or args.answerer, args.model)

    c_selector = build_selector(args.selector)

    all_rows: List[dict] = []
    for cond in args.conditions:
        selector: Selector = FullDumpSelector() if cond == "A" else c_selector
        print(f"== condition {cond} (selector={selector.name}, answerer={answerer.name}) ==")
        rows = run_condition(
            cond, selector, answerer, items, args.k, args.dump_fps, args.max_dump_frames, judge_fn
        )
        all_rows.extend(rows)

    summary = summarize(all_rows)

    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.jsonl")
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {runs_path} and {summary_path}")

    # echo the go/no-go gate (SPEC §8)
    a, c = summary.get("A"), summary.get("C")
    if a and c and a["accuracy"] is not None and c["accuracy"] is not None:
        acc_ok = c["accuracy"] >= a["accuracy"] - 0.02
        tok_ok = (c["token_reduction_vs_A"] or 0) >= 5
        print(
            f"\nGO/NO-GO gate: accuracy_within_2pts={acc_ok}, token_reduction>=5x={tok_ok} "
            f"-> {'PROCEED' if (acc_ok and tok_ok) else 'STOP / investigate'}"
        )


if __name__ == "__main__":
    main()
