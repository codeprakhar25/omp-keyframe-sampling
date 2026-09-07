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
from .metrics import exact_match, hit_at_k, llm_judge, recall_at_k
from .selectors import EmbeddingSelector, FullDumpSelector, Selector, UniformSelector
from harness.text import question_stem


def build_selector(name: str, model_id: str | None = None, longest_edge: int | None = None,
                   vlm_model: str | None = None, reasoning_effort: str | None = None) -> Selector:
    if name == "uniform":
        return UniformSelector()
    if name == "embedding":
        kwargs = {"model_id": model_id} if model_id else {}
        return EmbeddingSelector(**kwargs)
    if name == "hier":
        from .selectors import HierarchicalSelector
        kwargs = {"model_id": model_id} if model_id else {}
        return HierarchicalSelector(**kwargs)
    if name == "beam":
        from .selectors import BeamCoarseToFineSelector
        kwargs = {"model_id": model_id} if model_id else {}
        return BeamCoarseToFineSelector(**kwargs)
    if name == "transcript":
        from .selectors import TranscriptGatedSelector
        kwargs = {"model_id": model_id} if model_id else {}
        return TranscriptGatedSelector(**kwargs)
    if name == "videoret":
        from .selectors import VideoRetSelector
        kwargs = {"model_id": model_id} if model_id else {}
        return VideoRetSelector(**kwargs)
    if name == "smolvlm":
        from .selectors import SmolVLMSelector
        kwargs = {}
        if model_id:
            kwargs["model_id"] = model_id
        if longest_edge:
            kwargs["longest_edge"] = longest_edge
        return SmolVLMSelector(**kwargs)
    if name == "pe":
        from .selectors import PESelector
        # for PE the selector "model_id" is the PE-Core config (default = fair-peer L14-336)
        kwargs = {"config": model_id} if model_id else {}
        return PESelector(**kwargs)
    if name == "twostage":
        from .selectors import TwoStageVLMSelector
        kwargs = {}
        if model_id:
            kwargs["model_id"] = model_id
        if vlm_model:
            kwargs["vlm_model"] = vlm_model
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return TwoStageVLMSelector(**kwargs)
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
    arm: str | None = None,
) -> List[dict]:
    rows: List[dict] = []
    for item in items:
        frames = load_frames(item, dump_fps=dump_fps, max_frames=max_dump_frames)
        selector._item = item  # transcript-gated selector reads subtitle path here; others ignore
        selected = selector.select(frames, question_stem(item), k)

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
                # arm = fine-grained label for Fork-A analysis (e.g. "A","knob","U","C-so400m");
                # `condition` stays A/C so the go/no-go gate still works.
                "arm": arm or cond,
                "item_id": item.get("id"),
                # video length is the Fork-A independent variable (for length-bin crossover plot)
                "video_seconds": item.get("video_seconds"),
                "length_bin": item.get("length_bin"),
                "selector": selector.name,
                "n_candidate_frames": len(frames),
                "n_selected_frames": len(selected),
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
                "latency_s": round(latency, 3),
                "recall_at_k": recall_at_k(selected, item),
                "hit_at_k": hit_at_k(selected, item),
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
            "mean_hit_at_k": _mean_or_none([r["hit_at_k"] for r in rs]),
            "mean_selected_frames": _mean_or_none([r["n_selected_frames"] for r in rs]),
            "mean_latency_s": _mean_or_none([r["latency_s"] for r in rs]),
        }

    # token reduction vs full-dump condition A (the SPEC's headline ratio)
    base = summary.get("A", {}).get("mean_input_tokens")
    for cond, s in summary.items():
        mit = s.get("mean_input_tokens")
        s["token_reduction_vs_A"] = round(base / mit, 2) if base and mit else None
    return summary


def summarize_by_bin(rows: List[dict]) -> Dict[str, dict]:
    """Fork-A view: accuracy / hit@k / tokens per (length_bin, arm).

    length_bin comes from the manifest; falls back to 'all' when absent so this is
    harmless on non-binned manifests.
    """
    by_bin: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_bin[r.get("length_bin") or "all"][r.get("arm") or r["condition"]].append(r)

    out: Dict[str, dict] = {}
    for b in sorted(by_bin):
        out[b] = {}
        for arm, rs in by_bin[b].items():
            out[b][arm] = {
                "n": len(rs),
                "accuracy": _mean_or_none([r["accuracy"] for r in rs]),
                "hit_at_k": _mean_or_none([r["hit_at_k"] for r in rs]),
                "mean_input_tokens": _mean_or_none([r["input_tokens"] for r in rs]),
                "mean_latency_s": _mean_or_none([r["latency_s"] for r in rs]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="S0 visual-evidence-compression harness")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conditions", nargs="+", default=["A", "C"], choices=["A", "C"])
    ap.add_argument("--answerer", default="echo", choices=["echo", "anthropic", "openai"])
    ap.add_argument("--model", default=None, help="answerer model id (provider-specific)")
    ap.add_argument("--selector", default="uniform",
                    choices=["uniform", "embedding", "hier", "beam", "transcript", "videoret", "smolvlm", "pe", "twostage"],
                    help="selector used for condition C")
    ap.add_argument("--selector-model", default=None,
                    help="override the selector's model id (e.g. HuggingFaceTB/SmolVLM2-2.2B-Instruct)")
    ap.add_argument("--selector-res", type=int, default=None,
                    help="smolvlm only: processor longest_edge (raise to read fine UI detail)")
    ap.add_argument("--vlm-model", default=None,
                    help="twostage only: Stage-2 VLM model id (default gpt-5.5)")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="twostage only: gpt-5.5 reasoning_effort (default low)")
    ap.add_argument("--k", type=int, default=6, help="frames to select in condition C")
    ap.add_argument("--dump-fps", type=float, default=1.0, help="candidate-frame sampling rate for video")
    ap.add_argument("--max-dump-frames", type=int, default=64, help="cap on condition A frames")
    ap.add_argument("--max-side", type=int, default=768, help="downscale long image side before sending")
    ap.add_argument("--judge", action="store_true", help="use LLM judge for accuracy (else exact match)")
    ap.add_argument("--judge-provider", default=None, help="defaults to --answerer provider")
    ap.add_argument("--judge-model", default="gpt-4.1",
                    help="judge model (cheap non-reasoning; decoupled from answerer --model)")
    ap.add_argument("--detail", default=None, choices=["low", "high", "auto"],
                    help="openai image detail: low=~85 tok/frame (full-dump), high=fine-grained (top-k)")
    ap.add_argument("--label", default=None,
                    help="arm label for Fork-A analysis (e.g. knob / U / C-so400m); condition stays A/C")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    answerer = build_answerer(args.answerer, args.model, max_side=args.max_side, detail=args.detail)
    judge_fn = None
    if args.judge:
        judge_fn = make_text_judge(args.judge_provider or args.answerer, args.judge_model)

    c_selector = build_selector(args.selector, args.selector_model, args.selector_res,
                               args.vlm_model, args.reasoning_effort)

    all_rows: List[dict] = []
    for cond in args.conditions:
        selector: Selector = FullDumpSelector() if cond == "A" else c_selector
        print(f"== condition {cond} (selector={selector.name}, answerer={answerer.name}) ==")
        rows = run_condition(
            cond, selector, answerer, items, args.k, args.dump_fps, args.max_dump_frames, judge_fn,
            arm=args.label,
        )
        all_rows.extend(rows)

    summary = summarize(all_rows)
    by_bin = summarize_by_bin(all_rows)

    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.jsonl")
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    bin_path = os.path.join(args.out, "summary_by_bin.json")
    with open(bin_path, "w", encoding="utf-8") as f:
        json.dump(by_bin, f, indent=2)

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
