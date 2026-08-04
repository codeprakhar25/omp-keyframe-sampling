#!/usr/bin/env python3
"""Holdout path eval for Fireworks fine-tuned placer LoRA."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

import importlib.util

from harness.store import path_key  # noqa: E402


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def soft_score(gold: list[str], pred: list[str] | None) -> dict:
    if not pred:
        return {
            "exact": False,
            "soft_hit": False,
            "branch_ok": False,
            "soft_relation": "empty",
        }
    exact = gold == pred
    same_root = bool(gold) and bool(pred) and gold[0] == pred[0]
    if exact:
        rel = "exact"
    elif len(gold) <= len(pred) and pred[: len(gold)] == gold:
        rel = "gold_prefix"
    elif len(pred) <= len(gold) and gold[: len(pred)] == pred:
        rel = "pred_prefix"
    elif same_root:
        rel = "same_root"
    else:
        rel = "diff_root"
    return {
        "exact": exact,
        "soft_hit": rel in ("exact", "gold_prefix", "pred_prefix"),
        "branch_ok": same_root,
        "soft_relation": rel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data" / "multitree_synth_smoke" / "place_holdout.jsonl",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "fireworks_placer_smoke")
    ap.add_argument(
        "--model",
        default=(
            "accounts/prakharkhatri123-edp/models/placer-smoke-llama31-8b"
            "#accounts/prakharkhatri123-edp/deployments/placer-smoke-llama31-8b-live"
        ),
        help="Fireworks model id; live-merge needs model#deployment or deployment id",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--provider",
        default="fireworks",
        choices=["fireworks", "openrouter", "openai"],
        help="openrouter serves the untuned base weights; Fireworks dropped "
        "llama-3.1-8b from serverless and the LoRA deployments are stopped. "
        "openai is here so every arm goes through one prompt/parser/scorer",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=None,
        help="HierStore sqlite — shared existing_dirs for all tasks (user-dir transfer)",
    )
    args = ap.parse_args()

    if args.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
        base_url = "https://openrouter.ai/api/v1"
    elif args.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        base_url = None
    else:
        key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API")
        base_url = "https://api.fireworks.ai/inference/v1"
    if not key:
        raise SystemExit(f"missing API key for provider={args.provider}")

    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    probe = _load_probe()
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    shared_dirs = None
    if args.store:
        shared_dirs = probe.dirs_from_store(
            args.store, tasks[0].get("roots") or ["work", "personal", "inbox"]
        )
        print(f"existing_dirs from {args.store}: n={len(shared_dirs)}", flush=True)

    rows = []
    pt = ct = 0
    parse_fail = 0
    for t in tasks:
        roots = t["roots"]
        if shared_dirs is not None:
            existing = shared_dirs
        else:
            existing = t.get("existing_dirs") or [[r] for r in roots]
        try:
            pred = probe.call_placer(
                client,
                model=args.model,
                roots=roots,
                max_depth=5,
                text=t["text"],
                kind="place",
                cue=None,
                from_path=None,
                existing_dirs=existing,
            )
            path = pred["path"]
            pt += pred["usage"]["prompt_tokens"]
            ct += pred["usage"]["completion_tokens"]
            raw = pred.get("raw", "")
        except Exception as e:  # noqa: BLE001
            path = None
            parse_fail += 1
            raw = str(e)
            print(f"FAIL {t['id']} {e}", flush=True)

        sc = soft_score(t["gold_path"], path)
        rows.append(
            {
                "id": t["id"],
                "tree_id": t.get("tree_id"),
                "gold_path": t["gold_path"],
                "pred_path": path,
                "raw": raw[:500] if isinstance(raw, str) else raw,
                **sc,
            }
        )
        mark = "OK" if sc["exact"] else ("SOFT" if sc["soft_hit"] else "MISS")
        print(
            f"{mark} {t['id']} gold={path_key(t['gold_path'])} "
            f"pred={path_key(path) if path else None}",
            flush=True,
        )

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "model": args.model,
        "provider": args.provider,
        "tasks": str(args.tasks),
        "path_exact": sum(r["exact"] for r in rows) / n,
        "path_soft": sum(r["soft_hit"] for r in rows) / n,
        "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        "parse_fail": parse_fail,
        "chat_prompt_tokens": pt,
        "chat_completion_tokens": ct,
        "gpt4o_baseline": {
            "path_exact": 0.5222222222222223,
            "path_soft": 0.9888888888888889,
            "branch_ok": 0.9888888888888889,
            "n": 90,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
