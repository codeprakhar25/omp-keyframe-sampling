#!/usr/bin/env python3
"""Token audit for CLEAN stem scorer query (post options bug).

Scorer text = question_stem only (SigLIP 64 / LongCLIP 248).
Also reports fused full-question tokens for contrast (what the bug used).
Answerer = lmms-eval prompt text tokens (no truncate in harness).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SLM = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/slm-lab")
IDS = Path(sys.argv[2])
OUT = Path(sys.argv[3])

sys.path.insert(0, str(SLM))
from harness.text import question_stem  # noqa: E402

ids = [l.strip() for l in IDS.read_text().splitlines() if l.strip()]
manifest = {}
for mf in ("data/manifest.lvb.full1560.json", "data/manifest.lvb.long976.json"):
    p = SLM / mf
    if p.exists():
        for x in json.loads(p.read_text()):
            manifest[x["id"]] = x

SIG_MAX, LC_MAX = 64, 248

from transformers import AutoProcessor, AutoTokenizer

sig_tok = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384").tokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)

sys.path.insert(0, str(SLM / "Long-CLIP"))
from model import longclip  # noqa: E402
from model.simple_tokenizer import SimpleTokenizer


def lc_stats(text: str):
    st = SimpleTokenizer()
    n_raw = len(st.encode(text)) + 2
    tok = longclip.tokenize([text], truncate=True)
    kept = int((tok[0] != 0).sum().item()) if hasattr(tok, "sum") else LC_MAX
    return n_raw, n_raw > LC_MAX, kept


def sig_stats(text: str):
    full = sig_tok(text, add_special_tokens=True, truncation=False)["input_ids"]
    kept = sig_tok(text, add_special_tokens=True, truncation=True, max_length=SIG_MAX)["input_ids"]
    kept_text = sig_tok.decode(kept, skip_special_tokens=True)
    return len(full), len(kept), len(full) > SIG_MAX, kept_text


rows = []
for qid in ids:
    item = manifest[qid]
    stem = question_stem(item)
    fused = item["question"]  # old buggy scorer input
    # lmms-eval style answerer prompt (approx from sample input pattern)
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(item["candidates"]))
    ans_prompt = f"{stem}\n{opts}\nAnswer with the option's letter from the given choices directly."

    s_n, s_k, s_t, s_kept = sig_stats(stem)
    sf_n, sf_k, sf_t, sf_kept = sig_stats(fused)
    l_n, l_t, l_k = lc_stats(stem)
    lf_n, lf_t, lf_k = lc_stats(fused)
    qwen_n = len(qwen_tok(ans_prompt, add_special_tokens=True)["input_ids"])

    rows.append({
        "id": qid,
        "stem": stem,
        "stem_chars": len(stem),
        "siglip_stem": {
            "max_length": SIG_MAX,
            "n_tokens_full": s_n,
            "n_tokens_kept": s_k,
            "truncated": s_t,
            "kept_text": s_kept,
        },
        "longclip_stem": {
            "max_length": LC_MAX,
            "n_tokens_full": l_n,
            "n_tokens_kept": l_k,
            "truncated": l_t,
        },
        "siglip_fused_contrast": {
            "n_tokens_full": sf_n,
            "truncated": sf_t,
            "kept_text": sf_kept,
            "note": "VOID path — what old hunt scored",
        },
        "longclip_fused_contrast": {
            "n_tokens_full": lf_n,
            "truncated": lf_t,
            "note": "VOID path — what old hunt scored",
        },
        "qwen_answerer_text": {
            "prompt_n_tokens": qwen_n,
            "truncated_in_harness": False,
        },
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2))
print(
    f"wrote {OUT} n={len(rows)} "
    f"sig_stem_trunc={sum(1 for r in rows if r['siglip_stem']['truncated'])} "
    f"lc_stem_trunc={sum(1 for r in rows if r['longclip_stem']['truncated'])}"
)
