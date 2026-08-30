#!/usr/bin/env python3
"""Per-frame scores + text embeds under the FUSED query (stem + options).

Deliberately separate from score_from_embeds.py: that script's stem path carries a
hard refusal if option text reaches the scorer, and that guard exists because a
fused query was once a BUG. Here fused is the experimental condition (R12), so the
guard is inverted -- we refuse if options are ABSENT -- rather than weakened in the
audited stem path.

Fused query = the manifest "question" minus the answerer instruction. It is NOT
it["question"] verbatim: that ends with "Answer with the single option letter.",
which is an instruction to the answerer, not part of the query being studied.
"""
import os

# Sentinel: with no --emb-dir the stem-embedding lookup must miss, which is what
# the 600s run did. Kept explicit so the fallback path stays visible.
_NO_STEM_EMBEDS = "/nonexistent_emb_dir"
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("SLM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness.embeds import l2, load_image_embed
from scripts.score_from_embeds import build_text_encoder

INSTR = "\n\nAnswer with the single option letter."

def question_fused(item) -> str:
    q = item.get("question", "") or ""
    if q.endswith(INSTR):
        q = q[: -len(INSTR)]
    return q.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bins", nargs="+", required=True)
    ap.add_argument("--emb-lc-dir", required=True)
    # 3600s LongCLIP lives under key "longclip" in results/embeds/, not as
    # "emb" in embeds_lc/ (which only ever held the 600s fill). Default None
    # keeps the audited 600s invocation byte-identical.
    ap.add_argument("--emb-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-embeds-out", required=True)
    ap.add_argument("--repo-dir", default="/data/slm-lab/Long-CLIP")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    bins = set(b.rstrip("s") for b in a.bins)
    items = [it for it in man if str(it.get("length_bin","")).rstrip("s") in bins]
    print(f"manifest: {len(items)} items in bins={sorted(bins)}")

    resolved, missing = [], []
    for it in items:
        t, e = load_image_embed(it["id"], "lc", a.emb_dir or _NO_STEM_EMBEDS, a.emb_lc_dir)
        (missing if t is None else resolved).append((it, t, e))
    print(f"image embeds: {len(resolved)} cached, {len(missing)} MISSING")
    if missing:
        print("  missing:", [m[0]["id"] for m in missing[:5]])
    if not resolved:
        raise SystemExit("REFUSING: no cached image embeds")

    texts = [question_fused(it) for it, _, _ in resolved]
    print("\n=== QUERY THE SCORER ACTUALLY RECEIVES (first 2) ===")
    for t in texts[:2]:
        print(f"  {t!r}\n")
    # inverted guard: fused MUST contain options
    n_opt = sum(1 for t in texts if "Options:" in t)
    n_instr = sum(1 for t in texts if "single option letter" in t)
    print(f"=== {len(texts)} queries | {n_opt} contain Options: | {n_instr} still carry the answer instruction ===")
    if n_opt != len(texts):
        raise SystemExit(f"REFUSING: {len(texts)-n_opt} fused queries have NO options — not a fused arm.")
    if n_instr:
        raise SystemExit(f"REFUSING: {n_instr} queries still carry the answerer instruction.")

    enc = build_text_encoder("lc", a.device, a.repo_dir, "BeichenZhang/LongCLIP-L", "longclip-L.pt")
    T = enc(texts)                      # (B, D) L2-normalised float32
    print("text embeds:", T.shape)

    ids = [it["id"] for it, _, _ in resolved]
    os.makedirs(os.path.dirname(a.text_embeds_out) or ".", exist_ok=True)
    np.savez(a.text_embeds_out, ids=np.array(ids), text=T)
    print("wrote", a.text_embeds_out)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for (it, times, emb), t in zip(resolved, T):
            E = l2(emb.astype(np.float32))
            s = (E @ t.astype(np.float32)).tolist()
            f.write(json.dumps({"id": it["id"],
                                "length_bin": str(it.get("length_bin")),
                                "times": [float(x) for x in times],
                                "scores": s,
                                "gold_evidence_seconds": it.get("gold_evidence_seconds")}) + "\n")
    print(f"wrote {a.out}: {len(resolved)} rows")

if __name__ == "__main__":
    main()
