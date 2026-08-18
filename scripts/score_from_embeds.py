#!/usr/bin/env python3
"""Per-frame scores from CACHED image embeds + a freshly encoded STEM-ONLY query.

Why this exists: dump_scores.py / dump_scores_longclip.py re-decode every video at
1 fps and re-run the image tower -- hours for the 600s bin. But the image towers were
never contaminated (dump_embeds.py only ever calls get_image_features/encode_image),
so the cached embeds are reusable as-is. Only the TEXT side was wrong. Re-encoding
~760 short strings is seconds; re-encoding ~180k frames is not.

Output schema is byte-identical to dump_scores.py ({id, length_bin, times, scores,
gold_evidence_seconds}) so export_picks_lmmseval.py --from-scores works unchanged.

Embed sources (per bin, resolved automatically):
  results/embeds/<qid>.npz     keys: times, siglip (N,1152), longclip (N,768), dinov2
  results/embeds_lc/<qid>.npz  keys: times, emb (N,768)   <- the 600s LongCLIP fill

RANKING EQUIVALENCE (why cosine here == the original scorers):
  * LongCLIP: dump_scores_longclip.py scores cosine of L2-normalized pairs. Identical.
  * SigLIP:   dump_scores.py uses logits_per_image = logit_scale*cosine + logit_bias.
    For a FIXED text that is a strictly monotone transform of cosine, so argsort --
    and therefore top-k -- is identical. Absolute values differ; ranks do not. Since
    every consumer of this file ranks, that is the property that matters.

Precision: cached embeds are fp16, the original dumps ran fp32. Scores agree to ~1e-3,
ranks to Spearman ~1.0. --validate quantifies this against an archived score file
instead of asserting it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from harness.embeds import l2, load_image_embed
from harness.manifest import check_gold_reliable
from harness.text import question_stem


def build_text_encoder(scorer: str, device: str, repo_dir: str, ckpt_repo: str, ckpt_file: str):
    """-> encode(list[str]) -> (B, D) L2-normalized float32."""
    import torch

    if scorer == "sig":
        from transformers import AutoModel, AutoProcessor

        mid = "google/siglip-so400m-patch14-384"
        model = AutoModel.from_pretrained(mid).to(device).eval()
        proc = AutoProcessor.from_pretrained(mid)

        def _pooled(o):
            """get_text_features returns a bare tensor on some transformers versions and
            a BaseModelOutputWithPooling on others. SiglipModel.forward uses
            text_outputs[1] (the pooled vector) to build logits_per_image, so pooler_output
            is the tensor that reproduces the original scorer -- last_hidden_state is not.
            """
            if torch.is_tensor(o):
                return o
            po = getattr(o, "pooler_output", None)
            if po is None:
                raise TypeError(f"SigLIP text output has no pooler_output: {type(o)}")
            return po

        def encode(texts):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    # tokenization MUST match dump_scores.py exactly: 64 positions,
                    # padded to max_length, truncated. Anything else silently shifts
                    # which tokens the tower sees.
                    inp = proc(text=texts[i:i + 64], return_tensors="pt",
                               padding="max_length", max_length=64,
                               truncation=True).to(device)
                    out.append(_pooled(model.get_text_features(**inp)).float().cpu().numpy())
            return l2(np.concatenate(out))

        return encode

    sys.path.insert(0, repo_dir)
    from huggingface_hub import hf_hub_download
    from model import longclip  # noqa: E402

    ck = hf_hub_download(ckpt_repo, ckpt_file)
    model, _ = longclip.load(ck, device=device)
    model = model.float().eval()  # CLIP keeps LayerNorm fp32; mixing crashes

    def encode(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                tok = longclip.tokenize(texts[i:i + 64], truncate=True).to(device)
                out.append(model.encode_text(tok).float().cpu().numpy())
        return l2(np.concatenate(out))

    return encode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scorer", required=True, choices=["sig", "lc"])
    ap.add_argument("--bins", nargs="+", default=None)
    ap.add_argument("--emb-dir", default="results/embeds")
    ap.add_argument("--emb-lc-dir", default="results/embeds_lc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--repo-dir", default="Long-CLIP")
    ap.add_argument("--ckpt-repo", default="BeichenZhang/LongCLIP-L")
    ap.add_argument("--ckpt-file", default="longclip-L.pt")
    ap.add_argument("--gold-reliable-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--text-embeds-out", default=None,
                    help="also dump the stem text embeds (qid-keyed) — OMP needs these")
    ap.add_argument("--validate-against", default=None,
                    help="an OLD scores jsonl; report score/rank agreement. Use with "
                         "--validate-query to prove this path reproduces the original.")
    ap.add_argument("--validate-query", choices=["stem", "full"], default="stem",
                    help="'full' = re-encode the ORIGINAL fused prompt, which should "
                         "reproduce an archived tainted file and thus prove the pipeline")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    check_gold_reliable(manifest, args.gold_reliable_only, args.manifest)
    bins = set(b.rstrip("s") for b in args.bins) if args.bins else None

    items = []
    for it in manifest:
        b = str(it.get("length_bin", "all")).rstrip("s")
        if bins and b not in bins:
            continue
        if args.gold_reliable_only and not it.get("gold_reliable"):
            continue
        items.append((it, b))
    print(f"manifest {args.manifest}: {len(items)} items in bins={sorted(bins) if bins else 'ALL'}")

    # resolve embeds FIRST -> never load a model just to discover the cache is empty
    resolved, missing = [], []
    for it, b in items:
        t, e = load_image_embed(it["id"], args.scorer, args.emb_dir, args.emb_lc_dir)
        (missing if t is None else resolved).append((it, b, t, e))
    print(f"image embeds: {len(resolved)} cached, {len(missing)} MISSING")
    if missing:
        print("  missing ids (first 5): " + ", ".join(m[0]["id"] for m in missing[:5]))
    if not resolved:
        raise SystemExit("REFUSING: no cached image embeds matched — wrong --emb-dir or bin?")

    if args.validate_query == "full":
        texts = [it.get("question", "") for it, _, _, _ in resolved]
        print("!! --validate-query=full: encoding the ORIGINAL FUSED prompt (stem+options).")
        print("!! This output is for pipeline validation ONLY. Never feed it to picks.")
    else:
        texts = [question_stem(it) for it, _, _, _ in resolved]

    # loud proof of what the tower actually sees — the check that was missing all week
    print("\n=== QUERY THE SCORER ACTUALLY RECEIVES (first 3) ===")
    for t in texts[:3]:
        print(f"  {t!r}")
    bad = [t for t in texts if "Options:" in t or "single option letter" in t]
    if args.validate_query == "stem":
        if bad:
            raise SystemExit(f"REFUSING: {len(bad)} queries still contain option text.")
        print(f"=== {len(texts)} queries, 0 option leaks ===\n")
    else:
        # never print "0 option leaks" over prompts that are visibly full of options —
        # a log that contradicts its own data is how the original bug survived a week
        print(f"=== {len(texts)} queries, {len(bad)} CONTAIN OPTIONS (expected: "
              f"--validate-query=full) ===\n")

    dev = args.device
    try:
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
            print("cuda unavailable -> cpu")
    except Exception:
        dev = "cpu"

    enc = build_text_encoder(args.scorer, dev, args.repo_dir, args.ckpt_repo, args.ckpt_file)
    print(f"encoding {len(texts)} stem queries with {args.scorer} text tower on {dev} ...")
    T = enc(texts)  # (n, D) normalized

    if args.text_embeds_out:
        os.makedirs(os.path.dirname(args.text_embeds_out) or ".", exist_ok=True)
        np.savez(args.text_embeds_out,
                 ids=np.array([it["id"] for it, _, _, _ in resolved]),
                 text=T.astype(np.float16))
        print(f"wrote {args.text_embeds_out}  ({T.shape[0]} qids, STEM-ONLY)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (it, b, times, emb), tv in zip(resolved, T):
            scores = (l2(emb) @ tv.astype(np.float32)).tolist()
            f.write(json.dumps({
                "id": it["id"],
                "length_bin": b,
                "times": [float(x) for x in times],
                "scores": [float(x) for x in scores],
                "gold_evidence_seconds": it.get("gold_evidence_seconds"),
            }) + "\n")
            n += 1
    print(f"wrote {args.out} ({n} items)")

    if args.validate_against and os.path.exists(args.validate_against):
        old = {}
        for line in open(args.validate_against):
            if line.strip():
                d = json.loads(line)
                old[d["id"]] = d
        rs, ks, both = [], [], 0
        new = {}
        for line in open(args.out):
            d = json.loads(line)
            new[d["id"]] = d
        for qid in set(old) & set(new):
            a = np.asarray(old[qid]["scores"], dtype=np.float64)
            c = np.asarray(new[qid]["scores"], dtype=np.float64)
            if a.shape != c.shape or a.size < 3:
                continue
            both += 1
            ra = np.argsort(np.argsort(-a))
            rc = np.argsort(np.argsort(-c))
            rs.append(np.corrcoef(ra, rc)[0, 1])
            for k in (8,):
                ks.append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-c)[:k])) / k)
        if both:
            print(f"\n=== VALIDATION vs {args.validate_against} (query={args.validate_query}) ===")
            print(f"  items compared     : {both}")
            print(f"  rank corr (Spearman): mean={np.mean(rs):.4f} min={np.min(rs):.4f}")
            print(f"  top-8 overlap       : mean={np.mean(ks):.4f}")
            print("  (query=full should be ~1.0 -> this path reproduces the original scorer.")
            print("   query=stem SHOULD differ -> that difference IS the bug being fixed.)")
        else:
            print("\n!! validation: no overlapping ids with comparable shapes")


if __name__ == "__main__":
    main()
