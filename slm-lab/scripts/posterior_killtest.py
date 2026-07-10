#!/usr/bin/env python3
"""Idea 1 kill-test: option-posterior frame selection with NO answerer model.

Question: does SigLIP alone -- frame embeddings x the 4 MCQA option strings, with no
VLM/GPT reasoning loop at all -- carry enough option-discriminative signal to answer LVB
MCQA from cosine similarities directly (a naive-Bayes posterior over per-frame option
cosines)? If the per-frame option margin is tiny and this selector-only posterior can't
beat the blind-GPT floor, frame *choice* is not the bottleneck and a cheap SigLIP-only
selector idea is dead on arrival -- no need to build the more elaborate scorer.

Pipeline per question:
  1. Embed every candidate frame + the 4 option strings + the bare question with SigLIP
     (google/siglip-so400m-patch14-384), cosine similarity via normalized get_image_features /
     get_text_features (NOT logits_per_image -- that bakes in the model's learned scale/bias,
     we want raw cosine units so the 0.01 kill threshold is meaningful).
  2. Per-frame option margin = top1 - top2 of the 4 option cosines for that frame.
  3. Selector-only MCQA answer: take the top-R frames by question-relevance cosine, sum their
     4 option-cosine vectors, divide by tau, softmax, argmax -> predicted letter. NOTE: dividing
     a single accumulated logit vector by a scalar tau does not change its argmax, so the
     predicted letter (and therefore accuracy) is mathematically invariant to tau here -- tau
     only reshapes the softmax *confidence*. This is expected, not a bug; see summary note.
  4. Greedy KL-movement selection @ k: sequential Bayesian-style update, each round pick the
     unpicked frame whose option-cosine contribution maximizes KL(posterior_after||before).
     Compare the resulting index set to the plain top-k-by-question-cosine set.

Frame IDs / bins come from results/scores/scores.jsonl (the Stage-1 cache) so this reuses
exactly the same question set as the rest of the fork; falls back to "all manifest items with
an extracted frame dir" if that cache is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

LETTERS = "ABCDEFGH"  # LVB has both 4-way and 5-way MCQA items; cover a safe margin
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def bare_question(question: str) -> str:
    idx = question.find("Options:")
    q = question[:idx] if idx != -1 else question
    return q.strip()


def list_frame_files(frames_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)


def percentile(vals, p):
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def load_rows(scores_path: str, frames_root: str, manifest: dict) -> List[dict]:
    if os.path.exists(scores_path):
        rows = [json.loads(l) for l in open(scores_path) if l.strip()]
        out = []
        for r in rows:
            item = manifest.get(r["id"])
            if item is None:
                continue
            out.append({"id": r["id"], "bin": str(r["length_bin"]), "item": item})
        return out
    # fallback: every manifest item with an extracted frame dir
    out = []
    for iid, item in manifest.items():
        d = os.path.join(frames_root, iid)
        if os.path.isdir(d):
            out.append({"id": iid, "bin": str(item.get("length_bin", "?")), "item": item})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.frames.100.json")
    ap.add_argument("--scores", default="results/scores/scores.jsonl")
    ap.add_argument("--frames-root", default="data/frames")
    ap.add_argument("--model-id", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau-grid", default="0.005,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--greedy-tau", type=float, default=0.05,
                     help="tau for the greedy KL selector. Step-3 accuracy is provably "
                          "tau-invariant (see module docstring) so there is no empirical "
                          "'best tau' to inherit -- this is a documented tie-break at the "
                          "middle of the grid, not a tuned value.")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of questions")
    ap.add_argument("--out", default="results/posterior_killtest.json")
    args = ap.parse_args()

    tau_grid = [float(x) for x in args.tau_grid.split(",")]

    manifest = {it["id"]: it for it in json.load(open(args.manifest))}
    rows = load_rows(args.scores, args.frames_root, manifest)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} questions to process", file=sys.stderr)

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    if device == "cuda":
        model = model.half()
    processor = AutoProcessor.from_pretrained(args.model_id)

    @torch.no_grad()
    def embed_images(images):
        feats = []
        for i in range(0, len(images), args.batch_size):
            chunk = images[i:i + args.batch_size]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            if device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            out = model.get_image_features(**inputs)
            # transformers>=5 returns BaseModelOutputWithPooling (vision_model's raw output)
            # instead of a bare tensor; the pooled embedding is .pooler_output. Support both
            # so this keeps working across transformers versions.
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.float().cpu())
        return torch.cat(feats, dim=0).numpy()

    @torch.no_grad()
    def embed_texts(texts):
        inputs = processor(text=texts, return_tensors="pt", padding="max_length",
                            max_length=64, truncation=True).to(device)
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    records = []
    all_margins = {"60": [], "600": []}
    selector_correct = {"60": {t: 0 for t in tau_grid}, "600": {t: 0 for t in tau_grid}}
    selector_total = {"60": 0, "600": 0}
    overlaps = {"60": [], "600": []}

    for n_done, row in enumerate(rows):
        iid, bin_, item = row["id"], row["bin"], row["item"]
        if bin_ not in all_margins:
            all_margins[bin_] = []
            selector_correct[bin_] = {t: 0 for t in tau_grid}
            selector_total[bin_] = 0
            overlaps[bin_] = []

        frames_dir = os.path.join(args.frames_root, iid)
        if not os.path.isdir(frames_dir):
            print(f"  !! missing frames dir {frames_dir}, skip", file=sys.stderr)
            continue
        files = list_frame_files(frames_dir)
        if not files:
            print(f"  !! no frame files in {frames_dir}, skip", file=sys.stderr)
            continue

        cands = item["candidates"]
        n_opt = len(cands)
        gold = item["gold_letter"]
        qtext = bare_question(item["question"])
        # LVB is NOT uniformly 4-way MCQA: 274/400 manifest items (124/200 in the
        # scores.jsonl subset) have 5 candidates. Handle any option count generically
        # rather than hard-coding 4 -- hard-coding would silently drop the majority class.

        images = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in files]
        img_emb = embed_images(images)                  # [F, D]
        txt_emb = embed_texts(cands + [qtext])           # [n_opt+1, D] -> 0..n_opt-1 options, last = question

        sims = img_emb @ txt_emb.T                       # [F, n_opt+1]
        opt_sims = sims[:, :n_opt]                        # [F, n_opt]
        q_sims = sims[:, n_opt]                            # [F]

        # --- per-frame option margin (top1 - top2 of the n_opt option cosines) ---
        sorted_opts = np.sort(opt_sims, axis=1)
        margin = sorted_opts[:, -1] - sorted_opts[:, -2]
        all_margins[bin_].extend(margin.tolist())

        # --- selector-only naive-Bayes MCQA, R frames by question-relevance ---
        R = min(args.R, len(files))
        top_r_idx = np.argsort(-q_sims)[:R]
        S = opt_sims[top_r_idx].sum(axis=0)              # [n_opt] accumulated logit-ish sum

        selector_preds = {}
        for tau in tau_grid:
            logits = S / tau
            m = logits.max()
            probs = np.exp(logits - m)
            probs = probs / probs.sum()
            pred_idx = int(np.argmax(logits))
            selector_preds[str(tau)] = {"pred": LETTERS[pred_idx], "probs": probs.tolist()}
        selector_total[bin_] += 1
        for tau in tau_grid:
            if selector_preds[str(tau)]["pred"] == gold:
                selector_correct[bin_][tau] += 1

        # --- greedy KL-movement selection @ k ---
        tau_g = args.greedy_tau
        k = min(args.k, len(files))
        picked: List[int] = []
        remaining = list(range(len(files)))
        acc_logits = np.zeros(n_opt)
        post_before = np.full(n_opt, 1.0 / n_opt)
        for _ in range(k):
            best_i, best_kl = None, -1.0
            for i in remaining:
                cand_logits = acc_logits + opt_sims[i] / tau_g
                m = cand_logits.max()
                p_after = np.exp(cand_logits - m)
                p_after = p_after / p_after.sum()
                kl = float(np.sum(p_after * (np.log(np.clip(p_after, 1e-12, None)) -
                                              np.log(np.clip(post_before, 1e-12, None)))))
                if kl > best_kl:
                    best_kl, best_i = kl, i
            picked.append(best_i)
            remaining.remove(best_i)
            acc_logits = acc_logits + opt_sims[best_i] / tau_g
            m = acc_logits.max()
            post_before = np.exp(acc_logits - m)
            post_before = post_before / post_before.sum()

        greedy_idx = sorted(picked)
        greedy_files = [files[i] for i in greedy_idx]
        topk_idx = sorted(np.argsort(-q_sims)[:k].tolist())
        topk_files = [files[i] for i in topk_idx]
        overlap = len(set(greedy_idx) & set(topk_idx)) / float(k)
        overlaps[bin_].append(overlap)

        records.append({
            "id": iid, "bin": bin_, "gold_letter": gold, "n_frames": len(files), "n_options": n_opt,
            "selector_pred": selector_preds,
            "margin_p10": percentile(margin.tolist(), 10),
            "margin_p50": percentile(margin.tolist(), 50),
            "margin_p90": percentile(margin.tolist(), 90),
            "greedy_indices": greedy_idx, "greedy_files": greedy_files,
            "topk_indices": topk_idx, "topk_files": topk_files,
            "overlap": overlap,
        })

        pred_mid = selector_preds.get(str(args.greedy_tau), {}).get("pred")
        print(f"[{n_done + 1}/{len(rows)}] {iid} bin={bin_} gold={gold} pred={pred_mid} "
              f"overlap={overlap:.2f} margin_p50={percentile(margin.tolist(), 50):.4f}",
              file=sys.stderr)

    bins_seen = sorted(selector_total.keys())
    summary = {"per_bin": {}, "margins_per_bin": {}, "overlap_mean": {}}
    for b in bins_seen:
        n = selector_total[b]
        summary["per_bin"][b] = {
            "n": n,
            "selector_acc": {str(t): (selector_correct[b][t] / n if n else None) for t in tau_grid},
        }
        summary["margins_per_bin"][b] = {
            "p10": percentile(all_margins[b], 10),
            "p50": percentile(all_margins[b], 50),
            "p90": percentile(all_margins[b], 90),
        }
        summary["overlap_mean"][b] = float(np.mean(overlaps[b])) if overlaps[b] else None

    pooled_margins = [x for b in bins_seen for x in all_margins[b]]
    summary["margins_pooled"] = {
        "p10": percentile(pooled_margins, 10),
        "p50": percentile(pooled_margins, 50),
        "p90": percentile(pooled_margins, 90),
    }
    pooled_overlap = [x for b in bins_seen for x in overlaps[b]]
    summary["overlap_mean"]["overall"] = float(np.mean(pooled_overlap)) if pooled_overlap else None

    summary["params"] = {
        "model_id": args.model_id, "R": args.R, "k": args.k, "tau_grid": tau_grid,
        "greedy_tau": args.greedy_tau,
        "note_tau_invariance": (
            "Selector-only argmax (step 3) is mathematically invariant to tau: dividing the "
            "summed option-cosine logits by one positive scalar tau does not change which "
            "option has the max logit, so softmax argmax -- and hence accuracy -- is identical "
            "across the whole tau grid by construction (mod float noise at extreme tau). tau "
            "only reshapes softmax *confidence*, not the predicted letter. This is an expected "
            "property of this formulation, not a bug; the tau sweep is informative for the "
            "greedy-KL step (which is NOT tau-invariant) but not for step-3 accuracy."
        ),
    }

    best_acc_600 = (max(summary["per_bin"]["600"]["selector_acc"].values())
                     if "600" in summary["per_bin"] and selector_total.get("600") else None)
    summary["kill_test"] = {
        "median_margin_pooled": summary["margins_pooled"]["p50"],
        "best_selector_acc_600": best_acc_600,
        "mean_overlap_overall": summary["overlap_mean"]["overall"],
    }

    out = {"records": records, "summary": summary}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
