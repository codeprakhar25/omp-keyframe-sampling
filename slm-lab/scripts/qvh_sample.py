#!/usr/bin/env python3
"""QVHighlights step-by-step sample: download a few videos, SigLIP cosine curve per 2-s clip,
run OUR peak-NMS on the curve, plot vs gold window for MANUAL inspection. Not an experiment —
a visual sanity check that the curve rides the gold plateau and our peak picker lands on it.

Per sample qid: yt-dlp the 150-s section -> 1 frame / 2-s clip -> SigLIP so400m cosine(query, clip)
-> build_union_indices (our peak-NMS) + plain top-k -> matplotlib curve (gold shaded, picks marked)
-> save frames + curve.png + meta.json. scp the outdir home to eyeball.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from harness.union_retrieval import build_union_indices


def dl_video(ytid, start, end, out):
    if os.path.exists(out):
        return True
    url = f"https://www.youtube.com/watch?v={ytid}"
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=480]+ba/b[height<=480]",
           "--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts",
           "-o", out, url]
    return subprocess.run(cmd).returncode == 0 and os.path.exists(out)


def extract_clips(video, outdir, clip_sec=2.0):
    os.makedirs(outdir, exist_ok=True)
    # one frame at the midpoint of each 2-s clip
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-vf", f"fps=1/{clip_sec}", os.path.join(outdir, "c%03d.jpg")])
    return sorted(os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".jpg"))


def siglip_scores(frame_paths, query, model, proc, torch, device):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    scores = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            chunk = imgs[i:i + 64]
            inp = proc(text=[query], images=chunk, return_tensors="pt",
                       padding="max_length", max_length=64, truncation=True).to(device)
            scores.append(model(**inp).logits_per_image.squeeze(-1).float().cpu())
    return torch.cat(scores).numpy()


def plot(times, scores, gold_windows, peak_idx, topk_idx, query, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, scores, "-o", ms=3, color="#444", lw=1, label="SigLIP cosine")
    for (b, e) in gold_windows:
        ax.axvspan(b, e, color="#4caf50", alpha=0.22, label="gold moment")
    ax.scatter([times[i] for i in topk_idx], [scores[i] for i in topk_idx],
               color="#1e88e5", s=60, zorder=5, label=f"top-{len(topk_idx)}")
    ax.scatter([times[i] for i in peak_idx], [scores[i] for i in peak_idx],
               facecolors="none", edgecolors="#e53935", s=140, lw=2, zorder=6, label="peak-NMS pick")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cosine"); ax.set_title(query[:90])
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h)); ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--outdir", default="results/qvh_sample")
    ap.add_argument("--clip-sec", type=float, default=2.0)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    samples = json.load(open(args.samples))
    summary = []
    for s in samples:
        qid = s["qid"]; ytid, st, en = s["vid"].split("_"); st, en = float(st), float(en)
        d = os.path.join(args.outdir, str(qid)); os.makedirs(d, exist_ok=True)
        vid_mp4 = os.path.join(d, "video.mp4")
        clipdir = os.path.join(d, "clips")
        print(f"\n=== qid{qid} {ytid} [{st:.0f}-{en:.0f}] :: {s['query']}")
        existing = sorted(os.path.join(clipdir, f) for f in os.listdir(clipdir)) if os.path.isdir(clipdir) else []
        existing = [f for f in existing if f.endswith(".jpg")]
        if existing:
            print(f"  reusing {len(existing)} pre-extracted frames")
            frames = existing
        else:
            if not dl_video(ytid, st, en, vid_mp4):
                print("  !! download failed, skip"); summary.append({"qid": qid, "ok": False}); continue
            frames = extract_clips(vid_mp4, clipdir, args.clip_sec)
        if not frames:
            print("  !! no frames"); continue
        scores = siglip_scores(frames, s["query"], model, proc, torch, device)
        times = [args.clip_sec * i + args.clip_sec / 2 for i in range(len(scores))]
        peak_idx = build_union_indices(scores.tolist(), times, budget=args.budget)
        topk_idx = sorted(np.argsort(-scores)[: args.topk].tolist())
        gold = s["relevant_windows"]
        gold_clip = set(s["relevant_clip_ids"])
        peak_hit = [i for i in peak_idx if i in gold_clip]
        plot(times, scores, gold, peak_idx, topk_idx, s["query"], os.path.join(d, "curve.png"))
        meta = {"qid": qid, "ok": True, "query": s["query"], "n_clips": len(scores),
                "gold_windows": gold, "gold_clip_ids": sorted(gold_clip),
                "peak_nms_idx": peak_idx, "topk_idx": topk_idx,
                "peak_in_gold": peak_hit, "peak_precision": round(len(peak_hit)/max(1,len(peak_idx)),2),
                "topk_in_gold": [i for i in topk_idx if i in gold_clip],
                "plateau_gap": round(float(scores[list(gold_clip)].mean() -
                                    np.delete(scores, list(gold_clip)).mean()), 4) if gold_clip else None}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
        print(f"  clips {len(scores)} | peak-NMS {peak_idx} | in-gold {peak_hit} "
              f"| plateau_gap {meta['plateau_gap']}")
        summary.append(meta)
    json.dump(summary, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {args.outdir}/summary.json")


if __name__ == "__main__":
    main()
