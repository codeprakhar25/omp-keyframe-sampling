#!/usr/bin/env python3
"""Build local clean compare HTML packs (fail/success × topk-lc × OMP-lc).

Read-only vs pod: pull samples/picks/embeds/videos first, then run here.

Example:
  python scripts/_build_clean_compare_pack.py --bin 3600 --k 8 \\
    --samples results/lmmseval_matrix_clean/k8_3600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VID_SRC = ROOT / "data/videos"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

try:
    from harness.text import question_stem
except Exception:
    def question_stem(item):
        q = item["question"] if isinstance(item, dict) else item
        idx = q.find("\n\nOptions:")
        if idx == -1:
            idx = q.find("Options:")
        return (q[:idx] if idx != -1 else q).strip()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(["pip", "install", "matplotlib", "-q"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def load_manifest():
    out = {}
    for mf in [
        ROOT / "data/manifest.lvb.long976.json",
        ROOT / "data/manifest.lvb.full1560.json",
        ROOT / "data/manifest.lvb.full1560.pod.json",
    ]:
        if not mf.exists():
            continue
        for x in json.loads(mf.read_text()):
            if isinstance(x, dict) and "id" in x:
                out[x["id"]] = x
    return out


def stable(ids):
    return sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())


def parse_options(input_text: str):
    opts = []
    for m in re.finditer(r"^([A-E])\.\s*(.+)$", input_text or "", re.M):
        opts.append({"letter": m.group(1), "text": m.group(2).strip()})
    return opts


def duration(video: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


def grab(video: Path, sec: float, out_jpg: Path):
    if out_jpg.exists() and out_jpg.stat().st_size > 1000:
        return True
    dur = duration(video)
    t = min(sec, max(0.0, dur - 0.05)) if dur else sec
    for ss in (t, max(0.0, t - 0.5), 0.0):
        cmd = ["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(video),
               "-frames:v", "1", "-q:v", "2", str(out_jpg)]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and out_jpg.exists() and out_jpg.stat().st_size > 1000:
            return True
    return False


def link_video(vf: str, vdir: Path):
    src = VID_SRC / vf
    dst = vdir / vf
    if not src.exists():
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def find_samples_root(samples: Path) -> Path:
    if any(samples.glob("*samples*.jsonl")):
        return samples
    hits = list(samples.rglob("*samples*.jsonl"))
    if not hits:
        raise FileNotFoundError(f"no samples under {samples}")
    return hits[0].parent


def load_arm_rows(samples_root: Path, substr: str):
    paths = [p for p in samples_root.glob("*samples*.jsonl") if substr in p.name]
    if not paths:
        paths = [p for p in samples_root.rglob("*samples*.jsonl") if substr in p.name]
    if not paths:
        raise FileNotFoundError(substr)
    rows = {}
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        a = s["lvb_acc"]
        qid = a["id"]
        gold, pred = a["answer"], a["parsed_pred"]
        rows[qid] = {
            "id": qid,
            "gold": gold,
            "pred": pred,
            "ok": pred == gold,
            "raw": s.get("filtered_resps"),
            "input": s.get("input", ""),
            "category": a.get("question_category"),
        }
    return rows, paths[0]


def acc_from_rows(rows: dict) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows.values() if r["ok"]) / len(rows)


def load_results_acc(samples_root: Path, key: str):
    paths = list(samples_root.glob("*_results.json")) + list(samples_root.rglob("*_results.json"))
    if not paths:
        return None
    res = json.loads(paths[0].read_text())["results"]
    if key in res:
        return res[key].get("lvb_acc,none")
    return None


def letter_text(opts, letter):
    for o in opts:
        if o["letter"] == letter:
            return o["text"]
    return None


def gold_info(m: dict):
    if not m:
        return {}
    centers = []
    spans = m.get("gold_evidence_seconds") or []
    for lo, hi in spans:
        centers.append(round((float(lo) + float(hi)) / 2.0, 2))
    return {
        "position": m.get("gold_position"),
        "native_fps": m.get("native_fps"),
        "spans": spans,
        "centers": centers,
        "reliable": m.get("gold_reliable"),
        "note": "LVB position → sec via manifest gold_evidence_seconds (±1s)",
    }


def load_text_index(bin_: int):
    # prefer bin-specific, fall back
    for name in [f"text_lc_{bin_}.npz", "text_lc_600.npz", "text_lc_3600.npz"]:
        p = ROOT / "results/embeds_text" / name
        if p.exists():
            z = np.load(p, allow_pickle=True)
            ids = [str(x) for x in z["ids"]]
            text = z["text"].astype(np.float32)
            text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
            return {i: text[k] for k, i in enumerate(ids)}, p
    return {}, None


def score_curve(qid: str, text_by_id: dict, bin_: int):
    if qid not in text_by_id:
        return None
    # 600: embeds_lc/emb ; 3600: embeds/longclip
    candidates = []
    if bin_ >= 3600:
        candidates.append((ROOT / "results/embeds" / f"{qid}.npz", "longclip"))
        candidates.append((ROOT / "results/embeds_lc" / f"{qid}.npz", "emb"))
    else:
        candidates.append((ROOT / "results/embeds_lc" / f"{qid}.npz", "emb"))
        candidates.append((ROOT / "results/embeds" / f"{qid}.npz", "longclip"))
    for p, key in candidates:
        if not p.exists():
            continue
        z = np.load(p)
        if key not in z.files:
            # try fallbacks
            for k in ("emb", "longclip"):
                if k in z.files:
                    key = k
                    break
            else:
                continue
        times = z["times"].astype(np.float32)
        emb = z[key].astype(np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        scores = emb @ text_by_id[qid]
        return times, scores
    return None


def plot_score(qid, times, scores, secs_self, secs_peer, self_label, peer_label, out_png,
               gold_spans=None, gold_centers=None):
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=120)
    ax.plot(times, scores, color="#444", lw=0.9, alpha=0.85, label="LongCLIP cosine (stem)")
    if gold_spans:
        for i, (lo, hi) in enumerate(gold_spans):
            ax.axvspan(float(lo), float(hi), color="#f1c40f", alpha=0.35,
                       label="LVB gold ±1s" if i == 0 else None)

    def mark(secs, color, marker, label, z=5):
        if not secs:
            return
        xs, ys = [], []
        for s in secs:
            i = int(np.argmin(np.abs(times - float(s))))
            xs.append(float(times[i]))
            ys.append(float(scores[i]))
        ax.scatter(xs, ys, c=color, s=42, zorder=z, marker=marker, label=label,
                   edgecolors="k", linewidths=0.4)

    self_color = "#c0392b" if "OMP" in self_label or "omp" in self_label.lower() else "#2980b9"
    mark(secs_self, self_color, "o", f"{self_label} pick")
    mark(secs_peer, "#27ae60", "D", f"{peer_label} pick")
    if gold_centers:
        mark(gold_centers, "#e67e22", "*", "gold center", z=6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(qid, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def tokens_html(tok):
    if not tok:
        return ""
    s, l = tok.get("siglip_stem", {}), tok.get("longclip_stem", {})
    return (
        f"<div class=meta><b>tokens (STEM)</b> — "
        f"SigLIP≈{s.get('n_tokens_full')}/64 {'TRUNC' if s.get('truncated') else 'ok'} · "
        f"LongCLIP {l.get('n_tokens_full')}/248 {'TRUNC' if l.get('truncated') else 'ok'}</div>"
    )


def write_enriched_html(sdir: Path, pack: dict, self_label: str, peer_label: str, bin_: int, k: int, kind: str):
    meta = pack["meta"]
    title = f"CLEAN {bin_}s · {self_label} @k={k} · {kind.upper()} (+ {peer_label} compare)"
    border = "ok" if kind == "success" else "fail"
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        f"<title>{title}</title>",
        "<style>body{font:14px/1.45 system-ui;margin:24px;max-width:1180px}"
        ".case{margin:1.8em 0;padding:1em;background:#f6f6f6}"
        ".fail{border-left:4px solid #c00}.ok{border-left:4px solid #080}"
        ".row{display:flex;flex-wrap:wrap;gap:6px}.row img{height:100px;border:2px solid #666}"
        ".peer img{border-color:#27ae60}.self img{border-color:#c0392b}"
        ".gold img{border-color:#e67e22;border-width:3px}"
        ".meta{color:#444;font-size:13px}.tag{background:#cfc;padding:1px 6px;margin-right:6px}"
        ".warn{background:#ffe8a0;padding:8px;margin:12px 0}"
        "img.plot{width:100%;max-width:1000px;border:1px solid #ccc;background:#fff}</style></head><body>",
        f"<h1>{title}</h1>",
        "<div class=warn>Clean stem LongCLIP scores. Curve = cosine(stem, frame). "
        f"Colored markers = {self_label} / {peer_label} / LVB gold. "
        "Built locally from pulled artifacts (no pod writes).</div>",
        f"<p>acc={meta.get('acc')} · uniform={meta.get('uniform_acc')} · n={meta.get('n')} · "
        f"shown={len(pack['cases'])}</p>",
        '<p><a href="../../index.html">pack overview</a></p>',
    ]
    for i, c in enumerate(pack["cases"], 1):
        secs = ", ".join(f"{float(s):.1f}s" for s in c["secs"])
        peer = c.get("peer") or {}
        psecs = ", ".join(f"{float(s):.1f}s" for s in peer.get("secs", []))
        cls = "ok" if c.get("ok") else "fail"
        html.append(f"<div class='case {cls}'>")
        html.append(
            f"<div><span class=tag>{bin_}s stem</span><b>{i}. {c['id']}</b> · "
            f"<code>{c.get('video_file')}</code> ({c.get('video_seconds')}s)</div>"
        )
        html.append(
            f"<div class=meta>gold <b>{c['gold_letter']}</b>: {c['gold_answer']}</div>"
        )
        status = "OK" if c.get("ok") else "FAIL"
        html.append(
            f"<div class=meta><b>{self_label}</b> {status} · pred <b>{c['pred']}</b>: "
            f"{c.get('pred_text')} · frames [{secs}]</div>"
        )
        pok = peer.get("ok")
        pok_s = "OK" if pok else ("FAIL" if pok is False else "?")
        html.append(
            f"<div class=meta><b>{peer_label}</b> {pok_s} · pred <b>{peer.get('pred')}</b>: "
            f"{peer.get('pred_text')} · frames [{psecs}]</div>"
        )
        g = c.get("gold") or {}
        if g.get("centers"):
            gsecs = ", ".join(f"{float(s):.1f}s" for s in g["centers"])
            rel = "reliable" if g.get("reliable") else "UNRELIABLE"
            html.append(
                f"<div class=meta><b>LVB gold</b> ({rel}, fps={g.get('native_fps')}, "
                f"pos={g.get('position')}) · centers [{gsecs}]</div>"
            )
        else:
            html.append("<div class=meta><b>LVB gold</b>: not resolved</div>")
        html.append(tokens_html(c.get("tokens")))
        html.append(f"<p>{c['question_stem']}</p><ul>")
        for o in c["options"]:
            marks = []
            if o["letter"] == c["gold_letter"]:
                marks.append("GOLD")
            if o["letter"] == c["pred"]:
                marks.append(f"{self_label}-pred")
            if o["letter"] == peer.get("pred"):
                marks.append(f"{peer_label}-pred")
            extra = f" ← {', '.join(marks)}" if marks else ""
            html.append(f"<li><b>{o['letter']}</b>) {o['text']}{extra}</li>")
        html.append("</ul>")

        if c.get("score_plot"):
            html.append("<div class=meta><b>score vs time</b> (LongCLIP stem cosine)</div>")
            html.append(f"<img class=plot src='{c['score_plot']}' alt='score plot'>")
        else:
            html.append("<div class=meta><i>score plot missing (pull embeds for this qid)</i></div>")

        html.append(f"<div class=meta><b>{self_label} clips</b></div><div class='row self'>")
        for rank, sec in enumerate(c["secs"], 1):
            rel = f"data/frames/{c['id']}/r{rank}_t{float(sec):.1f}s.jpg"
            html.append(f"<div><img src='{rel}'><div class=meta>#{rank} @{float(sec):.1f}s</div></div>")
        html.append("</div>")

        html.append(f"<div class=meta><b>{peer_label} clips</b></div><div class='row peer'>")
        for rel, sec in zip(peer.get("frame_jpgs", []), peer.get("secs", [])):
            html.append(f"<div><img src='{rel}'><div class=meta>@{float(sec):.1f}s</div></div>")
        if not peer.get("frame_jpgs"):
            html.append("<div class=meta>(no peer frames)</div>")
        html.append("</div>")

        html.append("<div class=meta><b>LVB gold clips</b></div><div class='row gold'>")
        for rel, sec in zip(g.get("frame_jpgs", []), g.get("centers", [])):
            html.append(f"<div><img src='{rel}'><div class=meta>gold @{float(sec):.1f}s</div></div>")
        if not g.get("frame_jpgs"):
            html.append("<div class=meta>(no gold frames)</div>")
        html.append("</div></div>")

    html.append("</body></html>")
    (sdir / "index.html").write_text("\n".join(html))
    _ = border  # silence lint


def build_case(qid, row, arm_id, picks, manifest, peer_row=None):
    m = manifest.get(qid, {})
    opts = parse_options(row["input"])
    if not opts and m.get("candidates"):
        opts = [{"letter": LETTERS[i], "text": c} for i, c in enumerate(m["candidates"])]
    stem = question_stem(m) if m else (row["input"].split("\nA.")[0].strip() if row.get("input") else "")
    secs = list(picks.get(qid, []))
    gold, pred = row["gold"], row["pred"]
    gold_text = next((o["text"] for o in opts if o["letter"] == gold), m.get("gold_answer"))
    pred_text = next((o["text"] for o in opts if o["letter"] == pred), None)
    case = {
        "id": qid,
        "arm": arm_id,
        "video_file": m.get("video_file"),
        "media_path": m.get("media_path"),
        "video_seconds": m.get("video_seconds"),
        "category": row.get("category") or m.get("question_category"),
        "question_stem": stem,
        "options": opts,
        "gold_letter": gold,
        "gold_answer": gold_text,
        "pred": pred,
        "pred_text": pred_text,
        "ok": row["ok"],
        "raw": row["raw"],
        "secs": secs,
        "scorer_query": "stem-only (clean)",
    }
    if peer_row:
        case["peer"] = {
            "arm": "omp_lc" if arm_id == "topk_lc" else "topk_lc",
            "ok": peer_row["ok"],
            "pred": peer_row["pred"],
        }
    return case


def enrich_cases(cases, arm_id, peer_id, self_label, peer_label, picks, peer_rows, text_by_id, bin_, sdir):
    plot_dir = sdir / "data/plots"
    peer_frame_dir = sdir / "data/frames_peer"
    gold_frame_dir = sdir / "data/frames_gold"
    plot_dir.mkdir(parents=True, exist_ok=True)
    peer_frame_dir.mkdir(parents=True, exist_ok=True)
    gold_frame_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    for c in cases:
        qid = c["id"]
        secs_self = list(picks[arm_id].get(qid, c.get("secs", [])))
        secs_peer = list(picks[peer_id].get(qid, []))
        peer_pred = peer_rows.get(qid, {})
        ginfo = gold_info(manifest.get(qid, {}))
        c["secs"] = secs_self
        c["gold"] = ginfo
        c["peer"] = {
            "arm": peer_id,
            "label": peer_label,
            "ok": peer_pred.get("ok"),
            "pred": peer_pred.get("pred"),
            "pred_text": letter_text(c["options"], peer_pred.get("pred")),
            "gold": peer_pred.get("gold"),
            "secs": secs_peer,
            "raw": peer_pred.get("raw"),
        }
        vid = ROOT / c["media_path"] if c.get("media_path") else None
        peer_jpgs, gold_jpgs = [], []
        if vid and vid.exists():
            dst = peer_frame_dir / qid
            dst.mkdir(parents=True, exist_ok=True)
            for rank, sec in enumerate(secs_peer, 1):
                name = f"r{rank}_t{float(sec):.1f}s.jpg"
                out = dst / name
                grab(vid, float(sec), out)
                if out.exists():
                    peer_jpgs.append(f"data/frames_peer/{qid}/{name}")
            gdst = gold_frame_dir / qid
            gdst.mkdir(parents=True, exist_ok=True)
            for rank, sec in enumerate(ginfo.get("centers") or [], 1):
                name = f"g{rank}_t{float(sec):.1f}s.jpg"
                out = gdst / name
                grab(vid, float(sec), out)
                if out.exists():
                    gold_jpgs.append(f"data/frames_gold/{qid}/{name}")
        c["peer"]["frame_jpgs"] = peer_jpgs
        c["gold"]["frame_jpgs"] = gold_jpgs

        curve = score_curve(qid, text_by_id, bin_)
        plot_rel = None
        if curve is not None:
            times, scores = curve
            png = plot_dir / f"{qid}_score.png"
            plot_score(
                qid, times, scores, secs_self, secs_peer, self_label, peer_label, png,
                gold_spans=ginfo.get("spans"), gold_centers=ginfo.get("centers"),
            )
            plot_rel = f"data/plots/{qid}_score.png"
        c["score_plot"] = plot_rel
    return cases


def write_arm_kind(out, arm_id, kind, cases, meta, label, bin_, k, picks_all, peer_id, peer_label, peer_rows, text_by_id):
    sdir = out / arm_id / kind
    if sdir.exists():
        shutil.rmtree(sdir)
    data = sdir / "data"
    vdir, fdir = data / "videos", data / "frames"
    vdir.mkdir(parents=True)
    fdir.mkdir(parents=True)

    videos = set()
    for c in cases:
        if c.get("video_file"):
            videos.add(c["video_file"])
        dst = fdir / c["id"]
        dst.mkdir(parents=True, exist_ok=True)
        vid = ROOT / c["media_path"] if c.get("media_path") else None
        frames = []
        for rank, sec in enumerate(c["secs"], 1):
            name = f"r{rank}_t{float(sec):.1f}s.jpg"
            outj = dst / name
            if vid and vid.exists():
                grab(vid, float(sec), outj)
            if outj.exists():
                frames.append(f"data/frames/{c['id']}/{name}")
        c["frame_jpgs"] = frames

    for vf in sorted(videos):
        if not link_video(vf, vdir):
            print("MISSING_VIDEO", arm_id, kind, vf)

    # enrich peer/gold/plots
    self_label = label
    cases = enrich_cases(
        cases, arm_id, peer_id, self_label, peer_label,
        picks_all, peer_rows, text_by_id, bin_, sdir,
    )

    pack = {
        "arm": arm_id,
        "label": label,
        "kind": kind,
        "k": k,
        "bin": bin_,
        "query": "stem-only",
        "harness": "lmms-eval",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "meta": meta,
        "cases": cases,
    }
    (data / "cases.json").write_text(json.dumps(pack, indent=2))
    write_enriched_html(sdir, pack, self_label, peer_label, bin_, k, kind)
    nfr = len(list(fdir.rglob("*.jpg")))
    print(arm_id, kind, "videos", len(list(vdir.glob("*.mp4"))), "frames", nfr,
          "plots", len(list((sdir / "data/plots").glob("*.png"))))
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--samples", type=Path, required=True, help="result root containing samples jsonl")
    ap.add_argument("--n", type=int, default=10, help="cases per fail/success bucket")
    ap.add_argument("--fail-only", action="store_true", help="only build FAIL buckets (skip success)")
    ap.add_argument("--no-plots", action="store_true", help="skip score plots (no embeds needed)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bin_, k = args.bin, args.k
    samples_root = find_samples_root(ROOT / args.samples if not args.samples.is_absolute() else args.samples)
    out = args.out or (ROOT / f"results/clean_pack_k{k}_{bin_}")
    if not out.is_absolute():
        out = ROOT / out

    picks = {
        "topk_lc": json.loads((ROOT / f"results/picks_lmmseval/picks_lc_k{k}.json").read_text()),
        "omp_lc": json.loads((ROOT / f"results/picks_lmmseval/picks_omp_lc_k{k}.json").read_text()),
    }
    manifest = load_manifest()
    text_by_id, text_path = ({}, None) if args.no_plots else load_text_index(bin_)
    print("samples_root", samples_root)
    print("text", text_path, "n", len(text_by_id), "plots", not args.no_plots, "fail_only", args.fail_only)

    topk_rows, _ = load_arm_rows(samples_root, f"picks_lc_{bin_}s_k{k}")
    omp_rows, _ = load_arm_rows(samples_root, f"picks_omp_lc_{bin_}s_k{k}")
    try:
        uni_rows, _ = load_arm_rows(samples_root, f"_i_{bin_}s_k{k}")
    except FileNotFoundError:
        uni_rows = {}

    topk_acc = load_results_acc(samples_root, f"longvideobench_val_picks_lc_{bin_}s_k{k}")
    omp_acc = load_results_acc(samples_root, f"longvideobench_val_picks_omp_lc_{bin_}s_k{k}")
    uacc = load_results_acc(samples_root, f"longvideobench_val_i_{bin_}s_k{k}")
    if topk_acc is None:
        topk_acc = round(acc_from_rows(topk_rows), 5)
    if omp_acc is None:
        omp_acc = round(acc_from_rows(omp_rows), 5)
    if uacc is None:
        uacc = round(acc_from_rows(uni_rows), 5) if uni_rows else None

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    meta_base = {"uniform_acc": uacc, "n": len(topk_rows), "bin": bin_, "k": k}
    all_ids = []
    all_vids = set()

    arms = [
        ("topk_lc", topk_rows, topk_acc, "topk-lc", "omp_lc", "OMP-lc", omp_rows),
        ("omp_lc", omp_rows, omp_acc, "OMP-lc", "topk_lc", "topk-lc", topk_rows),
    ]
    kinds = [("fail", False)] if args.fail_only else [("fail", False), ("success", True)]
    for arm_id, rows, acc, label, peer_id, peer_label, peer_rows in arms:
        meta = dict(meta_base, acc=acc, arm=arm_id)
        for kind, want_ok in kinds:
            ids = stable([q for q, r in rows.items() if r["ok"] == want_ok])[: args.n]
            all_ids.extend(ids)
            cases = [
                build_case(qid, rows[qid], arm_id, picks[arm_id], manifest, peer_rows.get(qid))
                for qid in ids
            ]
            vids = write_arm_kind(
                out, arm_id, kind, cases, meta, label, bin_, k,
                picks, peer_id, peer_label, peer_rows, text_by_id,
            )
            all_vids.update(vids)
        print(arm_id, "acc", acc, "fail", sum(1 for r in rows.values() if not r["ok"]),
              "ok", sum(1 for r in rows.values() if r["ok"]))

    (out / "ids.txt").write_text("\n".join(dict.fromkeys(all_ids)) + "\n")
    miss = [v for v in sorted(all_vids) if v and not (VID_SRC / v).exists()]
    (out / "videos_needed.txt").write_text("\n".join(miss) + ("\n" if miss else ""))
    # embeds needed only when plots enabled
    emb_needed = []
    if not args.no_plots:
        for qid in dict.fromkeys(all_ids):
            if bin_ >= 3600:
                p = ROOT / "results/embeds" / f"{qid}.npz"
            else:
                p = ROOT / "results/embeds_lc" / f"{qid}.npz"
            if not p.exists():
                emb_needed.append(qid)
    (out / "embeds_needed.txt").write_text("\n".join(emb_needed) + ("\n" if emb_needed else ""))
    summary = {
        "uniform_acc": uacc,
        "topk_lc_acc": topk_acc,
        "omp_lc_acc": omp_acc,
        "n": len(topk_rows),
        "bin": bin_,
        "k": k,
        "samples": str(samples_root.relative_to(ROOT)) if samples_root.is_relative_to(ROOT) else str(samples_root),
        "videos_needed": miss,
        "embeds_needed": emb_needed,
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2))
    links = []
    for arm_id, label in [("topk_lc", "topk-lc"), ("omp_lc", "OMP-lc")]:
        links.append(f'<li><a href="{arm_id}/fail/index.html">{label} FAIL</a>')
        if not args.fail_only:
            links[-1] += f' · <a href="{arm_id}/success/index.html">SUCCESS</a>'
        links[-1] += "</li>"
    plot_note = "no score plots (--no-plots)" if args.no_plots else "score plots if embeds present"
    (out / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>clean pack {bin_}s k={k}</title>
<style>body{{font:16px/1.4 system-ui;margin:40px}}.warn{{background:#ffe8a0;padding:10px}}</style>
</head><body>
<h1>Clean stem · {bin_}s · k={k}{" · FAIL only" if args.fail_only else ""}</h1>
<div class=warn>Local pack from pulled lmms-eval samples. Peer frames + LVB gold. {plot_note}.</div>
<p>uniform=<b>{uacc}</b> · topk-lc=<b>{topk_acc}</b> · OMP-lc=<b>{omp_acc}</b> · n={len(topk_rows)}</p>
<ul>
{"".join(links)}
</ul>
<p class=warn>videos_needed={len(miss)} · embeds_needed={len(emb_needed)} — pull only these, then re-run</p>
</body></html>
""")
    print("OUT", out)
    print("videos_needed", len(miss))
    print("embeds_needed", len(emb_needed))


if __name__ == "__main__":
    main()
