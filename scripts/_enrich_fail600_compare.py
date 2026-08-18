#!/usr/bin/env python3
"""Enrich 600s fail HTMLs: peer method frames+pred + LongCLIP score-vs-time plots.

Updates:
  results/clean_pack_k8_600/omp_lc/fail/
  results/clean_pack_k8_600/topk_lc/fail/

Does not touch pod processes. Uses local caches:
  results/embeds_lc/<qid>.npz + results/embeds_text/text_lc_600.npz
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/clean_pack_k8_600"
PICKS = {
    "topk_lc": json.loads((ROOT / "results/picks_lmmseval/picks_lc_k8.json").read_text()),
    "omp_lc": json.loads((ROOT / "results/picks_lmmseval/picks_omp_lc_k8.json").read_text()),
}
SAMPLES = ROOT / "results/lmmseval_matrix_clean/k8_600/Qwen__Qwen3-VL-8B-Instruct"
VID_SRC = ROOT / "data/videos"
GOLD = json.loads((OUT / "fail600_gold.json").read_text()) if (OUT / "fail600_gold.json").exists() else {}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(["pip", "install", "matplotlib", "-q"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def load_preds(arm_substr: str) -> dict:
    path = next(SAMPLES.glob(f"*{arm_substr}*.jsonl"))
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        a = s["lvb_acc"]
        out[a["id"]] = {
            "pred": a["parsed_pred"],
            "gold": a["answer"],
            "ok": a["parsed_pred"] == a["answer"],
            "raw": s.get("filtered_resps"),
            "input": s.get("input", ""),
        }
    return out


def load_text_index():
    z = np.load(ROOT / "results/embeds_text/text_lc_600.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    text = z["text"].astype(np.float32)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    return {i: text[k] for k, i in enumerate(ids)}


def score_curve(qid: str, text_by_id: dict):
    if qid not in text_by_id:
        return None
    p = ROOT / "results/embeds_lc" / f"{qid}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    times = z["times"].astype(np.float32)
    emb = z["emb"].astype(np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    scores = emb @ text_by_id[qid]
    return times, scores


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
    mark(secs_self, "#c0392b" if "OMP" in self_label or "omp" in self_label.lower() else "#2980b9",
         "o", f"{self_label} pick")
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


def letter_text(opts, letter):
    for o in opts:
        if o["letter"] == letter:
            return o["text"]
    return None


def enrich_arm(arm_id: str, peer_id: str, self_label: str, peer_label: str,
               preds_self: dict, preds_peer: dict, text_by_id: dict):
    sdir = OUT / arm_id / "fail"
    cases_path = sdir / "data/cases.json"
    pack = json.loads(cases_path.read_text())
    plot_dir = sdir / "data/plots"
    peer_frame_dir = sdir / "data/frames_peer"
    gold_frame_dir = sdir / "data/frames_gold"
    plot_dir.mkdir(parents=True, exist_ok=True)
    peer_frame_dir.mkdir(parents=True, exist_ok=True)
    gold_frame_dir.mkdir(parents=True, exist_ok=True)

    for c in pack["cases"]:
        qid = c["id"]
        secs_self = list(PICKS[arm_id].get(qid, c.get("secs", [])))
        secs_peer = list(PICKS[peer_id].get(qid, []))
        peer_pred = preds_peer.get(qid, {})
        ginfo = GOLD.get(qid) or {}
        gold_centers = list(ginfo.get("gold_center_seconds") or [])
        gold_spans = list(ginfo.get("gold_evidence_seconds") or [])
        c["secs"] = secs_self
        c["gold"] = {
            "position": ginfo.get("gold_position"),
            "native_fps": ginfo.get("native_fps"),
            "spans": gold_spans,
            "centers": gold_centers,
            "reliable": ginfo.get("gold_reliable"),
            "note": "LVB position → sec = pos/native_fps; span = center±1s (fetch_lvb_subset)",
        }
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
        # peer + gold frame thumbs
        vid = ROOT / c["media_path"] if c.get("media_path") else None
        peer_jpgs = []
        gold_jpgs = []
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
            for rank, sec in enumerate(gold_centers, 1):
                name = f"g{rank}_t{float(sec):.1f}s.jpg"
                out = gdst / name
                grab(vid, float(sec), out)
                if out.exists():
                    gold_jpgs.append(f"data/frames_gold/{qid}/{name}")
        c["peer"]["frame_jpgs"] = peer_jpgs
        c["gold"]["frame_jpgs"] = gold_jpgs

        # score plot
        curve = score_curve(qid, text_by_id)
        plot_rel = None
        if curve is not None:
            times, scores = curve
            png = plot_dir / f"{qid}_score.png"
            plot_score(qid, times, scores, secs_self, secs_peer, self_label, peer_label, png,
                       gold_spans=gold_spans, gold_centers=gold_centers)
            plot_rel = f"data/plots/{qid}_score.png"
        c["score_plot"] = plot_rel

    cases_path.write_text(json.dumps(pack, indent=2))
    write_html(sdir, pack, self_label, peer_label)
    print(arm_id, "fail cases", len(pack["cases"]),
          "plots", len(list(plot_dir.glob("*.png"))),
          "peer_frames", len(list(peer_frame_dir.rglob("*.jpg"))))


def tokens_html(tok):
    if not tok:
        return ""
    s, l = tok.get("siglip_stem", {}), tok.get("longclip_stem", {})
    return (
        f"<div class=meta><b>tokens (STEM)</b> — "
        f"SigLIP≈{s.get('n_tokens_full')}/64 {'TRUNC' if s.get('truncated') else 'ok'} · "
        f"LongCLIP {l.get('n_tokens_full')}/248 {'TRUNC' if l.get('truncated') else 'ok'}</div>"
    )


def write_html(sdir: Path, pack: dict, self_label: str, peer_label: str):
    meta = pack["meta"]
    title = f"CLEAN 600s · {self_label} @k=8 · FAIL (+ {peer_label} compare)"
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
        f"Red = {self_label} · green = {peer_label} · yellow band/orange★ = LVB gold "
        "(position/fps ±1s).</div>",
        f"<p>acc={meta.get('acc')} · uniform={meta.get('uniform_acc')} · n={meta.get('n')}</p>",
        '<p><a href="../../index.html">pack overview</a></p>',
    ]
    for i, c in enumerate(pack["cases"], 1):
        secs = ", ".join(f"{float(s):.1f}s" for s in c["secs"])
        peer = c.get("peer") or {}
        psecs = ", ".join(f"{float(s):.1f}s" for s in peer.get("secs", []))
        html.append("<div class='case fail'>")
        html.append(
            f"<div><span class=tag>600s stem</span><b>{i}. {c['id']}</b> · "
            f"<code>{c['video_file']}</code> ({c.get('video_seconds')}s)</div>"
        )
        html.append(
            f"<div class=meta>gold <b>{c['gold_letter']}</b>: {c['gold_answer']}</div>"
        )
        html.append(
            f"<div class=meta><b>{self_label}</b> FAIL · pred <b>{c['pred']}</b>: {c['pred_text']} · "
            f"frames [{secs}]</div>"
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
            html.append(f"<div class=meta><b>score vs time</b> (LongCLIP stem cosine)</div>")
            html.append(f"<img class=plot src='{c['score_plot']}' alt='score plot'>")

        html.append(f"<div class=meta><b>{self_label} clips</b></div><div class='row self'>")
        for rank, sec in enumerate(c["secs"], 1):
            rel = f"data/frames/{c['id']}/r{rank}_t{float(sec):.1f}s.jpg"
            html.append(f"<div><img src='{rel}'><div class=meta>#{rank} @{float(sec):.1f}s</div></div>")
        html.append("</div>")

        html.append(f"<div class=meta><b>{peer_label} clips</b></div><div class='row peer'>")
        for rel, sec in zip(peer.get("frame_jpgs", []), peer.get("secs", [])):
            html.append(f"<div><img src='{rel}'><div class=meta>@{float(sec):.1f}s</div></div>")
        if not peer.get("frame_jpgs"):
            html.append("<div class=meta>(no peer frames extracted)</div>")
        html.append("</div>")

        g = c.get("gold") or {}
        html.append("<div class=meta><b>LVB gold clips</b> (center of ±1s span)</div><div class='row gold'>")
        for rel, sec in zip(g.get("frame_jpgs", []), g.get("centers", [])):
            html.append(f"<div><img src='{rel}'><div class=meta>gold @{float(sec):.1f}s</div></div>")
        if not g.get("frame_jpgs"):
            html.append("<div class=meta>(no gold frames)</div>")
        html.append("</div></div>")

    html.append("</body></html>")
    (sdir / "index.html").write_text("\n".join(html))


def main():
    text_by_id = load_text_index()
    preds_topk = load_preds("picks_lc_600s_k8")
    preds_omp = load_preds("picks_omp_lc_600s_k8")
    enrich_arm("omp_lc", "topk_lc", "OMP-lc", "topk-lc", preds_omp, preds_topk, text_by_id)
    enrich_arm("topk_lc", "omp_lc", "topk-lc", "OMP-lc", preds_topk, preds_omp, text_by_id)
    print("done")


if __name__ == "__main__":
    main()
