#!/usr/bin/env python3
"""Clean stem 600s pack: fail/success for topk-lc AND omp-lc (k=8).

Source: lmmseval_matrix_clean/k8_600 — NOT archived hunt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "results/lmmseval_matrix_clean/k8_600/Qwen__Qwen3-VL-8B-Instruct"
PICKS = {
    "topk_lc": json.loads((ROOT / "results/picks_lmmseval/picks_lc_k8.json").read_text()),
    "omp_lc": json.loads((ROOT / "results/picks_lmmseval/picks_omp_lc_k8.json").read_text()),
}
MANIFEST = {}
for mf in [
    ROOT / "data/manifest.lvb.long976.json",
    ROOT / "data/manifest.lvb.full1560.json",
]:
    if mf.exists():
        for x in json.loads(mf.read_text()):
            MANIFEST[x["id"]] = x

OUT = ROOT / "results/clean_pack_k8_600"
VID_SRC = ROOT / "data/videos"
TOKEN_AUDIT = OUT / "token_audit_stem.json"
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

ARMS = {
    "topk_lc": {
        "sample_substr": "picks_lc_600s_k8",
        "picks_key": "topk_lc",
        "label": "topk-LongCLIP",
        "acc_key": "longvideobench_val_picks_lc_600s_k8",
    },
    "omp_lc": {
        "sample_substr": "picks_omp_lc_600s_k8",
        "picks_key": "omp_lc",
        "label": "OMP-LongCLIP",
        "acc_key": "longvideobench_val_picks_omp_lc_600s_k8",
    },
}


def stable(ids):
    return sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())


def parse_options(input_text: str):
    opts = []
    for m in re.finditer(r"^([A-E])\.\s*(.+)$", input_text, re.M):
        opts.append({"letter": m.group(1), "text": m.group(2).strip()})
    return opts


def load_arm(arm_id: str):
    cfg = ARMS[arm_id]
    paths = [p for p in CLEAN.glob("*samples*.jsonl") if cfg["sample_substr"] in p.name]
    if not paths:
        raise FileNotFoundError(cfg["sample_substr"])
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
    res = json.load(open(next(CLEAN.glob("*_results.json"))))["results"]
    acc = res[cfg["acc_key"]].get("lvb_acc,none")
    uacc = res["longvideobench_val_i_600s_k8"].get("lvb_acc,none")
    return rows, acc, uacc


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


def load_tokens():
    if TOKEN_AUDIT.exists():
        return {r["id"]: r for r in json.loads(TOKEN_AUDIT.read_text())}
    return {}


def tokens_html(tok):
    if not tok:
        return "<div class=meta>tokens: (audit pending)</div>"
    s, l = tok["siglip_stem"], tok["longclip_stem"]
    q = tok["qwen_answerer_text"]
    out = (
        f"<div class=meta><b>tokens (STEM)</b> — "
        f"SigLIP: {s['n_tokens_full']}→{s['n_tokens_kept']}/{s['max_length']} "
        f"{'TRUNCATED' if s['truncated'] else 'ok'} · "
        f"LongCLIP: {l['n_tokens_full']}→~{l['n_tokens_kept']}/{l['max_length']} "
        f"{'TRUNCATED' if l['truncated'] else 'ok'} · "
        f"Qwen text: {q['prompt_n_tokens']}</div>"
    )
    if s.get("kept_text") and s["truncated"]:
        kt = s["kept_text"]
        out += f"<div class=meta>SigLIP kept: <code>{kt[:240]}{'…' if len(kt)>240 else ''}</code></div>"
    return out


def build_case(qid, row, arm_id, tokens_by_id, peer_row=None):
    m = MANIFEST.get(qid, {})
    opts = parse_options(row["input"])
    if not opts and m.get("candidates"):
        opts = [{"letter": LETTERS[i], "text": c} for i, c in enumerate(m["candidates"])]
    stem = question_stem(m) if m else row["input"].split("\nA.")[0].strip()
    secs = list(PICKS[ARMS[arm_id]["picks_key"]].get(qid, []))
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
        "tokens": tokens_by_id.get(qid),
        "scorer_query": "stem-only (clean)",
        "harness": f"lmms-eval {ARMS[arm_id]['sample_substr']}",
    }
    if peer_row:
        case["peer"] = {
            "arm": "omp_lc" if arm_id == "topk_lc" else "topk_lc",
            "ok": peer_row["ok"],
            "pred": peer_row["pred"],
        }
    return case


def write_arm_kind(arm_id: str, kind: str, cases: list, meta: dict):
    sdir = OUT / arm_id / kind
    if sdir.exists():
        shutil.rmtree(sdir)
    data = sdir / "data"
    vdir, fdir = data / "videos", data / "frames"
    vdir.mkdir(parents=True)
    fdir.mkdir(parents=True)

    label = ARMS[arm_id]["label"]
    pack = {
        "arm": arm_id,
        "label": label,
        "kind": kind,
        "k": 8,
        "bin": 600,
        "query": "stem-only",
        "harness": "lmms-eval",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "meta": meta,
        "cases": [],
    }
    videos = set()
    for c in cases:
        if c.get("video_file"):
            videos.add(c["video_file"])
        dst = fdir / c["id"]
        dst.mkdir(parents=True, exist_ok=True)
        vid = ROOT / c["media_path"] if c.get("media_path") else None
        frames = []
        for rank, sec in enumerate(c["secs"], 1):
            name = f"r{rank}_t{sec:.1f}s.jpg"
            out = dst / name
            if vid and vid.exists():
                grab(vid, float(sec), out)
            if out.exists():
                frames.append(f"data/frames/{c['id']}/{name}")
        c2 = dict(c)
        c2["frame_jpgs"] = frames
        pack["cases"].append(c2)

    for vf in sorted(videos):
        if not link_video(vf, vdir):
            print("MISSING_VIDEO", arm_id, kind, vf)

    (data / "cases.json").write_text(json.dumps(pack, indent=2))

    title = f"CLEAN 600s · {label} @k=8 · {kind.upper()}"
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        f"<title>{title}</title>",
        "<style>body{font:14px/1.45 system-ui;margin:24px;max-width:1100px}"
        ".case{margin:1.5em 0;padding:1em;background:#f6f6f6}"
        ".fail{border-left:4px solid #c00}.ok{border-left:4px solid #080}"
        ".row{display:flex;flex-wrap:wrap;gap:6px}.row img{height:110px;border:2px solid #666}"
        ".meta{color:#444;font-size:13px}.tag{background:#cfc;padding:1px 6px;margin-right:6px}"
        ".warn{background:#ffe8a0;padding:8px;margin:12px 0}</style></head><body>",
        f"<h1>{title}</h1>",
        "<div class=warn>Clean stem query. Source: lmmseval_matrix_clean/k8_600. "
        "Old hunt packs VOID.</div>",
        f"<p>acc={meta.get('acc')} · uniform={meta.get('uniform_acc')} · n={meta.get('n')} · "
        f"10 {kind} cases shown</p>",
        '<p><a href="../../index.html">pack overview</a></p>',
    ]
    md = [f"# {title}\n"]

    for i, c in enumerate(pack["cases"], 1):
        cls = "ok" if c["ok"] else "fail"
        secs = ", ".join(f"{float(s):.1f}s" for s in c["secs"])
        html.append(f"<div class='case {cls}'>")
        html.append(
            f"<div><span class=tag>600s stem</span><b>{i}. {c['id']}</b> · "
            f"<code>data/videos/{c['video_file']}</code> ({c.get('video_seconds')}s)</div>"
        )
        html.append(
            f"<div class=meta>gold <b>{c['gold_letter']}</b>: {c['gold_answer']} · "
            f"pred <b>{c['pred']}</b>: {c['pred_text']} · frames [{secs}]</div>"
        )
        if c.get("peer"):
            p = c["peer"]
            html.append(
                f"<div class=meta>peer {p['arm']}: {'OK' if p['ok'] else 'FAIL'} pred={p['pred']}</div>"
            )
        html.append(tokens_html(c.get("tokens")))
        html.append(f"<p>{c['question_stem']}</p><ul>")
        for o in c["options"]:
            g = " ← GOLD" if o["letter"] == c["gold_letter"] else ""
            pr = " ← PRED" if o["letter"] == c["pred"] else ""
            html.append(f"<li><b>{o['letter']}</b>) {o['text']}{g}{pr}</li>")
        html.append("</ul>")
        html.append(f"<div class=meta><b>{label} clips</b></div><div class=row>")
        for rank, sec in enumerate(c["secs"], 1):
            rel = f"data/frames/{c['id']}/r{rank}_t{float(sec):.1f}s.jpg"
            html.append(f"<div><img src='{rel}'><div class=meta>#{rank} @{float(sec):.1f}s</div></div>")
        html.append("</div></div>")

        md.append(f"### {i}. `{c['id']}` {'OK' if c['ok'] else 'FAIL'}")
        md.append(f"- gold {c['gold_letter']} / pred {c['pred']}")
        md.append(f"- frames [{secs}]")
        md.append(f"- Q: {c['question_stem']}")
        for o in c["options"]:
            md.append(f"  - {o['letter']}) {o['text']}")
        md.append("")

    html.append("</body></html>")
    (sdir / "index.html").write_text("\n".join(html))
    (sdir / f"{kind.upper()}.md").write_text("\n".join(md))
    nfr = len(list(fdir.rglob("*.jpg")))
    print(arm_id, kind, "videos", len(list(vdir.glob("*.mp4"))), "frames", nfr)
    return videos


def main():
    saved = TOKEN_AUDIT.read_text() if TOKEN_AUDIT.exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if saved:
        TOKEN_AUDIT.write_text(saved)
    tokens_by_id = load_tokens()

    # load both arms for peer annotation
    topk_rows, topk_acc, uacc = load_arm("topk_lc")
    omp_rows, omp_acc, _ = load_arm("omp_lc")
    meta_base = {"uniform_acc": uacc, "n": len(topk_rows), "bin": 600, "k": 8}

    all_ids = []
    all_vids = set()
    summary = {
        "uniform_acc": uacc,
        "topk_lc_acc": topk_acc,
        "omp_lc_acc": omp_acc,
        "n": len(topk_rows),
    }

    for arm_id, rows, acc in [
        ("topk_lc", topk_rows, topk_acc),
        ("omp_lc", omp_rows, omp_acc),
    ]:
        peer = omp_rows if arm_id == "topk_lc" else topk_rows
        meta = dict(meta_base, acc=acc, arm=arm_id)
        for kind, want_ok in [("fail", False), ("success", True)]:
            ids = stable([q for q, r in rows.items() if r["ok"] == want_ok])[:10]
            all_ids.extend(ids)
            cases = [
                build_case(qid, rows[qid], arm_id, tokens_by_id, peer.get(qid))
                for qid in ids
            ]
            vids = write_arm_kind(arm_id, kind, cases, meta)
            all_vids.update(vids)
        print(
            arm_id, "acc", acc,
            "fail_pool", sum(1 for r in rows.values() if not r["ok"]),
            "ok_pool", sum(1 for r in rows.values() if r["ok"]),
        )

    (OUT / "ids.txt").write_text("\n".join(dict.fromkeys(all_ids)) + "\n")
    miss = [v for v in sorted(all_vids) if v and not (VID_SRC / v).exists()]
    (OUT / "videos_needed.txt").write_text("\n".join(miss) + ("\n" if miss else ""))
    (OUT / "meta.json").write_text(json.dumps(summary, indent=2))

    (OUT / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>clean pack 600s k=8</title>
<style>body{{font:16px/1.4 system-ui;margin:40px}}.warn{{background:#ffe8a0;padding:10px}}</style>
</head><body>
<h1>Clean stem · 600s · k=8</h1>
<div class=warn>lmms-eval matrix_clean/k8_600. Fail + success for topk-lc and OMP-lc.
Old hunt packs VOID.</div>
<p>uniform=<b>{uacc}</b> · topk-lc=<b>{topk_acc}</b> (+{(topk_acc-uacc)*100:.1f}pt) ·
OMP-lc=<b>{omp_acc}</b> (+{(omp_acc-uacc)*100:.1f}pt) · n=412</p>
<ul>
<li><a href="topk_lc/fail/index.html">topk-lc FAIL</a> ·
    <a href="topk_lc/success/index.html">SUCCESS</a></li>
<li><a href="omp_lc/fail/index.html">OMP-lc FAIL</a> ·
    <a href="omp_lc/success/index.html">SUCCESS</a></li>
</ul>
</body></html>
""")
    print("OUT", OUT)
    print("videos_needed", len(miss))


if __name__ == "__main__":
    main()
