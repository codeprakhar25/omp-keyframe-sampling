#!/usr/bin/env python3
"""Build fail/success HTML packs from CLEAN stem-query lmms-eval results (LongCLIP).

Source: results/lmmseval_matrix_clean/k8_{15,60} picks_lc samples + picks_lc_k8.json
NOT the archived gpt_mcqa hunt (fused-options scorer).
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
CLEAN = ROOT / "results/lmmseval_matrix_clean"
PICKS_LC = json.loads((ROOT / "results/picks_lmmseval/picks_lc_k8.json").read_text())
PICKS_SIG = json.loads((ROOT / "results/picks_lmmseval/picks_sig_k8.json").read_text())
MANIFEST = {x["id"]: x for x in json.loads((ROOT / "data/manifest.lvb.full1560.json").read_text())}
OUT = ROOT / "results/clean_pack_k8"
VID_SRC = ROOT / "data/videos"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TOKEN_AUDIT = ROOT / "results/clean_pack_k8/token_audit_stem.json"

try:
    from harness.text import question_stem
except Exception:
    def question_stem(item):
        q = item["question"] if isinstance(item, dict) else item
        idx = q.find("\n\nOptions:")
        if idx == -1:
            idx = q.find("Options:")
        return (q[:idx] if idx != -1 else q).strip()


def stable(ids):
    return sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())


def parse_options(input_text: str):
    """Parse A-E lines from lmms-eval prompt."""
    opts = []
    for m in re.finditer(r"^([A-E])\.\s*(.+)$", input_text, re.M):
        opts.append({"letter": m.group(1), "text": m.group(2).strip()})
    return opts


def load_samples(bin_: int, arm: str):
    """arm: i | picks_lc | picks_omp_lc"""
    d = CLEAN / f"k8_{bin_}" / "Qwen__Qwen3-VL-8B-Instruct"
    paths = list(d.glob(f"*samples_longvideobench_val_{arm}_{bin_}s_k8.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no samples for {bin_} {arm} under {d}")
    rows = {}
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        a = s["lvb_acc"]
        qid = a["id"]
        gold = a["answer"]
        pred = a["parsed_pred"]
        rows[qid] = {
            "id": qid,
            "gold": gold,
            "pred": pred,
            "ok": pred == gold,
            "raw": s.get("filtered_resps"),
            "input": s.get("input", ""),
            "category": a.get("question_category"),
            "duration_group": a.get("duration_group"),
        }
    # acc from results json
    rpath = list(d.glob("*_results.json"))[0]
    res = json.load(open(rpath))["results"]
    key = f"longvideobench_val_{arm}_{bin_}s_k8"
    acc = res[key].get("lvb_acc,none")
    return rows, acc, key


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


def pick_ids(rows: dict, want_ok: bool, n: int = 10):
    pool = [qid for qid, r in rows.items() if r["ok"] == want_ok]
    return stable(pool)[:n]


def load_tokens():
    if TOKEN_AUDIT.exists():
        return {r["id"]: r for r in json.loads(TOKEN_AUDIT.read_text())}
    return {}


def tokens_html(tok) -> str:
    if not tok:
        return "<div class=meta>tokens: (audit pending)</div>"
    s, l = tok["siglip_stem"], tok["longclip_stem"]
    q = tok["qwen_answerer_text"]
    out = (
        f"<div class=meta><b>tokens (STEM scorer query)</b> — "
        f"SigLIP: {s['n_tokens_full']}→{s['n_tokens_kept']}/{s['max_length']} "
        f"{'TRUNCATED' if s['truncated'] else 'ok'} · "
        f"LongCLIP: {l['n_tokens_full']}→kept~{l['n_tokens_kept']}/{l['max_length']} "
        f"{'TRUNCATED' if l['truncated'] else 'ok'} · "
        f"Qwen answerer text: {q['prompt_n_tokens']} tok (no text truncate)</div>"
    )
    if s.get("kept_text") and s["truncated"]:
        kt = s["kept_text"]
        out += (
            f"<div class=meta>SigLIP kept: <code>{kt[:240]}{'…' if len(kt)>240 else ''}</code></div>"
        )
    # contrast: fused void path
    sf = tok.get("siglip_fused_contrast") or {}
    if sf:
        out += (
            f"<div class=meta>contrast VOID fused: SigLIP {sf.get('n_tokens_full')} tok "
            f"{'TRUNC' if sf.get('truncated') else 'ok'}</div>"
        )
    return out


def build_case(qid: str, row: dict, bin_: int, tokens_by_id: dict):
    m = MANIFEST.get(qid, {})
    opts = parse_options(row["input"])
    if not opts and m.get("candidates"):
        opts = [{"letter": LETTERS[i], "text": c} for i, c in enumerate(m["candidates"])]
    stem = question_stem(m) if m else row["input"].split("\nA.")[0].strip()
    secs_lc = list(PICKS_LC.get(qid, []))
    secs_sig = list(PICKS_SIG.get(qid, []))
    gold = row["gold"]
    pred = row["pred"]
    gold_text = next((o["text"] for o in opts if o["letter"] == gold), m.get("gold_answer"))
    pred_text = next((o["text"] for o in opts if o["letter"] == pred), None)
    return {
        "id": qid,
        "bin": bin_,
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
        "secs_longclip": secs_lc,
        "secs_siglip": secs_sig,  # for compare only; no sig answerer yet
        "tokens": tokens_by_id.get(qid),
        "scorer_query": "stem-only (clean)",
        "harness": "lmms-eval picks_lc k=8",
    }


def write_pack(kind: str, bins_cases: dict, meta: dict):
    """kind = fail | success. Under OUT/longclip/{kind}/"""
    sdir = OUT / "longclip" / kind
    if sdir.exists():
        shutil.rmtree(sdir)
    data = sdir / "data"
    vdir = data / "videos"
    fdir = data / "frames"
    vdir.mkdir(parents=True)
    fdir.mkdir(parents=True)

    pack = {
        "scorer": "longclip",
        "kind": kind,
        "k": 8,
        "query": "stem-only (post 2026-07-16 fix)",
        "harness": "lmms-eval",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "note": "NOT from archived gpt_mcqa hunt. Clean picks_lc.",
        "meta": meta,
        "bins": {},
    }
    videos = set()

    for bin_, cases in bins_cases.items():
        out_cases = []
        for c in cases:
            videos.add(c["video_file"])
            dst = fdir / f"{bin_}s" / c["id"]
            dst.mkdir(parents=True, exist_ok=True)
            vid = ROOT / c["media_path"] if c.get("media_path") else None
            frames = []
            for rank, sec in enumerate(c["secs_longclip"], 1):
                name = f"r{rank}_t{sec:.1f}s.jpg"
                out = dst / name
                if vid and vid.exists():
                    grab(vid, sec, out)
                if out.exists():
                    frames.append(f"data/frames/{bin_}s/{c['id']}/{name}")
            c2 = dict(c)
            c2["frame_jpgs"] = frames
            out_cases.append(c2)
        pack["bins"][str(bin_)] = {"cases": out_cases}

    for vf in sorted(x for x in videos if x):
        if not link_video(vf, vdir):
            print("MISSING_VIDEO", vf)

    (data / "cases.json").write_text(json.dumps(pack, indent=2))

    title = f"CLEAN stem · LongCLIP topk@8 · {kind.upper()}"
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        f"<title>{title}</title>",
        "<style>body{font:14px/1.45 system-ui;margin:24px;max-width:1100px}"
        ".case{margin:1.5em 0;padding:1em;background:#f6f6f6}"
        ".fail{border-left:4px solid #c00}.ok{border-left:4px solid #080}"
        ".row{display:flex;flex-wrap:wrap;gap:6px}.row img{height:120px;border:2px solid #666}"
        ".meta{color:#444;font-size:13px}.tag{background:#cfc;padding:1px 6px;margin-right:6px}"
        ".warn{background:#ffe8a0;padding:8px;margin:12px 0}</style></head><body>",
        f"<h1>{title}</h1>",
        "<div class=warn>Clean stem-query only (CORRECT_FINDINGS). Old hunt/fail_pack HTML = VOID.</div>",
        f"<p>harness=lmms-eval · picks_lc · k=8 · model=Qwen3-VL-8B · "
        f"15s acc={meta.get('acc_15')} · 60s acc={meta.get('acc_60')}</p>",
        '<p><a href="../../index.html">pack overview</a></p>',
    ]
    md = [f"# {title}\n", "Clean stem scorer query. lmms-eval picks_lc k=8.\n"]

    for bin_, blob in pack["bins"].items():
        acc = meta.get(f"acc_{bin_}")
        html.append(f"<h2>{bin_}s · picks_lc@8 acc={acc}</h2>")
        md.append(f"\n## {bin_}s · acc={acc}\n")
        for i, c in enumerate(blob["cases"], 1):
            cls = "ok" if c["ok"] else "fail"
            secs = ", ".join(f"{s:.1f}s" for s in c["secs_longclip"])
            html.append(f"<div class='case {cls}'>")
            html.append(
                f"<div><span class=tag>stem-clean</span><b>{i}. {c['id']}</b> · "
                f"<code>data/videos/{c['video_file']}</code> "
                f"({c.get('video_seconds')}s)</div>"
            )
            html.append(
                f"<div class=meta>gold <b>{c['gold_letter']}</b>: {c['gold_answer']} · "
                f"pred <b>{c['pred']}</b>: {c['pred_text']} · LC frames [{secs}]</div>"
            )
            html.append(tokens_html(c.get("tokens")))
            if c.get("secs_siglip"):
                ss = ", ".join(f"{s:.1f}s" for s in c["secs_siglip"])
                html.append(
                    f"<div class=meta>SigLIP picks (frames only, no answerer yet): [{ss}]</div>"
                )
            html.append(f"<p>{c['question_stem']}</p><ul>")
            for o in c["options"]:
                g = " ← GOLD" if o["letter"] == c["gold_letter"] else ""
                p = " ← PRED" if o["letter"] == c["pred"] else ""
                html.append(f"<li><b>{o['letter']}</b>) {o['text']}{g}{p}</li>")
            html.append("</ul>")
            html.append("<div class=meta><b>LongCLIP clips</b></div><div class=row>")
            for rank, sec in enumerate(c["secs_longclip"], 1):
                rel = f"data/frames/{bin_}s/{c['id']}/r{rank}_t{sec:.1f}s.jpg"
                html.append(f"<div><img src='{rel}'><div class=meta>#{rank} @{sec:.1f}s</div></div>")
            html.append("</div></div>")

            md.append(f"### {i}. `{c['id']}` {'OK' if c['ok'] else 'FAIL'}")
            md.append(f"- video: `{c['video_file']}`")
            md.append(f"- gold **{c['gold_letter']}** / pred **{c['pred']}**")
            md.append(f"- LC frames: [{secs}]")
            if c.get("tokens"):
                t = c["tokens"]
                md.append(
                    f"- tokens STEM SigLIP {t['siglip_stem']['n_tokens_full']}/64 "
                    f"{'TRUNC' if t['siglip_stem']['truncated'] else 'ok'}; "
                    f"LongCLIP {t['longclip_stem']['n_tokens_full']}/248 "
                    f"{'TRUNC' if t['longclip_stem']['truncated'] else 'ok'}; "
                    f"Qwen {t['qwen_answerer_text']['prompt_n_tokens']}"
                )
            md.append(f"- Q: {c['question_stem']}")
            for o in c["options"]:
                g = " ← GOLD" if o["letter"] == c["gold_letter"] else ""
                p = " ← PRED" if o["letter"] == c["pred"] else ""
                md.append(f"  - {o['letter']}) {o['text']}{g}{p}")
            md.append("")

    html.append("</body></html>")
    (sdir / "index.html").write_text("\n".join(html))
    (sdir / f"{kind.upper()}.md").write_text("\n".join(md))
    print(kind, "videos", len(list(vdir.glob("*.mp4"))), "frames", len(list(fdir.rglob("*.jpg"))))
    return sorted(videos)


def main():
    # keep token audit if present
    saved_tok = TOKEN_AUDIT.read_text() if TOKEN_AUDIT.exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if saved_tok:
        TOKEN_AUDIT.write_text(saved_tok)

    tokens_by_id = load_tokens()
    meta = {}
    fail_bins, ok_bins = {}, {}
    all_vids = set()
    all_ids = []

    for bin_ in (15, 60):
        rows, acc, key = load_samples(bin_, "picks_lc")
        meta[f"acc_{bin_}"] = acc
        meta[f"n_{bin_}"] = len(rows)
        meta[f"task_{bin_}"] = key
        # also store uniform for context
        _, uacc, _ = load_samples(bin_, "i")
        meta[f"uniform_acc_{bin_}"] = uacc

        fail_ids = pick_ids(rows, want_ok=False, n=10)
        ok_ids = pick_ids(rows, want_ok=True, n=10)
        all_ids.extend(fail_ids + ok_ids)
        fail_bins[bin_] = [build_case(qid, rows[qid], bin_, tokens_by_id) for qid in fail_ids]
        ok_bins[bin_] = [build_case(qid, rows[qid], bin_, tokens_by_id) for qid in ok_ids]
        print(f"bin{bin_} lc_acc={acc} uniform={uacc} fail_pool="
              f"{sum(1 for r in rows.values() if not r['ok'])} ok_pool="
              f"{sum(1 for r in rows.values() if r['ok'])}")

    (OUT / "ids.txt").write_text("\n".join(dict.fromkeys(all_ids)) + "\n")

    vids = write_pack("fail", fail_bins, meta)
    all_vids.update(vids)
    vids = write_pack("success", ok_bins, meta)
    all_vids.update(vids)

    # SigLIP stub — picks exist, answerer missing
    sig = OUT / "siglip"
    sig.mkdir(parents=True)
    (sig / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>SigLIP clean pack — waiting</title></head><body>
<h1>SigLIP clean stem — no answerer results yet</h1>
<p><code>picks_sig_k8.json</code> is on disk (stem query, 773 qids) but
<code>lmmseval_matrix_clean</code> has <b>no</b> <code>picks_sig_*</code> sample jsonl
for 15s/60s.</p>
<p>LongCLIP pack is ready: <a href="../longclip/fail/index.html">fail</a> ·
<a href="../longclip/success/index.html">success</a></p>
<p>Need other agent to run lmms-eval arms:<br>
<code>longvideobench_val_picks_sig_15s_k8</code> and
<code>longvideobench_val_picks_sig_60s_k8</code> (bs=1).</p>
<p>Clean LC accs: 15s={meta['acc_15']} · 60s={meta['acc_60']}
(uniform 15s={meta['uniform_acc_15']} · 60s={meta['uniform_acc_60']})</p>
</body></html>
""")
    (sig / "README.md").write_text(
        "SigLIP picks exist (stem). Answerer samples not run yet. See index.html.\n"
    )

    (OUT / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>clean pack k=8</title>
<style>body{{font:16px/1.4 system-ui;margin:40px}}.warn{{background:#ffe8a0;padding:10px}}</style>
</head><body>
<h1>Clean stem-query pack · k=8</h1>
<div class=warn>Source: <code>lmmseval_matrix_clean</code> + <code>picks_*_k8.json</code>.
Old <code>results/hunt/fail_pack_topk6</code> = tainted fused-options — do not use.</div>
<ul>
<li><a href="longclip/fail/index.html"><b>LongCLIP FAIL</b></a> (10/bin × 15s+60s)</li>
<li><a href="longclip/success/index.html"><b>LongCLIP SUCCESS</b></a></li>
<li><a href="siglip/index.html">SigLIP</a> — waiting on answerer runs</li>
</ul>
<p>LC picks_lc@8: 15s=<b>{meta['acc_15']}</b> (uniform {meta['uniform_acc_15']}) ·
60s=<b>{meta['acc_60']}</b> (uniform {meta['uniform_acc_60']})</p>
</body></html>
""")

    miss = [v for v in sorted(all_vids) if v and not (VID_SRC / v).exists()]
    (OUT / "videos_needed.txt").write_text("\n".join(miss) + ("\n" if miss else ""))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("OUT", OUT)
    print("videos needed", len(miss), miss)


if __name__ == "__main__":
    main()
