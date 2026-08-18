#!/usr/bin/env python3
"""Short-lived OMP-fail visual browser for pod (port 5902).

Tabs: bin → k → category → case.
Case: frames (topk vs OMP) + optional video + on-demand score graph + k16 strip
+ on-demand OMP residual trace (k=8): score curves after each pick × α sweep.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
import uvicorn

ROOT = Path(os.environ.get("SLM_ROOT", "/workspace/slm-lab"))
VID_DIR = ROOT / "data/videos"
CACHE = ROOT / "results/viz_cache"
PICKS_DIR = ROOT / "results/picks_lmmseval"
RESULTS = ROOT / "results/lmmseval_matrix_clean"

BINS = (600, 3600)
KS = (8, 16)
ALPHAS = (0.0, 0.5, 0.75, 1.0)  # α=0 topk · α=1 textbook OMP

CAT_NAME = {
    "S2E": "Scene-referred Event",
    "S2O": "Scene-referred Object Existence",
    "S2A": "Scene-referred Object Attribute",
    "E2O": "Event-referred Object",
    "O2E": "Object-referred Event",
    "T2E": "Text-referred Event",
    "T2O": "Text-referred Object Existence",
    "T2A": "Text-referred Object Attribute",
    "E3E": "Event before/after Event",
    "O3O": "Object before/after Object",
    "SSS": "Sequence of Scenes",
    "SOS": "Scene-referred Object Tracking",
    "SAA": "Scene-referred Object Attribute Change",
    "T3E": "Event before/after Text",
    "T3O": "Object before/after Text",
    "TOS": "Text-referred Object Tracking",
    "TAA": "Text-referred Object Attribute Change",
}

SAMPLE_ROOTS = {
    (600, 8): RESULTS / "k8_600",
    (600, 16): RESULTS / "k16_600",
    (3600, 8): RESULTS / "k8_3600",
    (3600, 16): RESULTS / "k16_3600_merged",
}

app = FastAPI()
STATE: dict = {}


def _stem(q: str) -> str:
    idx = q.find("\n\nOptions:")
    if idx == -1:
        idx = q.find("Options:")
    return (q[:idx] if idx != -1 else q).strip()


def _parse_opts(text: str):
    return [{"letter": m.group(1), "text": m.group(2).strip()}
            for m in re.finditer(r"^([A-E])\.\s*(.+)$", text or "", re.M)]


def _load_arm_rows(samples_root: Path, substr: str) -> dict:
    paths = [p for p in samples_root.rglob("*samples*.jsonl") if substr in p.name]
    if not paths:
        raise FileNotFoundError(f"{substr} under {samples_root}")
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
            "ok": pred == gold,
            "pred": pred,
            "gold": gold,
            "cat": a.get("question_category") or "?",
            "input": s.get("input", ""),
            "question": a.get("question") or "",
        }
    return rows


def _load_manifest() -> dict:
    out = {}
    for name in ("manifest.lvb.full1560.json", "manifest.lvb.long976.json",
                 "manifest.lvb.full1560.pod.json"):
        p = ROOT / "data" / name
        if not p.exists():
            continue
        for x in json.loads(p.read_text()):
            if isinstance(x, dict) and "id" in x:
                out[x["id"]] = x
    return out


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def build_index():
    picks = {
        ("topk", k): json.loads((PICKS_DIR / f"picks_lc_k{k}.json").read_text())
        for k in KS
    }
    picks.update({
        ("omp", k): json.loads((PICKS_DIR / f"picks_omp_lc_k{k}.json").read_text())
        for k in KS
    })
    manifest = _load_manifest()
    cases = {}  # (bin,k) -> {qid: case}
    by_cat = {}  # (bin,k) -> cat -> [qid]
    outcomes = {}  # (bin,k,arm,qid) -> ok
    cat_stats = {}  # (bin,k) -> cat -> {n, omp_fail, topk_fail}

    for bin_ in BINS:
        for k in KS:
            root = SAMPLE_ROOTS[(bin_, k)]
            topk = _load_arm_rows(root, f"picks_lc_{bin_}s_k{k}")
            omp = _load_arm_rows(root, f"picks_omp_lc_{bin_}s_k{k}")
            for qid, r in topk.items():
                outcomes[(bin_, k, "topk", qid)] = r["ok"]
            for qid, r in omp.items():
                outcomes[(bin_, k, "omp", qid)] = r["ok"]

            # full-N category fail counts (OMP + topk)
            stats = defaultdict(lambda: {"n": 0, "omp_fail": 0, "topk_fail": 0})
            ids = sorted(set(topk) & set(omp))
            for qid in ids:
                cat = omp[qid]["cat"]
                stats[cat]["n"] += 1
                if not omp[qid]["ok"]:
                    stats[cat]["omp_fail"] += 1
                if not topk[qid]["ok"]:
                    stats[cat]["topk_fail"] += 1
            cat_stats[(bin_, k)] = dict(stats)

            fail_ids = sorted(qid for qid, r in omp.items() if not r["ok"])
            catmap = defaultdict(list)
            cmap = {}
            for qid in fail_ids:
                o = omp[qid]
                t = topk.get(qid, {})
                m = manifest.get(qid, {})
                vf = m.get("video_file") or (Path(m["media_path"]).name if m.get("media_path") else None)
                if not vf and "_" in qid:
                    vf = qid.rsplit("_", 1)[0] + ".mp4"
                cat = o["cat"]
                catmap[cat].append(qid)
                cmap[qid] = {
                    "id": qid,
                    "bin": bin_,
                    "k": k,
                    "cat": cat,
                    "cat_name": CAT_NAME.get(cat, cat),
                    "gold": o["gold"],
                    "omp_pred": o["pred"],
                    "topk_pred": t.get("pred"),
                    "topk_ok": t.get("ok"),
                    "stem": _stem(o.get("question") or o.get("input") or ""),
                    "options": _parse_opts(o.get("input") or ""),
                    "video_file": vf,
                    "video_seconds": m.get("video_seconds"),
                    "omp_secs": list(picks[("omp", k)].get(qid, [])),
                    "topk_secs": list(picks[("topk", k)].get(qid, [])),
                }
            cases[(bin_, k)] = cmap
            by_cat[(bin_, k)] = {c: sorted(ids) for c, ids in sorted(catmap.items())}

    # text embeds by bin
    text_by_bin = {}
    for bin_ in BINS:
        for name in (f"text_lc_{bin_}.npz", "text_lc_all.npz"):
            p = ROOT / "results/embeds_text" / name
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            ids = [str(x) for x in z["ids"]]
            text = _l2(z["text"].astype(np.float32))
            text_by_bin[bin_] = {i: text[j] for j, i in enumerate(ids)}
            break

    STATE.update(
        cases=cases, by_cat=by_cat, outcomes=outcomes, cat_stats=cat_stats,
        text_by_bin=text_by_bin, manifest=manifest,
    )
    n = sum(len(v) for v in cases.values())
    print(f"indexed OMP fails: {n} across {len(cases)} settings", flush=True)


@app.on_event("startup")
def _startup():
    CACHE.mkdir(parents=True, exist_ok=True)
    build_index()


def _css() -> str:
    return """
:root { --bg:#f7f6f2; --card:#fff; --ink:#1a1a1a; --mut:#666; --line:#ddd8ce;
  --omp:#1f6feb; --topk:#c45c26; --ok:#1f7a4d; --bad:#a33; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.45 system-ui,sans-serif; color:var(--ink); background:var(--bg); }
header { background:var(--card); border-bottom:1px solid var(--line); padding:14px 18px;
  position:sticky; top:0; z-index:5; }
h1 { margin:0 0 4px; font-size:1.2rem; }
.sub { color:var(--mut); font-size:13px; margin:0 0 10px; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0; }
.tab { border:1px solid var(--line); background:#eeebe4; padding:6px 11px; border-radius:6px;
  text-decoration:none; color:var(--ink); font-size:13px; }
.tab.active, .tab:hover { background:var(--ink); color:#fff; border-color:var(--ink); }
main { max-width:1200px; margin:0 auto; padding:16px 18px 48px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }
.pill { display:inline-block; font-size:11px; font-weight:700; color:#fff; padding:2px 7px; border-radius:4px; }
.pill.bad { background:var(--bad); } .pill.ok { background:var(--ok); }
.pill.omp { background:var(--omp); } .pill.topk { background:var(--topk); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:8px; }
.cat a { display:block; padding:10px 12px; border:1px solid var(--line); border-radius:8px;
  background:var(--card); text-decoration:none; color:var(--ink); }
.cat a:hover { border-color:#999; }
.case-list a { display:block; padding:7px 10px; border-bottom:1px solid var(--line);
  text-decoration:none; color:var(--ink); font-size:13px; }
.case-list a:hover { background:#f0eee8; }
.frames { display:flex; flex-wrap:wrap; gap:6px; }
.frames img { height:86px; border:2px solid #444; background:#111; }
.frames.omp img { border-color:var(--omp); }
.frames.topk img { border-color:var(--topk); }
.cap { font-size:10px; color:var(--mut); }
.btn { border:1px solid var(--line); background:#1a1a1a; color:#fff; padding:8px 14px;
  border-radius:6px; cursor:pointer; font-size:13px; }
.btn:disabled { opacity:.5; }
.muted { color:var(--mut); font-size:13px; }
.opts { margin:8px 0; padding-left:1.1em; font-size:13px; }
.marks { color:var(--omp); }
video { width:100%; max-height:360px; background:#000; border-radius:8px; }
.strip { background:#fff6df; border:1px solid #ead9a8; border-radius:8px; padding:10px 12px; margin-top:14px; }
code { font-size:12px; }
table.cmp { width:100%; border-collapse:collapse; font-size:13px; }
table.cmp th, table.cmp td { border:1px solid var(--line); padding:6px 8px; text-align:right; }
table.cmp th { background:#f0eee8; text-align:center; font-weight:600; }
table.cmp td.cat, table.cmp th.cat { text-align:left; white-space:nowrap; }
table.cmp td.num { font-variant-numeric:tabular-nums; }
.delta-down { background:#d8f0e0; color:#145c32; font-weight:600; }  /* fewer fails = better */
.delta-up { background:#f8d4d4; color:#8a1f1f; font-weight:600; }    /* more fails = worse */
.delta-flat { background:#f0eee8; color:#555; }
.cell-omp { background:#e8f0fe; }
.cell-topk { background:#fff1e8; }
.legend span { display:inline-block; padding:2px 8px; border-radius:4px; margin-right:8px; font-size:12px; }
.btn-link { display:inline-block; border:1px solid var(--line); background:#1a1a1a; color:#fff;
  padding:10px 16px; border-radius:8px; text-decoration:none; font-size:14px; font-weight:600; }
.btn-link:hover { background:#333; }
"""


def _page(title: str, body: str, crumbs: str = "") -> HTMLResponse:
    html_doc = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_css()}</style></head><body>
<header><h1>OMP-fail visual browser</h1>
<p class="sub">OMP-lc fails only · frames+video · score graph on demand · short-lived</p>
{crumbs}</header><main>{body}</main></body></html>"""
    return HTMLResponse(html_doc)


def _crumbs(*parts):
    bits = ['<a class="tab" href="/">home</a>']
    for label, href in parts:
        if href:
            bits.append(f'<a class="tab" href="{href}">{html.escape(label)}</a>')
        else:
            bits.append(f'<span class="tab active">{html.escape(label)}</span>')
    return '<div class="tabs">' + "".join(bits) + "</div>"


def _pct(fail: int, n: int) -> str:
    return f"{100 * fail / n:.0f}%" if n else "—"


def _delta_cls(d: int) -> str:
    if d < 0:
        return "delta-down"
    if d > 0:
        return "delta-up"
    return "delta-flat"


def _compare_k8_k16_table(bin_: int) -> str:
    """4-col: OMP@k8 | OMP@k16 | topk@k8 | topk@k16 + deltas."""
    s8 = STATE["cat_stats"][(bin_, 8)]
    s16 = STATE["cat_stats"][(bin_, 16)]
    cats = sorted(
        set(s8) | set(s16),
        key=lambda c: -(s8.get(c, {}).get("omp_fail", 0) + s16.get(c, {}).get("omp_fail", 0)),
    )
    rows = []
    t = {k: {"n": 0, "omp": 0, "topk": 0} for k in (8, 16)}
    for cat in cats:
        a = s8.get(cat, {"n": 0, "omp_fail": 0, "topk_fail": 0})
        b = s16.get(cat, {"n": 0, "omp_fail": 0, "topk_fail": 0})
        for kk, st in ((8, a), (16, b)):
            t[kk]["n"] += st["n"]
            t[kk]["omp"] += st["omp_fail"]
            t[kk]["topk"] += st["topk_fail"]
        d_omp = b["omp_fail"] - a["omp_fail"]
        d_topk = b["topk_fail"] - a["topk_fail"]
        name = CAT_NAME.get(cat, cat)
        rows.append(
            f"<tr>"
            f"<td class='cat'><b>{html.escape(cat)}</b><br><span class='muted'>{html.escape(name)}</span></td>"
            f"<td class='num cell-omp'>{a['omp_fail']}<br><span class='muted'>{_pct(a['omp_fail'], a['n'])} · n={a['n']}</span></td>"
            f"<td class='num cell-omp {_delta_cls(d_omp)}'>{b['omp_fail']}"
            f"<br><span class='muted'>{_pct(b['omp_fail'], b['n'])} · Δ{d_omp:+d}</span></td>"
            f"<td class='num cell-topk'>{a['topk_fail']}<br><span class='muted'>{_pct(a['topk_fail'], a['n'])}</span></td>"
            f"<td class='num cell-topk {_delta_cls(d_topk)}'>{b['topk_fail']}"
            f"<br><span class='muted'>{_pct(b['topk_fail'], b['n'])} · Δ{d_topk:+d}</span></td>"
            f"</tr>"
        )
    d_omp_t = t[16]["omp"] - t[8]["omp"]
    d_topk_t = t[16]["topk"] - t[8]["topk"]
    rows.append(
        f"<tr>"
        f"<td class='cat'><b>TOTAL</b></td>"
        f"<td class='num cell-omp'><b>{t[8]['omp']}</b><br><span class='muted'>{_pct(t[8]['omp'], t[8]['n'])}</span></td>"
        f"<td class='num cell-omp {_delta_cls(d_omp_t)}'><b>{t[16]['omp']}</b>"
        f"<br><span class='muted'>{_pct(t[16]['omp'], t[16]['n'])} · Δ{d_omp_t:+d}</span></td>"
        f"<td class='num cell-topk'><b>{t[8]['topk']}</b><br><span class='muted'>{_pct(t[8]['topk'], t[8]['n'])}</span></td>"
        f"<td class='num cell-topk {_delta_cls(d_topk_t)}'><b>{t[16]['topk']}</b>"
        f"<br><span class='muted'>{_pct(t[16]['topk'], t[16]['n'])} · Δ{d_topk_t:+d}</span></td>"
        f"</tr>"
    )
    head = (
        "<table class='cmp'><thead><tr>"
        "<th class='cat'>category</th>"
        "<th class='cell-omp'>OMP fail<br>k=8</th>"
        "<th class='cell-omp'>OMP fail<br>k=16 (Δ vs k8)</th>"
        "<th class='cell-topk'>topk fail<br>k=8</th>"
        "<th class='cell-topk'>topk fail<br>k=16 (Δ vs k8)</th>"
        "</tr></thead><tbody>"
    )
    return head + "".join(rows) + "</tbody></table>"


def _compare_omp_vs_topk_table(bin_: int, k: int) -> str:
    """At fixed k: OMP fail | OMP% | topk fail | topk% · color by OMP−topk (green = OMP fewer fails)."""
    st = STATE["cat_stats"][(bin_, k)]
    cats = sorted(st.keys(), key=lambda c: -st[c]["omp_fail"])
    rows = []
    tot_n = tot_o = tot_t = 0
    for cat in cats:
        s = st[cat]
        n, of, tf = s["n"], s["omp_fail"], s["topk_fail"]
        tot_n += n
        tot_o += of
        tot_t += tf
        d = of - tf  # negative => OMP better
        name = CAT_NAME.get(cat, cat)
        rows.append(
            f"<tr>"
            f"<td class='cat'><b>{html.escape(cat)}</b><br><span class='muted'>{html.escape(name)}</span>"
            f" · <a href='/b/{bin_}/k/{k}/c/{cat}'>browse</a></td>"
            f"<td class='num cell-omp'>{of}<br><span class='muted'>{_pct(of, n)} · n={n}</span></td>"
            f"<td class='num cell-topk'>{tf}<br><span class='muted'>{_pct(tf, n)}</span></td>"
            f"<td class='num {_delta_cls(d)}'>Δ OMP−topk {d:+d}</td>"
            f"</tr>"
        )
    d_t = tot_o - tot_t
    rows.append(
        f"<tr><td class='cat'><b>TOTAL</b></td>"
        f"<td class='num cell-omp'><b>{tot_o}</b><br><span class='muted'>{_pct(tot_o, tot_n)}</span></td>"
        f"<td class='num cell-topk'><b>{tot_t}</b><br><span class='muted'>{_pct(tot_t, tot_n)}</span></td>"
        f"<td class='num {_delta_cls(d_t)}'><b>Δ {d_t:+d}</b></td></tr>"
    )
    head = (
        "<table class='cmp'><thead><tr>"
        "<th class='cat'>category</th>"
        f"<th class='cell-omp'>OMP fail<br>k={k}</th>"
        f"<th class='cell-topk'>topk fail<br>k={k}</th>"
        "<th>Δ (OMP − topk)<br>green = OMP fewer fails</th>"
        "</tr></thead><tbody>"
    )
    return head + "".join(rows) + "</tbody></table>"


@app.get("/", response_class=HTMLResponse)
def home():
    # k=16 cards first (current focus), then k=8
    cards16, cards8 = [], []
    for bin_ in BINS:
        for k, bucket in ((16, cards16), (8, cards8)):
            n = len(STATE["cases"][(bin_, k)])
            nc = len(STATE["by_cat"][(bin_, k)])
            bucket.append(
                f'<div class="cat"><a href="/b/{bin_}/k/{k}"><b>{bin_}s · k={k}</b><br>'
                f'<span class="muted">{n} OMP fails · {nc} cats</span></a></div>'
            )
    body = (
        '<div class="card" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">'
        '<a class="btn-link" href="/compare?mode=k16">Compare OMP vs topk @ k=16</a>'
        '<a class="btn-link" href="/compare?mode=budget">Compare k=8 vs k=16</a>'
        '<span class="muted">Case pages: frames + video + <b>Generate score graph</b> button</span>'
        '</div>'
        '<h2 style="margin:18px 0 8px;font-size:1.05rem">k=16 (focus)</h2>'
        f'<div class="grid">{"".join(cards16)}</div>'
        '<h2 style="margin:18px 0 8px;font-size:1.05rem">k=8</h2>'
        f'<div class="grid">{"".join(cards8)}</div>'
    )
    return _page("home", body, _crumbs(("home", None)))


@app.get("/compare", response_class=HTMLResponse)
def compare(bin_: int = Query(600), mode: str = Query("k16")):
    if bin_ not in BINS:
        bin_ = 600
    if mode not in ("k16", "budget", "k8"):
        mode = "k16"
    bin_tabs = "".join(
        f'<a class="tab {"active" if b==bin_ else ""}" href="/compare?bin_={b}&mode={mode}">{b}s</a>'
        for b in BINS
    )
    mode_tabs = (
        f'<a class="tab {"active" if mode=="k16" else ""}" href="/compare?bin_={bin_}&mode=k16">OMP vs topk @ k=16</a>'
        f'<a class="tab {"active" if mode=="k8" else ""}" href="/compare?bin_={bin_}&mode=k8">OMP vs topk @ k=8</a>'
        f'<a class="tab {"active" if mode=="budget" else ""}" href="/compare?bin_={bin_}&mode=budget">k=8 vs k=16 budget</a>'
    )
    if mode == "budget":
        legend = (
            '<p class="legend">'
            '<span class="delta-down">green / ↓ fewer fails when k↑</span>'
            '<span class="delta-up">red / ↑ more fails when k↑</span>'
            '<span class="cell-omp">blue = OMP</span><span class="cell-topk">orange = topk</span></p>'
        )
        table = _compare_k8_k16_table(bin_)
        title = f"{bin_}s — k8 vs k16 fail counts"
        note = "Full-N. Δ = k16 − k8 fail count."
    else:
        kk = 16 if mode == "k16" else 8
        legend = (
            '<p class="legend">'
            '<span class="delta-down">green = OMP has fewer fails than topk</span>'
            '<span class="delta-up">red = OMP has more fails than topk</span>'
            '<span class="cell-omp">blue = OMP</span><span class="cell-topk">orange = topk</span></p>'
        )
        table = _compare_omp_vs_topk_table(bin_, kk)
        title = f"{bin_}s · k={kk} — OMP vs topk fails by category"
        note = f"Full-N at k={kk}. Click browse → OMP-fail cases (frames + graph button)."
    links = []
    for cat in sorted(STATE["cat_stats"][(bin_, 16 if mode != "k8" else 8)].keys()):
        kshow = 16 if mode != "k8" else 8
        n_fail = len(STATE["by_cat"][(bin_, kshow)].get(cat, []))
        links.append(f'<a class="tab" href="/b/{bin_}/k/{kshow}/c/{cat}">{cat} · k{kshow} ({n_fail})</a>')
    body = (
        f'<div class="tabs">{bin_tabs}</div><div class="tabs">{mode_tabs}</div>{legend}'
        f'<div class="card"><h2>{title}</h2><p class="muted">{note}</p>{table}</div>'
        f'<div class="card"><h3>Browse OMP fails</h3><div class="tabs">{"".join(links)}</div></div>'
    )
    return _page("compare", body, _crumbs(("compare", None)))


@app.get("/b/{bin_}/k/{k}", response_class=HTMLResponse)
def bin_k(bin_: int, k: int):
    if (bin_, k) not in STATE["by_cat"]:
        raise HTTPException(404)
    # k tabs + bin tabs
    bin_tabs = "".join(
        f'<a class="tab {"active" if b==bin_ else ""}" href="/b/{b}/k/{k}">{b}s</a>'
        for b in BINS
    )
    k_tabs = "".join(
        f'<a class="tab {"active" if kk==k else ""}" href="/b/{bin_}/k/{kk}">k={kk}</a>'
        for kk in KS
    )
    cats = STATE["by_cat"][(bin_, k)]
    stats = STATE["cat_stats"][(bin_, k)]
    items = []
    for cat, ids in cats.items():
        name = CAT_NAME.get(cat, cat)
        st = stats.get(cat, {})
        of, tf, n = st.get("omp_fail", len(ids)), st.get("topk_fail", 0), st.get("n", 0)
        items.append(
            f'<div class="cat"><a href="/b/{bin_}/k/{k}/c/{cat}">'
            f'<b>{cat}</b> · {html.escape(name)}<br>'
            f'<span class="muted">OMP fail {of}/{n} ({_pct(of, n)}) · topk {tf}/{n} ({_pct(tf, n)})</span></a></div>'
        )
    body = f'<div class="tabs">{bin_tabs}</div><div class="tabs">{k_tabs}</div>'
    body += (
        f'<p><a class="btn-link" href="/compare?bin_={bin_}&mode=k{k}">OMP vs topk @ k={k}</a> '
        f'<a class="btn-link" href="/compare?bin_={bin_}&mode=budget">k8 vs k16 budget</a> '
        f'<span class="muted">{sum(len(v) for v in cats.values())} OMP fails @k={k}</span></p>'
    )
    body += f'<div class="grid">{"".join(items)}</div>'
    return _page(f"{bin_}s k={k}", body, _crumbs((f"{bin_}s k={k}", None)))


@app.get("/b/{bin_}/k/{k}/c/{cat}", response_class=HTMLResponse)
def cat_list(bin_: int, k: int, cat: str):
    ids = STATE["by_cat"].get((bin_, k), {}).get(cat)
    if ids is None:
        raise HTTPException(404)
    cases = STATE["cases"][(bin_, k)]
    rows = []
    for qid in ids:
        c = cases[qid]
        rescue = " · <span class='pill ok'>topk✓</span>" if c["topk_ok"] else " · <span class='pill bad'>both✗</span>"
        rows.append(
            f'<a href="/b/{bin_}/k/{k}/c/{cat}/q/{qid}">'
            f'<code>{html.escape(qid)}</code> · OMP→{c["omp_pred"]} gold={c["gold"]}{rescue}</a>'
        )
    name = CAT_NAME.get(cat, cat)
    body = f'<div class="card"><h2>{cat} — {html.escape(name)}</h2>'
    body += f'<p class="muted">{len(ids)} cases · {bin_}s · k={k}</p>'
    body += f'<div class="case-list card" style="padding:0">{"".join(rows)}</div></div>'
    return _page(f"{cat}", body, _crumbs(
        (f"{bin_}s", f"/b/{bin_}/k/{k}"),
        (f"k={k}", f"/b/{bin_}/k/{k}"),
        (cat, None),
    ))


def _ensure_frame(video: Path, sec: float, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 800:
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    t = max(0.0, float(sec))
    for ss in (t, max(0.0, t - 0.4), 0.0):
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 800:
            return True
    return False


def _frame_urls(bin_: int, k: int, qid: str, arm: str, secs: list, n: int = 8) -> list[tuple[str, float]]:
    c = STATE["cases"][(bin_, k)][qid]
    vf = c.get("video_file")
    if not vf:
        return []
    video = VID_DIR / vf
    if not video.exists():
        return []
    out = []
    for i, sec in enumerate(secs[:n]):
        jp = CACHE / qid / f"k{k}_{arm}_{i}_{float(sec):.2f}.jpg"
        if _ensure_frame(video, float(sec), jp):
            out.append((f"/frame/{qid}/k{k}/{arm}/{i}/{float(sec):.2f}", float(sec)))
    return out


def _k16_strip(qid: str, bin_: int, k: int) -> str:
    oc = STATE["outcomes"]
    parts = ["<div class='strip'><b>Cross-k outcome</b><br>"]
    for kk in KS:
        o = oc.get((bin_, kk, "omp", qid))
        t = oc.get((bin_, kk, "topk", qid))
        if o is None:
            parts.append(f"k={kk}: <span class='muted'>n/a</span> · ")
            continue
        op = '<span class="pill ok">OMP✓</span>' if o else '<span class="pill bad">OMP✗</span>'
        tp = '<span class="pill ok">topk✓</span>' if t else '<span class="pill bad">topk✗</span>'
        parts.append(f"k={kk}: {op} {tp} · ")
    o8 = oc.get((bin_, 8, "omp", qid))
    o16 = oc.get((bin_, 16, "omp", qid))
    if o8 is False and o16 is True:
        parts.append('<br><b style="color:var(--ok)">k=16 HELPED — OMP fails@k8, passes@k16</b>')
    elif o8 is False and o16 is False:
        parts.append('<br><span class="muted">OMP still fails @k16</span>')
    elif o16 is True and k == 16:
        parts.append('<br><span class="muted">OMP pass @k16 (this page is from fail list of another filter — should not happen)</span>')
    # if viewing k16 fail but would... it's a fail so o16 False
    if k == 16 and o8 is True and o16 is False:
        parts.append('<br><b style="color:var(--bad)">REGRESS — OMP pass@k8, fail@k16</b>')
    parts.append("</div>")
    return "".join(parts)


@app.get("/b/{bin_}/k/{k}/c/{cat}/q/{qid}", response_class=HTMLResponse)
def case_page(bin_: int, k: int, cat: str, qid: str):
    cases = STATE["cases"].get((bin_, k), {})
    c = cases.get(qid)
    if not c or c["cat"] != cat:
        raise HTTPException(404)

    omp_frames = _frame_urls(bin_, k, qid, "omp", c["omp_secs"])
    topk_frames = _frame_urls(bin_, k, qid, "topk", c["topk_secs"])

    def frame_html(frames, cls, label):
        if not frames:
            return f'<p class="muted">{label}: no frames (missing video?)</p>'
        imgs = "".join(
            f'<div><img src="{u}" loading="lazy"><div class="cap">{s:.1f}s</div></div>'
            for u, s in frames
        )
        return f'<div class="muted">{label}</div><div class="frames {cls}">{imgs}</div>'

    opts = []
    for o in c["options"]:
        marks = []
        if o["letter"] == c["gold"]:
            marks.append("GOLD")
        if o["letter"] == c["omp_pred"]:
            marks.append("OMP")
        if o["letter"] == c["topk_pred"]:
            marks.append("topk")
        extra = f' <span class="marks">← {", ".join(marks)}</span>' if marks else ""
        opts.append(f'<li><b>{o["letter"]}</b>) {html.escape(o["text"])}{extra}</li>')

    vf = c.get("video_file")
    vid_ok = bool(vf and (VID_DIR / vf).exists())
    video_block = ""
    if vid_ok:
        video_block = f'''
<details class="card"><summary>Show video ({html.escape(vf)})</summary>
<video controls preload="none" src="/video/{html.escape(vf)}"></video>
</details>'''
    else:
        video_block = f'<p class="muted">Video missing: {html.escape(str(vf))}</p>'

    rescue = "topk rescues this OMP fail" if c["topk_ok"] else "both fail"
    trace_card = ""
    if k == 8:
        trace_card = """
<div class="card">
  <h3>OMP residual trace (k=8)</h3>
  <p class="muted">On the fly. After each pick: score curve under current q_res · pick marked.
  α∈{0=topk, 0.5, 0.75, 1=OMP}. Scalars: ‖q_res‖, cos(q₀,q_res).</p>
  <button class="btn" id="tbtn" onclick="loadTrace()">Generate OMP residual trace</button>
  <div id="twrap" style="margin-top:10px"></div>
</div>
"""
    body = f'''
<div class="card">
  <div><span class="pill bad">OMP FAIL</span>
    <span class="pill omp">OMP→{c["omp_pred"]}</span>
    <span class="pill topk">topk→{c["topk_pred"]} {"✓" if c["topk_ok"] else "✗"}</span>
    <span class="muted">gold={c["gold"]} · {rescue}</span>
  </div>
  <p><code>{html.escape(qid)}</code> · {c["cat"]} · {html.escape(c["cat_name"])} · {bin_}s · k={k}</p>
  <p>{html.escape(c["stem"])}</p>
  <ul class="opts">{"".join(opts)}</ul>
  <p class="muted">OMP secs: {", ".join(f"{s:.1f}" for s in c["omp_secs"][:12])}{"…" if len(c["omp_secs"])>12 else ""}</p>
  <p class="muted">topk secs: {", ".join(f"{s:.1f}" for s in c["topk_secs"][:12])}{"…" if len(c["topk_secs"])>12 else ""}</p>
</div>
<div class="card">
  <h3>Frames</h3>
  {frame_html(omp_frames, "omp", "OMP-lc picks")}
  <div style="height:10px"></div>
  {frame_html(topk_frames, "topk", "topk-lc picks")}
</div>
{video_block}
<div class="card">
  <h3>Score graph</h3>
  <p class="muted">Not loaded by default (heavy). LongCLIP stem cosine over time; markers = picks.</p>
  <button class="btn" id="gbtn" onclick="loadGraph()">Generate score graph</button>
  <div id="gwrap" style="margin-top:10px"></div>
</div>
{trace_card}
{_k16_strip(qid, bin_, k)}
<script>
async function loadGraph() {{
  const btn = document.getElementById('gbtn');
  const wrap = document.getElementById('gwrap');
  btn.disabled = true; btn.textContent = 'Generating…';
  wrap.innerHTML = '<p class="muted">working…</p>';
  try {{
    const r = await fetch('/api/scoreplot/{bin_}/{k}/{qid}');
    if (!r.ok) {{ wrap.innerHTML = '<p class="muted">failed: '+r.status+' '+await r.text()+'</p>'; return; }}
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    wrap.innerHTML = '<img style="max-width:100%;border:1px solid #ddd" src="'+url+'">';
    btn.textContent = 'Regenerate';
  }} catch(e) {{
    wrap.innerHTML = '<p class="muted">'+e+'</p>';
  }} finally {{ btn.disabled = false; }}
}}
async function loadTrace() {{
  const btn = document.getElementById('tbtn');
  const wrap = document.getElementById('twrap');
  if (!btn) return;
  btn.disabled = true; btn.textContent = 'Generating…';
  wrap.innerHTML = '<p class="muted">replaying OMP × 4 alphas…</p>';
  try {{
    const r = await fetch('/api/omptrace/{bin_}/{qid}');
    if (!r.ok) {{ wrap.innerHTML = '<p class="muted">failed: '+r.status+' '+await r.text()+'</p>'; return; }}
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    wrap.innerHTML = '<img style="max-width:100%;border:1px solid #ddd" src="'+url+'">';
    btn.textContent = 'Regenerate';
  }} catch(e) {{
    wrap.innerHTML = '<p class="muted">'+e+'</p>';
  }} finally {{ btn.disabled = false; }}
}}
</script>
'''
    return _page(qid, body, _crumbs(
        (f"{bin_}s", f"/b/{bin_}/k/{k}"),
        (f"k={k}", f"/b/{bin_}/k/{k}"),
        (cat, f"/b/{bin_}/k/{k}/c/{cat}"),
        (qid, None),
    ))


@app.get("/frame/{qid}/k{k}/{arm}/{i}/{sec}")
def frame(qid: str, k: int, arm: str, i: int, sec: float):
    jp = CACHE / qid / f"k{k}_{arm}_{i}_{float(sec):.2f}.jpg"
    if not jp.exists():
        raise HTTPException(404)
    return FileResponse(jp, media_type="image/jpeg")


@app.get("/video/{name}")
def video(name: str):
    # prevent path escape
    name = Path(name).name
    p = VID_DIR / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


def _load_lc(qid: str, bin_: int):
    """Return (times, emb, q0) L2-normalized LongCLIP pair, or (None, err)."""
    text_map = STATE["text_by_bin"].get(bin_) or STATE["text_by_bin"].get(600) or {}
    if qid not in text_map:
        return None, "no text embed"
    q = text_map[qid].astype(np.float32)
    candidates = []
    if bin_ >= 3600:
        candidates += [(ROOT / "results/embeds" / f"{qid}.npz", "longclip"),
                       (ROOT / "results/embeds_lc" / f"{qid}.npz", "emb")]
    else:
        candidates += [(ROOT / "results/embeds_lc" / f"{qid}.npz", "emb"),
                       (ROOT / "results/embeds" / f"{qid}.npz", "longclip")]
    for p, key in candidates:
        if not p.exists():
            continue
        z = np.load(p)
        key_use = key if key in z.files else (
            "emb" if "emb" in z.files else ("longclip" if "longclip" in z.files else None)
        )
        if key_use is None or "times" not in z.files:
            continue
        emb = _l2(z[key_use].astype(np.float32))
        times = z["times"].astype(np.float32)
        return (times, emb, q), None
    return None, "no image embed"


def _score_curve(qid: str, bin_: int):
    pair, err = _load_lc(qid, bin_)
    if pair is None:
        return None, err
    times, emb, q = pair
    return (times, emb @ q), None


def _omp_trace(q0: np.ndarray, emb: np.ndarray, times: np.ndarray, k: int, alpha: float):
    """Step-by-step partial-deflation OMP. α=0 → topk; α=1 → shipped OMP.

    Returns list of step dicts (len=k): scores, pick idx/sec/score, residual scalars.
    """
    n = emb.shape[0]
    k = min(k, n)
    q = q0.astype(np.float32).copy()
    E = emb.astype(np.float32)
    q0n = float(np.linalg.norm(q0))
    basis: list = []
    chosen = np.zeros(n, dtype=bool)
    steps = []
    for step in range(k):
        scores = E @ q
        scores_view = scores.copy()
        scores_view[chosen] = -np.inf
        best = int(np.argmax(scores_view))
        pick_score = float(scores[best])
        q_norm = float(np.linalg.norm(q))
        cos0 = float((q @ q0) / max(q_norm * q0n, 1e-12))
        explained = 1.0 - (q_norm / max(q0n, 1e-12)) ** 2
        steps.append({
            "step": step,
            "scores": scores.astype(np.float32),
            "pick_i": best,
            "pick_sec": float(times[best]),
            "pick_score": pick_score,
            "q_norm": q_norm,
            "cos_q0": cos0,
            "explained": float(max(0.0, explained)),
        })
        chosen[best] = True
        v = E[best].copy()
        for b in basis:
            v -= (v @ b) * b
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v /= norm
            basis.append(v)
            q = q - float(alpha) * (q @ v) * v
    return steps


@app.get("/api/scoreplot/{bin_}/{k}/{qid}")
def scoreplot(bin_: int, k: int, qid: str):
    if qid not in STATE["cases"].get((bin_, k), {}):
        raise HTTPException(404, "not an OMP-fail case for this setting")
    c = STATE["cases"][(bin_, k)][qid]
    curve, err = _score_curve(qid, bin_)
    if curve is None:
        raise HTTPException(404, err or "no curve")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, scores = curve
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
    ax.plot(times, scores, color="#444", lw=0.9, alpha=0.85, label="LC stem cosine")
    for secs, color, marker, lab in (
        (c["omp_secs"], "#1f6feb", "o", "OMP"),
        (c["topk_secs"], "#c45c26", "D", "topk"),
    ):
        if not secs:
            continue
        xs, ys = [], []
        for s in secs:
            i = int(np.argmin(np.abs(times - float(s))))
            xs.append(float(times[i]))
            ys.append(float(scores[i]))
        ax.scatter(xs, ys, c=color, marker=marker, s=28, zorder=5, label=lab)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(f"{qid} · {bin_}s · k={k}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    # also cache
    out = CACHE / qid / f"score_k{k}_{bin_}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/api/omptrace/{bin_}/{qid}")
def omptrace(bin_: int, qid: str):
    """On-the-fly OMP residual visualization for k=8 OMP-fail cases.

    Layout (most info): scalar trajectories for all α, then per-α small-multiples
    of residual score curves with the pick at that step marked.
    """
    k = 8
    if qid not in STATE["cases"].get((bin_, k), {}):
        raise HTTPException(404, "not an OMP-fail case at k=8 for this bin")
    pair, err = _load_lc(qid, bin_)
    if pair is None:
        raise HTTPException(404, err or "no embeds")
    times, emb, q0 = pair

    traces = {a: _omp_trace(q0, emb, times, k, a) for a in ALPHAS}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # colors per alpha
    acol = {0.0: "#c45c26", 0.5: "#6a4c93", 0.75: "#2a9d8f", 1.0: "#1f6feb"}
    n_alpha = len(ALPHAS)
    # 2 scalar rows + one strip of k panels per alpha
    fig = plt.figure(figsize=(14, 2.4 + 2.1 * n_alpha), dpi=110)
    gs = GridSpec(2 + n_alpha, k, figure=fig, height_ratios=[1.1, 1.1] + [1.3] * n_alpha,
                  hspace=0.55, wspace=0.28)

    ax_norm = fig.add_subplot(gs[0, :])
    ax_cos = fig.add_subplot(gs[1, :])
    steps_x = list(range(k))
    for a in ALPHAS:
        tr = traces[a]
        ax_norm.plot(steps_x, [s["q_norm"] for s in tr], "-o", ms=4, color=acol[a],
                     label=f"α={a:g}", lw=1.4)
        ax_cos.plot(steps_x, [s["cos_q0"] for s in tr], "-o", ms=4, color=acol[a],
                    label=f"α={a:g}", lw=1.4)
    ax_norm.set_ylabel("‖q_res‖")
    ax_norm.set_title(f"{qid} · {bin_}s · k=8 · residual scalars (α=0 topk … α=1 OMP)")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.legend(loc="upper right", fontsize=8, ncol=4)
    ax_norm.set_xticks(steps_x)
    ax_cos.set_ylabel("cos(q₀, q_res)")
    ax_cos.set_xlabel("pick step (0 = before any deflation)")
    ax_cos.grid(True, alpha=0.25)
    ax_cos.set_xticks(steps_x)
    ax_cos.legend(loc="upper right", fontsize=8, ncol=4)

    # pick-order annotation under title area via text in first alpha row
    for ai, a in enumerate(ALPHAS):
        tr = traces[a]
        all_sc = np.concatenate([s["scores"] for s in tr])
        ymin, ymax = float(np.min(all_sc)), float(np.max(all_sc))
        pad = 0.05 * (ymax - ymin + 1e-6)
        picks_txt = "→".join(f"{s['pick_sec']:.0f}" for s in tr)
        for si, st in enumerate(tr):
            ax = fig.add_subplot(gs[2 + ai, si])
            ax.plot(times, st["scores"], color="#444", lw=0.7, alpha=0.85)
            ax.axvline(st["pick_sec"], color=acol[a], lw=1.2, alpha=0.9)
            ax.scatter([st["pick_sec"]], [st["pick_score"]], c=acol[a], s=22, zorder=5)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(True, alpha=0.2)
            if si == 0:
                ax.set_ylabel(f"α={a:g}\n{picks_txt}s", fontsize=7, color=acol[a])
            ax.set_title(
                f"step{st['step']} · {st['pick_sec']:.0f}s · ‖q‖={st['q_norm']:.2f}",
                fontsize=7,
            )
            if ai == n_alpha - 1:
                ax.set_xlabel("t (s)", fontsize=7)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=6)

    fig.suptitle(
        "emb·q_res after each pick (chosen excluded next). α=0=topk · α=1=OMP. "
        "Scalars = residual shrink.",
        fontsize=9, y=0.995,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    out = CACHE / qid / f"omptrace_k8_{bin_}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5902"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
