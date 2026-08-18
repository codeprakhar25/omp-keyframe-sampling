#!/usr/bin/env python3
"""Build tabbed visual analysis HTML for 600s + 3600s clean packs.

Output: results/clean_pack_analysis/index.html
Self vs peer frames only (no gold).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/clean_pack_analysis"
RESULTS = ROOT / "results"

# Concise visual notes (no gold). Key: (bin, k, arm, qid)
NOTES: dict[tuple, dict] = {
    # ---- 3600 k8 ----
    (3600, 8, "topk_lc", "iwXp1fT89-M_2"): {
        "blurb": "SSS order. topk stuck mid-video (ties on bed ~297s). OMP hits early blue-sleeveless woman (creamer / hands-on-face).",
    },
    (3600, 8, "topk_lc", "1D9TgBrW6Sw_1"): {
        "blurb": "Wrong chapter. topk: arrest + Packers. OMP: Airport/707 cockpit graphic. Same doc, different segment.",
    },
    (3600, 8, "topk_lc", "yFAuXmcGk2Y_2"): {
        "blurb": "Cluster collapse: all 8 topk picks in ~7s on green A/B board. OMP spreads more; both still wrong attribute.",
    },
    (3600, 8, "topk_lc", "x4UBaEojM6U_2"): {
        "blurb": "Right bathroom scene. Both say index finger; need middle finger. Fine-motor, not retrieval miss.",
    },
    (3600, 8, "topk_lc", "to7vCdkLi4s_2"): {
        "blurb": "Right ML slides (cutout/jitter). Same wrong subtitle line. Subs off → expected hard.",
    },
    (3600, 8, "topk_lc", "bwDfdTh0VYs_0"): {
        "blurb": "UI/icons visible. Both under-count (“two circular icons”).",
    },
    (3600, 8, "topk_lc", "_ZIa6SEJEyg_0"): {
        "blurb": "topk glued to first ~13s talking head. OMP mixes later chemistry slide. Count still wrong.",
    },
    (3600, 8, "topk_lc", "TMe7oXMJoSM_0"): {
        "blurb": "Stage/singing frames. Both “kneeling” @k8. Becomes OMP flip @k32.",
    },
    (3600, 8, "topk_lc", "Bwnkg6GbXwU_0"): {
        "blurb": "Horses on highway clearly visible. Direction/action nuance wrong.",
    },
    (3600, 8, "topk_lc", "fPLjjr8w6DU_1"): {
        "blurb": "Bathtub/petals scene. Pink vs white floral dress.",
    },
    # ---- 3600 k32 ----
    (3600, 32, "topk_lc", "o2F-N42Ufo4_2"): {
        "blurb": "SSS montage. Shared “1981” cards early; order still differs. More k ≠ agreement (j~0.25).",
    },
    (3600, 32, "topk_lc", "TMe7oXMJoSM_0"): {
        "blurb": "OMP wins via diversity: wide stage (“oh freedom”) + later beats. topk dens30=23 on singing peaks.",
    },
    (3600, 32, "omp_lc", "YcbKamVxDzI_2"): {
        "blurb": "Reverse flip: OMP picks wrong person (man/white) vs topk woman/yellow.",
    },
    (3600, 32, "topk_lc", "Bwnkg6GbXwU_0"): {
        "blurb": "Still wrong direction @k32 (option drifts, not to gold).",
    },
    (3600, 32, "topk_lc", "M5YKW6fhlss_2"): {
        "blurb": "Both wrong, different distractors (insect vs noseless character).",
    },
    (3600, 32, "omp_lc", "M5YKW6fhlss_1"): {
        "blurb": "Shared cartoon distractor vs cave/river answer.",
    },
    # ---- 600 k8 ----
    (600, 8, "topk_lc", "JASFwBtUK40_0"): {
        "blurb": "Name ID. topk ~89s workshop apparatus → “Patrick Craine”. OMP ~25s interview lower-third Andrés Jaque ✓.",
    },
    (600, 8, "topk_lc", "0oALTLKRWBA_1"): {
        "blurb": "SSS. Both see blonde/rose early; order of later beats wrong for topk.",
    },
    (600, 8, "topk_lc", "X0U3fP0tZyY_1"): {
        "blurb": "Museum woman. topk “white guitar” vs OMP “black/red string instrument”.",
    },
    (600, 8, "topk_lc", "mH9LdC7IFH8_1"): {
        "blurb": "Before armored speaker: OMP wine-glass girl; topk car-on-grass distractor.",
    },
    (600, 8, "omp_lc", "8905KCkLDYc_0"): {
        "blurb": "Beach party. OMP→phone; topk→American flag. Classic object swap on similar scenes.",
    },
    (600, 8, "omp_lc", "F8Ma1qs0Rkg_1"): {
        "blurb": "Subtitle appear/disappear. OMP “extra line”; topk correct disappear. OCR/timing.",
    },
    (600, 8, "omp_lc", "UwlKYM2Sotg_0"): {
        "blurb": "Three men. OMP folded arms; topk hand-on-shoulder. Near-identical people shots.",
    },
    (600, 8, "topk_lc", "kj3Po7zUeyw_1"): {
        "blurb": "Who dies first → both Sarah not Philip. Likely dialogue/plot; visuals ambiguous.",
    },
    (600, 8, "topk_lc", "SO3czkzeFjw_0"): {
        "blurb": "SSS sticky both-fail. Different wrong orders; neither covers full chain.",
    },
    (600, 8, "topk_lc", "UbNyMSwoT5A_0"): {
        "blurb": "Elephant trek. Same wrong “helmet without scabbard” (shield vs scabbard).",
    },
    (600, 8, "topk_lc", "Ng2rNm6Nwsg_0"): {
        "blurb": "Paintings→old man. Same wrong letter @k8; OMP flips OK @k32.",
    },
    (600, 8, "omp_lc", "XR3Ov2nQ39s_0"): {
        "blurb": "Tank/military TOS. Same wrong subtitle fragment. Subs off.",
    },
    # ---- 600 k32 ----
    (600, 32, "topk_lc", "Ng2rNm6Nwsg_0"): {
        "blurb": "Now flip: OMP gets old-man-after-paintings; topk still wrong variant of same man.",
    },
    (600, 32, "topk_lc", "pFtKaT3GF9I_0"): {
        "blurb": "SSS flip @k32: OMP correct order; topk still green/white clothes opener. High jaccard but order differs.",
    },
    (600, 32, "topk_lc", "X0U3fP0tZyY_1"): {
        "blurb": "Still flip @k32: white guitar vs black/red instrument. Sticky object ID.",
    },
    (600, 32, "omp_lc", "JwoBdRC2fzE_0"): {
        "blurb": "OMP “running race” vs topk “wrestling” near statues/fists. Action category.",
    },
    (600, 32, "omp_lc", "vJ9hYCUDHTo_0"): {
        "blurb": "Mirror lady. OMP→picture/timestamps; topk→earrings ✓.",
    },
    (600, 32, "omp_lc", "IN0osLg-Mn8_1"): {
        "blurb": "Fight ground. OMP→hat; topk→watch ✓. Small object.",
    },
    (600, 32, "topk_lc", "QHS9ZZBdK-g_0"): {
        "blurb": "Both wrong hand laterality (right vs left fingers up).",
    },
    (600, 32, "topk_lc", "kj3Po7zUeyw_1"): {
        "blurb": "Still both Sarah≠Philip @k32. Not fixed by budget.",
    },
    (600, 32, "topk_lc", "jdbG9gmg_SA_1"): {
        "blurb": "SSS both still sunny/queue opener vs evening camping. Sticky.",
    },
    # ---- 3600 k16 ----
    (3600, 16, "topk_lc", "bwDfdTh0VYs_0"): {
        "blurb": "Flip @k16: OMP count OK; topk still under-counts icons. Was both-fail @k8.",
    },
    (3600, 16, "topk_lc", "o2F-N42Ufo4_2"): {
        "blurb": "SSS flip. Shared early cards; OMP order wins. Same family as k32 flip.",
    },
    (3600, 16, "omp_lc", "mfS6gyP0mwo_2"): {
        "blurb": "Reverse flip: OMP wrong attribute; topk OK. SAA fine-detail.",
    },
    # ---- 600 k16 ----
    (600, 16, "topk_lc", "JASFwBtUK40_0"): {
        "blurb": "Name ID sticky flip: topk workshop→Patrick; OMP lower-third Andrés. Same as k8.",
    },
    (600, 16, "topk_lc", "0oALTLKRWBA_1"): {
        "blurb": "SSS flip. Blonde/rose early; OMP later-beat order correct.",
    },
    (600, 16, "topk_lc", "pFtKaT3GF9I_0"): {
        "blurb": "SSS flip (also @k32): OMP order; topk green/white opener.",
    },
    (600, 16, "topk_lc", "X0U3fP0tZyY_1"): {
        "blurb": "Museum instrument sticky: white guitar vs black/red. Flip across k.",
    },
    (600, 16, "omp_lc", "JwoBdRC2fzE_0"): {
        "blurb": "OMP “running race” vs topk “wrestling”. Action category (also @k32).",
    },
    (600, 16, "omp_lc", "F8Ma1qs0Rkg_1"): {
        "blurb": "Subtitle appear/disappear. OMP extra line; topk OK. OCR/timing.",
    },
    (600, 16, "omp_lc", "UwlKYM2Sotg_0"): {
        "blurb": "Three men pose. OMP folded arms; topk hand-on-shoulder.",
    },
}

# Copy sticky 3600 both-fail notes to later k where missing
for qid in [
    "yFAuXmcGk2Y_2", "x4UBaEojM6U_2", "to7vCdkLi4s_2",
    "_ZIa6SEJEyg_0", "fPLjjr8w6DU_1", "iwXp1fT89-M_2",
]:
    src = NOTES.get((3600, 8, "topk_lc", qid))
    if not src:
        continue
    for kk in (16, 32):
        if (3600, kk, "topk_lc", qid) not in NOTES:
            NOTES[(3600, kk, "topk_lc", qid)] = {
                "blurb": src["blurb"] + f" Sticky @k{kk}.",
            }

# Sticky 600 both-fails → k16
for qid, blurb in [
    ("SO3czkzeFjw_0", "SSS sticky both-fail. Different wrong orders; neither covers full chain."),
    ("UbNyMSwoT5A_0", "Elephant trek. Same wrong “helmet without scabbard” (shield vs scabbard)."),
    ("Ng2rNm6Nwsg_0", "Paintings→old man. Wrong @k16; OMP flips OK @k32."),
    ("jdbG9gmg_SA_1", "SSS both still sunny/queue opener vs evening camping. Sticky."),
    ("kj3Po7zUeyw_1", "Who dies first → both Sarah not Philip. Likely dialogue/plot."),
]:
    if (600, 16, "topk_lc", qid) not in NOTES:
        NOTES[(600, 16, "topk_lc", qid)] = {"blurb": blurb + " @k16."}


def load(pack: str, arm: str, kind: str) -> dict:
    return json.loads((RESULTS / pack / arm / kind / "data/cases.json").read_text())


def jacc(a, b) -> float:
    s = {round(float(x), 1) for x in a}
    p = {round(float(x), 1) for x in b}
    return len(s & p) / len(s | p) if s or p else 0.0


def dens30(secs) -> int:
    secs = [float(x) for x in secs]
    if not secs:
        return 0
    return max(sum(1 for u in secs if abs(u - t) <= 30) for t in secs)


def thumbs(base: str, paths: list | None, n: int = 4) -> str:
    parts = []
    for rel in (paths or [])[:n]:
        src = f"{base}/{rel}"
        parts.append(
            f'<div class="thumb"><img src="{html.escape(src)}" loading="lazy" alt="">'
            f'<div class="cap">{html.escape(Path(rel).name)}</div></div>'
        )
    return "".join(parts) if parts else '<div class="meta">(no frames)</div>'


def case_panel(bin_: int, k: int, pack: str, arm: str, label: str, peer_label: str, c: dict, idx: int) -> str:
    peer = c.get("peer") or {}
    pok = peer.get("ok")
    mode = "FLIP" if pok else "BOTH"
    note = (
        NOTES.get((bin_, k, arm, c["id"]))
        or NOTES.get((bin_, k, "topk_lc", c["id"]))
        or {}
    )
    blurb = note.get("blurb") or f"{mode} · {c.get('category')} — see frames."
    j = jacc(c.get("secs") or [], peer.get("secs") or [])
    d = dens30(c.get("secs") or [])
    base = f"../{pack}/{arm}/fail"
    cid = f"c-{bin_}-k{k}-{arm}-{c['id']}"
    opts = []
    for o in c.get("options") or []:
        marks = []
        if o["letter"] == c["gold_letter"]:
            marks.append("GOLD")
        if o["letter"] == c["pred"]:
            marks.append(f"{label}")
        if o["letter"] == peer.get("pred"):
            marks.append(f"{peer_label}")
        extra = f' <span class="marks">← {", ".join(marks)}</span>' if marks else ""
        opts.append(f'<li><b>{o["letter"]}</b>) {html.escape(o["text"])}{extra}</li>')
    secs_s = ", ".join(f"{float(s):.0f}" for s in (c.get("secs") or [])[:8])
    if len(c.get("secs") or []) > 8:
        secs_s += "…"
    psecs = ", ".join(f"{float(s):.0f}" for s in (peer.get("secs") or [])[:8])
    if len(peer.get("secs") or []) > 8:
        psecs += "…"
    return f'''
<div class="case-panel" id="{cid}" data-mode="{mode}" hidden>
  <div class="case-head">
    <span class="pill {mode.lower()}">{mode}</span>
    <code>{html.escape(c["id"])}</code>
    <span class="dim">{html.escape(str(c.get("category")))} · {c.get("video_seconds")}s</span>
    <a class="packlink" href="{base}/index.html" target="_blank">pack ↗</a>
  </div>
  <p class="blurb">{html.escape(blurb)}</p>
  <p class="q">{html.escape(c.get("question_stem") or "")}</p>
  <ul class="opts">{"".join(opts)}</ul>
  <div class="statline">
    <b>{label}</b> → {c["pred"]} · dens30={d} · [{secs_s}]
    &nbsp;|&nbsp;
    <b>{peer_label}</b> {"OK" if pok else "FAIL"} → {peer.get("pred")} · j={j:.2f} · [{psecs}]
  </div>
  <div class="frameblock">
    <div class="flabel">{label}</div>
    <div class="row">{thumbs(base, c.get("frame_jpgs"), 4)}</div>
  </div>
  <div class="frameblock">
    <div class="flabel">{peer_label}</div>
    <div class="row peer">{thumbs(base, peer.get("frame_jpgs"), 4)}</div>
  </div>
</div>'''


def build_k_section(bin_: int, k: int, pack: str) -> str:
    parts = []
    for arm, label, peer_label in [
        ("topk_lc", "topk", "OMP"),
        ("omp_lc", "OMP", "topk"),
    ]:
        pk = load(pack, arm, "fail")
        flips = [c for c in pk["cases"] if (c.get("peer") or {}).get("ok")]
        both = [c for c in pk["cases"] if not (c.get("peer") or {}).get("ok")]
        acc = pk["meta"].get("acc")
        sid = f"{bin_}-k{k}-{arm}"
        # case buttons
        btns = []
        panels = []
        for i, c in enumerate(flips + both, 1):
            mode = "FLIP" if (c.get("peer") or {}).get("ok") else "BOTH"
            cid = f"c-{bin_}-k{k}-{arm}-{c['id']}"
            btns.append(
                f'<button type="button" class="case-btn {mode.lower()}" data-panel="{cid}">'
                f'{mode} · {html.escape(c["id"])}</button>'
            )
            panels.append(case_panel(bin_, k, pack, arm, label, peer_label, c, i))
        parts.append(f'''
<section class="arm" data-arm="{arm}">
  <div class="arm-head">
    <h3>{label}-lc FAIL <span class="dim">acc={acc} · flips={len(flips)} · both={len(both)}</span></h3>
  </div>
  <div class="mode-tabs" data-group="{sid}-mode">
    <button type="button" class="tab active" data-modefilter="ALL">All ({len(pk["cases"])})</button>
    <button type="button" class="tab" data-modefilter="FLIP">Flips ({len(flips)})</button>
    <button type="button" class="tab" data-modefilter="BOTH">Both-fail ({len(both)})</button>
  </div>
  <div class="case-nav" id="{sid}-nav">{"".join(btns)}</div>
  <div class="case-stage" id="{sid}-stage">
    <div class="placeholder">Pick a case ↑</div>
    {"".join(panels)}
  </div>
</section>''')
    return "\n".join(parts)


def overview(bin_: int) -> str:
    if bin_ == 3600:
        return '''
<div class="overview">
  <p><b>Signal:</b> most pack fails are <em>both-fail + same letter</em>. Flips rare but crisp (wrong chapter / SSS coverage / TAA state).</p>
  <ul>
    <li>k8 topk flips 2/10; OMP flips 0/10</li>
    <li>k16 topk flips 2/10; OMP flips 1/10 (incl. reverse mfS6…)</li>
    <li>k32 topk flips 2/10; OMP flips 1/10 (incl. reverse)</li>
    <li>Many frames already “look relevant” — errors = attribute / order / subtitle</li>
  </ul>
  <p class="dim">Acc: k8 .511 / .546 · k16 .546 / .580 · k32 .573 / .585 (topk / OMP)</p>
</div>'''
    return '''
<div class="overview">
  <p><b>Signal:</b> more flips than 3600 — selection sometimes decides (nameplate, object, action).</p>
  <ul>
    <li>k8 topk flips 4/10; OMP flips 3/10</li>
    <li>k16 topk flips 4/10; OMP flips 3/10</li>
    <li>k32 topk flips 3/10; OMP flips 4/10</li>
    <li>Sticky: SSS chains, death-order (kj3…), TOS without subs · name/instrument flips persist across k</li>
  </ul>
  <p class="dim">Acc: k8 .614 / .631 · k16 .650 / .658 · k32 .648 / .665 (topk / OMP)</p>
</div>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bins = [
        (3600, {
            8: "clean_pack_k8_3600",
            16: "clean_pack_k16_3600",
            32: "clean_pack_k32_3600",
        }),
        (600, {
            8: "clean_pack_k8_600",
            16: "clean_pack_k16_600",
            32: "clean_pack_k32_600",
        }),
    ]

    bin_panels = []
    for bin_, packs in bins:
        k_tabs = []
        k_bodies = []
        # overview tab
        k_tabs.append(f'<button type="button" class="tab active" data-ktab="{bin_}-ov">Overview</button>')
        k_bodies.append(f'<div class="ktab-panel active" id="{bin_}-ov">{overview(bin_)}</div>')
        for k, pack in packs.items():
            k_tabs.append(f'<button type="button" class="tab" data-ktab="{bin_}-k{k}">k={k}</button>')
            k_bodies.append(
                f'<div class="ktab-panel" id="{bin_}-k{k}" hidden>'
                f'<p class="dim">Pack: <a href="../{pack}/index.html" target="_blank">{pack}</a> · self vs peer only</p>'
                f'{build_k_section(bin_, k, pack)}</div>'
            )
        bin_panels.append(f'''
<div class="bin-panel {"active" if bin_==3600 else ""}" id="bin-{bin_}" {"hidden" if bin_!=3600 else ""}>
  <div class="ktabs">{"".join(k_tabs)}</div>
  {"".join(k_bodies)}
</div>''')

    page = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clean pack analysis · 3600 & 600</title>
<style>
:root {{
  --bg:#fafaf8; --card:#fff; --ink:#1a1a1a; --muted:#666;
  --line:#e4e2dc; --flip:#1f7a4d; --both:#a33; --accent:#2c5aa0;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font:15px/1.45 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  color:var(--ink); background:var(--bg);
}}
header {{
  padding:20px 24px 12px; border-bottom:1px solid var(--line); background:var(--card);
  position:sticky; top:0; z-index:5;
}}
header h1 {{ margin:0 0 4px; font-size:1.35rem; font-weight:700; letter-spacing:-0.02em; }}
header .sub {{ color:var(--muted); font-size:13px; margin:0 0 12px; }}
.bintabs, .ktabs, .mode-tabs {{ display:flex; flex-wrap:wrap; gap:6px; }}
.tab, .case-btn {{
  font:13px/1.2 system-ui,sans-serif; border:1px solid var(--line); background:#f3f2ee;
  color:var(--ink); padding:7px 12px; border-radius:6px; cursor:pointer;
}}
.tab:hover, .case-btn:hover {{ background:#ebeae4; }}
.tab.active {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
main {{ max-width:1100px; margin:0 auto; padding:18px 24px 48px; }}
.bin-panel[hidden], .ktab-panel[hidden], .case-panel[hidden] {{ display:none !important; }}
.overview {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
.overview ul {{ margin:8px 0 0; padding-left:1.2em; }}
.dim {{ color:var(--muted); font-size:13px; font-family:system-ui,sans-serif; }}
.arm {{ margin:18px 0 28px; }}
.arm-head h3 {{ margin:0 0 8px; font-size:1.05rem; }}
.case-nav {{
  display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 12px; max-height:9.5em; overflow:auto;
  padding:8px; background:var(--card); border:1px solid var(--line); border-radius:8px;
}}
.case-btn.flip {{ border-color:#b7d7c5; }}
.case-btn.both {{ border-color:#e0b4b4; }}
.case-btn.active {{ outline:2px solid var(--accent); background:#e8eef8; }}
.case-btn[hidden] {{ display:none !important; }}
.case-stage {{
  background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px;
  min-height:200px;
}}
.placeholder {{ color:var(--muted); font-family:system-ui,sans-serif; font-size:13px; padding:24px 0; text-align:center; }}
.case-head {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; font-family:system-ui,sans-serif; font-size:13px; }}
.pill {{ font-size:11px; font-weight:700; color:#fff; padding:2px 7px; border-radius:4px; }}
.pill.flip {{ background:var(--flip); }}
.pill.both {{ background:var(--both); }}
.blurb {{ margin:0 0 8px; font-size:15px; }}
.q {{ margin:0 0 8px; font-size:14px; color:#333; }}
.opts {{ margin:0 0 10px; padding-left:1.1em; font-size:13px; font-family:system-ui,sans-serif; }}
.marks {{ color:var(--accent); font-size:12px; }}
.statline {{ font:12px/1.4 ui-monospace,Menlo,monospace; color:#444; margin:0 0 10px; word-break:break-word; }}
.frameblock {{ margin-top:8px; }}
.flabel {{ font:12px system-ui,sans-serif; color:var(--muted); margin-bottom:4px; }}
.row {{ display:flex; flex-wrap:wrap; gap:6px; }}
.thumb img {{ height:88px; border:2px solid #555; background:#111; display:block; }}
.row.peer img {{ border-color:var(--flip); }}
.cap {{ font:10px system-ui,sans-serif; color:#777; max-width:130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.packlink {{ margin-left:auto; color:var(--accent); text-decoration:none; font-size:12px; }}
.warn {{
  font:13px system-ui,sans-serif; background:#fff6df; border:1px solid #ead9a8;
  padding:8px 10px; border-radius:6px; margin:0 0 14px;
}}
</style>
</head>
<body>
<header>
  <h1>Clean pack analysis</h1>
  <p class="sub">3600s + 600s · fail cases · self vs peer frames · no gold</p>
  <div class="bintabs">
    <button type="button" class="tab active" data-bin="3600">3600s</button>
    <button type="button" class="tab" data-bin="600">600s</button>
  </div>
</header>
<main>
  <div class="warn">Click a bin → Overview or k → filter Flips/Both → one case. Only that panel shows.</div>
  {"".join(bin_panels)}
</main>
<script>
(function() {{
  function show(el, on) {{ if (!el) return; el.hidden = !on; el.classList.toggle('active', on); }}

  // bin tabs
  document.querySelectorAll('.bintabs .tab').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.bintabs .tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const bin = btn.dataset.bin;
      document.querySelectorAll('.bin-panel').forEach(p => show(p, p.id === 'bin-' + bin));
    }});
  }});

  // k tabs inside each bin
  document.querySelectorAll('.bin-panel').forEach(bin => {{
    bin.querySelectorAll('.ktabs .tab').forEach(btn => {{
      btn.addEventListener('click', () => {{
        bin.querySelectorAll('.ktabs .tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const id = btn.dataset.ktab;
        bin.querySelectorAll('.ktab-panel').forEach(p => show(p, p.id === id));
      }});
    }});
  }});

  // mode filter + case select
  document.querySelectorAll('.arm').forEach(arm => {{
    const nav = arm.querySelector('.case-nav');
    const stage = arm.querySelector('.case-stage');
    const modeTabs = arm.querySelector('.mode-tabs');

    function applyMode(mode) {{
      modeTabs.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.modefilter === mode));
      nav.querySelectorAll('.case-btn').forEach(b => {{
        const panel = document.getElementById(b.dataset.panel);
        const m = panel?.dataset.mode;
        const ok = mode === 'ALL' || m === mode;
        b.hidden = !ok;
        if (!ok && b.classList.contains('active')) {{
          b.classList.remove('active');
          stage.querySelectorAll('.case-panel').forEach(p => p.hidden = true);
          const ph = stage.querySelector('.placeholder');
          if (ph) ph.hidden = false;
        }}
      }});
    }}

    modeTabs.querySelectorAll('.tab').forEach(t => {{
      t.addEventListener('click', () => applyMode(t.dataset.modefilter));
    }});

    nav.querySelectorAll('.case-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        nav.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const id = btn.dataset.panel;
        stage.querySelectorAll('.case-panel').forEach(p => p.hidden = p.id !== id);
        const ph = stage.querySelector('.placeholder');
        if (ph) ph.hidden = true;
      }});
    }});
  }});
}})();
</script>
</body></html>
'''
    (OUT / "index.html").write_text(page)
    # point old 3600 analysis at new hub
    old = RESULTS / "clean_pack_3600_analysis" / "index.html"
    if old.parent.exists():
        old.write_text(
            '<!DOCTYPE html><meta charset=utf-8>'
            '<meta http-equiv="refresh" content="0;url=../clean_pack_analysis/index.html">'
            '<p>Moved → <a href="../clean_pack_analysis/index.html">clean_pack_analysis</a></p>'
        )
    print("WROTE", OUT / "index.html", (OUT / "index.html").stat().st_size)


if __name__ == "__main__":
    main()
