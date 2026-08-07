#!/usr/bin/env python3
"""Serve data/ lab pages + save dataset reviews to data/reviews/<id>.json

  .venv/bin/python scripts/serve_lab.py
  → http://127.0.0.1:8765/dataset-review.html
"""
from __future__ import annotations

import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVIEWS = DATA / "reviews"
SMOKE = DATA / "smoke"
SCALE = DATA / "scale"
HOST, PORT = "127.0.0.1", 8765
ID_RE = re.compile(r"^[a-z0-9_]+$")
TAG_RE = re.compile(r"<([^<>]+)>")

# freeze-ish tab order (events first — what IV actually covers)
IV_TAB_ORDER = [
    "thinking",
    "inhaling",
    "breathing",
    "sigh",
    "whispering",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
    "pause",
    "laugh",
    "angry",
    "sad",
    "excited",
    "fear",
    "surprise",
    "disgust",
    "none",
]


def _resolve_wav_url(clip: dict) -> str | None:
    """Return path relative to data/ for lab static serve, or None."""
    f = clip.get("file") or ""
    hp = clip.get("hf_or_path") or ""
    cid = clip.get("id") or ""
    name = Path(f).name if f else ""
    cands = [
        hp,
        f,
        f"samples/{f}" if f else "",
        f"samples/scale/{name}" if name else "",
        f"samples/smoke/{name}" if name else "",
        f"scale/train/wavs/{cid}.wav" if cid else "",
    ]
    for rel in cands:
        if not rel:
            continue
        p = DATA / rel
        if p.is_file():
            return rel.replace("\\", "/")
    return None


def _load_iv_cut_proposals() -> dict:
    fp = SCALE / "iv_cut_proposals.json"
    if not fp.exists():
        return {"proposals": [], "by_source_id": {}, "by_tag": {}}
    return json.loads(fp.read_text(encoding="utf-8"))


def build_indicvoices_catalog() -> dict:
    """IV + IVR clips from drafts.json for browser page."""
    fp = SCALE / "drafts.json"
    if not fp.exists():
        return {"error": "missing drafts.json", "clips": [], "tabs": []}
    doc = json.loads(fp.read_text(encoding="utf-8"))
    props_doc = _load_iv_cut_proposals()
    by_src = props_doc.get("by_source_id") or {}
    out = []
    tab_counts: dict[str, int] = {}
    for c in doc.get("clips") or []:
        src = (c.get("source") or "").strip()
        if src not in ("iv", "ivr"):
            continue
        wav = _resolve_wav_url(c)
        if not wav:
            continue
        t1 = (c.get("final_tag1") or c.get("human_tag1") or c.get("tag1") or "none").strip() or "none"
        t2 = (c.get("final_tag2") or c.get("human_tag2") or c.get("tag2") or "").strip()
        tags = [t1] + ([t2] if t2 else [])
        for t in tags:
            tab_counts[t] = tab_counts.get(t, 0) + 1
        text = c.get("text") or ""
        raw_events = [m.lower() for m in TAG_RE.findall(text)]
        cuts = by_src.get(c.get("id") or "", [])
        out.append(
            {
                "id": c.get("id"),
                "source": src,
                "dataset": c.get("dataset") or "",
                "native_label": c.get("native_label") or "",
                "duration_s": c.get("duration_s"),
                "tag1": t1,
                "tag2": t2,
                "tags": tags,
                "caption": c.get("caption") or "",
                "text": text,
                "raw_events": raw_events,
                "wav": wav,
                "confidence": c.get("confidence") or "",
                "notes": c.get("notes") or "",
                "status": c.get("status") or "",
                "cuts": cuts,
            }
        )
    # proposal-centric tabs for cut review (order from splitter)
    prop_by_tag = props_doc.get("by_tag") or {}
    tab_order = props_doc.get("tab_order") or [
        t for t, n in sorted(prop_by_tag.items(), key=lambda x: -x[1]) if n
    ]
    tabs = [{"id": "all", "label": "all", "n": len(out)}]
    cut_tab_ids = set()
    for t in tab_order:
        n = int(prop_by_tag.get(t) or 0)
        if n:
            tabs.append({"id": f"cuts_{t}", "label": f"cuts:{t}", "n": n})
            cut_tab_ids.add(f"cuts_{t}")
    seen = set(cut_tab_ids)
    for t in IV_TAB_ORDER:
        if t in tab_counts:
            tabs.append({"id": t, "label": t, "n": tab_counts[t]})
            seen.add(t)
    for t, n in sorted(tab_counts.items(), key=lambda x: (-x[1], x[0])):
        if t not in seen:
            tabs.append({"id": t, "label": t, "n": n})
    return {
        "n": len(out),
        "note": "IV/IVR drafts. cuts:* = catalog-only IV <> tags (v3). breathing≠inhaling. Accept=sanity not retag.",
        "tabs": tabs,
        "clips": out,
        "cut_meta": {
            "n_proposals": props_doc.get("n_proposals", 0),
            "by_tag": prop_by_tag,
            "emit_tags": props_doc.get("emit_tags") or props_doc.get("target_tags") or [],
            "gate_raw_tags": props_doc.get("gate_raw_tags") or [],
            "n_gated_clips": props_doc.get("n_gated_clips", 0),
        },
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DATA), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[lab] {self.address_string()} {fmt % args}")

    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/reviews":
            ids = sorted(
                p.stem
                for p in REVIEWS.glob("*.json")
                if p.name != "_meta.json" and not p.name.startswith("_")
            )
            self._json(200, {"ids": ids})
            return
        if path.startswith("/api/reviews/"):
            rid = path.rsplit("/", 1)[-1]
            if not ID_RE.match(rid):
                self._json(400, {"error": "bad id"})
                return
            fp = REVIEWS / f"{rid}.json"
            if not fp.exists():
                self._json(404, {"error": "not found"})
                return
            self._json(200, json.loads(fp.read_text(encoding="utf-8")))
            return
        if path == "/api/meta":
            self._json(200, json.loads((REVIEWS / "_meta.json").read_text(encoding="utf-8")))
            return
        if path == "/api/smoke/meta":
            self._json(200, json.loads((SMOKE / "meta.json").read_text(encoding="utf-8")))
            return
        if path == "/api/smoke/clips":
            self._json(200, json.loads((SMOKE / "clips.json").read_text(encoding="utf-8")))
            return
        if path == "/api/scale/drafts":
            fp = SCALE / "drafts.json"
            if not fp.exists():
                self._json(404, {"error": "run scripts/draft_tags.py --from-smoke first"})
                return
            self._json(200, json.loads(fp.read_text(encoding="utf-8")))
            return
        if path == "/api/scale/eval":
            fp = SCALE / "eval.json"
            if not fp.exists():
                self._json(404, {"error": "run scripts/tag_swap_eval.py --init first"})
                return
            self._json(200, json.loads(fp.read_text(encoding="utf-8")))
            return
        if path == "/api/scale/eval_orpheus":
            fp = SCALE / "eval_orpheus.json"
            if not fp.exists():
                self._json(404, {"error": "missing data/scale/eval_orpheus.json"})
                return
            self._json(200, json.loads(fp.read_text(encoding="utf-8")))
            return
        if path == "/api/scale/eval_mio_native":
            fp = SCALE / "eval_mio_native.json"
            if not fp.exists():
                self._json(404, {"error": "missing data/scale/eval_mio_native.json"})
                return
            self._json(200, json.loads(fp.read_text(encoding="utf-8")))
            return
        if path == "/api/scale/indicvoices":
            self._json(200, build_indicvoices_catalog())
            return
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        if path == "/api/smoke/clips":
            if not isinstance(data, dict) or "clips" not in data:
                self._json(400, {"error": "need {clips: [...]}"})
                return
            fp = SMOKE / "clips.json"
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/drafts":
            if not isinstance(data, dict) or "clips" not in data:
                self._json(400, {"error": "need {clips: [...]}"})
                return
            SCALE.mkdir(parents=True, exist_ok=True)
            fp = SCALE / "drafts.json"
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/eval":
            if not isinstance(data, dict) or "pairs" not in data:
                self._json(400, {"error": "need {pairs: [...]}"})
                return
            SCALE.mkdir(parents=True, exist_ok=True)
            fp = SCALE / "eval.json"
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/audit_v8":
            if not isinstance(data, dict) or "clips" not in data:
                self._json(400, {"error": "need {clips: [...]}"})
                return
            SCALE.mkdir(parents=True, exist_ok=True)
            fp = SCALE / "audit_v8_manifest.json"
            fp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/eval_orpheus":
            if not isinstance(data, dict) or "pairs" not in data:
                self._json(400, {"error": "need {pairs: [...]}"})
                return
            SCALE.mkdir(parents=True, exist_ok=True)
            fp = SCALE / "eval_orpheus.json"
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/eval_mio_native":
            if not isinstance(data, dict) or "pairs" not in data:
                self._json(400, {"error": "need {pairs: [...]}"})
                return
            SCALE.mkdir(parents=True, exist_ok=True)
            fp = SCALE / "eval_mio_native.json"
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})
            return

        if path == "/api/scale/indicvoices":
            # merge confidence + notes into drafts.json by clip id (no new schema)
            updates = data.get("updates") if isinstance(data, dict) else None
            cut_updates = data.get("cut_updates") if isinstance(data, dict) else None
            if updates is None and cut_updates is None:
                self._json(
                    400,
                    {
                        "error": "need {updates: [...]} and/or {cut_updates: [{cut_id, status?, caption_inline?, notes?}]}"
                    },
                )
                return
            n = 0
            n_cuts = 0
            paths = []
            if isinstance(updates, list):
                fp = SCALE / "drafts.json"
                if not fp.exists():
                    self._json(404, {"error": "missing drafts.json"})
                    return
                doc = json.loads(fp.read_text(encoding="utf-8"))
                by_id = {c.get("id"): c for c in doc.get("clips") or [] if c.get("id")}
                for u in updates:
                    if not isinstance(u, dict):
                        continue
                    cid = u.get("id")
                    if not cid or cid not in by_id:
                        continue
                    clip = by_id[cid]
                    if "confidence" in u:
                        conf = str(u.get("confidence") or "").strip().upper()
                        if conf in ("H", "M", "L"):
                            clip["confidence"] = conf
                    if "notes" in u:
                        clip["notes"] = str(u.get("notes") or "")
                    n += 1
                fp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                paths.append(str(fp.relative_to(ROOT)))
            if isinstance(cut_updates, list):
                cfp = SCALE / "iv_cut_proposals.json"
                if not cfp.exists():
                    self._json(404, {"error": "missing iv_cut_proposals.json — run iv_caption_split.py"})
                    return
                cdoc = json.loads(cfp.read_text(encoding="utf-8"))
                by_cut = {p.get("cut_id"): p for p in cdoc.get("proposals") or [] if p.get("cut_id")}
                for u in cut_updates:
                    if not isinstance(u, dict):
                        continue
                    cid = u.get("cut_id")
                    if not cid or cid not in by_cut:
                        continue
                    prop = by_cut[cid]
                    if "status" in u:
                        st = str(u.get("status") or "").strip()
                        if st in ("proposed", "accepted", "rejected"):
                            prop["status"] = st
                    if "caption_inline" in u:
                        prop["caption_inline"] = str(u.get("caption_inline") or "")
                    if "notes" in u:
                        prop["notes"] = str(u.get("notes") or "")
                    n_cuts += 1
                # rebuild by_source_id
                by_src: dict = {}
                for p in cdoc.get("proposals") or []:
                    sid = p.get("source_id") or ""
                    by_src.setdefault(sid, []).append(p)
                cdoc["by_source_id"] = by_src
                cfp.write_text(json.dumps(cdoc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                paths.append(str(cfp.relative_to(ROOT)))
            self._json(
                200,
                {"ok": True, "updated": n, "cut_updated": n_cuts, "path": paths},
            )
            return

        if not path.startswith("/api/reviews/"):
            self._json(404, {"error": "not found"})
            return
        rid = path.rsplit("/", 1)[-1]
        if not ID_RE.match(rid):
            self._json(400, {"error": "bad id"})
            return
        data["id"] = rid
        if "name" not in data:
            data["name"] = rid
        fp = REVIEWS / f"{rid}.json"
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._json(200, {"ok": True, "path": str(fp.relative_to(ROOT))})


def main():
    REVIEWS.mkdir(parents=True, exist_ok=True)
    SMOKE.mkdir(parents=True, exist_ok=True)
    SCALE.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"lab → http://{HOST}:{PORT}/dataset-review.html")
    print(f"smoke → http://{HOST}:{PORT}/smoke.html")
    print(f"review queue → http://{HOST}:{PORT}/review-queue.html")
    print(f"also → http://{HOST}:{PORT}/datasets.html")
    print(f"mio tag-swap → http://{HOST}:{PORT}/tag-swap.html")
    print(f"mio base vs lora → http://{HOST}:{PORT}/base-vs-lora.html")
    print(f"orpheus tag-swap → http://{HOST}:{PORT}/tag-swap-orpheus.html")
    print(f"orpheus base vs lora → http://{HOST}:{PORT}/base-vs-lora-orpheus.html")
    print(f"mio native ceiling → http://{HOST}:{PORT}/tag-swap-mio-native.html")
    print(f"indicvoices browse → http://{HOST}:{PORT}/indicvoices.html")
    print(f"reviews → {REVIEWS}")
    print(f"smoke data → {SMOKE}")
    print(f"scale drafts → {SCALE}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
