#!/usr/bin/env python3
"""ASR-backfill empty draft texts (OpenAI). Fast path for leftovers.

  .venv/bin/python scripts/asr_backfill.py
  .venv/bin/python scripts/asr_backfill.py --ids sc130,sc131
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples" / "scale"
DRAFTS = ROOT / "data" / "scale" / "drafts.json"


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key() -> str:
    return (
        os.environ.get("OPENAI_API")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_KEY")
        or ""
    ).strip()


def resolve_wav(file_field: str) -> Path | None:
    name = Path(file_field).name
    for p in (SAMPLES / name, ROOT / "data" / "samples" / file_field):
        if p.exists():
            return p
    return None


def transcribe(wav: Path, key: str, model: str) -> str:
    """multipart whisper / gpt-4o-mini-transcribe."""
    boundary = "----rvboundary7"
    data = wav.read_bytes()
    body = b""
    fields = {
        b"model": model.encode(),
        b"language": b"hi",
        b"response_format": b"text",
    }
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k.decode()}"\r\n\r\n'.encode()
        body += v + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read().decode("utf-8").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma ids; default=all empty")
    ap.add_argument("--model", default="gpt-4o-mini-transcribe")
    args = ap.parse_args()
    load_env()
    key = api_key()
    if not key:
        raise SystemExit("no OPENAI_API / OPENAI_API_KEY")

    drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))
    want = {x.strip() for x in args.ids.split(",") if x.strip()}
    n = 0
    for c in drafts["clips"]:
        if want and c.get("id") not in want:
            continue
        if (c.get("text") or "").strip():
            continue
        p = resolve_wav(c["file"])
        if not p:
            print("miss wav", c["id"], flush=True)
            continue
        print(f"ASR {c['id']} {p.name} …", flush=True)
        try:
            text = transcribe(p, key, args.model)
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            continue
        c["text"] = text
        c["text_source"] = f"asr:{args.model}"
        side = Path(str(p) + ".json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        meta["text"] = text
        meta["text_source"] = c["text_source"]
        side.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n += 1
        print(f"  → {text[:80]}", flush=True)
        DRAFTS.write_text(json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    still = sum(1 for c in drafts["clips"] if not (c.get("text") or "").strip())
    print(f"DONE asr_filled={n} still_empty={still}", flush=True)


if __name__ == "__main__":
    main()
