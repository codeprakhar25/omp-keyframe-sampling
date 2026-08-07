#!/usr/bin/env python3
"""Optional multimodal audio draft for tags.

Uses OpenAI audio-capable chat if OPENAI_API / OPENAI_API_KEY / OPENAI_KEY in .env.
Gemini hook if GEMINI_API_KEY / GOOGLE_API_KEY present.

  .venv/bin/python scripts/audio_draft.py --wav path.wav --text "..."

Returns JSON: {tag1, tag2, confidence, notes, provider}
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = [
    "none",
    "angry",
    "sad",
    "excited",
    "surprise",
    "fear",
    "disgust",
    "pause",
    "inhaling",
    "thinking",
    "sigh",
    "whispering",
    "laugh",
]


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def openai_key() -> str:
    return (
        os.environ.get("OPENAI_API")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_KEY")
        or ""
    ).strip()


def gemini_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


def available_providers() -> list[str]:
    out = []
    if openai_key():
        out.append("openai")
    if gemini_key():
        out.append("gemini")
    return out


def _parse_json_blob(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


PROMPT = """You label expressive speech for Hindi/Hinglish TTS research.
Pick at most 2 tags from this freeze ONLY:
{tags}

Definitions:
- pause = near-silence gap, little/no voice
- thinking = filled hesitation (umm/uhh/hmm)
- inhaling = audible in-breath; sigh = audible sigh out-breath
- whispering = clearly whispered speech
- emotions: angry/sad/excited/surprise/fear/disgust
- none = flat / no clear expressive tag
- laugh = audible laughter (rare; only if clearly heard — short clips often have other tags instead)

Human calibration (gold labels from a finished 55-clip smoke sheet — match this ear style):
{fewshot}

Trust EAR over transcript <event> markers (markers can be wrong/inaudible).
inhaling vs sigh are close; pick the clearer one; conf M if unsure.

Transcript (context only):
{text}

Return ONLY JSON:
{{"tag1":"...","tag2":"","confidence":"H|M|L","notes":"short"}}
"""


def build_fewshot(max_examples: int = 12) -> str:
    """Load human smoke tags as text few-shot for calibration."""
    smoke_path = ROOT / "data" / "smoke" / "clips.json"
    if not smoke_path.exists():
        return "(no smoke sheet yet)"
    clips = json.loads(smoke_path.read_text(encoding="utf-8")).get("clips") or []
    lines = []
    # diversify by tag1
    seen = set()
    for c in clips:
        t1 = (c.get("pass2_tag1") or c.get("tag1") or "").strip()
        if not t1 or t1 in seen:
            continue
        if c.get("bonus") and t1 == "laugh":
            continue
        seen.add(t1)
        t2 = (c.get("tag2") or "").strip()
        conf = (c.get("confidence") or "").strip() or "?"
        native = (c.get("native_label") or "")[:40]
        lines.append(
            f"- human: tag1={t1}"
            + (f" tag2={t2}" if t2 else "")
            + f" conf={conf} | source={c.get('source')} native≈{native}"
        )
        if len(lines) >= max_examples:
            break
    return "\n".join(lines) if lines else "(empty)"


def draft_openai(wav_path: Path, text: str, model: str = "") -> dict:
    key = openai_key()
    if not key:
        raise RuntimeError("no OPENAI_API")

    # Account-available audio chat models (prefer mini for cost)
    candidates = [m for m in [model, "gpt-audio-mini", "gpt-audio", "gpt-audio-1.5"] if m]

    mime = mimetypes.guess_type(str(wav_path))[0] or "audio/wav"
    b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    last_err = ""
    for model in candidates:
        body = {
            "model": model,
            "modalities": ["text"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(
                                tags=" · ".join(TAGS),
                                text=(text or "")[:800],
                                fewshot=build_fewshot(),
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64,
                                "format": "wav" if "wav" in (mime or "") else "mp3",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e.read().decode("utf-8", errors="replace")[:400]
            continue

        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        parsed = _parse_json_blob(str(content))
        return _normalize(parsed, provider="openai", model=model)

    raise RuntimeError(f"openai audio failed: {last_err}")


def draft_gemini(wav_path: Path, text: str, model: str = "gemini-2.0-flash") -> dict:
    """Gemini generateContent with inline audio bytes."""
    key = gemini_key()
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")

    mime = mimetypes.guess_type(str(wav_path))[0] or "audio/wav"
    b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": PROMPT.format(
                            tags=" · ".join(TAGS),
                            text=(text or "")[:800],
                            fewshot=build_fewshot(),
                        )
                    },
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"gemini HTTP {e.code}: {err}") from e

    parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    content = "".join(p.get("text", "") for p in parts)
    parsed = _parse_json_blob(content)
    return _normalize(parsed, provider="gemini", model=model)


def _normalize(parsed: dict, provider: str, model: str) -> dict:
    t1 = str(parsed.get("tag1") or "none").strip()
    t2 = str(parsed.get("tag2") or "").strip()
    if t1 not in TAGS:
        t1 = "none"
    if t2 and t2 not in TAGS:
        t2 = ""
    conf = str(parsed.get("confidence") or "M").strip().upper()
    if conf not in ("H", "M", "L"):
        conf = "M"
    return {
        "tag1": t1,
        "tag2": t2,
        "confidence": conf,
        "notes": str(parsed.get("notes") or "")[:200],
        "provider": provider,
        "model": model,
    }


def draft_audio(wav_path: Path, text: str = "", provider: str = "auto") -> dict:
    load_env()
    provs = available_providers()
    if provider == "auto":
        if "openai" in provs:
            provider = "openai"
        elif "gemini" in provs:
            provider = "gemini"
        else:
            raise RuntimeError(
                "no audio API key — set OPENAI_API or OPENAI_API_KEY or GEMINI_API_KEY in .env"
            )
    if provider == "openai":
        return draft_openai(wav_path, text)
    if provider == "gemini":
        return draft_gemini(wav_path, text)
    raise RuntimeError(f"unknown provider {provider}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multimodal audio tag draft")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--provider", default="auto", choices=["auto", "openai", "gemini"])
    args = ap.parse_args()
    load_env()
    print("providers:", available_providers() or ["(none)"])
    out = draft_audio(args.wav, args.text, provider=args.provider)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
