#!/usr/bin/env python3
"""ASR bake-off on HiACC, scored against the corpus's own human transcripts.

HiACC ships human transcriptions, so it is a free labelled validation set. This
measures whether Hinglish ASR is viable *before* pointing it at YouTube, where
there is no ground truth.

  .venv/bin/python scripts/asr_hiacc_bakeoff.py --models whisper-1,gpt-4o-transcribe
  .venv/bin/python scripts/asr_hiacc_bakeoff.py --models gemini-2.5-flash
  .venv/bin/python scripts/asr_hiacc_bakeoff.py --models whisper-1 --no-lang-hint

Metrics
  wer / cer            standard, after punctuation + case normalisation
  deva_frac delta      script drift: reference Devanagari share vs hypothesis
  latin_kept           share of reference English word types still in Latin in
                       the hypothesis (catches transliteration of English into
                       Devanagari, which WER alone under-penalises)

Output: data/scale/asr_hiacc_bakeoff.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    ("adult", ROOT / "hiacc-new" / "code_switched_labels.json"),
    ("child", ROOT / "hiacc-new" / "hiacc-ch" / "code_switched_labels.json"),
]
WAV_DIRS = [ROOT, ROOT / "hiacc-new", ROOT / "hiacc-new" / "hiacc-ch"]
OUT = ROOT / "data" / "scale" / "asr_hiacc_bakeoff.json"

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

GEMINI_PROMPT = (
    "Transcribe this Hindi/English code-switched audio verbatim. "
    "Write Hindi words in Devanagari and English words in Latin script, "
    "exactly as spoken. Do not translate. Do not add punctuation that was not "
    "spoken. Output only the transcript."
)


# ---------------------------------------------------------------- env / data


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def openai_key() -> str:
    for k in ("OPENAI_API", "OPENAI_API_KEY", "OPENAI_KEY"):
        if os.environ.get(k):
            return os.environ[k].strip()
    return ""


def gemini_key() -> str:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(k):
            return os.environ[k].strip()
    return ""


def build_set() -> list[dict]:
    idx: dict[str, dict] = {}
    for src, path in LABELS:
        for r in json.loads(path.read_text(encoding="utf-8")):
            key = r.get("audio") or r.get("audio_filepath") or ""
            idx[os.path.basename(key)] = {"set": src, **r}

    clips = []
    for d in WAV_DIRS:
        for p in sorted(d.glob("*.wav")):
            name = p.name
            hit = idx.get(name) or idx.get(name.replace(" (1)", ""))
            if not hit:
                continue
            with wave.open(str(p)) as w:
                dur = w.getnframes() / w.getframerate()
            clips.append(
                {
                    "id": p.stem,
                    "wav": str(p.relative_to(ROOT)),
                    "set": hit["set"],
                    "cs_label": hit.get("label"),
                    "ref": (hit.get("transcription") or "").strip(),
                    "duration_s": round(dur, 2),
                }
            )
    return clips


# ---------------------------------------------------------------- providers


def asr_openai(wav: Path, model: str, key: str, lang_hint: bool) -> str:
    boundary = "----rvbakeoff"
    fields = {b"model": model.encode(), b"response_format": b"text"}
    if lang_hint:
        fields[b"language"] = b"hi"
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k.decode()}"\r\n\r\n'.encode()
        body += v + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n" + wav.read_bytes() + b"\r\n"
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
    with urllib.request.urlopen(req, timeout=240) as r:
        return r.read().decode("utf-8").strip()


def asr_gemini(wav: Path, model: str, key: str, lang_hint: bool) -> str:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": GEMINI_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(wav.read_bytes()).decode(),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read().decode("utf-8"))
    parts = d["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def transcribe(wav: Path, model: str, lang_hint: bool) -> str:
    if model.startswith("gemini"):
        key = gemini_key()
        if not key:
            raise SystemExit("no GEMINI_API_KEY / GOOGLE_API_KEY in .env")
        return asr_gemini(wav, model, key, lang_hint)
    key = openai_key()
    if not key:
        raise SystemExit("no OPENAI_API_KEY in .env")
    return asr_openai(wav, model, key, lang_hint)


# ---------------------------------------------------------------- scoring


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[^\w\sऀ-ॿ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def deva_frac(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if DEVA.search(w)) / len(words)


def latin_types(words: list[str]) -> set[str]:
    return {w for w in words if LATIN.search(w) and not DEVA.search(w)}


def score(ref: str, hyp: str) -> dict:
    rw, hw = norm(ref).split(), norm(hyp).split()
    rc, hc = list(norm(ref).replace(" ", "")), list(norm(hyp).replace(" ", ""))
    rl = latin_types(rw)
    kept = len(rl & latin_types(hw)) / len(rl) if rl else None
    return {
        "n_ref_words": len(rw),
        "wer": round(edit_distance(rw, hw) / len(rw), 4) if rw else None,
        "cer": round(edit_distance(rc, hc) / len(rc), 4) if rc else None,
        "ref_deva_frac": round(deva_frac(rw), 4),
        "hyp_deva_frac": round(deva_frac(hw), 4),
        "n_ref_latin_types": len(rl),
        "latin_kept": round(kept, 4) if kept is not None else None,
        "empty_hyp": not hw,
    }


def agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("wer") is not None]
    if not ok:
        return {"n": 0}
    tw = sum(r["n_ref_words"] for r in ok)
    # micro: weight by reference length, not by clip
    micro_wer = sum(r["wer"] * r["n_ref_words"] for r in ok) / tw
    lat = [r for r in ok if r["latin_kept"] is not None]
    return {
        "n": len(ok),
        "wer_micro": round(micro_wer, 4),
        "wer_macro": round(sum(r["wer"] for r in ok) / len(ok), 4),
        "cer_macro": round(sum(r["cer"] for r in ok) / len(ok), 4),
        "ref_deva_frac": round(sum(r["ref_deva_frac"] for r in ok) / len(ok), 4),
        "hyp_deva_frac": round(sum(r["hyp_deva_frac"] for r in ok) / len(ok), 4),
        "latin_kept": round(sum(r["latin_kept"] for r in lat) / len(lat), 4) if lat else None,
        "n_empty": sum(1 for r in rows if r.get("empty_hyp")),
        "n_error": sum(1 for r in rows if r.get("error")),
    }


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="whisper-1")
    ap.add_argument("--no-lang-hint", action="store_true",
                    help="omit language=hi; forcing hi can transliterate English into Devanagari")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_env()
    clips = build_set()
    if args.limit:
        clips = clips[: args.limit]
    total_s = sum(c["duration_s"] for c in clips)
    print(f"clips={len(clips)}  audio={total_s:.0f}s ({total_s/60:.1f} min)")

    prev = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    results = prev.get("results", {})
    lang_hint = not args.no_lang_hint

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        arm = model if lang_hint else f"{model}[no-lang-hint]"
        print(f"\n=== {arm} ===", flush=True)
        rows = []
        for i, c in enumerate(clips, 1):
            row = {k: c[k] for k in ("id", "set", "cs_label", "duration_s")}
            try:
                t0 = time.time()
                hyp = transcribe(ROOT / c["wav"], model, lang_hint)
                row["rtf"] = round((time.time() - t0) / c["duration_s"], 2)
                row["hyp"] = hyp
                row.update(score(c["ref"], hyp))
            except urllib.error.HTTPError as e:
                row["error"] = f"HTTP {e.code}: {e.read().decode()[:200]}"
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            mark = row.get("error") or f"wer={row.get('wer')}"
            print(f"  [{i}/{len(clips)}] {row['id']:10} {mark}", flush=True)

        a = agg(rows)
        results[arm] = {"model": model, "lang_hint": lang_hint, "agg": a, "rows": rows}
        print(f"  -> {a}", flush=True)

        args.out.write_text(
            json.dumps(
                {
                    "dataset": "HiACC (adult+child), human transcripts as reference",
                    "n_clips": len(clips),
                    "audio_s": round(total_s, 1),
                    "gemini_prompt_note": "Gemini arms get an explicit script instruction; "
                    "Whisper arms get none. Not a like-for-like prompt, but it does "
                    "reflect how each would actually be used.",
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\nwrote {args.out}")
    print(f"\n{'arm':34} {'WERµ':>7} {'CER':>7} {'devaΔ':>7} {'latin_kept':>11} {'err':>4}")
    for arm, r in results.items():
        a = r["agg"]
        if not a.get("n"):
            continue
        delta = a["hyp_deva_frac"] - a["ref_deva_frac"]
        lk = a["latin_kept"]
        print(
            f"{arm:34} {a['wer_micro']:>7.3f} {a['cer_macro']:>7.3f} "
            f"{delta:>+7.3f} {(f'{lk:.3f}' if lk is not None else '-'):>11} "
            f"{a['n_error']:>4}"
        )


if __name__ == "__main__":
    main()
