#!/usr/bin/env python3
"""GPT frontier MCQA on banked picks — LDDR F.3 / lmms-eval protocol.

Templates match LDDR paper Appendix F.3 (same as lmms-eval defaults where
available). Used consistently across selectors; only frames change.

  LVBench:
    {question(+options)}
    Answer the question with the option letter

  Video-MME:
    Select the best answer ... (A, B, C, or D) ...
    {question}
    {options}
    Answer with the option's letter from the given choices directly.

  LongVideoBench:
    Select the best answer ... (A, B, C, D or E) ...
    {question}
    A. {opt0} ...
    Answer with the option's letter from the given choices directly.

Defaults:
  - NO system prompt
  - NO [t=Xs] (opt-in --timestamps)
  - parse = harness.mcq_extract.extract_mcq_answer
  - do NOT use qwen3_vl Video-MME override (that was open-VLM path only)

Decode: gpt-5* → --effort low|minimal (no temp=0). Classic → --effort none.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image

# Ensure repo root (slm-lab/) is on path for harness.mcq_extract
_SLM_ROOT = Path(__file__).resolve().parents[1]
if str(_SLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLM_ROOT))

from harness.mcq_extract import extract_mcq_answer  # noqa: E402

# LDDR F.3 / lmms-eval default (NOT qwen3_vl override).
POST_PROMPTS = {
    "lvbench": "Answer the question with the option letter",
    "videomme": "Answer with the option's letter from the given choices directly.",
    "longvideobench": "Answer with the option's letter from the given choices directly.",
}

PRE_PROMPTS = {
    "lvbench": "",
    "videomme": (
        "Select the best answer to the following multiple-choice question based "
        "on the video and the subtitles. Respond with only the letter (A, B, C, or D) "
        "of the correct option."
    ),
    "longvideobench": (
        "Select the best answer to the following multiple-choice question based "
        "on the video and the subtitles. Respond with only the letter (A, B, C, D or E) "
        "of the correct option."
    ),
}

# Strip glued post / qwen3_vl tails / LDDR preambles if re-formatting.
_KNOWN_POST_RE = re.compile(
    r"\n*(Answer the question with the option letter\.?"
    r"|Answer with the option'?s? letter from the given choices directly\.?"
    r"|Answer with the option letter only\.?"
    r"|Answer with the single option letter\.?"
    r"|Answer the question with A, B, C, or D\.?"
    r"|The best answer is:?"
    r"|Only give the best option\.?"
    r"|Best option:?)\s*$",
    re.IGNORECASE,
)
_LETTERS = "ABCDE"
_KNOWN_PRE_RE = re.compile(
    r"^Select the best answer to the following multiple-choice question based "
    r"on the video and the subtitles\.[^\n]*\n+",
    re.IGNORECASE,
)
_QWEN3VL_PREFIX_RE = re.compile(r"^Question:\s*", re.IGNORECASE)
_OPTIONS_HEADER_RE = re.compile(r"\nOptions:\n", re.IGNORECASE)


def _load_env_file(path: str) -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


# Picks landing within ~0.5s of a video's true duration make ffmpeg's -ss
# seek land past the last decodable frame → empty output, exit 0 → PIL
# UnidentifiedImageError. Retry at progressively earlier timestamps instead
# of needing per-video duration lookups.
_EXTRACT_BACKOFFS = (0.0, 0.15, 0.4, 0.8, 1.5, 3.0)

# Floor matches gen_resprop_picks FLOOR (~min_pixels / baseline frame).
# Applied AFTER max_side so GPT "full" = max_side-capped; D-arms shrink from that.
_MIN_PIXELS_FLOOR = 200_704


def normalize_pick_entries(raw: Any) -> list[tuple[float, float]]:
    """Normalize picks to ``[(sec, frac), ...]``.

    Accepts:
      - ``[17.0, 76.0, ...]``                         → frac=1.0 each (full-res)
      - ``[[17.0, 0.5], [76.0, 1.0], ...]``            → resprop/restier
      - ``[{"sec":17,"frac":0.5}, ...]``               → tolerant
      - wrapper ``{"secs":[...]}`` / ``{"picks":...}`` → unwrap once

    Refuses empty / malformed so a silent full-res bug can't hide.
    """
    if isinstance(raw, dict):
        if "picks" in raw:
            raw = raw["picks"]
        elif "secs" in raw:
            # secs-only dict → full-res
            raw = raw["secs"]
        else:
            raise ValueError(f"unsupported picks dict keys={list(raw)[:8]}")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"empty/non-list picks: {type(raw)}")

    out: list[tuple[float, float]] = []
    for i, e in enumerate(raw):
        if isinstance(e, (int, float)):
            out.append((float(e), 1.0))
            continue
        if isinstance(e, dict):
            sec = e.get("sec", e.get("t", e.get("time")))
            frac = e.get("frac", e.get("scale", 1.0))
            if sec is None:
                raise ValueError(f"pick[{i}] dict missing sec: {e}")
            out.append((float(sec), float(frac)))
            continue
        if isinstance(e, (list, tuple)) and len(e) >= 1:
            sec = float(e[0])
            frac = float(e[1]) if len(e) >= 2 else 1.0
            out.append((sec, frac))
            continue
        raise ValueError(f"pick[{i}] bad entry type={type(e)} val={e!r}")

    for i, (sec, frac) in enumerate(out):
        if sec < 0:
            raise ValueError(f"pick[{i}] negative sec={sec}")
        if not (0.0 < frac <= 1.0 + 1e-6):
            raise ValueError(f"pick[{i}] frac out of (0,1]: {frac}")
        if frac > 1.0:
            out[i] = (sec, 1.0)
    return out


_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)


def apply_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """GPT full-res protocol: cap longest side (thumbnail, keeps aspect)."""
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), _RESAMPLE)
    return img


def apply_frac_resize(img: Image.Image, frac: float) -> Image.Image:
    """Shrink pixel area by ``frac`` relative to current size (post max_side).

    Mirrors restier/resprop intent (tokens ∝ pixels) without qwen_vl_utils.
    ``frac >= 0.999`` → no-op. Floor at ``_MIN_PIXELS_FLOOR`` (= restier MINP)
    so tiny fracs don't collapse to unreadable stamps. Note: with
    ``max_side=768``, a 768×432 cap is already ~332k px, so frac≲0.60 may
    hit the floor — pixel_ratio will exceed pick mean_frac; still valid
    compression vs full, just disclose in protocol.
    """
    if frac >= 0.999:
        return img
    w, h = img.size
    cur = w * h
    target = max(_MIN_PIXELS_FLOOR, int(frac * cur))
    if target >= cur:
        return img
    scale = (target / cur) ** 0.5
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), _RESAMPLE)


def encode_jpeg_b64(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_frame_image(video: Path, t: float) -> Image.Image:
    """ffmpeg seek → RGB PIL. Raises on failure."""
    last_err: Exception = RuntimeError("extract_frame_image: no attempts made")
    for backoff in _EXTRACT_BACKOFFS:
        seek_t = max(0.0, t - backoff)
        out = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                out = tmp.name
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{seek_t:.3f}", "-i", str(video),
                "-frames:v", "1", "-q:v", "2", "-y", out,
            ]
            subprocess.run(cmd, check=True, timeout=60)
            return Image.open(out).convert("RGB")
        except Exception as e:
            last_err = e
        finally:
            if out:
                try:
                    os.unlink(out)
                except OSError:
                    pass
    raise last_err


def extract_frame(
    video: Path,
    t: float,
    max_side: int,
    quality: int,
    *,
    frac: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    """ffmpeg → max_side → optional frac shrink → JPEG b64 + size audit."""
    img = extract_frame_image(video, t)
    native = img.size
    img = apply_max_side(img, max_side)
    after_cap = img.size
    img = apply_frac_resize(img, frac)
    final = img.size
    b64 = encode_jpeg_b64(img, quality)
    meta = {
        "sec": float(t),
        "frac": float(frac),
        "native_wh": list(native),
        "capped_wh": list(after_cap),
        "final_wh": list(final),
        "final_pixels": int(final[0] * final[1]),
        "capped_pixels": int(after_cap[0] * after_cap[1]),
    }
    return b64, meta


def extract_frame_b64(video: Path, t: float, max_side: int, quality: int) -> str:
    """Back-compat: full-res (frac=1) JPEG only."""
    b64, _ = extract_frame(video, t, max_side, quality, frac=1.0)
    return b64


def picks_mean_frac(entries: list[tuple[float, float]]) -> float:
    if not entries:
        return 1.0
    return sum(f for _, f in entries) / len(entries)


# LVBench: "(A) …" · LongVideoBench/VMM: "A. …" / "A) …"
_OPTION_LINE_RE = re.compile(r"^(?:\([A-E]\)|[A-E][.)])\s", re.MULTILINE)
_OPTION_LETTER_RE = re.compile(
    r"^(?:\(([A-E])\)|([A-E])[.)])\s", re.MULTILINE
)


def normalize_option_lines(options: list[Any], *, style: str = ".") -> list[str]:
    """Turn raw choice strings into ``A. …`` / ``A) …`` lines.

    Accepts either bare texts (``['cat','dog']``) or already-lettered
    lines (``['A. cat', 'B) dog']``). Never invents letters past E.
    """
    out: list[str] = []
    for i, opt in enumerate(options):
        if i >= len(_LETTERS):
            break
        s = str(opt).strip()
        if not s:
            continue
        if _OPTION_LINE_RE.match(s):
            out.append(s)
        else:
            out.append(f"{_LETTERS[i]}{style} {s}")
    return out


def item_option_lines(item: dict, *, bench: str) -> list[str]:
    """Pull MCQ choices from whatever field the bench manifest uses.

    - Video-MME: ``options`` (often already ``A. …``)
    - LongVideoBench: ``candidates`` (bare texts) — question may ALSO
      already include fused ``Options:\\nA) …``; caller still passes these
      as fallback if the stem was stripped bare.
    - LVBench: usually inline in ``question``; ``options`` rare.
    """
    raw = item.get("options")
    if raw is None:
        raw = item.get("candidates")
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    # LVB fused questions use A); LDDR F.3 LVB template uses A.
    style = "." if bench == "longvideobench" else "."
    return normalize_option_lines(raw, style=style)


def _letters_in_text(text: str) -> list[str]:
    found = []
    for m in _OPTION_LETTER_RE.finditer(text or ""):
        ch = m.group(1) or m.group(2)
        if ch and ch not in found:
            found.append(ch)
    return found


def valid_letters_for(item: dict, bench: str) -> str:
    """ABCD or ABCDE from option count / bench default."""
    lines = item_option_lines(item, bench=bench)
    if lines:
        return _LETTERS[: len(lines)]
    found = _letters_in_text(item.get("question") or "")
    if found:
        return "".join(ch for ch in _LETTERS if ch in found)
    return "ABCDE" if bench == "longvideobench" else "ABCD"


def format_question(question: str, bench: str, options: Optional[list[str]] = None) -> str:
    """Build LDDR F.3 / lmms-eval text for bench.

    ``question`` from LongVideoBench/LVBench manifests often already
    includes options inline (LVBench ``(A)…``, LVB ``A) …`` after an
    ``Options:`` header). Video-MME keeps ``question`` as bare stem —
    ``options`` must be passed and gets appended only if the stem has no
    option lines (no double-insert).
    """
    stem = question.rstrip()
    stem = _KNOWN_POST_RE.sub("", stem).rstrip()
    stem = _KNOWN_PRE_RE.sub("", stem).lstrip()
    stem = _QWEN3VL_PREFIX_RE.sub("", stem)
    stem = _OPTIONS_HEADER_RE.sub("\n", stem)
    if options and not _OPTION_LINE_RE.search(stem):
        # allow bare texts or preformatted lines
        if options and not _OPTION_LINE_RE.match(str(options[0]).strip()):
            opt_lines = normalize_option_lines(options, style=".")
        else:
            opt_lines = [str(o).rstrip() for o in options]
        stem = stem.rstrip() + "\n" + "\n".join(opt_lines)
    pre = PRE_PROMPTS[bench]
    post = POST_PROMPTS[bench]
    if pre:
        return f"{pre}\n{stem}\n{post}"
    return f"{stem}\n{post}"


def parse_letter(text: str, valid: str = "ABCD") -> str:
    letter = extract_mcq_answer(text or "", choices=list(valid))
    return letter if letter else "?"


def prompt_has_options(text: str) -> bool:
    """True if ≥2 lettered choice lines present (A/B minimum).

    Accepts ``(A) …`` (LVBench), ``A. …`` / ``A) …`` (LVB / VMM).
    """
    letters = set(_letters_in_text(text))
    return "A" in letters and "B" in letters


def question_text_for_item(item: dict) -> str:
    """Choose stem vs full question without dropping MCQ options.

    LVBench ships ``question_stem`` (bare) + options fused only into
    ``question`` as ``(A)…``. Preferring stem blindly → chance-level.
    Use stem only when options/candidates exist to re-attach, or when
    the stem already carries lettered lines.
    """
    stem = (item.get("question_stem") or "").strip()
    full = (item.get("question") or "").strip()
    if not stem:
        return full
    if prompt_has_options(stem):
        return stem
    if item.get("options") or item.get("candidates"):
        return stem  # caller passes option lines into format_question
    # stem bare, options only in full question
    return full or stem


def normalize_gold(gold: Any) -> Any:
    """Prefer single letter; tolerate ``C) text`` / ``C. text``."""
    if gold is None:
        return None
    s = str(gold).strip()
    if len(s) == 1 and s.upper() in _LETTERS:
        return s.upper()
    m = re.match(r"^([A-E])[.)]\s", s, re.I)
    if m:
        return m.group(1).upper()
    if s and s[0].upper() in _LETTERS and (len(s) == 1 or not s[1].isalnum()):
        return s[0].upper()
    return gold


def build_user_content(
    question: str,
    frames_b64: list[tuple[float, str]],
    *,
    bench: str,
    timestamps: bool,
    options: Optional[list[str]] = None,
) -> list[dict]:
    q = format_question(question, bench, options=options)
    parts: list[dict] = [{"type": "text", "text": q}]
    for t, b64 in frames_b64:
        if timestamps:
            parts.append({"type": "text", "text": f"[t={t:.1f}s]"})
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return parts


def ask(
    client: Any,
    model: str,
    content: list[dict],
    *,
    effort: str,
    max_tokens: int,
) -> str:
    # No system prompt — matches lmms-eval open-VLM path.
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_tokens,
    )
    if effort == "none":
        kwargs["temperature"] = 0
    else:
        kwargs["reasoning_effort"] = effort
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    return (msg.content or "") if msg is not None else ""


def resolve_video(item: dict, video_root: Path) -> Path:
    vf = item.get("video_file")
    if vf:
        p = video_root / vf
        if p.is_file():
            return p
    mp = item.get("media_path")
    if mp and Path(mp).is_file():
        return Path(mp)
    if vf:
        return video_root / Path(vf).name
    raise FileNotFoundError(f"no video for {item.get('id')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--picks", required=True,
        help="qid -> [sec,...] OR [[sec,frac],...] (resprop/restier)",
    )
    ap.add_argument("--video-root", default="/workspace/hf/lvbench")
    ap.add_argument("--bench", default="lvbench", choices=sorted(POST_PROMPTS))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qids", default=None, help="optional JSON list of qids")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument(
        "--effort", default="low",
        choices=["none", "minimal", "low", "medium", "high"],
        help="gpt-5*: low/minimal (no temp=0). none=temperature=0 for classic models.",
    )
    ap.add_argument(
        "--timestamps", action="store_true",
        help="inject [t=Xs] before each frame (OFF by default for lmms-eval parity)",
    )
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--max-tokens", type=int, default=1024,
        help="max_completion_tokens; reasoning models need headroom",
    )
    ap.add_argument("--env-file", default="/workspace/slm-lab/.env")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="per-qid checkpoint jsonl")
    ap.add_argument(
        "--preflight-only", action="store_true",
        help="validate picks + prompt options for queue; no ffmpeg/API",
    )
    ap.add_argument(
        "--frames-only", action="store_true",
        help="extract+resize frames for queue; print pixel audit; no API",
    )
    ap.add_argument(
        "--require-compression", action="store_true",
        help="refuse if mean pick frac >= 0.95 (guards silent full-res on D-arms)",
    )
    args = ap.parse_args()

    _load_env_file(args.env_file)
    need_api = not args.preflight_only and not args.frames_only
    if need_api and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing (pass via --env-file or env)")

    client = None
    if need_api:
        from openai import OpenAI

        client = OpenAI()

    video_root = Path(args.video_root)
    mani = {it["id"]: it for it in json.load(open(args.manifest))}
    picks_raw = json.load(open(args.picks))

    # Normalize all picks up front — fail loud before spending API $.
    picks: dict[str, list[tuple[float, float]]] = {}
    bad_picks: list[str] = []
    for qid, raw in picks_raw.items():
        try:
            picks[qid] = normalize_pick_entries(raw)
        except Exception as e:
            bad_picks.append(f"{qid}: {e}")
    if bad_picks:
        sys.exit(
            f"PICKS_INVALID n={len(bad_picks)} eg: " + "; ".join(bad_picks[:5])
        )

    mean_frac_all = (
        sum(picks_mean_frac(v) for v in picks.values()) / len(picks) if picks else 1.0
    )
    n_compressed = sum(
        1 for v in picks.values() if any(f < 0.999 for _, f in v)
    )
    print(
        f"picks loaded n={len(picks)} mean_frac={mean_frac_all:.4f} "
        f"qids_with_any_frac<1={n_compressed} max_side={args.max_side}",
        flush=True,
    )
    if args.require_compression and mean_frac_all >= 0.95:
        sys.exit(
            f"REQUIRE_COMPRESSION failed: mean_frac={mean_frac_all:.4f} >= 0.95 "
            f"(this looks like a full-res picks file)"
        )

    if args.qids:
        qids = json.load(open(args.qids))
        if isinstance(qids, dict):
            qids = qids.get("qids") or qids.get("ids") or list(qids)
    else:
        import random

        ids = [qid for qid in mani if qid in picks]
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        qids = ids[: args.n]

    missing = [q for q in qids if q not in picks]
    if missing:
        sys.exit(f"queue qids missing from picks: {missing[:5]} n={len(missing)}")

    # Prompt preflight on full queue (cheap, no IO).
    prompt_bad: list[str] = []
    for qid in qids:
        item = mani[qid]
        opt_lines = item_option_lines(item, bench=args.bench)
        q_text = question_text_for_item(item)
        text = format_question(q_text, args.bench, options=opt_lines or None)
        if not prompt_has_options(text):
            prompt_bad.append(qid)
    if prompt_bad:
        sys.exit(
            f"PROMPT_MISSING_OPTIONS n={len(prompt_bad)} eg={prompt_bad[:5]} "
            f"(refusing chance-level run)"
        )
    print(f"preflight prompts_ok n={len(qids)}", flush=True)

    if args.preflight_only:
        # sample one formatted prompt + pick fracs
        q0 = qids[0]
        item = mani[q0]
        opt_lines = item_option_lines(item, bench=args.bench)
        text = format_question(
            question_text_for_item(item),
            args.bench, options=opt_lines or None,
        )
        print("--- SAMPLE PROMPT TAIL ---")
        print(text[-400:])
        print("--- PICKS ---", picks[q0][:4], ("..." if len(picks[q0]) > 4 else ""))
        print("PREFLIGHT_OK")
        return

    ckpt_path = Path(args.ckpt or (str(args.out) + ".ckpt.jsonl"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt_path.is_file() and not args.frames_only:
        for line in ckpt_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[rec["id"]] = rec
        print(f"resumed {len(done)} from {ckpt_path}")

    lock = threading.Lock()
    pre = PRE_PROMPTS[args.bench]
    post = POST_PROMPTS[args.bench]
    print(
        f"protocol bench={args.bench} pre={pre[:60]+('…' if len(pre)>60 else '')!r} "
        f"post={post!r} timestamps={args.timestamps} "
        f"parse=extract_mcq_answer system=none (LDDR F.3) "
        f"frac_resize={'on' if n_compressed else 'off(full)'}",
        flush=True,
    )

    def run_one(qid: str) -> Optional[dict]:
        if qid in done and not args.frames_only:
            return done[qid]
        item = mani[qid]
        gold = normalize_gold(item.get("gold_letter") or item.get("gold_answer"))
        entries = picks[qid]
        secs = [t for t, _ in entries]
        fracs = [f for _, f in entries]
        frame_meta: list[dict] = []
        try:
            video = resolve_video(item, video_root)
            if not video.is_file():
                raise FileNotFoundError(str(video))
            frames: list[tuple[float, str]] = []
            for t, frac in entries:
                b64, meta = extract_frame(
                    video, t, args.max_side, args.quality, frac=frac,
                )
                frames.append((t, b64))
                frame_meta.append(meta)
            # Guard: D-arm must actually shrink when frac < 1
            for meta in frame_meta:
                if meta["frac"] < 0.999 and meta["final_pixels"] >= meta["capped_pixels"]:
                    # floor can pin size equal on already-tiny caps — allow ==
                    # only if capped already at/below floor
                    if meta["capped_pixels"] > _MIN_PIXELS_FLOOR:
                        raise RuntimeError(
                            f"frac={meta['frac']} but pixels did not shrink: "
                            f"capped={meta['capped_pixels']} final={meta['final_pixels']}"
                        )
            if args.frames_only:
                mean_px = sum(m["final_pixels"] for m in frame_meta) / len(frame_meta)
                mean_cap = sum(m["capped_pixels"] for m in frame_meta) / len(frame_meta)
                print(
                    f"[frames] {qid} mean_frac={sum(fracs)/len(fracs):.3f} "
                    f"mean_px={mean_px:.0f} mean_capped={mean_cap:.0f} "
                    f"ratio={mean_px/mean_cap:.3f} "
                    f"sizes={[m['final_wh'] for m in frame_meta]}",
                    flush=True,
                )
                rec = {
                    "id": qid,
                    "pred": None,
                    "gold": gold,
                    "ok": None,
                    "secs": secs,
                    "fracs": fracs,
                    "frame_meta": frame_meta,
                    "mean_frac": round(sum(fracs) / len(fracs), 4),
                    "mean_final_pixels": int(mean_px),
                    "mean_capped_pixels": int(mean_cap),
                    "pixel_ratio": round(mean_px / mean_cap, 4) if mean_cap else None,
                    "video": video.name,
                    "bench": args.bench,
                    "frames_only": True,
                }
                done[qid] = rec
                return rec

            opt_lines = item_option_lines(item, bench=args.bench)
            q_text = question_text_for_item(item)
            content = build_user_content(
                q_text, frames,
                bench=args.bench, timestamps=args.timestamps,
                options=opt_lines or None,
            )
            prompt_text = next(
                (p["text"] for p in content if p.get("type") == "text"), ""
            )
            if not prompt_has_options(prompt_text):
                raise RuntimeError(
                    f"prompt missing MCQ options for {qid} "
                    f"(bench={args.bench}; refused chance-level run)"
                )
            valid = valid_letters_for(item, args.bench)
            assert client is not None
            raw = ask(
                client, args.model, content,
                effort=args.effort, max_tokens=args.max_tokens,
            )
            pred = parse_letter(raw, valid=valid)
            mean_px = sum(m["final_pixels"] for m in frame_meta) / len(frame_meta)
            mean_cap = sum(m["capped_pixels"] for m in frame_meta) / len(frame_meta)
            rec = {
                "id": qid,
                "pred": pred,
                "gold": gold,
                "ok": pred == gold,
                "secs": secs,
                "fracs": fracs,
                "mean_frac": round(sum(fracs) / len(fracs), 4),
                "mean_final_pixels": int(mean_px),
                "mean_capped_pixels": int(mean_cap),
                "pixel_ratio": round(mean_px / mean_cap, 4) if mean_cap else None,
                "raw": (raw or "")[:200],
                "video": video.name,
                "bench": args.bench,
                "timestamps": args.timestamps,
                "valid_letters": valid,
                "prompt_has_options": True,
            }
        except Exception as e:
            rec = {
                "id": qid,
                "pred": "?",
                "gold": gold,
                "ok": False,
                "secs": secs,
                "fracs": fracs,
                "raw": "",
                "error": f"{type(e).__name__}: {e}",
                "video": item.get("video_file"),
                "bench": args.bench,
                "timestamps": args.timestamps,
            }
        if not args.frames_only:
            with lock:
                with open(ckpt_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[qid] = rec
        tag = "OK" if rec.get("ok") else "x"
        err = f" err={rec['error'][:80]}" if rec.get("error") else ""
        pr = rec.get("pixel_ratio")
        pr_s = f" pxr={pr}" if pr is not None else ""
        print(
            f"[{len(done)}] {qid} pred={rec['pred']} gold={rec['gold']} "
            f"{tag}{pr_s}{err}",
            flush=True,
        )
        return rec

    todo = [q for q in qids if q not in done]
    print(
        f"queue {len(todo)} / {len(qids)}  model={args.model} "
        f"effort={args.effort} workers={args.workers} "
        f"mode={'frames_only' if args.frames_only else 'api'}",
        flush=True,
    )
    if args.workers > 1 and todo:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, todo))
    else:
        for q in todo:
            run_one(q)

    records = [done[q] for q in qids if q in done]
    if args.frames_only:
        ratios = [r["pixel_ratio"] for r in records if r.get("pixel_ratio") is not None]
        out = {
            "mode": "frames_only",
            "protocol": {
                "bench": args.bench,
                "max_side": args.max_side,
                "frac_resize": True,
                "min_pixels_floor": _MIN_PIXELS_FLOOR,
                "note": "pixel_ratio = mean_final / mean_capped; cite acc deltas not token%",
            },
            "picks_mean_frac": round(mean_frac_all, 4),
            "n": len(records),
            "mean_pixel_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "records": records,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(
            f"\n== FRAMES_ONLY == n={len(records)} mean_pixel_ratio="
            f"{out['mean_pixel_ratio']} picks_mean_frac={mean_frac_all:.4f} "
            f"wrote {args.out}",
            flush=True,
        )
        return

    answered = [r for r in records if not r.get("error") and r.get("pred") != "?"]
    scored = [r for r in records if not r.get("error")]
    n = len(scored)
    correct = sum(1 for r in scored if r.get("ok"))
    acc = round(correct / n, 4) if n else None
    ratios = [r["pixel_ratio"] for r in scored if r.get("pixel_ratio") is not None]
    out = {
        "protocol": {
            "bench": args.bench,
            "pre_prompt": pre or None,
            "post_prompt": post,
            "system_prompt": None,
            "timestamps": args.timestamps,
            "parser": "harness.mcq_extract.extract_mcq_answer",
            "template_source": "LDDR F.3 / lmms-eval default",
            "max_side": args.max_side,
            "frac_resize": n_compressed > 0,
            "min_pixels_floor": _MIN_PIXELS_FLOOR,
            "picks_mean_frac": round(mean_frac_all, 4),
            "mean_pixel_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "token_note": "GPT path: report acc deltas; do not equate pixel_ratio to Qwen vis-token %",
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_requested": len(qids),
        "n": n,
        "n_ok_parse": len(answered),
        "coverage": round(n / len(qids), 4) if qids else 0,
        "accuracy": acc,
        "correct": correct,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(
        f"\n== GPT {args.bench} == acc={acc} n={n}/{len(qids)} "
        f"coverage={out['coverage']} mean_px_ratio="
        f"{out['protocol']['mean_pixel_ratio']} wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
