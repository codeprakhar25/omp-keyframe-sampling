#!/usr/bin/env python3
"""Map IV cut proposals → time spans given word-level alignment.

Pure logic (no GPU). Used by iv_cut_slice locally / on Modal.

Alignment JSON per source clip:
  {"words": [{"word": "...", "start": 0.1, "end": 0.3}, ...], "duration_s": 7.7}
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

TOKEN_RE = re.compile(r"<([^<>]+)>|[^\s<>]+")
BRACKET_BODY_RE = re.compile(r"\[([^\]]+)\]")


def normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFC", (w or "").strip())
    w = w.replace("\u200d", "").replace("\u200c", "")
    w = BRACKET_BODY_RE.sub(r"\1", w)
    return w


def plain_words_from_transcript(text: str) -> list[str]:
    out = []
    for m in TOKEN_RE.finditer(text or ""):
        if m.group(1) is not None:
            continue
        w = normalize_word(m.group(0))
        if w:
            out.append(w)
    return out


def word_index_before_tag(text: str, token_idx: int) -> int:
    """Count plain words strictly before token_idx in tokenize order."""
    n = 0
    i = 0
    for m in TOKEN_RE.finditer(text or ""):
        if i == token_idx:
            break
        if m.group(1) is None:
            w = normalize_word(m.group(0))
            if w:
                n += 1
        i += 1
    return n


def align_words_to_ref(
    ref_words: list[str], aligned: list[dict]
) -> list[dict | None]:
    """Map each ref word → aligned {start,end,word} by sequential greedy match.

    Returns list len(ref_words); None if unmatched.
    """
    out: list[dict | None] = [None] * len(ref_words)
    j = 0
    for i, rw in enumerate(ref_words):
        rw_n = normalize_word(rw).lower()
        best = None
        best_j = None
        # search forward window
        for k in range(j, min(len(aligned), j + 8)):
            aw = normalize_word(str(aligned[k].get("word") or "")).lower()
            if not aw:
                continue
            if aw == rw_n or aw in rw_n or rw_n in aw:
                best = aligned[k]
                best_j = k
                break
        if best is None and j < len(aligned):
            # positional fallback within small drift
            best = aligned[j]
            best_j = j
        if best is not None and best_j is not None:
            out[i] = {
                "word": rw,
                "start": float(best["start"]),
                "end": float(best["end"]),
            }
            j = best_j + 1
    return out


def cut_times_from_alignment(
    text: str,
    token_idx: int,
    n_left: int,
    n_right: int,
    aligned_words: list[dict],
    duration_s: float,
    *,
    pad_s: float = 0.03,
    min_dur_s: float = 1.2,
) -> dict:
    """Compute t_start/t_end for a cut. Returns dict with ok + times + reason."""
    ref = plain_words_from_transcript(text)
    if not ref:
        return {"ok": False, "reason": "no_words"}
    mapped = align_words_to_ref(ref, aligned_words)
    n_before = word_index_before_tag(text, token_idx)
    # left words are the n_left words immediately before tag
    left_i0 = max(0, n_before - n_left)
    left_i1 = n_before  # exclusive
    right_i0 = n_before
    right_i1 = min(len(ref), n_before + n_right)

    left_span = [mapped[i] for i in range(left_i0, left_i1) if mapped[i]]
    right_span = [mapped[i] for i in range(right_i0, right_i1) if mapped[i]]

    if not left_span and not right_span:
        return {"ok": False, "reason": "no_aligned_words"}

    if left_span:
        t0 = left_span[0]["start"]
    else:
        # clip-initial tag: start a bit before first right word
        t0 = max(0.0, right_span[0]["start"] - 0.15)

    if right_span:
        t1 = right_span[-1]["end"]
    else:
        t1 = min(duration_s, left_span[-1]["end"] + 0.15)

    t0 = max(0.0, t0 - pad_s)
    t1 = min(duration_s, t1 + pad_s)

    if t1 <= t0:
        return {"ok": False, "reason": "bad_order", "t_start_s": t0, "t_end_s": t1}

    dur = t1 - t0
    if dur < min_dur_s:
        # expand symmetrically toward clip edges
        need = min_dur_s - dur
        t0 = max(0.0, t0 - need / 2)
        t1 = min(duration_s, t1 + need / 2)
        if t1 - t0 < min_dur_s * 0.85:
            return {
                "ok": False,
                "reason": "too_short",
                "t_start_s": round(t0, 3),
                "t_end_s": round(t1, 3),
                "dur_s": round(t1 - t0, 3),
            }

    n_mapped = sum(1 for x in mapped if x is not None)
    return {
        "ok": True,
        "reason": "",
        "t_start_s": round(t0, 3),
        "t_end_s": round(t1, 3),
        "dur_s": round(t1 - t0, 3),
        "n_ref_words": len(ref),
        "n_aligned_matched": n_mapped,
        "word_i0": left_i0 if left_span else right_i0,
        "word_i1": right_i1 if right_span else left_i1,
    }


def resolve_source_wav(root: Path, clip: dict) -> Path | None:
    """Same candidates as serve_lab (absolute paths)."""
    data = root / "data"
    f = clip.get("file") or ""
    hp = clip.get("hf_or_path") or ""
    cid = clip.get("id") or ""
    name = Path(f).name if f else ""
    cands = [
        data / hp if hp else None,
        data / f if f else None,
        data / f"samples/{f}" if f else None,
        data / f"samples/scale/{name}" if name else None,
        data / f"samples/smoke/{name}" if name else None,
        data / f"scale/train/wavs/{cid}.wav" if cid else None,
    ]
    for p in cands:
        if p and p.is_file():
            return p
    return None
