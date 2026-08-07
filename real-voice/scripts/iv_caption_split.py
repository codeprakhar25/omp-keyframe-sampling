#!/usr/bin/env python3
"""Split IndicVoices tagged transcripts into single-tag caption cut proposals.

Catalog-only: labels come from IV <> tags in the transcript. Human accept/reject
is caption/cut sanity QA — not retagging.

Tag lock / gate / emit: see scripts/iv_tags.py
  Locked add-ons: gasp, throat_clearing, cough, tsk, ugh, sniffle
  breathing ≠ inhaling

Policy v4:
  - Non-overlapping windows: exclusive zones between primaries
  - Stack of ≥2 emit-class tags → no auto row (stack_ambiguous)
  - Hard-stop at talking / child* / unintelligible / singing
  - Min ≥4 words; ≥1 word each side unless that side is clip edge
  - Cap ≤2 inhaling rows/clip, ≤3 rows/clip total
  - cut_id = {clip}_{tag}_t{token_idx}
  - Other emit-class tag inside built span → drop (catalog incomplete)

  .venv/bin/python scripts/iv_caption_split.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from iv_tags import (  # noqa: E402
    EMIT_TAGS,
    GATE_RAW,
    HARD_STOP_TAGS,
    NOISE_TAGS,
    SKIP_VOCAL,
    TAG_MAP,
    mapped_tag as _mapped_tag,
)

SCALE = ROOT / "data" / "scale"
DRAFTS = SCALE / "drafts.json"
OUT = SCALE / "iv_cut_proposals.json"

ALGO_VERSION = 7

TOKEN_RE = re.compile(r"<([^<>]+)>|[^\s<>]+")
BRACKET_BODY_RE = re.compile(r"\[([^\]]+)\]")
TAB_ORDER = list(EMIT_TAGS)


def normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFC", w)
    w = w.replace("\u200d", "").replace("\u200c", "")
    return BRACKET_BODY_RE.sub(r"\1", w)


def tokenize(text: str) -> list[dict]:
    out = []
    for m in TOKEN_RE.finditer(text or ""):
        if m.group(1) is not None:
            raw = m.group(1).strip().lower()
            out.append({"kind": "tag", "value": raw, "raw_tag": raw})
        else:
            w = normalize_word(m.group(0).strip())
            if w:
                out.append({"kind": "word", "value": w})
    return out


def mapped_tag(raw: str) -> str | None:
    return _mapped_tag(raw)


def words_join(tokens: list[dict]) -> str:
    return " ".join(t["value"] for t in tokens if t["kind"] == "word").strip()


def clip_has_gate(tokens: list[dict]) -> bool:
    return any(t["kind"] == "tag" and t["value"] in GATE_RAW for t in tokens)


def primary_tag_indices(
    tokens: list[dict], emit_tags: set[str]
) -> tuple[list[tuple[int, str]], list[dict]]:
    primaries: list[tuple[int, str]] = []
    skipped_stacks: list[dict] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i]["kind"] != "tag":
            i += 1
            continue
        j = i
        emit_in_stack: list[tuple[int, str, str]] = []
        while j < n and tokens[j]["kind"] == "tag":
            want = mapped_tag(tokens[j]["value"])
            if want in emit_tags:
                emit_in_stack.append((j, want, tokens[j]["value"]))
            j += 1
        if len(emit_in_stack) == 1:
            idx, want, _ = emit_in_stack[0]
            primaries.append((idx, want))
        elif len(emit_in_stack) >= 2:
            skipped_stacks.append(
                {
                    "token_idx": emit_in_stack[0][0],
                    "candidates": [
                        {"token_idx": a, "tag": b, "raw_tag": c}
                        for a, b, c in emit_in_stack
                    ],
                    "reason": "stack_ambiguous",
                }
            )
        i = j if j > i else i + 1
    return primaries, skipped_stacks


def collect_side(
    tokens: list[dict],
    start: int,
    step: int,
    bound: int,
    max_words: int,
) -> tuple[list[dict], list[int]]:
    """Collect words toward bound. Returns (word_tokens, indices_touched)."""
    words: list[dict] = []
    touched: list[int] = []
    i = start
    while 0 <= i < len(tokens) and (
        (step < 0 and i >= bound) or (step > 0 and i < bound)
    ):
        if len(words) >= max_words:
            break
        t = tokens[i]
        if t["kind"] == "tag":
            if t["value"] in HARD_STOP_TAGS:
                break
            touched.append(i)
            i += step
            continue
        words.append(t)
        touched.append(i)
        i += step
    if step < 0:
        words.reverse()
        touched.reverse()
    return words, touched


def emit_tags_in_indices(
    tokens: list[dict], indices: list[int], primary_idx: int, emit_tags: set[str]
) -> list[str]:
    found = []
    for i in indices:
        if i == primary_idx:
            continue
        t = tokens[i]
        if t["kind"] != "tag":
            continue
        want = mapped_tag(t["value"])
        if want in emit_tags:
            found.append(want)
    return found


def window_around(
    tokens: list[dict],
    tag_idx: int,
    want: str,
    left_words: int,
    right_words: int,
    prev_primary: int | None,
    next_primary: int | None,
    emit_tags: set[str],
) -> tuple[dict | None, str]:
    """Build non-overlapping window. Returns (prop, drop_reason)."""
    left_bound = (prev_primary + 1) if prev_primary is not None else 0
    right_bound = next_primary if next_primary is not None else len(tokens)

    left, left_idx = collect_side(tokens, tag_idx - 1, -1, left_bound, left_words)
    right, right_idx = collect_side(tokens, tag_idx + 1, +1, right_bound, right_words)

    span_indices = left_idx + [tag_idx] + right_idx
    others = emit_tags_in_indices(tokens, span_indices, tag_idx, emit_tags)
    if others:
        return None, "unlabeled_emit"

    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
    if n_total < 4:
        return None, "short"

    # each side ≥1 unless that side is true clip edge (no neighboring primary)
    if n_left < 1 and prev_primary is not None:
        return None, "empty_side"
    if n_right < 1 and next_primary is not None:
        return None, "empty_side"
    if n_left < 1 and prev_primary is None and n_right < 4:
        return None, "short"
    if n_right < 1 and next_primary is None and n_left < 4:
        return None, "short"

    left_txt = words_join(left)
    right_txt = words_join(right)
    parts = []
    if left_txt:
        parts.append(left_txt)
    parts.append(f"[{want}]")
    if right_txt:
        parts.append(right_txt)

    n_words = sum(1 for t in tokens if t["kind"] == "word")
    words_before = sum(1 for j in range(tag_idx) if tokens[j]["kind"] == "word")
    frac = (words_before / n_words) if n_words else 0.5

    return {
        "tag": want,
        "caption_inline": " ".join(parts),
        "text_left": left_txt,
        "text_right": right_txt,
        "text_plain": words_join(left + right),
        "n_words_left": n_left,
        "n_words_right": n_right,
        "approx_word_frac": round(frac, 3),
        "raw_tag": tokens[tag_idx].get("raw_tag") or tokens[tag_idx]["value"],
        "token_idx": tag_idx,
        "status": "proposed",
        "notes": "",
        "reject_reason": "",
    }, ""


def apply_caps(proposals: list[dict], max_inhaling: int, max_rows: int) -> tuple[list[dict], int]:
    kept: list[dict] = []
    n_inh = 0
    dropped = 0
    for p in sorted(proposals, key=lambda x: x["token_idx"]):
        if len(kept) >= max_rows:
            dropped += 1
            continue
        if p["tag"] == "inhaling":
            if n_inh >= max_inhaling:
                dropped += 1
                continue
            n_inh += 1
        kept.append(p)
    return kept, dropped


def split_clip(
    clip: dict,
    emit_tags: set[str],
    left_words: int,
    right_words: int,
    max_inhaling: int,
    max_rows: int,
) -> tuple[list[dict], dict, list[dict]]:
    tokens = tokenize(clip.get("text") or "")
    stats = {
        "stack_ambiguous": 0,
        "drop_short": 0,
        "drop_empty_side": 0,
        "drop_unlabeled_emit": 0,
        "drop_cap": 0,
    }
    if not clip_has_gate(tokens):
        return [], stats, []

    primaries, skipped_stacks = primary_tag_indices(tokens, emit_tags)
    stats["stack_ambiguous"] = len(skipped_stacks)
    primary_idxs = [i for i, _ in primaries]

    raw_props: list[dict] = []
    for k, (idx, want) in enumerate(primaries):
        prev_p = primary_idxs[k - 1] if k > 0 else None
        next_p = primary_idxs[k + 1] if k + 1 < len(primary_idxs) else None
        prop, reason = window_around(
            tokens,
            idx,
            want,
            left_words,
            right_words,
            prev_p,
            next_p,
            emit_tags,
        )
        if not prop:
            if reason == "unlabeled_emit":
                stats["drop_unlabeled_emit"] += 1
            elif reason == "empty_side":
                stats["drop_empty_side"] += 1
            else:
                stats["drop_short"] += 1
            continue
        prop["cut_id"] = f"{clip['id']}_{want}_t{idx}"
        prop["source_id"] = clip["id"]
        prop["algo_version"] = ALGO_VERSION
        raw_props.append(prop)

    capped, n_cap = apply_caps(raw_props, max_inhaling=max_inhaling, max_rows=max_rows)
    stats["drop_cap"] = n_cap
    return capped, stats, skipped_stacks


def merge_prior_status(new_props: list[dict], old_path: Path) -> None:
    """Keep human QA when cut_id matches and algo_version is v3 (or caption identical)."""
    if not old_path.exists():
        return
    try:
        old = json.loads(old_path.read_text(encoding="utf-8"))
    except Exception:
        return
    by_id = {p.get("cut_id"): p for p in old.get("proposals") or [] if p.get("cut_id")}
    for p in new_props:
        prev = by_id.get(p["cut_id"])
        if not prev:
            continue
        prev_ver = prev.get("algo_version")
        if prev_ver is not None and prev_ver != ALGO_VERSION:
            continue
        if prev_ver is None and prev.get("caption_inline") != p.get("caption_inline"):
            continue
        if prev.get("status") in ("accepted", "rejected"):
            p["status"] = prev["status"]
            if prev.get("caption_inline"):
                p["caption_inline"] = prev["caption_inline"]
        if prev.get("notes"):
            p["notes"] = prev["notes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-tags", default=",".join(TAB_ORDER))
    ap.add_argument("--source", default="iv")
    ap.add_argument("--left-words", type=int, default=8)
    ap.add_argument("--right-words", type=int, default=8)
    ap.add_argument("--max-inhaling-per-clip", type=int, default=2)
    ap.add_argument("--max-rows-per-clip", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    emit_tags = {t.strip() for t in args.emit_tags.split(",") if t.strip()}
    doc = json.loads(DRAFTS.read_text(encoding="utf-8"))
    clips = [c for c in doc.get("clips") or [] if (c.get("source") or "") == args.source]

    all_props: list[dict] = []
    by_src: dict[str, list[dict]] = {}
    n_gated = 0
    agg = {
        "stack_ambiguous": 0,
        "drop_short": 0,
        "drop_empty_side": 0,
        "drop_unlabeled_emit": 0,
        "drop_cap": 0,
    }
    skipped_stack_rows: list[dict] = []

    for c in clips:
        tokens = tokenize(c.get("text") or "")
        if not clip_has_gate(tokens):
            continue
        n_gated += 1
        props, st, stacks = split_clip(
            c,
            emit_tags,
            args.left_words,
            args.right_words,
            args.max_inhaling_per_clip,
            args.max_rows_per_clip,
        )
        for k in agg:
            agg[k] += st.get(k, 0)
        for s in stacks:
            skipped_stack_rows.append({"source_id": c["id"], **s})
        if not props:
            continue
        by_src[c["id"]] = props
        all_props.extend(props)

    merge_prior_status(all_props, args.out)

    by_tag = {t: sum(1 for p in all_props if p["tag"] == t) for t in TAB_ORDER if t in emit_tags}
    for t in sorted(emit_tags):
        if t not in by_tag:
            by_tag[t] = sum(1 for p in all_props if p["tag"] == t)
    by_tag_clips = {
        t: len({p["source_id"] for p in all_props if p["tag"] == t}) for t in by_tag
    }

    out = {
        "version": ALGO_VERSION,
        "algo_version": ALGO_VERSION,
        "source_filter": args.source,
        "label_source": "indicvoices_verbatim_tags",
        "gate_raw_tags": sorted(GATE_RAW),
        "emit_tags": [t for t in TAB_ORDER if t in emit_tags],
        "stack_policy": "reject_multi_emit_stack",
        "window_policy": "non_overlap_exclusive_zones",
        "breathing_vs_inhaling": "separate",
        "left_words": args.left_words,
        "right_words": args.right_words,
        "max_inhaling_per_clip": args.max_inhaling_per_clip,
        "max_rows_per_clip": args.max_rows_per_clip,
        "min_words": 4,
        "n_gated_clips": n_gated,
        "n_source_clips": len(by_src),
        "n_proposals": len(all_props),
        "by_tag": by_tag,
        "by_tag_distinct_clips": by_tag_clips,
        "drop_stats": agg,
        "n_stacks_skipped": len(skipped_stack_rows),
        "stacks_skipped": skipped_stack_rows,
        "tab_order": [t for t in TAB_ORDER if by_tag.get(t, 0) > 0],
        "note": (
            "Catalog-only IV <> tags. Gate/emit from iv_tags.py "
            "(+gasp,throat_clearing,cough,tsk,ugh,sniffle). breathing≠inhaling. "
            "Stacks rejected. Non-overlap. Accept=sanity not retag."
        ),
        "tag_lock": {
            "source": "scripts/iv_tags.py",
            "locked_addons": [
                "gasp",
                "throat_clearing",
                "cough",
                "tsk",
                "ugh",
                "sniffle",
            ],
            "gate_raw": sorted(GATE_RAW),
            "emit_tags": list(EMIT_TAGS),
            "ignore_emit": ["inhaling"],
            "skip_vocal": sorted(SKIP_VOCAL),
        },
        "proposals": all_props,
        "by_source_id": by_src,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"gated={n_gated} clips_with_props={len(by_src)} "
        f"proposals={len(all_props)} by_tag={by_tag} "
        f"by_tag_clips={by_tag_clips} drops={agg} stacks={len(skipped_stack_rows)}"
    )
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
