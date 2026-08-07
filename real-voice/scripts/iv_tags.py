"""Locked IndicVoices event-tag config (single source of truth).

Catalog-only. breathing ≠ inhaling.

Balance rules (v7):
  - inhaling BACK in emit/gate, hard-capped ~5k source clips
  - thinking NOT in gate (co-occur only) + hard cap — stops 33k thinking flood
  - breathing gated but hard-capped
  - rare tags (whisper, sniffle, laugh, sigh, …): keep all survivors
  - noise never blocks a gated clip
"""
from __future__ import annotations

TAG_MAP: dict[str, str] = {
    "sigh": "sigh",
    "whispering": "whispering",
    "inhaling": "inhaling",
    "breathing": "breathing",
    "uhh": "thinking",
    "umm": "thinking",
    "hmm": "thinking",
    "laughter": "laugh",
    "laugh": "laugh",
    "gasp": "gasp",
    "throat_clearing": "throat_clearing",
    "cough": "cough",
    "tsk": "tsk",
    "ugh": "ugh",
    "sniffle": "sniffle",
}

# Open clip if ≥1 of these. thinking (uhh/umm/hmm) intentionally OUT —
# still emitted when co-occurring with an event below.
GATE_RAW: set[str] = {
    "sigh",
    "whispering",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
    "laughter",
    "laugh",
    "breathing",
    "inhaling",
}

EMIT_TAGS: list[str] = [
    "sigh",
    "whispering",
    "breathing",
    "inhaling",
    "thinking",
    "laugh",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
]

EMIT_SET: set[str] = set(EMIT_TAGS)

IGNORE_EMIT: set[str] = set()  # none for now

# After split: max distinct source clips per tag (None / missing = unlimited)
TAG_SOURCE_CAPS: dict[str, int] = {
    "thinking": 3000,  # co-occur only + cap
    "breathing": 5000,
    "inhaling": 5000,
}

SKIP_VOCAL: set[str] = {
    "stammers",
    "singing",
    "smack",
    "whistling",
    "hum",
    "groan",
    "swallowing",
    "sneezing",
    "yawning",
    "snorting",
    "wheezing",
    "nose_blowing",
    "uh-huh",
    "yelling",
    "trill",
}

NOISE_TAGS: set[str] = {
    "persistent-noise-start",
    "persistent-noise-end",
    "noise",
    "talking",
    "bird_squawk",
    "bird",
    "click",
    "clicking",
    "clink",
    "clinking",
    "clanking",
    "tapping",
    "thumping",
    "pounding",
    "horn",
    "motorcycle",
    "dishes",
    "barking",
    "rustling",
    "squeak",
    "squawking",
    "screeching",
    "beep",
    "bell",
    "static",
    "child",
    "child_talking",
    "child_yelling",
    "child_crying",
    "child_whining",
    "child_laughing",
    "children",
    "children_talking",
    "children_yelling",
    "baby",
    "baby_crying",
    "baby_talking",
    "unintelligible",
    "hiss",
    "music",
    "clanging",
    "rattling",
    "animal",
    "phone_vibrating",
    "phone_ringing",
    "popping",
    "buzz",
    "buzzer",
    "ringing",
    "scratching",
    "chiming",
    "siren",
    "tone",
    "tones",
    "footsteps",
    "meow",
    "typewriter",
    "sniffing",
    *SKIP_VOCAL,
}

HARD_STOP_TAGS: set[str] = {
    "talking",
    "child",
    "child_talking",
    "child_yelling",
    "unintelligible",
    "singing",
    "yelling",
    "children_yelling",
}


def mapped_tag(raw: str) -> str | None:
    r = (raw or "").strip().lower()
    if r in NOISE_TAGS or r in SKIP_VOCAL:
        return None
    want = TAG_MAP.get(r)
    if want in IGNORE_EMIT:
        return None
    return want


def apply_tag_source_caps(
    proposals: list[dict],
    caps: dict[str, int] | None = None,
) -> tuple[list[dict], dict]:
    """Drop excess rows for capped tags (keep earliest by cut_id / source).

    Preference: keep a source's non-capped tags always; for capped tags keep
    sources that also have an uncapped emit tag first, then fill.
    """
    caps = caps if caps is not None else TAG_SOURCE_CAPS
    if not caps:
        return proposals, {}

    uncapped_tags = set(EMIT_TAGS) - set(caps)
    # sources that carry at least one uncapped tag
    sources_with_uncapped: set[str] = set()
    for p in proposals:
        if p["tag"] in uncapped_tags:
            sources_with_uncapped.add(p["source_id"])

    kept_sources_for_tag: dict[str, set[str]] = {t: set() for t in caps}
    dropped = {t: 0 for t in caps}
    out: list[dict] = []

    # pass 1: all uncapped rows
    for p in proposals:
        if p["tag"] not in caps:
            out.append(p)

    # pass 2: capped tags — prefer multi-event sources
    def sort_key(p: dict) -> tuple:
        sid = p["source_id"]
        prefer = 0 if sid in sources_with_uncapped else 1
        return (prefer, sid, p.get("cut_id") or "")

    for p in sorted((x for x in proposals if x["tag"] in caps), key=sort_key):
        t = p["tag"]
        sid = p["source_id"]
        cap = caps[t]
        if sid in kept_sources_for_tag[t]:
            out.append(p)
            continue
        if len(kept_sources_for_tag[t]) >= cap:
            dropped[t] += 1
            continue
        kept_sources_for_tag[t].add(sid)
        out.append(p)

    stats = {
        t: {
            "cap": caps[t],
            "sources_kept": len(kept_sources_for_tag[t]),
            "rows_dropped": dropped[t],
        }
        for t in caps
    }
    return out, stats
