#!/usr/bin/env python3
"""Curate CogVideoX identity LoRA clips from raw avatar-lab videos.

- Uses personal sources (WhatsApp_*, pers.mp4, new.mp4, new2.mp4)
- Splits long clips into 2.5–3.5s segments (multiple per source OK)
- Skips stock/watermarked files
- Dedupes near-identical segments via perceptual hash on mid-frame
- Normalizes to portrait 480x848-ish (even dimensions) for training

Outputs:
  data/video_lora/
    clips/clip_001.mp4 + clip_001.txt
    manifest.json
    REPORT.md
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: uv pip install pillow", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "video"
OUT = ROOT / "data" / "video_lora"
CLIPS = OUT / "clips"
PREVIEW = OUT / "preview"

TRIGGER = "PRAKH_PERSON"
CLIP_MIN = 2.5
CLIP_TARGET = 3.0
CLIP_MAX = 3.5
# Gap between windows on same source — reduces near-duplicate motion
WINDOW_STRIDE = 3.2
MIN_GAP_BETWEEN_CLIPS = 0.8
DEDUPE_HAMMING = 8  # lower = stricter

INCLUDE_PATTERNS = (
    re.compile(r"^WhatsApp Video", re.I),
    re.compile(r"^pers\.mp4$", re.I),
    re.compile(r"^new\.mp4$", re.I),
    re.compile(r"^new2\.mp4$", re.I),
)

# Explicit tags for manually named diversity clips
SOURCE_TAGS = {
    "new.mp4": "low_light",
    "new2.mp4": "outdoor_setting",
}
EXCLUDE_NAMES = {
    "istockphoto-479312547-640_adpp_is.mp4",
    "watermarked_preview.mp4",
}


@dataclass
class SourceInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    portrait: bool


@dataclass
class ClipPlan:
    source: str
    start: float
    duration: float
    tag: str  # category hint for caption


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def ffprobe(path: Path) -> SourceInfo:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
    return SourceInfo(path, dur, w, h, fps, h > w)


def dhash(image_path: Path, size: int = 16) -> int:
    img = Image.open(image_path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_mid_frame(video: Path, start: float, duration: float, out_jpg: Path) -> None:
    t = start + duration / 2
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_jpg),
        ]
    )


def export_clip(video: Path, start: float, duration: float, out_mp4: Path) -> None:
    """Portrait crop center + scale to 480x848 (even), 30fps, h264, no audio."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        "scale=480:848:force_original_aspect_ratio=increase,"
        "crop=480:848,fps=30,format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )


def guess_tag(source_name: str, start: float, duration: float, portrait: bool) -> str:
    if source_name in SOURCE_TAGS:
        return SOURCE_TAGS[source_name]
    if not portrait:
        return "landscape_talking"
    if duration <= 2.5:
        return "short_reaction"
    if "19.42.18" in source_name:
        return "outdoor_bright"
    if "19.42.19" in source_name:
        return "indoor_talking"
    if "19.42.20" in source_name:
        return "indoor_varied"
    if source_name.lower().startswith("pers"):
        return "reference_talking"
    return "talking_head"


def caption_for(tag: str) -> str:
    base = f"{TRIGGER}, a man talking to camera"
    extras = {
        "low_light": "indoors in low light, dim room, speaking naturally",
        "outdoor_setting": "outdoors, natural daylight, casual outdoor setting",
        "outdoor_bright": "outdoors in daylight, casual setting",
        "indoor_talking": "indoors, neutral background, speaking naturally",
        "indoor_varied": "indoors, varied expressions while speaking",
        "landscape_talking": "landscape frame, face visible while speaking",
        "short_reaction": "brief clip, natural expression",
        "reference_talking": "clear frontal talking head, reference style",
        "talking_head": "talking head, natural motion",
    }
    return f"{base}, {extras.get(tag, 'natural motion')}"


def plan_segments(info: SourceInfo) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    name = info.path.name
    d = info.duration

    if d < CLIP_MIN:
        return plans  # too short — likely unusable for training

    if d <= CLIP_MAX:
        dur = min(d - 0.05, CLIP_MAX)
        if dur >= CLIP_MIN:
            plans.append(ClipPlan(name, 0.05, dur, guess_tag(name, 0, dur, info.portrait)))
        return plans

    start = 0.15
    while start + CLIP_MIN <= d - 0.1:
        dur = min(CLIP_TARGET, d - start - 0.05)
        if dur > CLIP_MAX:
            dur = CLIP_TARGET
        if dur < CLIP_MIN:
            break
        plans.append(ClipPlan(name, start, dur, guess_tag(name, start, dur, info.portrait)))
        start += WINDOW_STRIDE

    return plans


def is_duplicate(h: int, seen: list[int]) -> bool:
    return any(hamming(h, s) <= DEDUPE_HAMMING for s in seen)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Missing source dir: {SRC}")

    sources: list[SourceInfo] = []
    for p in sorted(SRC.glob("*.mp4")):
        if p.name in EXCLUDE_NAMES:
            continue
        if not any(rx.search(p.name) for rx in INCLUDE_PATTERNS):
            continue
        sources.append(ffprobe(p))

    # Clean output
    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    CLIPS.mkdir(parents=True)
    PREVIEW.mkdir(parents=True)

    all_plans: list[ClipPlan] = []
    for info in sources:
        all_plans.extend(plan_segments(info))

    accepted: list[dict] = []
    seen_hashes: list[int] = []
    rejected_dup = 0
    clip_idx = 0

    for plan in all_plans:
        src_path = SRC / plan.source
        preview_jpg = PREVIEW / f"candidate_{clip_idx:03d}.jpg"
        extract_mid_frame(src_path, plan.start, plan.duration, preview_jpg)
        h = dhash(preview_jpg)
        if is_duplicate(h, seen_hashes):
            rejected_dup += 1
            preview_jpg.unlink(missing_ok=True)
            continue
        seen_hashes.append(h)

        clip_idx += 1
        stem = f"clip_{clip_idx:03d}"
        out_mp4 = CLIPS / f"{stem}.mp4"
        out_txt = CLIPS / f"{stem}.txt"
        export_clip(src_path, plan.start, plan.duration, out_mp4)
        out_txt.write_text(caption_for(plan.tag) + "\n", encoding="utf-8")

        accepted.append(
            {
                "id": stem,
                "source": plan.source,
                "start_sec": round(plan.start, 3),
                "duration_sec": round(plan.duration, 3),
                "tag": plan.tag,
                "caption": caption_for(plan.tag),
                "dhash": hex(h),
            }
        )

    manifest = {
        "trigger_word": TRIGGER,
        "clip_count": len(accepted),
        "rejected_near_duplicates": rejected_dup,
        "sources_used": len(sources),
        "sources_excluded": list(EXCLUDE_NAMES),
        "settings": {
            "clip_min_sec": CLIP_MIN,
            "clip_target_sec": CLIP_TARGET,
            "clip_max_sec": CLIP_MAX,
            "window_stride_sec": WINDOW_STRIDE,
            "dedupe_hamming_max": DEDUPE_HAMMING,
            "output_resolution": "480x848",
            "fps": 30,
        },
        "clips": accepted,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Report
    by_source: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    for c in accepted:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
        by_tag[c["tag"]] = by_tag.get(c["tag"], 0) + 1

    lines = [
        "# CogVideoX LoRA dataset report",
        "",
        f"- **Clips accepted:** {len(accepted)}",
        f"- **Near-duplicates rejected:** {rejected_dup}",
        f"- **Sources scanned:** {len(sources)} personal clips",
        f"- **Output:** `{CLIPS.relative_to(ROOT)}`",
        "",
        "## Clips per source",
        "",
    ]
    for k, v in sorted(by_source.items()):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Clips per category", ""])
    for k, v in sorted(by_tag.items()):
        lines.append(f"- {k}: {v}")
    lines.extend(
        [
            "",
            "## Gaps to fill (record new if needed)",
            "",
            "Target 15–25 clips with variety. Check you have:",
            "- [ ] Different shirts across clips",
            "- [ ] Different rooms/backgrounds",
            "- [ ] Side angles (not only frontal)",
            "- [ ] Different lighting (window vs overhead)",
            "- [ ] Expressions (smile, serious, laugh)",
            "",
            f"Current count **{len(accepted)}** — "
            + ("OK to start training." if len(accepted) >= 15 else "**record more** to reach 15+."),
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"clips": len(accepted), "rejected_dup": rejected_dup, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
