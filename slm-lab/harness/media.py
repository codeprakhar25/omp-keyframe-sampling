"""Media loading: turn a manifest item into a list of candidate Frames.

A `Frame` is one still image plus, for video, its timestamp in seconds. Selectors
operate on the candidate frame list; answerers consume the selected subset.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Frame:
    index: int
    image: Image.Image
    seconds: Optional[float] = None  # wall-clock position for video frames; None for stills


def load_frames(item: dict, dump_fps: float = 1.0, max_frames: int = 64) -> List[Frame]:
    """Load the *candidate* frame set for an item (this is condition A's payload)."""
    mtype = item.get("media_type", "video")
    if mtype == "video":
        return _load_video_frames(item["media_path"], dump_fps, max_frames)
    if mtype == "images":
        return _load_image_dir(item.get("media_path"), item.get("media_paths"), max_frames)
    if mtype == "image":
        img = Image.open(item["media_path"]).convert("RGB")
        return [Frame(index=0, image=img, seconds=None)]
    raise ValueError(f"unknown media_type: {mtype!r}")


def _load_video_frames(path: str, dump_fps: float, max_frames: int) -> List[Frame]:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "opencv-python is required for media_type 'video' (pip install opencv-python)"
        ) from e

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / max(dump_fps, 1e-6))), 1)

    frames: List[Frame] = []
    src_idx = 0
    out_idx = 0
    while True:
        if not cap.grab():
            break
        if src_idx % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Frame(index=out_idx, image=Image.fromarray(rgb), seconds=src_idx / native_fps))
            out_idx += 1
            if len(frames) >= max_frames:
                break
        src_idx += 1
    cap.release()
    return frames


def _load_image_dir(media_path: Optional[str], media_paths: Optional[List[str]], max_frames: int) -> List[Frame]:
    if media_paths:
        paths = list(media_paths)
    elif media_path and os.path.isdir(media_path):
        paths = sorted(
            os.path.join(media_path, f)
            for f in os.listdir(media_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    elif media_path:
        paths = [media_path]
    else:
        raise ValueError("media_type 'images' needs media_path (file/dir) or media_paths (list)")

    return [
        Frame(index=i, image=Image.open(p).convert("RGB"), seconds=float(i))
        for i, p in enumerate(paths[:max_frames])
    ]


def frame_to_base64(frame: Frame, fmt: str = "JPEG", max_side: int = 768, quality: int = 85) -> str:
    """Encode a frame as base64 for vision APIs, downscaling the long side to cap tokens."""
    img = frame.image
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
