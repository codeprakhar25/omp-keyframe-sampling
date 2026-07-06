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
        return _load_image_dir(item.get("media_path"), item.get("media_paths"), max_frames, dump_fps)
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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[Frame] = []
    if total > 0:
        # Spread the budget across the WHOLE clip. When the request exceeds max_frames the
        # cap becomes a *coarser fps* (honest full-dump ~ model-knob), NOT a truncation to the
        # opening frames — otherwise long-bin full-dump silently ignores late needles and the
        # Fork-A crossover / hit@k is corrupted. (Mirrors the images-path uniform-to-cap fix.)
        import numpy as np

        want = max(1, int(round((total / native_fps) * max(dump_fps, 1e-6))))
        n_take = min(want, max_frames) if max_frames else want
        target = sorted({int(x) for x in np.linspace(0, total - 1, n_take)})
        target_set = set(target)
        src_idx = out_idx = 0
        while len(frames) < n_take and cap.grab():
            if src_idx in target_set:
                ok, bgr = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(Frame(index=out_idx, image=Image.fromarray(rgb), seconds=src_idx / native_fps))
                    out_idx += 1
            src_idx += 1
        cap.release()
        return frames

    # fallback: unknown length -> sequential step sampling (old behaviour)
    step = max(int(round(native_fps / max(dump_fps, 1e-6))), 1)
    src_idx = out_idx = 0
    while cap.grab():
        if src_idx % step == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Frame(index=out_idx, image=Image.fromarray(rgb), seconds=src_idx / native_fps))
            out_idx += 1
            if max_frames and len(frames) >= max_frames:
                break
        src_idx += 1
    cap.release()
    return frames


def _frame_second(fname: str) -> Optional[int]:
    """Parse the wall-clock second from an extracted frame name `f<sec>.jpg`."""
    base = os.path.splitext(os.path.basename(fname))[0]
    if base.startswith("f") and base[1:].isdigit():
        return int(base[1:])
    return None


def _load_image_dir(
    media_path: Optional[str],
    media_paths: Optional[List[str]],
    max_frames: int,
    dump_fps: float = 1.0,
) -> List[Frame]:
    """Load an extracted 1fps frame set. `seconds`/`index` are the real wall-clock
    second (from the `f<sec>.jpg` name), so selection spread and hit@k are temporal.

    dump_fps<1 subsamples in time (model-knob). When the frame count still exceeds
    max_frames, we subsample *uniformly across the whole video* — not the first N —
    so full-dump under a provider image cap stays representative of the entire clip.
    """
    if media_paths:
        paths = list(media_paths)
        secs: List[int] = list(range(len(paths)))
    elif media_path and os.path.isdir(media_path):
        files = sorted(
            f for f in os.listdir(media_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        paths = [os.path.join(media_path, f) for f in files]
        secs = [(_frame_second(f) if _frame_second(f) is not None else i) for i, f in enumerate(files)]
    elif media_path:
        paths, secs = [media_path], [0]
    else:
        raise ValueError("media_type 'images' needs media_path (file/dir) or media_paths (list)")

    # model-knob: temporal downsample (frames are 1 per second)
    if dump_fps and dump_fps < 1.0:
        step = max(int(round(1.0 / dump_fps)), 1)
        paths, secs = paths[::step], secs[::step]

    # cap to provider limit by spreading uniformly across the whole clip
    n = len(paths)
    if max_frames and n > max_frames:
        keep = [round(j * (n - 1) / (max_frames - 1)) for j in range(max_frames)]
        paths = [paths[j] for j in keep]
        secs = [secs[j] for j in keep]

    return [
        Frame(index=secs[i], image=Image.open(p).convert("RGB"), seconds=float(secs[i]))
        for i, p in enumerate(paths)
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
