"""MCP server: visual evidence compression for agents.

Exposes one tool, `select_evidence`: given a video / directory of screenshots and a
question, score every candidate frame with SigLIP-so400m (the best selector from
FINDINGS.md §S2: hit@k 0.90 on n=20 stitched GUI needles, 13.5x token reduction at
equal gpt-4.1 accuracy) and save the top-k frames as PNGs. The calling agent then
sends only those k frames to its own model instead of the full dump.

Run:
    pip install "mcp[cli]"          # plus the harness deps (torch, transformers, ...)
    python mcp_server.py            # stdio transport

Claude Code registration:
    claude mcp add visual-evidence -- python /path/to/slm-lab/mcp_server.py
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("visual-evidence")

_MODEL_ID = os.environ.get("VE_SELECTOR_MODEL", "google/siglip-so400m-patch14-384")
_state: dict = {}


def _selector():
    """Lazy singleton so the server starts instantly and only pays model load on first call."""
    if "sel" not in _state:
        from harness.selectors import EmbeddingSelector

        _state["sel"] = EmbeddingSelector(model_id=_MODEL_ID)
    return _state["sel"]


@mcp.tool()
def select_evidence(
    media_path: str,
    question: str,
    k: int = 6,
    dump_fps: float = 1.0,
    max_frames: int = 100,
    out_dir: Optional[str] = None,
) -> dict:
    """Pick the k frames of a video or screenshot directory most relevant to a question.

    Use this instead of sending a whole screen recording / frame dump to a vision model:
    it cuts input tokens ~13x at equal answer accuracy on GUI/video QA.

    Args:
        media_path: video file (.mp4/.mov/...), directory of images, or single image.
        question: what you want to answer from the media; scoring is question-conditioned.
        k: how many frames to keep.
        dump_fps: candidate sampling rate for videos (frames/second).
        max_frames: cap on candidate frames considered.
        out_dir: where to write the selected PNGs (default: a fresh temp dir).

    Returns:
        {frames: [{path, frame_index, seconds, score}], n_candidates, model}
        Frames are ordered by original timeline; read/attach the paths to answer.
    """
    from harness.media import load_frames

    if os.path.isdir(media_path):
        item = {"media_type": "images", "media_path": media_path}
    elif os.path.splitext(media_path)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        item = {"media_type": "image", "media_path": media_path}
    else:
        item = {"media_type": "video", "media_path": media_path}

    frames = load_frames(item, dump_fps=dump_fps, max_frames=max_frames)
    if not frames:
        raise ValueError(f"no frames loaded from {media_path}")

    sel = _selector()
    torch = sel.torch
    with torch.no_grad():
        inputs = sel.processor(
            text=[question], images=[f.image for f in frames], return_tensors="pt",
            padding="max_length", max_length=64, truncation=True,
        ).to(sel.device)
        scores = sel.model(**inputs).logits_per_image.squeeze(-1)
    top = torch.topk(scores, min(k, len(frames))).indices.tolist()

    dest = out_dir or tempfile.mkdtemp(prefix="evidence_")
    os.makedirs(dest, exist_ok=True)
    out = []
    for i in sorted(top):  # preserve timeline order
        f = frames[i]
        path = os.path.join(dest, f"frame_{f.index:04d}.png")
        f.image.save(path)
        out.append(
            {
                "path": path,
                "frame_index": f.index,
                "seconds": f.seconds,
                "score": round(float(scores[i]), 4),
            }
        )
    return {"frames": out, "n_candidates": len(frames), "model": _MODEL_ID}


if __name__ == "__main__":
    mcp.run()
