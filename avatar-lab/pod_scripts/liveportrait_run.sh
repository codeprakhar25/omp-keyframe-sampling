#!/usr/bin/env bash
# LivePortrait (KwaiVGI) — video-driven reenactment for the TALKING-HEAD base.
# Animates a still keyframe with head/expression motion from a DRIVING clip
# (implicit keypoints + warp + STITCHING -> clean boundary, no paste seam).
# NOT audio-driven: motion comes from the driving video; output length = driving length.
#
# Pairs with a relip stage (audio -> lips). NOTE: LatentSync relip is BROKEN on
# Blackwell (see README gotchas) -> use SadTalker for the talking-head instead,
# OR run LatentSync on a non-Blackwell GPU.
set -euo pipefail

ROOT=/workspace/avatar
LP=$ROOT/LivePortrait

# --- setup (once) ---
# git clone https://github.com/KwaiVGI/LivePortrait $LP
# python3 -m venv $ROOT/lpenv && source $ROOT/lpenv/bin/activate
# pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# # GOTCHA: requirements.txt 1st line is "-r requirements_base.txt" -> install FROM the repo dir (cd $LP) so it resolves.
# pip install -r requirements.txt
# pip install --force-reinstall --no-deps torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# # GOTCHA: do NOT pass extra positionals to `hf download` (they get read as explicit filenames -> fetches junk):
# hf download KwaiVGI/LivePortrait --local-dir $LP/pretrained_weights
# apt-get install -y ffmpeg   # overlay wiped on pod restart

# --- run ---
source $ROOT/lpenv/bin/activate
cd "$LP"
SRC="${1:-$ROOT/wan_in/key_threeq_s0.png}"     # still keyframe (FLUX-LoRA)
DRV="${2:-$ROOT/wan_in/pers.mp4}"              # driving clip (any natural head motion; 12s -> 12s out)
python inference.py -s "$SRC" -d "$DRV" --output-dir "$ROOT/lp_out"
# onnx face-detect/landmark fall to CPU (libcublasLt.so.11 cu11-vs-cu12) = harmless; warp on GPU.
# Out: lp_out/<src>--<drv>.mp4  (+_concat side-by-side). identity/glasses HOLD, clean boundary.
