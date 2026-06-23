#!/usr/bin/env bash
# SadTalker: still image + audio -> talking-head video.
# Run from avatar-lab/ root. 6GB-tuned: size 256 + gfpgan enhancer, --still.
#
# Usage:
#   scripts/run_sadtalker.sh data/photos/profile.jpeg out/test.wav out/
#
set -euo pipefail

IMG="${1:?source image}"
AUD="${2:?driven audio wav}"
OUTDIR="${3:-out}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/envs/sadtalker/bin/python"
ST="$ROOT/tools/SadTalker"

# absolutize before cd
IMG="$(readlink -f "$IMG")"
AUD="$(readlink -f "$AUD")"
OUTDIR="$(readlink -f "$OUTDIR")"

# inference.py resolves ./checkpoints + ./gfpgan relative to its own dir
cd "$ST"
"$PY" inference.py \
  --source_image "$IMG" \
  --driven_audio "$AUD" \
  --result_dir "$OUTDIR" \
  --preprocess full \
  --still \
  --size 256 \
  --enhancer gfpgan
