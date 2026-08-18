#!/usr/bin/env bash
# Resume Video-MME Stage 0 unzip only (download already done).
# Fixed: no xargs -I{} vs find -exec {} clash.
set -euo pipefail

ROOT="${SLM_ROOT:-/workspace/slm-lab}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
SRC="$HF_HOME/videomme_src"
DST="$HF_HOME/videomme"
LOG="$ROOT/results/videomme_stage0.log"
UNZIP_JOBS="${UNZIP_JOBS:-8}"

mkdir -p "$DST/data" "$DST/subtitle" "$ROOT/results"
exec > >(tee -a "$LOG") 2>&1

echo "======== UNZIP RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) jobs=$UNZIP_JOBS ========"
test -d "$ROOT/slmenv"
command -v unzip >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip; }

mapfile -t CHUNKS < <(find "$SRC" -maxdepth 1 -type f -name 'videos_chunked_*.zip' | sort)
echo "chunks=${#CHUNKS[@]}"
ls -lh "$SRC"/videos_chunked_*.zip | awk '{print $5, $9}'

unpack_one() {
  local z="$1"
  local dst="$2"
  local tmp
  tmp=$(mktemp -d "$dst/_uz.XXXXXX")
  unzip -q -o "$z" -d "$tmp"
  # -t target avoids xargs/find {} clash; + batches
  find "$tmp" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) \
    -exec mv -n -t "$dst/data/" {} +
  rm -rf "$tmp"
  echo "done $(basename "$z") videos_now=$(find "$dst/data" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) | wc -l)"
}
export -f unpack_one

# parallel over zip paths; placeholder ZIPPATH not used inside find -exec {}
printf '%s\n' "${CHUNKS[@]}" | xargs -P "$UNZIP_JOBS" -n 1 bash -c 'unpack_one "$1" "'"$DST"'"' _

SUB=$(find "$SRC" -maxdepth 1 -type f -iname 'subtitle*.zip' | head -1 || true)
if [ -n "${SUB:-}" ]; then
  echo "======== UNZIP subtitles ========"
  rm -rf "$DST/subtitle_tmp"
  mkdir -p "$DST/subtitle_tmp"
  unzip -q -o "$SUB" -d "$DST/subtitle_tmp"
  if [ -d "$DST/subtitle_tmp/subtitle" ]; then
    find "$DST/subtitle_tmp/subtitle" -type f -exec mv -n {} "$DST/subtitle/" \;
  else
    find "$DST/subtitle_tmp" -type f -exec mv -n {} "$DST/subtitle/" \;
  fi
  rm -rf "$DST/subtitle_tmp"
fi

echo "======== PREFLIGHT ========"
python3 -m pip install -q datasets pyarrow 2>&1 | tail -2
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

ROOT="$ROOT" python3 - <<'PY'
import json, os, sys
from pathlib import Path
from datasets import load_dataset

HF_HOME = Path(os.environ["HF_HOME"])
DST = HF_HOME / "videomme" / "data"
ROOT = Path(os.environ["ROOT"])
local = {p.stem for p in DST.iterdir() if p.suffix.lower() in {".mp4", ".mkv"}}
ds = load_dataset("lmms-lab/Video-MME", split="test")
col = "videoID" if "videoID" in ds.column_names else "video_id"
ids = set(ds[col])
missing = sorted(ids - local)
extra = sorted(local - ids)
print(f"parquet_videoIDs={len(ids)} local_videos={len(local)} missing={len(missing)} extra={len(extra)}")
if missing[:10]:
    print("missing_sample", missing[:10])
out = {"parquet_n": len(ids), "local_n": len(local), "missing": len(missing), "extra": len(extra), "missing_sample": missing[:20]}
(ROOT / "results/videomme_stage0_preflight.json").write_text(json.dumps(out, indent=2))
if missing:
    print("FATAL: join incomplete")
    sys.exit(6)
print("JOIN_OK")
PY

echo "======== ARCHIVE SIZE before delete ========"
du -sh "$SRC"
find "$SRC" -maxdepth 1 -type f -name 'videos_chunked_*.zip' -printf '%s\n' \
  | awk '{s+=$1; n++} END{printf "zip_count=%d zip_GiB=%.1f\n", n, s/2^30}'
du -sh "$DST" "$DST/data" "$DST/subtitle"

echo "======== DELETE chunk zips ========"
find "$SRC" -maxdepth 1 -type f -name 'videos_chunked_*.zip' -print -delete
if [ -n "${SUB:-}" ] && [ -f "$SUB" ]; then
  du -sh "$SUB"
  rm -f "$SUB"
  echo "deleted subtitle zip"
fi
echo "after_delete:"
du -sh "$SRC" "$DST"
echo "======== DONE RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
