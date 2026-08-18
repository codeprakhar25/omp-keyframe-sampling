#!/usr/bin/env bash
# Video-MME Stage 0 — download + unzip onto network volume (CPU pod).
# Rules: no git; kill by PID only; never echo HF_TOKEN; never touch slmenv.
set -euo pipefail

ROOT="${SLM_ROOT:-/workspace/slm-lab}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
SRC="$HF_HOME/videomme_src"
DST="$HF_HOME/videomme"
LOG="$ROOT/results/videomme_stage0.log"
NPROC="${NPROC:-$(nproc)}"
UNZIP_JOBS="${UNZIP_JOBS:-8}"   # parallel unzip after first-zip layout check

mkdir -p "$ROOT/results" "$SRC" "$DST"
exec > >(tee -a "$LOG") 2>&1

echo "======== Video-MME Stage 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
echo "HF_HOME=$HF_HOME SRC=$SRC DST=$DST nproc=$NPROC unzip_jobs=$UNZIP_JOBS"

# token — never print value
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ROOT/.env"
  set +a
fi
: "${HF_TOKEN:?HF_TOKEN missing (load via $ROOT/.env)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1

# protect
test -d "$ROOT/slmenv"
test -d /workspace/persist/slm600

# space gate: need ~200G free for peak (zips+extract). df is cluster-wide — use VOLUME_GB.
USED_G=$(du -sm /workspace/slm-lab /workspace/hf /workspace/lmmsenv /workspace/lmms-eval /workspace/persist 2>/dev/null | awk '{s+=$1} END{printf "%d", s/1024}')
echo "workspace_used_approx_GiB=$USED_G (major trees)"
VOL_G="${VOLUME_GB:-300}"
FREE_G=$((VOL_G - USED_G))
echo "assume_volume_GiB=$VOL_G free_approx_GiB=$FREE_G"
if [ "$FREE_G" -lt 190 ]; then
  echo "FATAL: need ~200GiB free for peak. Expand volume to ≥300GB first. (free_approx=$FREE_G)"
  exit 2
fi

# deps
python3 -m pip install -q -U 'pip<25' huggingface_hub hf_transfer 2>&1 | tail -3
command -v unzip >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip; }

echo "======== DOWNLOAD ========"
# resume-safe
huggingface-cli download lmms-lab/Video-MME \
  --repo-type dataset \
  --local-dir "$SRC" \
  --max-workers "$NPROC"

echo "======== SRC LISTING ========"
du -sh "$SRC"
ls -lh "$SRC" | head -50
ls "$SRC"/*.zip 2>/dev/null | wc -l || true

# find chunk zips + subtitle
mapfile -t CHUNKS < <(find "$SRC" -maxdepth 2 -type f -name 'videos_chunked_*.zip' | sort)
SUB=$(find "$SRC" -maxdepth 2 -type f -iname 'subtitle*.zip' | head -1 || true)
echo "chunks=${#CHUNKS[@]} subtitle=${SUB:-NONE}"
if [ "${#CHUNKS[@]}" -lt 1 ]; then
  echo "FATAL: no videos_chunked_*.zip under $SRC"
  find "$SRC" -maxdepth 3 -type f | head -80
  exit 3
fi

echo "======== UNZIP first chunk (layout check) ========"
mkdir -p "$DST/data" "$DST/subtitle"
FIRST="${CHUNKS[0]}"
echo "first=$FIRST"
TMP="$DST/_unpack_tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -q -o "$FIRST" -d "$TMP"
# normalize into data/
if ls "$TMP"/*.mp4 >/dev/null 2>&1 || ls "$TMP"/*.MP4 >/dev/null 2>&1; then
  find "$TMP" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' \) -exec mv -n {} "$DST/data/" \;
elif [ -d "$TMP/data" ]; then
  find "$TMP/data" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) -exec mv -n {} "$DST/data/" \;
else
  find "$TMP" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) -exec mv -n {} "$DST/data/" \;
fi
rm -rf "$TMP"
N1=$(find "$DST/data" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) | wc -l)
echo "after_first_chunk videos_in_data=$N1"
if [ "$N1" -lt 1 ]; then
  echo "FATAL: first zip produced no videos in $DST/data"
  exit 4
fi
ls "$DST/data" | head -5

echo "======== UNZIP remaining chunks (parallel $UNZIP_JOBS) ========"
# NOTE: never use xargs -I{} with find -exec {} — clashes; use -n1 + $1.
REST=("${CHUNKS[@]:1}")
unpack_one() {
  local z="$1" dst="$2" tmp
  tmp=$(mktemp -d "$dst/_uz.XXXXXX")
  unzip -q -o "$z" -d "$tmp"
  find "$tmp" -type f \( -iname '*.mp4' -o -iname '*.mkv' \) \
    -exec mv -n -t "$dst/data/" {} +
  rm -rf "$tmp"
  echo "done $(basename "$z")"
}
export -f unpack_one
if [ "${#REST[@]}" -gt 0 ]; then
  printf '%s\n' "${REST[@]}" | xargs -P "$UNZIP_JOBS" -n 1 bash -c 'unpack_one "$1" "'"$DST"'"' _
fi

if [ -n "${SUB:-}" ]; then
  echo "======== UNZIP subtitles ========"
  unzip -q -o "$SUB" -d "$DST/subtitle_tmp" || unzip -q -o "$SUB" -d "$DST"
  if [ -d "$DST/subtitle_tmp" ]; then
    # flatten into subtitle/
    if [ -d "$DST/subtitle_tmp/subtitle" ]; then
      mv "$DST/subtitle_tmp/subtitle"/* "$DST/subtitle/" 2>/dev/null || true
    else
      find "$DST/subtitle_tmp" -type f -exec mv -n {} "$DST/subtitle/" \;
    fi
    rm -rf "$DST/subtitle_tmp"
  fi
fi

echo "======== PREFLIGHT join (parquet videoID) ========"
python3 -m pip install -q datasets pyarrow 2>&1 | tail -2
ROOT="$ROOT" python3 - <<'PY'
import json, os, sys
from pathlib import Path

HF_HOME = Path(os.environ["HF_HOME"])
DST = HF_HOME / "videomme" / "data"
ROOT = Path(os.environ["ROOT"])

local = {p.stem for p in DST.iterdir() if p.suffix.lower() in {".mp4", ".mkv"}} if DST.exists() else set()

from datasets import load_dataset
ds = load_dataset("lmms-lab/Video-MME", split="test")
col = "videoID" if "videoID" in ds.column_names else ("video_id" if "video_id" in ds.column_names else None)
if col is None:
    print("FATAL cols", ds.column_names)
    sys.exit(5)
ids = set(ds[col])
missing = sorted(ids - local)
extra = sorted(local - ids)
print(f"parquet_videoIDs={len(ids)}")
print(f"local_videos={len(local)}")
print(f"missing={len(missing)} extra={len(extra)}")
if missing[:10]:
    print("missing_sample", missing[:10])
out = {
    "parquet_n": len(ids),
    "local_n": len(local),
    "missing": len(missing),
    "extra": len(extra),
    "missing_sample": missing[:20],
}
(ROOT / "results/videomme_stage0_preflight.json").write_text(json.dumps(out, indent=2))
if missing:
    print("FATAL: join incomplete")
    sys.exit(6)
print("JOIN_OK")
PY

echo "======== ARCHIVE SIZE before delete ========"
du -sh "$SRC"
du -ch "$SRC"/videos_chunked_*.zip 2>/dev/null | tail -5 || du -ch "$SRC"/**/videos_chunked_*.zip 2>/dev/null | tail -5
find "$SRC" -type f -name 'videos_chunked_*.zip' -printf '%s %p\n' | awk '{s+=$1; n++} END{printf "zip_count=%d zip_bytes=%d zip_GiB=%.1f\n", n, s, s/2^30}'
du -sh "$DST" "$DST/data" "$DST/subtitle"

echo "======== DELETE chunk zips (reclaim) ========"
find "$SRC" -type f -name 'videos_chunked_*.zip' -print -delete
# keep subtitle zip? optional keep small; delete if large
if [ -n "${SUB:-}" ] && [ -f "$SUB" ]; then
  du -sh "$SUB"
  rm -f "$SUB"
  echo "deleted subtitle zip"
fi
echo "after_delete SRC:"
du -sh "$SRC"
du -sh "$DST"

echo "======== DONE Stage 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
echo "HF_HOME=$HF_HOME  data=$DST/data  log=$LOG"
