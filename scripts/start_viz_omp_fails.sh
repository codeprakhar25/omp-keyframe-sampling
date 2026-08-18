#!/usr/bin/env bash
# Start OMP-fail visual browser on CPU (or any) pod with /workspace network volume.
# Usage: bash /workspace/slm-lab/scripts/start_viz_omp_fails.sh
set -euo pipefail

ROOT="${SLM_ROOT:-/workspace/slm-lab}"
PORT="${PORT:-5902}"
cd "$ROOT"

echo "[viz] root=$ROOT port=$PORT"

# system deps (idempotent)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[viz] installing ffmpeg…"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
fi

python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "$ROOT/scripts/requirements_viz.txt"

mkdir -p "$ROOT/results/viz_cache"
pkill -f viz_omp_fails_server.py 2>/dev/null || true
sleep 1

export SLM_ROOT="$ROOT"
export PORT="$PORT"
nohup python3 "$ROOT/scripts/viz_omp_fails_server.py" \
  > "$ROOT/results/viz_omp_fails.log" 2>&1 &
echo "[viz] pid $!  log=$ROOT/results/viz_omp_fails.log"
sleep 2
curl -s -o /dev/null -w "[viz] HTTP %{http_code} on :$PORT\n" "http://127.0.0.1:${PORT}/" || true
echo "[viz] tunnel from laptop:"
echo "  ssh -i ~/.ssh/runpod -p <CPU_PORT> -N -L ${PORT}:127.0.0.1:${PORT} root@<CPU_HOST>"
echo "  then open http://127.0.0.1:${PORT}/"
