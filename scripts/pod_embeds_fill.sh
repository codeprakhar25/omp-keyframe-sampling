#!/usr/bin/env bash
# Fill the embedding cache so ALL pick-math (OMP / AdaRD / peak-NMS / FOCUS) can
# replay on CPU forever, for every in-scope bin, in BOTH scorer geometries.
#
# Coverage as of 2026-07-15 (before this run):
#   15s   0/189  embeds   <- blocks every pick-math method on the 15s bin
#   60s 172/172  embeds
#   600s 411/412 embeds   <- the 1 missing item aborts a fail-loud batched run
#   and NO npz has longclip embeds at all -> "LongCLIP pick-math" so far has
#   really meant LongCLIP scores on SigLIP geometry.
#
# This pass adds: 15s bin, the missing 600s item, and the longclip key on every
# npz (refill, --longclip default on). One GPU walk, then pick-math is free.
#
# --no-gold-reliable-only is REQUIRED: the default True silently drops text-Qs
# and would leave 15s at ~107/189, re-creating the incomplete-bin trap.
#
# Waits for the gate to release the GPU (vLLM/lmms-eval both want the card).
# Usage (from /workspace/slm-lab):  bash scripts/pod_embeds_fill.sh [wait|now]
set -euo pipefail

MODE="${1:-wait}"
SLM=${SLM:-/workspace/slm-lab}
OUT=$SLM/results/embeds
LOG=$SLM/results/embeds_fill.log
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="${PYTHONPATH:-$SLM}"
# MUST be slmenv: this box is RTX PRO 4500 Blackwell (sm_120) and the system
# python's torch 2.4.1+cu124 has no sm_120 kernels -> any GPU op dies with
# "no kernel image is available for execution on the device". slmenv carries the
# only Blackwell-capable torch. (The existing 585 npz were dumped with it.)
PY=${PY:-$SLM/slmenv/bin/python}
[ -x "$PY" ] || { echo "FATAL: $PY missing — system python3 CANNOT run on this GPU" >&2; exit 1; }
# slmenv's own bin/python finds its nvidia libs via RPATH normally, but be
# explicit — the same libnvrtc-builtins.so.13.0 loader miss bit the gate.
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

cd "$SLM"
mkdir -p "$OUT"

gpu_busy () { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }

if [ "$MODE" = "wait" ]; then
  echo "[$(date -u +%H:%M:%S)] waiting for GPU to free (gate holds it)..." | tee -a "$LOG"
  for i in $(seq 1 240); do   # up to ~40 min
    gpu_busy || { echo "[$(date -u +%H:%M:%S)] GPU free after ~$((i*10))s" | tee -a "$LOG"; break; }
    sleep 10
  done
  if gpu_busy; then
    echo "[$(date -u +%H:%M:%S)] GPU STILL busy after 40min — not starting, would OOM the gate" | tee -a "$LOG"
    exit 1
  fi
fi

run_bin () {
  local bin=$1 mani=$2
  echo "=== embeds bin=${bin} manifest=$(basename "$mani") $(date -u) ===" | tee -a "$LOG"
  "$PY" scripts/dump_embeds.py \
    --manifest "$mani" --bins "$bin" \
    --no-gold-reliable-only --longclip \
    --out-dir "$OUT" 2>&1 | tee -a "$LOG" | tail -3
}

# 15s + 60s live in the full1560 manifest; 600s in long976.
run_bin 15  data/manifest.lvb.full1560.json
run_bin 60  data/manifest.lvb.full1560.json    # refills longclip key on existing npz
run_bin 600 data/manifest.lvb.long976.json     # picks up the 1 missing + longclip refill

echo "=== coverage after fill ===" | tee -a "$LOG"
"$PY" - << 'PY' 2>&1 | tee -a "$LOG"
import json, os, collections
import numpy as np
emb = "results/embeds"
have = {f[:-4] for f in os.listdir(emb) if f.endswith(".npz")}
for name, path in (("15/60", "data/manifest.lvb.full1560.json"),
                   ("600", "data/manifest.lvb.long976.json")):
    if not os.path.exists(path):
        continue
    m = json.load(open(path))
    items = m if isinstance(m, list) else m.get("items", [])
    byb = collections.defaultdict(list)
    for it in items:
        byb[str(it.get("length_bin"))].append(it["id"])
    for b, ids in sorted(byb.items()):
        n = sum(1 for i in ids if i in have)
        lc = sum(1 for i in ids if i in have and "longclip" in np.load(f"{emb}/{i}.npz").files)
        flag = "" if n == len(ids) else "   <-- INCOMPLETE"
        print(f"  bin {b}: {n}/{len(ids)} embeds, {lc}/{len(ids)} with longclip{flag}")
PY
echo "DONE $(date -u)" | tee -a "$LOG"
