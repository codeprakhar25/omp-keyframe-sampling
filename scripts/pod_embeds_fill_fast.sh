#!/usr/bin/env bash
# SHARDED embeds fill. Replaces the serial pod_embeds_fill.sh.
#
# Why sharding wins here (measured on this pod 2026-07-15):
#   112 cores, 251G RAM, GPU util 88% -> 0% -> 0% -> 0% between bursts.
#   ffmpeg decode is the bottleneck (~500-590 frames per 600s video) and it is
#   CPU/IO-bound, so the GPU idles ~75% of the time. One serial process leaves
#   111 cores unused. N shards over disjoint items scale near-linearly.
#
# The cap is GPU MEMORY, not CPU: each worker loads SigLIP so400m + DINOv2-large
# + LongCLIP-L (~6.5G of weights) plus activations. Serial run measured 11.9G at
# batch 64. batch 16 shrinks activations so 3 workers fit in 32G with headroom.
# (Jul 14 note: 3-4 fp32 scoring shards OOM'd a 32G card at batch 32 — hence
# batch 16 and 3 workers, not 4.)
#
# Usage (from /workspace/slm-lab):
#   bash scripts/pod_embeds_fill_fast.sh 600      # one bin
#   bash scripts/pod_embeds_fill_fast.sh 60 600   # several, in order
set -euo pipefail

BINS_ARG=("$@")
[ ${#BINS_ARG[@]} -eq 0 ] && BINS_ARG=(15 60 600)

SLM=/workspace/slm-lab
OUT=$SLM/results/embeds
LOGD=$SLM/results/embeds_log
SHARDS=${SHARDS:-3}
BATCH=${BATCH:-16}
mkdir -p "$OUT" "$LOGD"
cd "$SLM"

export HF_HOME=/workspace/hf
export PYTHONPATH=$SLM
PY=$SLM/slmenv/bin/python   # MUST be slmenv: Blackwell sm_120, system torch has no kernels
[ -x "$PY" ] || { echo "FATAL: $PY missing" >&2; exit 1; }
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

manifest_for () { case "$1" in 15|60) echo data/manifest.lvb.full1560.json;; 600) echo data/manifest.lvb.long976.json;; esac; }

for bin in "${BINS_ARG[@]}"; do
  mani=$(manifest_for "$bin")
  echo "=== bin=${bin} : $SHARDS shards x batch $BATCH  $(date -u) ==="
  pids=()
  for s in $(seq 0 $((SHARDS - 1))); do
    "$PY" scripts/dump_embeds.py \
      --manifest "$mani" --bins "$bin" \
      --no-gold-reliable-only --longclip \
      --shard "$s" --num-shards "$SHARDS" --batch-size "$BATCH" \
      --out-dir "$OUT" > "$LOGD/fill_${bin}_s${s}.log" 2>&1 &
    pids+=($!)
    echo "  shard $s pid ${pids[-1]}"
    sleep 20   # stagger model loads: 3 simultaneous loads is the OOM peak
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || { echo "  !! shard pid $p FAILED"; fail=1; }; done
  echo "=== bin=${bin} shards done (fail=$fail) $(date -u) ==="
  grep -h "NO FRAMES" "$LOGD"/fill_${bin}_s*.log 2>/dev/null | head -5 || true
done

echo "=== coverage ==="
"$PY" - << 'PY'
import json, os, collections
import numpy as np
emb = "results/embeds"
have = {f[:-4] for f in os.listdir(emb) if f.endswith(".npz")}
for path in ("data/manifest.lvb.full1560.json", "data/manifest.lvb.long976.json"):
    if not os.path.exists(path):
        continue
    m = json.load(open(path))
    items = m if isinstance(m, list) else m.get("items", [])
    byb = collections.defaultdict(list)
    for it in items:
        byb[str(it.get("length_bin"))].append(it["id"])
    for b, ids in sorted(byb.items()):
        if b.rstrip("s") not in ("15", "60", "600"):
            continue
        n = sum(1 for i in ids if i in have)
        lc = 0
        for i in ids:
            if i in have:
                try:
                    if "longclip" in np.load(f"{emb}/{i}.npz").files:
                        lc += 1
                except Exception:
                    pass
        flag = "" if (n == len(ids) and lc == len(ids)) else "   <-- INCOMPLETE"
        print(f"  bin {b}: {n}/{len(ids)} embeds, {lc}/{len(ids)} longclip{flag}")
PY
echo "DONE $(date -u)"
