#!/usr/bin/env bash
# k=32 kill-shot probe: does SigLIP top-k still beat uniform at a generous budget,
# with a leaderboard-class local answerer (Qwen3-VL-8B)?
#
# Runs on the pod against the 200 cached-score items (100x60s + 100x600s in
# results/scores/scores.jsonl) — the SigLIP scoring half is already paid, so this
# is answerer-only. Arms per bin: blind / uniform@32 / topk@32 (uniform@32 IS the
# budget-matched control; full==uniform at k=32 so no separate full arm).
#
# Decision rule: topk > uniform at k=32 on the 600s bin -> full Stage A/B run is
# justified. Gap dead -> stop, nothing to scale.
#
# Setup (once per pod restart, overlay env is wiped):
#   export PIP_BREAK_SYSTEM_PACKAGES=1
#   pip install -U vllm            # needs recent vllm (Qwen3-VL support, cu128/sm_120)
#   pip install openai
#
# Usage: bash scripts/pod_probe_k32.sh   (from /workspace/slm-lab)
# Long-run safe: server started via nohup with explicit PID; killed by PID at exit.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
K=32
N=100
WORKERS="${WORKERS:-8}"
LOG=results/probe32
mkdir -p "$LOG"

echo "== starting vLLM server: $MODEL =="
nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 40}' \
    > "$LOG/vllm.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$LOG/vllm.pid"
echo "vLLM PID $VLLM_PID (log: $LOG/vllm.log)"

cleanup() {
    echo "killing vLLM PID $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "== waiting for server health =="
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "server up after ~$((i * 5))s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM died during startup — tail of log:" >&2
        tail -20 "$LOG/vllm.log" >&2
        exit 1
    fi
    sleep 5
done

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY=dummy

for BIN in 60 600; do
    for COND in blind uniform topk; do
        OUT="results/mcqa_probe32_${BIN}_${COND}.json"
        if [ -f "$OUT" ]; then
            echo "== skip $OUT (exists) =="
            continue
        fi
        echo "== bin=$BIN cond=$COND k=$K =="
        python3 scripts/gpt_mcqa.py \
            --bin "$BIN" --cond "$COND" --k "$K" --n "$N" \
            --model "$MODEL" --effort none --workers "$WORKERS" \
            --out "$OUT" 2>&1 | tee "$LOG/run_${BIN}_${COND}.log" | tail -3
    done
done

echo "== probe summary =="
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("results/mcqa_probe32_*.json")):
    d = json.load(open(f))
    print(f"{f}: acc={d['accuracy']} n={d['n']}")
EOF
