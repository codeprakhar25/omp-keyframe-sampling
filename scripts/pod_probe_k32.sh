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

# vLLM 0.25 ships CUDA-13 extensions; the .so lives in the venv's nvidia wheels
# but isn't on the default loader path — expose every nvidia lib dir explicitly.
NV_LIBS=$(ls -d "$(dirname "$(command -v python3)")"/../lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"   # gpt_mcqa.py imports harness.* from repo root

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
K="${K:-32}"
N="${N:-100}"
WORKERS="${WORKERS:-8}"
BINS="${BINS:-60 600}"
SCORES="${SCORES:-results/scores/scores.jsonl}"
LOG=results/probe32
mkdir -p "$LOG"

VLLM_PID=""
if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "== reusing already-healthy vLLM server on :$PORT =="
else
    echo "== starting vLLM server: $MODEL =="
    # first boot: flashinfer autotune + CUDA graph capture can exceed vLLM's default
    # 600s engine-ready timeout on this card — give it 30 min
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    # flashinfer's JIT arch check doesn't recognize Blackwell sm_120 and hard-fails
    # in the sampler ("requires sm75 or higher"); torch sampler is fine
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export TORCH_CUDA_ARCH_LIST=12.0
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
        if [ -n "$VLLM_PID" ]; then
            echo "killing vLLM PID $VLLM_PID"
            kill "$VLLM_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    # First boot can take >10 min (flashinfer autotune + CUDA graph capture);
    # 20-min budget, and HARD ABORT if still down — never run arms against a dead port.
    echo "== waiting for server health =="
    UP=0
    for i in $(seq 1 360); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "server up after ~$((i * 5))s"
            UP=1
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM died during startup — tail of log:" >&2
            tail -20 "$LOG/vllm.log" >&2
            exit 1
        fi
        sleep 5
    done
    if [ "$UP" != 1 ]; then
        echo "server never became healthy within 30 min — aborting" >&2
        exit 1
    fi
fi

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY=dummy

for BIN in $BINS; do
    for COND in blind uniform topk; do
        OUT="results/mcqa_probe32_${BIN}_${COND}.json"
        # skip only GOOD results (n>0) — a crashed/portless run writes n=0 and must rerun
        if [ -f "$OUT" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT'))['n'] > 0 else 1)" 2>/dev/null; then
            echo "== skip $OUT (exists, n>0) =="
            continue
        fi
        echo "== bin=$BIN cond=$COND k=$K =="
        python3 scripts/gpt_mcqa.py \
            --scores "$SCORES" \
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
