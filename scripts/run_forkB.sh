#!/usr/bin/env bash
# Fork B PAID: coarse-to-fine (hier) selector with gpt-5.5 answerer, vs the Fork-A arms.
# Run ONLY after fb_echo.sh shows hier lifts hit@k over flat top-k on 600/3600s.
# Primary bar = hit@k recall on long bins (accuracy is guessing-contaminated -> secondary sanity only).
set -uo pipefail
cd /workspace/slm-lab
set -a; . ./.env; set +a
export HF_HOME=/workspace/hf
MANIFEST=${MANIFEST:-data/manifest.lvb.frames.local.json}
MODEL=gpt-5.5
K=6; CAP_SEL=3600

echo "===== CF: hier (coarse-to-fine) so400m k=$K, detail high ====="
python3 -m harness.run --manifest $MANIFEST --conditions C \
  --answerer openai --model $MODEL --detail high --selector hier --k $K --max-dump-frames $CAP_SEL \
  --judge --judge-model gpt-4.1 --label CF-hier --out results/fb_CF 2>&1 | tail -20

echo "===== Fork B analysis (all arms: A / knob / U / C-flat / CF-hier) ====="
for MET in hit_at_k accuracy; do
  python3 scripts/analyze_forkA.py \
    results/p0_A/runs.jsonl results/p0_knob/runs.jsonl results/p0_U/runs.jsonl \
    results/p0_C/runs.jsonl results/fb_CF/runs.jsonl \
    --arm-full C-so400m --arm-topk CF-hier --metric $MET --out results/forkB_$MET.json
done
echo "FORKB_DONE"
