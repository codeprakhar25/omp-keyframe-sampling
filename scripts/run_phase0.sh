#!/usr/bin/env bash
# Phase 0 paid arms on the LongVideoBench subset (frames manifest = pre-extracted 1fps JPEGs).
# Answerer gpt-5.5. Primary = MCQA accuracy (+ hit@k diagnostic). Judge = cheap gpt-4.1.
#
# Four operating points on the accuracy-vs-cost curve:
#   A     full-dump           detail low,  <=500 imgs uniform across whole clip (provider wall)
#   knob  model-knob          full-dump @ dump-fps 0.5, detail low (provider's own cheap knob)
#   U     uniform-k           k frames evenly spaced, detail high (dumb cheap baseline)
#   C     so400m top-k        k frames by SigLIP similarity, detail high (smart cheap)
# The selector only earns its keep if C beats BOTH U and knob at equal-or-lower cost.
set -uo pipefail
cd /workspace/slm-lab
set -a; . ./.env; set +a
MANIFEST=${MANIFEST:-data/manifest.lvb.frames.local.json}
MODEL=gpt-5.5
CAP_A=500     # gpt-5.5 hard limit = 500 images/request -> full-dump's real token wall
CAP_SEL=3600  # selectors see ALL frames (whole video) then pick k -> compression's advantage
K=6

echo "===== A: full-dump (detail low, <=500 imgs, uniform-to-cap) ====="
python3 -m harness.run --manifest $MANIFEST --conditions A \
  --answerer openai --model $MODEL --detail low --max-dump-frames $CAP_A \
  --judge --judge-model gpt-4.1 --label A --out results/p0_A 2>&1 | tail -20

echo "===== knob: model-knob full-dump @ 0.5fps detail low ====="
python3 -m harness.run --manifest $MANIFEST --conditions A \
  --answerer openai --model $MODEL --detail low --dump-fps 0.5 --max-dump-frames $CAP_A \
  --judge --judge-model gpt-4.1 --label knob --out results/p0_knob 2>&1 | tail -20

echo "===== U: uniform k=$K over whole video (detail high) ====="
python3 -m harness.run --manifest $MANIFEST --conditions C \
  --answerer openai --model $MODEL --detail high --selector uniform --k $K --max-dump-frames $CAP_SEL \
  --judge --judge-model gpt-4.1 --label U --out results/p0_U 2>&1 | tail -20

echo "===== C: so400m top-k k=$K over whole video (detail high) ====="
python3 -m harness.run --manifest $MANIFEST --conditions C \
  --answerer openai --model $MODEL --detail high --selector embedding --k $K --max-dump-frames $CAP_SEL \
  --judge --judge-model gpt-4.1 --label C-so400m --out results/p0_C 2>&1 | tail -20

echo "===== Fork A analysis ====="
for MET in accuracy hit_at_k; do
  python3 scripts/analyze_forkA.py \
    results/p0_A/runs.jsonl results/p0_knob/runs.jsonl results/p0_U/runs.jsonl results/p0_C/runs.jsonl \
    --arm-full A --arm-topk C-so400m --metric $MET --out results/forkA_$MET.json
done
echo "PHASE0_DONE"
