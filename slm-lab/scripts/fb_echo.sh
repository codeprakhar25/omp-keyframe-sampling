#!/usr/bin/env bash
# Fork B FREE validation: does coarse-to-fine (hier) lift long-video selection recall
# over flat top-k (embedding)? echo answerer = $0 API cost; hit@k is the whole question.
# Compare summary_by_bin.json hit@k across selectors before paying for accuracy.
set -uo pipefail
cd /workspace/slm-lab
export HF_HOME=/workspace/hf
MANIFEST=${MANIFEST:-data/manifest.lvb.frames.local.json}
K=6; CAP=3600

for sel in embedding hier uniform; do
  echo "===== echo + $sel k=$K ====="
  python3 -m harness.run --manifest $MANIFEST --conditions C --answerer echo \
    --selector $sel --k $K --max-dump-frames $CAP --label "echo-$sel" \
    --out results/fb_echo_$sel 2>&1 | tail -6
done

echo "===== hit@k by bin ====="
python3 - <<'PY'
import json
for sel in ["embedding","hier","uniform"]:
    d=json.load(open(f"results/fb_echo_{sel}/summary_by_bin.json"))
    def hit(b):
        arm=next(iter(d[b]))            # one arm per echo run
        return d[b][arm]["hit_at_k"]
    row=" ".join(f"{b}:{hit(b):.2f}" for b in sorted(d,key=lambda x:float(x[:-1])))
    print(f"{sel:10} {row}")
PY
echo FB_ECHO_DONE
