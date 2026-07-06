#!/usr/bin/env bash
# Fork B close-out: is the long-video localization wall SigLIP, or cheap selection itself?
# Swaps ONLY the scorer, everything else fixed (same manifest, 4 bins, k=6, echo answerer=$0).
# Primary metric hit@k via echo. See research/scorer_swap_spec.md.
#
# Usage: bash scripts/scorer_swap.sh            # runs the free arms (uniform/so400m/siglip2)
#        ARM=videoret bash scripts/scorer_swap.sh   # single arm once VideoRetSelector lands
set -uo pipefail
cd /workspace/slm-lab
export HF_HOME=/workspace/hf
MANIFEST=${MANIFEST:-data/manifest.lvb.frames.local.json}
K=6; CAP=3600

run() {  # name  selector  [model]
  local name=$1 sel=$2 model=${3:-}
  echo "===== echo + $name (selector=$sel model=${model:-default}) k=$K ====="
  local margs=""; [ -n "$model" ] && margs="--selector-model $model"
  python3 -m harness.run --manifest "$MANIFEST" --conditions C --answerer echo \
    --selector "$sel" $margs --k $K --max-dump-frames $CAP --label "echo-$name" \
    --out "results/swap_$name" 2>&1 | tail -6
}

if [ -n "${ARM:-}" ]; then
  case "$ARM" in
    videoret) run videoret videoret "${VIDEORET_MODEL:-LanguageBind/LanguageBind_Video_FT}";;
    siglip2)  run siglip2 embedding google/siglip2-so400m-patch14-384;;
    *) echo "unknown ARM=$ARM"; exit 1;;
  esac
else
  run uniform  uniform
  run so400m   embedding                                      # google/siglip-so400m (Fork-A/B incumbent)
  run siglip2  embedding google/siglip2-so400m-patch14-384    # control: better IMAGE encoder, not video
fi

echo "===== hit@k by bin ====="
python3 - <<'PY'
import json, glob, os
rows=[]
for d in sorted(glob.glob("results/swap_*")):
    p=os.path.join(d,"summary_by_bin.json")
    if not os.path.exists(p): continue
    name=os.path.basename(d)[5:]
    bb=json.load(open(p))
    def hit(b):
        arm=next(iter(bb[b])); return bb[b][arm]["hit_at_k"]
    order=sorted(bb, key=lambda x: float(x[:-1]) if x[:-1].replace('.','').isdigit() else 1e9)
    rows.append((name, " ".join(f"{b}:{hit(b):.2f}" for b in order)))
for name,r in rows: print(f"{name:10} {r}")
PY
echo SWAP_DONE
