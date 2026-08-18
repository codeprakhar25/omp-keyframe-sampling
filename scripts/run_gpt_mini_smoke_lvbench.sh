#!/bin/bash
# LVBench GPT-5-mini smoke: pull ONE video at a time (pod disk ~20G; subset ~26G).
# Usage on pod: bash scripts/run_gpt_mini_smoke_lvbench.sh
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
VID_ROOT=${VID_ROOT:-/workspace/hf/lvbench}
OUT=${OUT:-$SLM/results/gpt_mini/lvbench_omp_k8_smoke50.json}
SMOKE=${SMOKE:-$SLM/data/smoke50.json}
MANI=${MANI:-$SLM/data/manifest.smoke50.json}
PICKS=${PICKS:-$SLM/data/picks_omp_lc_lvbench_k8.json}
MODEL=${MODEL:-gpt-5-mini}
WORKERS=${WORKERS:-4}
PYTHON=${PYTHON:-python3.11}
PATH=/workspace/bin:$PATH

set -a
# shellcheck disable=SC1091
source "$SLM/.azure_backup.env"
# shellcheck disable=SC1091
source "$SLM/.env"
set +a

AZURE_DEST=${AZURE_DEST%/}
AZURE_SAS=${AZURE_SAS#\?}
mkdir -p "$VID_ROOT" "$(dirname "$OUT")"

$PYTHON - <<'PY' "$SMOKE" "$SLM/data/_smoke_by_video.json"
import json, sys
from collections import defaultdict
smoke = json.load(open(sys.argv[1]))
mani = {it["id"]: it for it in json.load(open(sys.argv[1].replace("smoke50.json", "manifest.smoke50.json")))}
by = defaultdict(list)
for qid in smoke["qids"]:
    by[mani[qid]["video_file"]].append(qid)
out = {v: qs for v, qs in sorted(by.items())}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print(f"videos={len(out)} questions={sum(len(v) for v in out.values())}")
PY

BYV=$SLM/data/_smoke_by_video.json
mapfile -t VIDS < <($PYTHON -c "import json; print('\n'.join(json.load(open('$BYV'))))")

n=0
CKPT="${OUT}.ckpt.jsonl"
for vid in "${VIDS[@]}"; do
  n=$((n+1))
  QJSON=$SLM/data/_qids_cur.json
  # skip video if all its qids already have a parseable answer in ckpt
  SKIP=$($PYTHON - "$BYV" "$vid" "$QJSON" "$CKPT" <<'PY'
import json, sys
from pathlib import Path
by, vid, out, ckpt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
qids = json.load(open(by))[vid]
json.dump(qids, open(out, "w"))
done = {}
if Path(ckpt).is_file():
    for line in open(ckpt):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("pred") and r["pred"] != "?" and (r.get("raw") or "").strip() and not r.get("error"):
            done[r["id"]] = r
pending = [q for q in qids if q not in done]
print("SKIP" if not pending else f"NEED {len(pending)}/{len(qids)}")
PY
)
  echo "===== [$n/${#VIDS[@]}] $vid  $SKIP ====="
  if [ "$SKIP" = "SKIP" ]; then
    continue
  fi
  dest="$VID_ROOT/$vid"
  if [ ! -s "$dest" ]; then
    /workspace/bin/azcopy copy \
      "${AZURE_DEST}/hf/lvbench/${vid}?${AZURE_SAS}" \
      "$dest" --overwrite=true
  fi
  # lmms-eval parity: --bench lvbench, NO --timestamps
  $PYTHON "$SLM/scripts/gpt_mini_lvbench.py" \
    --manifest "$MANI" \
    --picks "$PICKS" \
    --video-root "$VID_ROOT" \
    --bench "${BENCH:-lvbench}" \
    --qids "$QJSON" \
    --model "$MODEL" \
    --effort "${EFFORT:-low}" \
    --max-tokens "${MAX_TOKENS:-1024}" \
    --workers "$WORKERS" \
    --env-file "$SLM/.env" \
    --out "$SLM/results/gpt_mini/_partial.json" \
    --ckpt "$CKPT"
  rm -f "$dest"
  df -h / | tail -1
done

# rebuild full summary from ckpt + smoke50 qids
$PYTHON - "$SMOKE" "$CKPT" "$OUT" <<'PY'
import json, sys
qids = json.load(open(sys.argv[1]))["qids"]
done = {}
for line in open(sys.argv[2]):
    if line.strip():
        r = json.loads(line)
        done[r["id"]] = r
recs = [done[q] for q in qids if q in done]
scored = [r for r in recs if not r.get("error")]
n = len(scored)
correct = sum(1 for r in scored if r.get("ok"))
acc = round(correct / n, 4) if n else None
out = {
    "model": "gpt-5-mini",
    "method": "omp_lc_k8",
    "bench": "lvbench",
    "n_requested": len(qids),
    "n": n,
    "coverage": round(n / len(qids), 4) if qids else 0,
    "accuracy": acc,
    "correct": correct,
    "records": recs,
}
json.dump(out, open(sys.argv[3], "w"), indent=2)
print(f"FINAL acc={acc} n={n}/{len(qids)} coverage={out['coverage']} -> {sys.argv[3]}")
errs = [r for r in recs if r.get("error")]
if errs:
    print(f"errors={len(errs)} e.g. {errs[0].get('error','')[:120]}")
PY
