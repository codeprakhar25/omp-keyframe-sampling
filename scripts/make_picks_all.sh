#!/usr/bin/env bash
# Build every picks file across every bin that has scores. CPU, seconds.
# Supersedes make_picks.sh: merges 3600s in when it exists, skips it when it doesn't,
# so the overnight chain can call it before OR after the 3600s scores land.
#
# ONE FILE PER (method, scorer, k), SPANNING ALL BINS — forced by the task design:
# longvideobench_val_picks_lc_15s_k8 and ..._3600s_k8 both read $LVB_PICKS_LC_K8.
# The per-bin restriction comes from each task's process_docs filter.
set -uo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
PY=$SLM/slmenv/bin/python
S=$SLM/results/scores
E=$SLM/results/embeds_text
P=$SLM/results/picks_lmmseval
mkdir -p "$P"
KS="8 16 32 64"

for sc in lc sig; do
  parts=(); tparts=()
  for tag in 1560 600 3600; do
    [ -s "$S/scores_${sc}_${tag}.jsonl" ] && { parts+=("$S/scores_${sc}_${tag}.jsonl"); tparts+=("$E/text_${sc}_${tag}.npz"); }
  done
  [ ${#parts[@]} -gt 0 ] || { echo "no scores for $sc — skipping"; continue; }
  cat "${parts[@]}" > "$S/scores_${sc}_all.jsonl"
  echo "merged $S/scores_${sc}_all.jsonl: $(wc -l < "$S/scores_${sc}_all.jsonl") rows from ${#parts[@]} bins"
  $PY - "$E/text_${sc}_all.npz" "${tparts[@]}" <<'PY'
import sys, numpy as np
out, parts = sys.argv[1], sys.argv[2:]
ids, txt = [], []
for p in parts:
    d = np.load(p); ids.append(d["ids"]); txt.append(d["text"])
ids = np.concatenate(ids); txt = np.concatenate(txt)
assert len(set(ids.tolist())) == len(ids), "duplicate qid across bins"
np.savez(out, ids=ids, text=txt)
print("merged %s: %d qids" % (out, len(ids)))
PY
  for k in $KS; do
    $PY scripts/export_picks_lmmseval.py --from-scores "$S/scores_${sc}_all.jsonl" \
      --method topk --k "$k" --out "$P/picks_${sc}_k${k}.json" 2>&1 | sed "s/^/  topk ${sc} k${k}: /"
    $PY scripts/gen_omp_picks.py --scores "$S/scores_${sc}_all.jsonl" --scorer "$sc" \
      --text-embeds "$E/text_${sc}_all.npz" --k "$k" \
      --out "$P/picks_omp_${sc}_k${k}.json" 2>&1 | sed "s/^/  omp ${sc} k${k}: /"
  done
done

echo
echo "=== PICKS FILES ==="
for f in "$P"/*.json; do
  case "$f" in *.meta.json) continue;; esac
  echo "  $(basename "$f")  n=$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f")"
done
