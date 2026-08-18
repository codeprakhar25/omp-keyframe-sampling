#!/usr/bin/env bash
# Build every picks file the lmms-eval matrix can consume. CPU, seconds.
#
# k costs NOTHING at pick time -- it is an argsort cutoff over scores that already
# exist -- so every k is emitted now and the run plan just chooses arms. Same for OMP:
# the inputs (stem text embeds + cached image embeds) are already on disk.
#
# ONE FILE PER (method, scorer, k), SPANNING ALL BINS. This is forced by the task
# design, not a shortcut: longvideobench_val_picks_lc_15s_k8 and ..._600s_k8 both read
# $LVB_PICKS_LC_K8. The per-bin restriction comes from the task's process_docs filter,
# so the picks file must cover every bin that any task reading that env var evaluates.
#
#   bash scripts/make_picks.sh
set -euo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
PY=$SLM/slmenv/bin/python
S=$SLM/results/scores
E=$SLM/results/embeds_text
P=$SLM/results/picks_lmmseval
mkdir -p "$P"

KS="8 16 32"

# --- merge per-scorer: one scores file + one text-embed file covering all bins ------
for sc in lc sig; do
  cat "$S/scores_${sc}_1560.jsonl" "$S/scores_${sc}_600.jsonl" > "$S/scores_${sc}_all.jsonl"
  echo "merged $S/scores_${sc}_all.jsonl: $(wc -l < "$S/scores_${sc}_all.jsonl") rows"
  $PY - "$E/text_${sc}_1560.npz" "$E/text_${sc}_600.npz" "$E/text_${sc}_all.npz" <<'PY'
import sys, numpy as np
a, b, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
ids = np.concatenate([a["ids"], b["ids"]])
txt = np.concatenate([a["text"], b["text"]])
assert len(set(ids.tolist())) == len(ids), "duplicate qid across bins"
np.savez(out, ids=ids, text=txt)
print("merged %s: %d qids" % (out, len(ids)))
PY
done

# --- top-k picks --------------------------------------------------------------------
for sc in lc sig; do
  for k in $KS; do
    $PY scripts/export_picks_lmmseval.py \
      --from-scores "$S/scores_${sc}_all.jsonl" --method topk --k "$k" \
      --out "$P/picks_${sc}_k${k}.json" | sed "s/^/  topk ${sc} k${k}: /"
  done
done

# --- OMP picks ----------------------------------------------------------------------
for sc in lc sig; do
  for k in $KS; do
    $PY scripts/gen_omp_picks.py \
      --scores "$S/scores_${sc}_all.jsonl" --scorer "$sc" \
      --text-embeds "$E/text_${sc}_all.npz" --k "$k" \
      --out "$P/picks_omp_${sc}_k${k}.json" | sed "s/^/  omp ${sc} k${k}: /"
  done
done

echo
echo "=== PICKS FILES ==="
for f in "$P"/*.json; do
  case "$f" in *.meta.json) continue;; esac
  n=$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f")
  echo "  $(basename "$f")  n=$n"
done
