#!/usr/bin/env bash
# Clean STEM-ONLY rescore, all bins, both scorers, off CACHED image embeds.
#
# Costs minutes, not hours: the image towers were never contaminated, so no video is
# re-decoded and no frame is re-encoded. Only ~760 short strings go through a text
# tower. Proven equivalent to the original GPU scorer on 2026-07-15:
#   full-query replay vs archived scores_600full_lc_all.jsonl -> Spearman 0.9999, top8 0.9967
# The same comparison with the stem query gives Spearman 0.838 / top8 0.582, i.e. the
# fix moves 42% of the picked frames at 600s. That gap is the whole point.
#
# Also emits stem text embeds per (scorer,bin) -> OMP needs them and they are free here.
#
#   bash scripts/rescore_clean.sh
set -euo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
export HF_HOME=/workspace/hf
export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
PY=$SLM/slmenv/bin/python
OUT=$SLM/results/scores
EMB=$SLM/results/embeds_text
mkdir -p "$OUT" "$EMB"

run () {  # scorer manifest bins tag
  local sc=$1 mani=$2 bins=$3 tag=$4
  local log="$OUT/../rescore_${sc}_${tag}.log"
  echo "=================== $sc / $tag ==================="
  # FULL output to a log; only a summary to stdout. The first version piped straight
  # into `grep -E <allowlist>`, which swallowed an AttributeError traceback whole and
  # reported success -- the run just silently produced no sig scores. Never filter a
  # stream you have not first written down.
  if ! $PY scripts/score_from_embeds.py \
        --manifest "$mani" --scorer "$sc" --bins $bins \
        --out "$OUT/scores_${sc}_${tag}.jsonl" \
        --text-embeds-out "$EMB/text_${sc}_${tag}.npz" > "$log" 2>&1; then
    echo "!! FAILED ($sc/$tag) — last 15 lines of $log:"
    tail -15 "$log"
    return 1
  fi
  grep -E "manifest |image embeds|option leaks|wrote |encoding " "$log" || true
  # a 0-row output must never pass as success
  local n; n=$(wc -l < "$OUT/scores_${sc}_${tag}.jsonl")
  [ "$n" -gt 0 ] || { echo "!! REFUSING: scores_${sc}_${tag}.jsonl has 0 rows"; return 1; }
}

run lc  data/manifest.lvb.full1560.json "15 60" 1560
run lc  data/manifest.lvb.long976.json  "600"   600
run sig data/manifest.lvb.full1560.json "15 60" 1560
run sig data/manifest.lvb.long976.json  "600"   600

echo
echo "=== SCORE FILES ==="
wc -l "$OUT"/scores_*.jsonl
echo "=== STEM TEXT EMBEDS (OMP inputs) ==="
ls -la "$EMB"/
