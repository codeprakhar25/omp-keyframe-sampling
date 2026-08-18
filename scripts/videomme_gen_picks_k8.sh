#!/usr/bin/env bash
# After embeds_vmm complete: scores + text embeds + topk/OMP k=8 picks (CPU after text encode).
set -euo pipefail
cd /workspace/slm-lab
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/slm-lab
PY=/workspace/lmmsenv/bin/python
MAN=data/manifest.videomme.json
EMB=results/embeds_vmm
mkdir -p results/scores results/embeds_text results/picks_lmmseval results/videomme_logs

# gate: short+medium image embeds present for all 1800 qids
$PY - <<'PY'
import json, os
from pathlib import Path
m=[x for x in json.load(open("data/manifest.videomme.json")) if x["length_bin"] in ("short","medium")]
miss=[x["id"] for x in m if not (Path("results/embeds_vmm")/f"{x['id']}.npz").exists()]
print(f"embeds present {len(m)-len(miss)}/{len(m)}")
if miss:
    raise SystemExit(f"REFUSING: {len(miss)} missing embeds e.g. {miss[:5]}")
# longclip key
import numpy as np
z=np.load(f"results/embeds_vmm/{m[0]['id']}.npz")
assert "longclip" in z.files and "times" in z.files, z.files
print("OK longclip key", z["longclip"].shape)
PY

for BIN in short medium; do
  echo "=== score+text $BIN ==="
  $PY scripts/score_from_embeds.py \
    --manifest "$MAN" --scorer lc --bins "$BIN" \
    --emb-dir "$EMB" --emb-lc-dir "$EMB" \
    --device cuda \
    --out "results/scores/scores_lc_vmm_${BIN}.jsonl" \
    --text-embeds-out "results/embeds_text/text_lc_vmm_${BIN}.npz"
done

for BIN in short medium; do
  echo "=== topk k8 $BIN ==="
  $PY scripts/export_picks_lmmseval.py \
    --from-scores "results/scores/scores_lc_vmm_${BIN}.jsonl" \
    --method topk --k 8 --bin "$BIN" \
    --out "results/picks_lmmseval/picks_topk_lc_vmm_${BIN}_k8.json"

  echo "=== OMP k8 $BIN ==="
  $PY scripts/gen_omp_picks.py \
    --scores "results/scores/scores_lc_vmm_${BIN}.jsonl" \
    --scorer lc --bin "$BIN" --k 8 \
    --embeds-dir "$EMB" --embeds-lc-dir "$EMB" \
    --text-embeds "results/embeds_text/text_lc_vmm_${BIN}.npz" \
    --out "results/picks_lmmseval/picks_omp_lc_vmm_${BIN}_k8.json"
  # gen_omp may write {id:{idx,secs}} — export flat secs for lmms-eval
  $PY scripts/export_picks_lmmseval.py \
    --from-picks "results/picks_lmmseval/picks_omp_lc_vmm_${BIN}_k8.json" \
    --out "results/picks_lmmseval/picks_omp_lc_vmm_${BIN}_k8.json"
done

# coverage gate
$PY - <<'PY'
import json
from pathlib import Path
for meth in ("topk","omp"):
  for b in ("short","medium"):
    p=Path(f"results/picks_lmmseval/picks_{meth}_lc_vmm_{b}_k8.json")
    d=json.load(open(p))
    assert len(d)==900, (p, len(d))
    assert all(len(v)==8 for v in d.values()), p
    print(f"OK {p.name} n=900 k=8")
print("PICKS_READY")
PY
