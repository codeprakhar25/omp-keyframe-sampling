#!/usr/bin/env bash
set -euo pipefail
cd /workspace/slm-lab
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1
PY=/workspace/lmmsenv/bin/python
echo "======== Video-MME Stage1 VERIFY $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
$PY - <<'PY'
import json, os, yaml, random
from pathlib import Path
from collections import Counter

pf=json.load(open("results/videomme_stage0_preflight.json"))
assert pf["missing"]==0 and pf["parquet_n"]==900 and pf["local_n"]==900
print("OK videos join 900/900 (stage0 preflight)")

m=json.load(open("data/manifest.videomme.json"))
assert len(m)==2700
assert Counter(x["length_bin"] for x in m)=={"short":900,"medium":900,"long":900}
assert len({x["id"] for x in m})==2700
# sample media exists (full 2700 isfile is slow on network vol)
rng=random.Random(0)
for it in rng.sample(m, 30):
    assert os.path.isfile(it["media_path"]), it["media_path"]
bad=[x["id"] for x in m if "Options:" in (x.get("question_stem") or "")]
assert not bad
sm=[x for x in m if x["length_bin"] in ("short","medium")]
print(f"OK manifest 2700; short+med={len(sm)}q/{len({x['videoID'] for x in sm})}vid; media sample ok")

src=Path("scripts/dump_embeds.py").read_text()
assert "def _norm_bin" in src and "bins = set(_norm_bin(b) for b in args.bins)" in src
ns={}; exec(src[src.index("def _norm_bin"):src.index("\ndef main")], ns)
assert ns["_norm_bin"]("short")=="short" and ns["_norm_bin"]("600s")=="600"
print("OK dump_embeds _norm_bin")

assert Path("Long-CLIP/model/longclip.py").is_file()
vmm=Path("/workspace/lmms-eval/lmms_eval/tasks/videomme")
assert (vmm/"picks_utils.py").is_file()
need=["videomme_picks_omp_lc_short_k8","videomme_picks_omp_lc_medium_k8",
      "videomme_picks_topk_lc_short_k8","videomme_picks_topk_lc_medium_k8",
      "videomme_uniform_k8_short","videomme_uniform_k8_medium"]
for t in need:
    yp=vmm/f"{t}.yaml"
    cfg=yaml.safe_load("".join(l for l in yp.read_text().splitlines(True) if "!function" not in l))
    assert cfg["task"]==t and cfg["dataset_kwargs"]["cache_dir"]=="videomme"
# ast-parse picks_utils (no lmms_eval import — that pulls TaskManager-scale cost)
import ast
ast.parse((vmm/"picks_utils.py").read_text())
pu=(vmm/"picks_utils.py").read_text()
for name in ("filter_short","filter_medium","videomme_doc_to_visual_picks_omp_lc_k8",
             "videomme_doc_to_visual_picks_topk_lc_k8","videomme_doc_to_visual_uniform_k8"):
    assert f"def {name}" in pu, name
print("OK 6 yamls + picks_utils")

for s in ("videomme_launch_embeds.sh","videomme_gen_picks_k8.sh","videomme_run_eval.sh","build_videomme_manifest.py"):
    assert Path("scripts",s).is_file(), s
print("OK scripts")
print("======== ALL GATES GREEN — PREP READY ========")
print("When GPUs free:  bash scripts/videomme_launch_embeds.sh")
print("Then:            bash scripts/videomme_gen_picks_k8.sh")
print("Then eval:       scripts/videomme_run_eval.sh")
PY
