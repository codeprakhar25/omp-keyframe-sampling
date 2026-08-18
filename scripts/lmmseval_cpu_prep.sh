#!/bin/bash
# CPU-side prep that overlaps GPU eval. Idempotent. Safe to run anytime.
# - verify video symlinks for Sig/LC picks
# - preload HF LongVideoBench metadata (annotations) into cache
# - write coverage report
set -euo pipefail

SLM=${SLM:-/workspace/slm-lab}
OUT=${OUT:-$SLM/results/lmmseval}
PY=${PY:-/workspace/lmmsenv/bin/python}
export HF_HOME="${HF_HOME:-/workspace/hf}"
# lmmsenv only — never slmenv
export PYTHONPATH="/workspace/lmms-eval"
mkdir -p "$OUT"

log() { echo "[cpu-prep $(date -u +%H:%M:%S)] $*"; }

log "start"
VIDDIR="$HF_HOME/datasets/longvideobench/videos"
mkdir -p "$VIDDIR"

# ensure all local mp4s are symlinked (cheap)
"$PY" - <<'PY'
import os, json, glob
src="/workspace/slm-lab/data/videos"
dst=os.path.expanduser(os.environ.get("HF_HOME","/workspace/hf")+"/datasets/longvideobench/videos")
os.makedirs(dst, exist_ok=True)
linked=skip=0
for name in os.listdir(src):
    if not name.endswith(".mp4"): continue
    t=os.path.join(dst,name); s=os.path.join(src,name)
    if os.path.lexists(t): skip+=1; continue
    os.symlink(s,t); linked+=1
print(f"symlink linked={linked} skip={skip} dst={len(os.listdir(dst))}")

# coverage: picks qids → video files exist?
miss=[]
ok=0
for p in glob.glob("/workspace/slm-lab/results/picks_lmmseval/*.json"):
    picks=json.load(open(p))
    # qid like Ip9DbdOtqF4_0 → video often Ip9DbdOtqF4.mp4
    for qid in picks:
        base=qid.rsplit("_",1)[0]+".mp4"
        # also try qid itself
        cands=[base, qid+".mp4"]
        if any(os.path.exists(os.path.join(dst,c)) for c in cands):
            ok+=1
        else:
            miss.append((os.path.basename(p), qid, base))
print(f"picks_video_ok≈{ok} miss={len(miss)}")
if miss[:8]:
    print("miss_sample", miss[:8])
open("/workspace/slm-lab/results/lmmseval/cpu_prep_coverage.json","w").write(
    json.dumps({"ok":ok,"miss":len(miss),"miss_sample":miss[:50]}, indent=2)
)
PY

# warm HF dataset annotations (CPU) — helps first lmms-eval task
log "warming HF LongVideoBench validation split metadata"
"$PY" - <<'PY' || log "HF warm failed (non-fatal)"
import os
os.environ.setdefault("HF_HOME","/workspace/hf")
from datasets import load_dataset
ds=load_dataset("longvideobench/LongVideoBench", split="validation")
print("n", len(ds), "cols", ds.column_names[:12])
# duration groups
from collections import Counter
c=Counter(int(str(x).rstrip("s")) for x in ds["duration_group"])
print("bins", dict(c))
# video_path style
print("video_path0", ds[0]["video_path"], "id0", ds[0]["id"])
# check first 20 short videos resolve
import os
vid=os.path.join(os.environ["HF_HOME"],"datasets/longvideobench/videos")
hit=0
for i in range(min(40,len(ds))):
    if int(str(ds[i]["duration_group"]).rstrip("s"))!=15: continue
    p=os.path.join(vid, ds[i]["video_path"])
    # video_path may already include videos/ prefix or just filename
    alts=[p, os.path.join(vid, os.path.basename(ds[i]["video_path"]))]
    if any(os.path.exists(a) for a in alts): hit+=1
    if hit>=5: break
print("resolved_15s_sample", hit)
PY

log "done → $OUT/cpu_prep_coverage.json"
