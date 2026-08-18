#!/bin/bash
# Fast thesis-first lmms-eval pipeline (ONE GPU).
# 1) uniform@8 on 15s only → gate vs LDDR 70.9
# 2) if PASS (±1.5pt): kick 600s uniform@8 (+ topk@8), then 60s
# 3) never touches 3600s
# Parallelism: CPU pick-export overlaps; GPU arms sequential (batch_size>1).
set -euo pipefail

LMMS=${LMMS:-/workspace/lmms-eval}
SLM=${SLM:-/workspace/slm-lab}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
NFRAMES=${NFRAMES:-8}
BATCH=${BATCH:-2}                 # VRAM allowing; fall back handled by OOM → rerun bs=1
OUT=${OUT:-$SLM/results/lmmseval}
GATE15=${GATE15:-0.709}           # LDDR Uniform@8 15s
TOL=${TOL:-0.015}                 # ±1.5pt
mkdir -p "$OUT" "$SLM/results/picks_lmmseval" "$OUT/log"

TASKDIR=$LMMS/lmms_eval/tasks/longvideobench
cp -f "$SLM/harness/lmmseval_patch/"*.py "$TASKDIR/"
cp -f "$SLM/harness/lmmseval_patch/"*.yaml "$TASKDIR/"
# pin _i max_num_frames in every i_* yaml
for f in "$TASKDIR"/longvideobench_val_i_*.yaml; do
  sed -i -E "s/max_num_frames:[[:space:]]*[0-9]+/max_num_frames: $NFRAMES/" "$f"
done

# ONE env only: lmmsenv. NEVER prepend slmenv site-packages (shadows torchvision).
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="/workspace/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/slmenv/' | paste -sd: -)
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
# optional belt-and-suspenders; with clean PYTHONPATH, lmmsenv tv has read_video
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-}"
PY=${PY:-/workspace/lmmsenv/bin/python}
MARGS="pretrained=$MODEL,max_num_frames=$NFRAMES,device_map=auto"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

preflight_stack () {
  log "preflight: interpreter + torchvision + max_num_frames wiring"
  "$PY" - <<PY
import sys, os
bad=[p for p in sys.path if "/slmenv/" in p]
assert not bad, f"slmenv still on sys.path: {bad}"
import torch, torchvision
assert hasattr(torchvision.io, "read_video"), (
    f"torchvision {torchvision.__version__} missing read_video — wrong stack"
)
print(f"OK torch={torch.__version__} tv={torchvision.__version__} "
      f"cuda={torch.cuda.is_available()} tv_file={torchvision.__file__}")
# prove model_args max_num_frames lands on simple adapter (force_simple path)
sys.path.insert(0, "/workspace/lmms-eval")
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
# don't load weights — just check __init__ signature default vs our arg path
import inspect
sig = inspect.signature(Qwen3_VL.__init__)
assert "max_num_frames" in sig.parameters, "simple Qwen3_VL missing max_num_frames"
# _build_video_kwargs: fps=None → nframes=max_num_frames
class _Tmp:
    fps=None; total_pixels=None; max_num_frames=int("$NFRAMES")
    min_pixels=1; max_pixels=1
kw = Qwen3_VL._build_video_kwargs(_Tmp())
assert kw.get("nframes") == int("$NFRAMES"), f"nframes wiring broken: {kw}"
print(f"OK max_num_frames→nframes={kw.get('nframes')} (force_simple path)")
PY
}

arm_done () {
  # resume: skip if prior successful results exist under OUT/tag
  local tag=$1
  local d="$OUT/$tag"
  [ -d "$d" ] || return 1
  # any results json with non-trivial size, and log without ModuleNotFound/Traceback crash
  local n
  n=$(find "$d" -type f \( -name "*.json" -o -name "*.jsonl" \) -size +200c 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || return 1
  if [ -f "$OUT/log/${tag}.log" ] && grep -qE "ModuleNotFoundError|Traceback \(most recent call last\)" "$OUT/log/${tag}.log"; then
    # crashed log — only trust if results clearly complete AND log also has success marker
    grep -qE "longvideobench.*acc|Results saved|Evaluation complete|DONE" "$OUT/log/${tag}.log" || return 1
  fi
  return 0
}

run_arm () {
  local task=$1 tag=$2 bs=${3:-$BATCH}
  if arm_done "$tag"; then
    log "SKIP $tag (results already present)"
    return 0
  fi
  log "RUN $tag task=$task bs=$bs nframes=$NFRAMES force_simple=1"
  local extra_env=()
  [ -n "${FORCE_QWENVL_VIDEO_READER}" ] && extra_env+=(FORCE_QWENVL_VIDEO_READER="$FORCE_QWENVL_VIDEO_READER")
  if ! env "${extra_env[@]}" "$PY" -m lmms_eval --model qwen3_vl --force_simple \
      --model_args "$MARGS" \
      --tasks "$task" --batch_size "$bs" \
      --output_path "$OUT/$tag" --log_samples \
      >"$OUT/log/${tag}.log" 2>&1; then
    if [ "$bs" -gt 1 ]; then
      log "OOM/fail bs=$bs → retry bs=1"
      env "${extra_env[@]}" "$PY" -m lmms_eval --model qwen3_vl --force_simple \
        --model_args "$MARGS" \
        --tasks "$task" --batch_size 1 \
        --output_path "$OUT/$tag" --log_samples \
        >"$OUT/log/${tag}.log" 2>&1
    else
      log "FAIL $tag"; return 1
    fi
  fi
  log "DONE $tag"
}

# parse per-bin acc from lmms-eval samples / results json
# prints e.g. 0.709 or empty
bin_acc () {
  local tag=$1 bin=$2
  "$PY" - "$OUT" "$tag" "$bin" <<'PY'
import json, glob, os, sys
out, tag, bin_s = sys.argv[1], sys.argv[2], sys.argv[3]
bin_i = int(str(bin_s).rstrip("s"))
acc = None

def walk(o):
    found = None
    if isinstance(o, dict):
        for k, v in o.items():
            ks = str(k).rstrip("s")
            if ks == str(bin_i) and isinstance(v, dict) and "acc" in v:
                return float(v["acc"])
        for v in o.values():
            found = walk(v)
            if found is not None:
                return found
    elif isinstance(o, list):
        for x in o:
            found = walk(x)
            if found is not None:
                return found
    return None

cands = sorted(glob.glob(os.path.join(out, tag, "**", "*.json"), recursive=True))
for p in cands:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    acc = walk(d)
    if acc is not None:
        break

if acc is None:
    samples = sorted(glob.glob(os.path.join(out, tag, "**", "*samples*.json*"), recursive=True))
    for p in samples:
        try:
            if p.endswith(".jsonl"):
                rows = [json.loads(l) for l in open(p) if l.strip()]
            else:
                obj = json.load(open(p))
                rows = obj if isinstance(obj, list) else obj.get("samples") or obj.get("data") or []
        except Exception:
            continue
        ok = tot = 0
        for r in rows:
            doc = r.get("doc") or r.get("target_doc") or r
            g = doc.get("duration_group") if isinstance(doc, dict) else None
            if g is None:
                g = r.get("duration_group")
            try:
                g = int(str(g).rstrip("s"))
            except Exception:
                continue
            if g != bin_i:
                continue
            tot += 1
            correct = r.get("exact_match") or r.get("lvb_acc")
            if isinstance(correct, dict):
                pred, gold = correct.get("parsed_pred"), correct.get("answer")
                hit = (pred == gold) if pred and gold else None
            elif isinstance(correct, (int, float, bool)):
                hit = bool(correct)
            else:
                hit = r.get("correct")
            if hit is None:
                continue
            ok += int(hit)
        if tot:
            acc = ok / tot
            break

# also scrape arm log for printed aggregate like "'15': {'acc': 0.709"
if acc is None:
    logp = os.path.join(out, "log", f"{tag}.log")
    if os.path.isfile(logp):
        import re
        txt = open(logp, errors="ignore").read()
        m = re.search(rf"['\"]?{bin_i}['\"]?\s*:\s*\{{[^}}]*'acc'\s*:\s*([0-9.]+)", txt)
        if not m:
            m = re.search(rf"['\"]?{bin_i}['\"]?\s*:\s*\{{[^}}]*\"acc\"\s*:\s*([0-9.]+)", txt)
        if m:
            acc = float(m.group(1))

if acc is None:
    sys.exit(0)
print(f"{acc:.6f}")
PY
}

export_picks () {
  local scores=$1 k=$2 out=$3 bin=${4:-}
  local args=(--from-scores "$scores" --method topk --k "$k" --out "$out")
  [ -n "$bin" ] && args+=(--bin "$bin")
  "$PY" "$SLM/scripts/export_picks_lmmseval.py" "${args[@]}"
}

rm -f "$OUT/PIPELINE_COMPLETE" "$OUT/GATE15_ACC.txt" "$OUT/GATE15_PASS"

preflight_stack

log "=== CPU prep: export SigLIP + LongCLIP topk picks (LDDR uses LongCLIP) ==="
export_picks "$SLM/results/scores/scores_1560_full.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_15.json" 15
export_picks "$SLM/results/scores/scores_1560_full.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_60.json" 60
export_picks "$SLM/results/scores/scores_600full_sig_all.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_600.json" 600
# LongCLIP = LDDR's encoder — head-to-head with SigLIP under same lmms-eval pathway
export_picks "$SLM/results/scores/scores_longclip_full1560.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_15.json" 15
export_picks "$SLM/results/scores/scores_longclip_full1560.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_60.json" 60
export_picks "$SLM/results/scores/scores_600full_lc_all.jsonl" "$NFRAMES" \
  "$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_600.json" 600

log "=== GATE: uniform@${NFRAMES} native-video on 15s only (target ${GATE15}) ==="
run_arm longvideobench_val_v_15s "gate15_uniform${NFRAMES}_v"
ACC15=$(bin_acc "gate15_uniform${NFRAMES}_v" 15 || true)
log "15s uniform@${NFRAMES}_v acc=${ACC15:-MISSING}"
if [ -z "${ACC15}" ]; then
  log "Could not parse 15s acc — see $OUT/gate15_uniform${NFRAMES}_v and log/"
  exit 1
fi
# durable gate artifact — watcher requires this (not log "DONE")
printf "%s\n" "$ACC15" > "$OUT/GATE15_ACC.txt"
"$PY" - <<PY
acc=float("$ACC15"); gate=float("$GATE15"); tol=float("$TOL")
ok = abs(acc-gate) <= tol
print(f"GATE15 {'PASS' if ok else 'FAIL'}: {acc:.3f} vs {gate:.3f} ±{tol}")
raise SystemExit(0 if ok else 3)
PY
GATE_RC=$?

if [ "$GATE_RC" -ne 0 ]; then
  log "15s FAIL — NOT kicking 600s. Fix pathway first."
  exit 3
fi
touch "$OUT/GATE15_PASS"

log "=== 15s PASS → kick 600s uniform + SigLIP topk + LongCLIP topk (no 3600) ==="
run_arm longvideobench_val_v_600s "b600_uniform${NFRAMES}_v"
export LVB_PICKS="$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_600.json"
run_arm longvideobench_val_picks_600s "b600_topk${NFRAMES}_sig"
export LVB_PICKS="$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_600.json"
run_arm longvideobench_val_picks_600s "b600_topk${NFRAMES}_lc"

log "=== finish short bins: 60s uniform_v + 15/60 Sig+LC topk + fair uniform_i ==="
run_arm longvideobench_val_v_60s "b60_uniform${NFRAMES}_v"
run_arm longvideobench_val_i_15s "b15_uniform${NFRAMES}_i"
run_arm longvideobench_val_i_60s "b60_uniform${NFRAMES}_i"
export LVB_PICKS="$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_15.json"
run_arm longvideobench_val_picks_15s "b15_topk${NFRAMES}_sig"
export LVB_PICKS="$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_15.json"
run_arm longvideobench_val_picks_15s "b15_topk${NFRAMES}_lc"
export LVB_PICKS="$SLM/results/picks_lmmseval/sig_topk${NFRAMES}_60.json"
run_arm longvideobench_val_picks_60s "b60_topk${NFRAMES}_sig"
export LVB_PICKS="$SLM/results/picks_lmmseval/lc_topk${NFRAMES}_60.json"
run_arm longvideobench_val_picks_60s "b60_topk${NFRAMES}_lc"

log "=== SUMMARY (SigLIP vs LongCLIP under lmms-eval) ==="
for tag_bin in \
  "gate15_uniform${NFRAMES}_v:15" \
  "b60_uniform${NFRAMES}_v:60" \
  "b600_uniform${NFRAMES}_v:600" \
  "b15_uniform${NFRAMES}_i:15" \
  "b60_uniform${NFRAMES}_i:60" \
  "b15_topk${NFRAMES}_sig:15" \
  "b15_topk${NFRAMES}_lc:15" \
  "b60_topk${NFRAMES}_sig:60" \
  "b60_topk${NFRAMES}_lc:60" \
  "b600_topk${NFRAMES}_sig:600" \
  "b600_topk${NFRAMES}_lc:600"
do
  tag=${tag_bin%%:*}; b=${tag_bin##*:}
  a=$(bin_acc "$tag" "$b" || true)
  printf "  %-28s bin=%-4s acc=%s\n" "$tag" "$b" "${a:-?}"
done
# watcher DONE gate: requires this file + GATE15_ACC.txt (parsed number)
date -u +"PIPELINE_COMPLETE %Y-%m-%dT%H:%M:%SZ" > "$OUT/PIPELINE_COMPLETE"
log "PIPELINE_COMPLETE (gate15_acc=$(cat "$OUT/GATE15_ACC.txt"))"
