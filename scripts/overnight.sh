#!/usr/bin/env bash
# UNATTENDED OVERNIGHT CHAIN — 2026-07-15. Runs after the k=8 matrix lands.
#
# Ordered by VALUE, not by wish-list order, because the night is ~6h and the full
# wish-list is ~15h of answerer time. Everything above the line is LDDR-comparable;
# everything below it is an internal curve that compares to no published number.
#
#   #F=8 is the ONLY budget LDDR publishes for Qwen3-VL-8B (Table 1, subtitles off,
#   LongCLIP for every baseline). k=16/32/64 have no published counterpart.
#
#   3600s #F=8: Uniform 45.2 | AKS 51.4 | Q-frame 49.8 | FOCUS 50.2 | MDP3 51.1
#               LD 55.9 | LDDR 56.2      <- biggest selection gain of any bin
#
# Stage order:
#   0  wait for k8.done (15/60/600 @ k=8, already running)
#   1  3600s image embeds   (6 shards x dedup-by-video, no DINO -> ~9x less work)
#   2  3600s scores         (CPU, off those embeds)
#   3  picks, all bins/ks   (CPU, free)
#   4  3600s k=8 arms       <-- completes the LDDR grid across ALL FOUR bins
#   ------------------------------ everything below is bonus -------------------
#   5  60s k=16 arms        (~35m)
#   6  600s k=32 arms       (~2h45m)
#   7  3600s k=32 arms      (~3h45m)      # k=64 omitted: ~7.5h alone, and 64 frames
#                                         # x ~1024 tok risks blowing context anyway
#
# Every stage: .done marker written ONLY after real output is verified. lmms-eval
# exits 0 on a failed eval, so exit status is never the gate (a watcher trusting it
# reported a false DONE on 2026-07-15). Resumable: rerun and finished stages skip.
set -uo pipefail
SLM=/workspace/slm-lab
cd "$SLM"
export HF_HOME=/workspace/hf
export FORCE_QWENVL_VIDEO_READER=decord
NV_LIBS=$(ls -d "$SLM"/slmenv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
PY=$SLM/slmenv/bin/python
PYBIN=/workspace/lmmsenv/bin/python
M=$SLM/results/lmmseval_matrix_clean
ST=$SLM/results/stages
mkdir -p "$ST" "$M"
log () { echo "[$(date -u +%H:%M:%S)] $*"; }

# ------------------------------------------- 0: the PRIMARY k=8 matrix (15/60/600)
# Re-run from scratch: the 2026-07-15 attempt aborted at 1498/2491 on a missing qid,
# so no results survived. Picks are regenerated at 773 (was 772) and preflighted.
while pgrep -f "lmms[_]eval" >/dev/null; do log "another lmms_eval is up; waiting"; sleep 60; done

# -------------------------------------------------- lmms-eval arm runner
P=$SLM/results/picks_lmmseval
run_arms () {   # tag  tasks_csv  env_assignments...
  local tag=$1; shift
  local tasks=$1; shift
  [ -f "$ST/$tag.done" ] && { log "SKIP $tag (done)"; return 0; }
  for kv in "$@"; do export "$kv"; done
  local files=()
  for kv in "$@"; do
    local f=${kv#*=}
    [ -s "$f" ] || { log "!! $tag: ${kv%%=*} -> $f missing/empty — SKIPPING stage"; return 1; }
    files+=("$f")
  done
  # COVERAGE preflight. "non-empty" is not coverage: one absent qid killed the k=8
  # matrix at 60% on 2026-07-15 because picks_utils (correctly) fails loud INSIDE
  # lmms-eval, after the model load and ~70min of GPU. Check before spending it.
  local bin=${tag##*_}
  PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages" \
    $PY scripts/preflight_picks.py "$bin" "${files[@]}" 2>&1 | while read -r l; do log "  $l"; done
  PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages" \
    $PY scripts/preflight_picks.py "$bin" "${files[@]}" >/dev/null 2>&1 \
    || { log "!! $tag: picks do not cover bin ${bin}s — SKIPPING (would die mid-run)"; return 1; }
  log "RUN $tag :: $tasks"
  export PYTHONPATH="/workspace/lmms-eval:$SLM/slmenv/lib/python3.11/site-packages"
  "$PYBIN" -m lmms_eval --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,device_map=auto \
    --tasks "$tasks" --batch_size 1 \
    --output_path "$M/$tag" --log_samples > "$M/$tag.log" 2>&1
  local errs got
  # grep -c ALREADY prints 0 and exits 1 on no-match; the old `|| echo 0` appended a
  # SECOND 0 -> errs="0\n0" -> `[: 0\n0: integer expression expected` -> the test errored,
  # was treated as false, and the stage was marked DONE without ever checking. A gate that
  # cannot fail is not a gate.
  errs=$(grep -c "Error during evaluation" "$M/$tag.log" 2>/dev/null | head -1)
  errs=${errs:-0}
  got=$(find "$M/$tag" -name "*results*.json" 2>/dev/null | wc -l)
  if [ "$errs" -gt 0 ] || [ "$got" -eq 0 ]; then
    log "  !! $tag FAILED (errors=$errs results=$got)"; return 1
  fi
  touch "$ST/$tag.done"; log "  $tag DONE"
}

# ============================ 0: PRIMARY k=8 matrix (15/60/600) ============================
# RUNS FIRST, before the 3600s embeds. The embed dump is hours long with an unknown
# tail (sequential cv2 decode), and on 2026-07-15 putting the primary result behind a
# long unattended stage meant the pod died with NOTHING banked. Bank the paper table
# first; 3600s is upside.
# Per-bin = per model load: one bin failing can no longer take the other two with it.
for b in 15 60 600; do
  run_arms "k8_$b" \
    "longvideobench_val_i_${b}s_k8,longvideobench_val_picks_lc_${b}s_k8,longvideobench_val_picks_omp_lc_${b}s_k8" \
    "LVB_PICKS_LC_K8=$P/picks_lc_k8.json" "LVB_PICKS_OMP_LC_K8=$P/picks_omp_lc_k8.json"
done
# SigLIP scorer arm, 60s only (per plan: SigLIP paused elsewhere)
run_arms k8sig_60 \
  "longvideobench_val_picks_sig_60s_k8" "LVB_PICKS_SIG_K8=$P/picks_sig_k8.json"

# ---------------------------------------------------- 1: 3600s image embeds
if [ ! -f "$ST/embeds3600.done" ]; then
  # STREAMING encoder (2026-07-16): dump_embeds now decodes+encodes per batch and frees the
  # frames, so PEAK RSS is ~4GB per shard REGARDLESS of video length -- it was the whole-video
  # frame list (~6GB, up to ~10GB for a 3574s clip) that blew the 57.7GB cgroup at 11 shards
  # (free(1) reports the HOST's 251GB inside the container, so the first sizing was 4x too high).
  # Verified byte-identical to the old path on one video. Measured clean rate: 33.5s/video
  # single-process (decode 12s + encode 22s), not the 13-15h the THRASHING run showed.
  #
  # Encode (~18ms/frame) is GPU-bound and bigger than decode (~10ms/frame, CPU-parallel), so
  # total encode ~80min on ONE GPU; a 2nd GPU pod halves it. Multi-pod = one GLOBAL shard space:
  # every pod passes the SAME EMB_NUM_SHARDS and its own EMB_SHARD_LO..EMB_SHARD_HI slice. The
  # video->shard map is a sorted-index split (deterministic across processes), so disjoint slices
  # never overlap and never double-decode. Solo default: 12 shards, 0..11.
  #
  # --no-use-siglip is a DELIBERATE scope call: LongCLIP is LDDR's standardized encoder and our
  # primary arm; SigLIP is parked and scored BELOW uniform at 60s k=8. The npz carries no siglip
  # key, so harness.embeds.load_image_embed reports that scorer uncached, never wrong vectors.
  NSH=${EMB_NUM_SHARDS:-12}          # size of the GLOBAL shard space
  LO=${EMB_SHARD_LO:-0}              # this pod's first shard index (inclusive)
  HI=${EMB_SHARD_HI:-$((NSH - 1))}  # this pod's last shard index (inclusive)
  log "STAGE 1: 3600s embeds — global $NSH shards, THIS pod runs $LO..$HI, streaming, LongCLIP only"
  export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
  pids=()
  for sh in $(seq "$LO" "$HI"); do
    $PY scripts/dump_embeds.py --manifest data/manifest.lvb.long976.json \
      --bins 3600 --no-gold-reliable-only --no-use-dino --no-use-siglip --dedup-by-video \
      --num-shards "$NSH" --shard "$sh" --batch-size 32 \
      --out-dir "$SLM/results/embeds" > "$SLM/results/emb3600_sh$sh.log" 2>&1 &
    pids+=($!)
  done
  log "  $(( HI - LO + 1 )) shards launched on this pod (~4GB RSS each, streaming): ${pids[*]}"
  for p in "${pids[@]}"; do wait "$p"; done
  got=$($PY - <<'PY'
import json, os
m = json.load(open("/workspace/slm-lab/data/manifest.lvb.long976.json"))
ids = [x["id"] for x in m if str(x.get("length_bin")) == "3600s"]
have = sum(1 for i in ids if os.path.exists(f"/workspace/slm-lab/results/embeds/{i}.npz"))
print(f"{have}/{len(ids)}")
PY
)
  log "  3600s embeds: $got"
  case "$got" in
    564/564) touch "$ST/embeds3600.done"; log "  STAGE 1 DONE" ;;
    *) log "  !! 3600s embeds INCOMPLETE ($got) — arms would fail-loud on a missing qid."
       log "  !! continuing to score whatever exists; 3600s answering will be skipped." ;;
  esac
fi

# ---------------------------------------------------------- 2: 3600s scores
if [ ! -f "$ST/scores3600.done" ] && [ -f "$ST/embeds3600.done" ]; then
  log "STAGE 2: 3600s stem-only scores (CPU off cached embeds)"
  export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
  ok=1
  # lc ONLY. Stage 1 runs --no-use-siglip, so no 3600s npz carries a siglip key and a
  # `sig` pass here would write 0 rows -> ok=0 -> scores3600.done never touched ->
  # k8_3600 silently skipped AFTER the ~5h embed run. Keep this list in sync with the
  # towers stage 1 actually builds.
  for sc in lc; do
    $PY scripts/score_from_embeds.py --manifest data/manifest.lvb.long976.json \
      --scorer $sc --bins 3600 \
      --out "$SLM/results/scores/scores_${sc}_3600.jsonl" \
      --text-embeds-out "$SLM/results/embeds_text/text_${sc}_3600.npz" \
      > "$SLM/results/score3600_$sc.log" 2>&1 || ok=0
    n=$(wc -l < "$SLM/results/scores/scores_${sc}_3600.jsonl" 2>/dev/null || echo 0)
    log "  scores_${sc}_3600.jsonl: $n rows"
    [ "$n" -gt 0 ] || ok=0
  done
  [ "$ok" -eq 1 ] && { touch "$ST/scores3600.done"; log "  STAGE 2 DONE"; } \
                  || log "  !! STAGE 2 FAILED — see results/score3600_*.log"
fi

# ------------------------------------------------------------- 3: all picks
if [ ! -f "$ST/picks_all.done" ]; then
  log "STAGE 3: picks for every bin x k x {topk,omp}"
  export PYTHONPATH="$SLM:$SLM/slmenv/lib/python3.11/site-packages"
  bash scripts/make_picks_all.sh > "$SLM/results/make_picks_all.log" 2>&1
  nf=$(ls "$SLM"/results/picks_lmmseval/*.json 2>/dev/null | grep -vc meta)
  log "  picks files: $nf"
  [ "$nf" -ge 8 ] && { touch "$ST/picks_all.done"; log "  STAGE 3 DONE"; } \
                  || log "  !! STAGE 3 FAILED — see results/make_picks_all.log"
fi


# ------------------------------- 4: 3600s @ k=8  — completes the LDDR grid
if [ -f "$ST/scores3600.done" ]; then
  run_arms k8_3600 \
    "longvideobench_val_i_3600s_k8,longvideobench_val_picks_lc_3600s_k8,longvideobench_val_picks_omp_lc_3600s_k8" \
    "LVB_PICKS_LC_K8=$P/picks_lc_k8.json" "LVB_PICKS_OMP_LC_K8=$P/picks_omp_lc_k8.json"
else
  log "SKIP k8_3600 — no 3600s scores"
fi

# --------------------------------------------- 5+: bonus, non-LDDR budgets
run_arms k16_60 \
  "longvideobench_val_i_60s_k16,longvideobench_val_picks_lc_60s_k16,longvideobench_val_picks_omp_lc_60s_k16" \
  "LVB_PICKS_LC_K16=$P/picks_lc_k16.json" "LVB_PICKS_OMP_LC_K16=$P/picks_omp_lc_k16.json"

run_arms k32_600 \
  "longvideobench_val_i_600s_k32,longvideobench_val_picks_lc_600s_k32,longvideobench_val_picks_omp_lc_600s_k32" \
  "LVB_PICKS_LC_K32=$P/picks_lc_k32.json" "LVB_PICKS_OMP_LC_K32=$P/picks_omp_lc_k32.json"

if [ -f "$ST/scores3600.done" ]; then
  run_arms k32_3600 \
    "longvideobench_val_i_3600s_k32,longvideobench_val_picks_lc_3600s_k32,longvideobench_val_picks_omp_lc_3600s_k32" \
    "LVB_PICKS_LC_K32=$P/picks_lc_k32.json" "LVB_PICKS_OMP_LC_K32=$P/picks_omp_lc_k32.json"
fi

log "=== OVERNIGHT CHAIN FINISHED ==="
ls -la "$ST"/
