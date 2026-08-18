#!/usr/bin/env bash
# Wait for the two 600 contender legs (shared-volume .done), then launch single-arm 3600 jobs
# on this pod. Args: ARM:GPU specs, e.g. `alpha0.5:0 alpha0.75:1`. Each arm coverage-gated.
set -u
SLM=/workspace/slm-lab; cd "$SLM"
while ! { [ -f results/valid_rerun/600_alpha.done ] && [ -f results/valid_rerun/600_rfloor.done ]; }; do sleep 30; done
echo "600 contenders done $(date -u) -> launching single-arm 3600 on this pod"
for spec in "$@"; do
  arm=${spec%%:*}; gpu=${spec##*:}
  setsid bash -c "BIN=3600 ARMS=$arm LABEL=3600_$arm CUDA_VISIBLE_DEVICES=$gpu bash $SLM/scripts/run_valid_arms.sh" \
    > "$SLM/results/valid_rerun/3600_${arm}.launch.log" 2>&1 < /dev/null &
  echo "launched $arm on GPU$gpu $(date -u)"
done
