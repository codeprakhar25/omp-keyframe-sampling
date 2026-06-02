#!/usr/bin/env bash
# =============================================================================
# POD SETUP — run FIRST, with network OPEN, with NO agent running.
#
# Does everything that needs GitHub/PyPI/astral so the locked phase needs none
# of them for cloning or dep-resolution:
#   1. install toolchain (uv, node, claude CLI, python deps)
#   2. FULL-clone all 3 repos into repos/   (base_sha local -> offline checkout)
#   3. warm the uv cache + build firebase .venv via calibrate_eval (NO API)
#
# After this succeeds, run firewall.sh to blackhole GitHub, then verify_lock.sh.
#
# Idempotent: re-runnable. Safe to re-run if a step fails.
# =============================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"
echo "==> experiment dir: $EXP_DIR"

# Persistent uv cache (on the network volume if EXP_DIR is on it) so warmed
# wheels survive a pod restart and the locked phase resolves offline-ish.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EXP_DIR/.uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/agents-experiment-cache}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
echo "==> UV_CACHE_DIR=$UV_CACHE_DIR"

# ---- 1. toolchain --------------------------------------------------------
# tmux (run_repeats.sh) + rsync (laptop pull needs rsync on BOTH ends).
if ! command -v tmux >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  echo "==> installing tmux + rsync"
  apt-get update -y >/dev/null 2>&1 && apt-get install -y tmux rsync >/dev/null 2>&1 \
    || echo "WARN: apt install tmux/rsync failed — install manually if needed"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "FATAL: uv not on PATH (add \$HOME/.local/bin)"; exit 1; }

if ! command -v node >/dev/null 2>&1; then
  echo "==> installing node (for claude CLI)"
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
  apt-get install -y nodejs >/dev/null 2>&1 || echo "WARN: install node manually if this failed"
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "==> installing claude code CLI"
  npm install -g @anthropic-ai/claude-code
fi
claude --version || { echo "FATAL: claude CLI not installed"; exit 1; }

echo "==> python deps"
uv pip install --system -r requirements.txt 2>/dev/null \
  || pip install -r requirements.txt

# ---- 2. FULL-clone all repos (network open) ------------------------------
# Harness clone_repo() now does a full clone; calling it here pre-populates
# repos/ so the locked agent phase never clones. We clone directly to be sure.
declare -A REPOS=(
  ["firebase__firebase-admin-python"]="https://github.com/firebase/firebase-admin-python.git"
  ["pdm-project__pdm"]="https://github.com/pdm-project/pdm.git"
  ["OpShin__opshin"]="https://github.com/OpShin/opshin.git"
)
mkdir -p repos
for slug in "${!REPOS[@]}"; do
  if [ -d "repos/$slug/.git" ]; then
    echo "==> repos/$slug already present (skip clone)"
  else
    echo "==> FULL-cloning $slug"
    git clone "${REPOS[$slug]}" "repos/$slug"
  fi
done

# ---- 3. warm uv cache + build venvs (NO API, NO agent) -------------------
# calibrate_eval runs the gold source + tests for one task per repo. This:
#   - builds firebase's .venv template + installs its deps (caches wheels)
#   - resolves pdm/opshin `uv run --group` envs (caches wheels)
# so the per-cell eval under lock hits the warm cache, not a cold network.
echo "==> warming caches via calibrate_eval (one task per repo, no API)"
WARM_TASKS=(
  "firebase__firebase-admin-python__940"
  "pdm-project__pdm__3769"
  "OpShin__opshin__595"
)
for tid in "${WARM_TASKS[@]}"; do
  echo "    -- calibrate $tid"
  python3 calibrate_eval.py --task-id "$tid" || echo "    WARN: calibrate $tid non-zero (inspect, may still be warm)"
done

echo
echo "==> SETUP COMPLETE."
echo "    next: bash pod/firewall.sh   (blackhole GitHub)"
echo "    then: bash pod/verify_lock.sh (hard gate)"
