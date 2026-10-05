#!/usr/bin/env bash
# STEP 3 driver: the real 90-iteration run, 8 GPUs, inside the interactive alloc.
#
#   bash port/run_8xl40s_interactive.sh
#
# Run this under tmux (the guide's STEP 3): an interactive allocation dies with
# the ssh connection, tmux keeps it alive. Safe to re-run -- it auto-resumes
# from ckpt/latest_checkpointed_iteration.txt.
#
# Same environment facts as steps 1 and 2:
#   - PORT_ROOT is on /gscratch/cse (krishna is 100% full).
#   - Node default compiler is GCC 8.5; anything that JIT-compiles at runtime
#     (flashinfer attention kernels) needs >= 9, so keep gcc 13 on PATH.
# The launcher handles proxies, browser staging, secrets, and the GPU split.
#
# This wrapper adds the checkpoint pruner alongside the run. At SAVE_INTERVAL=1
# a ~34 GB checkpoint lands every iteration; 90 of them is ~3.1 TB. Retention is
# every 10th iteration plus the most recent (see prune_checkpoints.py for the
# two safety properties: never delete what the tracker names, and never prune
# concurrently with a save).

set -uo pipefail

export PORT_ROOT=/gpfs/scrubbed/heqj3/owrl
export EXP_NAME="${EXP_NAME:-owrl_native_frepro_l40s}"
ENV_DIR="$PORT_ROOT/envs/webrl"
SAVE_DIR="$PORT_ROOT/outputs/$EXP_NAME/ckpt"

# Retention policy -- override at launch if you want more safety margin.
KEEP_EVERY="${KEEP_EVERY:-10}"
KEEP_RECENT="${KEEP_RECENT:-1}"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load gcc/13.4.0 2>/dev/null || true
if [ "$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)" -lt 9 ] 2>/dev/null; then
  echo "[step3] WARN: g++ < 9 on PATH and gcc/13.4.0 module missing; JIT builds will fail"
fi
export TMPDIR=/tmp/heqj3_build
mkdir -p "$TMPDIR" "$PORT_ROOT/logs" "$SAVE_DIR"

# The preflight below runs BEFORE the launcher exports these, and
# megatron.training resolves only from the Megatron-LM checkout (pip installs
# just megatron.core), so set them here to check what the run will actually see.
export PYTHONPATH="$PORT_ROOT/OpenWebRL:$PORT_ROOT/build/Megatron-LM"
NV="$ENV_DIR/lib/python3.12/site-packages/nvidia"
export CUDA_HOME="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ENV_DIR/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LIBRARY_PATH:-}"

fail() { echo "[step3] PREFLIGHT FAILED: $*" >&2; exit 1; }

# ---- preflight ------------------------------------------------------------
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "$NGPU" -ge "${NUM_GPUS:-8}" ] || fail "need ${NUM_GPUS:-8} GPUs, see $NGPU (are we inside the allocation?)"

# The env pieces that took five smoke attempts to get right -- fail in seconds
# rather than an hour into the first rollout.
"$ENV_DIR/bin/python" - <<'PY' || fail "environment check failed"
import sys
import numpy as np
assert np.__version__.startswith("1."), f"numpy {np.__version__} -- slime asserts 1.x"
import torch
assert torch.cuda.is_available(), "cuda not available"
v = torch.backends.cudnn.version()
assert v and v >= 91500, f"cudnn {v} -- sglang requires >= 9.15 with torch 2.9.1"
import scipy.sparse            # breaks under numpy 1.x if scipy is a 2.x build
import megatron.core, megatron.training
from megatron.bridge import AutoBridge   # silently missing after a bare setup_env.sh
import sglang, transformer_engine, flash_attn
print(f"[step3] env OK: numpy {np.__version__} torch {torch.__version__} cudnn {v}")
PY

# Free space: ~34 GB per checkpoint, and we hold ~2 between milestones.
AVAIL_GB=$(df -BG --output=avail "$SAVE_DIR" | tail -1 | tr -dc "0-9")
[ "${AVAIL_GB:-0}" -ge 200 ] || fail "only ${AVAIL_GB}G free under $SAVE_DIR; need >=200G"
echo "[step3] ${AVAIL_GB}G free under $SAVE_DIR"

# A save that dies half-way (2026-09-06: scrubbed hit its quota during iter 9's
# common.pt write) leaves iter_N with N > tracker. Megatron would resume from the
# tracker and re-save into that dir; move the leftover aside instead of deleting.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  TRK=$(tr -dc '0-9' < "$SAVE_DIR/latest_checkpointed_iteration.txt")
  for d in "$SAVE_DIR"/iter_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d" | sed 's/iter_0*//'); n=${n:-0}
    if [ "$n" -gt "${TRK:-0}" ]; then
      mv "$d" "$SAVE_DIR/../broken_$(basename "$d")_$(date +%Y%m%d_%H%M%S)" && echo "[step3] moved stale partial checkpoint $(basename "$d") aside (tracker=$TRK)"
    fi
  done
fi
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "[step3] RESUMING from iteration $(cat "$SAVE_DIR/latest_checkpointed_iteration.txt")"
else
  echo "[step3] fresh start (no checkpoint present)"
fi

# ---- host-resource sampler, alongside the run -----------------------------
# On 2026-08-14 g3103 went unresponsive mid-rollout and had to be drained and
# rebooted, and we had no memory data from the period at all -- run.log stopped
# mid-line with no error. The job cgroup caps memory at --mem but leaves
# memory.swap.max unlimited, so going over the cap swaps rather than OOM-killing,
# which is silent. This records it so next time there is something to read.
SAMPLE_LOG="$PORT_ROOT/logs/resources.log"
bash "$PORT_ROOT/port/resource_sampler.sh" 60 >>"$SAMPLE_LOG" 2>&1 &
SAMPLER_PID=$!
echo "[step3] resource sampler pid=$SAMPLER_PID -> $SAMPLE_LOG"

# ---- checkpoint pruner, alongside the run ---------------------------------
PRUNE_LOG="$PORT_ROOT/logs/prune.log"
"$ENV_DIR/bin/python" "$PORT_ROOT/port/prune_checkpoints.py" \
  --save-dir "$SAVE_DIR" --keep-every "$KEEP_EVERY" --keep-recent "$KEEP_RECENT" \
  >>"$PRUNE_LOG" 2>&1 &
PRUNER_PID=$!
echo "[step3] pruner pid=$PRUNER_PID (keep every ${KEEP_EVERY}th + ${KEEP_RECENT} most recent) -> $PRUNE_LOG"
cleanup() {
  echo "[step3] stopping pruner pid=$PRUNER_PID sampler pid=$SAMPLER_PID"
  kill "$PRUNER_PID" "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[step3] started $(date -Is) on $(hostname)"
cd "$PORT_ROOT"
bash port/run_8xl40s_interactive.sh
rc=$?
echo "[step3] launcher exited rc=$rc at $(date -Is)"
echo "[step3] re-run this script to resume from the last checkpoint."
