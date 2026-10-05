#!/usr/bin/env bash
# ============================================================================
# Launch the training run INSIDE an existing interactive allocation
# (salloc -A krishna -p gpu-l40s --nodes=1 --gpus=8 ...).
#
# Run it under tmux so an ssh drop does not kill the run:
#     tmux new -s owrl
#     bash /gscratch/krishna/xunmei/owrl/port/run_8xl40s_interactive.sh
#     # detach with Ctrl-b d ; reattach later with: tmux attach -t owrl
#
# Safe to re-run: it auto-resumes from the last checkpoint. If the allocation
# expires mid-iteration you lose at most one iteration (SAVE_INTERVAL=1).
#
# Smoke test instead of the full run:   SMOKE=1 bash run_8xl40s_interactive.sh
# ============================================================================
set -euo pipefail
PORT_ROOT="${PORT_ROOT:-/gscratch/krishna/xunmei/owrl}"
ENV_DIR="$PORT_ROOT/envs/webrl"
REPO_DIR="$PORT_ROOT/OpenWebRL"
BUILD_DIR="$PORT_ROOT/build"
EXP_NAME="${EXP_NAME:-owrl_native_frepro_l40s}"

# ---- sanity: we must be on a node that actually has the GPUs --------------
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "[hyak] visible GPUs: $NGPU"
if [ "${SMOKE:-0}" = "1" ]; then
  WANT=2
else
  WANT="${NUM_GPUS:-8}"   # 4-GPU twin config (2 rollout + 2 train) is allowed: GPU split is infrastructure (guide §7)
fi
if [ "$NGPU" -lt "$WANT" ]; then
  echo "[hyak] FATAL: need $WANT GPUs, see $NGPU. Are you inside the salloc shell?" >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
# 2026-09-09: home (10 GB quota) filled up and every SGLang engine died at init with
# "OSError: [Errno 122] Disk quota exceeded" in ~/.triton/cache -- and the rollout
# loop then spun for hours against dead engines. Put all JIT/compile caches on the
# node-local NVMe (/tmp, 2.8 TB). These are inherited by ray workers (ray start
# runs in this shell) and by the SGLang server subprocesses.
export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/inductor_cache"
export FLASHINFER_WORKSPACE_BASE="/tmp/$USER/flashinfer"
export XDG_CACHE_HOME="/tmp/$USER/xdg_cache"
export CUDA_CACHE_PATH="/tmp/$USER/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
export PYTHONPATH="$REPO_DIR:$BUILD_DIR/Megatron-LM"
export CUDA_HOME="$ENV_DIR"
NV="$ENV_DIR/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"

# LIBRARY_PATH (link time) is NOT the same as LD_LIBRARY_PATH (run time).
# flashinfer JIT-compiles attention kernels on first use and links them with
# -lcudart; with only LD_LIBRARY_PATH set, ld fails with
#     /usr/bin/ld: cannot find -lcudart
# and SGLang dies during cuda-graph capture. Give the linker the same dirs.
export LIBRARY_PATH="$ENV_DIR/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LIBRARY_PATH:-}"
# Anything that JIT-compiles at runtime also needs a compiler newer than the
# node default (GCC 8.5 < torch 2.9.1's floor of 9) and a working nvcc.
export CUDA_HOME="$ENV_DIR"
[ -d /sw/gcc/13.2.0/bin ] && export PATH="/sw/gcc/13.2.0/bin:$PATH"

# CRITICAL (Hyak): the node exports http_proxy, direct egress works, and the
# proxy BREAKS the OpenAI judge calls. Clear every proxy var.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"

# Browsers: prebuilt bundle staged to node-local disk (NFS cold-start is a killer).
#
# The bundle moved. /gscratch/krishna/xunmei/ms-playwright holds chromium-1117 +
# ffmpeg-1009, built for playwright ~1.49; setup_env.sh pins playwright 1.58.0,
# which wants chromium-1208, chromium_headless_shell-1208 and ffmpeg-1011 and
# dies with "Executable doesn't exist at .../chromium_headless_shell-1208/...".
# The matching set is installed at /gscratch/cse/xunmei/ms-playwright (and
# krishna is at 100% disk, so it could not be refreshed in place anyway).
PLAYWRIGHT_BUNDLE="${PLAYWRIGHT_BUNDLE:-/gscratch/cse/xunmei/ms-playwright}"
rsync -a --exclude "*.deps" "$PLAYWRIGHT_BUNDLE/" /tmp/ms-playwright-$USER/ 2>/dev/null; rc=$?
# rc 23 = "some files not transferred": Xunmei's bundle has two root-only *.deps
# files that are irrelevant to the browsers; the copy itself is complete (622 MB).
if [ $rc -eq 0 ] || [ $rc -eq 23 ]; then
  export PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright-$USER
  echo "[hyak] playwright staged to /tmp/ms-playwright from $PLAYWRIGHT_BUNDLE"
else
  export PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BUNDLE"
  echo "[hyak] WARN: staging failed, using $PLAYWRIGHT_BUNDLE (slow browser cold-start)"
fi

# Judge key
[ -f "$PORT_ROOT/secrets.env" ] && . "$PORT_ROOT/secrets.env"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (see $PORT_ROOT/secrets.env)}"
export JUDGE_API_MODE=served
export JUDGE_API_BASE="https://api.openai.com/v1"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
# Ablation B (2026-09-16): opposite-bias judge pair as the training reward.
#   JUDGE_MODEL_2=gpt-5.6-luna enables it (both judges run on every COMPLETED trajectory);
#   JUDGE_PAIR_MODE=drop|fail|half decides what a disagreement is worth (drop = reward None).
export JUDGE_MODEL_2="${JUDGE_MODEL_2:-}"
export JUDGE_PAIR_MODE="${JUDGE_PAIR_MODE:-drop}"
# Partial credit for failed trajectories (verified requirements / total, cap 0.8); empty = off.
export JUDGE_PARTIAL_MODEL="${JUDGE_PARTIAL_MODEL:-}"
export JUDGE_PARTIAL_CAP="${JUDGE_PARTIAL_CAP:-0.8}"
# Save groups the dynamic filter discards (all-fail / all-success) for partial-credit / RM work.
export SLIME_SAVE_DROPPED_GROUPS_DIR="${SLIME_SAVE_DROPPED_GROUPS_DIR:-$PORT_ROOT/outputs/$EXP_NAME/dropped_groups}"
export JUDGE_API_KEY="$OPENAI_API_KEY"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Browser pool (plain subprocess Chromium, no docker)
export SLIME_BROWSER_ENV_MODE=local_process
# The browser pool is the usual throughput ceiling, so size it to the cores we
# actually hold (the OnDemand session on g3103 has 128). Each Chromium wants
# roughly one core and 1-2 GB RAM. Lower it if the node starts swapping.
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${BROWSER_SLOTS:-96}"
export SLIME_BROWSER_LOCAL_PROCESS_HOST=127.0.0.1
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START=18000
export SLIME_BROWSER_LOCAL_PROCESS_PORT_END=18999
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=/tmp/slime_browser_logs
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=/tmp/slime_browser_ports
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON="$ENV_DIR/bin/python"

export SLIME_MODEL_ROOT="$PORT_ROOT/models"
export SLIME_SAVE_ROOT="$PORT_ROOT/outputs/$EXP_NAME"
export SLIME_SAVE_DIR="$SLIME_SAVE_ROOT/ckpt"
mkdir -p "$SLIME_SAVE_DIR" "$PORT_ROOT/logs"

# Auto-resume
if [ -f "$SLIME_SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  export SLIME_LOAD_CHECKPOINT="$SLIME_SAVE_DIR"
  export SLIME_CKPT_STEP="$(tr -dc '0-9' < "$SLIME_SAVE_DIR/latest_checkpointed_iteration.txt")"
  echo "[hyak] RESUMING @ iter $SLIME_CKPT_STEP"
else
  echo "[hyak] fresh start"
fi

export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# torch_memory_saver refuses to run with it (entrypoint.py:182 checks the var
# literally) and hard-fails SGLang startup with:
#     RuntimeError: TorchMemorySaver is disabled for the current process
#                   because expandable_segments is not supported yet.
# slime enables the memory saver whenever offload_rollout is on
# (sglang_engine.py:557 "enable_memory_saver": args.offload_rollout). That is
# what lets SGLang hand its KV-cache memory back before a training step, which
# matters far more on a 48 GB card than the fragmentation win from expandable
# segments. Setting PYTORCH_ALLOC_CONF (the torch 2.9 spelling) instead would be
# worse, not better: the allocator would use expandable segments while
# torch_memory_saver's check no longer sees it, breaking it silently.
# If fragmentation ever looks like the OOM cause, follow §8 (recompute, TP=2)
# rather than reinstating this.

# ---- L40S / no-NVLink NCCL fallbacks: uncomment ONLY if NCCL hangs at init --
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

cd "$REPO_DIR"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[hyak] SMOKE MODE (2 GPUs, colocated, few episodes)"
  export NUM_GPUS=2 ROLLOUT_NUM_GPUS="" BROWSER_SLOTS=8
  export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=8
  export SLIME_SAVE_ROOT="$PORT_ROOT/outputs/owrl_smoke"
  # SLIME_SAVE_DIR was already exported above as the REAL run's ckpt path, and
  # the repo script uses SAVE_DIR="${SLIME_SAVE_DIR:-${DEFAULT_SAVE_DIR}}" --
  # so overriding only SLIME_SAVE_ROOT left the smoke test writing its 58 GB
  # checkpoint into outputs/owrl_native_frepro_l40s/ckpt. That then makes the
  # real run's auto-resume find latest_checkpointed_iteration.txt and print
  # "RESUMING @ iter 0", silently continuing from a 2-GPU tiny-batch smoke
  # instead of starting the fresh seed the guide calls for. Redirect both.
  export SLIME_SAVE_DIR="$SLIME_SAVE_ROOT/ckpt"
  mkdir -p "$SLIME_SAVE_DIR"
  exec bash scripts/run_browser_smoke_2gpu.sh 2>&1 | tee -a "$SLIME_SAVE_ROOT/run.log"
fi

# ---- Full run: paper-aligned, identical recipe to the AI2 run --------------

# expandable_segments belongs HERE, not above the SMOKE branch.
#
# It must NOT be set for the colocated smoke path: torch_memory_saver checks
# PYTORCH_CUDA_ALLOC_CONF literally (entrypoint.py:182) and hard-fails SGLang
# startup with "TorchMemorySaver is disabled for the current process because
# expandable_segments is not supported yet."
#
# But the full run is disaggregated (ROLLOUT_NUM_GPUS=4 + ACTOR_GPUS=4), where
# the run's own args report colocate=False and offload_rollout=False. Since
# slime sets enable_memory_saver = args.offload_rollout (sglang_engine.py:557),
# the memory saver is never constructed here and there is nothing to conflict
# with. Meanwhile the AI2 run found the training step OOMing by a few hundred
# MiB with ~700 MiB sitting fragmented-free, which expandable_segments
# recovered exactly -- and 48 GB cards have thinner headroom than their 80 GB,
# so fragmentation is more likely to bite here, not less.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_GPUS="${NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"   # SGLang generation GPUs (2 on the 4-GPU twin)
export ACTOR_GPUS="${ACTOR_GPUS:-4}"               # Megatron training GPUs (2 on the 4-GPU twin -> TP=2 x DP=1, same global batch)
# TP=2 since 2026-08-21. Was TP=1 (DP=4): no NVLink on L40S means tensor-parallel
# traffic crosses PCIe, so TP=1 is the faster choice when it fits. It stopped
# fitting -- iteration 7's training step died with
#     torch.OutOfMemoryError: Tried to allocate 272.00 MiB.
#     GPU 0 has a total capacity of 44.39 GiB of which 237.62 MiB is free.
#     this process has 41.61 GiB memory in use.
# The guide's section 8 order was followed: the recompute fix is live (grep for
# _checkpoint_with_sequence_args returns 2, and the run uses
# --recompute-granularity full --recompute-method uniform), and
# expandable_segments was already set with only 139.60 MiB reserved-but-
# unallocated, so fragmentation was not the cause. That leaves TP=2.
#
# Note the shortfall was only 34 MiB, and 2.55 GiB of GPU 0 was held by four
# foreign 638 MiB CUDA contexts -- one per peer rank. Those are NOT a
# misconfiguration to fix: slime sets RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
# (ray/utils.py:18) on purpose so every actor sees all 8 GPUs, because the
# disaggregated train->rollout weight transfer needs that visibility. Forcing
# per-actor CUDA_VISIBLE_DEVICES would reclaim the 2.55 GiB but risks breaking
# weight sync, so it is not worth it on a fidelity run.
#
# TP=2 with ACTOR_GPUS=4 gives TP=2 x DP=2. Section 7 hyperparameters are
# untouched: global-batch-size stays 256, only gradient accumulation doubles.
# Checkpoints are torch_dist (.distcp), which reshards across TP on load, so
# iter_0000004 written under TP=1 resumes cleanly here.
export TP=2                        # was 1; see the OOM note above
export BROWSER_MAX_STEPS=15        # stage 1 (repo default 30 is stage 2)
export NUM_ROLLOUT="${NUM_ROLLOUT:-90}"
export SAVE_INTERVAL=1             # one checkpoint per iteration
export ROLLOUT_MAX_CONTEXT_LEN=21504
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"

# Optional memory diagnostic (snapshot at >45GiB or on OOM):
# export SLIME_MEM_SNAPSHOT=1

echo "[hyak] EXP=$EXP_NAME gpus=$NUM_GPUS (rollout $ROLLOUT_NUM_GPUS + train $ACTOR_GPUS, TP=$TP) num_rollout=$NUM_ROLLOUT judge=$JUDGE_MODEL judge2=${JUDGE_MODEL_2:-none} pair_mode=$JUDGE_PAIR_MODE"
exec bash scripts/run_browser_full.sh 2>&1 | tee -a "$SLIME_SAVE_ROOT/run.log"
