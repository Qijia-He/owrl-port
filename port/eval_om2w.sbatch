#!/usr/bin/env bash
#SBATCH --job-name=owrl_eval
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --exclude=g3121
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/eval/eval_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/eval/eval_%j.out
# Online-Mind2Web eval, mirroring David's scripts/sbatch_eval_owrlsft_om2w.sh
# (1 GPU, 8 parallel browsers, gpt-4.1 judge, greedy by default, 65536 ctx,
# eval_samples kept so analysis/score_v2.sh works). Usage:
# NOTE 2026-09-14: attention backend forced to triton (same as training). In this env the
# flashinfer default never answers health checks (detokenizer heartbeat stuck) and the
# triton server needs ~9 min to come up on a cold node (GPFS model read + JIT).
#   MODEL_PATH=<hf dir> RUN_NAME=<name> [TASK_FILE=...] [TEMPERATURE=0.7 TOP_P=0.8] sbatch port/eval_om2w.sbatch
set -uo pipefail
ROOT=/gscratch/cse/heqj3/owrl; E=$ROOT/OpenWebRL_eval
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"; RUN_NAME="${RUN_NAME:?set RUN_NAME}"
export PATH="$ROOT/envs/webrl/bin:$PATH"
export PYTHONPATH="$E:$ROOT/build/Megatron-LM"
NV="$ROOT/envs/webrl/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$ROOT/envs/webrl/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ROOT/envs/webrl/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LIBRARY_PATH:-}"
export CUDA_HOME="$ROOT/envs/webrl"
[ -d /sw/gcc/13.2.0/bin ] && export PATH="/sw/gcc/13.2.0/bin:$PATH"
# node-local caches and tmp (home is 10 GB; krishna is at its inode cap)
export TMPDIR=/tmp/$USER/job_$SLURM_JOB_ID; mkdir -p $TMPDIR
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_cache FLASHINFER_WORKSPACE_BASE=/tmp/$USER/flashinfer XDG_CACHE_HOME=/tmp/$USER/xdg_cache CUDA_CACHE_PATH=/tmp/$USER/nv_cache
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $FLASHINFER_WORKSPACE_BASE $XDG_CACHE_HOME $CUDA_CACHE_PATH
export HF_HOME=/tmp/$USER/hf HF_HUB_OFFLINE=1; mkdir -p $HF_HOME
# browsers: stage Xunmei's bundle to node-local disk (two root-only *.deps files -> rc 23 is fine)
rsync -a --exclude '*.deps' /gscratch/cse/xunmei/ms-playwright/ /tmp/ms-playwright-$USER/ 2>/dev/null; rc=$?
if [ $rc -eq 0 ] || [ $rc -eq 23 ]; then export PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright-$USER; else export PLAYWRIGHT_BROWSERS_PATH=/gscratch/cse/xunmei/ms-playwright; echo "WARN: playwright staging failed (rc=$rc)"; fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,0.0.0.0,localhost" NO_PROXY="127.0.0.1,0.0.0.0,localhost"
set +x; . $ROOT/secrets.env; export JUDGE_API_KEY="$OPENAI_API_KEY"
echo "[eval] job=$SLURM_JOB_ID node=$(hostname) model=$MODEL_PATH run=$RUN_NAME temp=${TEMPERATURE:-0.0} top_p=${TOP_P:-1.0} start=$(date -Is)"
nvidia-smi -L
cd $E
MODEL_PATH="$MODEL_PATH" \
TASK_FILE="${TASK_FILE:-openwebrl/data/online-mind2web.jsonl}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/eval_om2w/$RUN_NAME}" \
N_PARALLEL=8 \
SLIME_BROWSER_ENV_MODE=local_process \
SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=8 \
SLIME_BROWSER_LOCAL_PROCESS_PYTHON="$ROOT/envs/webrl/bin/python" \
SGLANG_TP=1 SGLANG_DP=1 \
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}" JUDGE_API_MODE=served JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com/v1}" \
TEMPERATURE="${TEMPERATURE:-0.0}" TOP_P="${TOP_P:-1.0}" \
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---attention-backend triton}" \
INFERENCE_STEP_TIMEOUT_SECS=120 \
SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=300 \
SGLANG_STARTUP_TIMEOUT_SECS=2700 \
SGLANG_CONTEXT_LENGTH=65536 \
bash scripts/run_evaluation_local.sh
rc=$?
echo "[eval] finished rc=$rc at $(date -Is)"
echo "[eval] results: $(ls ${OUTPUT_ROOT:-$ROOT/outputs/eval_om2w/$RUN_NAME}/*/eval_*turn_*/results_task_*.jsonl 2>/dev/null | wc -l) task files"
