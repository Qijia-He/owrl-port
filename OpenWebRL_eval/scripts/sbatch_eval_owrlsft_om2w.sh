#!/bin/bash
#SBATCH --job-name=owrlsft-om2w-eval
#SBATCH --account=gpu-a40-krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/dan29/logs/owrlsft_om2w_eval_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/dan29/logs/owrlsft_om2w_eval_%j.err
#
# Online-Mind2Web eval for the owrlsft-om2w RL checkpoints (RL-trained on top of
# the released OpenWebRL SFT model). Usage:
#
#   MODEL_PATH=/…/zixian-checkpoints/owrlsft-om2w-lr1e6-step32 \
#   RUN_NAME=owrlsft-om2w-lr1e6-step32_gpt41_run1 \
#   sbatch scripts/sbatch_eval_owrlsft_om2w.sh
#
# WHY A DEDICATED SCRIPT rather than reusing sbatch_eval_online_mind2web.sh:
# that script sets NO timeouts and defaults the judge to Gemini, so every arm
# has to pass ~8 env vars at submit time and any one of them can be forgotten.
# The values below are the ones job 37863465 (gemini-cua, a published
# comparator) actually ran with, read off its log -- baking them in makes these
# runs comparable to the 37.50% baseline by construction.
#
# DO NOT use scripts/sbatch_eval_qwen35.sh for these. It pins the qwen35-serve
# env and forces --attention-backend triton, which Qwen3.5-4B needs and a
# Qwen3-VL checkpoint does not; it puts the arm on a different inference stack
# from every comparator and shifted a score by ~3 points on 2026-07-30.

set -ex

export HF_HOME=/mmfs1/gscratch/krishna/dan29/hf
export PLAYWRIGHT_BROWSERS_PATH=/mmfs1/gscratch/krishna/dan29/OpenWebRL/.cache/ms-playwright
# STAGE THE BROWSER BUNDLE TO NODE-LOCAL /tmp (added 2026-08-25). With the krishna
# pool at 100%, chromium's startup reads of the GPFS-resident bundle exceed the 180s
# BrowserType.launch cap even on lightly-loaded nodes -- killed 3 eval attempts in a
# row (g3067/g3072/g3075; load theory refuted on g3075 at load 5). A bulk cp is
# tolerant of slow reads in a way chromium's startup is not; same fix as browser
# generation (reference_browser_gen_needs_unsaturated_node). Falls back to the GPFS
# path if the copy fails.
PW_LOCAL=/tmp/pw_${SLURM_JOB_ID}
if mkdir -p "$PW_LOCAL" && cp -r "$PLAYWRIGHT_BROWSERS_PATH/." "$PW_LOCAL/" 2>/dev/null; then
  export PLAYWRIGHT_BROWSERS_PATH="$PW_LOCAL"
  trap 'rm -rf "$PW_LOCAL"' EXIT
else
  echo "WARN: playwright staging to /tmp failed; using GPFS bundle" >&2
fi
export PATH="/mmfs1/gscratch/krishna/dan29/conda/envs/openwebrl/bin:${PATH}"
# per-job Triton cache: a shared one leaks tens of thousands of inodes into the
# fileset's 61M file cap, which killed an eval on 2026-07-30.
# node-local (2026-08-25): the krishna per-job dir made sglang startup JIT writes die
# with Errno 122 while the pool sat at 100% (killed 38838034 during cuda-graph capture)
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
# node-local TMPDIR (2026-08-25): ~/.bashrc exports TMPDIR on krishna, so chromium
# put its ephemeral profiles (--user-data-dir) on the 100%-full pool; profile I/O
# stalls -> BrowserType.launch preflight timeout (38843617) and mid-run Page crashes.
export TMPDIR=/tmp/job_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

cd /mmfs1/gscratch/krishna/dan29/OpenWebRL

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME, e.g. owrlsft-om2w-lr1e6-step32_gpt41_run1}"

set +x
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY}}"
export JUDGE_API_KEY
set -x

# No --no-save-eval-samples: these runs KEEP eval_samples (~1.6 GB each), so
# they score with analysis/score_v2.sh, the canonical v2 path.
#
# Backends are left at the harness defaults (flashinfer/flashinfer) to match
# every comparator. TEMPERATURE=0 means the sampler is never entered, so the
# --sampling-backend pytorch workaround is not needed here.
#
# TEMPERATURE/TOP_P are now overridable so a sampled arm can reuse THIS script
# and stay byte-comparable to the greedy baseline (same 65536 context, same
# judge, same parallelism). The defaults are unchanged, so every existing
# comparator still runs greedy exactly as before.
#
# !! IF YOU SET TEMPERATURE>0 OR TOP_P<1 YOU MUST ALSO PASS
#    SGLANG_EXTRA_ARGS="--sampling-backend pytorch".
# The sampled path calls FlashInfer's top_k_top_p_sampling_from_probs, which
# JIT-builds via /usr/local/cuda/bin/nvcc -- a path that does not exist on these
# nodes. That killed all three top-p arms of 2026-07-29: 300/300 ABORTED while
# SLURM still reported COMPLETED 0:0. See
# analysis/flashinfer_topp_failure_2026-07-29/PROVENANCE.md.
# SGLANG_CONTEXT_LENGTH pinned to 65536 rather than inherited from
# run_evaluation_local.sh, so the served context does not silently revert
# to 32768 if that working-tree edit is lost.
MODEL_PATH="${MODEL_PATH}" \
TASK_FILE="${TASK_FILE:-openwebrl/data/online-mind2web.jsonl}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval_online_mind2web_full/${RUN_NAME}}" \
N_PARALLEL=8 \
SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=8 \
SGLANG_TP=1 \
SGLANG_DP=1 \
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}" \
JUDGE_API_MODE=served \
JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com/v1}" \
TEMPERATURE="${TEMPERATURE:-0.0}" \
NO_REPEAT_RESAMPLE="${NO_REPEAT_RESAMPLE:-}" \
FINAL_ANSWER_AT_STEP="${FINAL_ANSWER_AT_STEP:-}" \
INEFFECTIVE_HINT="${INEFFECTIVE_HINT:-}" \
TOP_P="${TOP_P:-1.0}" \
INFERENCE_STEP_TIMEOUT_SECS=120 \
SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=300 \
SGLANG_STARTUP_TIMEOUT_SECS="${SGLANG_STARTUP_TIMEOUT_SECS:-1800}" \
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}" \
bash scripts/run_evaluation_local.sh
