#!/bin/bash
# STANDARD 3-seed Online-Mind2Web eval launcher — the serving config of record
# (updated 2026-08-26): PLAIN NUCLEUS 0.7/0.8, NO no-repeat resample.
# (The 2026-08-25 standard added NRR k=3; retired by user decision 2026-08-26 —
#  set NRR=3 explicitly to reproduce old-standard runs.)
#
#   MODEL_DIR=/…/checkpoints/<ckpt>  RUN_PREFIX=<arm>_gpt41_nucresample \
#   bash scripts/launch_standard_evals.sh
#
# Comparator on the unfrozen-projector checkpoint at this config (plain nucleus):
#   29.78% n_success/300  /  42.60% canonical v2.
# The greedy (26.89/36.57) and NRR-k=3 (30.78±1.02/44.18±1.38) figures are NOT
# comparable to runs launched here — match configs when comparing.
#
# Hardening baked in (each killed at least one run on 2026-08-25):
#  - --sampling-backend pytorch: FlashInfer's top-p JIT needs an absent nvcc;
#    without this, 300/300 ABORT under a COMPLETED 0:0 exit.
#  - OUTPUT_BASE defaults to cse: the krishna pool at 100% killed three seeds
#    mid-write (Errno 122). Point back at krishna only when the pool has room.
#  - The sbatch stages the Playwright bundle and Triton cache to node-local
#    /tmp (startup deaths at 180s browser launch / ~90s sglang JIT write).
#  - Node groups are DISJOINT and cover the stragglers (g3044/g3045 were
#    reachable by every seed before -> two evals co-located on g3044).
set -euo pipefail
R=/mmfs1/gscratch/krishna/dan29/OpenWebRL
cd "$R"
export HF_HOME=/mmfs1/gscratch/krishna/dan29/hf

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the post-processed checkpoint dir}"
RUN_PREFIX="${RUN_PREFIX:?set RUN_PREFIX, e.g. myarm_gpt41_nucresample}"
OUTPUT_BASE="${OUTPUT_BASE:-/mmfs1/gscratch/cse/dan29/eval_online_mind2web_full}"
TEMP="${TEMP:-0.7}"; TOPP="${TOPP:-0.8}"; NRR="${NRR:-}"

[ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: OPENAI_API_KEY unset" >&2; exit 1; }
[ -d "$MODEL_DIR/llamafactory_saved" ] || { echo "ERROR: $MODEL_DIR not post-processed (no llamafactory_saved/)" >&2; exit 1; }
# every link of the guard's flag chain must exist or the arm silently runs plain nucleus
grep -q 'NO_REPEAT_RESAMPLE:-' scripts/sbatch_eval_owrlsft_om2w.sh || { echo "ERROR: sbatch lacks NO_REPEAT_RESAMPLE passthrough" >&2; exit 1; }
grep -q 'no-repeat-resample' scripts/run_evaluation_local.sh || { echo "ERROR: runner lacks the flag append" >&2; exit 1; }
grep -q 'no-repeat-resample' openwebrl/run_evaluate.py || { echo "ERROR: run_evaluate.py lacks the flag" >&2; exit 1; }
grep -q '_nrr_action_key' openwebrl/generate_browser.py || { echo "ERROR: generate_browser.py lacks the guard" >&2; exit 1; }

# disjoint node groups INCLUDING the former stragglers; g3070 bad ECC.
# Extend BAD with any node currently running a training or another eval.
G1=g3043,g3044,g3046,g3051,g3054,g3061,g3067
G2=g3047,g3052,g3057,g3062,g3064,g3065,g3074
G3=g3045,g3050,g3053,g3060,g3063,g3066,g3072,g3075
BAD="${BAD:-g3070}"

for r in 1 2 3; do
  case $r in
    1) excl="${G2},${G3},${BAD}" ;;
    2) excl="${G1},${G3},${BAD}" ;;
    3) excl="${G1},${G2},${BAD}" ;;
  esac
  name="${RUN_PREFIX}_run${r}"
  [ -d "${OUTPUT_BASE}/${name}" ] && { echo "SKIP ${name}: exists"; continue; }
  jid=$(MODEL_PATH="$MODEL_DIR" \
        RUN_NAME="$name" \
        OUTPUT_ROOT="${OUTPUT_BASE}/${name}" \
        TEMPERATURE="$TEMP" TOP_P="$TOPP" NO_REPEAT_RESAMPLE="$NRR" \
        SGLANG_EXTRA_ARGS="--sampling-backend pytorch" \
        JUDGE_MODEL=gpt-4.1 \
        JUDGE_API_KEY="$OPENAI_API_KEY" \
        SLIME_BROWSER_ENV_MODE=local_process \
        sbatch --parsable --exclude="$excl" scripts/sbatch_eval_owrlsft_om2w.sh)
  echo "submitted ${name}: job ${jid}  temp=${TEMP} top_p=${TOPP} nrr=${NRR:-off} out=${OUTPUT_BASE}"
done
