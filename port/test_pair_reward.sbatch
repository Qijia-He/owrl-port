#!/usr/bin/env bash
#SBATCH --job-name=owrl_pairtest
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/eval/pairtest_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/eval/pairtest_%j.out
set -uo pipefail
ROOT=/gscratch/cse/heqj3/owrl; export PATH="$ROOT/envs/webrl/bin:$PATH"; export PYTHONPATH="$ROOT/OpenWebRL:$ROOT/build/Megatron-LM"
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p $TRITON_CACHE_DIR
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; . $ROOT/secrets.env
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.2} JUDGE_MODEL_2="${JUDGE_MODEL_2:-}" JUDGE_PAIR_MODE=${JUDGE_PAIR_MODE:-drop} JUDGE_PARTIAL_MODEL="${JUDGE_PARTIAL_MODEL:-}" JUDGE_PARTIAL_CAP=${JUDGE_PARTIAL_CAP:-0.8}
python $ROOT/port/test_pair_reward.py --eval-dir "outputs/eval_om2w/iter60_gpt41_greedy_run1/owrl_iter60/eval_owrl_iter60_gpt-4.1_turn_20260914_215937" --limit ${LIMIT:-12} ${TASK_IDS:+--task-ids $TASK_IDS}
