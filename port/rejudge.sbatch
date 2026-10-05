#!/usr/bin/env bash
#SBATCH --job-name=owrl_rejudge
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/eval/rejudge_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/eval/rejudge_%j.out
# usage: JUDGE=gpt-5.6-luna [RUNS="sft_gpt41_greedy_run1 ..."] [LIMIT=5] sbatch port/rejudge.sbatch
set -uo pipefail
ROOT=/gscratch/cse/heqj3/owrl; export PATH="$ROOT/envs/webrl/bin:$PATH"; export PYTHONPATH="$ROOT/OpenWebRL_eval"
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p $TRITON_CACHE_DIR
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; set +x; . $ROOT/secrets.env
JUDGE="${JUDGE:?}"; TAG=$(echo "$JUDGE" | tr -c 'a-zA-Z0-9\n' '_')
for r in ${RUNS:-sft_gpt41_greedy_run1 iter30_gpt41_greedy_run1 iter60_gpt41_greedy_run1 iter89_gpt41_greedy_run1}; do
  E=$(ls -d $ROOT/outputs/eval_om2w/$r/*/eval_*turn_* | head -1)
  python $ROOT/port/rejudge_eval.py --eval-dir "$E" --judge-model "$JUDGE" --out $ROOT/outputs/rejudge/${r}__${TAG}.jsonl --concurrency 8 ${LIMIT:+--limit $LIMIT}
done
echo "[rejudge] all done $(date -Is)"
