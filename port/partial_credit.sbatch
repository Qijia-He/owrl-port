#!/usr/bin/env bash
#SBATCH --job-name=owrl_pcscore
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/eval/pcscore_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/eval/pcscore_%j.out
# usage: JUDGE=gpt-5.2 RUNS="sft_gpt41_greedy_run1 iter60_gpt41_greedy_run1" [LIMIT=N] sbatch port/partial_credit.sbatch
set -uo pipefail
ROOT=/gscratch/cse/heqj3/owrl; export PATH="$ROOT/envs/webrl/bin:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; . $ROOT/secrets.env
JUDGE="${JUDGE:?}"; TAG=$(echo "$JUDGE" | tr -c 'a-zA-Z0-9\n' '_'); mkdir -p $ROOT/outputs/partial_credit
for r in ${RUNS:?}; do
  E=$(ls -d $ROOT/outputs/eval_om2w/$r/*/eval_*turn_* | head -1)
  python $ROOT/port/partial_credit_score.py --eval-dir "$E" --judge-model "$JUDGE" --out $ROOT/outputs/partial_credit/${r}__${TAG}.jsonl ${LIMIT:+--limit $LIMIT} ${INCLUDE_SUCCESS:+--include-success}
done
echo "[pc] all done $(date -Is)"
