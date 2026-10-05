#!/usr/bin/env bash
# Ablation B, cheap version: one arm, opposite-bias judge pair as the training reward.
#   reward = gpt-4.1 AND gpt-5.6-luna agree -> that verdict; disagree -> reward None (dropped
#   from the GRPO group by the dynamic filter). 60 iterations, everything else identical to
#   owrl_native_frepro_l40s (which stays the single-lenient-judge control, judge gpt-5.2).
set -euo pipefail
cd /gscratch/cse/heqj3/owrl
J=$(sbatch --parsable --export=ALL,EXP_NAME=owrl_pairjudge_l40s,NUM_ROLLOUT=60,JUDGE_MODEL=gpt-4.1,JUDGE_MODEL_2=gpt-5.6-luna,JUDGE_PAIR_MODE=drop,BROWSER_SLOTS=64 port/run_owrl_8xl40s.sbatch)
echo "$J" > logs/current_job_pairjudge.txt
echo "submitted pair-judge training job $J (EXP_NAME=owrl_pairjudge_l40s, 60 iters, judges gpt-4.1 + gpt-5.6-luna, mode=drop)"
squeue -j "$J" -O 'JobID:12,State:10,Reason:24,TimeLimit:12,NumCPUs:6'
