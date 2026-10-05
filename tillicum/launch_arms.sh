#!/usr/bin/env bash
# Run ON TILLICUM login node. Submits the two 60-iteration arms, one 8xH200 node each.
#   bash launch_arms.sh            # both
#   bash launch_arms.sh base41     # one of: base41 partial41
# Edit --cpus-per-task / --mem in run_owrl_tillicum.sbatch after checking the node:
#   scontrol show node g006 | grep -E "CPUTot|RealMemory"
set -uo pipefail
PORT_ROOT="${PORT_ROOT:-/gpfs/scrubbed/heqj3/owrl}"; cd "$PORT_ROOT"
SB="$PORT_ROOT/tillicum/run_owrl_tillicum.sbatch"
COMMON="BROWSER_SLOTS=${BROWSER_SLOTS:-48},NUM_ROLLOUT=60,MEM_RESTART_GB=${MEM_RESTART_GB:-800},JUDGE_MODEL=gpt-4.1"
for arm in ${@:-base41 partial41}; do
  case $arm in
    base41)    EXTRA="" ;;
    partial41) EXTRA=",JUDGE_PARTIAL_MODEL=gpt-4.1,JUDGE_PARTIAL_CAP=0.8" ;;
    *) echo "unknown arm $arm"; exit 1;;
  esac
  EXP="owrl_${arm}_tlc"
  jid=$(sbatch --parsable --job-name="owrl_$arm" --export=ALL,EXP_NAME=$EXP,$COMMON$EXTRA "$SB")
  echo "$arm -> job $jid  exp=$EXP  (log: logs/slurm_$jid.out, run.log: outputs/$EXP/run.log)"
done
squeue -u "$USER" -o "%.10i %.14j %.10P %.8T %.10M %R"
