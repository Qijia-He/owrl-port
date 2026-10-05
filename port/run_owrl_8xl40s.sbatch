#!/usr/bin/env bash
#SBATCH --job-name=owrl_train
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
#SBATCH --time=7-00:00:00
#SBATCH --time-min=1-00:00:00
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/slurm_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/slurm_%j.out
#
# Whole-job supervisor for the 90-iteration OpenWebRL run.
#   * outer loop: relaunch step3 whenever the launcher exits (crash, or a
#     deliberate restart) until the tracker reaches NUM_ROLLOUT-1. step3 resumes
#     from ckpt/latest_checkpointed_iteration.txt.
#   * memory guard: host RSS grows ~14 GiB per iteration and only resets on
#     restart (RUNBOOK §0.3). When the sampler shows > MEM_RESTART_GB during a
#     training phase, wait for the next checkpoint to land, then restart.
#   * iteration guard: also restart every MAX_ITERS_PER_LAUNCH iterations as a
#     belt-and-braces bound on memory growth.
# Knobs (sbatch --export=ALL,BROWSER_SLOTS=64,...):
#   BROWSER_SLOTS (64)  MEM_RESTART_GB (760)  MAX_ITERS_PER_LAUNCH (14)  EXP_NAME
set -uo pipefail
# Twin/takeover mutual exclusion (memory: project_ckpt_pool_takeover_pattern): when the same
# run is queued on two partitions, whichever starts first cancels its siblings so two
# launchers never write the same checkpoint dir.
SIB_FILE="/gscratch/cse/heqj3/owrl/logs/siblings_${EXP_NAME:-owrl_native_frepro_l40s}.txt"
for sib in ${SIBLING_JOBS:-} $( [ -f "$SIB_FILE" ] && cat "$SIB_FILE" ); do
  [ "$sib" != "$SLURM_JOB_ID" ] && scancel "$sib" 2>/dev/null && echo "[sbatch] cancelled sibling job $sib"
done
echo "[sbatch] gpus=$(nvidia-smi -L | wc -l) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB split=${ROLLOUT_NUM_GPUS:-4}+${ACTOR_GPUS:-4}"
export BROWSER_SLOTS="${BROWSER_SLOTS:-64}"
export EXP_NAME="${EXP_NAME:-owrl_native_frepro_l40s}"
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS:-120}"
# wandb: key comes from secrets.env (sourced by the launcher); entity must override upstream's yangrui_rl default
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-heqj3-university-of-washington}"
export WANDB_PROJECT="${WANDB_PROJECT:-openwebrl}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
MEM_RESTART_GB="${MEM_RESTART_GB:-760}"
MAX_ITERS_PER_LAUNCH="${MAX_ITERS_PER_LAUNCH:-14}"
LAST_ITER=$(( ${NUM_ROLLOUT:-90} - 1 ))

ROOT=/gscratch/cse/heqj3/owrl
TRK="$ROOT/outputs/$EXP_NAME/ckpt/latest_checkpointed_iteration.txt"
RES="$ROOT/logs/resources.log"
LOG="$ROOT/logs/step3.$SLURM_JOB_ID.log"
tracker() { local t; t=$(tr -dc '0-9' < "$TRK" 2>/dev/null); echo "${t:--1}"; }

echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) slots=$BROWSER_SLOTS exp=$EXP_NAME start=$(date -Is) tracker=$(tracker)"
nvidia-smi -L
cd "$ROOT"

# ---- guard: restart the launcher at a checkpoint boundary when memory is high
guard() {
  local armed=0 armed_at=-1 start_iter cur mem chrome reason="" dead stuck dead_seen
  start_iter=$(tracker)
  dead_seen=$(grep -c 'Server process terminated unexpectedly\|Scheduler hit an exception' "$LOG" 2>/dev/null); sleep 900   # let the new launcher get past init before judging it
  while :; do
    sleep 60
    cur=$(tracker)
    mem=$(grep '^\[sampler\]' "$RES" 2>/dev/null | tail -1 | awk '{split($3,a,"/"); print int(a[1])}')
    chrome=$(grep '^\[sampler\]' "$RES" 2>/dev/null | tail -1 | awk '{print $8}')
    # 2026-09-09: all four SGLang engines died at init (home quota) but the rollout
    # loop kept going for 7 h with zero accepted groups. Restart right away when an
    # engine dies or when a rollout has completed >600 groups with none accepted.
    local dead=$(grep -c 'Server process terminated unexpectedly\|Scheduler hit an exception' "$LOG" 2>/dev/null)
    local stuck=$(tail -1 "$ROOT/outputs/$EXP_NAME/ckpt/progress.log" 2>/dev/null | grep -oE 'groups=0/48 completed_groups=[0-9]+' | grep -oE '[0-9]+$')
    if [ "${dead:-0}" -gt "${dead_seen:-0}" ] || [ "${stuck:-0}" -gt 600 ]; then
      echo "[guard] $(date -Is) engine dead (${dead:-0} new signals) or rollout stuck (${stuck:-0} groups, 0 accepted): restarting launcher now" | tee -a "$LOG"
      bash port/restart_launcher.sh "engine-dead-or-stuck" 2>&1 | tee -a "$LOG"
      return 0
    fi
    if [ "$armed" = 0 ]; then
      if [ "${mem:-0}" -ge "$MEM_RESTART_GB" ] && [ "${chrome:-1}" = 0 ]; then armed=1; armed_at=$cur; reason="mem=${mem}G>=${MEM_RESTART_GB}G"; fi
      if [ "$cur" -ge 0 ] && [ $((cur - start_iter)) -ge "$MAX_ITERS_PER_LAUNCH" ]; then armed=1; armed_at=$((cur-1)); reason="iters_since_launch=$((cur - start_iter))"; fi
      [ "$armed" = 1 ] && echo "[guard] $(date -Is) armed ($reason) at tracker=$cur; will restart after the next checkpoint" | tee -a "$LOG"
    elif [ "$cur" -gt "$armed_at" ] && [ "$cur" -lt "$LAST_ITER" ]; then
      echo "[guard] $(date -Is) checkpoint $cur landed; restarting launcher ($reason)" | tee -a "$LOG"
      sleep 90   # let the save fully settle on GPFS before the kill sweep
      bash port/restart_launcher.sh "$reason" 2>&1 | tee -a "$LOG"
      return 0
    fi
  done
}

# ---- outer loop
attempt=0
while :; do
  attempt=$((attempt+1))
  if [ "$(tracker)" -ge "$LAST_ITER" ]; then echo "[sbatch] tracker=$(tracker) >= $LAST_ITER: run complete"; break; fi
  echo "[sbatch] launch attempt=$attempt tracker=$(tracker) at $(date -Is)" | tee -a "$LOG"
  guard & GUARD_PID=$!
  bash port/step3_train.sh 2>&1 | tee -a "$LOG"
  kill $GUARD_PID 2>/dev/null; wait $GUARD_PID 2>/dev/null
  echo "[sbatch] launcher returned at $(date -Is) tracker=$(tracker)" | tee -a "$LOG"
  bash port/restart_launcher.sh "post-exit cleanup" >>"$LOG" 2>&1
  [ "$attempt" -ge 40 ] && { echo "[sbatch] too many launches, giving up"; break; }
  # 2026-09-09: relaunching 60 s after the kill sweep made the new SGLang servers die
  # at init ("Server process terminated unexpectedly") -- the old engines were still
  # releasing GPU memory / ports. Wait until every GPU reads 0 MiB and no engine or
  # browser process is left (up to 10 min) before launching again.
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    left=$(ps -u "$(id -u)" -o args= | grep -cE 'sglang|env_server|chrome|raylet|MegatronTrainRayActor' || true)
    [ "${used:-1}" -eq 0 ] && [ "${left:-1}" -eq 0 ] && break
    [ $((i % 6)) -eq 0 ] && echo "[sbatch] waiting for node to drain: gpu_mem_used=${used}MiB leftover_procs=$left" | tee -a "$LOG"
    sleep 10
  done
  sleep 30
done
echo "[sbatch] end=$(date -Is) tracker=$(tracker)"
