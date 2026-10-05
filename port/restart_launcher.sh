#!/usr/bin/env bash
# Tear down a running OpenWebRL launcher on THIS node so the outer loop in
# run_owrl_8xl40s.sbatch can relaunch step3 (which resumes from the tracker).
# Kill list follows RUNBOOK §6b; matching goes through /proc/<pid>/cmdline so
# this shell (and the sbatch/guard shells) are never matched by their own text.
#   usage: bash restart_launcher.sh "<reason>"
REASON="${1:-manual}"
SELF=$$; PARENTS="$SELF $PPID $(ps -o ppid= -p $PPID 2>/dev/null | tr -d ' ')"
echo "[restart] $(date -Is) reason=$REASON"
# 1) stop the ray job politely first so actors flush logs
timeout 120 ray stop --force >/dev/null 2>&1 || true
# 2) kill sweep
for p in $(ps -u "$(id -u)" -o pid= --no-headers); do
  case " $PARENTS " in *" $p "*) continue;; esac
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
  case "$cmd" in
    *keep_timelimit_fit*|*wandb_backfill*|*restart_launcher*|*run_owrl_8xl40s.sbatch*|*slurm_script*) continue;;
    *openwebrl.docker.env_server*|*chrome*|*MegatronTrainRayActor*|*RolloutManager*|*raylet*|*gcs_server*|*sglang*|*python3\ train.py*|*resource_sampler*|*prune_checkpoints*|*ray\ job\ submit*|*run_browser_full.sh*|*run_8xl40s_interactive.sh*|*step3_train.sh*|*dashboard*|*log_monitor*|*ray/*)
      kill -9 "$p" 2>/dev/null ;;
  esac
done
sleep 5
# 3) browser temp state
rm -rf /tmp/slime_browser_ports/* /tmp/slime_browser_logs/* 2>/dev/null
left=$(ps -u "$(id -u)" -o pid=,args= | grep -cE 'env_server|chrome|raylet|sglang|train.py' || true)
echo "[restart] done; leftover matching procs=$left"
