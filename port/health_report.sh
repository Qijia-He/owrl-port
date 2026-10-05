#!/usr/bin/env bash
# Rollout health from run.log: browser session survival, groups needed per
# iteration, and rollout time. Usage: bash port/health_report.sh [run.log]
LOG="${1:-/gscratch/cse/heqj3/owrl/outputs/${EXP_NAME:-owrl_native_frepro_l40s}/run.log}"
echo "== $LOG"
echo "-- browser sessions (healthy=True is a session that survived /reset):"
grep -oE 'LocalProcessExit.*healthy=(True|False)' "$LOG" | grep -oE 'healthy=\w+' | sort | uniq -c
echo "-- why /reset died:"
grep -oE '/reset failed for task webvoyager/[0-9]+: [A-Za-z]+\(.{0,60}' "$LOG" \
  | sed -E 's#task webvoyager/[0-9]+#task#; s/[0-9]+ms/Nms/g' | cut -c1-75 | sort | uniq -c | sort -rn | head -5
echo "-- env_server spawn timeouts:"; grep -c 'was not healthy within' "$LOG"
echo "-- groups completed to get 48 accepted, per iteration (lower is better; Xunmei: 860-1670):"
grep -E 'RolloutProgress.*accepted_groups=48/48' "$LOG" | sed -E 's/.*rollout_id=([0-9]+) .*completed_groups=([0-9]+).*/iter \1: \2 groups/' | awk '!s[$2]++' | tail -10
echo "-- rollout seconds per iteration (Xunmei: 5200-7500):"
grep -oE "perf ([0-9]+):.*'perf/rollout_time': [0-9.]+" "$LOG" | sed -E "s/perf ([0-9]+):.*rollout_time': ([0-9]+).*/iter \1: \2 s/" | tail -10
echo "-- must stay 0:"; echo "OOM=$(grep -c OutOfMemoryError "$LOG") ActorDied=$(grep -c ActorDiedError "$LOG")"
