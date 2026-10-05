#!/usr/bin/env bash
# Host-resource sampler. Runs alongside training and writes one line a minute.
#
# Why this exists: on 2026-08-14 g3103 went unresponsive mid-rollout, the job
# stuck in CG, and the node had to be drained and rebooted. run.log truncated
# mid-line with no error of any kind -- no OOM, no MemoryError, no cgroup
# oom-kill. We could not tell afterwards whether we had been anywhere near the
# job's memory cap, because nothing was recording it.
#
# Slurm gives the job cgroup memory.max = --mem but leaves memory.swap.max = max,
# so exceeding the cap does NOT trigger an OOM kill -- the kernel swaps instead.
# That degrades quietly and piles processes into uninterruptible D state.
#
# WHAT TO ALARM ON. Not the size of swap. Two false alarms came from that: on
# g3101 the kernel parks cold anonymous pages at every rollout->train transition
# (4 GiB once, then 24 GiB) while the cgroup sits at ~78% and memory.events "max"
# stays 0, and the figure then holds perfectly still. That is ordinary reclaim
# under a burst allocation, not distress. The two signals that actually separate
# g3101's benign parking from g3103's death spiral are:
#
#   * memory.events "max" > 0  -- the cgroup really did hit memory.max
#   * swap still GROWING between consecutive samples -- reclaim is not keeping up
#
# so those are what raise WARN, plus occupancy above 90% of the cap.
#
#   bash resource_sampler.sh [interval_secs] >> logs/resources.log

set -u
INT="${1:-60}"

CG="/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
# Walk up to the job-level cgroup. The leaf (step_extern/user/task_0) has a
# memory.max too, but its value is the literal "max" -- the real --mem cap only
# appears on the job_<id> level, so keep climbing until the value is numeric.
while [ "$CG" != "/sys/fs/cgroup" ]; do
  v="$(cat "$CG/memory.max" 2>/dev/null)"
  case "$v" in ''|max) CG="$(dirname "$CG")" ;; *) break ;; esac
done
rd() { v="$(cat "$CG/$1" 2>/dev/null)"; case "$v" in ''|max) echo 0 ;; *) echo "$v" ;; esac; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

MAXB="$(rd memory.max)"
PREV_SWP=""
echo "[sampler] cgroup=$CG memory.max=$(gb "$MAXB")GiB interval=${INT}s started $(date -Is)"
echo "[sampler] time            mem_gb/max  swap_gb  peak_gb  load1  procs  chromium  tmp_free  maxhit"

while true; do
  # How many times the cgroup actually hit memory.max. This is the real signal;
  # swap size is not.
  MAXHIT="$(awk '/^max /{print $2}' "$CG/memory.events" 2>/dev/null)"
  MAXHIT="${MAXHIT:-0}"
  MEM="$(gb "$(rd memory.current)")"
  SWP="$(gb "$(rd memory.swap.current)")"
  PEAK="$(gb "$(rd memory.peak)")"
  LOAD="$(cut -d' ' -f1 /proc/loadavg)"
  PROCS="$(ps -u "$(id -u)" --no-headers 2>/dev/null | wc -l)"
  # pgrep -c prints 0 AND exits 1 when nothing matches, so a `|| echo 0`
  # fallback would append a second 0. Take the first field only.
  CHROME="$(pgrep -u "$(id -u)" -c -f chrome-headless-shell 2>/dev/null | head -1)"
  CHROME="${CHROME:-0}"
  TMP="$(df -h --output=avail /tmp 2>/dev/null | tail -1 | tr -d ' ')"
  printf "[sampler] %s  %s/%s  %s  %s  %s  %s  %s  %s  maxhit=%s\n" \
    "$(date +%H:%M:%S)" "$MEM" "$(gb "$MAXB")" "$SWP" "$PEAK" "$LOAD" "$PROCS" "$CHROME" "$TMP" "$MAXHIT"
  # Loud line so a grep for WARN finds the moment things turned.
  # SWAP CARRIES NO INDEPENDENT SIGNAL IN THIS WORKLOAD -- do not alarm on it.
  # Three false alarms were raised trying to: on absolute size (4 GiB), on a
  # bigger absolute size (24 GiB), and on growth between samples (15.0 -> 22.8).
  # All three fired at the same place, the rollout->train transition, where the
  # browser pool is torn down and Megatron allocates ~147 GiB inside one sampling
  # interval; the kernel parks a few GiB of cold anon pages, then the figure sits
  # perfectly still until the next transition. It happens every iteration, at
  # ~79% occupancy, with maxhit=0 throughout. Swap here is the shadow of that
  # burst, nothing more.
  #
  # Only two things mean real trouble, and g3103 showed both: the cgroup actually
  # reaching memory.max (memory.events "max" ticks up), and sustained occupancy
  # near the cap. swap_gb stays in the log as evidence, never as a trigger.
  awk -v m="$MEM" -v x="$(gb "$MAXB")" -v h="$MAXHIT" 'BEGIN{
    if (x > 0 && m > 0.90*x) printf "[sampler] WARN memory at %.0f%% of cap (%s/%s GiB)\n", 100*m/x, m, x;
    if (h+0 > 0)             printf "[sampler] WARN cgroup HIT its memory cap %s time(s) -- the g3103 pattern, act now\n", h;
  }'
  PREV_SWP="$SWP"
  sleep "$INT"
done
