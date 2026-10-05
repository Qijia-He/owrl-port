#!/bin/bash
# Apply methodology v2 to eval run dirs, guarding the two known gotchas:
#  1. v2 needs the eval_*turn_<ts> INVOCATION dir, not the run dir
#     (tell you got it wrong: confirmed_excluded=0)
#  2. a run dir can hold several eval_*turn_<ts> siblings from crashed/resumed
#     attempts -- pick the one with the most results_task_*.jsonl
set -u
ROOT=/mmfs1/gscratch/krishna/dan29/OpenWebRL/outputs/eval_online_mind2web_full
DIRS=()
for d in "$@"; do
  best=""; bestn=-1
  for sub in "$d"/*/eval_*turn_* "$d"/eval_*turn_*; do
    [ -d "$sub" ] || continue
    n=$(ls "$sub"/results_task_*.jsonl 2>/dev/null | wc -l)
    echo "    candidate $(basename "$sub"): ${n} results" >&2
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$sub; fi
  done
  if [ -z "$best" ]; then echo "  NO eval_*turn_* dir under $d" >&2; continue; fi
  echo "  -> $d : picked $(basename "$best") (${bestn} results)" >&2
  # absolute: the cd below would otherwise break relative paths, and v2 then
  # silently reports denom=0 rather than erroring on the missing directory
  DIRS+=("$(realpath "$best")")
done
[ ${#DIRS[@]} -eq 0 ] && { echo "no dirs resolved" >&2; exit 1; }
cd /mmfs1/gscratch/krishna/dan29/OpenWebRL
/mmfs1/gscratch/krishna/dan29/conda/envs/openwebrl/bin/python \
  openwebrl/compute_eval_success_rate_woaborted_v2.py "${DIRS[@]}"
