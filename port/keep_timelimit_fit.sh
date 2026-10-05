#!/usr/bin/env bash
# While job $1 is PENDING, keep its TimeLimit just inside the next maintenance
# reservation start ($2, ISO). TimeMin alone did not make SLURM shrink the limit:
# on 09-04 07:27 g3102 freed 8 GPUs and the job was skipped with
# "ReqNodeNotAvail, Reserved for maintenance" because TimeLimit was 4-12.
J=$1; MAINT=${2:-2026-09-08T04:00:00}
while :; do
  st=$(squeue -j "$J" -h -o %T 2>/dev/null)
  [ -z "$st" ] && { echo "$(date -Is) job gone: $(sacct -j $J -n -X -o State | head -1)"; exit 0; }
  [ "$st" = RUNNING ] && { echo "$(date -Is) RUNNING on $(squeue -j $J -h -o %N)"; exit 0; }
  rem=$(( $(date -d "$MAINT" +%s) - $(date +%s) - 1200 ))   # 20 min margin
  if [ $rem -lt 43200 ]; then echo "$(date -Is) <12h left before maintenance, stop shrinking"; exit 0; fi
  d=$((rem/86400)); h=$(( (rem%86400)/3600 )); m=$(( (rem%3600)/60 ))
  new=$(printf '%d-%02d:%02d:00' $d $h $m)
  cur=$(squeue -j "$J" -h -o %l)
  if [ "$cur" != "$new" ]; then scontrol update jobid="$J" TimeLimit="$new" && echo "$(date -Is) TimeLimit $cur -> $new (reason: $(squeue -j $J -h -o %r))"; fi
  sleep 900
done
