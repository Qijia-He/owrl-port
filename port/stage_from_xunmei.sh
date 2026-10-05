#!/usr/bin/env bash
# One-time staging of Xunmei's OpenWebRL tree into my own root.
# Sizes: OpenWebRL+port+build ~2 GB, models ~8 GB, envs/webrl 31 GB (~15 min on GPFS).
# Checkpoints go to /gscratch/scrubbed (13 TB free, 21-day purge) via the outputs symlink.
set -euo pipefail
SRC=/gscratch/cse/xunmei/owrl
MY_ROOT=/gscratch/cse/heqj3/owrl
mkdir -p "$MY_ROOT"

echo "[stage] repo + patches + build + models"
rsync -a --info=progress2 "$SRC/OpenWebRL" "$SRC/build" "$SRC/models" "$MY_ROOT/"
echo "[stage] conda env (31 GB)"
rsync -a --info=progress2 "$SRC/envs" "$MY_ROOT/"

# outputs live on scrubbed; the launcher derives SLIME_SAVE_ROOT from PORT_ROOT/outputs
mkdir -p /gscratch/scrubbed/heqj3/owrl/outputs
[ -e "$MY_ROOT/outputs" ] || ln -s /gscratch/scrubbed/heqj3/owrl/outputs "$MY_ROOT/outputs"

if [ ! -f "$MY_ROOT/secrets.env" ]; then
  printf 'export OPENAI_API_KEY=sk-REPLACE_ME\n' > "$MY_ROOT/secrets.env"
  chmod 600 "$MY_ROOT/secrets.env"
  echo "[stage] wrote template $MY_ROOT/secrets.env -- put YOUR key in it"
fi
git config --global --add safe.directory "$MY_ROOT/OpenWebRL" || true
echo "[stage] done. Tree:"; ls -la "$MY_ROOT"
