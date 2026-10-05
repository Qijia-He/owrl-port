#!/usr/bin/env bash
# Run ON KLONE. Snapshot the code half of /gscratch/cse/heqj3/owrl into the private
# GitHub repo (Qijia-He/owrl-port) so Tillicum can `git clone` it and a scrubbed purge
# costs nothing. Only code/configs/scripts/data go in; the env, model, browsers,
# checkpoints, logs and secrets are excluded (see EXCLUDES) -- they are rebuilt by
# tillicum/setup_env_tillicum.sh.
#   bash sync_to_github.sh "commit message"
set -euo pipefail
SRC=/gscratch/cse/heqj3/owrl; DST=/gscratch/cse/heqj3/owrl_git; MSG="${1:-sync $(date -Is)}"
EXCLUDES=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude '*.orig' --exclude '*.orig_*' --exclude wandb --exclude '*.tgz' --exclude '*.zip')
mkdir -p "$DST/build"
rsync -a --delete "${EXCLUDES[@]}" "$SRC/OpenWebRL/"        "$DST/OpenWebRL/"
rsync -a --delete "${EXCLUDES[@]}" "$SRC/OpenWebRL_eval/"   "$DST/OpenWebRL_eval/"
rsync -a --delete "${EXCLUDES[@]}" "$SRC/build/Megatron-LM/" "$DST/build/Megatron-LM/"
rsync -a --delete "${EXCLUDES[@]}" --exclude '*.log' "$SRC/port/"     "$DST/port/"
rsync -a --delete "${EXCLUDES[@]}" "$SRC/tillicum/"         "$DST/tillicum/"
cd "$DST"
# secret gate: the live key values must not appear anywhere in the snapshot
. "$SRC/secrets.env"
for k in "$OPENAI_API_KEY" "$WANDB_API_KEY"; do
  if grep -rqF --exclude-dir=.git -- "$k" .; then echo "ABORT: a live API key is inside the snapshot:"; grep -rlF --exclude-dir=.git -- "$k" .; exit 1; fi
done
if grep -rqE "sk-proj-[A-Za-z0-9_-]{20}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}" . --exclude-dir=.git; then echo "ABORT: key-shaped string found:"; grep -rlE "sk-proj-[A-Za-z0-9_-]{20}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}" . --exclude-dir=.git; exit 1; fi
big=$(find . -path ./.git -prune -o -type f -size +90M -print); [ -z "$big" ] || { echo "ABORT: files over GitHub's 100 MB limit:"; echo "$big"; exit 1; }
git add -A
git -c user.name=Qijia-He -c user.email=heqijia23@gmail.com commit -q -m "$MSG" || { echo "nothing to commit"; exit 0; }
git push -q origin main && echo "pushed: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
