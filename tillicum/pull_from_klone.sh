#!/usr/bin/env bash
# Run ON TILLICUM (login node). Pulls the OpenWebRL tree from klone into $DEST.
# One Duo prompt: an ssh ControlMaster keeps a single authenticated connection
# open and every rsync below reuses it.
#   bash pull_from_klone.sh            # everything needed for training (~31 GB)
#   WITH_HF_CKPTS=1 bash pull_from_klone.sh   # + hf_ckpts for eval (+50 GB)
set -uo pipefail
SRC_HOST="${SRC_HOST:-heqj3@klone.hyak.uw.edu}"
SRC=/gscratch/cse/heqj3/owrl
DEST="${DEST:-/gpfs/scrubbed/heqj3/owrl}"
CM=/tmp/ssh-cm-klone-$USER
mkdir -p "$DEST"
ssh -o ControlMaster=auto -o ControlPath="$CM" -o ControlPersist=4h -fN "$SRC_HOST" || { echo "ssh to klone failed"; exit 1; }
R="rsync -a --info=progress2 --partial -e 'ssh -o ControlPath=$CM'"
pull() { echo "== $1"; eval $R "$SRC_HOST:$SRC/$1" "$DEST/${2:-$1}"; }
# small, start editing while the rest streams
pull port/                      port/
pull tillicum/                  tillicum/
pull OpenWebRL/                 OpenWebRL/          # 7 MB, includes openwebrl/data/
pull OpenWebRL_eval/            OpenWebRL_eval/
pull build/                     build/              # 1.1 GB (Megatron-LM patched + sglang source)
# browser bundle lives in Xunmei's dir on klone
echo "== ms-playwright"; eval $R "$SRC_HOST:/gscratch/cse/xunmei/ms-playwright/" "$DEST/ms-playwright/" --exclude '*.deps'
pull models/                    models/             # 9.1 GB SFT start point
pull envs/webrl/                envs/webrl/         # 20 GB; then run fix_env_paths.sh
[ "${WITH_HF_CKPTS:-0}" = 1 ] && pull hf_ckpts/ hf_ckpts/
mkdir -p "$DEST/logs" "$DEST/outputs"
ssh -o ControlPath="$CM" -O exit "$SRC_HOST" 2>/dev/null
echo "== done. NEXT: create $DEST/secrets.env by hand (chmod 600), then bash $DEST/tillicum/fix_env_paths.sh"
du -sh "$DEST"/* 2>/dev/null
