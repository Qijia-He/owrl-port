#!/usr/bin/env bash
# Run ON TILLICUM after pull_from_klone.sh. The conda env was built on klone and
# carries absolute paths; rewrite them to the Tillicum PORT_ROOT.
#   - 137 bin/* shebangs (#!/mmfs1/gscratch/cse/{heqj3,xunmei}/owrl/envs/webrl/bin/python)
#   - the two editable-install finders (openwebrl, megatron-core) -> point at our copies
#   - *.pth / direct_url.json entries
# Idempotent. Prints a verification at the end.
set -uo pipefail
PORT_ROOT="${PORT_ROOT:-/gpfs/scrubbed/heqj3/owrl}"
PERSIST_ROOT="${PERSIST_ROOT:-/gpfs/projects/krishna/heqj3/owrl_persist}"
ENV="$PERSIST_ROOT/envs/webrl"   # real path (PORT_ROOT/envs is a symlink to it)
SP="$ENV/lib/python3.12/site-packages"
[ -x "$ENV/bin/python3.12" ] || { echo "no env at $ENV"; exit 1; }
n=0
for f in "$ENV"/bin/*; do
  [ -f "$f" ] && head -c 2 "$f" 2>/dev/null | grep -q '#!' || continue
  if head -1 "$f" | grep -qE '^#!.*/gscratch/(cse|krishna)/(heqj3|xunmei)/owrl/envs/webrl/'; then
    sed -i -E "1s#/mmfs1/gscratch/(cse|krishna)/(heqj3|xunmei)/owrl/envs/webrl#$ENV#; 1s#/gscratch/(cse|krishna)/(heqj3|xunmei)/owrl/envs/webrl#$ENV#" "$f"; n=$((n+1))
  fi
done
echo "[fix] shebangs rewritten: $n"
for f in "$SP"/__editable__*finder.py "$SP"/*.pth "$SP"/*.dist-info/direct_url.json; do
  [ -f "$f" ] || continue
  grep -qE '/gscratch/(cse|krishna)/(heqj3|xunmei)/owrl' "$f" || continue
  sed -i -E "s#(/mmfs1)?/gscratch/(cse|krishna)/(heqj3|xunmei)/owrl#$PORT_ROOT#g" "$f"
  echo "[fix] rewrote $f"
done
echo "[fix] leftover klone paths in bin/ and site-packages top level:"
grep -lE '/gscratch/(cse|krishna)/(heqj3|xunmei)/' "$ENV"/bin/* "$SP"/__editable__* "$SP"/*.pth 2>/dev/null || echo "  none"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$PORT_ROOT/OpenWebRL:$PORT_ROOT/build/Megatron-LM"
python - <<'PY'
import numpy, torch, sglang, megatron.core as mc, openwebrl, transformer_engine, flash_attn, playwright, openai, wandb, ray
print("[fix] imports OK  numpy", numpy.__version__, "torch", torch.__version__, "sglang", sglang.__version__)
print("[fix] megatron.core from", mc.__file__)
print("[fix] openwebrl from", openwebrl.__file__)
PY
