#!/usr/bin/env bash
# Run ON A TILLICUM GPU NODE before the first real submission (10 min, 1 GPU):
#   srun -A zixianma -p gpu-h200 --gpus=1 -c 8 --mem=64G --time=00:30:00 bash preflight_gpu_node.sh
# Checks exactly what the launcher will need: CUDA via the env's own toolkit, JIT
# compiler, Chromium from the staged bundle reaching a live site, OpenAI judge call,
# wandb auth, node-local /tmp. Prints no secrets.
set -uo pipefail
PORT_ROOT="${PORT_ROOT:-/gpfs/scrubbed/heqj3/owrl}"
ENV="$PORT_ROOT/envs/webrl"; NV="$ENV/lib/python3.12/site-packages/nvidia"
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true; module load gcc/13.4.0 2>/dev/null || true
export PATH="$ENV/bin:$PATH" PYTHONPATH="$PORT_ROOT/OpenWebRL:$PORT_ROOT/build/Megatron-LM" CUDA_HOME="$ENV"
export LD_LIBRARY_PATH="$ENV/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$LD_LIBRARY_PATH" TRITON_CACHE_DIR=/tmp/$USER/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_cache XDG_CACHE_HOME=/tmp/$USER/xdg_cache
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
ok(){ echo "[ok]  $*"; }; bad(){ echo "[BAD] $*"; FAIL=1; }; FAIL=0
echo "node=$(hostname) gcc=$(g++ -dumpversion) nvcc=$(nvcc --version 2>/dev/null | tail -1 | grep -o 'V[0-9.]*')"
[ "$(g++ -dumpversion | cut -d. -f1)" -ge 9 ] && ok "g++ >= 9" || bad "g++ < 9"
df -h /tmp | tail -1 | awk '{print "/tmp avail " $4}'
python - <<'PY' && ok "torch/cudnn/megatron/sglang/TE/flash-attn import on GPU" || bad "env import"
import numpy as np, torch; assert np.__version__.startswith("1."); assert torch.cuda.is_available()
v=torch.backends.cudnn.version(); assert v>=91500, v
import megatron.core, megatron.training; from megatron.bridge import AutoBridge
import sglang, transformer_engine, flash_attn
x=torch.randn(1024,1024,device="cuda"); (x@x).sum().item()
print("  gpu:", torch.cuda.get_device_name(0), "cudnn", v)
PY
python - <<'PY' && ok "flashinfer / triton JIT compile" || bad "JIT compile (check gcc + nvcc + LIBRARY_PATH)"
import torch, triton, triton.language as tl
@triton.jit
def k(X, n: tl.constexpr):
    tl.store(X + tl.arange(0, n), tl.arange(0, n))
x=torch.empty(16, device="cuda", dtype=torch.int32); k[(1,)](x, 16); assert x[15]==15
PY
rsync -a --exclude '*.deps' "$PORT_ROOT/ms-playwright/" /tmp/ms-playwright-$USER/ && ok "playwright bundle staged to /tmp" || bad "playwright staging"
PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright-$USER python - <<'PY' && ok "chromium reached a live site" || bad "chromium launch / navigation"
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(headless=True); pg=b.new_page(); pg.goto("https://www.amazon.com", timeout=30000); print("  title:", pg.title()[:60]); b.close()
PY
[ -f "$PORT_ROOT/secrets.env" ] && . "$PORT_ROOT/secrets.env"
[ "$(stat -c %a "$PORT_ROOT/secrets.env" 2>/dev/null)" = 600 ] && ok "secrets.env mode 600" || bad "secrets.env missing or not mode 600"
python - <<'PY' && ok "OpenAI judge call (gpt-4.1) from this node" || bad "OpenAI call"
import os, openai; c=openai.OpenAI(); r=c.chat.completions.create(model="gpt-4.1", messages=[{"role":"user","content":"Reply with the single word SUCCESS."}], max_tokens=5); print("  reply:", r.choices[0].message.content)
PY
python -c "import wandb; wandb.login(key=__import__('os').environ['WANDB_API_KEY'], relogin=True, verify=True)" >/dev/null 2>&1 && ok "wandb auth" || bad "wandb auth"
AV=$(df -BG --output=avail "$PORT_ROOT/outputs" 2>/dev/null | tail -1 | tr -dc 0-9); [ "${AV:-0}" -ge 1500 ] && ok "outputs has ${AV}G free" || bad "outputs has only ${AV:-0}G free (need >=1.5T for two arms)"
echo; [ "$FAIL" = 0 ] && echo "PREFLIGHT PASSED" || echo "PREFLIGHT FAILED (see [BAD] lines)"
