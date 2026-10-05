#!/usr/bin/env bash
# Rebuild the training env from scratch ON A TILLICUM GPU NODE (apex/TE/flash-attn
# compile against the env's own CUDA 12.8; 2-4 h, mostly compile). Use this when the
# rsync'd env is gone (scrubbed purge) or never copied. Mirrors how the klone env was
# built (conda-meta/history + Xunmei's setup_env.sh), pinned by requirements_freeze_klone.txt.
#   srun -A zixianma -p gpu-h200 --gpus=1 -c 32 --mem=200G --time=06:00:00 bash setup_env_tillicum.sh 2>&1 | tee logs/setup_env.log
# Layout: conda env on projects/krishna ($PERSIST_ROOT, not purged -- the only piece that is slow to rebuild);
# code, model, browsers, outputs on scrubbed ($PORT_ROOT). $PORT_ROOT/envs is a symlink into $PERSIST_ROOT.
# Needs: the git clone at $PORT_ROOT (OpenWebRL/, build/Megatron-LM/, tillicum/) and egress.
set -euxo pipefail
PORT_ROOT="${PORT_ROOT:-/gpfs/scrubbed/heqj3/owrl}"                       # code + outputs (scrubbed, purged)
PERSIST_ROOT="${PERSIST_ROOT:-/gpfs/projects/krishna/heqj3/owrl_persist}"  # conda env only (NOT purged; 18 GB; krishna has ~26 GB free)
BIG_ROOT="${BIG_ROOT:-$PORT_ROOT}"                                         # model (9 GB) + browsers (0.6 GB): scrubbed, re-downloaded in minutes if purged
ENV_DIR="$PERSIST_ROOT/envs/webrl"; REPO="$PORT_ROOT/OpenWebRL"; MEG="$PORT_ROOT/build/Megatron-LM"
mkdir -p "$PERSIST_ROOT/envs" "$BIG_ROOT/models/OpenWebRL" "$BIG_ROOT/ms-playwright"
# the launch chain only knows $PORT_ROOT/{envs,models,ms-playwright}: symlink whatever lives elsewhere
[ -e "$PORT_ROOT/envs" ] || ln -s "$PERSIST_ROOT/envs" "$PORT_ROOT/envs"
for d in models ms-playwright; do [ "$BIG_ROOT" = "$PORT_ROOT" ] || [ -e "$PORT_ROOT/$d" ] || ln -s "$BIG_ROOT/$d" "$PORT_ROOT/$d"; done
need_gb=21; free_gb=$(df -BG --output=avail "$PERSIST_ROOT" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge $need_gb ] || { echo "ABORT: only ${free_gb}G free under $PERSIST_ROOT; the env needs ~18G plus headroom"; exit 1; }
REQ="$PORT_ROOT/tillicum/requirements_freeze_klone.txt"
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load conda/Miniforge3-25.3.1-3 gcc/13.4.0
export TMPDIR=/tmp/$USER/build PIP_CACHE_DIR=/tmp/$USER/pip_cache CONDA_PKGS_DIRS=/tmp/$USER/conda_pkgs   # never home (10 GB)
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$PORT_ROOT/logs"
# 0. conda layer: python + CUDA 12.8 toolkit (nvcc, cudart, cublas...) exactly as on klone
[ -d "$ENV_DIR" ] || conda create -y -p "$ENV_DIR" python=3.12
conda install -y -p "$ENV_DIR" -c nvidia cuda-toolkit=12.8.1
# profilers/IDE bits of the toolkit are never used at runtime or by JIT (nvcc + headers + libs stay): -1.4 GB
[ "${TRIM:-1}" = 1 ] && rm -rf "$ENV_DIR"/nsight-compute-* "$ENV_DIR"/nsightee_plugins "$ENV_DIR"/bin/{nsys,ncu,ncu-ui,nvvp,nsight-sys,cuda-gdb,cuda-gdbserver,compute-sanitizer} 2>/dev/null
export PATH="$ENV_DIR/bin:$PATH" CUDA_HOME="$ENV_DIR"
NV="$ENV_DIR/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$LD_LIBRARY_PATH"
PIP="$ENV_DIR/bin/pip"
# 1. torch first (everything compiles against it)
$PIP install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
$PIP install "numpy==1.26.4" nvidia-cudnn-cu12==9.16.0.29 cmake ninja packaging
# 2. every pure wheel from the freeze (skip git/file sources, the compiled ones, and torch which is pinned above)
grep -vE "^(apex|mbridge|megatron-bridge|torch_memory_saver|pip) @|^(torch|torchvision|torchaudio|flash_attn|transformer_engine|transformer_engine_cu12|transformer_engine_torch|megatron-core|openwebrl|sglang|sgl-kernel)==" "$REQ" > /tmp/$USER/req_wheels.txt
$PIP install -r /tmp/$USER/req_wheels.txt
$PIP install sglang==0.5.6.post2 sgl-kernel==0.3.19          # pinned wheels (klone env has the wheel, not the source build)
# 3. compiled deps
MAX_JOBS=16 $PIP -v install flash-attn==2.7.4.post1 --no-build-isolation
$PIP install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
$PIP install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
NVCC_APPEND_FLAGS="--threads 4" $PIP -v install --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
$PIP install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall --no-deps
$PIP install git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a --no-build-isolation --no-deps
# 4. our source trees as editable installs (patched copies from the git clone)
(cd "$MEG" && $PIP install -e . --no-deps)
(cd "$REPO" && $PIP install -e . --no-deps)
# 5. browsers: playwright 1.58.0 -> chromium-1208 + headless shell + ffmpeg-1011, into $PORT_ROOT/ms-playwright
export PLAYWRIGHT_BROWSERS_PATH="$BIG_ROOT/ms-playwright"
[ -d "$PLAYWRIGHT_BROWSERS_PATH/chromium-1208" ] || "$ENV_DIR/bin/playwright" install chromium
# 6. SFT start model from HF (9.1 GB)
[ -f "$BIG_ROOT/models/OpenWebRL/OpenWebRL-4B-SFT/model.safetensors" ] || \
  "$ENV_DIR/bin/hf" download OpenWebRL/OpenWebRL-4B-SFT --local-dir "$BIG_ROOT/models/OpenWebRL/OpenWebRL-4B-SFT" 2>/dev/null || \
  "$ENV_DIR/bin/huggingface-cli" download OpenWebRL/OpenWebRL-4B-SFT --local-dir "$BIG_ROOT/models/OpenWebRL/OpenWebRL-4B-SFT"
# 7. verify (same checks as step3 preflight)
PYTHONPATH="$REPO:$MEG" "$ENV_DIR/bin/python" - <<'PY'
import numpy as np, torch; assert np.__version__.startswith("1.")
assert torch.cuda.is_available(); v=torch.backends.cudnn.version(); assert v>=91500, v
import scipy.sparse, megatron.core, megatron.training; from megatron.bridge import AutoBridge
import sglang, transformer_engine, flash_attn, apex, playwright, openai, wandb, ray
print("SETUP OK torch", torch.__version__, "cudnn", v, "sglang", sglang.__version__)
PY
echo "===== SETUP DONE ====="
