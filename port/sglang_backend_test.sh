#!/usr/bin/env bash
# Startup test for sglang.launch_server in MY env; full server logs kept on GPFS.
ROOT=/gscratch/cse/heqj3/owrl; OUT=$ROOT/logs/eval/sglang_test; mkdir -p $OUT
export PATH="$ROOT/envs/webrl/bin:/sw/gcc/13.2.0/bin:$PATH"
NV="$ROOT/envs/webrl/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$ROOT/envs/webrl/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$LD_LIBRARY_PATH:${LIBRARY_PATH:-}" CUDA_HOME="$ROOT/envs/webrl"
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache FLASHINFER_WORKSPACE_BASE=/tmp/$USER/flashinfer TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_cache XDG_CACHE_HOME=/tmp/$USER/xdg_cache; mkdir -p $TRITON_CACHE_DIR $FLASHINFER_WORKSPACE_BASE $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME
export HF_HUB_OFFLINE=1 no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost PYTHONUNBUFFERED=1 SGLANG_VLM_CACHE_SIZE_MB=4096
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
M=$ROOT/models/OpenWebRL/OpenWebRL-4B-SFT
echo "node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
run_variant() {  # name, extra args...
  local name=$1; shift; local port=$((30000 + RANDOM % 1000)); local log=$OUT/$name.log
  echo "=== $name port=$port start=$(date +%T) args: $*"
  python -m sglang.launch_server --model-path "$M" --host 127.0.0.1 --port $port --trust-remote-code --dtype bfloat16 --tp 1 --log-level info "$@" > $log 2>&1 &
  local pid=$! t0=$(date +%s) ok=0
  for i in $(seq 1 60); do sleep 10; curl -sf http://127.0.0.1:$port/health_generate >/dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; done
  echo "$name healthy=$ok after $(( $(date +%s) - t0 ))s"; grep -vE 'FutureWarning|warnings.warn|^\s*$' $log | tail -8 | cut -c1-220
  if [ $ok = 1 ]; then curl -s http://127.0.0.1:$port/generate -H 'Content-Type: application/json' -d '{"text":"Hello, my name is","sampling_params":{"max_new_tokens":8,"temperature":0}}' | cut -c1-200; echo; fi
  kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null; pkill -9 -f "port $port" 2>/dev/null; sleep 8
}
run_variant triton_ctx65k_warmupskip --attention-backend triton --mem-fraction-static 0.8 --context-length 65536 --skip-server-warmup
run_variant triton_ctx32k_nocudagraph --attention-backend triton --mem-fraction-static 0.5 --context-length 32768 --skip-server-warmup --disable-cuda-graph
run_variant flashinfer_ctx65k_warmupskip --attention-backend flashinfer --mem-fraction-static 0.8 --context-length 65536 --skip-server-warmup
run_variant triton_ctx65k_default --attention-backend triton --mem-fraction-static 0.8 --context-length 65536
