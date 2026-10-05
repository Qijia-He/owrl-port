#!/usr/bin/env bash
#SBATCH --job-name=owrl_convert_hf
#SBATCH --account=gpu-l40s-krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/cse/heqj3/owrl/logs/eval/convert_%j.out
#SBATCH --error=/gscratch/cse/heqj3/owrl/logs/eval/convert_%j.out
# Convert Megatron torch_dist checkpoints to HF (CPU, gloo, world_size 1).
# Calls tools/convert_torch_dist_to_hf_bridge.py DIRECTLY: the wrapper
# scripts/run_convert_hf.sh deletes the .distcp shards after converting.
set -uo pipefail
ROOT=/gscratch/cse/heqj3/owrl
export PATH="$ROOT/envs/webrl/bin:$PATH"
export PYTHONPATH="$ROOT/OpenWebRL:$ROOT/build/Megatron-LM"
NV="$ROOT/envs/webrl/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$ROOT/envs/webrl/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache; mkdir -p $TRITON_CACHE_DIR
# needs one GPU: the Megatron bridge asserts CUDA (2026-09-14: "No CUDA GPUs are available" on the CPU-only attempt)
ORIGIN=$ROOT/models/OpenWebRL/OpenWebRL-4B-SFT
SCR=/gscratch/scrubbed/heqj3/owrl/outputs/owrl_native_frepro_l40s/ckpt
declare -A SRC=( [iter30]=$SCR/iter_0000030 [iter60]=$SCR/iter_0000060 [iter89]=/gscratch/krishna/heqj3/owrl_ckpt/iter_0000089 )
# generic form: CONVERT_SPECS="tag=/abs/iter_dir tag2=/abs/iter_dir2" (overrides CONVERT_TAGS)
if [ -n "${CONVERT_SPECS:-}" ]; then unset SRC; declare -A SRC; TAGS=""; for kv in $CONVERT_SPECS; do SRC[${kv%%=*}]=${kv#*=}; TAGS="$TAGS ${kv%%=*}"; done; else TAGS="${CONVERT_TAGS:-iter89 iter60 iter30}"; fi
for tag in $TAGS; do
  in=${SRC[$tag]}; out=$ROOT/hf_ckpts/owrl_${tag}
  if [ -f "$out/model.safetensors" ] || ls "$out"/*.safetensors >/dev/null 2>&1; then echo "[convert] $tag exists, skip"; continue; fi
  echo "[convert] $tag: $in -> $out  $(date -Is)"
  ( cd $ROOT/OpenWebRL && python tools/convert_torch_dist_to_hf_bridge.py --input-dir "$in" --output-dir "$out" --origin-hf-dir "$ORIGIN" -f ) || { echo "[convert] $tag FAILED"; continue; }
  # sglang needs tokenizer/processor/chat template alongside the weights
  for f in chat_template.jinja chat_template.json tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json processor_config.json video_preprocessor_config.json generation_config.json; do
    [ -e "$out/$f" ] || cp "$ORIGIN/$f" "$out/" 2>/dev/null
  done
  ls -la "$out" | head -20; du -sh "$out"
done
echo "[convert] done $(date -Is)"
