# OpenWebRL on Tillicum — what has to move and what the run needs
(written 2026-10-02 from the klone tree /gscratch/cse/heqj3/owrl; nothing on Tillicum has been checked yet —
 run port/tillicum_probe.sh there first and fill §6)

## 1. Pieces that must exist on Tillicum  (≈31 GB mandatory, +50 GB optional)
| piece | klone path | size | notes |
|---|---|---|---|
| training code | OpenWebRL/ | 7 MB | Xunmei's fork + 4 patched files: openwebrl/reward_browser.py (pair judge, partial credit), slime/rollout/sglang_rollout.py (dropped-group dumps), scripts/run_browser_full.sh (env forwarding), openwebrl/env/config.yaml |
| Megatron-LM | build/Megatron-LM/ | 410 MB | megatron-core 0.16.0rc0 source; 1 patch: megatron/core/dist_checkpointing/strategies/common.py (torch.save retry) |
| sglang source | build/sglang/ | 739 MB | reference only — the running sglang is the pip wheel in the env; can skip |
| launcher scripts | port/ | 141 KB | run_owrl_8xl40s.sbatch (supervisor) → step3_train.sh → run_8xl40s_interactive.sh → OpenWebRL/scripts/run_browser_full.sh → ray job submit train.py; plus prune_checkpoints.py, resource_sampler.sh, restart_launcher.sh, convert_hf.sbatch, eval_om2w.sbatch, rejudge*, partial_credit*, wandb_backfill.py |
| conda env | envs/webrl/ | 20 GB | py3.12.13, torch 2.9.1+cu128, CUDA 12.8 toolkit *inside the env* (bin/nvcc, lib/libcudart), sglang 0.5.6.post2, sgl-kernel 0.3.19, flashinfer 0.5.3, flash-attn 2.7.4.post1, transformer_engine 2.10.0, apex 0.1, ray 2.57.0, playwright 1.58.0, openai 2.6.1, wandb 0.28.2, triton 3.5.1. NOT relocatable as-is: 137 shebangs with absolute paths (13 still → xunmei), 128 conda-meta refs, megatron-core editable finder → /mmfs1/gscratch/cse/xunmei/owrl/build/Megatron-LM (masked by PYTHONPATH). Preferred: rebuild with /gscratch/cse/xunmei/owrl/port/setup_env.sh (84 lines, pinned) on Tillicum; fallback: rsync + sed. |
| SFT start model | models/OpenWebRL/OpenWebRL-4B-SFT/ | 9.1 GB | HF safetensors; Megatron loads it directly (no Megatron ckpt to convert) |
| data | OpenWebRL/openwebrl/data/ | 2.4 MB | train: webgym_filtered_popular_2102_cleaned.parquet (+blacklist hosts); eval: online-mind2web.jsonl; inside the repo, moves with it |
| browser bundle | /gscratch/cse/xunmei/ms-playwright/ | 642 MB | chromium-1208 + chromium_headless_shell-1208 + ffmpeg-1011, must match playwright 1.58.0; staged to node-local /tmp at launch |
| secrets | secrets.env | 249 B | OPENAI_API_KEY + WANDB_API_KEY, mode 600 — recreate by hand on Tillicum, never through a shared dir |
| eval package | OpenWebRL_eval/ | 6 MB | David's OM2W eval repro (triton attention backend, startup timeout 2700 s) |
| HF ckpts for eval | hf_ckpts/ | 50 GB | optional; only if evals also move. Training outputs (58 GB torch_dist per saved iter) do not need to move — start the arms fresh |

## 2. Per-node resource profile of one run (measured on klone)
- GPUs: 8×L40S 48 GB = Megatron actor TP=2×DP=2 on 4 + 4 SGLang engines (TP=1); 4-GPU twin = TP=2×DP=1 on 2 + 2 engines, same global batch.
  On 80/141 GB cards the 4B model fits with room; the same split (4+4) or 2+6 both work; GPU is NOT the bottleneck.
- CPU: the browser pool is the throughput ceiling: 1 Chromium ≈ 1 core + 1–2 GB. 64 slots on 96 cores (8-GPU), 40 slots on 64 cores (4-GPU twin).
- RAM: 850 GB asked on klone (600 GB on the twin). Host RSS grows ~14 GiB/iter (known leak) → supervisor restarts the launcher every 14 iters or at MEM_RESTART_GB; a node with ≥1 TB lets two arms share one node.
- Node-local disk (/tmp): playwright bundle 0.65 GB + triton/inductor/flashinfer/XDG caches + per-browser logs ≈ 20–50 GB; must be writable and not home (home quota kills SGLang engines).
- Shared disk: checkpoints 58 GB per saved iteration; SAVE_INTERVAL=1 with pruner keep-every-10 + latest ⇒ ~400–500 GB per 60-iter arm; 7 copies while pruning lags. Put outputs on the big scratch, not on a file-count-quota'd fs.
- Wall time: 4×L40S twin today: rollout 26–41 min/iter (base41 2087–2472 s, partial41 1519–1701 s) + training; 60 iters ≈ 1.5–2 days. 8×L40S ≈ 1.6–2× faster. Ask 7 days.
- Network from COMPUTE nodes (go/no-go for Tillicum):
  1. outbound HTTPS to arbitrary public websites — rollouts drive live sites (2102 popular hosts). No allow-list proxy will work.
  2. api.openai.com (judge, 300–1300 calls/iter), api.wandb.ai.
  3. On klone the node sets http_proxy and that BREAKS the OpenAI client → launcher unsets all proxy vars. On Tillicum do the opposite if egress is proxy-only.
- Ports (all 127.0.0.1): browser env servers 18000–18999, Ray head + dashboard, SGLang servers. One run per node unless the port ranges are split.
- OS libs for Chromium (klone has them, Rocky 8.10): libnss3 libnspr4 libnssutil3 libatk-1.0 libatk-bridge-2.0 libcups libxkbcommon libXcomposite libXdamage libXfixes libXrandr libgbm libcairo libpango-1.0 libasound libdrm. If missing on Tillicum: conda-forge equivalents into the env + LD_LIBRARY_PATH, or an Apptainer image.
- Compiler: JIT (flashinfer, triton, TE) needs gcc ≥ 9 and the env's nvcc; klone: `module load gcc/13.2.0`.

## 3. Launch chain and the knobs a port must re-point
run_owrl_8xl40s.sbatch  (#SBATCH account/partition/gpus/cpus/mem/time; ROOT=/gscratch/cse/heqj3/owrl; logs/; sibling cancel; MAX_ITERS_PER_LAUNCH=14; MEM_RESTART_GB)
  └ step3_train.sh        (PORT_ROOT, ENV_DIR, SAVE_DIR, module load gcc, TMPDIR, PYTHONPATH=OpenWebRL:build/Megatron-LM, CUDA_HOME=$ENV_DIR, LD_/LIBRARY_PATH to env's nvidia libs, pruner + sampler)
    └ run_8xl40s_interactive.sh (PORT_ROOT, caches→/tmp/$USER, PLAYWRIGHT_BUNDLE→/tmp/ms-playwright-$USER, proxy unset, JUDGE_API_MODE=served, JUDGE_API_BASE, JUDGE_MODEL, JUDGE_MODEL_2, JUDGE_PAIR_MODE, JUDGE_PARTIAL_MODEL, JUDGE_PARTIAL_CAP, SLIME_SAVE_DROPPED_GROUPS_DIR, browser pool vars, SLIME_MODEL_ROOT, SLIME_SAVE_ROOT, NUM_GPUS/ROLLOUT_NUM_GPUS/ACTOR_GPUS/TP, NUM_ROLLOUT)
      └ OpenWebRL/scripts/run_browser_full.sh (ray start --head --num-gpus; RUNTIME_ENV_JSON; TRAIN_DATA; judge args: JUDGE_PROMPT_VARIANT=action_history, JUDGE_MAX_ATTACHED_IMGS=3; GRPO hparams: 48 tasks×5, global batch 256, 2 ppo epochs, lr 1e-6, clip 0.2/0.28, temp 0.8, 15 steps)
        └ ray job submit … train.py   (resume = outputs/$EXP/ckpt/latest_checkpointed_iteration.txt)
Grep targets when porting: `/gscratch/cse/heqj3/owrl`, `/gscratch/cse/xunmei`, `gpu-l40s-krishna`, `gpu-l40s`, `/tmp/`, `module load`, `gcc/13.2.0`.

## 4. The two arms to run there (identical to the klone twins)
- owrl_base41:    JUDGE_MODEL=gpt-4.1, NUM_ROLLOUT=60                                    (OpenWebRL native recipe on our infra = baseline)
- owrl_partial41: JUDGE_MODEL=gpt-4.1 JUDGE_PARTIAL_MODEL=gpt-4.1 JUDGE_PARTIAL_CAP=0.8 NUM_ROLLOUT=60
Then: convert_hf (iter 30/59) → OM2W eval (300 tasks, greedy, 30 steps) → rejudge luna + gpt-5.2 → v2 score.

## 5. Transfer plan
klone and Tillicum share no filesystem. Options: Globus (UW Hyak endpoints) for the 31 GB; or `rsync -av --partial` login→login.
Order: port/ + OpenWebRL/ + build/Megatron-LM (1 GB, start editing) → ms-playwright (0.65 GB) → model (9 GB) → env (20 GB, or rebuild) → hf_ckpts (optional).

## 6. Unknowns to fill from tillicum_probe.sh
account / partition names and GPU type (H100 80 GB vs H200 141 GB) · cores and RAM per GPU node · OS + glibc + which Chromium libs are missing ·
module list (gcc, cuda) · /tmp size on GPU nodes · scratch path + quota (space AND file count) · home quota ·
egress from a GPU node (curl https://www.google.com, https://api.openai.com/v1/models, https://api.wandb.ai) and whether a proxy is required ·
max walltime per partition · whether SLURM allows --gpus=4 sub-node jobs
