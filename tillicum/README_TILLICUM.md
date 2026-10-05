# OpenWebRL on Tillicum — runbook (2026-10-05)

Probe facts (from tillicum_probe.sh, 2026-10-05): account `zixianma`, partition `gpu-h200` (23 nodes × 8 H200 143 GB),
QOS debug/interactive/normal; Rocky 9.8 / glibc 2.34; modules `conda/Miniforge3-25.3.1-3`, `gcc/13.4.0`; no nvcc module
(the env ships CUDA 12.8); GPU nodes have direct egress (google/amazon/openai/pypi reachable, no proxy); all 16 Chromium
libs present; /tmp 3.5 TB on GPU nodes; storage: home 10 GB (86 % used — never write there), /gpfs/projects/krishna 1 TB
(97 % used — unusable), /gpfs/scrubbed/heqj3 100 TB (everything goes here; check its purge policy — klone's is 21 days).

Layout on Tillicum — two roots:
- `PORT_ROOT=/gpfs/scrubbed/heqj3/owrl` (purged): the git clone (code) + `outputs/` (checkpoints) + `logs/`.
- `PERSIST_ROOT=/gpfs/projects/krishna/heqj3/owrl_persist` (not purged): only `envs/webrl` (18 GB trimmed) — the one piece
  that takes hours to rebuild. `PORT_ROOT/envs` is a symlink to it. projects/krishna had 26 GB free on 2026-10-05, so the
  model (9 GB) and browsers (0.6 GB) stay on scrubbed; if purged, `setup_env_tillicum.sh` re-downloads them in minutes
  (it skips the env when it already exists).

## Where each piece comes from (scrubbed is purged, so nothing there is the master copy)
| piece | master | how it gets onto Tillicum |
|---|---|---|
| code (OpenWebRL, OpenWebRL_eval, build/Megatron-LM, port/, tillicum/) | private GitHub repo `Qijia-He/owrl-port` (synced from klone with `tillicum/sync_to_github.sh`) | `git clone` |
| conda env (18 GB, PERSIST_ROOT on projects/krishna) | rebuildable | `tillicum/setup_env_tillicum.sh` (2–4 h compile) — or rsync klone's env + `fix_env_paths.sh` (faster, needs klone reachable) |
| SFT model (9 GB, scrubbed) | HuggingFace `OpenWebRL/OpenWebRL-4B-SFT` | downloaded by setup_env_tillicum.sh |
| Chromium bundle (scrubbed) | playwright CDN | `playwright install chromium` (in setup_env_tillicum.sh) |
| secrets.env | you | typed by hand, mode 600 |
| checkpoints / outputs | nowhere else | back up `iter_*` you care about to klone (`/gscratch/krishna/heqj3/owrl_ckpt/`) before the purge window |

## Steps (all on Tillicum)
```bash
# 0. still unknown: cores / RAM per GPU node, max walltime, purge window -- paste these before step 5
scontrol show node g006 | grep -E "CPUTot|RealMemory|Gres="
scontrol show partition gpu-h200 | grep -E "MaxTime|DefMemPerGPU|MaxMemPer"
cat /gpfs/scrubbed/README* 2>/dev/null | head

# 1. code from GitHub (HTTPS; use a personal access token or `gh auth login`)
mkdir -p /gpfs/scrubbed/heqj3 && cd /gpfs/scrubbed/heqj3
git clone https://github.com/Qijia-He/owrl-port.git owrl && cd owrl && mkdir -p logs outputs

# 2. secrets: type the two lines by hand (same values as klone ~/.bashrc line 59 / owrl/secrets.env)
install -m 600 /dev/null secrets.env && $EDITOR secrets.env      # export OPENAI_API_KEY=...  export WANDB_API_KEY=...

# 3a. env + model + browsers, rebuilt on a GPU node (self-contained; ~3 h)
srun -A zixianma -p gpu-h200 --gpus=1 -c 32 --mem=200G --time=06:00:00 bash tillicum/setup_env_tillicum.sh 2>&1 | tee logs/setup_env.log
# 3b. OR copy klone's env/model/browsers (one Duo prompt, ~30 GB) and patch the env's absolute paths
bash tillicum/pull_from_klone.sh && module load conda/Miniforge3-25.3.1-3 gcc/13.4.0 && bash tillicum/fix_env_paths.sh

# 4. GPU-node preflight (CUDA, JIT, Chromium on a live site, OpenAI, wandb, disk) -- must print PREFLIGHT PASSED
srun -A zixianma -p gpu-h200 --gpus=1 -c 8 --mem=64G --time=00:30:00 bash tillicum/preflight_gpu_node.sh

# 5. set --cpus-per-task / --mem in tillicum/run_owrl_tillicum.sbatch from step 0, then submit both arms
bash tillicum/launch_arms.sh            # or: bash tillicum/launch_arms.sh partial41
```
Later code changes: edit on klone → `bash tillicum/sync_to_github.sh "msg"` → `git pull` on Tillicum. Editing on Tillicum and pushing works too; just pull on klone before the next sync (the sync rsyncs klone → snapshot and would overwrite).

## What the sbatch does (same supervisor as klone)
`run_owrl_tillicum.sbatch` → `step3_train.sh` (preflight, pruner keep-every-10, resource sampler) → `run_8xl40s_interactive.sh`
(caches → /tmp/$USER, playwright staged to /tmp, proxies unset, judge env, GPU split 4 rollout + 4 train, TP=2) →
`OpenWebRL/scripts/run_browser_full.sh` → ray. Auto-resumes from `outputs/$EXP/ckpt/latest_checkpointed_iteration.txt`;
restarts the launcher every 14 iters or when host RSS > MEM_RESTART_GB. One arm per node; the two arms are
`owrl_base41_tlc` (gpt-4.1 binary) and `owrl_partial41_tlc` (+ gpt-4.1 partial credit, cap 0.8), 60 iters each.

## Sizing on 8×H200 (vs the klone 4×L40S twins)
- GPU memory is 3× L40S; keep the klone split (4+4, TP=2) — it is infrastructure, not recipe. Could later try 2 rollout + 6 train or two arms on one node.
- Throughput ceiling is still the browser pool: BROWSER_SLOTS ≈ 0.75 × cores allocated (48 on 64 cores). Raise it if the node has 128+ cores.
- RAM: launcher RSS grows ~14 GiB/iter; ask ≥ 900 GB if the node has it, else lower MEM_RESTART_GB to ~0.85 × --mem.
- Disk: 58 GB per checkpoint, pruner keeps every 10th + latest → ~500 GB per arm; preflight demands ≥ 1.5 TB free under outputs/.
- Expected: rollout ~15–25 min/iter → 60 iters ≈ 1 day per arm (klone twins: 26–41 min/iter).

## Known differences to watch on the first iteration
- `module load gcc/13.4.0` replaces klone's gcc/13.2.0 (needed for flashinfer/triton JIT; env check in step3 fails fast if g++ < 9).
- NVLink present on H200: TP=2 traffic no longer crosses PCIe; NCCL fallbacks in the launcher stay commented out.
- If `[hyak] playwright staged` is missing from the log, the bundle path is wrong (PLAYWRIGHT_BUNDLE=$PORT_ROOT/ms-playwright).
- Judge calls: watch for `[PartialCredit]` lines in partial41's run.log and `Browser reward:` lines in both; zero `Server process terminated` lines.
- wandb project `openwebrl`, run names = EXP_NAME; needs WANDB_API_KEY in secrets.env (WANDB_MODE=online in the sbatch).

## After the arms finish
`port/convert_hf.sbatch` (CONVERT_SPECS="tag=/abs/iter_dir", --gpus=1) → `port/eval_om2w.sbatch` → `port/rejudge.sbatch` (luna, gpt-5.2) →
`compute_eval_success_rate_woaborted_v2.py`; these still carry klone paths/partitions — port them the same way (sed PORT_ROOT, account, partition) when needed.
