# owrl-port — OpenWebRL GRPO training port for UW Hyak (klone L40S / Tillicum H200)

Code half of `/gscratch/cse/heqj3/owrl` (klone). Layout = `PORT_ROOT`:

- `OpenWebRL/` — OpenWebRL @ 05f8ed4 + cluster patches + our reward changes (`openwebrl/reward_browser.py`: judge pair, partial credit; `slime/rollout/sglang_rollout.py`: dropped-group dumps); training data in `openwebrl/data/`.
- `OpenWebRL_eval/` — same base + David's Online-Mind2Web eval scripts (`compute_eval_success_rate_woaborted_v2.py`).
- `build/Megatron-LM/` — Megatron-LM @ 3714d81 + OpenWebRL's docker patch + `torch.save` retry in `dist_checkpointing/strategies/common.py`.
- `port/` — klone launch chain (sbatch supervisor → step3 → launcher), conversion / eval / rejudge / partial-credit tools, runbooks.
- `tillicum/` — the Tillicum port: `README_TILLICUM.md` (start here), `setup_env_tillicum.sh`, `fix_env_paths.sh`, `preflight_gpu_node.sh`, `launch_arms.sh`, ported sbatch/step3/launcher, `requirements_freeze_klone.txt`.

Not in git (rebuilt or copied): `envs/webrl` (conda, 20 GB — `tillicum/setup_env_tillicum.sh`), `models/OpenWebRL/OpenWebRL-4B-SFT` (HF), `ms-playwright/` (`playwright install chromium`), `secrets.env` (OPENAI_API_KEY, WANDB_API_KEY; mode 600), `outputs/` (checkpoints), `logs/`.

Synced from klone with `tillicum/sync_to_github.sh`.

## Action-level reward experiments

[Read the complete research report: motivation, method and citations, setup, results, limitations, and next steps](docs/action-reward-experiments/README.md).

The log records the ALFWorld TextWorld, ALFWorld THOR, and WebShop Sol-selection pilot. Seed 42 is complete; subsequent experiment versions will be appended to the same directory.
