# Official-split success evaluation of the trained selectors

## Status: submitted to Slurm, October 21, 2026

All four trained selectors are complete. Their task-success rates are **not available yet**. CPU preparation job **377085** passed, including 11 evaluation contract tests, checkpoint checksums, task separation and official task enumeration. Slurm controller job **377090** has been submitted. It becomes eligible at **12:04:49 Pacific on October 21**; this is an eligibility time, not a guaranteed GPU start time.

The controller first checks project quota, then submits a one-GPU deployment check. If that succeeds, it submits the eight environment/method evaluation lanes, with at most four active GPU jobs. Closing a laptop or terminal does not stop the workflow. No new evaluation GPU job was running at the time of this update.

The short delayed eligibility allows GPFS quota accounting to refresh after a verified lossless archive freed **1.922 GiB**. The existing **4 GiB free-space guard** remains in force. A failed quota check stops GPU submissions; it does not hold GPUs while waiting for storage.

## What question does this answer?

Can a trained local selector improve the success of a frozen Qwen3.5-4B actor, and how much of Sol's improvement does it retain? We compare four complete agent policies on the same tasks:

| Method | Actor generations at each step | Action executed |
| --- | --- | --- |
| Greedy | One deterministic generation | That generation's native action |
| Sol of 5 | Five independent sampled candidates | Sol's selected candidate |
| SelectionARM of 5 | The same sampling protocol | The trained generative selector's candidate |
| Selection head of 5 | The same sampling protocol | The trained listwise head's highest-scoring candidate |

After a policy chooses a different action, its subsequent environment state and candidate sets can differ. This is closed-loop policy evaluation, with pairing by task, not a comparison that forces different policies to share a single trajectory.

The main comparison is each method's **task-success difference versus Greedy**. Training losses and agreement with Sol are separate metrics, not task success. See [the training report](selector-training-results.md) for the four checkpoint results and upstream method references.

## Which tasks, and why not 50 per environment?

The earlier 50-task experiments used a reserve inside the training split. They were pilot experiments, not official test results. The present experiment uses the complete following evaluation sets:

| Environment | Official split | Tasks per method | Maximum executed actions |
| --- | --- | ---: | ---: |
| ALFWorld TextWorld | `valid_unseen` | 134 | 50 |
| WebShop | `test`, goal indices 0–499 | 500 | 30 |

This is **(134 + 500) × 4 methods × 1 execution seed = 2,536 evaluation episodes**. Execution seed is **42 only**. Seeds 43 and 44 remain deferred. ALFWorld also provides 140 `valid_seen` games; they are not included in this run. [ALFWorld official dataset description](https://github.com/alfworld/alfworld/blob/master/alfworld/data/README.md), [WebShop official split code](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/env.py).

The official task splits do not imply that every other setting matches every published leaderboard. In particular, WebShop's 30-action cap follows the user's requested protocol. Historical 100-action pilot results are not reused as controls.

The **500 development examples per environment used during training are decision states**, not 500 test episodes. They served checkpoint selection. Their task IDs do not overlap this evaluation. The official WebShop split is by goal index, not a disjoint-product split: the audit finds 14 shared product groups with our training/development cohort, but **zero shared task IDs**. We retain the official 500 tasks and report this fact rather than removing tasks or silently changing the benchmark.

## Frozen models and inputs

The actor remains `Qwen/Qwen3.5-4B`; its weights are unchanged. Each environment uses its separately trained SelectionARM adapter and listwise head from `v2-listwise-sol-preaction`. The teacher is the existing pinned `gpt-6.1-sol` configuration. Both learned methods trained on the same 5,000 training and 500 development panels per environment, labeled by Sol before action execution. These are not K=1 lookahead labels, and this run performs no actor SFT, DPO or iterative retraining.

Actor sampling preserves temperature 0.7, top-p 0.9, top-k disabled, thinking enabled, five independent candidate calls and a 1,024-token output cap. Greedy uses temperature zero and one generation. Actor context capacity remains 262,144 tokens. No invalid or truncated candidate is repaired or resampled.

TextWorld receives its native text observations. WebShop uses the full catalog and human goals, and passes every available native visible product image, including search-result images. Full actor history is preserved. The selector prompt uses the already pinned public Piotr/Zixian renderer, with its original selector-history convention; this must not be confused with the actor's full history.

The generative selector uses the same HF BF16/SDPA loading and deterministic, non-thinking, 32-token decoding as its post-training reload probe. It has no JSON grammar constraint. A malformed selection explicitly falls back to candidate 1 and is counted. This deployment recipe is recorded in the new experiment version rather than silently equated with the older vLLM structured-output selector.

The head uses the unchanged, full-history reference feature extractor, including exact actor prompt and candidate token IDs. It recomputes features using the frozen backbone; this run **does not establish the latency or FLOP savings of online hidden-state reuse**. Actor serving and the HF selector share one allocated H200. The actor's vLLM memory fraction is fixed at 0.60 across methods.

## Checks before reporting success

Deployment validation recomputes frozen development examples and checks predictions against the accepted checkpoints, including native-image examples. It then runs one complete reserved training-pilot episode for each learned environment/method combination. Those four deployment episodes are stored separately and **excluded from the 2,536 official evaluation episodes**.

Each evaluation lane audits its first five episodes before continuing. Those five belong to the official task list and remain in that lane's final denominator. A policy's failure to navigate to an image-bearing page is recorded as policy behavior, not used to replace the task.

The runner durably records each selected action before execution. After a planned interruption it resets the native environment and replays recorded actions, checking public observations, image hashes and native outcomes, then continues without reselecting already recorded decisions. Every executed action must belong to the offered candidate set.

## Reporting and recovery

- ALFWorld success uses the official environment's `won` flag. WebShop success requires native reward exactly 1; mean native reward is reported separately.
- Final denominators remain 134 or 500. Context-budget exhaustion counts as failure. Infrastructure errors remain unresolved coverage and prevent a partial subset from being presented as a final success rate.
- Differences versus Greedy use 10,000 paired task-bootstrap resamples and a 95% percentile interval. This interval describes uncertainty across tasks for the one execution seed; it does not measure across-seed variation.
- The report also records invalid actions, candidate and selected-action truncation, selector fallbacks, step counts, known API charges and actor/selector latency. The known API ledger is separate from GPU billing.
- GPU workers have an eight-hour allocation and a seven-hour soft deadline. Explicit checkpoint exit 75 triggers bounded continuation through a CPU Slurm callback. OOM, unexpected exceptions and other hard failures stop that lane for diagnosis instead of retrying blindly. Independent lanes can continue.
- Results and compressed trajectories are stored on project storage. Recomputable temporary feature inputs go to scratch. The controller regenerates `results.json` and `results.md` as jobs finish.

## Locations

- Code: `arm-generalization/experiments/selector-training/evaluation/`
- Frozen protocol and outputs: `arm-generalization/artifacts/selector-evaluation/v2-official-text134-web500-text50-web30-1024-seed42/`
- Protocol/task/checkpoint lock: `study.json`
- Workflow state: `workflow.json` (created when the controller starts)
- Job submissions and logs: `jobs/`
- Official episode results: `episodes/<environment>/<method>/`
- Separate deployment probes: `deployment-episodes/`
- Aggregated results: `results.json` and `results.md` (created by controller callbacks)

This page records a submitted experiment, not completed success rates. The historical pilot table is unchanged.
