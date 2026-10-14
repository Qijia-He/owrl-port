# Decisions, limitations, and next steps

Updated 2026-10-14. The [main report](README.md) is the self-contained explanation of the project. This file preserves the decisions and operational history behind it.

## Current scope

- Complete: six environment–actor combinations, four conditions, 50 tasks per condition, execution seed 42; 1,200 complete episodes.
- Complete: all 120 smoke episodes reused as part of the same task manifests, with no duplicate counting; trajectory reconciliation and all six expansion audits passed.
- Complete: the separate 2,048-token diagnostic, 40 fixed attempts (39 native completed episodes and one context-exhausted failure), both audits passed. Its GPU allocation has been released. Seeds 43 and 44 remain deferred because the user requested a summary after one seed.
- TextWorld and WebShop are selected for iterative selector training. Native-input SFT/DPO/deployment checks passed; initial research-data collection is submitted as job 361267. Full training and official held-out/test evaluation have not started.
- Earlier upstream inference checks and two optimizer steps per training recipe were installation checks, not full ARM training or a reproduction of all upstream results.

## Interpretation of the completed pilot

The clearest gain is **TextWorld × Qwen3.5**: success increases from 46% with random-of-five to 70% with Sol, a paired difference of +24 percentage points with a task-bootstrap interval of [+10, +38]. The supplementary task-group bootstrap also remains above zero.

The other five comparisons with random-of-five have positive point estimates, but intervals include or touch zero. This is insufficient evidence of a reliable gain across environments or execution seeds.

WebShop native mean reward increases from 0.5303 to 0.5893 for Qwen3.5, and from 0.6146 to 0.6542 for Qwen3-VL. Both paired reward intervals include zero. Report the estimates and uncertainty together.

## Limitations to investigate

| Observation | Evidence | Consequence for the next experiment |
|---|---|---|
| Qwen3.5 frequently exhausts its response budget | TextWorld selected-response truncation: random-of-five 35.7%, Sol 10.8%; in THOR, all five responses are truncated in 31.4% of Sol-visited pools | Separate avoiding unusable responses from better semantic choices. Any output-limit change requires a new version. |
| Qwen3 / Qwen3-VL often repeat the same action | All five actions identical in Sol-visited pools: TextWorld Qwen3 74.6%, THOR Qwen3-VL 92.9%, WebShop Qwen3-VL 71.8% | Check whether useful alternatives exist before expanding annotation. |
| Most THOR episodes hit the step cap | With Sol: Qwen3.5 82%, Qwen3-VL 86% | Inspect repeated actions, perception/navigation, and controller feedback. Do not infer end-to-end visual-control ability from high-level selection. |
| One execution seed and a small training-reserve sample | 50 tasks per cell; some ALFWorld tasks share a group | Treat this as exploratory. Task bootstrap does not measure execution-seed variation. |
| Extra selection compute is not cost-matched | Both five-candidate policies propose five actions, but only one calls Sol | Do not claim budget-optimal inference or equivalence to pass-at-five evaluation. |

Candidate diagnostics describe the states each policy actually visited. They are not counterfactual rollouts, oracle action-quality labels, or a causal decomposition of the gain.

## Research sequence

1. Analyze existing successes and failures to distinguish candidate quality, selection quality, and execution failures.
2. Use the completed, bounded 2,048-token diagnostic with Greedy as the reference to choose a future output-budget protocol. Retain the original 1,024-token snapshot unchanged; do not automatically expand.
3. Decide which settings justify ARM labels and training. Use the same public prompt convention for new teacher labels, training examples, and inference.
4. Evaluate on official held-out/test tasks once the protocol is fixed. Seeds 43 and 44 remain deferred.

## Method alignment

The [main report's method section](README.md#2-how-does-the-method-work) identifies Piotr's original implementation, Zixian's integration, and our changes. Its [upstream limitations section](README.md#upstream-limitations-we-found) explains the missing historical dependency, backbone-description mismatch, and released-data/public-prompt mismatch.

The completed pilot follows the public selection builder and inference helpers. Native benchmark observations/actions, Qwen actor checkpoints, and Sol are explicit adaptations. We do not claim byte-identical recovery of the unpublished historical teacher pipeline.

## 2026-10-06: completion and recovery

- Restricted execution to seed 42 following the user's instruction; no other seed generation was started.
- Recovered from one local inference connection interruption using bounded transport retries with the same seeded payload.
- Recovered from one Sol HTTP 500 interruption; retained request retry limits, explicit error records, and bounded shard recovery.
- Verified that all 1,137 / 1,186 completed summary hashes present before the respective recoveries remained unchanged.
- Retained interrupted attempts in operational/cost records; they were not counted as benchmark failures. Incomplete episodes were rerun from their start.
- Reconciled all 1,200 trajectories and summaries, passed all six protocol audits, froze the results, and released experiment resources.
- Established a Markdown log with tables rendered from frozen exported JSON.

Earlier prompt-grammar, serving-context, serving-mode, and model diagnostics are retained separately and excluded from this result version.

## 2026-10-07: publication and reader-facing report

- The user selected `Qijia-He/owrl-port`; documentation lives under `docs/action-reward-experiments/`, linked from the repository homepage.
- Rewrote the log in English and made the main README a complete introduction for a reader unfamiliar with the project.
- Added method attribution with pinned source citations, benchmark setup, baseline rationale, results, uncertainty, implementation status, and next steps to that single report.
- The table renderer preserves the explanatory narrative and updates only explicitly marked numerical blocks in the main report.
- This documentation update does not change the frozen numerical data or authorize additional experiments.

## 2026-10-14: output-length diagnostic and context exhaustion

- The user requested checking a larger generation limit before deciding on training data. The diagnostic changes the actor cap from 1,024 to 2,048 for frozen Qwen3.5-4B, TextWorld/WebShop, Greedy/Sol-of-five, and the first 10 original pilot tasks per environment, seed 42: 40 fixed task-condition attempts. Historical comparisons use exactly those same tasks. No extra seed or automatic expansion follows.
- Job 360313 failed before any episode because an inherited HTTP proxy intercepted loopback readiness probes. CPU-only job 360830 reproduced the fault. A loopback-only bypass passed 45 tests in job 360831.
- Job 360836 completed five TextWorld greedy and two WebShop greedy episodes. A third WebShop greedy task executed 94 steps, with 89 responses truncated at 2,048 tokens, then exceeded the unchanged 262,144-token actor context. Both workers stopped and the GPU was released. No Sol-selection episode had begun.
- Accounting policy `fixed-actor-context-failure-v1` was clarified after this observed error: explicitly identified actor context exhaustion is a separate failed termination, remains in the fixed task denominator, preserves the last native reward, and does not stop later tasks. Generic infrastructure errors still stop execution. No prompt/image/limit/task-list change is made to avoid the failure.
- Job 361085 passed 57 validation tests, including continuation, resume, audit rejection, recovery evidence, and denominator checks. CPU-only job 361087 preserved and hashed the original failed summary and trajectory before reclassifying that recorded attempt. Inference resume job 361088 was submitted; the seven completed episodes and one context failure are retained.
- Continuous monitoring followed resume through completion at 19:58 UTC. Job 361088 exited 0 after 53m 12s and released its GPU. Both environment audits passed; all 40 fixed attempts are accounted for, with zero teacher parse fallbacks and native WebShop images verified.
- On the matched ten-task subset, TextWorld Greedy/Sol success changed from 50%/70% at 1,024 tokens to 60%/80% at 2,048. WebShop changed from 20%/40% to 30%/40%. Sol-minus-Greedy point estimates stayed +20 pp in TextWorld and narrowed from +20 to +10 pp in WebShop; intervals remain wide.
- Greedy truncation decreased, but Sol selected-response truncation increased in both environments. WebShop Sol mean reward decreased from 0.5813 to 0.5413. These trajectory-level rates describe different visited states and do not isolate a causal mechanism.
- New estimated costs: Sol $6.4198; the three one-H200 allocations, including failures, $1.3318; combined $7.7516. The [final diagnostic report](runs/v4-sol-nativecmd-output2048-diagnostic10.md) and [portable metrics](diagnostics/v4-sol-nativecmd-output2048-diagnostic10.json) preserve the results and accounting caveats.
- ARM training/data collection remain paused until task selection. No automatic expansion or extra seed followed completion. The original published 1,200-episode snapshot is unchanged.

## 2026-10-14: literature audit of 1,024 and 2,048 output tokens

- Added the [model/length/success-rate audit](literature-length-alignment.md), separating prompting baselines from task-trained actors and WebShop success from continuous reward.
- Checked papers and pinned public code. A paper's training hyperparameters do not by themselves establish every baseline's evaluation settings. ICRL's current code overrides the configured per-turn output cap with the remaining context budget and may critique/retry during evaluation; historical execution remains unverified.
- Retained differences in checkpoint identity, thinking, images, task split, action lists, catalog/goals, and retry policy. No cross-paper score difference is interpreted as the effect of changing output length.
- Kept Greedy as our primary baseline. The review changes no frozen data, experiment configuration, task choice, or training scope. No compute allocation or new experiment was started for the audit.

## 2026-10-14: iterative training authorized for TextWorld and WebShop

- The user selected initial selector SFT followed by iterative refresh/DPO for these two environments. Keep the Qwen3.5 actor and 1,024-token candidate limit fixed; retain native WebShop images. Initial data target: 1,500 training states and 150 development states per environment, followed by two 750-state refreshes per environment, replay, and SFT controls. Separate versus joint selector checkpoints is being clarified.
- Frozen manifests exclude reserved pilot groups and official evaluation tasks. Native-input encoding passed, with 0/10-image train/deployment prefix parity. Job 361256 failed before optimization because the container lacked git, then released its GPU. CPU job 361258 passed after checkout verification was fixed. Job 361259 passed 67 checks, three SFT/DPO optimizer steps each, reference integrity, and deployment; its GPU was released. Initial per-environment collection is submitted as job 361267 and does not depend on the separate/joint checkpoint choice.
- Probe weights are discarded as research initializers. The [training record](iterative-training-plan.md) distinguishes planned counts from collected data and functional validation from scientific results. No historical numerical result was changed.
