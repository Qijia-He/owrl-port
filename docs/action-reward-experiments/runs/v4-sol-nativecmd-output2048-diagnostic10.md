# Actor output-length sensitivity: 1,024 versus 2,048 tokens

Completed and audited on **2026-10-14 at 19:58 UTC**. All **40 fixed task-condition attempts** are accounted for: **39 native completed episodes and one context-exhausted failure**. Both environment audits passed. Job 361088 exited successfully after 53 minutes 12 seconds and released its GPU. Iterative training and training-data collection remain paused pending task selection.

**Answer:** Increasing the actor cap improved Greedy by one success out of ten in each environment. Sol improved by one success in TextWorld and had identical task-level success outcomes in WebShop. Sol's advantage over Greedy stayed at +20 percentage points in TextWorld and narrowed from +20 to +10 points in WebShop. These are exploratory point estimates on ten tasks, with wide uncertainty. A larger cap did not eliminate truncation or uniformly improve the results.

Recovery history: job 360836 finished seven native episodes before WebShop task `webshop/4843` exhausted the fixed actor context after 94 executed steps. The supervisor stopped both workers and released the GPU after 20 minutes 28 seconds. No Sol-selection episodes had started at that point. CPU-only job 361085 passed 57 tests, and job 361087 preserved and reclassified the context-failure evidence. Job 361088 resumed the remaining 32 attempts without rerunning the seven completed episodes or the recorded context failure.

## Question

Does giving Qwen more output tokens reduce truncated actions, improve greedy success, and change the benefit of Sol selecting among five candidates?

The existing comparison remains valid at its declared output cap. This experiment measures sensitivity to that cap. It does not assume that longer output will help, or that avoiding truncated actions is an illegitimate benefit of selection.

## Fixed scope

- Actor: frozen Qwen3.5-4B.
- Environments: native ALFWorld TextWorld and WebShop, preserving all available native images.
- Conditions: greedy single-action decoding and Sol selection from five stochastic Qwen candidates.
- Tasks: the first 10 tasks in each original, audited 50-task pilot manifest, chosen by fixed order independently of previous outcomes.
- Execution seed: 42 only.
- Fixed task-condition attempts: 2 environments × 2 conditions × 10 tasks = 40. Native completed episodes and attempts ending in actor context exhaustion are reported separately.
- Changed parameter: actor `max_tokens`, 1,024 → 2,048, for both conditions.
- Preserved parameters: candidate count, sampling settings, prompts, model weights, 262,144-token serving context, Sol settings, native observations/actions/rewards, and episode step limits.
- Sol's completion cap remains 2,048 tokens. Training `cutoff_len` is unrelated and unchanged.
- Historical comparisons use exactly the matching 10 tasks, never the full 50-task average.
- No automatic expansion, additional seed, official test evaluation, or training follows this diagnostic.

The original results remain immutable. New results and API accounting use a separate project-storage directory. The public prompt-rendering protocol stays the same; the run configuration records the output-length variant explicitly.

The method is unchanged from the [main report](../README.md#2-how-does-the-method-work): frozen Qwen proposes actions, Sol chooses an existing candidate index, and only that action executes. The adapter imports [Piotr's public selection prompt](https://github.com/piotr-teterwak/action-reward-models/blob/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce/inference/selection_prompt.py) and [Zixian's inference helpers](https://github.com/zixianma/OpenWebRL/blob/134f064d867170eb833f6ab0f668a1b05050c031/openwebrl/arm_inference.py). Greedy uses temperature 0; Sol's five proposals use temperature 0.7, top_p 0.9, top_k -1, and thinking enabled. Duplicate and truncated candidates are retained without resampling. The selector is the same `gpt-6.1-sol` API alias, with its original settings. This diagnostic runs no SFT, DPO, or actor training.

## Success: compare Sol with Greedy

Every cell uses the **same ten tasks in its environment**. A ten-point change is one additional success in this subset; it is not a ten-percent relative improvement.

| Environment | Actor output cap | Greedy | Sol of 5 | Sol minus Greedy: pp [95% CI] |
|---|---:|---:|---:|---:|
| ALFWorld TextWorld | 1,024 | 5/10 (50%) | 7/10 (70%) | +20 [0, +50] |
| ALFWorld TextWorld | 2,048 | 6/10 (60%) | 8/10 (80%) | +20 [-20, +60] |
| WebShop | 1,024 | 2/10 (20%) | 4/10 (40%) | +20 [0, +50] |
| WebShop | 2,048 | 3/10 (30%) | 4/10 (40%) | +10 [0, +30] |

| Environment | Condition | Success change, 2,048 minus 1,024: pp [95% CI] |
|---|---|---:|
| ALFWorld TextWorld | Greedy | +10 [-20, +40] |
| ALFWorld TextWorld | Sol of 5 | +10 [0, +30] |
| WebShop | Greedy | +10 [0, +30] |
| WebShop | Sol of 5 | 0 [0, 0] |

Intervals use 10,000 task-paired bootstrap draws. The WebShop Sol cap-comparison interval is [0, 0] because all ten observed binary success outcomes match across caps; resampling those observed pairs cannot create a difference. It does **not** establish a zero effect on the wider task population. These intervals cover task resampling only, not execution-seed variation or changes/stochasticity behind the Sol API alias. The original 50-task averages are not substituted for these matched ten-task baselines.

## Truncation, invalid actions, and task length

“Candidate truncation” is the fraction of generated candidates ending at the output limit. “Selected truncation” is the fraction of executed decisions for which the chosen response was truncated. Invalid actions use the existing native-adapter diagnostics. Rates pool decisions within each condition; they are not averages over task-level rates. Context-exhausted attempts contribute their recorded steps and remain in the task denominator.

| Environment | Cap | Condition | Candidate truncation | Selected truncation | Invalid actions | Mean steps |
|---|---:|---|---:|---:|---:|---:|
| TextWorld | 1,024 | Greedy | 54.7% | 54.7% | 58.7% | 30.0 |
| TextWorld | 2,048 | Greedy | 49.6% | 49.6% | 58.0% | 27.4 |
| TextWorld | 1,024 | Sol of 5 | 21.8% | 8.2% | 33.7% | 24.3 |
| TextWorld | 2,048 | Sol of 5 | 18.3% | 12.4% | 14.3% | 21.7 |
| WebShop | 1,024 | Greedy | 80.5% | 80.5% | 80.5% | 34.8 |
| WebShop | 2,048 | Greedy | 31.5% | 31.5% | 31.5% | 28.9 |
| WebShop | 1,024 | Sol of 5 | 12.8% | 2.3% | 2.3% | 13.3 |
| WebShop | 2,048 | Sol of 5 | 21.7% | 7.0% | 7.0% | 22.7 |

Greedy truncation decreased in both environments, especially WebShop, but remained high. Sol's selected-response truncation increased in both environments despite the larger allowance. Different caps produce different actions and subsequent states, so these aggregate rates do not compare identical states or isolate a causal mechanism. Giving both conditions the same cap makes the declared comparison consistent; it does not guarantee that the cap affects them equally.

## WebShop native reward

Success requires reward 1; the native continuous reward also measures partial satisfaction of the shopping request.

| Cap | Greedy mean reward | Sol mean reward | Sol minus Greedy [95% CI] |
|---|---:|---:|---:|
| 1,024 | 0.4833 | 0.5813 | +0.0980 [+0.0267, +0.1810] |
| 2,048 | 0.5083 | 0.5413 | +0.0330 [0.0000, +0.0830] |

Increasing the cap changed Greedy reward by +0.0250 [0.0000, +0.0750] and Sol reward by -0.0400 [-0.1067, 0.0000]. Thus unchanged Sol success in WebShop does not mean every outcome stayed unchanged: its partial-credit reward decreased and its average task length increased.

## Interpretation and next decision

The diagnostic supports testing a stronger Greedy baseline: 2,048 tokens helped it on one extra task in each environment. It does not establish 2,048 as an optimal cap or show that longer output solves the remaining failures. Sol retains positive success-rate point estimates over Greedy, so the observed selection benefit did not disappear when the allowance increased. The small subset and wide intervals do not establish a reliable generalization result or prove which mechanism caused the gain.

The original 1,024-token, 50-task result remains a valid result at its declared budget. This diagnostic is a separate sensitivity check. No expansion, extra execution seed, training-data collection, or ARM training was launched after it. Task/data selection remains the next user decision.

## Cost and final audit

New Sol usage totaled **$6.4198**, matching the sum assigned to the finalized attempts: $2.4428 for TextWorld and $3.9771 for WebShop. The matched historical Sol subsets cost $1.9434 and $0.9757 respectively; those are old charges, not part of this run's bill.

| GPU job | Outcome | Elapsed |
|---|---|---:|
| 360313 | Startup/proxy failure; no episodes | 15m 07s |
| 360836 | Seven native episodes, then context rejection | 20m 28s |
| 361088 | Successful resume and final audit | 53m 12s |

The three one-H200 allocations total 88m 47s, or approximately **$1.3318** at the quoted $0.90/GPU-hour rate. Combined new API and GPU estimates are **$7.7516**. These are usage estimates, not invoices; the failed allocations are included. CPU-only checks and monitoring steps requested no additional GPUs.

The final audits account for 20 TextWorld attempts (20 native completed) and 20 WebShop attempts (19 native completed plus the one context failure), covering 1,007 executed decisions. They validate the task manifests, actor parameters, prompt/image hashes, and execution of candidates actually offered to Sol. TextWorld has no artificial images; WebShop exercised up to ten native images per observation. There were zero teacher parse fallbacks. The audits validate stored evidence; they do not independently replay every transition. The final Slurm job exited with code 0 and released its allocation.

The [portable diagnostic metrics and audit hashes](../diagnostics/v4-sol-nativecmd-output2048-diagnostic10.json) preserve all numerical values. The full local comparison artifact has SHA-256 `ce8dfa346f5d30994ae20970152804cc62e0d9adb046d342d2c91fd05d5e0702`. This diagnostic export is separate from the original pilot's frozen result format.

## Validation and execution

The CPU validation checked configuration isolation, native action grammar, reference prompt alignment, task isolation, episode auditing, transport behavior, and both source task manifests. It did not execute any new environment episodes.

The original 43 tests did not cover the compute-node proxy environment. Job 360313 successfully initialized vLLM, but the supervisor sent requests for `127.0.0.1` through the inherited HTTP proxy and never recognized the running server. The job exited after 15 minutes 7 seconds, with zero episodes and no Sol calls; its GPU allocation was released. At the quoted $0.90 per H200-hour rate, this unsuccessful attempt used approximately $0.23 of GPU time.

CPU-only job 360830 reproduced HTTP 503 with environment proxy routing and HTTP 200 with a direct connection to the same loopback service inside Apptainer. Local inference and readiness requests now explicitly bypass proxies only for loopback hosts; external API routing remains unchanged. A regression test reproduces a proxy environment that excludes `localhost` but not `127.0.0.1`. Readiness probes now record HTTP status or connection-error type, the startup deadline is five minutes, and execution receipts are preserved per attempt. These are infrastructure repairs; no scientific parameter changed.

Each inference job requested one H200, four CPUs, and at most four hours under normal QoS. The model server bound to loopback and ran inside the same Slurm allocation as the environment workers. The allocation was released when work ended. No GPU allocation remains reserved for subsequent training. Slurm quoted a maximum $3.60 per four-hour allocation; actual elapsed allocation times and the resulting estimate are listed above.

## Context exhaustion and accounting clarification

In job 360836, the actor rejected the 95th decision request for WebShop greedy task `webshop/4843`: its prompt contained at least 260,097 tokens and the request reserved 2,048 output tokens, exceeding the unchanged 262,144-token context. Of the preceding 94 responses, 89 ended at the 2,048-token output cap. This is an observed failure case, not a causal decomposition of the aggregate success-rate differences.

The runner originally treated every HTTP 400 as an infrastructure error and stopped both environment workers. Accounting policy `fixed-actor-context-failure-v1`, introduced after observing this error, distinguishes this explicit local-model length rejection from unrelated errors. Such an attempt has status `context_exhausted`, success false, and the last observed native reward. It remains in the fixed 10-task denominator and is listed separately from native completed episodes. The runner continues to the next task. Unrelated service errors still stop the job for diagnosis.

This is a post-observation accounting clarification, not a preregistered rule. The original failure summary, infrastructure event, and full trajectory are retained. A recovery receipt records their hashes and the reclassification, without rerunning 94 expensive steps. The seven completed episodes were retained unchanged. The interrupted TextWorld task restarted from its native reset with the same seed. No prompt history or image was removed, no output/context limit was adjusted, and no replacement task was selected.

The validation includes a simulated episode that reaches the context limit and verifies that later tasks still execute, a second invocation that executes no already-finalized tasks, strict recognition of local context errors, preservation of original recovery evidence, rejection of inconsistent audit records, and retention of failed attempts in the reporting denominator. All 57 checks passed before resume; the final native-trajectory audits subsequently passed as well.

The job wrote the audited comparison reported above. With only 10 tasks per environment, confidence intervals and success differences remain exploratory.

## Artifacts

- Configuration: `configs/output_length_2048.yaml`
- Launcher: `scripts/output_length_job.sbatch`
- Initial validation log: `reports/output-length-check-360316.out`
- Failed attempt log: `reports/output-length-360313.out`
- Loopback diagnostic: `reports/loopback-probe.json`
- Repair validation log: `reports/output-length-check-360831.out`
- Replacement job log: `reports/output-length-360836.out`
- Context-accounting validation log: `reports/output-length-check-361085.out`
- Context-recovery receipt: `reports/output-length-context-recovery.json`
- Resume job log: `reports/output-length-361088.out`
- Final execution receipt: `artifacts/output-length/v4-sol-nativecmd-output2048-diagnostic10/execution.json`
- Automatic final report: `artifacts/output-length/v4-sol-nativecmd-output2048-diagnostic10/comparison.md`

The 1,024-token results in the completed seed-42 report remain the original version, not a mixture of output limits.
