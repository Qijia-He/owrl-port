# Actor output-length sensitivity: 1,024 versus 2,048 tokens

Status on 2026-10-14: job 360836 started inference successfully and finished seven native episodes (five TextWorld greedy and two WebShop greedy). WebShop task `webshop/4843` then exhausted the fixed actor context after 94 executed steps. The supervisor stopped both workers and released the GPU after 20 minutes 28 seconds. No Sol-selection episodes had started. CPU-only job 361085 passed 57 tests, and job 361087 preserved and reclassified the context-failure evidence. Inference resume job 361088 has been submitted for the remaining 32 task-condition attempts. Iterative training and training-data collection remain paused pending task selection.

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

## Validation and execution

The CPU validation checked configuration isolation, native action grammar, reference prompt alignment, task isolation, episode auditing, transport behavior, and both source task manifests. It did not execute any new environment episodes.

The original 43 tests did not cover the compute-node proxy environment. Job 360313 successfully initialized vLLM, but the supervisor sent requests for `127.0.0.1` through the inherited HTTP proxy and never recognized the running server. The job exited after 15 minutes 7 seconds, with zero episodes and no Sol calls; its GPU allocation was released. At the quoted $0.90 per H200-hour rate, this unsuccessful attempt used approximately $0.23 of GPU time.

CPU-only job 360830 reproduced HTTP 503 with environment proxy routing and HTTP 200 with a direct connection to the same loopback service inside Apptainer. Local inference and readiness requests now explicitly bypass proxies only for loopback hosts; external API routing remains unchanged. A regression test reproduces a proxy environment that excludes `localhost` but not `127.0.0.1`. Readiness probes now record HTTP status or connection-error type, the startup deadline is five minutes, and execution receipts are preserved per attempt. These are infrastructure repairs; no scientific parameter changed.

The inference job requests one H200, four CPUs, and at most four hours under normal QoS. The model server binds to loopback, runs inside the same Slurm allocation as the environment workers, and exits when they finish or fail. No GPU allocation remains reserved for subsequent training. Estimated maximum GPU cost from Slurm is $3.60; actual elapsed allocation time determines billing. Sol charges are logged separately.

## Context exhaustion and accounting clarification

In job 360836, the actor rejected the 95th decision request for WebShop greedy task `webshop/4843`: its prompt contained at least 260,097 tokens and the request reserved 2,048 output tokens, exceeding the unchanged 262,144-token context. Of the preceding 94 responses, 89 ended at the 2,048-token output cap. This demonstrates that increasing the output allowance does not necessarily eliminate truncation or growing-history failures; it does not yet establish a success-rate comparison.

The runner originally treated every HTTP 400 as an infrastructure error and stopped both environment workers. Accounting policy `fixed-actor-context-failure-v1`, introduced after observing this error, distinguishes this explicit local-model length rejection from unrelated errors. Such an attempt has status `context_exhausted`, success false, and the last observed native reward. It remains in the fixed 10-task denominator and is listed separately from native completed episodes. The runner continues to the next task. Unrelated service errors still stop the job for diagnosis.

This is a post-observation accounting clarification, not a preregistered rule. The original failure summary, infrastructure event, and full trajectory are retained. A recovery receipt records their hashes and the reclassification, without rerunning 94 expensive steps. The seven completed episodes are retained unchanged. The interrupted TextWorld task restarts from its native reset with the same seed. No prompt history or image is removed, no output/context limit is adjusted, and no replacement task is selected.

The validation includes a simulated episode that reaches the context limit and verifies that later tasks still execute, a second invocation that executes no already-finalized tasks, strict recognition of local context errors, preservation of original recovery evidence, rejection of inconsistent audit records, and retention of failed attempts in the reporting denominator. This is runner validation; it does not substitute for the remaining native environment runs.

The job will write an audited comparison of success, native reward, selected-response truncation, invalid actions, and paired differences. With only 10 tasks per environment, confidence intervals and success differences are exploratory.

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
- Live execution receipt once the job starts: `artifacts/output-length/v4-sol-nativecmd-output2048-diagnostic10/execution.json`
- Automatic final report: `artifacts/output-length/v4-sol-nativecmd-output2048-diagnostic10/comparison.md`

The 1,024-token results in the completed seed-42 report remain the original version, not a mixture of output limits.
