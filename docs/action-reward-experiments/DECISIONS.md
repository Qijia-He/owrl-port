# Decisions, limitations, and next steps

Updated 2026-10-16 UTC. The [main report](README.md) is the self-contained explanation of the project. This file preserves the decisions and operational history behind it.

## Current scope

- Complete: six environment–actor combinations, four conditions, 50 tasks per condition, execution seed 42; 1,200 complete episodes.
- Complete: all 120 smoke episodes reused as part of the same task manifests, with no duplicate counting; trajectory reconciliation and all six expansion audits passed.
- Complete: the separate 2,048-token diagnostic, 40 fixed attempts (39 native completed episodes and one context-exhausted failure), both audits passed. Its GPU allocation has been released. Seeds 43 and 44 remain deferred because the user requested a summary after one seed.
- TextWorld and WebShop were selected for iterative selector training; the user now prioritizes one-round SelectionARM/head comparisons, with iterative DPO deferred. Native-input SFT/DPO/deployment checks passed. Collection job 361267 was cancelled for a data-size review after 22m 04s; its GPU is released and completed data preserved. Full training and official held-out/test evaluation have not started.
- Earlier upstream inference checks and two optimizer steps per training recipe were installation checks, not full ARM training or a reproduction of all upstream results.
- The user halved the subsequent collection proposal: 5,000 training and 500 development examples per environment initially, then 2,500 new examples per environment per conditional DPO refresh. The first stage totals 11,000 examples; the full three-stage budget totals 21,000. The user subsequently authorized background execution of first-round collection; the v6b chain later stopped on an image-download HTTP 502. Recovery bootstrap 362095 passed 75 checks and audited 265 examples. Subsequent collection completed, then a native-image encoding audit stopped the chain; see the October 16 entry.

## 2026-10-16 UTC — Table 1(c) takes priority; image parity blocks training

The user requested Piotr SelectionARM and Zixian's selection head for the paper before iterative training. Overleaf authentication succeeded and the actual panel was read. Its controls are single-sample, whereas the user's primary comparison is Greedy. The [new plan](selection-arm-head-plan.md) proposes distinct learned-method rows and documents the historical Qwen3.5 Greedy comparisons, 48→70% for TextWorld and 18→28% for WebShop. No paper changes were pushed. This is in-domain selector adaptation, not a claim of zero-shot web-to-ALFWorld/WebShop transfer. Existing labels have no lookahead evidence.

GPU jobs 362097 and 362666 completed in 3h 14m 30s and 5h 56m 56s (9.19 H200-hours combined, excluding earlier attempts). CPU audit 362264 admitted 687 examples: TextWorld 308 train/52 development and WebShop 277 train/50 development. Later counters total approximately 1,930 provisional examples, but audit 363560 failed on a train/deployment prefix mismatch at 2026-10-16 22:07:04 UTC. No GPU remained allocated.

Diagnostics first found that the deployment-side audit used a default 16,777,216-pixel maximum rather than the configured 262,144 maximum. Explicit bounds fixed that case. Version v6d now checks every retained example, rather than one example per image count, and disables automatic GPU successors. CPU bootstrap 364358 passed 76 regression checks but exposed another mismatch: LLaMA-Factory's preliminary resize and native processor resizing give different visual grids for a 2,138×2,000 image (30×34 versus 30×32). Diagnostic 364369 reproduced it. CPU-only probe 364373 passed both known cases using one native resize with explicit bounds; prefixes, grids and image tensors matched. The change is currently a diagnostic override, not integrated into research training. The complete dataset, including the 687 previously accepted examples, needs the stronger parity check before training. Existing source snapshots, data and failed audit receipts are preserved; no record was dropped to make the check pass.

Zixian's current selection-head branch was separately cloned at `da20e8e45123682627c732bc5f9828f441d9ea33`; the historical inference-helper checkout remains pinned. Main-head reproduction requires the pointwise flag, native-action equivalence, exact token/feature accounting, and Qwen3.5-specific layer/cache validation. The small head is not Piotr's scalar reward model and does not require DPO. No new research training, extra seed, or official test run was submitted.

The October 14 failure timestamp below has been corrected from a mislabeled local time to UTC; this changes documentation only.

## 2026-10-15 UTC — Native image-download failure and recovery

Collection 361323 stopped at 2026-10-14 23:20:25 UTC (16:20:25 Pacific) after 44m 59s when the native Amazon image host returned HTTP 502. The adapter previously made one request. Slurm cancelled dependent audit 361324 and released the GPU. The failure was discovered at the next status check; no collection ran between the failure and recovery. Previous statements that the job was running described an earlier check, not continuous monitoring.

The same original URL now returns HTTP 200. Recovery version v6c adds at most four same-URL attempts with 2/4/8-second backoffs and explicit transport-event logging. It never substitutes or omits images. Permanent failures and exhausted retries still stop the chain. No candidate-generation, observation, action, scoring, teacher, split, or seed parameter changed. A source snapshot verified against the v6b frozen hashes preserves the implementation that failed.

CPU-only bootstrap 362095 passed 75 checks, re-audited 48 completed episodes, and admitted **171 TextWorld + 94 WebShop training examples**. These include all 80 previously admitted examples. Native image hashes and train/deployment prefixes passed; no context exclusion occurred and the largest retained example has 11,658 image-expanded tokens. The two unfinished tasks restart; their old partial/error records remain outside the retained counts. Completed-episode teacher charges including imports total approximately $10.81, excluding unfinished calls and probes. There are still zero development examples.

At that recovery checkpoint, GPU successor 362097 was submitted and awaiting startup; it subsequently completed as recorded above. The same round-zero-only stopping conditions remain in force. The 35–60 H200-hour planning range excludes the interruption, scheduler queueing, and CPU audit time; the elapsed completion date must move accordingly. A new empirical projection will follow the first audited 250 training examples per environment.

## 2026-10-14 — Background first-round collection authorized

The user requested a job that continues after disconnecting their computer. Implemented a normal Slurm dependency chain: one-H200 collection, then a CPU-only admission/encoding audit, then another collection job only while accepted counts are short. Each GPU allocation is bounded to eight hours and released before CPU auditing. The chain stops at the first-round datasets and never submits training, DPO, extra execution seeds, or official test evaluation.

Bootstrap 361314 passed 70 checks. Source review found the new action-admission parser mistakenly treated TextWorld's bracketed `go to` template generator as literal action syntax. GPU successor 361316 was cancelled during startup after 25 seconds, with no fresh collection. The corrected rule and navigation regression coverage are preserved in a new v6b study; bootstrap 361322 repeats CPU validation and imported-data auditing. The superseded startup's data and accounting remain available.

The workflow freezes source/configuration hashes, preserves native images and full raw traces, checks observation parity when resuming interrupted episodes, and reuses only audited source data. Invalid or malformed labels never count toward targets. Separate dataset exports preserve a nested 2,500-example subset. Local status files update automatically; GitHub remains a dated human-facing snapshot. These implementation changes were necessary to make the user's background-execution request concrete.

Corrected bootstrap 361322 passed 70 checks and admitted 58 TextWorld + 22 WebShop saved examples, with all images preserved, no context exclusion, and a maximum of 11,300 encoded tokens. It submitted GPU collection 361323 and dependent CPU audit 361324. The GPU job later failed and the audit was cancelled, as recorded in the recovery entry above. The audited-yield extrapolation is approximately 51 H200-hours and $538 in teacher usage for the 11,000-example target, still highly uncertain and excluding queue/CPU audit time.

## 2026-10-14 — User halves the collection budget

Apply the reduction to both environments and all planned data stages. Initial audited collection milestones are now 250, 2,500, and 5,000 training examples per environment. The development panel is fixed at 500 per environment, and each later refresh targets 2,500 fresh examples per environment; approximately 1,071 replay examples preserve the 30% replay ratio. The first-stage teacher-cost extrapolation is approximately $407, with a rough $400–600 planning range, and approximately 45 H200-hours at the sparse measured throughput. These are estimates, not caps or guarantees.

Candidate count, actor/teacher parameters, native images, per-task cap, split isolation, and training hyperparameters are unchanged. Later refreshes still require the preceding trained selector and evaluation. This turn updates the plan and documentation; it does not resume collection or modify frozen v5 artifacts. The larger proposal below is retained only as superseded history.

## 2026-10-14 — Data-size rationale corrected; collection paused

The user questioned whether 1,500 SFT examples per environment was an assistant-selected number. It was: no source establishes that this amount is sufficient. The frozen original plan totals 3,000 initial SFT examples, 3,000 fresh examples over two refreshes, and 300 development examples. Replay and multiple preference pairs must not inflate the count of new teacher-labeled examples.

The [data-size review](iterative-training-plan.md#what-the-upstream-data-actually-contains) distinguishes Piotr's 39,155 released SFT training rows / 2,992 distinct training contexts from Zixian's 5,540-state joint actor SFT/DPO corpus. Our available training task pools are 2,969 TextWorld and 8,556 WebShop instances after separation; these are task counts, not ready-made SFT rows.

Collection job 361267 was deliberately cancelled. Its completed-episode snapshots preserve 58 eligible TextWorld states from nine episodes and 33 WebShop states from seven episodes. These require formal dataset audits before training. No replacement job is queued, and no full research optimizer run occurred. Existing frozen study/configuration and cancellation logs remain intact.

The superseded larger proposal budgeted 10,000 initial SFT examples and two 5,000-example refreshes per environment, plus 1,000 development examples per environment: 42,000 retained examples overall. It proposed a nested 5,000/10,000 SFT learning curve. This was an assistant recommendation, not an upstream requirement; the user subsequently halved it as recorded above. Checkpoint organization remains open, and the reduced plan still requires a new version with a sufficient task ceiling.

## Interpretation of the completed pilot

The user requested a concrete data-collection plan, then halved its quantity. The [reduced plan](iterative-training-plan.md#concrete-collection-proposal--2026-10-14) specifies round-zero data quantity, the unchanged five-candidate sampler, native image retention, group separation, accepted-sample audits, staged collection, and measured-yield cost/runtime estimates. It recommends separate environment selectors. Round-zero background collection has since been authorized; its interruption and recovery are recorded above. DPO refresh data is deferred until a previous trained selector exists.

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
