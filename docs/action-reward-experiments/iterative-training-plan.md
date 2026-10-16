# Iterative Qwen selector training on TextWorld and WebShop

Status, 2026-10-16 UTC: **iterative DPO is deferred in favor of the [Table 1(c) SelectionARM/head comparison](selection-arm-head-plan.md).** The historical collection target remains 5,000 training plus 500 development examples per environment. Recovery collection jobs 362097 and 362666 completed; CPU audit 363560 then stopped on a native-image encoding mismatch. The last completed audit contains 687 examples, with approximately 1,930 provisional examples indicated by later worker counters. A stronger every-example check in v6d found an additional image-resizing mismatch after the first processing-limit fix. The data is preserved, but is not ready for research training. Automatic GPU continuation is disabled and no GPU is running or queued. The original multi-round design below is retained for later use; it is not the immediate execution plan.

The earlier 1,500-state budget was an assistant-selected starting point, not a demonstrated sufficient size. Its collection job 361267 was stopped for review and all completed data was preserved. The later larger proposal was halved at the user's request. Future 2,500-example-per-environment refreshes remain conditional on the preceding trained selector and evaluation; they are outside the running collection chain.

## What is being trained?

A frozen Qwen actor proposes five actions for the current observation. A separate Qwen selector reads that same observation, history, and candidate set, then returns an index such as `{"selection": 3}`. We want the inexpensive selector to recover as much of Sol's closed-loop task performance as possible. Agreement with Sol is a diagnostic; task success remains the main measure.

The student is Qwen3.5-4B. All native WebShop product images remain part of both teacher and student input; text-only observations receive no artificial images. THOR is outside this training study. The frozen initial configuration mixes the two selected environments in one selector. **The revised recommendation is separate TextWorld and WebShop selectors**, so each environment's benefit can be measured independently; this is a recommendation, not a recorded user choice. Data is stored separately and remains usable for either organization.

## Iteration sequence

1. **Initial SFT:** collect states from the selected training tasks, ask Sol to select a candidate, and train the student to output that index.
2. **Refreshed DPO:** deploy the previous student on new training trajectories, ask Sol to label the same candidate pools at the visited states, and train on preferred/rejected selection indices. Prefer the student's disagreement as a negative when its action differs from Sol's selected action. Mix in earlier training examples.
3. **Further refresh:** repeat after checking the previous version. Keep an SFT control using the same refreshed data to assess whether DPO itself helps.

Actions identical to the winning action are not negative examples. Sol preferences are supervision, not proof that other actions cannot succeed. The DPO reference is the frozen initial SFT checkpoint. The actor generating the candidates stays frozen throughout. Each round's DPO and SFT control start from the same previous student and use the same states.

This is an adaptation of the selection ARM interface in [Piotr's repository](https://github.com/piotr-teterwak/action-reward-models) and the integration described in [Zixian's ARM summary](https://github.com/zixianma/OpenWebRL/blob/arm/openwebrl/docs/ARM_SUMMARY.md). Zixian's actor DPO optimizes a different model/output; index DPO here must not be described as an exact reproduction of that actor-training experiment.

## Training package and implementation status

Use the installed [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/ce9dc9e072f80fa3abe0989d4ab90da25f083438), with Transformers, PEFT LoRA, and TRL. Collection, dataset conversion, SFT, DPO, and evaluation entry points are implemented. Full native training/deployment validation must pass before research collection.

Task-group separation excludes the existing pilot and official test sets from optimizer data. Teacher prompts and all native images are preserved, with checks against silent token truncation. The candidate sampler retains the original five-candidate settings. The selector's 32,768-token image-expanded training context is separate from the actor's 1,024-token output allowance. Formal image/token audits and tokenization run in CPU-only jobs before a GPU is allocated. Checkpoints include optimizer state for resumption.

Training uses BF16 and gradient checkpointing, language-backbone LoRA (rank 32, alpha 64, dropout 0.05), and frozen vision tower/projector weights. The student returns a short index with thinking disabled; actor generation retains its original thinking setting. Initial SFT uses learning rate 1e-4 for two epochs. Refresh DPO and its SFT control use 1e-5 for one epoch; DPO uses sigmoid preference loss with beta 0.1 and the fixed initial SFT reference. Batch size is one with 16 accumulation steps.

## Data quantities and evaluation

A sample is one decision state: observation and native images, history, five candidates, and Sol's selected index. It is not a complete episode. Retain states with at least two distinct action strings and valid teacher choices, up to ten per task instance. Successful and unsuccessful trajectories can contribute labels.

### What the upstream data actually contains

- **Piotr selection SFT:** our complete audit of the [released train/validation files at revision `13905e69`](https://huggingface.co/datasets/PTeterwak/action-reward-models-data/tree/13905e69ef10169167eab388533997008694bd6d/openwebrl_actor/selection_sft) counted **39,155 training rows and 782 validation rows**. Training rows contain **2,992 distinct state contexts**. Multiple candidate draws from the same context explain why rows greatly outnumber states. The [pipeline description](https://github.com/piotr-teterwak/action-reward-models/blob/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce/README.md#stage-1--data-generation-data_generation) describes roughly 3,000 states expanded to 49,500 candidate sets; generated sets and retained SFT rows are different counts. The published internal validation split is by draw and shares state contexts with training; our task-group separation is intentionally stricter.
- **Zixian offline actor training:** the [pinned summary](https://github.com/zixianma/OpenWebRL/blob/134f064d867170eb833f6ab0f668a1b05050c031/openwebrl/docs/ARM_SUMMARY.md#2-offline-filtered-sft-and-preference-learning) records **8,394 turns** for C2/1A filtered SFT and **5,540 retained training states** for the matched joint SFT/DPO comparison (3,464 C2 + 2,076 Piotr). The [joint-data inventory](https://github.com/zixianma/OpenWebRL/blob/134f064d867170eb833f6ab0f668a1b05050c031/openwebrl/docs/ARM_JOINT_DATA.md) records **658 validation pairs**. These recipes train full actor responses, unlike our selection-index student. Their counts do not prescribe our required sample size.

Neither source establishes that 1,500 examples per environment will recover Sol's performance. Conversely, Piotr's 39,155 rows are not 39,155 independent states, so merely matching that row count would not establish comparable coverage or quality.

### Original budget, now paused

| Stage | TextWorld new training states | WebShop new training states | Development states | Update |
|---|---:|---:|---:|---|
| Round 0 | 1,500 | 1,500 | 150 per environment | SFT |
| Round 1 | 750 | 750 | Reuse the fixed panel | DPO and same-data SFT control |
| Round 2 | 750 | 750 | Reuse the fixed panel | DPO and same-data SFT control |

These were assistant-proposed retained counts, not already collected data or a user-specified total. The initial SFT target was 3,000 training samples across both environments. Including both refresh rounds, the original plan reserved **6,000 new training examples plus 300 development examples = 6,300 retained labels**. Each refresh also replays 321 earlier training states per environment, approximately 30% of its resulting training set. Replay and additional epochs are repeated exposures, not new data. DPO may derive up to two distinct rejected-action pairs per state; pairs are not additional teacher-labeled states. Development labels never enter optimizer updates.

### Reduced sizing study

Following the user's request to halve the proposal, use a **5,000-example initial SFT budget per environment**, with a nested 2,500-example fit to measure the benefit of increasing data. Cover diverse task/product groups before adding repeated draws from the same context. This is a research budget, not a claim that 5,000 is optimal or sufficient. Compare teacher agreement and closed-loop success on disjoint development data; do not choose a size using official test results.

If refresh is useful, provision **2,500 new decision examples per environment per refresh**, initially for two rounds. This corresponds to **20,000 new training examples across both environments and all three stages**, plus a fixed **500-example development panel per environment**: **21,000 retained examples in total**, before replay. Further quantity changes require a recorded revision. Strong early saturation can justify stopping below this budget; do not automatically expand beyond it. No guarantee of matching Sol follows from these counts.

The frozen v5 configuration still records the original budget and joint-checkpoint proposal. The v6b collection configuration and its v6c transport-recovery successor replace its 600-task ceiling with the available isolated task pools and records the reduced targets. Separate datasets are exported for each environment; choosing separate or joint training checkpoints does not affect this collection. No research optimizer run is part of the submitted workflow.

Available manifest pools contain 2,969 TextWorld training task instances and 281 development instances; WebShop has 8,556 and 895 respectively. Training/development use disjoint task/product groups, and both exclude reserved pilot groups. Evaluate checkpoints on the same 50 pilot tasks per environment and seed 42, with Greedy as the primary comparison and Sol as the teacher reference. Report success, native WebShop reward, paired uncertainty, invalid actions, truncation, steps, and cost. Repeated pilot evaluation is developmental evidence, not an untouched final test result. Official test evaluation and seeds 43/44 remain excluded.

Those task pools are not pre-existing SFT datasets: executing one task produces several decision states, and eligibility filtering determines the usable yield. The last completed-episode snapshots before cancellation contain **58 eligible retained states from nine TextWorld episodes and 33 from seven WebShop episodes**. These 91 states still require the formal conversion/encoding audit; interrupted episodes are not included. The completed-episode snapshots attribute approximately $3.10 in teacher usage, excluding probes and interrupted work, so this is not the total bill. The scheduler records deliberate cancellation; the runner's generic `Slurm signal 15` failure record is preserved.

The corrected v6b CPU audit now admits **58 TextWorld and 22 WebShop examples (80 total)** from those saved episodes. The lower WebShop count reflects the new selected-action syntax admission rule. There were no context exclusions among the 80 encoded examples; the largest input used 11,300 tokens, and zero-, one-, and ten-image cases passed. These accepted examples are reused toward the new targets. The earlier 91 was a provisional eligibility count, not an audited dataset size.

## Concrete collection proposal — 2026-10-14

The user requested a collection plan, halved its data budget, and then authorized background execution. The immediate deliverable is **10,000 usable SFT training examples and 1,000 development examples across two environments**. Future refresh budgets are conditional on the preceding trained selector and evaluation, not data to collect now. Sampling, native observations, split rules, and training hyperparameters are unchanged.

| Stage | TextWorld | WebShop | Behavior used to visit states |
|---|---:|---:|---|
| Round 0 training | 5,000 | 5,000 | Frozen actor proposes five; Sol selects and the environment executes that candidate |
| Fixed development panel | 500 | 500 | Same round-zero behavior on disjoint development task groups |
| Round 1, after initial SFT | 2,500 new | 2,500 new | Previous student selects; Sol labels the same candidate pool without controlling the executed choice |
| Round 2, if further refresh is justified | 2,500 new | 2,500 new | Updated student selects; Sol supplies new preference labels |

The immediate target is 11,000 retained examples. The full conditional three-stage budget is 21,000, including development. A sample contains one current observation, its native images, causal history, five original candidate responses, and one Sol index. Additional epochs, replay, candidate permutations, repeated exports, and multiple negative pairs do not increase the count of newly labeled states. Report exact state-context hashes separately from candidate-set hashes.

### Task coverage and split

Freeze task/product groups before collecting any labels. Keep the existing pilot reserve excluded, and keep all official ALFWorld held-out and WebShop test/development tasks untouched. Our development panel comes from the native training split, using the already separated 281 TextWorld / 895 WebShop development instances; optimizer data comes from the separate 2,969 / 8,556 training instances.

Visit distinct task groups before revisiting related task instances. In TextWorld, stratify the ordering by native task type while approximately preserving the available training-pool proportions. In WebShop, spread coverage across product groups. Report task type, unique instance and unique group counts rather than just the number of rows. Collect a complete episode to the unchanged native termination/step limit, and retain at most ten eligible states per task instance, sampled deterministically across its trajectory rather than taking only its first ten turns. Retain successful and unsuccessful episodes without success-only filtering.

At ten states per task, 5,000 training examples require at least 500 task instances per environment. The small existing collection yielded 58/9 = 6.4 states per TextWorld episode and 33/7 = 4.7 per WebShop episode. If those yields persisted, the training targets would require roughly 780 and 1,060 episodes respectively; post-encoding filtering may require more. These are extrapolations from only 16 episodes, not guaranteed yields. Use all available eligible task instances as the ceiling, and stop on the usable-state target; remove the obsolete 600-task ceiling in the next version. If the pool is exhausted, report the shortfall rather than duplicating records to reach the target.

### Collection parameters

| Setting | Proposed value | Reason |
|---|---|---|
| Actor | Frozen `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Same actor as the inference pilot |
| Teacher | Existing configured Sol model, `gpt-6.1-sol` | Same teacher interface as the pilot; record returned model identity |
| Candidates | 5; one candidate pool per visited state initially | Preserve the original selection problem |
| Actor sampling | Temperature 0.7, top-p 0.9, top-k disabled | Match the pinned method and completed pilot |
| Actor output | 1,024 tokens per candidate, thinking enabled | Includes reasoning and action within the same allowance; log finish reasons |
| Actor input capacity | 262,144 tokens | Preserve existing context behavior; count context exhaustion separately |
| Teacher output | `max_completion_tokens=2048`; selection index 1–5 | Preserve the current request; no extra temperature or reasoning-effort override |
| Observation | Native text; all native WebShop product images | No image removal, invented TextWorld images, hidden state, or future observations |
| Image preprocessing | Existing 65,536–262,144 pixel bounds per image | Preserve the tested multimodal pipeline |
| Image capacity | Existing maximum 10 images per observation | If a native observation exceeds capacity, fail explicitly and version a capacity change; never silently drop images |
| Episode limit | TextWorld 50 actions; WebShop 100 actions | Same wrapper limits as the completed pilot |
| Randomness | Master seed 42, with deterministic round/task/step candidate seeds | Preserve reproducibility; no execution seeds 43/44 |
| Usable selector context | 32,768 tokens after image expansion | Keep raw full inputs; audit before admission, never silently trim history or images |
| Training target | Only `{"selection": N}` | Same short-index interface at deployment; not actor-response imitation |

Sol sees only information available at the decision point. The terminal reward and future observation are saved for analysis after execution and are never included in its labeling input. The teacher is asked to select an offered candidate; it cannot supply a replacement action. Candidate duplicates and truncated raw responses stay in the offered pool exactly as generated.

### Admission and audit rules

1. Keep every raw episode, all candidate strings, execution metadata, native outcomes, teacher usage, and image hashes. A rejected training example is still available for analysis.
2. Admit a state only when the teacher returns a valid offered index without parser fallback, at least two candidate actions differ, and the selected action is nonempty and structurally parseable under the environment's action syntax. Syntax does not establish that the action is correct or succeeds; native action failures remain part of the recorded episode.
3. Exclude malformed or transport-failed teacher labels. Preserve original image bytes, image order, prompt hashes, and teacher/actor model identities. Verify teacher/training/deployment prompt parity and task-group separation.
4. Mark every actor cutoff. Do not discard a candidate merely because it reached 1,024 tokens if its action can still be parsed. Report results separately for pools with and without truncated candidates. Never repair the action text or silently increase its generation cap.
5. CPU-audit the full multimodal encoding before counting an example toward the final target. Quarantine examples that cannot fit the proposed selector context intact, record their task and length distribution, and replenish from additional eligible tasks. If this systematically removes long histories or particular task types, review a versioned larger-context recipe before continuing expansion.
6. Deduplicate exact state contexts, cap each task's contribution, and report residual repeated observations/loop states. Check all collection rounds against earlier state hashes; reused examples are replay, not fresh-state credit. Do not pad a short dataset with repeated candidate draws.

These selection and audit rules make the new benchmark collection an explicit adaptation. Piotr's published collection also uses repeated draws from saved demonstration states and a different internal split; our online trajectories, stricter task-group split, and fresh-state accounting are not an exact reproduction of that data distribution.

### Batch order and training handoff

First audit and reuse any of the 91 preserved states that satisfy the new protocol, recording the original study and source hashes. Probe samples stay excluded. Then collect in bounded batches:

1. Reach **250 accepted training examples per environment** and run a CPU encoding/coverage audit. Update per-example teacher cost, calls per retained example, task yield, context exclusions, and throughput. These are the first examples of the research dataset and count toward 5,000; this is not another disposable functional smoke run.
2. Expand to **2,500 training examples per environment** and build the fixed development panel. Freeze this dataset version for the smaller SFT fit.
3. Expand to **5,000 training examples per environment**, retaining the 2,500-example subset. Fit both sizes from the same base checkpoint with the same training recipe; report the extra update/token exposure as well as dataset size. Use development evidence to judge whether the additional data helps. Teacher agreement and closed-loop task success are separate measures; weak evidence does not justify claiming saturation.
4. Release collection GPUs and hand off audited manifests, image hashes, inclusion/exclusion counts, token-length statistics, costs, and task-group coverage to training. The original 50-task pilot remains a development comparison; official held-out evaluation is deferred until the protocol is fixed.
5. Only after initial SFT, collect round-one states under that trained selector. For DPO, use Sol's index as chosen, prefer the student's disagreeing action as a negative, and retain at most two distinct rejected actions per state. Do not label another index with the same winning action as a negative. With 2,500 fresh states, replay approximately 1,071 earlier states per environment for a 30% replay mixture. The initial SFT model remains the fixed DPO reference; keep a same-state SFT control. The second refresh depends on the first refresh's results.

For the proposed separate selectors, the training handoff uses one environment's data per model. The existing LLaMA-Factory recipe remains: BF16, language LoRA rank 32 / alpha 64 / dropout 0.05, frozen vision tower/projector, effective batch 16; initial SFT learning rate 1e-4 for two epochs, then DPO learning rate 1e-5 for one epoch with beta 0.1. These are starting settings, not tuned results, and research training starts from the original base rather than smoke-test weights.

### Cost, runtime, and implementation boundary

From completed-episode snapshots, teacher usage per retained state was approximately **$0.0265 for TextWorld** and **$0.0476 for WebShop**, using the existing logged rates. Labeling every executed step while retaining at most ten states per task explains why the cost is higher than one isolated short label call. Extending those rates to 5,500 retained examples per environment gives **about $407 for round-zero teacher calls**. A working planning range is **$400–600**, with possible upward revision after post-encoding filtering and the 250-example audit. This is a historical-rate extrapolation, not a current price quote or spending cap; it excludes GPU time, later rounds, training, and evaluation.

The cancelled one-H200 allocation used 22m 04s while the completed-episode snapshots retained 91 states. A naive aggregate projection for 11,000 states is **about 45 H200-hours**, roughly two compute-days on one GPU, excluding queue time and later training/evaluation. Different per-environment yields, startup overhead, API latency, context filtering, and utilization make this estimate highly uncertain. Treat 35–60 GPU-hours as a rough scheduling range, not a confidence interval or completion promise. Re-estimate from the next completed batch before committing the full schedule; do not assume linear multi-GPU speedup.

After the corrected imported-data audit, only 80 examples count toward the new protocol. The same sparse extrapolation becomes approximately **51 H200-hours** and **$538 in round-zero teacher usage**. These remain within the initial rough planning ranges; they are not full-run measurements. Queue time and CPU audit time are additional, and the first 250 accepted examples per environment will provide a better estimate.

Use one H200 for the existing serving recipe, with two environment workers and bounded API concurrency. Its measured reservation was about 107,581 MiB, so that recipe does not fit an 18 GiB MIG instance unchanged. The live scheduler permits account `zixianma` with normal QoS (24-hour limit), four default CPUs per H200, and 30 GiB per CPU. Each GPU job requests eight hours, four CPUs and 120 GiB; collection uses a 6.5-hour soft deadline and a 40-minute margin for starting new episodes. CPU audit successors request no GPU, two CPUs and 60 GiB. The scheduler settings are checked again before each automatic submission. Data remains in project storage; the launch quota check showed 964 GiB used in the group's 1 TiB fileset. A 20 GiB per-study artifact guard stops expansion for review if necessary.

The collection implementation now freezes configuration and source hashes, reuses unchanged native model/preprocessing validation with explicit provenance, imports completed source episodes, orders tasks by group/type, audits native action syntax and image-expanded token lengths, and preserves nested 2,500/5,000-example exports. Interrupted episodes resume by replaying recorded native actions and checking observations before generating any new candidates. Existing replayed steps are not relabeled. User cancellation or infrastructure failure breaks the chain; there is no repeated automatic failure retry. Three audits without progress, a material context-exclusion rate, or the artifact storage guard also stop expansion. Separate model-training checkpoint orchestration is future work; only separate datasets are produced now.

## Background execution record

- CPU bootstrap 361314 passed 70 checks. Its first audit exposed a new admission bug during source review: TextWorld's `go to` template uses bracketed generator notation that must not be interpreted as literal action brackets. Those audit counts are superseded.
- GPU successor 361316 was cancelled during server startup after 25 seconds, before fresh collection. Its dependent audit was cancelled by Slurm; artifacts and accounting remain preserved.
- The corrected syntax rule has a regression assertion for `go to cabinet 1`. A new configuration/study, v6b, preserves the original v6 startup record. CPU bootstrap 361322 passed 70 checks and re-audited the saved episodes in about 35 seconds, then submitted GPU collection 361323. That job subsequently failed on a native image-download HTTP 502; CPU audit successor 361324 was cancelled.
- On 2026-10-15 UTC, a status check found that job 361323 had failed at 2026-10-14 23:20:25 UTC (16:20:25 Pacific), after 44m 59s. The failure was a native WebShop image URL returning HTTP 502. The previously checked job had been healthy, but collection did not continue after this error. The allocation was released, and dependent audit 361324 never started. All completed summaries and partial traces were retained.
- The exact failing URL returned HTTP 200 during recovery. The v6c adapter now makes at most four attempts to fetch that same URL, with 2/4/8-second backoffs for transient network/HTTP/decode failures. It records retry events, never replaces or silently drops images, and still stops on permanent errors or exhausted retries. No automatic whole-job retry was added. A verified snapshot of the old source and its frozen configuration remains in the v6b artifacts.
- CPU bootstrap 362095 passed 75 regression checks, including transient HTTP recovery, native pixel preservation, cache reuse, permanent failures, and bounded retry exhaustion. Its audit retained **171 TextWorld examples from 25 completed episodes and 94 WebShop examples from 23 completed episodes**. The 265 examples include the 80 previously audited examples, not an additional 265. The maximum retained image-expanded length is 11,658 tokens; none exceeded the 32,768-token cutoff. Of the retained examples, 209 have no image, 41 have ten images, and 15 have one image; all image content hashes and deployment-prefix checks passed. Development collection has not started.
- The new study imports and re-audits completed episodes. The two interrupted tasks restart from their native initial states; their prior partial/error records remain preserved and are excluded from retained-data counts. Normal soft-deadline pauses within a study continue to use checked native replay. Completed-episode teacher charges, including imported episodes, total approximately **$10.81**; this excludes unfinished attempts/probes and is not a complete account invoice. Recovery GPU job 362097 is submitted and awaiting startup.
- October 16 update: jobs 362097 and 362666 completed, but the next CPU audit failed. Stronger validation in v6d is CPU-only and explicitly disables automatic collection successors while the Table 1(c) plan is prepared. No full research training or DPO was launched.
- Each successful GPU job has an `afterok` CPU audit successor. Each successful audit submits another GPU job only if the accepted counts are short. No login-node daemon, open terminal, or connected client is required. Data preparation never holds a GPU.
- The local study's `status.json` and `STATUS.md` update during execution and identify current/successor jobs, audited counts, and errors. This GitHub report is a dated snapshot, not an unattended uploader. Raw examples, images, credentials, and request payloads remain outside the public documentation.

## Validation and cost record

Job 361255 passed CPU checks and froze task manifests. Job 361256 collected native probes but failed before optimization because `git` was absent in the container; its GPU was released after 4m 36s. Revision verification now reads pinned local Git metadata. CPU job 361258 passed encoding/tokenization. Job 361259 passed 67 checks, three optimizer steps each for SFT and DPO, and trained-adapter deployment on both native input types; it completed in 5m 25s. Probe weights are never initial research-training weights. Collection job 361267 exits on completion or failure and can resume completed episodes.

The probe encoders preserve zero/ten images, reach 9,014 expanded tokens, and match deployment prompt prefixes without truncation. SFT used two probe samples; DPO used three preference pairs. Finite smoke losses establish functionality, not task improvement. No full trained selector or new benchmark score is available yet.

At the existing logged rates and an illustrative 5,000-input/500-output-token call, a label costs about $0.015. The 6,300 retained states across three rounds imply about $94.50 if every call yields one retained state. Actual collection labels full trajectories, discarding duplicate-action pools and excess states, so call counts can be substantially higher. Images, output length, and caching also change cost. Six probe calls cost approximately $0.0389; this sample is too small for a reliable full-study estimate. Actual per-environment costs and progress will update the projection. There is no hard API budget cap.

Training/serving use one allocated H200 at a time; services bind to loopback and exit with the job. Data and adapters are stored in project storage. No GPU is retained while waiting for a checkpoint-organization decision.
