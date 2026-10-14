# Iterative Qwen selector training on TextWorld and WebShop

Status, 2026-10-14: the user selected **ALFWorld TextWorld and WebShop**, with a frozen Qwen3.5-4B actor and the original 1,024-token candidate cap. Task manifests, CPU encoding, three-step SFT/DPO updates, frozen-reference integrity, and trained-selector deployment on zero/ten images all passed. After the user questioned the data-size rationale, collection job 361267 was cancelled after 22m 04s and its GPU was released. Its completed episodes and labels are preserved. **The 1,500-state target was an assistant-selected starting budget, not an upstream requirement or a demonstrated sufficient size.** Full SFT has not started. Data sizing is under review; the larger budget below is a proposal, not an active collection target. Existing seed-42 inference results are unchanged.

## What is being trained?

A frozen Qwen actor proposes five actions for the current observation. A separate Qwen selector reads that same observation, history, and candidate set, then returns an index such as `{"selection": 3}`. We want the inexpensive selector to recover as much of Sol's closed-loop task performance as possible. Agreement with Sol is a diagnostic; task success remains the main measure.

The student is Qwen3.5-4B. All native WebShop product images remain part of both teacher and student input; text-only observations receive no artificial images. THOR is outside this training study. The initial configuration mixes the two selected environments in one selector; separate selectors would each use their own environment's data. Checkpoint organization is being clarified, while data remains stored separately by environment and supports either choice.

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

### Proposed sizing study, not yet launched

Use a **10,000-example initial SFT budget per environment**, with a nested 5,000-example fit to measure the benefit of increasing data. Cover diverse task/product groups before adding repeated draws from the same context. This is a research budget informed by the upstream scale, not a claim that 10,000 is optimal or sufficient. Compare teacher agreement and closed-loop success on disjoint development data; do not choose a size using official test results.

If refresh is useful, provision **5,000 new decision examples per environment per refresh**, initially for two rounds. This corresponds to **40,000 new training examples across both environments and all three stages**, plus a proposed fixed **1,000-example development panel per environment**: **42,000 retained examples in total**, before replay. The actual amount should change only in a recorded version after examining the learning curve, task coverage, label yield, and student-versus-teacher failure cases. Strong early saturation can justify stopping below this budget; continuing improvement can justify expansion. No guarantee of matching Sol follows from these counts.

The frozen v5 configuration still records the original budget and joint-checkpoint proposal. Its 600-task collection ceiling would permit at most 6,000 retained states per environment under the ten-state cap, so the larger proposal requires a new version with a sufficient task ceiling, not just editing the sample target. No revised collection or fit is queued. Separate-versus-joint checkpoints remain unresolved; storage is already separated by environment.

Available manifest pools contain 2,969 TextWorld training task instances and 281 development instances; WebShop has 8,556 and 895 respectively. Training/development use disjoint task/product groups, and both exclude reserved pilot groups. Evaluate checkpoints on the same 50 pilot tasks per environment and seed 42, with Greedy as the primary comparison and Sol as the teacher reference. Report success, native WebShop reward, paired uncertainty, invalid actions, truncation, steps, and cost. Repeated pilot evaluation is developmental evidence, not an untouched final test result. Official test evaluation and seeds 43/44 remain excluded.

Those task pools are not pre-existing SFT datasets: executing one task produces several decision states, and eligibility filtering determines the usable yield. The last completed-episode snapshots before cancellation contain **58 eligible retained states from nine TextWorld episodes and 33 from seven WebShop episodes**. These 91 states still require the formal conversion/encoding audit; interrupted episodes are not included. The completed-episode snapshots attribute approximately $3.10 in teacher usage, excluding probes and interrupted work, so this is not the total bill. The scheduler records deliberate cancellation; the runner's generic `Slurm signal 15` failure record is preserved.

## Validation and cost record

Job 361255 passed CPU checks and froze task manifests. Job 361256 collected native probes but failed before optimization because `git` was absent in the container; its GPU was released after 4m 36s. Revision verification now reads pinned local Git metadata. CPU job 361258 passed encoding/tokenization. Job 361259 passed 67 checks, three optimizer steps each for SFT and DPO, and trained-adapter deployment on both native input types; it completed in 5m 25s. Probe weights are never initial research-training weights. Collection job 361267 exits on completion or failure and can resume completed episodes.

The probe encoders preserve zero/ten images, reach 9,014 expanded tokens, and match deployment prompt prefixes without truncation. SFT used two probe samples; DPO used three preference pairs. Finite smoke losses establish functionality, not task improvement. No full trained selector or new benchmark score is available yet.

At the existing logged rates and an illustrative 5,000-input/500-output-token call, a label costs about $0.015. The 6,300 retained states across three rounds imply about $94.50 if every call yields one retained state. Actual collection labels full trajectories, discarding duplicate-action pools and excess states, so call counts can be substantially higher. Images, output length, and caching also change cost. Six probe calls cost approximately $0.0389; this sample is too small for a reliable full-study estimate. Actual per-environment costs and progress will update the projection. There is no hard API budget cap.

Training/serving use one allocated H200 at a time; services bind to loopback and exit with the job. Data and adapters are stored in project storage. No GPU is retained while waiting for a checkpoint-organization decision.
