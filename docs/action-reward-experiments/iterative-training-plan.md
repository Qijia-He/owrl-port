# Iterative Qwen selector training on TextWorld and WebShop

Status, 2026-10-14: the user selected **ALFWorld TextWorld and WebShop**, with a frozen Qwen3.5-4B actor and the original 1,024-token candidate cap. Task manifests, CPU encoding, three-step SFT/DPO updates, frozen-reference integrity, and trained-selector deployment on zero/ten images all passed. Validation job 361259 completed and released its GPU. Round-zero research collection is submitted as job 361267; full SFT has not started. Collection remains separated by environment and is useful for either checkpoint organization. Existing seed-42 inference results are unchanged.

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

| Stage | TextWorld new training states | WebShop new training states | Development states | Update |
|---|---:|---:|---:|---|
| Round 0 | 1,500 | 1,500 | 150 per environment | SFT |
| Round 1 | 750 | 750 | Reuse the fixed panel | DPO and same-data SFT control |
| Round 2 | 750 | 750 | Reuse the fixed panel | DPO and same-data SFT control |

These are planned retained counts, not already collected data. Each refresh also replays 321 earlier training states per environment, approximately 30% of its resulting training set. If selectors are separate, each initial fit uses 1,500 training states; a joint fit uses 3,000. Development labels never enter optimizer updates.

Available manifest pools contain 2,969 TextWorld training task instances and 281 development instances; WebShop has 8,556 and 895 respectively. Training/development use disjoint task/product groups, and both exclude reserved pilot groups. Evaluate checkpoints on the same 50 pilot tasks per environment and seed 42, with Greedy as the primary comparison and Sol as the teacher reference. Report success, native WebShop reward, paired uncertainty, invalid actions, truncation, steps, and cost. Repeated pilot evaluation is developmental evidence, not an untouched final test result. Official test evaluation and seeds 43/44 remain excluded.

## Validation and cost record

Job 361255 passed CPU checks and froze task manifests. Job 361256 collected native probes but failed before optimization because `git` was absent in the container; its GPU was released after 4m 36s. Revision verification now reads pinned local Git metadata. CPU job 361258 passed encoding/tokenization. Job 361259 passed 67 checks, three optimizer steps each for SFT and DPO, and trained-adapter deployment on both native input types; it completed in 5m 25s. Probe weights are never initial research-training weights. Collection job 361267 exits on completion or failure and can resume completed episodes.

The probe encoders preserve zero/ten images, reach 9,014 expanded tokens, and match deployment prompt prefixes without truncation. SFT used two probe samples; DPO used three preference pairs. Finite smoke losses establish functionality, not task improvement. No full trained selector or new benchmark score is available yet.

At the existing logged rates and an illustrative 5,000-input/500-output-token call, a label costs about $0.015. The 6,300 retained states across three rounds imply about $94.50 if every call yields one retained state. Actual collection labels full trajectories, discarding duplicate-action pools and excess states, so call counts can be substantially higher. Images, output length, and caching also change cost. Six probe calls cost approximately $0.0389; this sample is too small for a reliable full-study estimate. Actual per-environment costs and progress will update the projection. There is no hard API budget cap.

Training/serving use one allocated H200 at a time; services bind to loopback and exit with the job. Data and adapters are stored in project storage. No GPU is retained while waiting for a checkpoint-organization decision.
