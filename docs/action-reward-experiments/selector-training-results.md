# First selector training round: four models completed

Reviewed October 21, 2026, Pacific time. **Both SelectionARM adapters and both listwise selection heads completed training.** The last GPU job ended at 06:37:45. The parallel workflow is complete and holds no GPU allocations. These are training/development results; closed-loop task success has not yet been measured for these checkpoints.

## What was trained

The actor remains frozen Qwen3.5-4B. Each decision state contains five actor-generated candidates and Sol's pre-execution choice. TextWorld and WebShop each contribute **5,000 training and 500 development states**, with the same frozen panels and group split for the two selector methods. WebShop retains native images. Labels do not use K=1 lookahead, and this round does not train the actor.

The generative SelectionARM adaptation trains a separate Qwen3.5-4B LoRA selector to emit a candidate index, using LLaMA-Factory for two epochs. The Zixian-style listwise selection head trains 22,562,818 parameters for 20 epochs, using frozen-actor features. The [method and adaptation record](qijia-training-workstream.md) gives the upstream citations, losses, hyperparameters and validation checks. The [independent-job record](parallel-training.md) explains scheduling and recovery.

## Execution and automatic recovery

| Stage | GPU jobs | Allocated elapsed time | Finished, Oct 21 PDT |
|---|---|---:|---|
| Shared full-history feature extraction | 374367 | 4h 25m 32s | 02:21:02 |
| TextWorld listwise head | 376496 | 1m 37s | 03:41:08 |
| WebShop listwise head | 376498 | 1m 37s | 03:41:08 |
| TextWorld generative SelectionARM | 374539 → 376607 | 8h 24m 24s total | 06:37:45 |
| WebShop generative SelectionARM | 374543 → 376613 | 8h 04m 04s total | 06:17:45 |

Both head fits actually ran on MIG instances; the node had become schedulable by submission time. Feature extraction and the SFT attempts ran on full H200s. Allocated elapsed time includes loading, validation, saving and reload checks. The jobs overlapped, so these durations should not be added to estimate elapsed calendar time.

Both first SFT allocations exited with code **75**, the configured checkpoint-and-continue signal. Slurm labels these attempts FAILED, but the controller identified the expected exit, submitted a continuation and preserved optimizer/scheduler state. TextWorld resumed at step **461**, WebShop at step **283**. Both finished step **626/626**, epoch **2.0**, and produced persistent final adapters. No unexpected training failure remains.

All **11,000** full-history feature panels completed, in 688 chunks. Actor histories remained intact, including the longest recorded WebShop prompt of 161,591 tokens. This reference extraction repeats full actor forwards; it does not establish low-overhead feature capture during online generation.

## Head development results: 500 states per environment

| Environment | Selected epoch, of 20 | Exact teacher-index agreement | Teacher-action agreement | Uniform-of-five expected teacher-action agreement |
|---|---:|---:|---:|---:|
| TextWorld | 14 | 43.4% (217/500) | **90.2% (451/500)** | 52.6% |
| WebShop | 8 | 54.6% (273/500) | **87.2% (436/500)** | 42.56% |

Exact-index agreement requires selecting the same numbered candidate as Sol. Teacher-action agreement accepts a different candidate when its parsed native command is identical to Sol's command. Several candidates can therefore count as equivalent. The random column averages the number of equivalent candidates divided by five at each state; it is an offline reference, not a closed-loop random-policy score. Equivalent commands need not have identical reasoning text or retained context.

These are **development metrics used to select the checkpoint**, not untouched-test estimates. A 90.2% teacher-action agreement is not 90.2% task success, nor proof that the head preserves 90.2% of Sol's closed-loop benefit. Both heads passed exact reload, candidate-permutation and train-only-normalizer checks.

## Generative SelectionARM validation

Both models trained for two epochs and completed loss evaluation on their 500-state development sets. The lowest development loss selected TextWorld checkpoint **626** (loss **0.069523**) and WebShop checkpoint **600** (loss **0.070111**); WebShop training itself continued to step 626. The final adapters contain the selected weights.

Each final adapter was reloaded and exercised on **eight development prompts**, including the recorded native image processing. Both returned valid candidate indices on all eight prompts. TextWorld matched Sol's exact index and native action on 8/8; WebShop matched the exact index on 5/8 and the native action on 7/8.

**These eight prompts are a reload/functionality probe, not a full 500-state generation evaluation.** Their agreement numbers cannot be compared directly with the head's 500-state metrics or used to rank the two model families. Token-level development loss, which includes the selection response format, is not selection accuracy or benchmark success.

## Artifacts and next evaluation

All four final weights are persistent on project storage beneath:

```text
/gpfs/projects/krishna/heqj3/arm-generalization/
  experiments/selector-training/models/research/
    alfworld_text/head/head.pt
    alfworld_text/sft-adapter/adapter_model.safetensors
    webshop/head/head.pt
    webshop/sft-adapter/adapter_model.safetensors
```

The workspace path is a navigation symlink to the original versioned artifact directory. Each head is 90,512,546 bytes; each adapter is 259,794,448 bytes. Final weights remain on project storage; intermediate features and optimizer checkpoints use temporary scratch. The [completion receipt](diagnostics/selector-training-complete-2026-10-21.json) records hashes, selected checkpoints, development metrics, scheduler accounting and the final integrity audit.

The separately authorized next stage is running: matched closed-loop evaluation of Greedy, Sol-of-five, generative SelectionARM and listwise head on **134 TextWorld unseen tasks and 500 WebShop test tasks**, seed 42. CPU preparation 377085 and deployment check 377104 passed. At 14:30 Pacific, 316 episodes were complete; three TextWorld lanes continued, while a native image 404 stopped the three WebShop lanes and storage blocked both heads. The user-approved missing-image amendment passed validation; recovery controller 377373 is queued after verified lossless storage archival. The primary comparison is against Greedy. WebShop uses matched 30-step controls; historical 100-step pilot scores cannot be substituted. Reserved-training deployment probes are excluded from official success rates. See the [dated partial results, protocol and recovery status](official-selector-evaluation.md). No actor SFT, DPO or extra execution seeds are included.
