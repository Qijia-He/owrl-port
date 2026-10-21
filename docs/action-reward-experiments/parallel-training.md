# Four independent selector training jobs

Updated October 20, 2026. This is a scheduling change requested after the final cohort was collected. It does not change the research protocol, data, training algorithms or random seeds.

**Completion update, October 21:** the shared feature extraction and all four fits completed. Both SFT branches resumed successfully once; the final worker ended at 06:37:45 PDT. No GPU allocations remain. See [verified training results](selector-training-results.md). The queue snapshot at the bottom of this page is historical.

## What is trained

The frozen Qwen3.5-4B actor generated five candidates at each recorded decision state. Sol selected a candidate before execution. Each environment has 5,000 training and 500 development states. We train two alternatives to Sol, separately for TextWorld and WebShop:

- **Generative SelectionARM:** a separate Qwen3.5-4B LoRA model learns to emit the selected candidate index. This is the Piotr-style generative selector adaptation.
- **Listwise selection head:** a 22,562,818-parameter network learns to compare candidate features extracted from the frozen actor. This uses Zixian's pinned listwise implementation.

The [method record](qijia-training-workstream.md) provides the upstream references, exact setup, declared adaptations and validation. Current labels are pre-execution choices, not K=1 lookahead labels. These are selector fits; the actor is not being fine-tuned.

## Actual dependencies

| Training job | Required inputs | Dependencies on the other three fits |
|---|---|---|
| TextWorld SelectionARM | Prepared TextWorld training/development data | None |
| WebShop SelectionARM | Prepared WebShop data, including native images | None |
| TextWorld listwise head | Completed shared full-history feature extraction | None |
| WebShop listwise head | Completed shared full-history feature extraction | None |

The previous controller created separate Slurm jobs but submitted them serially: features, both heads, then both generative selectors. The replacement submits both generative fits immediately and submits the two heads independently after the shared feature stage succeeds. Heads are not allocated GPUs while waiting for features. Four independent fits do not guarantee four simultaneous running allocations: actual starts depend on scheduler availability.

CPU preparation job **374309** completed successfully in **19m 54s**. All 11,000 examples passed the selector encoding audit. Maximum selector input lengths were **10,673 tokens for TextWorld** and **15,721 for WebShop**, below the unchanged 32,768 selector cutoff. Full actor histories used for head features are separate inputs and retain the much longer recorded lengths. Feature job **374367** keeps its original queue position during migration.

## Resources and recovery

Feature extraction and each generative fit request one full H200. Each head requests one MIG instance when the MIG node is healthy. At the live configuration check, the sole MIG node was draining after a GPU health-check timeout. If it remains unavailable when heads become ready, each head uses a short one-H200 job and records the reason. Research jobs use normal QoS; they are not restricted by debug QoS's one-running-job limit.

Each branch has its own checkpoints, output directories and CPU callback. A recorded checkpoint exit 75 resumes only that branch, with the original limit of 12 continuations. Unexpected failures stop the affected branch for review; unrelated fits can continue. The shared feature stage must succeed before either head starts. Source/data integrity and the 4 GiB project-space guard remain active. Intermediate checkpoints and features use the existing scratch locations; final weights stay on project storage.

The new control code lives in `experiments/selector-training/orchestration/`; frozen model-training code remains unchanged. Live state is `experiments/selector-training/models/parallel/workflow.json`, and per-branch logs and scheduler requests are in `models/parallel/jobs/`. The old sequential dispatcher has a STOP marker and its obsolete CPU callback is cancelled during migration. Never restart it alongside the new controller.

Finishing these jobs establishes trained selector artifacts and development metrics. It does not establish improved task success. Closed-loop comparisons against matched Greedy and Sol controls follow separately; the controller does not start actor SFT, DPO, extra execution seeds or official test evaluation.


## Installed workflow snapshot

CPU installation job **374537** adopted feature job **374367** and cancelled only the obsolete sequential callback **374368**. TextWorld SelectionARM is job **374539**. A failed Slurm configuration query prevented the first WebShop submission before any GPU was allocated to it. Reviewed controller revision 2 reuses one fresh scheduler-configuration snapshot per callback, avoiding repeated global queries. Recovery job **374541** passed **10 CPU scheduler-contract tests** and submitted WebShop SelectionARM as job **374543**. The original controller revision and failure receipt remain archived.

At this snapshot, all three GPU jobs are pending scheduler priority. Both head branches are waiting for completed features and have not yet requested GPU allocations. Their independent submissions are automatic. The [portable workflow receipt](diagnostics/parallel-training-2026-10-20.json) records this snapshot; it is not a live queue or a model-accuracy result.
