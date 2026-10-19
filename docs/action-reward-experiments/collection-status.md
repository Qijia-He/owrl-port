# Data collection and automatic recovery

Updated 2026-10-18 UTC. This page tracks collection for the one-round **generative SelectionARM versus pointwise selection-head** comparison. It contains no trained-selector benchmark results. The [main report](README.md) explains the full experiment, and the [training-data report](training-data-readiness.md) explains the methods and validation.

## Accepted data

| Environment | Accepted training | Accepted development | Total | Target |
|---|---:|---:|---:|---:|
| ALFWorld TextWorld | 1,429 | 189 | 1,618 | 5,000 + 500 |
| WebShop | 1,152 | 142 | 1,294 | 5,000 + 500 |
| Total | 2,581 | 331 | **2,912** | **11,000** |

There are **8,088 accepted states still to collect**, or 26.5% complete. One state contains five candidate responses and one teacher selection; it is not five independent labels or a complete episode. CPU recovery audit **369538** accepted 903 additional states, preserving the original 2,009. The stopped worker's progress snapshot understated the final saved yield. New states passed full prompt and native-pixel checks; none was excluded for selector-context overflow. The largest retained selector input remains 15,721 tokens. Full actor history is stored separately and is never shortened to that selector limit.

CPU bootstrap **369540** passed 87 regression checks and revalidated all 2,912 states and 583 native image panels. GPU collection **369558** is queued. CPU watcher **369551** completed its first real handoff after bootstrap; watcher **369559** now depends on collection job 369558 and will check its result. These are scheduler snapshots; pending means the GPU is not yet running.

## Why the previous job stopped

GPU job **364528** ran for 3h 47m 08s and failed on October 16 at 20:35 Pacific (October 17 at 03:35 UTC). A WebShop page contained an empty image source, and the adapter attempted `requests.get('')`. The resulting `MissingSchema` exception stopped collection. Its `afterok` CPU successor **364556** was cancelled, so the earlier chain did not automatically audit or recover. The GPU was released; it was not left idle overnight.

This was an adapter error handling a real benchmark edge case. The pinned WebShop [catalog loader](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/engine/engine.py) takes `images[0]` as `MainImage`, and the [results](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/templates/results_page.html) and [item](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/templates/item_page.html) templates emit it directly. Our local native-catalog check found **717 empty main-image sources among 1,181,430 loaded products**.

Version **v6g-native-empty-image-text-web-1024-seed42** preserves the previous study and fixes this case. It first resolves any selected-option image exactly as before. If the resulting native source is empty, it keeps the product text and actions and adds an explicit unavailable-image notice. It supplies no fabricated bitmap. All nonempty image URLs are still fetched, and real download failures remain errors. The empty-source event records its page hash and image position. Native results/item template checks passed; mixed empty/valid image and option-image regression checks preserve available images.

Sampling remains frozen Qwen3.5-4B, five independently sampled candidates, temperature 0.7, top-p 0.9, 1,024 generated tokens, thinking enabled, Sol selection, and seed 42. The new behavior applies only to the previously crashing native empty-source case. Official held-out/test tasks and reserved pilot groups remain excluded.

## How unattended recovery works

Slurm owns the process; the user's computer does not need to stay connected. A short **CPU-only supervisor** is submitted with an `afterany` dependency on the current stage. A dependency waiting in the queue holds no CPU or GPU allocation. When a stage finishes, the supervisor follows its successor or handles its failure. It does not keep an idle GPU server running while waiting.

- A successful collection stage is followed by CPU auditing and then the next required collection batch.
- A recognized transient network/API failure, node failure, or time limit can trigger automatic recovery after a five-minute scheduler delay. A failed GPU stage first goes through CPU auditing, so completed data is retained before another GPU allocation.
- Completed episodes are skipped. Normally paused episodes replay recorded native actions and verify observations/outcomes. An episode interrupted by an infrastructure error restarts from its task; its old trace remains stored, and matching teacher-cache entries can be reused. This is not a guarantee of exact mid-request continuation or zero repeated API cost.
- Recovery is capped at **two consecutive retries per phase** and **six total recoveries**. A successful CPU audit does not reset the GPU-phase failure count. Duplicate watcher/recovery submissions are prevented by a lock, handled-job records and Slurm job-name checks.
- Explicit cancellation, code/data mismatches, authentication failures, out-of-memory, storage failures and unclassified errors stop for review. Retrying unchanged code cannot fix the empty-URL bug that caused the original stop.
- The supervisor runs from a recorded source hash. Changing its code requires an explicit reviewed reinstall. A local `supervisor/STOP` file disables further recovery; cancelling active work remains a separate Slurm action.

The supervisor passed **21 tests**, including persistence/idempotency, failure classification, CPU auditing before GPU recovery, cancellation, retry limits and a fast successor that fails before its watcher starts. These tests do not constitute an observed production failure/recovery event.

## Storage and remaining work

The project fileset was approximately **99% used**, with about 10 GiB reported free when recovery was prepared. Every future GPU submission now checks the fileset quota and stops if fewer than **4 GiB** remain. The check runs once per multi-hour batch, not in a polling loop. Quota reporting can lag and other users share the space; this guard cannot guarantee enough capacity for the entire remaining collection. Irreplaceable observations, labels and exports remain in project storage.

After collection reaches 5,000 training and 500 development states per environment, both methods receive the same frozen cohort. Full feature extraction, research training, deployment validation and paired Greedy/Sol comparisons still follow. Iterative DPO, extra seeds and official test evaluation remain deferred. The supervisor does not authorize or launch those stages.

The [portable status receipt](diagnostics/v6g-collection-recovery.json) records the publication-time jobs and checks. Cluster `STATUS.md`, `status.json`, and `supervisor/status.json` update during execution. GitHub is a reviewed snapshot, not an unattended live uploader.
