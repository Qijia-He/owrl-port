# Data collection and automatic recovery

Updated 2026-10-19 UTC. This page tracks collection for the one-round **generative SelectionARM versus pointwise selection-head** comparison. It contains no trained-selector benchmark results. The [main report](README.md) explains the full experiment, and the [training-data report](training-data-readiness.md) explains the methods and validation.

## Accepted data

| Environment | Accepted training | Accepted development | Total | Target |
|---|---:|---:|---:|---:|
| ALFWorld TextWorld | 1,429 | 189 | 1,618 | 5,000 + 500 |
| WebShop | 1,152 | 142 | 1,294 | 5,000 + 500 |
| Total | 2,581 | 331 | **2,912** | **11,000** |

There are **8,088 accepted states still to collect**, or 26.5% complete. One state contains five candidate responses and one teacher selection; it is not five independent labels or a complete episode. CPU recovery audit **369538** accepted 903 additional states, preserving the original 2,009. The stopped worker's progress snapshot understated the final saved yield. New states passed full prompt and native-pixel checks; none was excluded for selector-context overflow. The largest retained selector input remains 15,721 tokens. Full actor history is stored separately and is never shortened to that selector limit.

CPU bootstrap **369540** previously passed 87 regression checks and revalidated all 2,912 states and 583 native image panels. Its GPU successor **369558 subsequently failed during startup and produced no new episodes**. Watcher **369559** ran and stopped for review because this failure was not yet recognized as retryable. The current recovery is **v6h-bounded-startup-text-web-1024-seed42**; its CPU bootstrap **369779 completed in 6m 12s**, revalidating all 2,912 states and 583 native image panels with zero selector-context exclusions. GPU continuation **369791 is now running**: it started at 21:34:49 Pacific on October 18, and model readiness took **105 seconds**. Worker logs contain **19 newly finished TextWorld tasks and 3 WebShop tasks** at the latest check. This confirms live collection has resumed; these task counts do not yet add audited states to the table above. Recent job-local GPU utilization samples were 91–100%. CPU audit successor **369796** waits for successful collection completion. Initial watcher **369781** handed monitoring to **369793**, which waits for collection to finish in any state. The [startup recovery receipt](diagnostics/v6h-startup-recovery.json) records the latest stage.

## New data awaiting audit

The latest check found **138 new provisionally eligible states** in completed task summaries from GPU job 369791, beyond the 2,912 audited states above:

| Environment | New completed tasks | New states awaiting full audit |
|---|---:|---:|
| TextWorld | 19 | 122 |
| WebShop | 3 | 16 |
| Total | **22** | **138** |

These counts come only from new summaries for this GPU job, excluding imported summaries. They do not subtract the worker's rolling counter from the previous audited total: the worker visits saved and new tasks in manifest order, so its rolling counter can temporarily omit previously accepted tasks that it has not reached yet. The previous audited states have not been lost.

Reporting should distinguish **2,912 audited + 138 newly screened, pending audit**, rather than describing the unchanged audited table as zero new output. The pending states already have candidate responses and teacher labels. The later CPU audit verifies native images and complete prompt encoding, checks context capacity, removes duplicate states, and exports the accepted cohort. Its accepted increment can be smaller than 138. This full audit runs after the current GPU batch; it does not run after every episode. Both raw traces and pending labels remain saved meanwhile.

## October 19: startup timeout and its repair

GPU job **369558** ran for 5m 14s, from October 18 at 18:39:32 to 18:44:46 Pacific. The old wrapper enforced a five-minute readiness limit. The log shows weights finishing at 18:44:29 and image-encoder profiling beginning at 18:44:30; the wrapper sent termination at 18:44:37. No fatal OOM, quota, or native-environment error preceded termination. The previously successful v6f service had initialized in about 90 seconds, but that observation did not justify treating all longer cold starts as failures. This attempt made no teacher calls and added no samples. The cancelled audit successor was **369591**.

Version v6h keeps the serving command, checkpoint, native image processing, sampling and splits unchanged. It raises the **total startup bound to 900 seconds**, writes a `server-startup.json` heartbeat with readiness status and elapsed time, and stops after **600 seconds without startup-log growth**. A process exit stops immediately. A progressing timeout receives a distinct `ServerStartupTimeout` error that the supervisor can retry within the existing two-per-phase/six-total limits. A stalled startup, wrong model IDs, OOM, storage failure, or other hard error still requires review. These bounds do not prove that any future cold start will succeed.

**27 startup/supervisor tests and all 87 existing collection checks passed** before continuation. The startup tests simulate readiness after seven minutes, a stalled service, a live progressing timeout, process exit, and adapter/model readiness. The tests themselves are not a GPU success result. Live readiness and newly finished collection tasks have now been observed separately, as recorded above. The previous v6g sources and failure logs remain preserved.

## Why the previous job stopped

GPU job **364528** ran for 3h 47m 08s and failed on October 16 at 20:35 Pacific (October 17 at 03:35 UTC). A WebShop page contained an empty image source, and the adapter attempted `requests.get('')`. The resulting `MissingSchema` exception stopped collection. Its `afterok` CPU successor **364556** was cancelled, so the earlier chain did not automatically audit or recover. The GPU was released; it was not left idle overnight.

This was an adapter error handling a real benchmark edge case. The pinned WebShop [catalog loader](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/engine/engine.py) takes `images[0]` as `MainImage`, and the [results](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/templates/results_page.html) and [item](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/templates/item_page.html) templates emit it directly. Our local native-catalog check found **717 empty main-image sources among 1,181,430 loaded products**.

Version **v6g-native-empty-image-text-web-1024-seed42** preserves the previous study and fixes this case. It first resolves any selected-option image exactly as before. If the resulting native source is empty, it keeps the product text and actions and adds an explicit unavailable-image notice. It supplies no fabricated bitmap. All nonempty image URLs are still fetched, and real download failures remain errors. The empty-source event records its page hash and image position. Native results/item template checks passed; mixed empty/valid image and option-image regression checks preserve available images.

Sampling remains frozen Qwen3.5-4B, five independently sampled candidates, temperature 0.7, top-p 0.9, 1,024 generated tokens, thinking enabled, Sol selection, and seed 42. The new behavior applies only to the previously crashing native empty-source case. Official held-out/test tasks and reserved pilot groups remain excluded.

## How unattended recovery works

Slurm owns the process; the user's computer does not need to stay connected. A short **CPU-only supervisor** is submitted with an `afterany` dependency on the current stage. A dependency waiting in the queue holds no CPU or GPU allocation. When a stage finishes, the supervisor follows its successor or handles its failure. It does not keep an idle GPU server running while waiting.

- A successful collection stage is followed by CPU auditing and then the next required collection batch.
- A recognized transient network/API failure, node failure, Slurm time limit, or progressing startup timeout can trigger automatic recovery after a five-minute scheduler delay. A failed GPU stage first goes through CPU auditing, so completed data is retained before another GPU allocation.
- Completed episodes are skipped. Normally paused episodes replay recorded native actions and verify observations/outcomes. An episode interrupted by an infrastructure error restarts from its task; its old trace remains stored, and matching teacher-cache entries can be reused. This is not a guarantee of exact mid-request continuation or zero repeated API cost.
- Recovery is capped at **two consecutive retries per phase** and **six total recoveries**. A successful CPU audit does not reset the GPU-phase failure count. Duplicate watcher/recovery submissions are prevented by a lock, handled-job records and Slurm job-name checks.
- Explicit cancellation, code/data mismatches, authentication failures, out-of-memory, storage failures and unclassified errors stop for review. Retrying unchanged code cannot fix the empty-URL bug that caused the original stop.
- The supervisor runs from a recorded source hash. Changing its code requires an explicit reviewed reinstall. A local `supervisor/STOP` file disables further recovery; cancelling active work remains a separate Slurm action.

The original supervisor passed **21 tests**; the startup-recovery revision passes **22 supervisor tests** plus five startup-wait tests, including persistence/idempotency, failure classification, CPU auditing before GPU recovery, cancellation, retry limits and a fast successor that fails before its watcher starts. These tests do not constitute an observed production failure/recovery event.

## Storage and remaining work

The project fileset was approximately **99% used**, with about 10 GiB reported free when recovery was prepared. Every future GPU submission now checks the fileset quota and stops if fewer than **4 GiB** remain. The check runs once per multi-hour batch, not in a polling loop. Quota reporting can lag and other users share the space; this guard cannot guarantee enough capacity for the entire remaining collection. Irreplaceable observations, labels and exports remain in project storage.

**Storage mitigation completed:** CPU-only job **369586** moved seven rebuildable audit-cache directories and four redundant smoke-test checkpoint directories to Scrubbed. All files passed SHA-256/size/mtime checks before the project originals were replaced with links. The final adapter weights remain on project and match those in the moved checkpoints. Original trajectories, teacher labels, native images, frozen datasets and source snapshots were not moved. Removed project file allocations total **3.51 GiB**. A subsequent quota reading at 04:28 UTC on October 19 reported **13.10 GiB free**, passing the unchanged 4 GiB guard; this is shared, time-varying headroom. The active audit-cache link also sends subsequent cache growth to Scrubbed. See the [storage receipt](diagnostics/storage-relocation-2026-10-18.json).

Using the previous 903-state increment as a rough basis, the remaining 8,088 states require about **8.37 GiB for raw collection/images/teacher cache plus 0.61 GiB for incremental exports**. Final duplicate cohort exports and training outputs are additional; other users can consume the shared headroom. Full-history head features and disposable training intermediates should use scratch, while the final dataset/model and paid raw data keep persistent copies. A further **20–30 GiB of project headroom or an approved persistent storage destination** is a planning recommendation for completing collection and training comfortably, not a measured training-space requirement. The 4 GiB guard remains.

After collection reaches 5,000 training and 500 development states per environment, both methods receive the same frozen cohort. Full feature extraction, research training, deployment validation and paired Greedy/Sol comparisons still follow. Iterative DPO, extra seeds and official test evaluation remain deferred. The supervisor does not authorize or launch those stages.

The [current startup-recovery receipt](diagnostics/v6h-startup-recovery.json) records the publication-time jobs and checks; the [earlier recovery receipt](diagnostics/v6g-collection-recovery.json) remains a historical snapshot. Cluster `STATUS.md`, `status.json`, and `supervisor/status.json` update during execution. GitHub is a reviewed snapshot, not an unattended live uploader.
