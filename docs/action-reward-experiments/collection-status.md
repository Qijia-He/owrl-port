# Data collection and automatic recovery

Updated 2026-10-19 at 21:15 UTC (14:15 Pacific). This page tracks collection for the one-round **generative SelectionARM versus pointwise selection-head** comparison. There are no trained-selector benchmark results. The [main report](README.md) explains the experiment.

## Current protocol and execution

The user requested **a maximum of 30 WebShop environment actions per episode**. Active study `v6i-webshop30-text50-1024-seed42` uses `configs/experiment-webshop30.yaml`; TextWorld stays at 50 actions. Five independently sampled candidates, Sol selection, temperature 0.7, top-p 0.9, 1,024 output tokens, thinking mode, seed 42, native images and full actor history are unchanged. A 30-step cap ends the rollout; it does not remove earlier history from the prompt.

Old GPU job **369791 was intentionally stopped** after 1h 13m 44s. Its original execution receipt records SIGTERM/exit 1; the protocol-transition receipt records the user-directed reason. Old audit 369796 and watcher 369793 were cancelled, and the old supervisor's STOP marker prevents an unwanted 100-step restart. Completed old tasks and raw partial traces remain saved; incomplete tasks restart under the new cap.

CPU bootstrap **370003** completed the fresh import/audit. Since then, **four GPU collection stages and their CPU audits have completed successfully**, with **zero automatic recoveries**. The fifth GPU stage, **371521**, is running; CPU audit **371558** and supervisor **371530** are queued on dependencies. Its model service became ready in approximately 105 seconds. See the [current progress receipt](diagnostics/v6i-collection-progress-2026-10-19.json); the [transition receipt](diagnostics/v6i-webshop30-transition.json) preserves the earlier launch snapshot.

| GPU collection job | GPU elapsed | CPU audit job | Accepted states after audit |
|---|---:|---|---:|
| 370030 | 4h 12m 11s | 370034 | 4,774 |
| 370622 | 2h 49m 49s | 370624 | 5,806 |
| 370757 | 3h 30m 18s | 370759 | 6,635 |
| 370991 | 3h 05m 55s | 370993 | **7,714** |

Completed summaries from the new version, excluding historical imports, cover **603 WebShop tasks with a maximum of 30 actions** and **371 TextWorld tasks with a maximum of 50**. None exceeds its configured cap. These are live collection checks, not learned-selector evaluation scores.

## Accepted data

| Environment | Accepted training | Accepted development | Total | Target |
|---|---:|---:|---:|---:|
| ALFWorld TextWorld | 3,675 | 500 | 4,175 | 5,000 + 500 |
| WebShop | 3,039 | 500 | 3,539 | 5,000 + 500 |
| Total | 6,714 | 1,000 | **7,714 (70.1%)** | **11,000** |

There are **3,286 accepted states still to collect**: 1,325 TextWorld training states and 1,961 WebShop training states. Both 500-state development cohorts are complete. The horizon-transition bootstrap previously increased the total from 2,912 to 3,183 by accepting 271 states saved before the intentional stop; those remain historical data. The four subsequent successful batches added another **4,531 audited states**.

<a id="new-data-awaiting-audit"></a>

At this snapshot, the current GPU job has saved **315 additional provisionally eligible states awaiting CPU audit**: 124 from 16 completed TextWorld tasks and 191 from 52 completed WebShop tasks. These are not included in 7,714 and may change after encoding, deduplication and task-cap checks. Unfinished tasks are not counted here.

One state contains five candidate responses and one teacher selection. Reporting must separate audited states from newly screened states awaiting audit. Worker rolling totals are not safe increments over the last audit because imported and new tasks are visited in manifest order. The full CPU audit checks native image processing, prompt encoding, selector-context capacity and duplicates before updating the accepted dataset.

The latest audit reports zero selector-context exclusions and a maximum retained selector input of 15,721 tokens. It performed 1,079 fresh prefix checks and 229 fresh native-image-panel checks; earlier checks are reused only through validated cache fingerprints. These figures do not mean full-history actor prompts are shortened to selector-prompt length.

## Historical data, validation and evaluation

Existing accepted data is preserved in order. Imported completed summaries record `collection_max_steps` and their source study; original traces are unchanged. Thus the eventual training cohort includes historical 100-step WebShop trajectories plus new 30-step trajectories. **Both SelectionARM and the pointwise head use exactly the same frozen cohort.** It must not be described as an entirely 30-step collection.

Six checks cover the explicit horizon migration, rejection of undeclared changes, rejection of hidden sampling/other-environment changes, and a collector that stops at exactly 30 actions when the environment never terminates. That rollout check confirms history lengths 0 through 29, with no history shortening. The 27 startup/supervisor checks and 87 collection regressions also passed. These CPU checks do not constitute a new live 30-step benchmark result.

The original WebShop RL [baseline default](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/baseline_models/train_rl.py) was 100 steps; examples of shorter recent training protocols include [GiGPO's 15 steps](https://arxiv.org/html/2505.10978#A5.SS1) and [ICRL's 30 interaction turns](https://arxiv.org/html/2605.15224v1#A3). Those papers differ in other settings and do not establish identical benchmark protocols.

New WebShop evaluations must use 30 steps consistently for Greedy, Sol, generative SelectionARM and the head. Historical 100-step Greedy/Sol scores stay unchanged and require new matched 30-step controls. Research fitting and evaluation are later stages and are not launched by this collection chain.

**Updated completion estimate:** approximately **10–15 more GPU hours, plus queue and CPU-audit time**, provided throughput continues and storage does not pause submission. The four completed collection stages used 13.64 GPU hours and added 2,342 TextWorld states and 2,189 WebShop states: approximately 172 and 161 accepted states per GPU hour respectively. WebShop is the remaining bottleneck; its 1,961-state audited shortfall takes about 12.2 hours at that rate, or about 11.0 hours if all 191 pending WebShop states pass. This estimate accounts for the separate environment targets rather than dividing the combined shortage by aggregate throughput. It supersedes the earlier 4–6 day estimate based on 100-step collection, and excludes training and evaluation.

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

**Latest quota reading, October 19:** approximately **7.37 GiB free**, about 3.37 GiB above the unchanged 4 GiB submission guard. This is a shared, delayed quota reading and does not establish enough room for all remaining raw data, final exports and training outputs. Collection can pause at the guard even if its throughput-based ETA has not elapsed.

The project fileset was approximately **99% used**, with about 10 GiB reported free when recovery was prepared. Every future GPU submission now checks the fileset quota and stops if fewer than **4 GiB** remain. The check runs once per multi-hour batch, not in a polling loop. Quota reporting can lag and other users share the space; this guard cannot guarantee enough capacity for the entire remaining collection. Irreplaceable observations, labels and exports remain in project storage.

**Storage mitigation completed:** CPU-only job **369586** moved seven rebuildable audit-cache directories and four redundant smoke-test checkpoint directories to Scrubbed. All files passed SHA-256/size/mtime checks before the project originals were replaced with links. The final adapter weights remain on project and match those in the moved checkpoints. Original trajectories, teacher labels, native images, frozen datasets and source snapshots were not moved. Removed project file allocations total **3.51 GiB**. A subsequent quota reading at 04:28 UTC on October 19 reported **13.10 GiB free**, passing the unchanged 4 GiB guard; this is shared, time-varying headroom. The active audit-cache link also sends subsequent cache growth to Scrubbed. See the [storage receipt](diagnostics/storage-relocation-2026-10-18.json).

Before the horizon change, using the previous 903-state increment as a rough basis, 8,088 remaining states were estimated to require about **8.37 GiB for raw collection/images/teacher cache plus 0.61 GiB for incremental exports**. Final duplicate cohort exports and training outputs are additional; other users can consume the shared headroom. Full-history head features and disposable training intermediates should use scratch, while the final dataset/model and paid raw data keep persistent copies. A further **20–30 GiB of project headroom or an approved persistent storage destination** is a planning recommendation for completing collection and training comfortably, not a measured training-space requirement. The 4 GiB guard remains.

After collection reaches 5,000 training and 500 development states per environment, both methods receive the same frozen cohort. Full feature extraction, research training, deployment validation and paired Greedy/Sol comparisons still follow. Iterative DPO, extra seeds and official test evaluation remain deferred. The supervisor does not authorize or launch those stages.

The [current progress receipt](diagnostics/v6i-collection-progress-2026-10-19.json) records the reviewed counts and jobs. The [horizon-transition receipt](diagnostics/v6i-webshop30-transition.json), [startup-recovery receipt](diagnostics/v6h-startup-recovery.json), and [earlier recovery receipt](diagnostics/v6g-collection-recovery.json) preserve historical snapshots. Cluster `STATUS.md`, `status.json`, and `supervisor/status.json` update during execution. GitHub is a reviewed snapshot, not an unattended live uploader.
