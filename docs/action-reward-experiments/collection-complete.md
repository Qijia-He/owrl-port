# Collection complete and automatic-training recovery

Reviewed October 20, 2026. The final cohort is ready; formal selector fitting has not started at this snapshot.

| Environment | Training | Development | Total |
|---|---:|---:|---:|
| TextWorld | 5,000 | 500 | 5,500 |
| WebShop | 5,000 | 500 | 5,500 |
| Total | 10,000 | 1,000 | **11,000** |

These are decision states, each with five candidates and one Sol choice: 55,000 candidate responses in total. Both selectors use the same frozen cohort, separately by environment. Labels remain pre-execution choices, not K=1 lookahead or direct-teacher actor actions.

## Completion and audit repair

GPU job **373009** completed the final collection batch in **1h 53m 06s**, from 02:12 to 04:05 Pacific on October 20. It started earlier than the previous scheduler prediction. No more collection is needed for the authorized target.

Final encoding accepted all 11,000 states. The additional head-input audit in job **373475** then failed: its image-processing kwargs overwrote the CLI `args` variable. After TextWorld, access to `args.dataset` for WebShop raised an AttributeError. The controller stopped and allocated no training GPU.

A reviewed copy renames only that dictionary and its use to `render_kwargs`. CPU job **374276** completed the full two-environment audit in **8m 03s**, preserving every frozen source file, final dataset hash, label and previously generated TextWorld panel. No new teacher calls or GPU work were required. The [final receipt](diagnostics/final-cohort-2026-10-20.json) records the repaired/original source hashes and data checks. The frozen original remains historical; future versions must incorporate the rename before reusing that audit entry point.

## Verified data properties

All 5,500 actor prompts per environment were reconstructed and hash-checked with native images and full history. Maximum actor input lengths are **39,992 tokens for TextWorld** and **161,591 for WebShop**. Captured candidate-token IDs exist for 4,440 TextWorld and 4,551 WebShop panels; the other 2,009 historical panels retain explicit text-reencoding provenance.

Native-test-task, reserved-pilot-group and train/development-group overlaps are all zero. These checks do not claim official WebShop catalog product-disjointness. Readiness is not model accuracy; full feature extraction, fitting and closed-loop evaluation are subsequent stages.

## Storage and automatic training

The pre-recovery project quota reading was **2.69 GiB free**, below the unchanged 4 GiB guard. CPU job **374286** freed **3.09 GiB of allocated project space** by moving five verified rebuildable mutable export views to scratch and creating persistent lossless gzip archives for 74 large immutable raw trajectories. Each archive was decompressed and SHA-256 checked against its original before the original path became a link to an uncompressed scratch working copy. Raw bytes retain persistent project copies; frozen cohorts, images, teacher caches and final weights remain persistent. This allocation reduction is not a refreshed shared-quota reading.

CPU job **374288** scheduled resume job **374295** after a 16-minute quota-refresh delay. The resume wrapper checks the recovery receipts, exercises archive restoration on a fixture, checks real quota and invokes the unchanged v2 controller. It prepares the data, extracts full-history features, fits both listwise heads and fits both generative selectors. Queue availability and existing source/quota gates still apply. No GPU is held during the delay.

The handoff does not start actor SFT, DPO, extra seeds or official evaluation. See [training scope](qijia-training-workstream.md) and [maintenance](MAINTENANCE.md).


## Subsequent scheduling update

CPU preparation subsequently passed in job 374309 (19m 54s). The user then requested four independent training jobs. The frozen v2 workers and cohort remain unchanged; the serial controller was replaced with independent branches. See the [current scheduling record](parallel-training.md). Earlier resume details above describe the recovery at that time.
