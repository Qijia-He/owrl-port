# Can we use the data already collected?

**Yes: the saved data supports a first supervised SelectionARM fit and supplies the labels for a selection-head fit.** The complete CPU audit now passes. The head still needs frozen-actor feature extraction, and the corrected preprocessing needs a fresh optimizer/deployment check before a research training run. No trained-model result is claimed here.

This report covers `v6e-native-image-data-audit-text-web-1024-seed42`, audited on 2026-10-16 UTC. [Portable audit evidence](diagnostics/v6e-training-data-readiness.json) accompanies the [Table 1(c) plan](selection-arm-head-plan.md). Iterative DPO remains deferred.

## What exists now

| Environment | Training states | Development states | Total | Training task groups | Development task groups |
|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | 871 | 189 | 1,060 | 132 | 33 |
| WebShop | 807 | 142 | 949 | 186 | 44 |
| Total | 1,678 | 331 | **2,009** | 318 | 77 |

One example is **one decision state with five candidate responses and one Sol selection**. It is not a whole task, and five candidates do not count as five independently labelled examples. Each environment has its own dataset; 1,678 is the sum of their training counts, not the size of either separate fit.

The audit inspected 398 completed/context-exhausted episode summaries. Retained examples cover 166 TextWorld tasks and 230 WebShop tasks. Every retained state is unique under the recorded content-state identity, no task contributes more than ten examples, and all 687 examples from the previous completed audit remain present. The earlier approximately 1,930 figure was a worker-counter estimate, not the final scan and deduplication result. This audit made **zero new teacher calls**.

All six TextWorld task types appear in both partitions. Training coverage is uneven: for example, the lamp/look-at type has 52 states, while the two-object placement type has 226. These counts describe the collected cohort; they do not establish learning sufficiency.

## What passed

- **Image and prompt parity:** 2,009 retained-example prompt checks passed. All 420 distinct image panels were compared at both the visual-grid and image-tensor level against native Hugging Face preprocessing with the recorded serving pixel limits. No image was replaced or dropped.
- **Length:** the longest SelectionARM training input is 15,721 image-expanded tokens. No considered example exceeded the 32,768-token training cutoff, and no retained example was silently truncated.
- **Native observations:** 1,419 examples contain no image, 167 contain one, and 423 contain ten. TextWorld is text-only; WebShop also has native pages without product images. A zero-image native page is not a missing-image error.
- **Label identity:** every saved Sol answer parses to the recorded candidate index and to the exported SFT target. Every complete actor request can be rebuilt from the saved observation/history/images with the original request hash.
- **Splits:** no retained native held-out task ID or pilot group enters training/development, and training and development groups are disjoint. WebShop uses native training indices ≥1,500. This is not a claim that the official WebShop test uses a disjoint product catalog.
- **Reconstruction:** all 2,009 saved candidate panels reconstruct their original thought/action fields from the saved response text. The same state IDs connect the SFT and head exports.
- **Regression and preservation:** 76 existing regression checks passed. Prior source versions, failed audit records, image bytes, labels and the completed inference-result packages are preserved.

CPU job 364409 completed the encoding/readiness work in 6m 36s; CPU job 364415 passed the deduplication, task-cap and old-record preservation checks in 7s. Neither allocated a GPU. These checks establish dataset consistency against a native preprocessing reference. They are not a new live vLLM/optimizer validation.

## What changed to fix the images

The old training path resized images once in LLaMA-Factory and again in the model processor. That could produce a different patch grid from the native processor acting directly on the original image, even with the same nominal pixel budget. The audit also previously omitted the deployment-side maximum-pixel override.

The new opt-in policy, `native-qwen-single-resize-v1`, loads the original image, converts it to RGB, and lets the native model processor perform the sole resize with minimum 65,536 and maximum 262,144 pixels. The pinned LLaMA-Factory checkout is unchanged; a versioned adapter is used by both auditing and the training entrypoint. Every retained prompt is checked, rather than only one example per image count. Cache fingerprints now include both processing and audit policies.

This changes training preprocessing, not the actor's generation, benchmark, images, teacher labels, or task split. A new training smoke/deployment check is still required because earlier optimizer checks used the previous preprocessing path.

## Which training formats are ready?

| Consumer | Available now | Still required |
|---|---|---|
| Piotr-style generative SelectionARM | Separate training/development JSON files, native image paths, teacher-index answers, provenance, hashes and passing encoding checks | Tokenized training cache and a small optimizer/reload/deployment check under the corrected policy |
| Zixian-style pointwise selection head | Matching candidate panels, native action-equivalence targets, complete actor-prompt reconstruction, re-encoded candidate sequences and finish reasons | Extract Qwen3.5 frozen features, validate pooling and token conventions, then verify features captured during online generation |

Both consume the **same existing Sol labels**. The head does not need a second round of Sol annotation just to obtain features. Its missing features must be computed by the frozen Qwen actor.

The existing logs did not store the original generated token IDs. The head export therefore labels its sequences as **re-encoded saved response text plus EOS**, following the upstream text-based extraction convention; it does not misrepresent them as captured generation IDs. Candidate text reconstruction passed, but that is not proof that every original tokenization is identical. Future collection should preserve exact IDs, and online capture must be checked against the chosen offline convention, including length-stopped responses and EOS handling. The upstream [feature code](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/selection_head_features.py) supports both text-derived and exact-ID feature construction.

**The head needs the actor's full prompt.** That prompt is longer than the separate selector's recent-history prompt. The observed maxima are 37,058 tokens in TextWorld and 161,591 in WebShop; WebShop's 95th percentile is 84,879. These fit the actor's recorded 262,144-token context, but cannot be silently cut to the SFT selector's 32,768-token limit. Feature extraction must handle the long tail and save pooled vectors rather than all token activations. Measure this cost on a small representative subset before extracting the full set.

## Label quality still needs care

The data teaches a model to imitate Sol's preferences. It is not a set of verified optimal actions or lookahead/outcome labels.

| Diagnostic over retained train + development examples | TextWorld | WebShop |
|---|---:|---:|
| Selected action flagged invalid by the environment adapter | 170 / 1,060 (16.0%) | 8 / 949 (0.8%) |
| Panel contains at least one length-stopped candidate | 526 / 1,060 (49.6%) | 574 / 949 (60.5%) |
| Sol selected a length-stopped candidate | 1 / 1,060 | 0 / 949 |
| Always select candidate 1: action-equivalent teacher agreement | 54.8% | 46.9% |
| Uniform selection: expected action-equivalent agreement | 53.2% | 45.9% |

The invalid-action flag is an adapter diagnostic, not a complete semantic judgement. In TextWorld it tests membership in the current native admissible-command list. **Native replay has now checked all 170 flagged TextWorld labels across 55 episodes**, with stored observations, rewards and termination flags matching the replay. All 170 selected actions returned `Nothing happens.` Of these, **146 panels contained no candidate in the native admissible-command list**, while **24 panels offered an admissible candidate that Sol did not choose**. Thus, the 16.0% flag rate is not a 16.0% avoidable teacher-error rate; the directly identified avoidable invalid selections are 24/1,060 (2.3%). Admissibility is not proof that an alternative would complete the task. The eight WebShop flags have not undergone the same replay analysis. CPU job 364417 completed this diagnostic; no alternatives were executed and no labels were changed.

Duplicates explain why action-equivalent random agreement is well above 20%. Head validation should compare with this baseline, not assume a five-way chance level of 20%. High teacher agreement alone does not establish improved task success.

Before retaining this cohort, the existing admission policy excluded 3,752 panels whose five executable actions were identical and 899 panels whose teacher-selected action failed native syntax checks. Further eligible states can fall outside the ten-state task cap; those two exclusion counts are not a complete denominator for every raw decision. Non-selected invalid or truncated candidates remain in retained panels. Completed failed episodes are not discarded merely because the task failed. No new outcome-based filtering was introduced by this audit.

## How much remains, and what happens next?

| Environment | Agreed train target | Training shortfall | Agreed development target | Development shortfall |
|---|---:|---:|---:|---:|
| TextWorld | 5,000 | 4,129 | 500 | 311 |
| WebShop | 5,000 | 4,193 | 500 | 358 |

The agreed 11,000-example target has **8,991 examples remaining**. Existing data is sufficient to exercise both methods on real examples and to run an exploratory first fit; whether it is enough to preserve Sol's task-success gains is an empirical question. It does not need to be discarded or labelled again wholesale.

The immediate order is: retain the completed TextWorld validity flags and inspect the small remaining WebShop flagged subset, verify a small Qwen3.5 feature-extraction subset including long/image-heavy inputs, add exact-token logging for future collection, and continue toward the agreed target using the same split and sampling settings. Before research training, run the corrected SFT optimizer/deployment check. Both methods must use a shared frozen cohort for their comparison. Iterative DPO, extra execution seeds and official test evaluation remain deferred.

Private training artifacts are under the study's `datasets/audited-current/<environment>/`: `sft_train.json`, `sft_development.json`, `dataset_info.json`, `records.json`, `provenance.json`, and `head-panels.jsonl`. Raw prompts/images remain in cluster storage. The public evidence contains counts, policies and hashes, not credentials or raw observations. The collection chain is paused; no new collection or training job was submitted by this audit.
