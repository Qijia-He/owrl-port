# Can we use the data already collected?

**Yes: the saved data supports both methods, and a shared small batch has now passed their training and checkpoint-reload checks.** The current CPU audit covers **8,795 accepted examples** after five successful v6i collection/audit batches; the original detailed v6e audit below covered 2,009. The historical GPU validation below covers 12 states, including native images and the longest actor history. A new [80-state training/resume validation](qijia-training-workstream.md) has passed for both methods in both environments, and the gated full-data handoff is waiting for collection. Full feature extraction for the final research cohort, serving/deployment validation, and learned-selector benchmark evaluation remain to be done.

This report covers the `v6e-native-image-data-audit-text-web-1024-seed42` audit and the subsequent `v6f` continuation, on 2026-10-16 UTC. [Data-audit evidence](diagnostics/v6e-training-data-readiness.json) and [GPU-validation evidence](diagnostics/v6f-pipeline-validation.json) accompany the [Table 1(c) plan](selection-arm-head-plan.md). Iterative DPO remains deferred.

## October 19, 21:15 UTC: four successful collection batches

The accepted cohort now contains TextWorld **3,675 training + 500 development** states and WebShop **3,039 + 500**. Both development targets are complete; 3,286 training states remain. The current GPU job has another 315 provisionally eligible states awaiting audit, excluded from the accepted count. The supervisor followed all four successful batches without a recovery. Completed new-version summaries contain 603 WebShop tasks, none exceeding 30 actions. The latest audit reports zero selector-context exclusions. See the [collection record](collection-status.md) and [progress receipt](diagnostics/v6i-collection-progress-2026-10-19.json). These collection checks do not establish trained-selector accuracy.

## October 19: WebShop maximum reduced to 30 actions

At the user's request, version v6i uses a 30-action maximum for new WebShop episodes and retains TextWorld's 50-action maximum. CPU bootstrap 370003 preserved the historical cohort and accepted 271 additional states, bringing the total to 3,183. Imported WebShop data retains its 100-step provenance; full saved actor histories and native images are not truncated. Both methods will train on the same frozen cohort, with collection-horizon provenance disclosed. New WebShop controls and trained-selector evaluations must all use 30 steps; historical 100-step scores are not substitutes. Six horizon/history tests, 27 startup/supervisor checks and 87 regressions passed. Current jobs and audit evidence are in the [collection record](collection-status.md).

## Earlier October 19 startup recovery

At that earlier checkpoint, the dataset contained **2,912 accepted states**. Job 369558 ended before collection when the five-minute service-readiness deadline interrupted ongoing startup. Version v6h changes only startup waiting/diagnostics and supervisor classification: 900 seconds total, 600 seconds without log growth, bounded retry for a progressing timeout. Native-input/token checks are inherited only after source-hash verification. New startup/supervisor tests (27) and existing collection regression tests (87) passed. See the [collection record](collection-status.md) for current execution and failure evidence; no new training result is implied.

## October 18 collection update

The [collection and recovery record](collection-status.md) reports **1,429 train + 189 development** TextWorld states and **1,152 train + 142 development** WebShop states. GPU job 364528 failed on an empty native image URL after 3h 47m; recovery audit 369538 retained another 903 states with zero selector-context exclusions. Version v6g fixes that native edge case and adds a tested, bounded Slurm recovery supervisor. Available images remain mandatory. The current shortage is **8,088 states**, and project quota is a material constraint. Full research training has not begun.

## Authorized continuation: exact tokens and a shared validation cohort

The October 16 continuation was `v6f-exact-token-collection-text-web-1024-seed42`; the subsequent recovery was v6g, followed by v6h/v6i as described above. It preserves the v6e source snapshot and all accepted examples. Collection targets remain **5,000 training and 500 development states per environment**: 8,991 additional accepted states were needed at the start of this continuation. Sampling stays at five independent candidates, temperature 0.7, top-p 0.9, 1,024 output tokens, thinking enabled, and seed 42. Native images, actions, rewards, task grouping, and pilot exclusion stay part of the protocol.

New actor requests ask vLLM to return the original prompt and generated token IDs, including its stopping metadata. The collector fails if required IDs are absent. Existing records keep the explicit provenance **re-encoded saved response plus EOS**; recording tokens now cannot retroactively recover their original generation IDs. Both groups remain usable, with provenance preserved.

Before bulk feature extraction, a fixed **12-state validation cohort** uses the same records for generative SFT and head validation: TextWorld has two training and two development states; WebShop has four of each. It includes native inputs with zero, one, and ten images and the longest actor histories in both partitions. Development examples never enter optimizer updates or feature normalization. CPU preprocessing passed for this cohort, including the **161,591-token full actor prompt**. The much shorter selector prompt has its own separately audited 32,768-token limit; that limit is never applied to actor history.

The head reference extracts frozen Qwen3.5 features at layers **16, 24, and 32**, adapting the upstream 36-layer actor's 18/27/36 choice. Following [Zixian's feature extractor](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/selection_head_features.py), it pools candidate thought tokens, action tokens, and the final token for each layer, and separately encodes the last prompt-token state. The last decoder layer is measured after the final normalization, matching Transformers' returned hidden states. Short examples are checked against those directly returned hidden states. Long examples use selected-layer hooks to avoid retaining every layer's activations, while processing the entire prompt. The upstream convention appends an end token when absent, including on truncated candidates: a 1,025-token feature sequence can therefore represent a 1,024-token actor response plus a synthetic end token. Original captured IDs remain unchanged in the raw records. This reference makes a full forward pass for each candidate; it does **not** establish efficient reuse of generation-time features. Online reuse still needs validation of the hybrid attention cache, numerical differences, and the extra end-token step.

The optimizer check uses the upstream **pointwise** head with soft cross-entropy over action-equivalent choices, train-only standardization, and no value loss. Three optimizer steps, permutation checks, and checkpoint reload are pipeline checks, not evidence of teacher-level accuracy. Generative SFT likewise runs three steps per environment and reloads the adapter. At the final collection target, one frozen `round0-5000` cohort exports both SFT messages and corresponding head panels. Collection runs as a Slurm GPU/CPU dependency chain. The original chain stopped on failures; v6g now has the bounded recovery policy described in the collection record. It does not start research training or DPO automatically.

The validation runtime adds [fla-core 0.5.2](https://pypi.org/project/fla-core/0.5.2/) in an isolated package directory, using its fused gated-delta attention kernels with native PyTorch causal convolution. Existing environment packages and central PyTorch/vLLM versions are preserved. Package metadata and the wheel hash are recorded. This runtime needs GPU validation because Transformers disables its CUDA kernel imports in a CPU-only process; a CPU availability flag alone cannot establish GPU compatibility.

Two validation errors were diagnosed separately from model quality: the initial reload check passed a `BatchEncoding` where the model expected an `input_ids` tensor; the corrected check reused the completed TextWorld adapter. WebShop's first backward pass then hit the home-directory quota while writing compiler artifacts, before any optimizer update. The Triton and TileLang caches were moved to scratch, cache locations are explicitly forwarded into containers, and compiler thread counts follow the Slurm allocation. Training data and adapters remain in project storage. Failed-attempt logs are preserved; these attempts are not benchmark episodes.

### Validation results

| Check | Observed result |
|---|---|
| Original token recording | 15 generated candidates decoded exactly to the server's returned text; three native prompts matched independently rebuilt IDs; included a response ending at the 1,024-token cap |
| Full-history head features | All 12 states / 60 candidates processed; longest prompt 161,591 tokens; no prompt shortening |
| Feature-pooling reference | 40 candidate-pooling and eight prefix-pooling checks matched returned hidden states exactly on eight short states |
| Image coverage | Native inputs with 0, 1, and 10 images passed |
| Feature extraction resources | 167.24 seconds for this deliberately varied batch, including JIT; peak allocated GPU memory 28.89 GiB |
| Pointwise head training | Separate three-step TextWorld and WebShop checks; finite losses; train-only standardizers; permutation and checkpoint-reload checks passed |
| Generative SelectionARM training | Separate three-step LoRA checks; finite losses; both saved adapters reloaded and produced finite logits |

Head and token checks passed in job **364454**. TextWorld reload passed in **364474**, and WebShop training/reload plus the combined validation receipt passed in **364491**. CPU continuation job **364492** completed in 4m 35s, passed 82 tests, and re-audited all 2,009 imported examples and 420 image panels with zero context exclusions. All previously selected IDs were preserved. It submitted GPU collection job **364528**, which subsequently failed on an empty native image URL. The October 18 recovery above supersedes that old queued status. The Slurm jobs persist after the client disconnects. Cluster status files update automatically; the GitHub document is a publication snapshot, not a live scheduler dashboard.

These checks do not measure task accuracy. Full-context extraction is a correctness reference, not a serving-speed result. Prefix-only and joined-sequence forwards showed a maximum absolute feature difference of **0.4025** across different forward shapes in BF16; the implementation follows upstream by using the separately encoded prefix state. This observation, the hybrid attention cache, and the end-token convention must be addressed in generation-time capture validation before claiming equivalent efficient online features.

## Original v6e audited cohort (October 16)

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

This changes training preprocessing while preserving the actor's generation, benchmark, images, teacher labels, and task split. The new-policy optimizer and adapter-reload checks now pass; live selector serving/deployment still needs validation before benchmark evaluation.

## Which training formats are ready?

| Consumer | Available now | Still required |
|---|---|---|
| Piotr-style generative SelectionARM | Audited SFT files; small tokenized cohort; per-environment optimizer and adapter-reload checks | Prepare the final frozen cohort, run research SFT, and validate live selector deployment |
| Zixian-style pointwise selection head | Matching panels and equivalence labels; full-history extraction and pooling checks; small per-environment head optimizer/reload checks | Extract features for the full frozen cohort, run the research fit, and verify generation-time capture and serving |

Both consume the **same existing Sol labels**. The head does not need a second round of Sol annotation just to obtain features. Its missing features must be computed by the frozen Qwen actor.

The original 2,009 examples did not store generated token IDs. Their head export labels sequences as **re-encoded saved response text plus EOS**, following the upstream text-based extraction convention. Candidate reconstruction does not prove original tokenization identity. New v6f collection records exact IDs, and the exporter preserves this distinction. Online capture must still be checked against the chosen offline convention, including length-stopped responses and EOS handling. The upstream [feature code](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/selection_head_features.py) supports both text-derived and exact-ID feature construction.

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

Small feature extraction, exact-token logging, and corrected-policy optimizer/reload checks are complete. The immediate work is to finish collection and CPU audits, retain the existing validity flags, and freeze one final cohort for both methods. The eight WebShop flagged selections still need their own replay analysis. Next come full feature extraction, research fits, live deployment checks, and matched pilot evaluation. Iterative DPO, extra execution seeds and official test evaluation remain deferred.

The original audited study exports SFT files, records, provenance, and `head-panels.jsonl`. The v6f chain writes incremental `datasets/audited-current/<environment>/` snapshots and a final frozen `datasets/round0-5000/<environment>/` cohort; final head-panel export consumes those same records. Validation artifacts are under `validation/`. Raw prompts/images remain in cluster storage. Public evidence contains counts, policies and hashes, not credentials or raw observations. Background continuation is authorized and submitted; the current Slurm stage is recorded in the portable validation snapshot and cluster status files.
