# Table 1(c): train and compare two action selectors

Updated 2026-10-20 UTC. **This is the next experiment plan, not a report of trained-model results.** The paper was successfully cloned and its actual ALFWorld/WebShop panel inspected. Iterative DPO is deferred. The first deliverable is a single training round comparing a generative SelectionARM with a small selection head, using the frozen Qwen3.5-4B actor on TextWorld and WebShop.

## 1. The question and the table we need

Qwen proposes five possible next actions. Sol currently chooses one of them. We want to replace that recurring API call with a trained local selector, then measure whether the resulting agent retains useful gains over **Greedy**. Training a selector does not update the actor that proposes actions.

The current paper panel uses **single-sample** controls: TextWorld 42% and WebShop 22% for Qwen3.5. Those numbers are valid results of that condition, but differ from the user's requested Greedy comparison. The corresponding Greedy results are 48% and 18%. The proposed first block is:

| Frozen Qwen3.5-4B system | TextWorld SR | Change vs Greedy | WebShop SR | Change vs Greedy |
|---|---:|---:|---:|---:|
| Greedy | 48% | — | 18% | — |
| + Sol, five candidates | 70% | +22 pp | 28% | +10 pp |
| + Piotr-style SelectionARM, five candidates | Pending | Pending | Pending | Pending |
| + selection head, five candidates | Pending | Pending | Pending | Pending |

The populated cells are the **historical 50-task training-reserve pilot**, execution seed 42 and 1,024 actor-output tokens. They are neither official test scores nor promised performance of the trained selectors. A new backend or changed generation protocol requires new matched controls; it cannot silently inherit these numbers. Full historical evidence is in the [frozen result package](data/v3-sol-nativecmd-seed42.json) and [per-run report](runs/v3-sol-nativecmd-seed42.md).

The paper currently has one `our WDM` row per actor and describes the web WDM as generative. The two learned methods need distinct labels. The wider panel also contains Qwen3 text, Qwen3-VL THOR/WebShop, and a pending Sol-as-actor block. They are outside this first two-environment Qwen3.5 training stage. A head depends on its actor's hidden representations, so it cannot simply be transferred to a different actor. An API-only actor exposes no hidden states for this head; a separate generative selector can still evaluate its proposals.

For the existing secondary actors, changing the baseline to Greedy would make TextWorld Qwen3 16%→22% (+6 pp), THOR Qwen3-VL 10%→14% (+4 pp), and WebShop Qwen3-VL 28%→30% (+2 pp). These are historical comparisons, not additional training commitments. The mixed label `Qwen3-VL-4B (THOR)` should clarify that THOR applies only to its ALFWorld cell.

The paper's Sol footnote says the version was unavailable from the spreadsheet. Our frozen experiment package records the requested alias **`gpt-6.1-sol`**. That resolves the missing request identity, but does not provide an immutable dated provider snapshot or justify assigning the medium/high reasoning effort used in other paper experiments to this pilot. No effort override was sent here.

## 2. What Piotr trains

**SelectionARM is a separate language/vision model that learns to output a choice.** Its input contains the current observation, native images if present, recent history, and all five candidate responses. Its supervised target is a short answer such as `{"selection": 3}`. Standard supervised fine-tuning increases the probability of the teacher's answer tokens. At deployment, it generates an index and the environment executes that existing candidate.

The pinned [SelectionARM recipe](https://github.com/piotr-teterwak/action-reward-models/blob/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce/training/llamafactory/arm_lora.yaml) uses LLaMA-Factory, an OpenWebRL-4B-SFT base, LoRA on language-model linear layers, rank 32, alpha 64, dropout 0.05, learning rate 1e-4, two epochs, cosine scheduling, 5% warmup, and frozen vision/projector weights. Its template is `qwen3_vl`; the released backbone is Qwen3-VL, despite the broader repository description previously identified as inconsistent with its artifacts.

This is **selection SFT**. Piotr's separate scalar-reward/Bradley–Terry recipe is another method. DPO is not necessary to train this selection model.

For our Qwen3.5 experiment, the analogous student would be a separate Qwen3.5-4B selector with the correct `qwen3_5` template. Call it **Piotr-style SelectionARM adapted to Qwen3.5 and native benchmark observations**, not the released Piotr checkpoint. Its existing preparation recipe uses a 32,768-token input cutoff instead of upstream 8,192 so that all native WebShop images can fit. It uses batch size 1 with accumulation 16, keeping 16 examples per optimizer step on one GPU. These are declared adaptations; the actor's 1,024-token response cap is a different parameter and stays fixed.

## 3. What Zixian trains

**The selection head trains a small network on features computed by the frozen actor.** The actor proposes five responses. For each response, the implementation pools intermediate vectors representing its reasoning, action text, and final token, together with the prompt's last-token vector. A small network assigns five scores; the largest score selects the action.

The [documented main implementation](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/docs/ARM_SELECTION_HEAD.md) uses a pointwise MLP with features from layers 18, 27, and 36 of its OpenWebRL/Qwen3-VL actor. Its teacher-labelled training set contains **2,692 states and 35,916 candidate panels**; multiple panels can come from one state. Those counts are different units. The reported Online-Mind2Web result is 38.4% for the head versus 39.4% for Piotr ARM across three repetitions. That comparison is useful background, not an ALFWorld/WebShop result or proof of equivalence.

The [head definition](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/selection_head.py) standardizes features using training data only. Each candidate is scored independently in the pointwise version, but training compares the five scores through a softmax selection loss. If multiple candidates have the same executable action as the teacher's choice, target probability is shared among them. This is soft cross-entropy, not DPO or a numerical outcome-reward target.

The [trainer](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/scripts/train_selection_head.py) supplies the starting recipe: explicitly pass `--pointwise`; use 512-dimensional projections, dropout 0.1, AdamW at 3e-4 with weight decay 0.05, batch size 256, 20 epochs, a OneCycle schedule with 5% warmup, gradient clipping at 1, candidate-order augmentation, and checkpoint selection by development teacher agreement. The code's default without `--pointwise` is a different, listwise architecture.

The efficiency claim comes from **capturing features during the actor's own generation**. Re-running a full actor forward pass over every saved response is useful as a correctness reference, but adds substantial compute and does not reproduce that claim. The relevant implementation is [feature extraction](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/selection_head_features.py), [server capture](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/actor_head_capture.py), and [feature scoring](https://github.com/zixianma/OpenWebRL/blob/da20e8e45123682627c732bc5f9828f441d9ea33/openwebrl/actor_head_scoring.py).

## 4. How we will adapt and compare them

| Item | Generative SelectionARM | Selection head |
|---|---|---|
| Trainable parameters | LoRA weights of a separate Qwen3.5-4B selector | Small pointwise MLP; actor remains frozen |
| Supervision | Sol's selected index, represented as JSON tokens | Sol's selected action-equivalence class, represented as a distribution over five indices |
| Training software | Pinned LLaMA-Factory | Pinned Zixian PyTorch trainer plus native-action adapter |
| Data | Same retained states, five-candidate panels, teacher labels and group split | Same retained states, five-candidate panels, teacher labels and group split |
| Environment | Separate TextWorld and WebShop fits | Separate TextWorld and WebShop fits |
| First stage | One SFT round | One head-training round |

This produces **four trained artifacts**: two methods × two environments. They replace the selector, not the actor. They need the same teacher data; collecting two separate label sets would be unnecessary.

The existing collection target remains 5,000 training plus 500 development states per environment, up to ten retained states per task, with group isolation from the pilot reserve and official evaluation sets. Current collection keeps one panel per retained content-state identity. It therefore should not be described as matching upstream merely because a state count is larger. A small functional fit can use a frozen audited subset; substantive comparison should use one identical frozen cohort for both methods and report its actual state, panel, task, and group counts.

Native TextWorld observations stay text-only. WebShop retains every supplied visible image, in the original order, within the recorded ten-image capacity; excess capacity or processing errors must be explicit. No candidate is regenerated, repaired, or removed at evaluation because it is duplicated, invalid, or truncated. Dataset admission already excludes teacher fallbacks, truncated teacher answers, panels with no action diversity, and syntactically invalid teacher-selected actions. Report that filtering; do not claim the training distribution includes every visited state.

For head targets, use equality of parsed native TextWorld/WebShop commands. Zixian's browser-coordinate tolerance is not the action semantics of these environments. For generative SFT, preserve the actual teacher index. Report both strict-index and action-equivalent agreement so duplicate candidates do not make the two training objectives look artificially different.

The two systems use different representations and loss encodings. Holding their panels fixed is valuable, but does not by itself make this an isolated architecture ablation.

## 5. Required implementation and validation

1. **Native image CPU parity and small optimizer/reload checks are complete.** All 2,009 retained examples passed the stronger v6e audit, and v6f validated both training paths on a shared subset. The adapter is shared by auditing and training. Live deployment remains to be checked; see the [data-readiness report](training-data-readiness.md).
2. **Freeze a common dataset manifest.** Record image hashes, group splits, state/panel identities, selected indices, native actions, token lengths, and exclusion reasons. Keep the pilot and official evaluation tasks out of both training and checkpoint selection.
3. **Complete generative selector deployment.** Keep the actor unchanged; train the separate Qwen3.5 LoRA student; reload its checkpoint; require a valid index in 1–5. Use the public selection-prompt convention with native observations and the same train/deployment image processing. Log any parse fallback explicitly.
4. **Implement a Qwen3.5 feature reference.** The local Qwen3.5-4B configuration has 32 decoder layers and hybrid attention. Layers 16/24/32 are a proposed depth-matched starting point, not an upstream setting or a measured optimum. First validate a full-forward reference. Do not copy the Qwen3-VL layer numbers or cache logic unchanged.
5. **Preserve generation identity.** New collection/inference should save the actual prompt and response token IDs, model/template versions, native image order, finish reasons, and candidate seeds. Existing logs preserve response text and request metadata, but not exact response IDs. Reconstructed historical sequences require an explicit validation record; they must not be labelled exact captured generation. Recover the actor's full causal prompt, not just the selector's shorter history.
6. **Train and connect the pointwise head.** Pool only the specified vectors; save compact features rather than full token-level activations. Fit normalizers on training data, choose checkpoints on development data, and test candidate permutation, equivalent-action targets, images, empty spans, EOS handling and truncated responses. All five original candidates remain selectable.
7. **Validate capture during generation.** Compare online captured features, logits and selected indices against the reference. Handle the last token explicitly, since ordinary generation does not feed its final emitted token back through the model. Report numerical differences and near-tie disagreements. The upstream code uses sglang; our current actor uses vLLM. A backend switch requires a new protocol and matched controls. Do not report the inexpensive head deployment until this path works.

These steps define a one-round selector study. They do not require iterative policy updates, DPO, outcome labels, or lookahead rollouts. In particular, our existing Sol labels use **pre-action observations**. The paper's lookahead-supervision section describes richer privileged evidence; those claims must not be attached to this dataset.

## 6. Evaluation and deliverables

Start with a complete-episode check, then the fixed 50 pilot tasks per environment at execution seed 42. Report Greedy, Sol-of-five, learned SelectionARM-of-five, and selection-head-of-five. This is 400 episodes if all four conditions are rerun. If the exact frozen historical controls remain applicable, only the two new methods require 200 additional episodes; document the reuse rather than double-counting. Keep the 1,024-token actor cap, temperature 0.7/top-p 0.9 for five-candidate generation, native observations/actions/scoring, and use the newly authorized horizons: TextWorld 50, WebShop 30. WebShop therefore requires fresh matched 30-step Greedy/Sol controls; its historical 100-step controls cannot be reused. Extra seeds remain deferred.

The primary outcome is task success and the paired difference from Greedy. Also report WebShop native mean reward, 95% paired task-bootstrap intervals, invalid actions, truncation, episode lengths, selector fallbacks, API usage, GPU hours, and separately measured generation/selection latency. Teacher agreement is an intermediate diagnostic, not evidence of task improvement. Reserve the official ALFWorld held-out and WebShop test scores for the fixed final protocol; the training-reserve pilot remains development evidence.

Final deliverables are the four checkpoints, their exact dataset/config/source manifests, deployment checks, and a table separating the two learned methods. The report must distinguish historical controls from any new cohort and distinguish native feature reuse from a full-forward prototype. The paper should retain dashes until verified results exist. No private paper content beyond this experiment scope or raw observations is included in the public log.

## 7. Current execution status and remaining work

The [training-data readiness report](training-data-readiness.md) is the current audit record. CPU job 364409 completed the repaired encoding and field-reconstruction audit in 6m 36s, and job 364415 verified the global ten-state task cap, unique states and preservation of all 687 previously accepted examples.

- **Available:** 8,795 accepted examples: TextWorld 4,234 train/500 development, WebShop 3,561 train/500 development. Both development targets are complete; unfinished/current-batch data is excluded from that count. Historical 100-step WebShop data is retained; new collection uses 30 steps. Both methods use the same frozen cohort, with provenance retained. Original 2,009-state head panels remain validated; final-cohort feature extraction is still pending.
- **Fixed and validated:** the processing-limit and preliminary-resize mismatch. The native image processor performs a single resize. Full CPU prefix/grid/pixel checks and new-policy SFT optimizer/reload checks passed. Live deployment remains to be validated.
- **Head inputs:** historical IDs remain explicitly re-encoded; new collection records exact IDs. All 12 validation panels passed full-history extraction, including the 161,591-token WebShop prompt. Pointwise optimizer/reload/permutation checks passed per environment. Full-cohort extraction and online capture parity remain outstanding.
- **Quality in the original 2,009-state cohort:** adapter-invalid selected actions occur in 170 TextWorld and eight WebShop examples. Replay of all 170 flagged TextWorld states found 146 without an admissible candidate and 24 with an admissible alternative. The larger flag rate therefore mostly reflects candidate quality, not avoidable invalid teacher selection. The eight WebShop flags have not undergone the same replay analysis. No outcome-based relabelling or filtering has been applied.
- **Continuation:** Active version v6i follows the user-requested WebShop 30-step cap. Five collection/audit batches have completed successfully with zero recoveries; the sixth GPU job is running. The previous running v6h job was intentionally stopped and its completed data audited. See the [current collection record](collection-status.md). The [new training framework](qijia-training-workstream.md) gates authorized research fits on successful small-data validation and final data readiness. Iterative DPO, extra seeds and official held-out/test evaluation remain deferred.
- **Paper access:** the paper is cloned locally; no changes have been pushed to Overleaf. Piotr's released selector is identified; a downloadable Zixian head checkpoint has not been verified, and our Qwen3.5 head needs its own fit regardless.

The original collection target remains 11,000 examples, leaving 2,205 after the latest audit (8,795 accepted). The completed checks permit collection to proceed while preserving existing data and flags. Both methods consume the same final frozen cohort. Historical failures and recovery evidence are preserved in the [decision log](DECISIONS.md).
