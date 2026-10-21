# Qijia's simulated-environment experiments and training pipeline

Updated October 20, 2026 UTC. This page maps the team's TODO list to Qijia's work on **ALFWorld TextWorld and WebShop**, explains what is being trained, and separates implemented checks from research results. The [main report](README.md) contains the completed inference pilot. The [collection record](collection-status.md) tracks the labels.

**October 21 training complete:** all four selectors have finished: TextWorld and WebShop generative SelectionARM, and one listwise head for each environment. Both SFT jobs automatically resumed once and finished two epochs. The heads' development action agreement with Sol is 90.2% and 87.2%, respectively, measured on 500 states per environment used for checkpoint selection; these are not task-success rates. Generative-selector agreement has only an eight-prompt reload probe so far. Final artifact hashes passed the CPU completion audit. See [training results, limits and next evaluation](selector-training-results.md). Closed-loop success remains unmeasured for these checkpoints.

## Scope and order

There are two workstreams for Qijia in the shared team update:

1. **Actor versus actor plus ARM in simulated environments.** Compare the base actor with stronger selection. The current experiment uses Qwen3.5-4B and compares Greedy, Sol-of-five, a trained generative SelectionARM, and a trained **listwise** selection head. Report success and the cost of the entire system.
2. **Actor SFT with ARM in simulated environments.** Compare the base actor, an actor fine-tuned on ARM-selected examples, and an actor fine-tuned on direct teacher-generated actions. Then iterate collection, filtering and actor SFT in versioned rounds.

The first workstream supplies the learned selector needed by the second. Real-web infrastructure, CUA and actor RL belong to other workstreams in the pasted assignment. They are not launched here. The broader API-actor/API-selector matrix can be added as a later simulated-environment baseline; the current training pipeline does not silently add more API collection, checkpoints or execution seeds.

**Training a selector and training an actor are different operations.** In the current stage, the actor keeps proposing actions with unchanged weights. The two learned selectors replace Sol's decision among those actions. Actor SFT later changes the model that writes the action itself.

## What the two selector trainers do

Piotr's [SelectionARM recipe](https://github.com/piotr-teterwak/action-reward-models) trains a separate multimodal model to return the teacher's chosen candidate index. Our adaptation uses Qwen3.5-4B, native benchmark observations and the same public selection-prompt convention. It trains LoRA weights using the pinned LLaMA-Factory implementation; the actor is frozen.

Zixian's [selection head](https://github.com/zixianma/OpenWebRL/blob/50e815f176a90223b473b2b23badfd9160792de6/openwebrl/docs/ARM_SELECTION_HEAD.md) scores candidate features from a frozen actor with a small network. The [model definition](https://github.com/zixianma/OpenWebRL/blob/50e815f176a90223b473b2b23badfd9160792de6/openwebrl/selection_head.py) and [trainer](https://github.com/zixianma/OpenWebRL/blob/50e815f176a90223b473b2b23badfd9160792de6/scripts/train_selection_head.py) at the reviewed latest revision are byte-identical to our previously pinned `da20e8e` copies. We retain that code pin and now choose the **listwise architecture used by the latest headline table**. Version v1 validated the documented pointwise variant; it was not the headline architecture. Its automatic training controller has been stopped before any research fit. Version v2 changes only the head architecture and output namespace, reusing the same data, generative-SFT implementation and full-history feature extractor.

| Setting | Generative SelectionARM | Listwise selection head |
|---|---|---|
| Inputs | Native observation/images, selector history and all five candidate responses | Full actor prompt and candidate hidden states; native images retained |
| Trainable weights | Separate Qwen3.5 LoRA, rank 32, alpha 64, dropout 0.05 | 512-dimensional projections and a two-layer candidate-set Transformer, eight attention heads, dropout 0.1 |
| Target | Actual Sol-selected index as JSON | Soft target over candidates with the same native action as Sol's choice |
| Optimizer | AdamW through LLaMA-Factory; learning rate 1e-4 | AdamW; learning rate 3e-4, weight decay 0.05 |
| Full-data schedule | Two epochs; batch 1, accumulation 16; cosine, 5% warmup | 20 epochs; batch 256; OneCycle, 5% warmup; clipping at 1 |
| Validation selection | Lowest development loss | Highest development action-equivalent teacher agreement |
| Outputs | One adapter per environment | One head per environment |

The common target is **5,000 training and 500 development states per environment**, with the same panels, teacher labels and group split for both methods. Each state is one five-candidate decision, not five independent examples. Historical WebShop examples retain their original 100-step provenance; new episodes use 30 steps. TextWorld remains at 50 steps. The 1,024-token actor-output cap, T=0.7, top-p=0.9 and execution seed 42 remain unchanged.

Qwen3.5 has 32 decoder layers, so reference feature extraction uses layers 16/24/32, preserving the relative depths of upstream's 18/27/36. It pools thought tokens, action tokens and the final token, plus the prompt's last token. Normalizers are fitted on training features only. The entire actor prompt is processed: the selector's 32,768-token training cutoff is never applied to actor history. These model/template/layer/input-length changes are declared adaptations, not an exact reproduction of another backbone.

## Current labels versus the latest lookahead results

The team update's **K=1 lookahead** result uses a teacher with evidence after candidate execution. Our current Sol labels were collected **before execution**. They are usable for teacher-choice distillation, including Zixian's original teacher-labelled head formulation, but do not reproduce the lookahead supervision experiment. We will not relabel them as K=1 or attribute the real-web result to ALFWorld/WebShop. A lookahead comparison would need a separate version with candidate execution, replay validation, new labels and matched data budgets.

Likewise, current Sol selects a Qwen-generated response; Sol does not generate that response. Such an example can support a selected-candidate actor-SFT baseline, but cannot be called **direct teacher-generated action SFT**. That latter baseline needs separately collected teacher actions at matched training states, using only pre-action inputs and the native action space. Training a selector to emit `{"selection": 3}` is not actor SFT at all.

## Implemented training framework

Active cluster code lives in `arm-generalization/training_framework_v2/`, isolated from the running collector's frozen source. The framework and vendor revisions are hashed and copied into `artifacts/selector-training/v2-listwise-sol-preaction/source-snapshot/`. No collection sampling or source file was modified.

The preparation stage freezes the shared cohort, verifies content-based state identity and task-group isolation, preserves original images and full actor history, and audits/tokenizes SFT inputs on a CPU job. GPU jobs consume the immutable manifest. Tokenized data, compact features, optimizer checkpoints and redundant research-preparation exports use rebuildable scratch; the authoritative frozen collection cohort, manifests and final trained artifacts remain on project storage. Scratch is temporary and subject to the cluster's inactivity purge; it is not the sole copy of raw paid labels or final weights.

The initial functional cohort is drawn deterministically from the already frozen `round0-2500` milestone: **32 training + 8 development states per environment, 80 total**. This avoids reading a changing `audited-current` export. Both methods use exactly these states. WebShop includes native 0-, 1- and 10-image inputs. CPU preparation verified full actor prompts up to **33,817 tokens for TextWorld and 46,051 for WebShop**; 25 and 26 panels respectively have original captured candidate token IDs, and older re-encoded examples keep explicit provenance.

Validation runs the actual research code paths:

- Extract frozen-actor features, comparing short cases with the independently returned hidden states. Save compact resumable chunks and record time, memory and a clearly labelled FLOP proxy.
- Train the head for 10 epochs, reload model/optimizer/scheduler/random state, and finish 20 epochs. Check train-only normalizers, candidate permutation equivariance and exact checkpoint reload logits.
- Train each generative selector for four optimizer steps, save its optimizer/scheduler/random state, and resume the same eight-step schedule to step eight. Evaluate development loss, reload the saved adapter, and generate selections on the native development inputs. Invalid JSON selections remain visible; no fallback is counted as agreement.
- Report strict-index and action-equivalent agreement separately. These tiny fits test the pipeline; they are not benchmark success or reliable model-quality estimates.

CPU preparation **372550 passed**, including seven identity/split tests. The first preparation attempt, **372546**, stopped because its validation compared legacy path-based state IDs against the collector's content-based image IDs. The comparison now uses the collector's actual identity convention; no source labels or IDs were changed. A regression test covers image relocation. A further **10 controller tests passed in job 372555**, covering missing validation, incomplete data, dependency waiting, duplicate suppression, stop markers, hard failures and bounded checkpoint continuation.

**GPU validation job 372553 completed successfully in 7m 16s on one H200.** All 80 panels / 400 candidate responses passed full-history feature extraction (203.1 seconds; peak allocated feature memory 14.51 GiB on this subset). Both heads resumed from epoch 10 to 20, passed permutation/normalizer checks and reloaded exactly. Both generative selectors resumed from step 4 to step 8, reloaded with finite logits and returned a valid candidate index on all eight development probes per environment. The v1 pointwise head has **16,784,898 parameters**. **Correction:** the earlier explanation attributing its difference from 22.6M to hidden width was incorrect; changing only pointwise to listwise at the same width produces **22,562,818 parameters**, matching the headline architecture's stated scale. Earlier validation of the longest historical prompts exceeded 18 GiB, so this combined feature/SFT check used a full GPU. Head-only research fits use one MIG instance. No teacher API calls were made.

These are functionality checks, not task success results. The following development agreement numbers each use only eight states and a checkpoint selected on development data; they cannot establish a reliable gain or teacher-level behavior.

| Environment | Generative SFT: strict / equivalent agreement | Historical v1 pointwise head: strict / equivalent agreement |
|---|---:|---:|
| TextWorld | 4/8 / 5/8 | 4/8 / 6/8 |
| WebShop | 6/8 / 8/8 | 5/8 / 7/8 |

The [v1 validation receipt](diagnostics/qijia-training-framework-v1.json) preserves these historical checks. **The new listwise heads have also passed** on the same 80-state cohort in MIG job **372614** (30 seconds). Both trained for 20 epochs with an actual epoch-10 resume, passed candidate-permutation and train-only-normalizer checks, and reloaded exactly. Development strict/equivalent agreement was 3/8 and 5/8 for TextWorld, and 5/8 and 7/8 for WebShop. These selected-checkpoint results on eight examples are functionality diagnostics, not task scores or evidence for choosing an architecture. The headline architecture was chosen from the reference, not from these numbers. See the [v2 alignment receipt](diagnostics/qijia-training-framework-v2-listwise.json).

Unchanged SFT and feature-extraction checks were reused only after source-function comparison (normalizing versioned entry-point/output paths), cohort checksums, feature-chunk checksums and adapter hashes passed. The first v2 CPU preparation stopped on a missing alignment-proof receipt; the proof was supplied, rerun checks passed, and no GPU model job was started by that failed preparation.

## Data-ready handoff

The old v1 watcher **372577** was cancelled and its STOP marker set. The v2 CPU controller follows the successful listwise validation. Final data readiness is complete; reviewed recovery reconnects it after the quota-refresh check without holding a GPU. It requires both:

1. A passing, source-matched 80-state validation receipt.
2. The final immutable `round0-5000` cohort with 5,000/500 states per environment and a passing head-input readiness audit.

It then prepares the full shared cohort, extracts features, fits the two heads and fits the two generative selectors. Queued dependency jobs reserve no GPU. Each compute stage releases its allocation on exit. Feature chunks and optimizer checkpoints permit continuation across time limits. Only explicit checkpoint exits continue automatically, at most 12 continuations per long stage; code, quota, OOM and unclassified errors stop for review. The controller checks live Slurm configuration, source hashes and the 4 GiB project-space guard before submissions. A `STOP` marker in the training artifact root prevents future submissions; active jobs must be cancelled separately.

The controller stops after the four trained artifacts. It does not launch closed-loop evaluations, actor SFT, iterative rounds, DPO or official test evaluation. Those have different datasets and protocol requirements. Preparing a ready handoff is not a claim that data collection, research training or deployment validation is already complete.

## Qijia's next evaluation and actor-SFT stages

The first closed-loop comparison uses the fixed training-reserve pilot, seed 42, and matched horizons for every condition. WebShop needs new 30-step Greedy and Sol controls; the historical 100-step 18%/28% scores are not interchangeable. The primary comparison is **each method versus Greedy**. Record native success, WebShop reward, paired task-bootstrap intervals, invalid actions, truncation, fallback count, steps, end-to-end and component latency, API spend and GPU allocation time.

Compute accounting must include the entire actor-plus-selector system. Report cached/uncached input tokens and generated tokens where telemetry exists. A `2 × parameters × processed tokens` FLOP proxy must state its assumptions: attention cost and vision/language differences are omitted. API FLOPs are unknown, not zero. Our current head feature reference repeats full actor forwards; it cannot claim Zixian's low serving overhead. Generation-time feature capture for Qwen3.5/vLLM still needs implementation and numerical validation before an efficiency claim.

Actor SFT then gets its own frozen data and versioned recipes. Compare equal training-example budgets and matched task-group splits for ARM-filtered and direct teacher-generated targets. Preserve native images and full causal history; exclude reserved pilot and official test tasks from updates. The iterative loop is: freeze the actor/selector versions, collect fresh training rollouts, filter candidates with the ARM, freeze the actor-SFT cohort, update the actor, and evaluate the new version. Updating the actor invalidates the head's old hidden-feature representation, so features and head compatibility must be revalidated for every actor version. The current selector controller does not silently perform this actor update.

This scope is also recorded in the machine-readable cluster file `training_framework_v2/qijia_scope.json`. No Slack message or Overleaf change is sent by this work.
