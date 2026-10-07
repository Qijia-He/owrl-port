# Decisions, limitations, and next steps

Updated 2026-10-07. The [main report](README.md) is the self-contained explanation of the project. This file preserves the decisions and operational history behind it.

## Current scope

- Complete: six environment–actor combinations, four conditions, 50 tasks per condition, execution seed 42; 1,200 complete episodes.
- Complete: all 120 smoke episodes reused as part of the same task manifests, with no duplicate counting; trajectory reconciliation and all six expansion audits passed.
- No experiments are running. Seeds 43 and 44 remain deferred because the user requested a summary after one seed.
- Benchmark ARM data collection/training and official held-out/test evaluation have not started.
- Earlier upstream inference checks and two optimizer steps per training recipe were installation checks, not full ARM training or a reproduction of all upstream results.

## Interpretation of the completed pilot

The clearest gain is **TextWorld × Qwen3.5**: success increases from 46% with random-of-five to 70% with Sol, a paired difference of +24 percentage points with a task-bootstrap interval of [+10, +38]. The supplementary task-group bootstrap also remains above zero.

The other five comparisons with random-of-five have positive point estimates, but intervals include or touch zero. This is insufficient evidence of a reliable gain across environments or execution seeds.

WebShop native mean reward increases from 0.5303 to 0.5893 for Qwen3.5, and from 0.6146 to 0.6542 for Qwen3-VL. Both paired reward intervals include zero. Report the estimates and uncertainty together.

## Limitations to investigate

| Observation | Evidence | Consequence for the next experiment |
|---|---|---|
| Qwen3.5 frequently exhausts its response budget | TextWorld selected-response truncation: random-of-five 35.7%, Sol 10.8%; in THOR, all five responses are truncated in 31.4% of Sol-visited pools | Separate avoiding unusable responses from better semantic choices. Any output-limit change requires a new version. |
| Qwen3 / Qwen3-VL often repeat the same action | All five actions identical in Sol-visited pools: TextWorld Qwen3 74.6%, THOR Qwen3-VL 92.9%, WebShop Qwen3-VL 71.8% | Check whether useful alternatives exist before expanding annotation. |
| Most THOR episodes hit the step cap | With Sol: Qwen3.5 82%, Qwen3-VL 86% | Inspect repeated actions, perception/navigation, and controller feedback. Do not infer end-to-end visual-control ability from high-level selection. |
| One execution seed and a small training-reserve sample | 50 tasks per cell; some ALFWorld tasks share a group | Treat this as exploratory. Task bootstrap does not measure execution-seed variation. |
| Extra selection compute is not cost-matched | Both five-candidate policies propose five actions, but only one calls Sol | Do not claim budget-optimal inference or equivalence to pass-at-five evaluation. |

Candidate diagnostics describe the states each policy actually visited. They are not counterfactual rollouts, oracle action-quality labels, or a causal decomposition of the gain.

## Proposed next work — not started

1. Analyze existing successes and failures to distinguish candidate quality, selection quality, and execution failures.
2. If a larger token limit or different diversity setting warrants a test, define a new version with comparable controls and retain the present snapshot.
3. Decide which settings justify ARM labels and training. Use the same public prompt convention for new teacher labels, training examples, and inference.
4. Evaluate on official held-out/test tasks once the protocol is fixed. Seeds 43 and 44 remain deferred.

## Method alignment

The [main report's method section](README.md#2-how-does-the-method-work) identifies Piotr's original implementation, Zixian's integration, and our changes. Its [upstream limitations section](README.md#upstream-limitations-we-found) explains the missing historical dependency, backbone-description mismatch, and released-data/public-prompt mismatch.

The completed pilot follows the public selection builder and inference helpers. Native benchmark observations/actions, Qwen actor checkpoints, and Sol are explicit adaptations. We do not claim byte-identical recovery of the unpublished historical teacher pipeline.

## 2026-10-06: completion and recovery

- Restricted execution to seed 42 following the user's instruction; no other seed generation was started.
- Recovered from one local inference connection interruption using bounded transport retries with the same seeded payload.
- Recovered from one Sol HTTP 500 interruption; retained request retry limits, explicit error records, and bounded shard recovery.
- Verified that all 1,137 / 1,186 completed summary hashes present before the respective recoveries remained unchanged.
- Retained interrupted attempts in operational/cost records; they were not counted as benchmark failures. Incomplete episodes were rerun from their start.
- Reconciled all 1,200 trajectories and summaries, passed all six protocol audits, froze the results, and released experiment resources.
- Established a Markdown log with tables rendered from frozen exported JSON.

Earlier prompt-grammar, serving-context, serving-mode, and model diagnostics are retained separately and excluded from this result version.

## 2026-10-07: publication and reader-facing report

- The user selected `Qijia-He/owrl-port`; documentation lives under `docs/action-reward-experiments/`, linked from the repository homepage.
- Rewrote the log in English and made the main README a complete introduction for a reader unfamiliar with the project.
- Added method attribution with pinned source citations, benchmark setup, baseline rationale, results, uncertainty, implementation status, and next steps to that single report.
- The table renderer preserves the explanatory narrative and updates only explicitly marked numerical blocks in the main report.
- This documentation update does not change the frozen numerical data or authorize additional experiments.
