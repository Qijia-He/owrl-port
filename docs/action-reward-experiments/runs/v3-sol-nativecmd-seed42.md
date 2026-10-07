# Closed-loop Sol pilot: v3-sol-nativecmd-seed42

Completed: **2026-10-06T09:40:38.962824+00:00**. Status: **complete and audited**.

Executed seeds: **[42]**. This snapshot contains **1,200 complete episodes**, including **120 reused smoke episodes**, each counted once. All Qwen actors are frozen 4B checkpoints.

[Overview](../README.md) · [Method and citations](../README.md#2-how-does-the-method-work) · [Setup](../README.md#3-what-environments-and-models-did-we-use) · [Interpretation and next steps](../DECISIONS.md) · [Source JSON](../data/v3-sol-nativecmd-seed42.json)

## What this experiment measures

At each decision, Qwen proposes actions and Sol chooses one of five proposals. Only the selected native action is executed; the environment then produces the next observation. This is an inference-only, closed-loop action-selection experiment. No actor updates, learned ARM training, or outcome-reward reshaping were performed in this pilot.

The four controls are greedy decoding, one stochastic sample, a uniform random choice from five stochastic samples, and Sol selection from five stochastic samples. Within each environment, all actors and controls use the same task manifest. Their trajectories can diverge, so they do not share a fixed candidate pool throughout an episode. Random-of-five controls the proposal count; it does not match the additional compute or cost of Sol.

## Task success

| Environment | Actor | Greedy | Single sample | Random of 5 | Sol of 5 | Sol minus random: pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 48.0% | 42.0% | 46.0% | 70.0% | +24.0 [+10.0, +38.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 16.0% | 20.0% | 18.0% | 22.0% | +4.0 [+0.0, +10.0] |
| ALFWorld THOR | Qwen3.5-4B | 14.0% | 10.0% | 8.0% | 18.0% | +10.0 [+0.0, +20.0] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 10.0% | 8.0% | 8.0% | 14.0% | +6.0 [+0.0, +14.0] |
| WebShop | Qwen3.5-4B | 18.0% | 22.0% | 22.0% | 28.0% | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | 28.0% | 28.0% | 28.0% | 30.0% | +2.0 [-6.0, +10.0] |

Each cell is the fraction of successful tasks; the seed-42 pilot has 50 tasks per cell. ALFWorld success follows native `won`; WebShop success requires native reward equal to 1. Differences are absolute percentage points (pp). Intervals apply to paired differences, not individual success rates. The internal strategy ID `gpt5` means five-candidate GPT selection; the actual selector is `gpt-6.1-sol`.

### Sol compared with every baseline

| Environment | Actor | Versus greedy: pp [95% CI] | Versus single sample: pp [95% CI] | Versus random of 5: pp [95% CI] |
|---|---|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | +22.0 [+10.0, +34.0] | +28.0 [+12.0, +44.0] | +24.0 [+10.0, +38.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | +6.0 [+0.0, +14.0] | +2.0 [-8.0, +12.0] | +4.0 [+0.0, +10.0] |
| ALFWorld THOR | Qwen3.5-4B | +4.0 [-4.0, +12.0] | +8.0 [+2.0, +16.0] | +10.0 [+0.0, +20.0] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | +4.0 [+0.0, +10.0] | +6.0 [+0.0, +14.0] | +6.0 [+0.0, +14.0] |
| WebShop | Qwen3.5-4B | +10.0 [+2.0, +18.0] | +6.0 [-2.0, +14.0] | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | +2.0 [-4.0, +10.0] | +2.0 [-4.0, +10.0] | +2.0 [-6.0, +10.0] |

## WebShop native mean reward

Reward is the native continuous score on a 0–1 scale, averaged over all episodes, including failures. It is reported separately from binary success.

| Actor | Greedy | Single sample | Random of 5 | Sol of 5 | Sol minus random [95% CI] |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 0.4163 | 0.5577 | 0.5303 | 0.5893 | +0.0590 [-0.0177, +0.1420] |
| Qwen3-VL-4B-Instruct | 0.6156 | 0.6026 | 0.6146 | 0.6542 | +0.0397 [-0.0163, +0.0993] |

| Actor | Reward versus greedy [95% CI] | Reward versus single sample [95% CI] |
|---|---:|---:|
| Qwen3.5-4B | +0.1729 [+0.0945, +0.2589] | +0.0316 [-0.0367, +0.1083] |
| Qwen3-VL-4B-Instruct | +0.0387 [-0.0157, +0.0973] | +0.0517 [+0.0103, +0.1060] |

## Steps, invalid actions, and truncation

Invalid-action and selected-truncation rates use executed steps as their denominator. Candidate truncation uses all generated candidates. Step-limit rate uses episodes. ALFWorld invalidity is checked against the native admissible commands before execution; WebShop checks syntax and current available interface actions. These diagnostics do not modify actions or filter candidates. Truncation means the actor response hit its output limit.

| Environment | Actor | Strategy | Mean steps | Invalid actions | Candidate truncation | Selected truncation | Step limit |
|---|---|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | Greedy | 30.30 | 60.1% | 51.7% | 51.7% | 52.0% |
| ALFWorld TextWorld | Qwen3.5-4B | Single sample | 34.86 | 59.9% | 40.1% | 40.1% | 58.0% |
| ALFWorld TextWorld | Qwen3.5-4B | Random of 5 | 32.12 | 61.1% | 36.4% | 35.7% | 54.0% |
| ALFWorld TextWorld | Qwen3.5-4B | Sol of 5 | 24.94 | 27.8% | 22.3% | 10.8% | 30.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | Greedy | 43.98 | 50.6% | 0.0% | 0.0% | 84.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | Single sample | 43.12 | 47.8% | 0.0% | 0.0% | 80.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | Random of 5 | 44.00 | 48.5% | 0.0% | 0.0% | 82.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | Sol of 5 | 42.12 | 41.3% | 0.0% | 0.0% | 78.0% |
| ALFWorld THOR | Qwen3.5-4B | Greedy | 43.72 | 89.2% | 80.6% | 80.6% | 86.0% |
| ALFWorld THOR | Qwen3.5-4B | Single sample | 45.48 | 89.5% | 72.9% | 72.9% | 90.0% |
| ALFWorld THOR | Qwen3.5-4B | Random of 5 | 46.36 | 86.2% | 74.2% | 74.2% | 92.0% |
| ALFWorld THOR | Qwen3.5-4B | Sol of 5 | 41.98 | 65.1% | 57.6% | 36.6% | 82.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | Greedy | 46.30 | 12.9% | 0.0% | 0.0% | 90.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | Single sample | 48.04 | 16.1% | 0.0% | 0.0% | 92.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | Random of 5 | 47.16 | 15.2% | 0.0% | 0.0% | 92.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | Sol of 5 | 45.38 | 15.8% | 0.0% | 0.0% | 86.0% |
| WebShop | Qwen3.5-4B | Greedy | 40.44 | 68.6% | 68.4% | 68.4% | 36.0% |
| WebShop | Qwen3.5-4B | Single sample | 22.60 | 40.1% | 39.9% | 39.9% | 16.0% |
| WebShop | Qwen3.5-4B | Random of 5 | 26.68 | 55.7% | 51.5% | 51.4% | 20.0% |
| WebShop | Qwen3.5-4B | Sol of 5 | 16.00 | 9.0% | 20.4% | 8.0% | 10.0% |
| WebShop | Qwen3-VL-4B-Instruct | Greedy | 7.26 | 0.6% | 0.0% | 0.0% | 4.0% |
| WebShop | Qwen3-VL-4B-Instruct | Single sample | 5.82 | 1.7% | 0.0% | 0.0% | 2.0% |
| WebShop | Qwen3-VL-4B-Instruct | Random of 5 | 8.42 | 0.2% | 0.0% | 0.0% | 4.0% |
| WebShop | Qwen3-VL-4B-Instruct | Sol of 5 | 8.30 | 0.2% | 0.0% | 0.0% | 2.0% |

## Candidate-pool diagnostics

The following table covers states visited by Sol. Distinct actions are counted by exact action-string equality; different reasoning with the same action does not count as a new choice. Because policies visit different states, these are descriptive diagnostics, not a causal decomposition of the gain.

| Environment | Actor | Mean distinct actions / pool | All 5 actions identical | All 5 responses truncated | Selector parse fallbacks |
|---|---|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 2.364 | 43.0% | 8.3% | 0 |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 1.292 | 74.6% | 0.0% | 0 |
| ALFWorld THOR | Qwen3.5-4B | 3.687 | 13.9% | 31.4% | 0 |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 1.075 | 92.9% | 0.0% | 0 |
| WebShop | Qwen3.5-4B | 2.328 | 39.4% | 4.5% | 0 |
| WebShop | Qwen3-VL-4B-Instruct | 1.400 | 71.8% | 0.0% | 0 |

## How the 95% confidence intervals are computed

1. Match treatment and baseline by task ID, requiring identical executed-seed coverage. For each task, compute `d_i = mean_seed(Sol outcome) - mean_seed(baseline outcome)`. Success outcomes are 0 or 1; the reward analysis uses continuous native reward.
2. Sample the matched tasks with replacement, keeping the sample size equal to the original task count. For this pilot, each draw contains 50 paired tasks. Compute the mean difference for each of 10,000 draws.
3. Report the 2.5th and 97.5th percentiles of those means. The statistical RNG is NumPy `default_rng(42)`; this bootstrap seed is separate from the number of experiment seeds.

Only one experiment seed was executed. These intervals describe uncertainty under task resampling, not variation across independent execution seeds. They are exploratory percentile-bootstrap intervals without multiple-comparison correction. An endpoint at zero should not be presented as clear evidence of improvement.

For TextWorld × Qwen3.5, Sol and random-of-five both succeed on 21 tasks; Sol alone succeeds on 14, random alone on 2, and both fail on 13. Thus the paired difference is `(14 - 2) / 50 = 0.24`, with a task-bootstrap interval of `[0.10, 0.38]`.

### Sensitivity to correlated task groups

ALFWorld includes related tasks within a task group. As a supplementary check, entire group IDs are resampled, retaining all their tasks and computing a task-weighted mean difference. This also uses 10,000 draws. It is a sensitivity analysis, not a replacement chosen after seeing the primary result.

| Environment | Actor | Tasks / groups | Task bootstrap: pp [95% CI] | Group bootstrap: pp [95% CI] |
|---|---|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 50 / 41 | +24.0 [+10.0, +38.0] | +24.0 [+11.5, +37.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 50 / 41 | +4.0 [+0.0, +10.0] | +4.0 [+0.0, +9.8] |
| ALFWorld THOR | Qwen3.5-4B | 50 / 42 | +10.0 [+0.0, +20.0] | +10.0 [+0.0, +20.4] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 50 / 42 | +6.0 [+0.0, +14.0] | +6.0 [+0.0, +13.5] |
| WebShop | Qwen3.5-4B | 50 / 50 | +6.0 [-4.0, +16.0] | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | 50 / 50 | +2.0 [-6.0, +10.0] | +2.0 [-6.0, +12.0] |

## Recorded protocol

| Item | Value |
|---|---|
| Selector | `gpt-6.1-sol`; maximum 2048 completion tokens |
| Stochastic proposals | 5 independent seeded draws; temperature 0.7; top_p 0.9; top_k -1 |
| Actor output / context limits | 1024 / 262144 tokens; enable_thinking `True` where supported |
| Selector context | Public prompt builder; last 5 history entries; complete candidate reasoning and action |
| Candidate handling | Keep duplicates and truncated responses; no action repair, filtering, or replacement draws |
| Version | `upstream-public-native-v3` / `v3-sol-nativecmd` / `native-alfworld-move-v1` |

See [the setup description](../README.md#3-what-environments-and-models-did-we-use) for checkpoint revisions, task sampling, native observations/actions, step limits, environments, and serving settings. The frozen method contract preserves the original three-seed plan; the executed-seed field and audit identify the actual one-seed scope.

## Costs

| Accounting scope | Returned-usage cost estimate | Calls |
|---|---:|---:|
| Native execution window, including smoke and interrupted attempts | $56.80 | 7,294 |
| Complete-episode attribution, including reused smoke once | $52.19 | — |
| All recorded Sol calls, including earlier diagnostics | $57.76 | 7,418 |

These are overlapping accounting views and must not be added together.

| Environment | Actor | Sol cost attributed to complete episodes |
|---|---|---:|
| ALFWorld TextWorld | Qwen3.5-4B | $9.82 |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | $1.80 |
| ALFWorld THOR | Qwen3.5-4B | $28.06 |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | $1.01 |
| WebShop | Qwen3.5-4B | $9.78 |
| WebShop | Qwen3-VL-4B-Instruct | $1.72 |

Tracked GPU allocation totals **46.32 H200-hours**, or **$41.69** at the planning rate of $0.90 per GPU-hour. This includes some earlier diagnostics and has a broader scope than this seed alone.

Costs are estimates from returned usage and recorded resource allocation, not reconciled invoices. Calls without returned usage may have charges that these records cannot resolve.

## Completion and audit evidence

- 1,200 summaries reconciled against 1,200 unique recorded trajectories.
- 6 environment–actor groups passed the launch/expansion gates, including episode completion, native image payloads, and selection membership.
- 0 unresolved infrastructure errors; 2 historical interruptions retained in the records.
- 120 smoke episodes reused as the prefix of the pilot manifests, without duplicate counting or selecting only successful outcomes.
- Completed-result hashes were preserved during recovery. Incomplete infrastructure attempts were recorded separately and rerun from the start of the episode.

Trajectory reconciliation verifies aggregation against recorded transitions; it is not an independent replay of the environments.

## Provenance and interpretation

Scientific configuration SHA-256: `78fc66d9a0ce60f25706e5f9537f6e184ae93117efd7cff0f1d02c53988ed7ee`.

The [immutable result package](../data/v3-sol-nativecmd-seed42.json) includes the frozen receipt hash, source-artifact hashes, protocol, per-seed summaries, paired intervals, candidate diagnostics, and accounting scopes.

- [Piotr reference repository at the inspected commit](https://github.com/piotr-teterwak/action-reward-models/tree/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce).
- [Zixian ARM integration at the inspected commit](https://github.com/zixianma/OpenWebRL/tree/134f064d867170eb833f6ab0f668a1b05050c031).
- [Method and reproduction boundaries](../README.md#upstream-limitations-we-found): source roles, public prompt alignment, and released-material inconsistencies.
- [Research interpretation and proposed next steps](../DECISIONS.md): what the results support and what remains untested.
