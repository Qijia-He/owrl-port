# ALFWorld and WebShop: matching output limits, models, and reported success

Reviewed 2026-10-14. This is a literature and public-code audit, not a reproduction. It supplements the [main experiment report](README.md). No new benchmark episodes, labels, or training were run for this review.

**Finding:** papers report relevant results alongside 1,024- and 2,048-token settings, but none of the references below establishes a controlled 1,024-versus-2,048 comparison matching our models and native observation protocol. In particular, ICRL's current request code overrides its configured 2,048-token limit. These numbers are reference points, not interchangeable leaderboard entries or evidence for the effect of increasing the cap.

Here, length means **new output tokens per actor turn**, not prompt length, total context, episode length, or a critic's allowance. ALFWorld means the text environment unless stated otherwise. Success rate (SR) is the fraction of tasks completed; WebShop reward also gives partial credit. The reward/score columns below use the papers' 0–100 scale, whereas our report uses 0–1.

## Published results associated with 1,024 tokens

[Self-Guide, Table 1 and Appendix A](https://arxiv.org/html/2604.03098v1) reports the following. Its appendix specifies 1,024 response tokens, 50 ALFWorld steps, 15 WebShop steps, and validation temperature 0.4 in its environment training settings. We have not verified a baseline-specific evaluation script establishing that every prompting row uses exactly that cap. Keep the model names as published; generic Qwen3-4B does not establish the Instruct-2507 checkpoint.

| Model | Method | Actor trained on these tasks? | ALFWorld SR % | WebShop SR % | WebShop score |
|---|---|---|---:|---:|---:|
| Qwen3-1.7B | ReAct | No RL | 10.2 | 4.7 | 41.6 |
| Qwen3-4B | ReAct | No RL | 20.3 | 11.3 | 31.1 |
| Qwen2.5-7B-Instruct | ReAct | No RL | 42.2 | 7.8 | 19.6 |
| Qwen3-4B | GRPO | Yes | 86.7 | 71.9 | 84.3 |
| Qwen3-4B | Full Self-Guide: GRPO + guidance + guidance reward | Yes | 96.9 | 78.1 | 87.8 |

ReAct describes a prompting method; it does not by itself imply temperature-zero Greedy decoding. Self-Guide also supplies admissible actions in its ALFWorld prompt, an observation difference from our actor.

[SkillRise, Section 4.1 and Table 1](https://arxiv.org/html/2607.26784v1) reports a 1,024-token setting and sampling evaluation at temperature 0.7 on 128 held-out instances per environment. Its WebShop uses **1,000 products and synthetic goals**. The entries below are pass@1, not its higher pass@3 results.

| Model | Method | Actor trained on these tasks? | ALFWorld SR % | WebShop SR % |
|---|---|---|---:|---:|
| Qwen3-4B | ReAct | No RL | 25.0 | 2.1 |
| Qwen3-4B | GRPO | Yes | 79.7 | 75.8 |
| Qwen3-4B | SkillRise | Yes | 85.9 | 84.4 |

The pinned [ALFWorld launcher](https://github.com/Within-yao/SkillRise/blob/2b38da3ad7b64414305d39609484803b07d3b9b0/examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh) and [WebShop launcher](https://github.com/Within-yao/SkillRise/blob/2b38da3ad7b64414305d39609484803b07d3b9b0/examples/skillrise_webshop/skillrise_webshop_qwen3_4b.sh) set `data.max_response_length=1024` and sampling validation at 0.7. The ALFWorld launcher selects `eval_in_distribution`; the WebShop launcher disables thinking and sets the small catalog with synthetic goals. These are checked launch settings, not a reproduction of every table row. They do not match our full-catalog, human-goal, image-preserving WebShop evaluation.

## Published results associated with 2,048 tokens — implementation caveat

[ICRL, Table 1 and Appendix C](https://arxiv.org/html/2605.15224v1) reports 2,048 response tokens per turn, a 16,384-token context, and 30 interaction turns in its training settings.

| Model as named in the paper | Method | Actor trained on these tasks? | ALFWorld SR % | WebShop SR % | WebShop reward |
|---|---|---|---:|---:|---:|
| Qwen3-4B | Prompting | No RL | 22.0 | 1.5 | 9.6 |
| Qwen3-8B | Prompting | No RL | 38.0 | 2.0 | 10.3 |
| Qwen3-30B-A3B | Prompting | No RL | 45.5 | 3.0 | 25.6 |
| Qwen3-4B | GRPO | Yes | 81.0 | 63.5 | 81.8 |
| Qwen3-4B | ICRL | Yes | 92.5 | 74.5 | 88.9 |
| Qwen3-8B | ICRL | Yes | 90.5 | 76.0 | 88.3 |

**These are not verified strict-2,048 evaluation scores.** At public commit `3ad79e421285db62e3d0c138dee1fce123f8c582`:

1. The [evaluation YAML](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/hydra_conf/eval/base.yaml) sets `eval_max_response_len: 2048`.
2. The [executor](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/generate.py#L163) instead computes a remaining budget from the context allowance minus the prompt, passes that budget to each generation call, and reduces it as the conversation grows.
3. The [request builder](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/runtime.py#L76) copies the sampling parameters and replaces `max_new_tokens` with that budget. This path does not clamp it to 2,048. Consequently, a 2,048 value in YAML does not establish a 2,048 cap in the actual request.
4. The [default custom configuration](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/hydra_conf/custom/icrl.yaml) allows two rounds. The [generation loop](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/generate.py#L58) can critique and retry a failed attempt during evaluation; the condition that would stop immediately for evaluation is commented out. Returning the final executor sample does not make this a single-attempt protocol.

This is a static finding about the released code, not proof of what generated the paper's historical results. Exact evaluation overrides, request logs, and task manifests are needed before claiming reproduction.

The current default [4B model configuration](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/hydra_conf/model/qwen3_4b_inst.yaml) references an Instruct-2507 model script; its [checkpoint configuration](https://github.com/brick-pid/ICRL/blob/3ad79e421285db62e3d0c138dee1fce123f8c582/ICRL/icrl/hydra_conf/checkpoint/base.yaml) also names that checkpoint. This is closer to our TextWorld Qwen3 actor, but does not independently identify the checkpoint behind every published row.

## Coverage of our exact models

| Our actor | What this review establishes |
|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | ICRL's current default points to this variant, with the execution caveats above. Other papers' generic Qwen3-4B labels are insufficient to establish an exact match. |
| Qwen/Qwen3.5-4B | No fully matched 1,024/2,048 result was found in the reviewed sources. |
| Qwen/Qwen3-VL-4B-Instruct | No fully matched 1,024/2,048 result was found in the reviewed sources. Text-model ALFWorld scores are not visual THOR scores. |

A relevant Qwen3.5 paper, [Self-Retrospection Distillation, Table 4 and Appendix C.5](https://arxiv.org/html/2610.08077v2), uses **8,192 actor-output tokens** for agentic training and evaluation. Its 2,048 figure is an auxiliary foresight allowance. It also uses sampled avg@8 evaluation. It therefore cannot supply a Qwen3.5-4B strict-2,048 Greedy reference.

## What alignment requires for our next protocol

Matching the cap is only one requirement. Before comparing absolute success with a paper, record:

| Dimension | Required match |
|---|---|
| Model | Exact checkpoint/revision, tokenizer/chat template, thinking mode, and any task-specific training |
| Generation | Actual per-request actor cap; Greedy versus sampling; stop rules; context budget and history handling |
| Observation | Text versus native images; admissible-action lists; extra guidance or critique |
| Environment | Version, task IDs and split; WebShop catalog size and human versus synthetic goals; TextWorld versus THOR |
| Evaluation | One attempt versus critique/retry; step cap; task weighting; success versus partial reward |

For our research question, the central comparison remains **the same frozen actor under Greedy versus Sol-of-five**, followed by the learned selector under the same protocol. The papers' task-trained actors answer a different question and should be labeled separately. External ReAct results should not be relabeled as our Greedy baseline without checking their decoder and prompts.

Our existing 50-task pilot uses reserved training tasks, native observations/images, and 50/100 decision limits for ALFWorld/WebShop. It is not an official test result. Preserve that record. If a separate paper-reproduction protocol is later chosen, version it explicitly rather than changing the native study to match a convenient score.

The completed [ten-task cap diagnostic](runs/v4-sol-nativecmd-output2048-diagnostic10.md) is our direct matched evidence about 1,024 versus 2,048, with its small-sample limitations. Comparing Self-Guide's 20.3 with ICRL's 22.0 would not isolate the length effect: model identity, prompts, tasks, decoding, and retry behavior are not established as equal.

This review does not choose a training task or authorize another run. Training-data collection, training, extra seeds, and official test evaluation remain pending the existing scope decisions.
