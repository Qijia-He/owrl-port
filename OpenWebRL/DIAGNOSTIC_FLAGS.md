# WebGym diagnostic flags (fork additions)

This fork adds four **env-gated, default-off** flags to the OpenWebRL eval harness.
With no flags set, the harness behaves byte-identically to upstream. They exist to
evaluate checkpoints that were RL-trained through the WebGym bridge
(`molmoweb-rl`), whose pre-2026-07-29 prompt taught models loose tool-call syntax,
and to reproduce WebGym's model-side eval conditions inside the native environment.

Added in commits `81c3d77` and `f385a51`.

## Flags

| flag | default | effect |
|---|---|---|
| `OWRL_TOOL_PARSER_LENIENT=1` | off | Forces the regex tool parser (bypassing sglang's qwen25 parser) and adds a loose-shape fallback: bare tool names (`<tool_call>go_back</tool_call>`), `"action"`/`"tool"`-keyed objects, arrays (first element), and name-on-own-line with optional JSON args. Argument-less `scroll` defaults to `direction=down`. Mirrors the WebGym bridge parser. |
| `OWRL_WEBGYM_BRIDGE_PROMPT=1` | off | Replaces the chat-template tools block with the WebGym bridge's pre-fix system prompt: `# Available tools` + `json.dumps(tool_list, indent=1)` appended to the policy, and no tools passed to the chat template. Byte-identical to the prompt bridge-trained checkpoints saw in training. |
| `--turn-history-reasoning-mode webgym_compress` | (CLI choice) | Exact port of the bridge's history compression: keep post-`</think>` text of past turns; unclosed `<think>` turns are reduced to their `<tool_call>` blocks; 2000-char cap per past turn. |
| `OWRL_BRIDGE_ACTION_PARITY=1` | off | Mirrors bridge execution semantics: only the first tool call of a turn executes; `drag`/`new_tab`/`switch_tab`/`close_tab` become a 1s `wait` (the bridge noops them). |

## When to use what

- **Stock upstream behavior** (benchmark-faithful): set nothing. Bridge-trained
  checkpoints will lose ~27% of episodes to `format_error_failed` — that is a real
  property of those checkpoints, not a harness bug.
- **Best-faith eval of a bridge-trained checkpoint** (e.g.
  [`zixianma/owrlsft-om2w-lr1e6-step32`](https://huggingface.co/zixianma/owrlsft-om2w-lr1e6-step32),
  [`zixianma/owrlsft-om2w-lr2e6-step40`](https://huggingface.co/zixianma/owrlsft-om2w-lr2e6-step40)):
  `OWRL_TOOL_PARSER_LENIENT=1` (+ `--context-num-screenshots 3
  --turn-history-reasoning-mode hide_thinking`, their trained input format).
- **Full WebGym-mimic** (WebGym's model-side conditions in the native env):
  all of the above plus `OWRL_WEBGYM_BRIDGE_PROMPT=1` and per-difficulty step
  budgets (run the task file split by difficulty with `--max-steps 30/50/70`).
- Checkpoints trained with the fixed native format stack (WebGym
  `WEBGYM_OWRL_STRICT_PARSER=1`, post-07-29 prompt) need **no flags**.

## Example

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <hf-checkpoint-dir> --port 9000 --tp-size 1 \
  --context-length 32768 --mem-fraction-static 0.6 &

export SLIME_BROWSER_ENV_MODE=local_process
export SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING=1
export SLIME_BROWSER_THINKING_OPEN_TAG='<think>'
export SLIME_BROWSER_THINKING_CLOSE_TAG='</think>'
export OPENAI_API_KEY=...                 # gpt-4.1 judge
export OWRL_TOOL_PARSER_LENIENT=1         # bridge-trained ckpt

python openwebrl/run_evaluate.py \
  --task-file openwebrl/data/online-mind2web.jsonl \
  --hf-checkpoint <hf-checkpoint-dir> \
  --output ./out --n-parallel 6 --sglang-ip 127.0.0.1 --sglang-port 9000 \
  --context-num-screenshots 3 --judge-max-attached-imgs 3 \
  --judge-model gpt-4.1 --judge-api-mode openai --judge-prompt-variant action_history \
  --turn-history-reasoning-mode hide_thinking \
  --browser-response-format-mode browser_env --browser-include-tool-response 1 \
  --max-steps 30 --turn-level
```

## Findings these flags produced (2026-07-29/30, OM2W test-300, temp 0, gpt-4.1)

- Parser leniency alone converts format-deaths into slow failures, not wins
  (lr1e6@32: 0.216 → 0.219 raw; 0 of 80 format-killed tasks became wins).
- With the judge and task set held fixed, the native environment out-scores the
  WebGym environment for both base and RL checkpoints (base 0.211→0.450,
  lr1e6 0.259→0.385 on shared-valid sets).
- Under fully identical conditions (full mimic bundle) base and lr1e6@32 tie
  (0.227 vs 0.243, p=0.64): the base's native advantage is a prompt-format
  advantage, not a capability gap.
