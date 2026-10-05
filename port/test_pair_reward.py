#!/usr/bin/env python
"""Offline test of the pair-judge reward path in the TRAINING repo: feed saved eval
trajectories (as last-turn Samples) through reward_browser.reward_func exactly as slime's
batched RM would, with JUDGE_MODEL / JUDGE_MODEL_2 / JUDGE_PAIR_MODE from the environment,
and print per-trajectory (judge1, judge2, agree, combined)."""
import argparse, asyncio, glob, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--eval-dir", required=True); ap.add_argument("--limit", type=int, default=8); ap.add_argument("--task-ids", default="", help="comma-separated task ids to test (overrides --limit)")
ap.add_argument("--repo", default="/gscratch/cse/heqj3/owrl/OpenWebRL"); a = ap.parse_args()
sys.path.insert(0, a.repo)
os.environ.setdefault("JUDGE_API_BASE", "https://api.openai.com/v1"); os.environ.setdefault("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
from openwebrl import reward_browser as rb
from slime.utils.types import Sample
class Args:
    judge_api_model = os.environ.get("JUDGE_MODEL", "gpt-4.1"); judge_api_mode = "served"; judge_timeout_secs = 180.0
    judge_max_attached_imgs = 3; context_num_screenshots = 1; browser_response_format_mode = "browser_env"; judge_prompt_variant = "action_history"
    def __getattr__(self, n): return None
args = Args()
results = {json.loads(open(f).readline())["task_id"]: json.loads(open(f).readline()) for f in glob.glob(f"{a.eval_dir}/results_task_*.jsonl")}
files = {}
for f in glob.glob(f"{a.eval_dir}/eval_samples/turn/*/*.json"):
    b = os.path.basename(f); t = next((t for t in results if b.startswith(t + "_")), None)
    if t and results[t]["status"] != "Status.ABORTED": files[t] = f
picked = [t for t in a.task_ids.split(",") if t in files] if a.task_ids else sorted(files)[: a.limit]
print(f"pair test: judge1={args.judge_api_model} judge2={os.environ.get('JUDGE_MODEL_2')} mode={os.environ.get('JUDGE_PAIR_MODE')} n={len(picked)}", flush=True)
async def main():
    outs = []
    for t in picked:
        d = json.load(open(files[t])); raw = d.get("images") or []; imgs = [(x[-1] if isinstance(x, list) else x) for x in raw if x]
        st = getattr(Sample.Status, results[t]["status"].split(".")[-1])
        s = Sample(prompt=d["llm_input_texts"], response=d["llm_response"], status=st,
                   metadata={"intent": results[t]["intent"], "task_id": t, "total_steps": int(results[t]["total_steps"]),
                             "turn_index": max(0, int(results[t]["total_steps"]) - 1), "is_last_turn": True,
                             "terminate_reason": results[t]["terminate_reason"]},
                   multimodal_inputs={"images": imgs[-3:], "partial_images": imgs})
        outs.append((t, await rb.reward_func(args, [s]), s))
    for t, r, s in outs:
        pj = s.metadata.get("pair_judge", {}); rm = s.metadata.get("reward", {}); pc = s.metadata.get("partial_credit", {})
        print(f"{t[:10]} {results[t]['status'].split('.')[-1]:9s} orig={results[t]['reward']} judge={rm.get('judge')} partial={pc.get('score')} -> reward_func={r} combined={rm.get('combined')}")
asyncio.run(main())
