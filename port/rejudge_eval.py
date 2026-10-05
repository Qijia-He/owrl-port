#!/usr/bin/env python
"""Re-judge saved Online-Mind2Web eval trajectories with another judge model.

Replays exactly the judge path the eval used (reward_browser.compute_judge_reward_actionhistory:
native OpenWebRL action-history prompt, last-3 screenshots with inferred step indices,
SUCCESS / NOT SUCCESS parsing) on the trajectories saved under eval_samples/, so the
policy trajectories are held fixed and only the judge changes.

Mirrors reward_func: only Status.COMPLETED trajectories are judged; everything else
scores 0 under every judge (same as the original run).

Approximation vs the live run: the action history is extracted from the last turn's
prompt+response (all assistant turns are embedded in the prompt) instead of the list of
turn samples; both paths reduce to the same per-step tool-call summaries.

usage: python rejudge_eval.py --eval-dir <eval_*turn_* dir> --judge-model gpt-5.6-luna \
           --out <jsonl> [--limit N] [--concurrency 8]
"""
import argparse, asyncio, glob, json, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repo", default="/gscratch/cse/heqj3/owrl/OpenWebRL_eval")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    os.environ.setdefault("JUDGE_API_BASE", "https://api.openai.com/v1")
    os.environ.setdefault("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    os.environ.setdefault("SLIME_BROWSER_THINKING_OPEN_TAG", "<think>")
    os.environ.setdefault("SLIME_BROWSER_THINKING_CLOSE_TAG", "</think>")
    from openwebrl import reward_browser as rb
    from slime.utils.types import Sample

    class Args:
        judge_api_model = a.judge_model
        judge_api_mode = "served"
        judge_timeout_secs = 180.0
        judge_max_attached_imgs = 3
        context_num_screenshots = 1
        browser_response_format_mode = "browser_env"
        judge_prompt_variant = "action_history"
        def __getattr__(self, name):   # anything else (debug-trace knobs etc.) -> off
            return None
    args = Args()

    results = {}
    for f in glob.glob(os.path.join(a.eval_dir, "results_task_*.jsonl")):
        d = json.loads(open(f).readline()); results[d["task_id"]] = d
    samples = {}
    unmatched = []
    for f in glob.glob(os.path.join(a.eval_dir, "eval_samples", "turn", "*", "*.json")):
        base = os.path.basename(f)
        # task ids may themselves contain "_" (e.g. <hex>_110325): match by full-id prefix
        hit = next((t for t in results if base.startswith(t + "_")), None)
        if hit: samples[hit] = f
        else: unmatched.append(f)
    # ~7% of sample files carry a different id in the filename; match those by the task
    # intent embedded in the saved prompt (the eval's own visualizer does the same).
    if unmatched:
        by_intent = [(results[t]["intent"][:80], t) for t in results if t not in samples and results[t].get("intent")]
        for f in unmatched:
            try: txt = json.load(open(f))["llm_input_texts"]
            except Exception: continue
            for intent, t in by_intent:
                if intent and intent in txt and t not in samples:
                    samples[t] = f; break
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l)
            except Exception: continue
            # retry COMPLETED tasks that were skipped earlier for lack of a matched sample
            if r["status"] == "Status.COMPLETED" and not r.get("judged") and r["task_id"] in samples:
                continue
            done[r["task_id"]] = r
    if os.path.exists(a.out):   # rewrite without the records we are about to redo
        with open(a.out, "w") as fw:
            for r in done.values(): fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    todo = [t for t in sorted(results) if t not in done]
    if a.limit: todo = todo[: a.limit]
    print(f"[rejudge] {a.eval_dir}\n[rejudge] judge={a.judge_model} tasks={len(results)} samples={len(samples)} already={len(done)} todo={len(todo)}", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    outf = open(a.out, "a")
    stats = {"judged": 0, "success": 0, "skipped": 0, "errors": 0}

    async def one(task_id):
        r = results[task_id]
        rec = {"task_id": task_id, "status": r["status"], "terminate_reason": r["terminate_reason"],
               "orig_reward": float(r["reward"]), "total_steps": int(r["total_steps"]),
               "judge_model": a.judge_model, "judge_score": 0.0, "judge_text": None, "answer": None, "judged": False}
        if r["status"] != "Status.COMPLETED" or task_id not in samples:
            rec["judge_text"] = "not judged (status != COMPLETED or no saved sample)"; stats["skipped"] += 1
            return rec
        d = json.load(open(samples[task_id]))
        prompt, resp = d["llm_input_texts"], d["llm_response"]
        # eval_samples store images as one list per turn; mirror _collect_recent_trajectory_images:
        # one representative (last) screenshot per turn, then the most recent 3.
        raw = d.get("images") or []
        imgs = [(t[-1] if isinstance(t, list) else t) for t in raw if t]
        s = Sample(prompt=prompt, response=resp, status=Sample.Status.COMPLETED,
                   metadata={"intent": r["intent"], "task_id": task_id, "total_steps": rec["total_steps"],
                             "turn_index": max(0, rec["total_steps"] - 1), "is_last_turn": True,
                             "terminate_reason": r["terminate_reason"]},
                   multimodal_inputs={"images": imgs[-3:]})
        async with sem:
            try:
                score, text, timeout = await rb.compute_judge_reward_actionhistory(
                    args, s, trajectory_action_history_response=prompt + resp)
                rec.update(judge_score=float(score), judge_text=text, judged=True, judge_timeout=bool(timeout),
                           answer=rb._extract_final_answer(args, resp))
                stats["judged"] += 1; stats["success"] += int(score >= 1.0)
            except Exception as e:
                rec["judge_text"] = f"ERROR: {e!r}"; stats["errors"] += 1
        return rec

    async def run():
        t0 = time.time(); n = 0
        for coro in asyncio.as_completed([one(t) for t in todo]):
            rec = await coro; outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush(); n += 1
            if n % 25 == 0: print(f"[rejudge] {n}/{len(todo)} done, {stats}, {time.time()-t0:.0f}s", flush=True)
    asyncio.run(run())
    outf.close()
    allrec = [json.loads(l) for l in open(a.out)]
    succ = sum(1 for r in allrec if r["judge_score"] >= 1.0); orig = sum(1 for r in allrec if r["orig_reward"] >= 1.0)
    print(f"[rejudge] DONE {a.judge_model}: success={succ}/{len(allrec)} (original gpt-4.1 run: {orig}) stats={stats}", flush=True)

if __name__ == "__main__":
    main()
