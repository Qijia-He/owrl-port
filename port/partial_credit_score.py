#!/usr/bin/env python
"""Partial-credit scorer for FAILED web-agent trajectories (offline prototype).

For every trajectory the original eval judged as not successful (COMPLETED with reward 0,
FAILED = max steps, TRUNCATED), ask a judge to
  1. decompose the task into concrete requirements R1..Rn,
  2. verify each against the screenshots (evidence must cite a screenshot; the agent's own
     claims are not evidence),
  3. name the first step at which the attempt became unrecoverable.
Score = verified / n, capped at CAP (successes stay strictly above every failure).

No torch / slime imports on purpose: runs anywhere with the `openai` package.
usage: python partial_credit_score.py --eval-dir <eval_*turn_*> --judge-model gpt-5.2 --out <jsonl>
"""
import argparse, asyncio, base64, glob, json, os, re, sys, time
from openai import AsyncOpenAI

CAP = 0.8
SYSTEM = """You are a strict evaluator of web-agent trajectories. You will be given a task, the agent's action history, several screenshots from the trajectory (labelled with their step index), and the agent's final message. The attempt was already judged NOT successful; your job is to measure how much of the task was genuinely accomplished.

Rules:
- Decompose the task into the smallest set of concrete, checkable requirements (R1, R2, ...). Typical web tasks have 2-6.
- Mark a requirement VERIFIED only if a screenshot shows the required state (right page, right filters, right item, right field value). The agent's own statements, plans, or final message are claims, not evidence.
- Mark it NOT VERIFIED if no screenshot supports it or a screenshot contradicts it.
- Then give the FIRST step index at which the attempt became unrecoverable (wrong site, wrong item, hallucinated completion, loop), or "none" if it simply ran out of steps on a reasonable path.

Output format (exact):
REQUIREMENTS:
R1: <text> -- VERIFIED|NOT VERIFIED -- <screenshot k or reason>
...
FIRST_ERROR_STEP: <integer or none>
PROGRESS: <verified>/<total>
"""

def turns(prompt, response):
    txt = (prompt or "") + (response or "")
    ts = re.findall(r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>", txt, re.S)
    return [t.strip() for t in ts if t.strip()]

def action_history(prompt, response):
    lines = []
    for i, t in enumerate(turns(prompt, response), 1):
        calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", t, re.S)
        acts = []
        for c in calls:
            try:
                d = json.loads(c); acts.append(f"{d.get('name')}({json.dumps(d.get('arguments', {}), ensure_ascii=False)[:160]})")
            except Exception:
                acts.append(c[:160])
        if not acts:
            acts = [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()[:160]]
        lines.append(f"Step {i}: " + "; ".join(acts))
    return "\n".join(lines) or "(no actions)"

def final_answer(response):
    m = re.search(r'"name"\s*:\s*"done".*?"response"\s*:\s*"(.+?)"\s*[},]', response or "", re.S)
    if m: return m.group(1)[:1500]
    tail = re.sub(r"<think>.*?</think>", "", response or "", flags=re.S).strip()
    return tail[-800:] if tail else "(no final message)"

def pick_frames(imgs, k=8):
    """Last frame of each turn; keep <= k frames evenly spaced, always including the last 3."""
    flat = [(i + 1, (x[-1] if isinstance(x, list) else x)) for i, x in enumerate(imgs) if x]
    if len(flat) <= k: return flat
    last3 = flat[-3:]; rest = flat[:-3]
    step = max(1, len(rest) // (k - 3))
    return rest[::step][: k - 3] + last3

def build_messages(intent, prompt, response, imgs):
    frames = pick_frames(imgs)
    user = [{"type": "text", "text": f"TASK:\n{intent}\n\nACTION HISTORY:\n{action_history(prompt, response)}\n\nAGENT'S FINAL MESSAGE (a claim, not evidence):\n{final_answer(response)}\n\nSCREENSHOTS ({len(frames)}):"}]
    for step, im in frames:
        url = im if im.startswith("data:") else f"data:image/png;base64,{im}"
        user.append({"type": "text", "text": f"Screenshot (step {step}):"})
        user.append({"type": "image_url", "image_url": {"url": url}})
    user.append({"type": "text", "text": "Your evaluation:\n"})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def parse(text):
    m = re.search(r"PROGRESS:\s*(\d+)\s*/\s*(\d+)", text or "")
    v, n = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    fe = re.search(r"FIRST_ERROR_STEP:\s*(\d+|none)", text or "", re.I)
    fe = (None if fe is None or fe.group(1).lower() == "none" else int(fe.group(1)))
    nreq = len(re.findall(r"^R\d+:", text or "", re.M))
    score = min(CAP, v / n) if (n and v is not None) else None
    return dict(verified=v, total=n, first_error_step=fe, n_requirements=nreq, score=score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True); ap.add_argument("--judge-model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--include-success", action="store_true")
    a = ap.parse_args()
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = {}
    for f in glob.glob(os.path.join(a.eval_dir, "results_task_*.jsonl")):
        d = json.loads(open(f).readline()); results[d["task_id"]] = d
    samples = {}
    for f in glob.glob(os.path.join(a.eval_dir, "eval_samples", "turn", "*", "*.json")):
        b = os.path.basename(f); t = next((t for t in results if b.startswith(t + "_")), None)
        if t: samples[t] = f
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out)}
    def wanted(r):
        st = r["status"].split(".")[-1]
        if st == "ABORTED": return False
        return a.include_success or float(r["reward"]) < 1
    todo = [t for t in sorted(results) if t in samples and wanted(results[t]) and t not in done]
    if a.limit: todo = todo[: a.limit]
    print(f"[pc] {a.eval_dir}\n[pc] judge={a.judge_model} todo={len(todo)} (done={len(done)})", flush=True)
    sem = asyncio.Semaphore(a.concurrency); outf = open(a.out, "a"); t0 = time.time(); n = [0]
    async def one(t):
        r = results[t]; d = json.load(open(samples[t]))
        msgs = build_messages(r["intent"], d["llm_input_texts"], d["llm_response"], d.get("images") or [])
        rec = dict(task_id=t, status=r["status"], orig_reward=float(r["reward"]), total_steps=int(r["total_steps"]), terminate_reason=r["terminate_reason"], judge_model=a.judge_model)
        async with sem:
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(client.chat.completions.create(model=a.judge_model, messages=msgs), timeout=240)
                    text = resp.choices[0].message.content or ""; rec.update(judge_text=text, **parse(text)); break
                except Exception as e:
                    rec["error"] = repr(e); await asyncio.sleep(3 ** (attempt + 1))
        return rec
    async def run():
        for coro in asyncio.as_completed([one(t) for t in todo]):
            rec = await coro; outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush(); n[0] += 1
            if n[0] % 25 == 0: print(f"[pc] {n[0]}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    asyncio.run(run()); outf.close()
    recs = [json.loads(l) for l in open(a.out)]; sc = [r["score"] for r in recs if r.get("score") is not None]
    print(f"[pc] DONE {a.judge_model}: scored={len(sc)}/{len(recs)} mean={sum(sc)/max(1,len(sc)):.3f} zeros={sum(1 for s in sc if s==0)} errors={sum(1 for r in recs if 'error' in r and r.get('score') is None)}", flush=True)

if __name__ == "__main__":
    main()
