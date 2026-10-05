#!/usr/bin/env python
"""Judge-ablation matrix from outputs/rejudge/<run>__<judge>.jsonl files.

Per checkpoint x judge: n_success/300 (raw), success excluding ABORTED, pair rules
(4.1 AND luna; 4.1 OR luna), disagreement/coverage, Cohen's kappa on COMPLETED,
and an optional canonical-v2 score computed through a shadow eval dir (results with
the judge's rewards + symlinked eval_samples) fed to compute_eval_success_rate_woaborted_v2.py.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys
RUNS = ["sft_gpt41_greedy_run1", "iter30_gpt41_greedy_run1", "iter60_gpt41_greedy_run1", "iter89_gpt41_greedy_run1"]
ROOT = "/gscratch/cse/heqj3/owrl"

def load(run, tag):
    p = f"{ROOT}/outputs/rejudge/{run}__{tag}.jsonl"
    if not os.path.exists(p): return None
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(p)}

def kappa(pairs):
    n = len(pairs)
    if not n: return float("nan")
    a = sum(1 for x, y in pairs if x == y) / n
    p1 = sum(x for x, _ in pairs) / n; p2 = sum(y for _, y in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (a - pe) / (1 - pe) if pe < 1 else float("nan")

def v2_score(run, tag, rewards):
    """Shadow dir with rewritten rewards -> canonical v2 scorer. Returns the success_rate line."""
    src = glob.glob(f"{ROOT}/outputs/eval_om2w/{run}/*/eval_*turn_*")[0]
    shadow = f"{ROOT}/outputs/rejudge_v2/{run}__{tag}/{os.path.basename(src)}"
    os.makedirs(shadow, exist_ok=True)
    for f in glob.glob(f"{src}/results_task_*.jsonl"):
        d = json.loads(open(f).readline())
        # the scorer's _reward_value only trusts a numeric top-level reward; a string falls
        # through to metadata.reward.combined (the ORIGINAL judge) -- write a float and patch both.
        rv = float(rewards.get(d["task_id"], 0.0)); d["reward"] = rv
        md = d.get("metadata")
        if isinstance(md, dict) and isinstance(md.get("reward"), dict):
            md["reward"]["combined"] = rv; md["reward"]["judge"] = rv
        open(os.path.join(shadow, os.path.basename(f)), "w").write(json.dumps(d) + "\n")
    link = os.path.join(shadow, "eval_samples")
    if not os.path.islink(link):
        if os.path.exists(link): shutil.rmtree(link)
        os.symlink(os.path.join(src, "eval_samples"), link)
    out = subprocess.run([f"{ROOT}/envs/webrl/bin/python", "openwebrl/compute_eval_success_rate_woaborted_v2.py", shadow],
                         cwd=f"{ROOT}/OpenWebRL_eval", capture_output=True, text=True, timeout=1800).stdout
    m = re.search(r"success_rate=([0-9.]+)%.*?denom=(\d+)", out)
    return (float(m.group(1)), int(m.group(2))) if m else (float("nan"), 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v2", action="store_true"); a = ap.parse_args()
    print(f"{'checkpoint':10s} {'judge':14s} {'succ/300':>9s} {'raw%':>6s} {'succ/nonabort':>13s} {'%':>6s}" + ("  v2%   denom" if a.v2 else ""))
    for run in RUNS:
        L = load(run, "gpt_5_6_luna"); G = load(run, "gpt_4_1")
        if not L: print(run, "no luna file yet"); continue
        tasks = sorted(L); n = len(tasks)
        orig = {t: L[t]["orig_reward"] >= 1 for t in tasks}
        luna = {t: L[t]["judge_score"] >= 1 for t in tasks}
        g41 = {t: G[t]["judge_score"] >= 1 for t in tasks} if G else None
        nonab = [t for t in tasks if L[t]["status"] != "Status.ABORTED"]
        arms = [("gpt-4.1 (orig)", orig), ("luna", luna),
                ("4.1 AND luna", {t: orig[t] and luna[t] for t in tasks}),
                ("4.1 OR luna", {t: orig[t] or luna[t] for t in tasks})]
        if g41: arms.insert(2, ("gpt-4.1 (rejudge)", g41))
        ck = run.split("_")[0]
        for name, r in arms:
            s = sum(r.values()); s2 = sum(r[t] for t in nonab)
            line = f"{ck:10s} {name:14s} {s:5d}/300 {100*s/300:6.1f} {s2:8d}/{len(nonab):<4d} {100*s2/len(nonab):6.1f}"
            if a.v2:
                v, d = v2_score(run, name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', ''), {t: float(r[t]) for t in tasks})
                line += f"  {v:5.1f}  {d:5d}"
            print(line)
        comp = [t for t in tasks if L[t]["judged"]]
        pairs = [(int(orig[t]), int(luna[t])) for t in comp]
        agree = sum(1 for x, y in pairs if x == y); both = sum(1 for x, y in pairs if x and y)
        o_only = sum(1 for x, y in pairs if x and not y); l_only = sum(1 for x, y in pairs if y and not x)
        print(f"{'':10s} judged(COMPLETED)={len(comp)} agree={agree} ({100*agree/max(1,len(comp)):.0f}%) kappa(4.1,luna)={kappa(pairs):.3f} bothS={both} 4.1-only-S={o_only} luna-only-S={l_only}")
        if g41:
            p2 = [(int(orig[t]), int(g41[t])) for t in comp]
            print(f"{'':10s} gpt-4.1 self-agreement={100*sum(1 for x,y in p2 if x==y)/max(1,len(p2)):.0f}% kappa={kappa(p2):.3f}")
        print()
main()
