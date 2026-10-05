#!/usr/bin/env python
"""Backfill / tail OpenWebRL run.log metrics into wandb.

The launched job had no --use-wandb (no WANDB_API_KEY at launch). slime still
prints every metric dict it would have logged, so this parses those lines:

  rollout.py:1467          - perf N: {...}     rollout/* + perf/* per iteration
  train_metric_utils.py:45 - perf N: {...}     perf/* train timing per iteration
  data.py:359              - rollout N: {...}  rollout/* reward/advantage stats
  model.py:784             - step K: {...}     train/* per optimizer step

plus two derived series from the browser pool:
  rollout/completed_groups  (groups finished to fill 48 accepted; lower = faster)
  browser/healthy_ratio, browser/sessions  (LocalProcessExit healthy=True/False)

x-axes: rollout/step (iteration) for rollout|perf|browser, train/step for train.
State (byte offset + sent keys) is kept next to the log so --follow can resume
without duplicates. Run id is fixed so re-runs append to the same wandb run.
"""
import argparse, ast, json, os, re, sys, time

DICT_RE = re.compile(r"(rollout\.py:1467 - perf|train_metric_utils\.py:\d+ - perf|data\.py:\d+ - rollout|model\.py:\d+ - step) (\d+): (\{.*\})\s*$")
DONE_RE = re.compile(r"RolloutProgress\] event=done rollout_id=(\d+) accepted_groups=(\d+)/(\d+) completed_groups=(\d+)")
EXIT_RE = re.compile(r"LocalProcessExit\] .*healthy=(True|False)")
STEP_END_RE = re.compile(r"train_one_step end rollout_id=(\d+) ppo_epoch=(\d+) step_id=(\d+)")
JUDGE_RE = re.compile(r"Browser reward: format=[0-9.]+ judge=([0-9.]+) combined=")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--project", default="openwebrl")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "heqj3-university-of-washington"))
    p.add_argument("--name", default="owrl_native_frepro_l40s")
    p.add_argument("--id", default=None, help="wandb run id (default: derived from name)")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--interval", type=float, default=120)
    p.add_argument("--state", default=None, help="state json path (default: <log>.wandb_state.json)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"offset": 0, "sent": [], "healthy": 0, "unhealthy": 0, "jn": 0, "js": 0, "rewards": {}, "succ": {}}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def main():
    a = parse_args()
    state_path = a.state or a.log + ".wandb_state.json"
    st = load_state(state_path)
    sent = set(tuple(x) for x in st["sent"])

    run = None
    if not a.dry_run:
        import wandb
        run = wandb.init(project=a.project, entity=a.entity, name=a.name,
                         id=a.id or re.sub(r"[^a-z0-9_-]", "-", a.name.lower()),
                         resume="allow", config={"source": "run.log backfill", "log": a.log})
        run.define_metric("rollout/step")
        run.define_metric("train/step")
        for pat in ("rollout/*", "perf/*", "browser/*"):
            run.define_metric(pat, step_metric="rollout/step")
        run.define_metric("train/*", step_metric="train/step")

    def emit(d):
        if a.dry_run:
            print(json.dumps(d)[:200])
        else:
            run.log(d)

    while True:
        size = os.path.getsize(a.log)
        if size < st["offset"]:      # log was rotated/truncated
            st["offset"] = 0
        n_new = 0
        with open(a.log, "r", errors="replace") as f:
            f.seek(st["offset"])
            for line in f:
                if not line.endswith("\n"):
                    break            # partial line; re-read next round
                st["offset"] += len(line.encode("utf-8", "replace"))
                m = EXIT_RE.search(line)
                if m:
                    st["healthy" if m.group(1) == "True" else "unhealthy"] += 1
                    continue
                m = JUDGE_RE.search(line)
                if m:
                    st["jn"] = st.get("jn", 0) + 1
                    if float(m.group(1)) >= 1.0:
                        st["js"] = st.get("js", 0) + 1
                    continue
                m = STEP_END_RE.search(line)
                if m:   # the 'step K:' dict follows this line; remember which iteration it belongs to
                    st["cur_rollout"] = int(m.group(1)); st["cur_step_in_iter"] = int(m.group(3))
                    continue
                m = DONE_RE.search(line)
                if m:
                    rid = int(m.group(1)); key = ("done", rid)
                    jn, js = st.get("jn", 0), st.get("js", 0)
                    st["jn"] = st["js"] = 0
                    if jn:
                        st.setdefault("succ", {})[str(rid)] = [jn, js]
                    if key not in sent:
                        tot = st["healthy"] + st["unhealthy"]
                        emit({"rollout/step": rid, "rollout/completed_groups": int(m.group(4)),
                              "rollout/accepted_groups": int(m.group(2)),
                              "browser/sessions": tot,
                              "browser/healthy_ratio": (st["healthy"] / tot) if tot else 0.0})
                        sent.add(key); n_new += 1
                    st["healthy"] = st["unhealthy"] = 0
                    skey = ("succ", rid)
                    if skey not in sent and jn:
                        succ = st["succ"]; win = [str(j) for j in range(rid - 9, rid + 1) if str(j) in succ]
                        N = sum(succ[j][0] for j in win); S = sum(succ[j][1] for j in win)
                        d = {"rollout/step": rid, "rollout/judge_success_all": js / jn, "rollout/judged_trajectories": jn,
                             "rollout/judge_success_all_ma10": S / N if N else 0.0}
                        rw = st.get("rewards", {}); rwin = [rw[str(j)] for j in range(rid - 9, rid + 1) if str(j) in rw]
                        if rwin:
                            d["rollout/raw_reward_ma10"] = sum(rwin) / len(rwin)
                        emit(d); sent.add(skey); n_new += 1
                    continue
                m = DICT_RE.search(line)
                if not m:
                    continue
                kind, idx, body = m.group(1).split(" - ")[1], int(m.group(2)), m.group(3)
                key = (kind + ("A" if "rollout.py" in m.group(1) else "B" if "train_metric" in m.group(1) else ""), idx)
                is_rollout_perf = kind == "perf" and "rollout.py" in m.group(1)
                if key in sent and not is_rollout_perf:
                    continue
                try:
                    d = ast.literal_eval(body)
                except Exception as e:
                    print(f"skip unparsable {kind} {idx}: {e}", file=sys.stderr)
                    continue
                d = {k: v for k, v in d.items() if isinstance(v, (int, float))}
                if is_rollout_perf and "rollout/raw_reward_mean" in d:
                    # 10-iteration moving average of the training reward (the raw series is +-0.05 noisy)
                    st.setdefault("rewards", {})[str(idx)] = float(d["rollout/raw_reward_mean"])
                    rw = st["rewards"]; rwin = [rw[str(j)] for j in range(idx - 9, idx + 1) if str(j) in rw]
                    ma = sum(rwin) / len(rwin)
                    d["rollout/raw_reward_ma10"] = ma
                    if ("ma", idx) not in sent:
                        if key in sent:   # perf dict already sent earlier: emit just the MA
                            emit({"rollout/step": idx, "rollout/raw_reward_ma10": ma})
                        sent.add(("ma", idx)); n_new += 1
                if key in sent:
                    continue
                if kind == "step":
                    d.setdefault("train/step", idx)
                    if "cur_rollout" in st:   # so train/* can also be viewed on the iteration axis
                        d["rollout/step"] = st["cur_rollout"]
                        d["train/step_in_iter"] = st.get("cur_step_in_iter", 0)
                else:
                    d["rollout/step"] = idx
                emit(d); sent.add(key); n_new += 1
        st["sent"] = sorted(sent)
        save_state(state_path, st)
        print(f"{time.strftime('%H:%M:%S')} offset={st['offset']} new={n_new} total_sent={len(sent)}", flush=True)
        if not a.follow:
            break
        time.sleep(a.interval)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
