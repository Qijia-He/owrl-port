"""Publish audited snapshots into a portable, versioned Markdown research log.

This only writes documentation locally. It neither launches experiments nor pushes
to GitHub. Re-rendering from exported data requires only the Python standard library.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/experiment-log'
ENV = {'alfworld_text': 'ALFWorld TextWorld', 'alfworld_thor': 'ALFWorld THOR', 'webshop': 'WebShop'}
ACTOR = {'qwen35': 'Qwen3.5-4B', 'qwen3_text': 'Qwen3-4B-Instruct-2507', 'qwen3_vl': 'Qwen3-VL-4B-Instruct'}
STRATEGIES = ('greedy', 'sample', 'random5', 'gpt5')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def register(receipt_path, run_id):
    import yaml
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', run_id):
        raise ValueError('Run ID must contain lowercase letters, digits and hyphens')
    receipt = json.loads(receipt_path.read_text())
    hashes = receipt['artifacts_sha256']
    for filename, expected in hashes.items():
        if Path(filename).name != filename or digest(receipt_path.parent / filename) != expected:
            raise ValueError('Frozen artifact verification failed: ' + filename)

    def artifact(suffix):
        matches = [name for name in hashes if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('Expected one artifact ending in ' + suffix)
        return json.loads((receipt_path.parent / matches[0]).read_text())

    summary = artifact('-final-summary.json')
    state = artifact('-final-state.json')
    audit = artifact('-final-audit.json')
    costs = artifact('-final-costs.json')
    sensitivity = artifact('-group-sensitivity.json')
    diagnostics = artifact('-final-candidate-diagnostics.json')
    config = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text())
    # Match armbench.core.json_hash, including its original whitespace encoding.
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != receipt['scientific_config_hash']:
        raise ValueError('Current configuration differs from the frozen snapshot; provide the matching configuration')
    if (state['status'] != 'completed' or not audit['passed'] or summary['infrastructure_errors'] or
            not all(row['expected_coverage_complete'] for row in summary['results']) or sensitivity['skipped']):
        raise ValueError('Only fully completed, audited experiments can be registered')
    method = json.loads((ROOT / 'configs/method.lock.json').read_text())
    gpu = {k: v for k, v in costs['gpu_usage'].items() if k != 'jobs'}
    package = {
        'schema_version': 1, 'id': run_id, 'completed_at': state['completed_at'],
        'completed_utc': datetime.fromtimestamp(state['completed_at'], timezone.utc).isoformat(),
        'run_tag': state['run_tag_suffix'], 'seeds': summary['expected_seeds'],
        'source_receipt': receipt_path.name, 'source_receipt_sha256': digest(receipt_path),
        'source_artifacts_sha256': hashes, 'scientific_config_hash': receipt['scientific_config_hash'],
        'protocol': {k: config[k] for k in ('actor_prompt_version', 'teacher_model', 'teacher_max_output_tokens',
            'candidate_count', 'temperature', 'top_p', 'top_k', 'actor_max_tokens', 'enable_thinking',
            'actor_context_tokens', 'inference_execution_mode')},
        'method': {'protocol': method['protocol'], 'piotr_commit': method['piotr_commit'],
                   'zixian_arm_commit': method['zixian_arm_commit'], 'contract': method['contract'],
                   'explicit_extensions': method['explicit_extensions']},
        'results': summary['results'], 'candidate_diagnostics': diagnostics['results'],
        'group_sensitivity': sensitivity['results'], 'audit': audit,
        'costs': {**{k: v for k, v in costs.items() if k != 'gpu_usage'}, 'gpu_usage': gpu},
    }
    path = DOCS / 'data' / (run_id + '.json')
    if path.exists() and json.loads(path.read_text()) != package:
        raise ValueError('An existing result version differs; register a new ID instead of overwriting history')
    atomic_write(path, json.dumps(package, ensure_ascii=False, indent=2) + '\n')


def pct(value):
    return f'{100 * value:.1f}%'


def interval(value, scale=100):
    lo, hi = value['ci95']
    precision = 1 if scale == 100 else 4
    return f"{scale * value['delta']:+.{precision}f} [{scale * lo:+.{precision}f}, {scale * hi:+.{precision}f}]"


def groups_of(data):
    groups = {}
    for row in data['results']:
        groups.setdefault((row['benchmark'], row['actor']), {})[row['strategy']] = row
    return sorted(groups.items())


LABEL = {'greedy': 'Greedy', 'sample': 'Single sample', 'random5': 'Random of 5', 'gpt5': 'Sol of 5'}


def success_table(data):
    lines = ['| Environment | Actor | Greedy | Single sample | Random of 5 | Sol of 5 | Sol minus random: pp [95% CI] |',
             '|---|---|---:|---:|---:|---:|---:|']
    for (benchmark, actor), group in groups_of(data):
        values = ' | '.join(pct(group[s]['success_rate']) for s in STRATEGIES)
        delta = interval(group['gpt5']['paired_comparisons']['random5']['success'])
        lines.append(f'| {ENV[benchmark]} | {ACTOR[actor]} | {values} | {delta} |')
    return lines


def render_run(data):
    run_id = data['id']
    rows = data['results']
    episodes = sum(r['episodes'] for r in rows)
    reused = sum(r['reused_smoke_episodes'] for r in rows)
    p = data['protocol']
    lines = [f'# Closed-loop Sol pilot: {run_id}', '',
        f"Completed: **{data['completed_utc']}**. Status: **complete and audited**.", '',
        f"Executed seeds: **{data['seeds']}**. This snapshot contains **{episodes:,} complete episodes**, including "
        f"**{reused} reused smoke episodes**, each counted once. All Qwen actors are frozen 4B checkpoints.", '',
        '[Overview](../README.md) · [Method and citations](../README.md#2-how-does-the-method-work) · [Setup](../README.md#3-what-environments-and-models-did-we-use) · '
        '[Interpretation and next steps](../DECISIONS.md) · [Source JSON](../data/' + run_id + '.json)', '',
        '## What this experiment measures', '',
        'At each decision, Qwen proposes actions and Sol chooses one of five proposals. Only the selected native action is executed; '
        'the environment then produces the next observation. This is an inference-only, closed-loop action-selection experiment. '
        'No actor updates, learned ARM training, or outcome-reward reshaping were performed in this pilot.', '',
        'The four controls are greedy decoding, one stochastic sample, a uniform random choice from five stochastic samples, '
        'and Sol selection from five stochastic samples. Within each environment, all actors and controls use the same task manifest. '
        'Their trajectories can diverge, so they do not share a fixed candidate pool throughout an episode. '
        'Random-of-five controls the proposal count; it does not match the additional compute or cost of Sol.', '',
        '## Task success', '', *success_table(data), '',
        'Each cell is the fraction of successful tasks; the seed-42 pilot has 50 tasks per cell. '
        'ALFWorld success follows native `won`; WebShop success requires native reward equal to 1. '
        'Differences are absolute percentage points (pp). Intervals apply to paired differences, not individual success rates. '
        f"The internal strategy ID `gpt5` means five-candidate GPT selection; the actual selector is `{p['teacher_model']}`.", '',
        '### Sol compared with every baseline', '',
        '| Environment | Actor | Versus greedy: pp [95% CI] | Versus single sample: pp [95% CI] | Versus random of 5: pp [95% CI] |',
        '|---|---|---:|---:|---:|']
    for (benchmark, actor), group in groups_of(data):
        values = ' | '.join(interval(group['gpt5']['paired_comparisons'][s]['success']) for s in STRATEGIES[:3])
        lines.append(f'| {ENV[benchmark]} | {ACTOR[actor]} | {values} |')
    lines += ['', '## WebShop native mean reward', '',
        'Reward is the native continuous score on a 0–1 scale, averaged over all episodes, including failures. '
        'It is reported separately from binary success.', '',
        '| Actor | Greedy | Single sample | Random of 5 | Sol of 5 | Sol minus random [95% CI] |',
        '|---|---:|---:|---:|---:|---:|']
    for (benchmark, actor), group in groups_of(data):
        if benchmark == 'webshop':
            values = ' | '.join(f"{group[s]['mean_reward']:.4f}" for s in STRATEGIES)
            lines.append(f"| {ACTOR[actor]} | {values} | {interval(group['gpt5']['paired_comparisons']['random5']['reward'], 1)} |")
    lines += ['', '| Actor | Reward versus greedy [95% CI] | Reward versus single sample [95% CI] |', '|---|---:|---:|']
    for (benchmark, actor), group in groups_of(data):
        if benchmark == 'webshop':
            values = ' | '.join(interval(group['gpt5']['paired_comparisons'][s]['reward'], 1) for s in STRATEGIES[:2])
            lines.append(f'| {ACTOR[actor]} | {values} |')
    lines += ['', '## Steps, invalid actions, and truncation', '',
        'Invalid-action and selected-truncation rates use executed steps as their denominator. '
        'Candidate truncation uses all generated candidates. Step-limit rate uses episodes. '
        'ALFWorld invalidity is checked against the native admissible commands before execution; WebShop checks syntax and current available interface actions. '
        'These diagnostics do not modify actions or filter candidates. Truncation means the actor response hit its output limit.', '',
        '| Environment | Actor | Strategy | Mean steps | Invalid actions | Candidate truncation | Selected truncation | Step limit |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for (benchmark, actor), group in groups_of(data):
        for strategy in STRATEGIES:
            row = group[strategy]
            values = ' | '.join(pct(row[k]) for k in ('invalid_action_rate', 'candidate_truncation_rate', 'selected_action_truncation_rate', 'step_limit_rate'))
            lines.append(f"| {ENV[benchmark]} | {ACTOR[actor]} | {LABEL[strategy]} | {row['mean_steps']:.2f} | {values} |")
    lines += ['', '## Candidate-pool diagnostics', '',
        'The following table covers states visited by Sol. Distinct actions are counted by exact action-string equality; '
        'different reasoning with the same action does not count as a new choice. '
        'Because policies visit different states, these are descriptive diagnostics, not a causal decomposition of the gain.', '',
        '| Environment | Actor | Mean distinct actions / pool | All 5 actions identical | All 5 responses truncated | Selector parse fallbacks |',
        '|---|---|---:|---:|---:|---:|']
    for row in data['candidate_diagnostics']:
        if row['strategy'] == 'gpt5':
            lines.append(f"| {ENV[row['benchmark']]} | {ACTOR[row['actor']]} | {row['mean_distinct_actions_per_pool']:.3f} | "
                         f"{pct(row['rates']['single_distinct_action_pools'])} | {pct(row['rates']['all_candidates_truncated'])} | {row['counts'].get('teacher_fallbacks', 0)} |")
    lines += ['', '## How the 95% confidence intervals are computed', '',
        '1. Match treatment and baseline by task ID, requiring identical executed-seed coverage. For each task, compute '
        '`d_i = mean_seed(Sol outcome) - mean_seed(baseline outcome)`. Success outcomes are 0 or 1; the reward analysis uses continuous native reward.',
        '2. Sample the matched tasks with replacement, keeping the sample size equal to the original task count. '
        'For this pilot, each draw contains 50 paired tasks. Compute the mean difference for each of 10,000 draws.',
        '3. Report the 2.5th and 97.5th percentiles of those means. The statistical RNG is NumPy `default_rng(42)`; '
        'this bootstrap seed is separate from the number of experiment seeds.', '',
        'Only one experiment seed was executed. These intervals describe uncertainty under task resampling, not variation across independent '
        'execution seeds. They are exploratory percentile-bootstrap intervals without multiple-comparison correction. '
        'An endpoint at zero should not be presented as clear evidence of improvement.']
    if run_id == 'v3-sol-nativecmd-seed42':
        lines += ['', 'For TextWorld × Qwen3.5, Sol and random-of-five both succeed on 21 tasks; Sol alone succeeds on 14, '
            'random alone on 2, and both fail on 13. Thus the paired difference is `(14 - 2) / 50 = 0.24`, '
            'with a task-bootstrap interval of `[0.10, 0.38]`.']
    lines += ['', '### Sensitivity to correlated task groups', '',
        'ALFWorld includes related tasks within a task group. As a supplementary check, entire group IDs are resampled, retaining all '
        'their tasks and computing a task-weighted mean difference. This also uses 10,000 draws. '
        'It is a sensitivity analysis, not a replacement chosen after seeing the primary result.', '',
        '| Environment | Actor | Tasks / groups | Task bootstrap: pp [95% CI] | Group bootstrap: pp [95% CI] |',
        '|---|---|---:|---:|---:|']
    for row in data['group_sensitivity']:
        if row['baseline'] == 'random5' and row['metric'] == 'success':
            group = row['supplementary_group_interval']
            lines.append(f"| {ENV[row['benchmark']]} | {ACTOR[row['actor']]} | {group['paired_tasks']} / {group['paired_groups']} | "
                         f"{interval(row['primary_task_interval'])} | {interval(group)} |")
    lines += ['', '## Recorded protocol', '', '| Item | Value |', '|---|---|',
        f"| Selector | `{p['teacher_model']}`; maximum {p['teacher_max_output_tokens']} completion tokens |",
        f"| Stochastic proposals | {p['candidate_count']} independent seeded draws; temperature {p['temperature']}; top_p {p['top_p']}; top_k {p['top_k']} |",
        f"| Actor output / context limits | {p['actor_max_tokens']} / {p['actor_context_tokens']} tokens; enable_thinking `{p['enable_thinking']}` where supported |",
        '| Selector context | Public prompt builder; last 5 history entries; complete candidate reasoning and action |',
        '| Candidate handling | Keep duplicates and truncated responses; no action repair, filtering, or replacement draws |',
        f"| Version | `{data['method']['protocol']}` / `{data['run_tag']}` / `{p['actor_prompt_version']}` |", '',
        'See [the setup description](../README.md#3-what-environments-and-models-did-we-use) for checkpoint revisions, task sampling, native observations/actions, '
        'step limits, environments, and serving settings. The frozen method contract preserves the original three-seed plan; '
        'the executed-seed field and audit identify the actual one-seed scope.', '',
        '## Costs', '']
    cost = data['costs']; window = cost['native_execution_window']; gpu = cost['gpu_usage']
    lines += ['| Accounting scope | Returned-usage cost estimate | Calls |', '|---|---:|---:|',
        f"| Native execution window, including smoke and interrupted attempts | ${window['returned_usage_cost_usd']:.2f} | {window['calls']:,} |",
        f"| Complete-episode attribution, including reused smoke once | ${cost['completed_episode_attributed_cost_usd']:.2f} | — |",
        f"| All recorded Sol calls, including earlier diagnostics | ${cost['all_recorded_sol']['returned_usage_cost_usd']:.2f} | {cost['all_recorded_sol']['calls']:,} |", '',
        'These are overlapping accounting views and must not be added together.', '',
        '| Environment | Actor | Sol cost attributed to complete episodes |', '|---|---|---:|']
    for (benchmark, actor), group in groups_of(data):
        lines.append(f"| {ENV[benchmark]} | {ACTOR[actor]} | ${group['gpt5']['charged_usd']:.2f} |")
    lines += ['', f"Tracked GPU allocation totals **{gpu['gpu_hours']:.2f} H200-hours**, or **${gpu['estimated_usd_at_0_90_per_gpu_hour']:.2f}** "
        'at the planning rate of $0.90 per GPU-hour. This includes some earlier diagnostics and has a broader scope than this seed alone.', '',
        'Costs are estimates from returned usage and recorded resource allocation, not reconciled invoices. '
        'Calls without returned usage may have charges that these records cannot resolve.', '',
        '## Completion and audit evidence', '',
        f"- {data['audit']['summaries_checked']:,} summaries reconciled against {data['audit']['unique_trajectories_checked']:,} unique recorded trajectories.",
        f"- {data['audit']['pilot_gates_passed']} environment–actor groups passed the launch/expansion gates, including episode completion, native image payloads, and selection membership.",
        f"- {data['audit']['current_infrastructure_errors']} unresolved infrastructure errors; {data['audit']['historical_infrastructure_errors']} historical interruptions retained in the records.",
        f"- {reused} smoke episodes reused as the prefix of the pilot manifests, without duplicate counting or selecting only successful outcomes.",
        '- Completed-result hashes were preserved during recovery. Incomplete infrastructure attempts were recorded separately and rerun from the start of the episode.', '',
        'Trajectory reconciliation verifies aggregation against recorded transitions; it is not an independent replay of the environments.', '',
        '## Provenance and interpretation', '',
        f"Scientific configuration SHA-256: `{data['scientific_config_hash']}`.", '',
        f"The [immutable result package](../data/{run_id}.json) includes the frozen receipt hash, source-artifact hashes, protocol, "
        "per-seed summaries, paired intervals, candidate diagnostics, and accounting scopes.", '',
        f"- [Piotr reference repository at the inspected commit](https://github.com/piotr-teterwak/action-reward-models/tree/{data['method']['piotr_commit']}).",
        f"- [Zixian ARM integration at the inspected commit](https://github.com/zixianma/OpenWebRL/tree/{data['method']['zixian_arm_commit']}).",
        '- [Method and reproduction boundaries](../README.md#upstream-limitations-we-found): source roles, public prompt alignment, and released-material inconsistencies.',
        '- [Research interpretation and proposed next steps](../DECISIONS.md): what the results support and what remains untested.', '']
    return '\n'.join(lines)


def render():
    packages = sorted((json.loads(p.read_text()) for p in (DOCS / 'data').glob('*.json')),
                      key=lambda row: (row['completed_at'], row['id']))
    if not packages:
        raise ValueError('No registered experiments')
    for data in packages:
        atomic_write(DOCS / 'runs' / (data['id'] + '.md'), render_run(data))
    latest = packages[-1]
    blocks = {
        'SNAPSHOT': f"Latest audited snapshot: **{latest['id']}**, completed **{latest['completed_utc']}**. "
                    f"Executed seeds: **{latest['seeds']}**; complete episodes: **{sum(r['episodes'] for r in latest['results']):,}**.",
        'SUCCESS': '\n'.join(success_table(latest)),
    }
    rewards = ['| Actor | Greedy | Single sample | Random of 5 | Sol of 5 | Sol minus random [95% CI] |',
               '|---|---:|---:|---:|---:|---:|']
    for (benchmark, actor), group in groups_of(latest):
        if benchmark == 'webshop':
            values = ' | '.join(f"{group[s]['mean_reward']:.4f}" for s in STRATEGIES)
            rewards.append(f"| {ACTOR[actor]} | {values} | {interval(group['gpt5']['paired_comparisons']['random5']['reward'], 1)} |")
    blocks['REWARD'] = '\n'.join(rewards)
    cost = latest['costs']; window = cost['native_execution_window']; gpu = cost['gpu_usage']
    blocks['COST'] = '\n'.join([
        '| Scope | Estimated cost |', '|---|---:|',
        f"| Sol during the native execution window, including smoke and interrupted attempts ({window['calls']:,} returned-usage calls) | **${window['returned_usage_cost_usd']:.2f}** |",
        f"| Sol attributed to complete episodes, including smoke once | ${cost['completed_episode_attributed_cost_usd']:.2f} |",
        f"| All recorded Sol calls, including earlier diagnostics | ${cost['all_recorded_sol']['returned_usage_cost_usd']:.2f} |",
        f"| Tracked GPU allocation ({gpu['gpu_hours']:.2f} H200-hours at $0.90/hour) | ${gpu['estimated_usd_at_0_90_per_gpu_hour']:.2f} |",
    ])
    history = ['## Experiment history', '', '| Completed (UTC) | Experiment | Executed seeds | Complete episodes | Status |',
               '|---|---|---|---:|---|']
    for data in reversed(packages):
        history.append(f"| {data['completed_utc'][:10]} | [{data['id']}](runs/{data['id']}.md) | {data['seeds']} | {sum(r['episodes'] for r in data['results'])} | Audited |")
    blocks['HISTORY'] = '\n'.join(history)
    # Preserve the self-contained narrative; update only explicit numeric blocks.
    # Future experiment updates must also review the surrounding interpretation.
    path = DOCS / 'README.md'
    overview = path.read_text(encoding='utf-8')
    for name, content in blocks.items():
        start, end = f'<!-- BEGIN GENERATED: {name} -->', f'<!-- END GENERATED: {name} -->'
        if overview.count(start) != 1 or overview.count(end) != 1:
            raise ValueError('Expected exactly one README block: ' + name)
        before, rest = overview.split(start, 1)
        _, after = rest.split(end, 1)
        overview = before + start + '\n\n' + content + '\n\n' + end + after
    atomic_write(path, overview)
    return len(packages)


def main():
    global DOCS
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', type=Path, help='Frozen artifact receipt to verify and export')
    parser.add_argument('--id', help='Stable experiment ID, e.g. v3-sol-nativecmd-seed42')
    parser.add_argument('--docs-root', type=Path, default=DOCS, help='Directory containing data/, runs/ and README.md')
    args = parser.parse_args()
    DOCS = args.docs_root.resolve()
    if bool(args.register) != bool(args.id):
        parser.error('--register and --id must be provided together')
    if args.register:
        register(args.register.resolve(), args.id)
    print(json.dumps({'registered_experiments': render(), 'entrypoint': str(DOCS / 'README.md')}))


if __name__ == '__main__':
    main()
