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


def success_table(data):
    groups = {}
    for row in data['results']:
        groups.setdefault((row['benchmark'], row['actor']), {})[row['strategy']] = row
    lines = ['| 环境 | Actor | Greedy | 单次采样 | 五候选随机 | Sol 五选一 | Sol−随机：百分点 [95% CI] |',
             '|---|---|---:|---:|---:|---:|---:|']
    for (benchmark, actor), group in sorted(groups.items()):
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
    lines = [f'# Sol 闭环实验：{run_id}', '', f"完成时间：{data['completed_utc']}。状态：已完成并通过审计。", '',
        f"Seeds：{data['seeds']}；共 {episodes} 条完整 episode，其中 {reused} 条复用已审计 smoke，计入同一任务清单且只统计一次。",
        '每组 50 个训练集预留任务、4 种策略；所有 Qwen actor 均为 4B，权重冻结。', '',
        '[返回实验首页](../README.md) · [结构化数据](../data/' + run_id + '.json) · [决策与后续](../DECISIONS.md)', '',
        '## 成功率', '', *success_table(data), '',
        '表中的区间针对配对成功率差值，不是单个成功率的区间。`gpt5` 是代码中的“五候选 GPT 选择”策略名；本轮模型为 `' + p['teacher_model'] + '`。', '',
        '## WebShop 原生平均 reward', '',
        '| Actor | Greedy | 单次采样 | 五候选随机 | Sol 五选一 | Sol−随机 [95% CI] |', '|---|---:|---:|---:|---:|---:|']
    for actor in sorted({r['actor'] for r in rows if r['benchmark'] == 'webshop'}):
        group = {r['strategy']: r for r in rows if r['benchmark'] == 'webshop' and r['actor'] == actor}
        values = ' | '.join(f"{group[s]['mean_reward']:.4f}" for s in STRATEGIES)
        lines.append(f"| {ACTOR[actor]} | {values} | {interval(group['gpt5']['paired_comparisons']['random5']['reward'], 1)} |")
    lines += ['', '## 步数、无效动作与截断', '',
        '无效动作率和选中动作截断率按执行步数统计；候选截断率按全部候选数量统计；达上限比例按 episode 统计。', '',
        '| 环境 | Actor | 策略 | 平均步数 | 无效动作 | 候选截断 | 选中动作截断 | 达步数上限 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values = ' | '.join(pct(row[k]) for k in ('invalid_action_rate', 'candidate_truncation_rate', 'selected_action_truncation_rate', 'step_limit_rate'))
        lines.append(f"| {ENV[row['benchmark']]} | {ACTOR[row['actor']]} | {row['strategy']} | {row['mean_steps']:.2f} | {values} |")
    lines += ['', '## 候选池诊断', '', '下面仅统计 Sol 所访问的状态。不同策略会访问不同状态，因此这些指标是描述性证据，不是因果分解。', '',
        '| 环境 | Actor | 每池平均不同动作数 | 五个动作相同的池比例 | 五个候选全截断的池比例 | 解析 fallback 次数 |',
        '|---|---|---:|---:|---:|---:|']
    for row in data['candidate_diagnostics']:
        if row['strategy'] == 'gpt5':
            lines.append(f"| {ENV[row['benchmark']]} | {ACTOR[row['actor']]} | {row['mean_distinct_actions_per_pool']:.3f} | "
                         f"{pct(row['rates']['single_distinct_action_pools'])} | {pct(row['rates']['all_candidates_truncated'])} | {row['counts'].get('teacher_fallbacks', 0)} |")
    lines += ['', '## 95% CI 如何计算', '',
        '按 task_id 配对：dᵢ = Sol 在任务 i 的成功值 − 基线在同一任务的成功值。成功值为 0/1；reward 分析使用原生连续 reward。', '',
        '从 50 个配对任务中有放回抽取 50 个，计算平均差值，重复 10,000 次；取 2.5% 和 97.5% 分位数。统计重采样 RNG 固定为 42，独立于“运行了几个实验 seed”这一问题。', '',
        '本轮只有运行 seed 42：区间反映任务重采样的不确定性，不估计跨运行 seed 的方差。任务有多个匹配 seed 时，当前实现先在任务内平均，再重采样任务。区间未做多重比较校正。', '',
        'ALFWorld 中存在同组相关任务，额外按整个 group_id 重采样，保留所有组内任务与任务加权均值。下面是补充分析，不替换主分析。', '',
        '| 环境 | Actor | 任务 / 组 | 主任务 bootstrap：百分点 | 补充组 bootstrap：百分点 |',
        '|---|---|---:|---:|---:|']
    for row in data['group_sensitivity']:
        if row['baseline'] == 'random5' and row['metric'] == 'success':
            group = row['supplementary_group_interval']
            lines.append(f"| {ENV[row['benchmark']]} | {ACTOR[row['actor']]} | {group['paired_tasks']} / {group['paired_groups']} | "
                         f"{interval(row['primary_task_interval'])} | {interval(group)} |")
    lines += ['', '## 固定协议与原生 benchmark', '',
        '| 项目 | 设置 |', '|---|---|',
        f"| Selector | `{p['teacher_model']}`，输出上限 {p['teacher_max_output_tokens']} tokens |",
        f"| 候选采样 | 每步 {p['candidate_count']} 次独立 seeded draw；T={p['temperature']}，top_p={p['top_p']}，top_k={p['top_k']} |",
        f"| Actor 输出 / 上下文 | {p['actor_max_tokens']} / {p['actor_context_tokens']} tokens；enable_thinking={p['enable_thinking']} |",
        '| 选择上下文 | 原公开 builder；最近 5 步历史；完整候选 thought/action |',
        '| ALFWorld TextWorld | 原生文本，无占位图片；50 步上限 |',
        '| ALFWorld THOR | 原生 RGB 与文本反馈，官方 mrcnn_astar 控制器；50 步上限 |',
        '| WebShop | 原生可见文本与全部当前可见商品图片，搜索结果最多 10 张；原生动作/评分；100 步上限 |',
        '| 数据隔离 | 原生训练集内按任务/商品组预留 10%；固定 pilot 清单；正式 held-out/test 未使用 |',
        '| 候选处理 | 保留重复与截断候选；不改写、不筛选、不补采样 |',
        f"| 版本 | `{data['method']['protocol']}` / `{data['run_tag']}` / `{p['actor_prompt_version']}` |", '',
        'THOR 是固定官方控制器上的高层动作选择实验；原生动作、Qwen actor 和跨 benchmark 观察适配属于明确记录的方法扩展。', '',
        '## 成本与核验', '']
    cost = data['costs']; window = cost['native_execution_window']; gpu = cost['gpu_usage']
    lines += [f"- 本轮 native 执行窗口：Sol 返回 usage 估算 **${window['returned_usage_cost_usd']:.2f}**，{window['calls']:,} 次调用；包括 smoke 和中断尝试。",
        f"- 完整 episode 归属费用：${cost['completed_episode_attributed_cost_usd']:.2f}，已包含 smoke；与上一项不能相加。",
        f"- 全部已记录 Sol 调用：${cost['all_recorded_sol']['returned_usage_cost_usd']:.2f}，含此前诊断。",
        f"- 已追踪 GPU：{gpu['gpu_hours']:.2f} H200 小时，按 $0.90/小时估算 ${gpu['estimated_usd_at_0_90_per_gpu_hour']:.2f}；包含前期部分诊断，范围宽于本轮。",
        '- 以上是 usage/资源估算，未与账单对账；未返回 usage 的潜在费用无法据此确认。', '',
        f"核验：{data['audit']['summaries_checked']} 条汇总与 {data['audit']['unique_trajectories_checked']} 条记录轨迹一致；"
        f"{data['audit']['pilot_gates_passed']} 组通过审计；当前基础设施错误 {data['audit']['current_infrastructure_errors']}；历史中断 {data['audit']['historical_infrastructure_errors']} 次。",
        '原始轨迹复算验证的是聚合一致性，不是独立重放环境。中断尝试保留但不作为任务失败，恢复前已完成结果的哈希保持不变。', '',
        '## 数据来源', '', f"冻结配置 hash：`{data['scientific_config_hash']}`。完整来源文件 hash 保存在随附结构化数据中。", '',
        f"- [Piotr 原方法](https://github.com/piotr-teterwak/action-reward-models/tree/{data['method']['piotr_commit']})",
        f"- [zixian ARM 集成](https://github.com/zixianma/OpenWebRL/tree/{data['method']['zixian_arm_commit']})",
        '- 原私有 `frontier_arbiter` 未发布；当前对齐公开 builder，不能声称复原了历史私有数据生成 prompt。',
        '- 已发布骨干为 qwen3_vl；发布数据与公开 inference builder 的 prompt 存在差异，新的标注/训练/推理链路统一采用公开版本。', '',
        '## 解释与下一步', '', '[查看持续维护的决策记录](../DECISIONS.md)。数值表由已审计 JSON 生成；分析判断与数值来源分开维护。', '']
    return '\n'.join(lines)


def render():
    packages = sorted((json.loads(p.read_text()) for p in (DOCS / 'data').glob('*.json')),
                      key=lambda row: (row['completed_at'], row['id']))
    if not packages:
        raise ValueError('No registered experiments')
    for data in packages:
        atomic_write(DOCS / 'runs' / (data['id'] + '.md'), render_run(data))
    latest = packages[-1]
    lines = ['# Action-level Rewards 实验日志', '',
        '目标：冻结 Qwen，由 Sol 在五个候选动作中选择，评估原生 benchmark 闭环成功率；随后再决定 ARM 数据收集与训练。', '',
        '[协议与复现边界](runs/' + latest['id'] + '.md#固定协议与原生-benchmark) · [决策、问题与下一步](DECISIONS.md) · [如何更新](MAINTENANCE.md)', '',
        f"最近完成：**{latest['id']}**，{latest['completed_utc']}。最新实验包含 seeds `{latest['seeds']}`。当前执行范围与待办见决策记录。", '',
        '## 最新结果', '', *success_table(latest), '',
        f"[完整实验记录：reward、步数、无效动作、截断、候选重复、CI 与费用](runs/{latest['id']}.md)", '',
        '所有 Qwen actor 为 4B。表中为训练集预留 pilot；单 seed 结果不能代替正式 held-out/test 评测。', '',
        '## 实验历史', '', '| 完成日期（UTC） | 实验版本 | Seeds | 完整 episodes | 状态 |', '|---|---|---|---:|---|']
    for data in reversed(packages):
        lines.append(f"| {data['completed_utc'][:10]} | [{data['id']}](runs/{data['id']}.md) | {data['seeds']} | {sum(r['episodes'] for r in data['results'])} | 已审计 |")
    lines += ['', '新实验以新 ID 追加；旧结果保留。数值来自已冻结、已审计的导出数据，后续重新渲染不会运行模型或提交实验。', '']
    atomic_write(DOCS / 'README.md', '\n'.join(lines))
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
