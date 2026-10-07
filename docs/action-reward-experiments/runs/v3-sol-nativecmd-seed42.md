# Sol 闭环实验：v3-sol-nativecmd-seed42

完成时间：2026-10-06T09:40:38.962824+00:00。状态：已完成并通过审计。

Seeds：[42]；共 1200 条完整 episode，其中 120 条复用已审计 smoke，计入同一任务清单且只统计一次。
每组 50 个训练集预留任务、4 种策略；所有 Qwen actor 均为 4B，权重冻结。

[返回实验首页](../README.md) · [结构化数据](../data/v3-sol-nativecmd-seed42.json) · [决策与后续](../DECISIONS.md)

## 成功率

| 环境 | Actor | Greedy | 单次采样 | 五候选随机 | Sol 五选一 | Sol−随机：百分点 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 48.0% | 42.0% | 46.0% | 70.0% | +24.0 [+10.0, +38.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 16.0% | 20.0% | 18.0% | 22.0% | +4.0 [+0.0, +10.0] |
| ALFWorld THOR | Qwen3.5-4B | 14.0% | 10.0% | 8.0% | 18.0% | +10.0 [+0.0, +20.0] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 10.0% | 8.0% | 8.0% | 14.0% | +6.0 [+0.0, +14.0] |
| WebShop | Qwen3.5-4B | 18.0% | 22.0% | 22.0% | 28.0% | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | 28.0% | 28.0% | 28.0% | 30.0% | +2.0 [-6.0, +10.0] |

表中的区间针对配对成功率差值，不是单个成功率的区间。`gpt5` 是代码中的“五候选 GPT 选择”策略名；本轮模型为 `gpt-6.1-sol`。

## WebShop 原生平均 reward

| Actor | Greedy | 单次采样 | 五候选随机 | Sol 五选一 | Sol−随机 [95% CI] |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 0.4163 | 0.5577 | 0.5303 | 0.5893 | +0.0590 [-0.0177, +0.1420] |
| Qwen3-VL-4B-Instruct | 0.6156 | 0.6026 | 0.6146 | 0.6542 | +0.0397 [-0.0163, +0.0993] |

## 步数、无效动作与截断

无效动作率和选中动作截断率按执行步数统计；候选截断率按全部候选数量统计；达上限比例按 episode 统计。

| 环境 | Actor | 策略 | 平均步数 | 无效动作 | 候选截断 | 选中动作截断 | 达步数上限 |
|---|---|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | gpt5 | 24.94 | 27.8% | 22.3% | 10.8% | 30.0% |
| ALFWorld TextWorld | Qwen3.5-4B | greedy | 30.30 | 60.1% | 51.7% | 51.7% | 52.0% |
| ALFWorld TextWorld | Qwen3.5-4B | random5 | 32.12 | 61.1% | 36.4% | 35.7% | 54.0% |
| ALFWorld TextWorld | Qwen3.5-4B | sample | 34.86 | 59.9% | 40.1% | 40.1% | 58.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | gpt5 | 42.12 | 41.3% | 0.0% | 0.0% | 78.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | greedy | 43.98 | 50.6% | 0.0% | 0.0% | 84.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | random5 | 44.00 | 48.5% | 0.0% | 0.0% | 82.0% |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | sample | 43.12 | 47.8% | 0.0% | 0.0% | 80.0% |
| ALFWorld THOR | Qwen3.5-4B | gpt5 | 41.98 | 65.1% | 57.6% | 36.6% | 82.0% |
| ALFWorld THOR | Qwen3.5-4B | greedy | 43.72 | 89.2% | 80.6% | 80.6% | 86.0% |
| ALFWorld THOR | Qwen3.5-4B | random5 | 46.36 | 86.2% | 74.2% | 74.2% | 92.0% |
| ALFWorld THOR | Qwen3.5-4B | sample | 45.48 | 89.5% | 72.9% | 72.9% | 90.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | gpt5 | 45.38 | 15.8% | 0.0% | 0.0% | 86.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | greedy | 46.30 | 12.9% | 0.0% | 0.0% | 90.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | random5 | 47.16 | 15.2% | 0.0% | 0.0% | 92.0% |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | sample | 48.04 | 16.1% | 0.0% | 0.0% | 92.0% |
| WebShop | Qwen3.5-4B | gpt5 | 16.00 | 9.0% | 20.4% | 8.0% | 10.0% |
| WebShop | Qwen3.5-4B | greedy | 40.44 | 68.6% | 68.4% | 68.4% | 36.0% |
| WebShop | Qwen3.5-4B | random5 | 26.68 | 55.7% | 51.5% | 51.4% | 20.0% |
| WebShop | Qwen3.5-4B | sample | 22.60 | 40.1% | 39.9% | 39.9% | 16.0% |
| WebShop | Qwen3-VL-4B-Instruct | gpt5 | 8.30 | 0.2% | 0.0% | 0.0% | 2.0% |
| WebShop | Qwen3-VL-4B-Instruct | greedy | 7.26 | 0.6% | 0.0% | 0.0% | 4.0% |
| WebShop | Qwen3-VL-4B-Instruct | random5 | 8.42 | 0.2% | 0.0% | 0.0% | 4.0% |
| WebShop | Qwen3-VL-4B-Instruct | sample | 5.82 | 1.7% | 0.0% | 0.0% | 2.0% |

## 候选池诊断

下面仅统计 Sol 所访问的状态。不同策略会访问不同状态，因此这些指标是描述性证据，不是因果分解。

| 环境 | Actor | 每池平均不同动作数 | 五个动作相同的池比例 | 五个候选全截断的池比例 | 解析 fallback 次数 |
|---|---|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 2.364 | 43.0% | 8.3% | 0 |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 1.292 | 74.6% | 0.0% | 0 |
| ALFWorld THOR | Qwen3.5-4B | 3.687 | 13.9% | 31.4% | 0 |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 1.075 | 92.9% | 0.0% | 0 |
| WebShop | Qwen3.5-4B | 2.328 | 39.4% | 4.5% | 0 |
| WebShop | Qwen3-VL-4B-Instruct | 1.400 | 71.8% | 0.0% | 0 |

## 95% CI 如何计算

按 task_id 配对：dᵢ = Sol 在任务 i 的成功值 − 基线在同一任务的成功值。成功值为 0/1；reward 分析使用原生连续 reward。

从 50 个配对任务中有放回抽取 50 个，计算平均差值，重复 10,000 次；取 2.5% 和 97.5% 分位数。统计重采样 RNG 固定为 42，独立于“运行了几个实验 seed”这一问题。

本轮只有运行 seed 42：区间反映任务重采样的不确定性，不估计跨运行 seed 的方差。任务有多个匹配 seed 时，当前实现先在任务内平均，再重采样任务。区间未做多重比较校正。

ALFWorld 中存在同组相关任务，额外按整个 group_id 重采样，保留所有组内任务与任务加权均值。下面是补充分析，不替换主分析。

| 环境 | Actor | 任务 / 组 | 主任务 bootstrap：百分点 | 补充组 bootstrap：百分点 |
|---|---|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 50 / 41 | +24.0 [+10.0, +38.0] | +24.0 [+11.5, +37.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 50 / 41 | +4.0 [+0.0, +10.0] | +4.0 [+0.0, +9.8] |
| ALFWorld THOR | Qwen3.5-4B | 50 / 42 | +10.0 [+0.0, +20.0] | +10.0 [+0.0, +20.4] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 50 / 42 | +6.0 [+0.0, +14.0] | +6.0 [+0.0, +13.5] |
| WebShop | Qwen3.5-4B | 50 / 50 | +6.0 [-4.0, +16.0] | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | 50 / 50 | +2.0 [-6.0, +10.0] | +2.0 [-6.0, +12.0] |

## 固定协议与原生 benchmark

| 项目 | 设置 |
|---|---|
| Selector | `gpt-6.1-sol`，输出上限 2048 tokens |
| 候选采样 | 每步 5 次独立 seeded draw；T=0.7，top_p=0.9，top_k=-1 |
| Actor 输出 / 上下文 | 1024 / 262144 tokens；enable_thinking=True |
| 选择上下文 | 原公开 builder；最近 5 步历史；完整候选 thought/action |
| ALFWorld TextWorld | 原生文本，无占位图片；50 步上限 |
| ALFWorld THOR | 原生 RGB 与文本反馈，官方 mrcnn_astar 控制器；50 步上限 |
| WebShop | 原生可见文本与全部当前可见商品图片，搜索结果最多 10 张；原生动作/评分；100 步上限 |
| 数据隔离 | 原生训练集内按任务/商品组预留 10%；固定 pilot 清单；正式 held-out/test 未使用 |
| 候选处理 | 保留重复与截断候选；不改写、不筛选、不补采样 |
| 版本 | `upstream-public-native-v3` / `v3-sol-nativecmd` / `native-alfworld-move-v1` |

THOR 是固定官方控制器上的高层动作选择实验；原生动作、Qwen actor 和跨 benchmark 观察适配属于明确记录的方法扩展。

## 成本与核验

- 本轮 native 执行窗口：Sol 返回 usage 估算 **$56.80**，7,294 次调用；包括 smoke 和中断尝试。
- 完整 episode 归属费用：$52.19，已包含 smoke；与上一项不能相加。
- 全部已记录 Sol 调用：$57.76，含此前诊断。
- 已追踪 GPU：46.32 H200 小时，按 $0.90/小时估算 $41.69；包含前期部分诊断，范围宽于本轮。
- 以上是 usage/资源估算，未与账单对账；未返回 usage 的潜在费用无法据此确认。

核验：1200 条汇总与 1200 条记录轨迹一致；6 组通过审计；当前基础设施错误 0；历史中断 2 次。
原始轨迹复算验证的是聚合一致性，不是独立重放环境。中断尝试保留但不作为任务失败，恢复前已完成结果的哈希保持不变。

## 数据来源

冻结配置 hash：`78fc66d9a0ce60f25706e5f9537f6e184ae93117efd7cff0f1d02c53988ed7ee`。完整来源文件 hash 保存在随附结构化数据中。

- [Piotr 原方法](https://github.com/piotr-teterwak/action-reward-models/tree/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce)
- [zixian ARM 集成](https://github.com/zixianma/OpenWebRL/tree/134f064d867170eb833f6ab0f668a1b05050c031)
- 原私有 `frontier_arbiter` 未发布；当前对齐公开 builder，不能声称复原了历史私有数据生成 prompt。
- 已发布骨干为 qwen3_vl；发布数据与公开 inference builder 的 prompt 存在差异，新的标注/训练/推理链路统一采用公开版本。

## 解释与下一步

[查看持续维护的决策记录](../DECISIONS.md)。数值表由已审计 JSON 生成；分析判断与数值来源分开维护。
