# Action-level Rewards 实验日志

目标：冻结 Qwen，由 Sol 在五个候选动作中选择，评估原生 benchmark 闭环成功率；随后再决定 ARM 数据收集与训练。

[协议与复现边界](runs/v3-sol-nativecmd-seed42.md#固定协议与原生-benchmark) · [决策、问题与下一步](DECISIONS.md) · [如何更新](MAINTENANCE.md)

最近完成：**v3-sol-nativecmd-seed42**，2026-10-06T09:40:38.962824+00:00。最新实验包含 seeds `[42]`。当前执行范围与待办见决策记录。

## 最新结果

| 环境 | Actor | Greedy | 单次采样 | 五候选随机 | Sol 五选一 | Sol−随机：百分点 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| ALFWorld TextWorld | Qwen3.5-4B | 48.0% | 42.0% | 46.0% | 70.0% | +24.0 [+10.0, +38.0] |
| ALFWorld TextWorld | Qwen3-4B-Instruct-2507 | 16.0% | 20.0% | 18.0% | 22.0% | +4.0 [+0.0, +10.0] |
| ALFWorld THOR | Qwen3.5-4B | 14.0% | 10.0% | 8.0% | 18.0% | +10.0 [+0.0, +20.0] |
| ALFWorld THOR | Qwen3-VL-4B-Instruct | 10.0% | 8.0% | 8.0% | 14.0% | +6.0 [+0.0, +14.0] |
| WebShop | Qwen3.5-4B | 18.0% | 22.0% | 22.0% | 28.0% | +6.0 [-4.0, +16.0] |
| WebShop | Qwen3-VL-4B-Instruct | 28.0% | 28.0% | 28.0% | 30.0% | +2.0 [-6.0, +10.0] |

[完整实验记录：reward、步数、无效动作、截断、候选重复、CI 与费用](runs/v3-sol-nativecmd-seed42.md)

所有 Qwen actor 为 4B。表中为训练集预留 pilot；单 seed 结果不能代替正式 held-out/test 评测。

## 实验历史

| 完成日期（UTC） | 实验版本 | Seeds | 完整 episodes | 状态 |
|---|---|---|---:|---|
| 2026-10-06 | [v3-sol-nativecmd-seed42](runs/v3-sol-nativecmd-seed42.md) | [42] | 1200 | 已审计 |

新实验以新 ID 追加；旧结果保留。数值来自已冻结、已审计的导出数据，后续重新渲染不会运行模型或提交实验。
