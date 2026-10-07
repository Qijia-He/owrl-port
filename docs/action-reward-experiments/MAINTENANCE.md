# 如何持续更新

[实验首页](README.md) · [决策记录](DECISIONS.md)

GitHub 固定位置：[`Qijia-He/owrl-port` / `docs/action-reward-experiments/`](https://github.com/Qijia-He/owrl-port/tree/main/docs/action-reward-experiments)，分支 `main`。此目录保存实验记录、导出结果和便携渲染脚本；收集/训练/运行脚本在集群上的独立 `arm-generalization` 实验项目中维护。

实验结果先冻结并通过审计，再登记进日志。文档更新不会启动模型、消耗 API 或提交 Slurm 作业。

## 文件组织

- `README.md`：自动生成的入口，展示最新已审计结果和实验历史。
- `runs/<实验ID>.md`：每个实验版本的完整 Markdown 表格与说明。
- `data/<实验ID>.json`：便于分享和重新渲染的结构化结果，以及原始冻结文件 SHA-256；不包含凭据、原始图片、机器路径或 API 请求内容。
- `DECISIONS.md`：人工维护的结论、用户决定、问题和下一步；重新生成表格不会覆盖此文件。
- 项目根目录的 `scripts/update_experiment_log.py` 负责校验和生成；发布到 GitHub 的副本保留在日志目录的 `tools/` 中。

## 当前实验的更新入口

在集群的 `arm-generalization` 实验项目根目录执行（不是 `owrl-port` 仓库根目录）：

```bash
source scripts/runtime.sh
"$ARM_PYTHON" scripts/update_experiment_log.py \
  --register reports/pilot-seed42-artifacts.json \
  --id v3-sol-nativecmd-seed42
```

登记时会检查冻结文件哈希、配置一致性、完整 task/seed 覆盖、最终审计状态，以及补充区间是否齐全。同一实验 ID 的结果不同会拒绝覆盖；新结果须使用新 ID。

只重新渲染现有导出数据，无需模型、集群、API 或第三方 Python 依赖：

```bash
python3 scripts/update_experiment_log.py
```

如果只下载了 GitHub 上的日志目录，在该目录下执行：

```bash
python3 tools/update_experiment_log.py --docs-root .
```

`scripts/freeze_seed42_report.py` 完成冻结后会调用上述登记入口。后续实验在生成相同结构的冻结 receipt 后，可用新 ID 登记；若实验设计改变，需同步更新该版本的协议说明，不能混入旧版本。

## 后续工作时的维护约定

1. 完成一轮已授权实验并审计后，登记对应冻结结果，生成该版本 Markdown 和首页。
2. 更新 `DECISIONS.md`：结果判断、失败原因、参数变动和用户决定；不要用新的判断改写旧数值。
3. 检查文档链接、导出数据与表格一致性，然后将本地 `docs/experiment-log/` 同步至 `Qijia-He/owrl-port` 的 `docs/action-reward-experiments/`。目录映射记录在实验项目的 `configs/experiment-log-publish.json`；提交前读取最新 `main`，使用 fast-forward/expected-head 检查保留并发修改。
4. 实验开始、暂停或范围改变时，先更新 `DECISIONS.md` 的当前状态；未完成结果不放进“已审计”结果表。

GitHub 同步由后续工作中的提交完成。当前生成脚本不运行定时后台上传；持续记录的约定不授权启动额外 seed、ARM 训练或正式测试。
