# 数据划分与推理保护

`scripts/data_tools.py` 只生成比赛数据的划分文件，不训练模型。输出使用新文件模式，拒绝覆盖已有对照。输入必须有唯一 ID。

```bash
python3 scripts/data_tools.py folds train.csv train_folds.csv --id sample_id --strategy stratified-group --group patient_id --target label --splits 5
```

支持 random、stratified、group、stratified-group、leave-group-out 和 time。leave-group-out 按指定 group 列每组留出一次，折数由组数决定；用于未见过的机构、地区或其他整组迁移。回归可在分层方式下加 `--regression-bins`；相同目标值归同一层。分组分层是平衡启发式，必须查看实际各折标签和组别分布。

```bash
python3 scripts/data_tools.py folds train.csv time_folds.csv --id sample_id --strategy time --time timestamp --splits 4 --gap 2 --plan time_plan.json
```

时间值须统一为数值或可排序的标准时间表示。time 将最早一块作为初始训练历史，fold=-1，其余块逐步作为验证；gap 以不同时间值的数量计算。**训练必须按 plan 的 train_ids 取数，不能使用 fold != k。** 后者会把未来数据带进训练。

```bash
python3 scripts/data_tools.py holdout train.csv development.csv sealed.csv --id sample_id --group patient_id --fraction 0.15
```

这会按独立组封存保留数据，输出摘要不展示标签。还需通过 workspace 的保留集登记操作登记该文件哈希，开发训练只使用 development 文件。普通文件封存是流程约束，不是阻止 Agent 读取文件的安全沙箱；跨会话也必须遵守未打开约定。样本少时评估损失与代表性，再决定是否封存。

## 隐藏重跑的可复用代码

`assets/inference_support.py` 可放进自己的 Notebook 或作为模块导入。它提供：

- `verify_artifacts(root, entries)`：核对相对路径和 SHA-256，不接受缺失或越界的模型。
- `RunBudget(seconds, reserve)`：为最终写出保留时间，只在剩余预算充足时运行可选增强。
- `predict_items(...)`：逐项记录耗时；只有明确指定异常和 fallback 才允许局部降级，模型加载错误应在调用前失败。
- `atomic_csv(...)`：核验列和有限数值后原子写出，失败时保留原文件。

调用前先对比赛数据身份、模型、方向、标签映射和运行时 sample 做校验。输出需按运行时 ID 对齐，随后运行比赛专用的结构检查；这里的工具不自动推断目标列或伪造缺失预测。

在可见运行中强制执行正式评分分支，测试极少/大量输入、乱序 ID、不同尺寸、空对象和无法联网的环境。分支不能因为只在隐藏重跑出现就省略验证。存储报告时使用序号或脱敏标识；不要公开病人路径和原始诊断日志。
