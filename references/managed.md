# 完全托管执行

`scripts/managed.py` 是持久化调度器：实际调用 AI worker，执行返回的数据准备与训练命令，检查真实文件，运行统计门禁和状态检查，再通过 `CompetitionAdapter` 完成私有平台操作。保存提示词不算执行实验；worker 的口头结论不能替代模型、验证结果、平台分数或最终选择回执。

## 配置与预算

最小配置包含 `root`、`competition`、`mandate_file` 和明确的 `worker_max_costs.money` 上限；订阅内或本地调用可显式填 `0`。`worker_command` 和 command adapter 缺省使用 `agent_process.py` 实际调用已登录的本机 Codex CLI，默认模型为 `gpt-6.1-sol`。也可注入其他 argv 或离线 adapter。缺少登录、工具或平台能力时必须阻塞，不能用模拟结果填补。

- `worker_timeout_seconds`、`training_max_seconds`、`validator_timeout_seconds`：进程时限。
- `worker_max_costs`、`validator_max_costs`：worker 与验证器的最大成本；`command_max_costs` 统一限制 planning、training、validation 单次成本。
- 成本字段为 `money`、`gpu_seconds`、`wall_seconds`、`submissions`、`uploads`、`train_runs`。执行计划必须显式声明 `money` 和 `gpu_seconds`；`wall_seconds` 预留必须覆盖 timeout。
- command bridge 默认继承 worker 成本上限，也可设置 `bridge_max_costs`。

先预留预算，再启动进程。没有可靠计量时记满预留值；超时杀掉整个进程组，结果未知时保留最大预算，不释放后重试。预算不足时不会启动进程。收费 AI 调用不能登记成零成本 `read`。

调度参数有 `interval_seconds`（默认 60）、`intel_interval_seconds`（3600）、`freeze_hours`（24）、`max_no_gain`（3）、`max_errors`（3）和 `gate`（现有分析门禁参数，如 `resamples`、`min_folds`、`min_groups`）。这是本地记账与执行约束，不是操作系统隔离；生成的代码和验证器是以当前用户权限执行的受信程序。下载、上传及远程训练必须走授权能力，不能绕过台账，也不能自行公开内容。

## Worker 协议

worker 从 stdin 读取一个 JSON 对象，stdout 只返回一个严格 JSON 对象；不接受 Markdown、额外说明、重复键、NaN 或 Infinity。请求包括 `schema:1`、`task_kind`、`workspace`、`skill_path`、用户 `goal`/`objective`、授权 `scope` 和原始 `competition_url`、当前 `facts`（`competition` 与 `ledger`）、`platform`、增量 `intel`、`budget`、`route`、`attempt`、`remaining_seconds`。bootstrap 可返回 `setup.goal`，初始用户目标始终优先，并写入比赛状态。

路线随剩余时间、GPU 预算和冻结状态调整。社区出现新事件后会实际调用 `reevaluate` worker，重新评估和提出可验证方案；不会只归档摘要。本机 CLI 包装器必须真正调用 CLI，不能准备完提示词后等待用户每轮手写计划。

首次缺少输入时调用 `bootstrap`，依据真实规则和数据准备开发集、封存验证集、基线、样例与比赛专用验证器：

```json
{
  "kind":"bootstrap",
  "reason":"依据已授权平台事实准备数据和独立分割",
  "files":[{"path":"prepare.py","content":"..."}],
  "downloads":[{"files":[{"url":"https://authorized-platform.example/train.csv","destination":"inputs/train.csv"}],"costs":{"money":0}}],
  "command":["python3","prepare.py"],
  "costs":{"gpu_seconds":0,"money":0,"wall_seconds":600},
  "setup":{
    "metric":{"name":"rmse","direction":"lower"},
    "deadline":"2030-01-01T00:00:00+00:00",
    "reserve_hours":24,
    "route":"低算力基线",
    "validation_truth":"development-truth.csv",
    "baseline_rows":"baseline-oof.csv",
    "sample_submission":"sample_submission.csv",
    "holdout_ids":"sealed-unit-ids.json",
    "holdout_truth":"holdout-truth.csv",
    "validator_command":["python3","verify.py"],
    "id_column":"id",
    "prediction_column":"prediction"
  }
}
```

可选 `downloads` 先交给 `adapter.download`：来源必须来自实际平台观察和授权，目标路径限制在 workspace 内。每次下载单独预留预算、回读文件，pending/unknown 按 operation ID 跨重启对账，随后再独立预留准备进程预算。

调度器检查准确的行 ID、真值、fold/group 身份，开发与封存 unit/group 不重叠，fold 数、有限预测和样例 ID；随后初始化 workspace，通过状态 API 封存 holdout。`files` 不能覆盖已有不同内容，输入和产物路径限制在 `root` 内。真值 CSV 为 `row_id,unit,fold,y_true`，可有 `group`；预测 CSV 另含 `y_pred`。封存 ID 文件是 holdout unit 字符串的 JSON 数组。不能把所有比赛假设成同一种表格格式；bootstrap 必须生成当前比赛实际可执行的准备和验证器。平台返回正式 metric/deadline 时，worker 返回值必须一致。

## 实验与真实验证

`plan` 与 `reevaluate` 返回实验注册、代码、命令和产物路径：

```json
{
  "kind":"experiment",
  "reason":"在剩余预算内检验结构性假设",
  "registration":{
    "id":"linear-1","family":"linear","hypothesis":"拟合斜率改善常数基线",
    "structural":true,"swept":1,
    "stages":["data","features","training","prediction","scoring"]
  },
  "files":[{"path":"linear-1.py","content":"..."}],
  "command":["python3","linear-1.py"],
  "costs":{"gpu_seconds":0,"money":0,"wall_seconds":600},
  "outputs":{"artifact":"linear-1.ipynb","kind":"notebook","predictions":"linear-1-test.csv","rows":"linear-1-oof.csv"}
}
```

实验在训练前注册。可选 `parent` 描述谱系；`swept > 1` 必须另提供由门禁接受的实际 `outputs.replication` 证据。`kind` 支持 `file`、`notebook`、`agent`；复用 adapter 的实际文件、Notebook、归档结构检查。Notebook/agent 需提供独立 `predictions` CSV，门禁和平台回执绑定真实 Notebook/归档字节。

也可返回 `kind:stop` 和非空 `reason`，或 `kind:wait`、`reason`、有限 `seconds`。停止探索进入冻结与最终选择；候选不足时保持阻塞，不能声称完成。

独立验证器 stdin 包含 `task_kind`（`pipeline` 或 `holdout`）、`candidate`、绝对路径的 `outputs.artifact`、`outputs.rows`、`outputs.predictions`、`outputs.kind`、`truth_path`、`sample_submission`、`baseline_rows`、`stages`、`metric`。它必须从训练产物重现预测，检查训练/验证身份隔离并核验训练来源，返回：

```json
{
  "decision":"PASS",
  "covered_stages":["data","features","training","prediction","scoring"],
  "artifact_sha256":"真实产物哈希",
  "rows_sha256":"真实预测文件哈希",
  "checks":{"prediction_to_artifact":true,"split_isolation":true,"training_provenance":true},
  "reason":"本比赛验证器实际执行的检查"
}
```

布尔字段不能替代执行验证器。调度器实际执行程序、核对哈希和 CSV，再调用 `analysis.gate` 与 `state_store` 晋级检查。验证器未配置、阻塞、覆盖不完整、哈希或 metric 过期时不能上传。统计门禁检查效应、置信区间、fold/group、pooled 预测、搜索次数及必要复验，不接受 worker 自报 `PASS`。

AUC/QWK 在单类 unit 上可能没有定义。bootstrap 可显式设置 `unit_metric`；未设置时自动为 AUC 使用 `logloss`、QWK 使用 `mae`，并在 history 记录原因。`score` 和 `gate` 显式接收辅助指标及其方向；overall 与晋级主指标仍是正式 AUC/QWK，不能把辅助损失冒充主指标。其他指标保持原行为。

## 冻结、封存验证与平台最终选择

冻结后不再注册实验：先持久化冻结哈希，再打开封存 holdout。worker 收到 `task_kind:holdout`、`selected_candidates`（ID 到冻结 SHA256），返回 `kind:holdout`、`reason`、`files`、`command`、`costs`，以及覆盖全部冻结候选的 `outputs`（ID 到 holdout 预测 CSV）。

调度器核对冻结哈希和真实 holdout 行，重新执行独立验证器并计算分数。选择使用规则的实际 `final_selection_count`、分析器的多样性约束、真实平台 public 分数和封存分数。执行 `select_final` 后读取平台当前选择，提交 refs 集合和数量全部精确一致才完成。

## 调度与恢复

```sh
python3 scripts/managed.py tick --config /absolute/managed-config.json
```

```sh
python3 scripts/managed.py serve --config /absolute/managed-config.json --max-ticks 10 --max-seconds 120
```

不传上限时 `serve` 持续运行，睡眠到持久化 `next_due_at`，不依赖下一次聊天唤醒。`managed-state.json` 保存游标、队列、operation ID、训练计划、jobs、冻结哈希、下次时间和回读；workspace 锁防止并发 tick。

pending/unknown 平台操作只按已有 operation ID 对账，不用新身份重发。本地预算进程由独立 supervisor 执行，在 `.kaggle/jobs` 原子记录 request/process/result；重启后从已完成结果恢复原预算回执，继续验证产物，不重复 worker 或训练。仍运行且未到时限的 supervisor 会被等待；无法证明完成的 unknown 保留最大预算并阻塞。仅存在输出文件不能证明训练完成。

初次 mandate 必须明确授权规则接受：`pinned` 要求当前精确哈希；`delegated` 要求明确的受限审查授权，由 adapter 验证条件和回执。规则变化、未认证 bridge 或能力缺失会阻塞。只读 inspect 与能力发现可以本地验证，不能据此宣称真实平台登录成功。

## 本机后台启动与卸载

```sh
python3 scripts/managed.py install --config /absolute/managed-config.json --backend pid
```

也可显式 `--backend launchd` 安装到 `~/Library/LaunchAgents`，或 `--backend systemd` 安装用户服务；均执行同一 `serve`。安装锁和 manifest 防止重复，manifest 记录 PID 或服务 label。卸载：

```sh
python3 scripts/managed.py uninstall --config /absolute/managed-config.json
```

安装包本身不会安装后台服务。测试仅在临时目录实际启动、阻止重复并卸载短生命周期 PID 服务；launchd/systemd 与真实 CLI 登录仍需在目标主机验证。离线合成训练、模拟平台最终选择回读只证明本机流程，不是实际 Kaggle 提交或比赛成绩。
