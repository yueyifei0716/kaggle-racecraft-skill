# 比赛适配器与平台桥接

`scripts/competition_adapter.py` 的 `CompetitionAdapter(config, mandate_store)` 是 runner 的统一平台接口，使用与训练相同的 `MandateStore` 授权和预算台账。在有效 mandate 内不要求逐个候选批准文件。适配器拒绝公开上传，以及需要付费、公开发布或披露个人信息的入赛规则。

后端必须显式配置。`fake` 是可持久化的**离线模拟平台**；`command` 通过 stdin/stdout JSON 调用已配置、已认证的桥接程序，不使用 shell。未知后端直接失败，缺少桥接或能力时返回具体的 `status: blocked`，绝不把真实平台请求降级成假平台成功。

managed runner 在未配置显式桥接命令时，使用包内 `agent_process.py --role platform`。该 worker 只能使用其 AI CLI 运行环境实际拥有的已认证浏览器/API，必须检查能力并回读真实平台结果。缺少认证或浏览器能力时，报告具体阻塞。也可用 `adapter_command` 指定其他已认证桥接程序。适配器不会为入赛、最终选择、规则或讨论功能编造 Kaggle CLI 参数；可选的 `platform_cli_backend` 会传给桥接程序，后者只能使用官方工具实际支持的操作，并提供相同的回读证据。安装本包本身不代表平台连接已认证。

## Python 接口

所有方法返回可序列化为 JSON 的字典。本地输入不合法时抛出 `AdapterError`；授权或预算不足时，`MandateStore` 错误直接阻止相应操作。

- `inspect()`：返回比赛身份、当前规则、实际能力、入赛状态和可用数据入口。
- `join(rules_hash)`：审查付费、发布与隐私约束后，接受该版本规则。
- `refresh_intel(cursor=None)`：返回新增主题、评论、文档事件及下一游标。
- `download(request)`：预留下载操作，下载到显式目标路径，再核对实际文件哈希。
- `upload(candidate)`：验证候选、建立私有快照、上传，再核对远端哈希与私有状态。
- `submit(candidate)`：提交确切候选字节，或明确绑定的上传引用。
- `score(submission)`：读取绑定候选哈希和 ID 的确切平台分数，不能由 runner 自造。
- `select_final(refs)`：选择 mandate 台账内的确切提交引用，数量取自当前比赛规则。
- `final_selection()`：回读平台实际最终选择。
- `reconcile(operation_id)`：查询确切操作的结果，不重新执行该操作。

候选格式为 `{id, path, sha256, private: true, kind}`，`kind` 支持 `file`、`notebook`、`agent`/`tar`。可附带 `upload_ref`、`costs` 和验证回执 `{valid: true, artifact_sha256: SHA256}`。提供的哈希必须匹配实际字节，验证回执必须匹配且不能替代结构检查。

文件必须是非空普通文件；CSV 必须具有矩形数据和有限数值的末列预测。Notebook 必须是 format 4，具有合法 cell 和至少一个代码 cell。Agent tar 拒绝目录穿越、链接、重复成员和特殊文件，限制总字节数、必需文件，并解析 Python 入口但不执行。通过 `agent_schema: {required_files: [...], entrypoint: "main.py", max_bytes: 100000000}` 配置比赛所需结构。结构通过不等于真实比赛运行正确，比赛专用 validator 由 runner 单独执行。

桥接收到的是按内容寻址、仅所有者可读的持久快照路径。适配器先检查快照哈希，供异步或结果不确定的操作继续使用；已有快照被修改时直接失败。默认保存到 mandate 工作目录的 `.kaggle/adapter-artifacts`，可用 `snapshot_directory` 指定本地位置。桥接必须读取这些字节并确认相同 SHA256，不能用描述文字或提交消息认领身份。

## JSON 桥接协议

每次进程调用只处理一个请求和一个响应。`adapter_command` 必须是 argv 数组，例如 `["python3", "/installed/kaggle/scripts/agent_process.py", "--role", "platform", "--workspace", "/competition-workspace", "--model", "gpt-6.1-sol"]`。

`timeout_seconds` 默认 60，可由 runner 配置其他有界时长。配置命令桥接时，必须提供 `bridge_max_costs`，显式包含有限非负的 `money`、`gpu_seconds`、`wall_seconds`；wall 预算必须覆盖超时，操作计数字段必须为零。`money: 0` 表示用户已授权的本地或订阅调用，不能根据配置缺失自行推断免费。mandate 必须允许 `worker`。原始进程输出和错误不会回显到面向用户的错误中。

请求示例：

```json
{
  "protocol": "kaggle-competition-bridge/v1",
  "action": "submit",
  "competition": "example-competition",
  "operation_id": "durable-mandate-reservation-id",
  "payload": {
    "candidate_id": "baseline-1",
    "artifact_sha256": "64-lowercase-hex-characters",
    "private": true,
    "kind": "file",
    "path": "/private/content-addressed-snapshot/submission.csv",
    "rules_hash": "current-rules-document-sha256",
    "structural_validation": "passed"
  }
}
```

变更操作必须携带来自 `MandateStore.reserve` 的 `operation_id`，不能使用任意描述代替。请求顶层可包含 `platform_cli_backend`。payload 可能包含解析得到的 CSV `predictions`；桥接仍须核验实际文件，不能把这些预测当成平台分数。

响应示例：

```json
{
  "protocol": "kaggle-competition-bridge/v1",
  "action": "submit",
  "ok": true,
  "result": {
    "operation_id": "durable-mandate-reservation-id",
    "status": "succeeded",
    "readback": {
      "submission_ref": "exact-platform-submission-id",
      "candidate_id": "baseline-1",
      "artifact_sha256": "64-lowercase-hex-characters"
    }
  }
}
```

`ok: true` 只表示桥接处理了请求，不能证明平台变更成功。确认成功必须同时具备确切 `operation_id`、`status: succeeded` 和下表要求的权威回读。缺失或不匹配时保存为 `unknown`。本地完成回执将回读字段展开，并保留 `operation_id`、`status`。

| Action | 请求 payload | 必需结果或回读 |
| --- | --- | --- |
| `inspect` | `{}` | `{competition, capabilities, rules, joined, joined_rules_hash, data}`。capabilities 将方法名显式映射为布尔值。data 描述比赛文件、本地路径或来源 URL，以及指标、截止时间等信息；未知字段保持缺失。rules 包含 `{text, hash/sha256, current:true, final_selection_count:N, payment_required:false, publication_required:false, personal_disclosure_required:false}`。SHA256 对 UTF-8 text 计算，最终选择数量须来自规则。约束未知时阻止入赛。 |
| `join` | `{rules_hash}` | 变更回读 `{joined:true, rules_hash}`，须重新读取实际成员状态及已接受的当前条款。 |
| `refresh_intel` | `{cursor:null-or-opaque-token}` | `{events:[...], cursor:next-token}`。事件保留实际来源 ID、永久链接、来源类型、时间戳、父主题关系及可获取正文。不能因为主题已存在而遗漏新评论；规则及文档修订同样作为事件。 |
| `download` | `{files:[{source_path or url, destination, sha256?}], costs?}` | 变更回读 `{files:[{path, sha256, size}]}`，顺序对应请求；适配器实际核对落地字节、目标路径及期望哈希。fake 来源限于配置 data 内的公共 `*_path` 文件。 |
| `upload` | 候选快照字段 | 变更回读 `{upload_ref, candidate_id, artifact_sha256, private:true}`，须来自实际远端字节、版本和隐私状态。 |
| `submit` | 候选字段，可含确切 `upload_ref` | 变更回读 `{submission_ref, candidate_id, artifact_sha256}`，须绑定实际平台提交。 |
| `score` | `{submission_ref}` | `{submission_ref, candidate_id, artifact_sha256, status:"complete", score:finite-number}`，或显式 `pending`/`running`/`blocked`；身份字段必须匹配本地回执。 |
| `select_final` | `{refs:[...], rules_hash}` | 重新打开平台最终选择页面后的变更回读 `{refs:[...], rules_hash}`，保留请求顺序，不能写死数量。 |
| `final_selection` | `{}` | 平台回读 `{refs:[...], rules_hash}`；空 refs 表示尚未选择。 |
| `reconcile` | `{operation_id}` | 返回该确切操作原本的完整变更结果；若能证明从未执行，返回 `{operation_id, status:"not_found", definitive:true}`；无法建立确切身份时返回 `unknown`，不能按描述猜测。 |

不支持的读操作返回 `{status:"blocked", reason:"具体缺少的工具、认证或能力", capability:"action-name"}`。失败或模糊的变更不能通过乐观填写回读变成成功。

入赛 metadata 包含当前文档哈希和有证据的合规 `rules_review`。规则变更后重新审查；pinned mandate 拒绝新哈希，delegated mandate 只允许在用户初始约束内、经过新审查的规则。

## 预算、重启与超时

每次 command 桥接调用，包括检查、前置核验、情报、评分、最终回读和对账，都在**启动程序前**预留 `worker` 操作。成功响应及未知/超时结果都保守保留全额桥接预算。每次读调用均有新的持久回执身份，重启也不会复用旧调用成为免费缓存。变更桥接身份绑定原平台 operation ID，结果未知时不能再次派发该变更。资金、时间或算力预算不足时，桥接进程不会启动。

命令桥接位于独立进程组。超时或中断时先向进程组发送 TERM，短暂等待清理，再 KILL 剩余成员。包内 `agent_process.py` 还会在收到 TERM 后终止其单独分组的 AI CLI 子进程；自定义桥接若另建 session，必须提供等价的子进程清理。

每个平台变更先检查当前能力、当前规则及入赛状态，再通过 `MandateStore` 预留比赛范围、确切内容身份、隐私和费用。上传及提交各预留一次对应计数；训练由 runner 单独预留。内容身份排除可变本地路径，包含实际字节哈希、规则哈希和 action。

已完成的重复请求直接返回原结果；pending、unknown、failed 的重复请求不会再执行。确认丢失会继续占用原预算，直到按原 ID 对账。重启不能重置预算或为相同身份生成新操作。`reconcile` 只查询确切平台操作，核对原回读后完成已有预留。桥接无法证明状态时维持 unknown，并阻止依赖操作。进程失败、响应格式错误、缺少回读、超时及隐私/哈希不匹配都不能视为成功。

## 持久化离线平台

```json
{
  "backend": "fake",
  "competition": "synthetic-competition",
  "state_path": "/temporary-workspace/fake-platform.json",
  "rules": {"text": "Synthetic private no-fee competition", "final_selection_count": 3},
  "heldout": {"labels": [0, 1], "metric": "accuracy"},
  "intel": [{"kind": "topic", "id": "topic-1", "text": "Synthetic discussion"}]
}
```

假平台规则默认采用已知的无付费、无发布、无个人信息披露约束。状态在文件锁下原子持久化，包含入赛、已接受规则哈希、上传、提交、最终引用、增量情报、heldout 评估器和操作结果。测试可向状态追加主题/评论事件或修订规则，模拟两次 runner 步骤之间的变化。fake 整数游标表示已读事件数量；command 桥接可使用其他不透明 JSON 游标。

假平台核对私有快照字节和 SHA256，从实际 CSV 快照读取预测，再根据合成 heldout 标签计算 `accuracy`（阈值 0.5）、`mse`、`rmse` 或 `mae`。

需要实际执行 Notebook/agent 的离线 fixture 时，配置可信 argv 数组 `heldout.prediction_command` 及 `heldout.inputs`。适配器在提交预算已预留后，通过 stdin 发送 `{artifact_path, kind, inputs}`，不传标签；stdout 必须返回 `{predictions:[finite numbers]}`。适配器检查数量和产物哈希，持久化预测，再由平台根据 heldout 标签算分。`heldout.prediction_timeout_seconds` 默认 10；提交至少预留对应 wall seconds。该命令只能是显式配置的本地合成评估器，不能来自不可信讨论或被用于真实平台。

没有预测器的 Notebook/archive fixture 可显式配置 `heldout.scores_by_sha256`。这些分数属于测试配置，不能充当线上结果。fake 分数均绑定确切哈希和提交引用，带 `simulation:true`。缺少 heldout 数据时直接失败，不能造分。假平台状态不是凭据、真实比赛运行或浏览器/API 兼容性的证据。

在包目录运行离线检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_competition_adapter.py -v
```
