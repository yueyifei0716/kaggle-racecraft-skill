# 一次授权与预算

`scripts/mandate.py` 只执行本地校验和记账，不访问网络、凭据、Kaggle 或付费服务。示例 `assets/mandate.example.json` 的 `user_confirmed:false` 刻意使它不可执行。Runner 不得替用户确认、提高预算、延长到期时间或扩大范围。

用户可在比赛开始时一次明确授权比赛内的训练、私有上传、提交和最终选择；授权范围和预算内不再逐候选版本请求批准。公开 Notebook、发文、外部消息和个人信息披露仍被禁止。授权不替代提交验证、竞争规则检查或最终选择的其他门禁。

## 授权文件

必填字段：

| 字段 | 要求 |
| --- | --- |
| `schema_version` | 整数 `1` |
| `user_confirmed` | 用户实际确认后才可写 `true`；不接受缺失、字符串或数字 |
| `scope.competition` | 单个比赛 slug |
| `scope.url` | 严格等于 `https://www.kaggle.com/competitions/<slug>` |
| `expires_at` | 含时区的 ISO8601；到达该时刻即禁止新预留 |
| `allowed_operations` | `read/worker/join/download/train/validate/upload/submit/select_final` 的非空子集 |
| `budgets` | 必须包含 `money/gpu_seconds/wall_seconds/submissions/uploads/train_runs` |
| `max_daily_submissions` | 非负整数，按 UTC 自然日记账 |
| `rules_acceptance` | 下述两种模式之一，由用户初次决定 |
| `privacy` | `private_only/no_publication/no_personal_disclosure` 均须为 `true` |

`status` 可省略，默认 `active`，也可为 `paused/revoked`。所有预算须非负且有限，后三项计数必须是整数；布尔值、NaN、Infinity、未知成本维度被拒绝。`money` 的单位由任务预算统一约定，模块不进行汇率换算。单次 `submit/upload/train` 必须分别预留恰好一个 `submissions/uploads/train_runs`。

条款授权模式必须显式填写：

- `pinned`：`{"mode":"pinned","allowed":true,"document_sha256":"<64位小写SHA256>"}`。该 hash 仅在向用户展示具体条款版本并获得授权后写入；`join` 提供的当前 hash 不一致时拒绝。`allowed:false` 表示禁止代接受条款。
- `delegated`：`{"mode":"delegated","allowed":true,"constraints":{"no_payment":true,"no_publication":true,"no_personal_disclosure":true}}`。用户事先委托 Agent 读取并评估当前条款，无需先取得具体 hash 或逐版本确认。每次加入仍须记录当前 `rules_document_sha256` 和 `rules_review:{"complies":true,"evidence":...}`；evidence 必须是非空文字或对象。Runner 应在 evidence 中记录具体条款和约束判断。新增付款、公开发布、个人信息披露要求或无法判断的条款须阻塞；不得只填写一个没有检查依据的 `complies:true`。

模块核对明确的本地授权字段和审查收据，不代替条款阅读，也不声称能从 hash 推断条款内容。元数据已标记 `payment_required/publication_required/personal_disclosure_required`，或相应 `no_*` 为 false 的委托审查会被拒绝。除 join 外，pinned 模式下提供的任何当前规则 hash 也必须匹配。

## API

```python
store = MandateStore(root, mandate_file)
receipt = store.reserve(operation, competition, identity, costs, metadata=None, now=None)
store.complete(receipt["id"], result, actual_costs=None)
store.fail(receipt["id"], reason, unknown=True)
store.inspect()
store.pause(reason="paused by user")
store.revoke(reason="revoked by user")
```

错误统一为 `MandateError`；返回值均可 JSON 序列化。`costs` 可只提供用到的维度，其余为零。`worker` 用于收费 Agent/规划调用；`validate` 用于验证子进程，均可预留时间和费用。`read` 只允许零成本；收费读取须作为有预算的 worker 调用处理。身份可为非空字符串或 JSON 对象，对象键按序规范化，身份和规则版本/文件 hash 等执行含义应由调用方明确构造；描述放在 metadata，避免描述修改绕过去重。

`reserve` 必须成功持久化后才执行。收据字段包括 `id/operation/competition/identity/status/reserved_costs/metadata/created_at/day_utc`，并返回 `duplicate`。去重键由比赛、操作及规范身份组成，与 metadata 的描述无关。相同身份不同成本被拒绝；相同身份返回原收据，包括已有 result。`duplicate:true` 时绝不能重新执行。已完成时复用 result；pending/unknown 时只允许使用平台回执或本地证据核对，不能重试不确定的变更。

合法轮询要使用不同的时间窗口身份，例如 `{"query":"operation-123","poll_window":"2026-11-01T00:01Z"}`。同一窗口重复读取使用同一收据，下一窗口可继续读取，不消耗预算。

`complete` 可在到期、暂停或撤销之后记入已预留操作的回执，但不能执行新操作。相同 result 和 actual_costs 的重复完成幂等，冲突结果被拒绝。actual_costs 未填写的字段沿用原预留；完整完成后，可用实际测量减少费用和时间占额，但操作计数不退款。任一实际值超出预留，收据记为 `overspend`，状态锁止，后续新预留全部拒绝；不会自动增加授权额度。未知结果可在证据核对后 complete；明确 failed 不可转成成功。

所有失败和崩溃后的 pending/unknown 都保守保留全额，不自动退款或清理；明确失败同样保留占额，避免已消耗时间或平台额度被反复利用。平台核验确定不存在时，可用 `fail(..., unknown=False)` 将 unknown 关闭为 failed，仍不退款、不允许同身份重新执行。每日提交按预留的 UTC 日期计入，跨日不重置总预算。

## 持久化与权限边界

文件均位于 root 的 `.kaggle/`：`mandate-state.json`、`mandate-audit.jsonl`、`mandate.lock`。锁保护读改写，账本原子替换并 fsync，审计使用追加及 fsync。审计先写，账本后写：若崩溃发生在两者之间，可能有未入账审计事件；没有成功返回的预留不得执行外部动作。持久化失败不会释放既有 pending。

首次写账时固定授权内容摘要，之后修改预算、操作或到期时间都会触发 `mandate_changed`。摘要只是防止 Runner 在同一次本地运行中悄然改授权；它不是用户签名，不能证明有本地写权限的人是否真的得到用户同意。Runner 不得删除账本或另建根目录来绕过预算。人工撤销/暂停使用 API 或 CLI；没有自动 resume、提升额度或初始化授权为真的命令。新授权需要用户明确决定，并保留旧账本以防重复动作。

## CLI

所有命令需 `--root <比赛目录> --mandate <授权文件>`，之后为 `validate/status/pause/revoke/reserve/complete/fail`。`reserve` 接受 `--operation --competition --identity --costs [--metadata]`，后三者为 JSON 文本；`complete` 接受 `--id --result [--actual-costs]`，结果及成本为 JSON；`fail` 接受 `--id --reason [--known]`。默认失败不确定，`--known` 仅用于有明确拒绝证据时。无效输入返回非零退出码。

离线验证：

```sh
python -m unittest discover -s open-source/kaggle/tests -p test_mandate.py -v
```
