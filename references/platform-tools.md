# 平台操作与运行预算

这些工具仅使用 Python 标准库。`platform_ops.py` 是具体版本审批的单步 CLI 工具；托管模式另用 [比赛授权](mandate.md) 与 [平台桥接](competition-adapter.md)，自动加入、下载、上传、提交和最终选择，无需逐版本审批。安装所需工具和账号登录属于初始环境；接受规则可按一次授权委托给 Agent。工具不猜测凭据，也不伪造登录。

本文件定义的是本包的接口，不替代平台规则。工具按本机当前官方 CLI 的参数与源码能力核对，不假定不同安装版本行为一致。平台接口会变化，旧版缺失参数会明确失败。没有进行真实账号、真实网络、GPU 或提交验收。

## 权限边界

- `--help` 只展示参数。所有操作默认 dry-run，也接受显式 `--dry-run`；不启动 CLI、不联网、不创建包、回执或输出目录。文件提交的 dry-run 会读取并检查本地 CSV，包上传的 dry-run 会核对本地 manifest，以产出准确审批请求。
- 本地准备、下载、查询、MCP 调用、写验证回执和实际 benchmark 均须 `--execute`。只读网络仍可能消耗平台请求额度。
- 本文件的单步 push、dataset create/version 和比赛提交使用 `--approval APPROVAL.json`，须对应具体版本。用户明确选择一次授权托管时改走 `managed.py`/`CompetitionAdapter` 的 mandate 路径，不替未来每个产物生成虚假的人工批准。公开发布、发帖和额外隐私披露仍不在托管范围内。
- 所有上传包创建为私有；kernel 禁用互联网为默认值。工具不提供公开发布开关。dataset version 继承远端隐私，当前不自动查证既有数据集是否私有，审批请求会明确标为 `inherited-unverified`，不能把成功回执写成已核验私有。
- 原始 CLI stdout/stderr 不回显到日志；错误只给通用原因。外部情报和 MCP 响应保存到用户指定的新本地文件，摘要仅显示 hash。上传 manifest 只含相对文件名、字节数和 SHA-256，不含本机绝对路径。message 是审批内容的一部分，调用者仍应避免个人信息。

## 认证状态与平台情报

`credentials` 只用文件属性检查 `kaggle.json`、`access_token` 是否存在、是否普通文件、权限是否只允许所有者及归属是否为当前用户。只报告环境变量是否存在，绝不读取文件值、输出用户名/key/token 或探测账号认证。

显式 `credentials --fix-perms --execute` 可把当前用户所有的普通凭据文件改为 0600；通过不跟随 symlink 的文件描述符再次核对所有者和 inode 后修改，不读文件内容。默认不动权限，只有 `--fix-perms` 时仍为 dry-run；symlink、非当前所有者和不支持安全打开的文件会拒绝。

```sh
python scripts/platform_ops.py credentials
```

`intel` 可执行 `version`、`quota`、`submission-limits`、`search`、`files`、`leaderboard`、`kernels` 和 `pull`；除 pull 外均须 `--out` 指定新的证据文件。`quota` 查询加速器配额，比赛提交余量用 `submission-limits`，两者不同。配额接口缺失或权限失败即失败，不推测余额。

```sh
python scripts/platform_ops.py intel version --out cli-version.txt --execute
```

```sh
python scripts/platform_ops.py intel submission-limits --competition example --out limits.txt --execute
```

```sh
python scripts/platform_ops.py intel search --query example --out competitions.csv --execute
```

```sh
python scripts/platform_ops.py intel kernels --competition example --out kernels.csv --execute
```

```sh
python scripts/platform_ops.py intel pull --kernel owner/example-kernel --out inspected-source --execute
```

pull 是查看最新源代码，不能作为指定版本运行产物的证据。下载代码只是数据，不自动执行。官方 CLI 分页输出默认只代表当前页，本工具不会声称已穷尽社区内容或榜单；自行记录查询条件、页范围和时间。

MCP 使用配置的 HTTPS Streamable HTTP 端点，先标准 JSON-RPC `initialize`、`notifications/initialized`，再动态 `tools/list`；不猜固定工具名。`--tool` 必须来自实际列表且带 `readOnlyHint: true`，然后 `tools/call`。服务端提示是信任边界，不是安全证明，调用前须核对工具描述和入参。列表最多 20 页，HTTP 30 秒超时、响应上限 10 MB。支持 JSON 和有界 SSE 响应，忽略通知并按请求 id 取得 JSON-RPC 结果；未知协商版本明确拒绝，通知的 HTTP 202 空响应可处理。token 仅从 `--token-env` 指定变量读取到请求头，不落盘、不回显，不允许含凭据/query 的 endpoint，也不跟随重定向转发 token。

```sh
python scripts/platform_ops.py mcp --endpoint https://example.invalid/mcp --token-env MCP_TOKEN --out tools.json --execute
```

```sh
python scripts/platform_ops.py mcp --endpoint https://example.invalid/mcp --token-env MCP_TOKEN --tool discovered_read_tool --arguments query.json --out intel.json --execute
```

## 准备运行包

源 `.py` 或 nbformat 4 `.ipynb` 写入全新目录。脚本包装器捕获异常并重新抛出；notebook 清空旧输出和 execution_count，插入运行日志函数及最后的完成事件。`runtime-events.jsonl` 记录阶段与耗时，退出时 `run-artifacts.json` 记录日志相对路径、产物相对文件名、大小和 hash。notebook 未执行到最后一格不能视为成功；仅有 process_exit 不是完成证据。可在模型代码中调用 `platform_log_event("inference", rows=count)`，不得把输入内容或凭据写入 metrics。

```sh
python scripts/platform_ops.py prepare-kernel inference.py --out kernel-package --id owner/example-kernel --title Example --competition example --gpu --execute
```

`--dataset owner/model-files` 可重复。`--internet` 明确启用后会进入包 metadata，上传审批的包 hash 覆盖此配置。该命令只是准备包，不启动训练、不上传。GPU 选项不证明远端硬件已经可用。

```sh
python scripts/platform_ops.py prepare-dataset model-files --out dataset-package --id owner/model-files --title Models --license CC0-1.0 --execute
```

数据许可必须由调用者核实并显式指定。源目录和目标必须分离，目标必须不存在；源文件会复制，原始证据保留。symlink、隐藏文件和明显凭据文件名被拒绝，这不是内容级秘密扫描，上传前仍需审阅清单。`package-manifest.json` 覆盖所有其他文件及 metadata，后续任何改动均使包校验失败。

## 具体审批、push 与提交

先运行预览以得到完整 `request`。审批 JSON 使用以下结构，其中 `request` 原样取自该次预览；下面只是结构说明，不是批准：

```json
{
  "request": {
    "operation": "submit-file",
    "competition": "example",
    "artifact_sha256": "完整的64位sha256",
    "file_name": "submission.csv",
    "message": "用户已经看过并批准的完整描述"
  },
  "expires_at": "2026-11-02T00:00:00Z",
  "quota": {"max_operations": 1}
}
```

approval 的 request 必须逐字段完全相同，expiry 必须含时区且仍有效；`max_operations` 为正整数。所有 pending/unknown/failed 尝试均消耗批准次数。批准记录、回执和输入是同一可信本地工作区的证据，不是可抵御本地文件被恶意篡改的签名系统。

```sh
python scripts/platform_ops.py kernel-push kernel-package
```

```sh
python scripts/platform_ops.py kernel-push kernel-package --approval approved-push.json --ledger platform-receipts --execute
```

dataset 上传对应 `dataset-create DATASET_PACKAGE`、`dataset-version DATASET_PACKAGE --message TEXT`，遵循相同预览、审批和回执流程。创建调用不含 `--public`。kernel push 会启动平台运行，回执必须从 CLI 响应取得确切版本号，缺失版本号即 unknown，不自动重新 push。

```sh
python scripts/platform_ops.py submit-file --competition example --file submission.csv --sample sample_submission.csv --id-column id --numeric-column prediction --message "已核实描述"
```

```sh
python scripts/platform_ops.py submit-file --competition example --file submission.csv --sample sample_submission.csv --id-column id --numeric-column prediction --message "已核实描述" --approval approved-submit.json --ledger platform-receipts --execute
```

该单步入口调用固定行数数值 CSV checker。托管平台桥接另支持文件、Notebook 和 agent 压缩包的结构校验、比赛专用验证器及精确产物身份；不能以通用结构 PASS 冒充比赛语义有效。

执行先把文件或整个 kernel/dataset 包复制到私有临时快照，核对快照 hash 与批准请求一致，CLI 只上传该快照，原目录随后变化不改变上传字节。随后在 ledger 中排他预留原子 pending 回执、fsync 文件和目录，再调用网络。相同比赛相同字节内容即使改文件名/message 也拒绝重提；代码提交以比赛、kernel、version、output_file 去重。并发预留用目录锁保护，崩溃遗留锁必须在检查所有 pending 后人工处理，工具不会自动删锁。始终使用同一比赛的同一 ledger；删除或更换 ledger 会丢失本地去重保护。

网络失败、超时、异常退出、缺少提交 ref、push 缺少 version 均落为 unknown 并返回非零。unknown 不能盲重试，成功退出和 unknown 都会留下可恢复回执。数据集上传成功仅表示 CLI 成功，不等于远端内容/隐私回读完成。

## 指定版本运行与产物

```sh
python scripts/platform_ops.py kernel-status --kernel owner/example-kernel --version 3 --execute
```

```sh
python scripts/platform_ops.py kernel-wait --kernel owner/example-kernel --version 3 --timeout 1800 --interval 5 --max-interval 60 --execute
```

wait 每轮输出状态，按 1.7 倍退避，单次 CLI timeout 不超过剩余预算，整体等待有上限。失败/取消/未知状态、超时返回非零，可继续 query 同一 version，不能借此重新 push。

```sh
python scripts/platform_ops.py kernel-output --kernel owner/example-kernel --version 3 --out version3-output --execute
```

官方新实现支持 `owner/slug/version`，但旧 CLI 可能忽略它。工具在本地检查所选可执行文件属于实际安装的 Kaggle 分发，并确认 status/output 源码向请求传入 version label，再将安装版本和源码 hash 写入产物回执。不导入 Kaggle 包，不触碰认证。检查失败直接拒绝，不退回 latest。可执行文件包装、非标准安装、源码布局变动均可能被保守拒绝，需要审查并升级适配器。

下载前确认指定版本 complete；下载到临时新目录，生成 `output-receipt.json`，含 kernel/version/各文件 hash 和页数，再原子移到目标。检查 CLI 支持显式 page-token 且可关闭内部自动翻页后，首请求传空 token，再按返回 token 逐页读取，始终保留同一 kernel/version，最多 100 页。重复 token、同名不同内容或超限均失败，不发布不完整产物；同名相同 hash 的重复文件仅保留一份。CLI output 以文件下载，不调用 unzip；可复用的 ZIP 解包函数预检路径穿越、绝对路径、symlink、覆写及总解压尺寸，不允许 archive 越界。

```sh
python scripts/platform_ops.py validate-output version3-output --kernel owner/example-kernel --version 3 --file submission.csv --sample sample_submission.csv --id-column id --numeric-column prediction --receipt validated-version3.json --execute
```

```sh
python scripts/platform_ops.py submit-code --competition example --kernel owner/example-kernel --version 3 --file submission.csv --output-dir version3-output --validated-receipt validated-version3.json --message "已核实版本3"
```

submit-code 用官方 `-k/-v/-f` 指定 kernel、版本和输出文件；要求 PASS 回执匹配 version、产物 hash 及 output receipt hash，并再次核对本地字节、validation hash 与下载清单 hash 三者一致。validate-output 在私有快照上做 hash 和 CSV 检查，不重复读取可变原文件来生成不同身份。审批还覆盖 validated receipt hash，防止换版本或换验证结果。执行时同样追加 `--approval ... --ledger ... --execute`。只证明结构及已下载版本身份，不证明隐藏数据语义或未来运行成功。

## 得分回读、恢复与核对

```sh
python scripts/platform_ops.py reconcile --ledger platform-receipts --attempt ATTEMPT_SHA256 --execute
```

reconcile 只读官方 submissions CSV/JSON，按 ref、完整 message 和可用文件名核对，取得 ref、状态和 public score。complete 且 score 为空、NaN、无限值都返回失败并标为 `score_unavailable`；数值 0 是有效分数。比赛故意不公开分数时只能保留未核验状态，不伪造 PASS。pending/running/submitted 只是等待评分，失败状态写入 failed 并返回非零。

超时上传没有 ref 时，仅以 message 找到的一条记录是候选，仍保留 unknown；须用户在平台核对实际产物后显式提供 `--remote-ref` 绑定，不能拿“最近一次提交”自动认领。CSV/JSON 未返回该 ref、分页遗漏、描述不符或多候选都拒绝认领。reconcile 不支持自动恢复未知 push/dataset 版本，需查看平台历史，再对已确定 kernel version 查询。历史中未找到记录并不授权重新提交。

## 测量与预算

`runtime.py benchmark` 从本地用户脚本加载 `decode(item, config)`、`infer(decoded, fold, tta, config)`，可选 `startup(config)`、`finalize(config)`、`synchronize()` 和 `gpu_memory_mb()`；显式 execute 后才运行。GPU 异步任务必须提供同步钩子，否则时间不能当 GPU 实际完成时间。禁止用未知社区代码作测量 driver。

items 是代表性单位输入的 JSON 列表。grid 是配置列表，例如 `[{"name":"base","folds":[0,1],"tta":[0],"passes":1,"loop_order":"unit-outer"}]`；一次 infer 为一个单位的一次 pass，多次 passes 的总时间会按次数归一。benchmark 写 `timings.csv` 和 `overheads.json`；记录每单位 decode_seconds、每 fold/TTA 的 seconds/passes、进程生命周期 RSS 峰值及可选 GPU 读数。GPU 内存不自动探测，缺失始终未测。

```sh
python scripts/runtime.py benchmark --driver local_driver.py --items representative-items.json --grid config-grid.json --out measured-runtime --execute
```

```sh
python scripts/runtime.py plan-csv measured-runtime/timings.csv --hidden-items 10000 --grid config-grid.json --overheads measured-runtime/overheads.json --queue-seconds 120 --crash-min-seconds 0 --crash-max-seconds 300 --budget-seconds 3600 --rss-limit-mb 16000 --gpu-limit-mb 15000
```

也可接收外部实测 CSV，但必须含 seconds/passes/fold/tta、decode_seconds；unit_index/rss_mb/gpu_mb 可选，有 config 时按 grid 中 name 选取。每个计划 fold/TTA 必须有实测数据；缺组合或 decode 不按 0 补齐。按每阶段 nearest-rank p95 外推 hidden n；unit-outer 解码每单位一次，fold-outer 按 fold×TTA 重解码。startup/finalize 来自实测 overheads，queue 单列，仅进入总墙钟预算，crash 区间和安全系数形成运行时间区间。

RSS/GPU 可按 unit_index 的观测最大正增长斜率外推，并与各自限额比较；只有一个单位时不声称已测增长，无数据时不输出“内存安全”。线性增长、相同设备、隐藏单位数量及最坏单位大小均是假设，需要远端 smoke run 验证。CSV p95、内存线性估算和 crash allowance 是预算证据，不是平台时限或隐藏集成功保证。benchmark 异常会直接失败，不把未完成测量补成通过。

简单已有批次 JSON 也可用 `runtime.py project`，字段为 startup_seconds/batch_seconds/measured_items_per_batch；它只做明确假设的批次估算，不替代上面的每单位实测。`runtime.py measure SCRIPT --artifact RESULT.json --execute` 可测独立脚本总耗时并在异常时保存 failed artifact。

## 本地离线验证

```sh
python -m unittest discover -s tests -p test_platform_ops.py -v
```

测试使用 mock CLI、mock MCP transport 和临时假可执行文件，覆盖超时/未知、审批时效和配额、重复阻断、pending 先落盘、空分数、exact version/hash 不符、ZIP 越界、私有默认、dry-run/help 无副作用以及解码/循环顺序/内存预算。它们不证明账号权限、在线 API、真实远端产物或隐藏数据已经验证。
