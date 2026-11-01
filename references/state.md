# 本地状态、台账与恢复

这里的命令只操作本地文件，不认证、不下载、不训练、不提交，也不连接 Kaggle。新比赛使用新目录。先把命令中的大写路径换成实际路径，并根据本次核验的事实填写模板中的 `null`；模板不是实验记录。

```sh
python scripts/workspace.py init WORKSPACE
python scripts/workspace.py config WORKSPACE --input PLAN.json
python scripts/workspace.py status WORKSPACE --now 2026-11-01T00:00:00Z
```

`config` 只允许合并 competition、metric、gates、deadline、reserve_hours、public_units、budget、route、goal、validation、next_action、late_join_days、started_at。metric 为 `{name, direction}`，direction 只接受 `higher` 或 `lower`；注册实验后不能更换指标。budget 中 gpu_hours、money、submissions 必须是有限非负数。public_units 以顶层字段为准；兼容 budget.public_units，但两处不能冲突。日期必须带时区。

competition gate 为 `null`，或 `{decision: "PASS"|"BLOCK", reason: "证据与决定"}`。五个阶段分别为 strategy、test_distribution、validation、midpoint、finals。PASS 必须是明确审查或分析后的结论，不能因为文件存在或公榜上涨就自动填写。

`competition.json` 保留 schema 1 及原有 deadline、reserve_hours、goal、route、budget、validation、holdout_opened_at、next_action、created_at 字段，扩展 metric、gates、public_units、late_join_days（默认 7）、started_at。

`ledger.json` 保存 schema 1、metric、public_units、validator `{authority, covered_stages}`、findings、candidates、trust、holdout `{sha256, opened_at, audit}`、final_selection，以及成功写入操作的 audit。留出集以 ledger.holdout 为准；competition.holdout_opened_at 仅保留旧接口字段，不能据此再次开封。

## 先注册，再记录结果

```sh
python scripts/workspace.py register WORKSPACE --input CANDIDATE.json
python scripts/workspace.py update WORKSPACE --id CANDIDATE_ID --artifact PREDICTIONS.csv
python scripts/workspace.py validator WORKSPACE --input VALIDATOR.json
python scripts/workspace.py score WORKSPACE --id CANDIDATE_ID --input SCORES.json --artifact PREDICTIONS.csv
python scripts/workspace.py update WORKSPACE --id CANDIDATE_ID --input PROMOTION.json
```

注册必须填写唯一 id、family、hypothesis、布尔 structural、整数 swept >= 1、非空 stages。stages 列出完整预测流程，包含预处理和后处理。旧的布尔 swept 统一计为一次试验。可选 parent 必须已经存在，root 从父候选的谱系推导，不能凭空填写。注册接口拒绝结果字段。

每个候选包含 id、family、parent、root、structural、hypothesis、swept、stages、artifact_sha256、gate_artifact_sha256、gate、oof、holdout、public、private、status、selection_units。初始分数为 null，status 为 registered。selection_units 初始为 `{}`，测量后填写 `{独立单位ID: 有限分数}`，供残差相关分析使用；不得放入可识别个人或敏感的原始 ID。

`update` 只允许 stages、selection_units、status 和 `--artifact`。artifact 选项读取实际文件字节，只保存 SHA256。文件哈希变化会清除旧分数、单位分数、候选 gate、trust 和 final_selection；修改流程阶段会使 gate 失效。id、hypothesis、parent、family、试验次数注册后不可重写；假设或搜索改变时新注册候选。可用 status 为 registered、running、failed、withdrawn、promoted。

`score` 接受有限数值 oof/holdout/public/private、selection_units、必填的当前 artifact_sha256，以及可选 gate。gate 使用完整统计输出 `{decision, artifact_sha256, validator_authority, covered_stages, evidence}`；decision 为 PASS 或 BLOCK，非空 evidence 必须来自统计分析，本工具不计算显著性。省略 gate_artifact_sha256 时从 gate.artifact_sha256 取值。传入 `--artifact` 还会核对当前字节。晋升和最终选择前应重新哈希；外部程序修改了文件后，工具只有再次读取时才能发现变化。

晋升输入为 `{"status":"promoted"}`。必须已配置 metric，strategy/test_distribution/validation 三个 gate 均 PASS，存在 OOF，候选 gate PASS 且匹配当前 artifact，两处 validator authority 均为 full，两处 covered_stages 均覆盖完整流程，且不存在未关闭的阻塞发现。

swept > 1 时，gate.evidence.search_trials 不得低于预注册试验次数；gate.evidence.replication 必须含 fresh true、independent true、与当前文件相同的 candidate_sha256、非空 evidence_id。该回执记录外部独立复验，并不代表本地台账执行了复验。partial/proxy/unknown/veto_only authority 可以留档，但不能授权晋升；统计 PASS 不能扩大验证器权限。

## 发现、信任判断、封存留出集与收尾

```sh
python scripts/workspace.py finding WORKSPACE --input FINDING.json
python scripts/workspace.py resolve WORKSPACE --id FINDING_ID --input RESOLUTION.json
python scripts/workspace.py trust WORKSPACE --input TRUST.json
python scripts/workspace.py holdout-register WORKSPACE --artifact SEALED_IDS.txt
python scripts/workspace.py holdout-open WORKSPACE --artifact SEALED_IDS.txt --input OPEN_REASON.json
python scripts/workspace.py final WORKSPACE --input FINAL.json
```

finding 必须含 id、severity（blocker/critical/high/warning/info）、description。blocker、critical、high 都会阻塞晋升和最终选择，直到明确 resolved 或 accepted。关闭输入为 `{status:"resolved"|"accepted", reason: ...}`；accepted 风险始终留档，不会被静默丢弃。

`trust` 保存分析 JSON：regime 为 ALIGNED/WEAK/DIVERGENT/INSUFFICIENT，并含唯一 candidate_ids 与 reason。computed_at、n_paired、public_units、local 及统计诊断原样保留。工具另加当前候选、验证器、发现、留出集、相关比赛配置的事实指纹；后续变化会清空判断或使指纹失效。状态查询只报告差异。INSUFFICIENT 是可以记录的仅用本地证据判断，不是信任公榜的许可。

留出集注册只保存事先准备并封存的单位 ID 文件的 SHA256，不保存内容和位置，不能覆盖旧封存。开封必须提供字节完全一致的文件、非空原因，且无未关闭阻塞。只记录一次成功开封及时间，再次开封失败。文件应由用户另行隔离和控制访问；该台账提供审计约束，不构成文件系统访问控制。开封前不能记录 holdout 分数。

`final` 必须含唯一 candidate_ids、reason，以及 holdout_decision `{decision:"used"|"not_opened"|"not_used", reason: ...}`。used 要求已有开封审计，且每个所选候选都有 holdout 分数。not_opened 必须与开封审计一致。not_used 明确解释为何未采用该证据，例如数据太少；它不会抹掉已经发生的开封。

每个最终候选必须已 promoted 且证据仍有效，已记录返回的 public 分数以证明提交评分成功；这一回执不授权按公榜分数排名。finals gate 必须 PASS，阻塞发现必须关闭或明确接受，当前 trust 必须覆盖所选候选。status 为 blocked 的选择分析不能保存为最终结果。先在 select 的分析 JSON 中补齐明确 reason 与 holdout_decision，再保存。trust 为 INSUFFICIENT 时，最终记录标记 evidence_basis local_only。该操作只保存本地决策，不提交比赛，也不证明私榜表现。

## 状态回读与保留旧工作

`status` 根据事实提出建议：阻塞发现、未通过阶段、剩余时间小于 reserve_hours、晚加入、提供 started_at 时的中期节点、至少五个已有成对 OOF/public 结果的 structural 候选但缺少当前 trust、以及建立基线或继续迭代。独立模型 root 同样计数，不要求一定有 parent。查询不写文件，不生成虚构发现或分数。candidate_errors、trust_current、final_current 显示配置变化后的失效状态。

原有 `inspect`、`record --event FILE`、`init` 接口保留。inspect 返回原配置、事件数与各 id 的最新事件。record 接受含 id/event 的 JSON 事实，保留 experiments.jsonl 追加记录，兼容末行缺少换行；已有日志损坏时拒绝追加。该通用日志不能绕过候选晋升条件。

文件写入使用同目录临时文件、fsync 和原子替换，所有写入通过本地 fcntl 锁串行化。一次 ledger 操作及其审计一起原子写入；失败时保留此前台账，之后可重试。

config 会同步 metric/public_units 到供统计工具读取的 ledger.json，清除旧 trust/final，并在替换两个文件前写入恢复记录。中断后 status/inspect 只读报告 recovery_required。运行以下命令完成已记录的事务；下一次写操作也会先恢复。不要删除恢复记录来掩盖部分写入。

```sh
python scripts/workspace.py recover WORKSPACE
```

初始化拒绝所有已有状态文件。发现旧 GAMEPLAN.md 或 competition_ledger.json 而新 schema 不完整时，inspect/status 报告 legacy_not_migrated，初始化拒绝覆盖。迁移必须逐项核对原始证据，不由工具猜测。配置、JSON 输入、事件日志和证据原因中都不得复制密钥、账号细节、私人数据行或凭据。
