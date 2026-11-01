# 能力清单与验收

本包按功能提供工具，不要求每项功能对应一个独立脚本。阶段说明、机器检查和平台操作共同构成工作流；没有把只写入说明的能力算作自动化工具。

| 能力 | 本地实现或入口 | 验证重点 |
| --- | --- | --- |
| 选赛、规则、目标与资源路线 | decisions、community；workspace config/status | 有证据的计划；截止与预算决定行动 |
| 阶段路由与跨会话状态 | workspace status | 缺失门禁、阻塞、结束时间、中期与旧信任提示 |
| 新工作区、原始证据保护 | workspace init | 不覆盖已有记录；旧格式需要明确迁移 |
| 实验预登记、父版本和台账 | workspace register/update/score/record | 先登记再记录；并发无丢更新 |
| 问题修复或明确接受 | workspace finding/resolve | 未解决关键问题阻止晋级和最终选择 |
| 保留集封存和打开审计 | data_tools holdout；workspace holdout-register/holdout-open | 分组不交叉、哈希一致、不可再次声明未打开 |
| 随机、分层、分组和时间划分 | data_tools folds | 固定种子、唯一 ID、同组不跨折、时间训练不见未来 |
| 通用指标与自定义评分 | analysis score；metrics | 加权指标、整体与单位分数分开、输入有限性 |
| 独立单位和组别噪声 | analysis noise | 按独立簇和目标测试规模计算，少簇证据不足 |
| 无训练结果时的模拟先验 | analysis noise 的 prior 参数 | 明示模拟假设、随机种子和参数，不冒充实测 |
| 分组差异、排名翻转与打乱对照 | analysis groups/null | 多候选同单位对比；解释可交换性假设 |
| 单位分数统计初筛 | paired_check | 显式方向、实际收益和折间一致性；非完整晋级 |
| 完整晋级门禁 | analysis gate | 行级与单位预测对齐、覆盖阶段、收益/噪声/尾部、搜索后复现、产物身份 |
| 公榜与本地信任检查 | analysis trust | 整体/结构/家族内相关，小样本与常量不能强判 |
| 排行榜奖牌分数线和同分平台 | analysis leaderboard | 由输入提供当场规则/阈值，不伪造通用奖牌资格 |
| 最终多候选建议 | analysis select | 当前门禁与评分、保留集/OOF、根路线与残差多样性、公榜作用受限 |
| 最终决策登记与过期检查 | workspace final/status | 当前信任与哈希、关键问题、保留集决策和已评分候选 |
| 赛后复盘 | analysis postmortem；finish | 公私榜相关、最佳未选候选与缺失证据 |
| 多模态建模、伪标签与集成 | modeling、experiments | fold 边界、教师训练范围、样本外集成、真实部署约束 |
| 认证状态检查 | platform_ops credentials | 只报告存在性和权限，不读取/显示密钥值 |
| 社区和平台信息采集 | platform_ops intel/mcp | 动态工具发现、只读调用、保留原始证据、无指令信任 |
| 私有 Notebook 与中间数据集准备 | platform_ops prepare-kernel/prepare-dataset | 清理执行输出、版本 manifest、日志和哈希 |
| 上传、运行状态与有界等待 | platform_ops kernel-push/kernel-status/kernel-wait | argv 调用、明确授权、回执、退避和终止失败 |
| 精确运行版本的输出 | platform_ops kernel-output/validate-output | 拒绝把 latest 当所需版本，输出哈希与版本共同校验 |
| 中间数据集发布新版本 | platform_ops dataset-create/dataset-version | 具体包哈希、许可、批准范围；继承的可见性不得假称已核实 |
| 推理测量与隐藏规模估算 | runtime benchmark/project/plan-csv | 启动、p95、解码、折数/增强、队列、写出和内存增长 |
| 隐藏失败时间定位 | runtime project/plan-csv | 仅估计运行进度，不能从错误反推标签 |
| 运行保护与原子输出 | assets/inference_support.py | 文件哈希、预算、显式降级、错误不吞、写出失败保留原文件 |
| 固定/可变 CSV 提交检查 | submission_check | 复合 ID、列、有限值、合法标签、概率范围与和 |
| 文件赛与代码赛正式提交 | platform_ops submit-file/submit-code | 默认 dry-run、冻结文件或版本、具体批准、先落盘 pending |
| 评分回收、未知结果与去重 | platform_ops reconcile | 空分数不记成功、未知不重传、消息变化不能绕过去重 |
| 任务委派与调度恢复 | managed tick/serve/install/uninstall；operations | 持久化到期时间、独立作业 supervisor、已完成回执恢复、不重复训练；明确区分服务生成与实际启动 |
| 一次授权与预算 | mandate；managed | 比赛范围、费用/GPU/时间、总与每日提交数；执行前预留、并发去重、未知保守占额、暂停/撤销 |
| 自动加入和规则接受 | CompetitionAdapter join + inspect | pinned 或 delegated 规则授权、约束审查、实际加入与条款哈希回读 |
| 初始数据准备 | managed bootstrap + adapter download | 无预制模型/划分也能启动；实际下载、哈希核对、生成准备脚本和验证器 |
| 持续讨论区追踪 | adapter refresh_intel + managed | 新帖子/评论游标、持久化到期时间、增量触发实际 Agent 复评 |
| 原生 Agent 工作进程 | agent_process | 实际启动所配 AI CLI、严格结果协议、scope 提示、无伪造成功、超时子进程清理 |
| 托管最终平台选择 | managed finalize + adapter select_final/final_selection | 规则决定数量、冻结模型、保留集一次评估、精确提交引用、平台独立回读后才 complete |

## 在本地复核

在技能目录运行完整测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

测试包含从空工作区开始的托管场景：自动加入离线平台、下载合成数据、真实分组训练两个模型并生成可运行 Notebook、独立重算训练与预测、统计门禁、私有上传与模拟平台实算分数、新评论复评、冻结和封存集评估、最终两份平台选择回读。另测预算不足、并发、超时/未知、重启和后台 PID 生命周期。测试不会访问真实 Kaggle、读取用户凭据或产生正式提交。

## 平台边界

精确版本输出要求安装的官方 CLI 能证明它将版本号传给相应接口；不满足时明确拒绝，不能偷偷下载最新输出。平台分页、权限和动态工具差异可能需要按当前平台证据调整适配。框架、GPU 环境和比赛专用结构仍须在真实运行前核验。

本地测试可以验收这些流程和工具的行为，不能替代一次完整线上比赛验证。托管需要配置的 AI CLI、可用的登录与平台工具及计算资源；提供的执行器可常驻，并可安装用户级调度，机器和服务仍需保持可用。预算账本不是操作系统沙箱，也不能代替外部云商的账户硬限额。
