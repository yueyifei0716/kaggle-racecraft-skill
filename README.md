# Kaggle Racecraft Skill

全流程 Kaggle 托管 Skill：从比赛研究、模型迭代到自动提交与最终方案选择。

供支持本地 Skill 的 AI 助手执行 Kaggle 比赛流程。包含策略选择、社区研究、样本外验证、实验台账、统计门禁、远程运行、评分对账和平台最终选择。托管模式支持一次授权后在预算内持续执行。入口为 [SKILL.md](SKILL.md)，操作覆盖见 [能力清单](references/capabilities.md)。

## 使用

克隆 [kaggle-racecraft-skill 仓库](https://github.com/yueyifei0716/kaggle-racecraft-skill)：

```bash
git clone https://github.com/yueyifei0716/kaggle-racecraft-skill.git
```

将克隆得到的 `kaggle-racecraft-skill` 文件夹复制到所用助手支持的 skills 目录，并命名为 `kaggle`；技能调用名保持为 `$kaggle`。不要只复制入口文件；保留 scripts、references、assets 和 agents。脚本需要 macOS/Linux 上的 Python 3.11 或更新版本，本地辅助工具仅使用标准库（状态锁使用 POSIX 文件锁）。连接 Kaggle 时另行安装并配置官方 Kaggle CLI；模型训练所需框架由具体比赛决定。

先告诉助手比赛链接、目标、工作目录、计算与提交预算，以及是否完全托管。助手应准备配置、确认一次授权并启动执行器；不要求你先编写模型、折划分或每轮批准文件。托管模式会自主准备基线、训练与验证代码、获取新增讨论、在预算内提交，并完成平台最终勾选与回读。账号登录、工具能力和必要资源必须可用；规则接受可按初始授权委托或固定文档版本。参见 [托管授权](references/mandate.md)、[托管执行](references/managed.md) 和 [平台桥接](references/competition-adapter.md)。

已有比赛记录不会自动迁移或覆盖；只执行单步工具时，在新目录初始化可运行：

```bash
python3 scripts/workspace.py init /path/to/competition
```

```bash
python3 scripts/workspace.py status /path/to/competition
```

各脚本均提供 `--help`。单步 `platform_ops.py` 保留具体版本审批模式；托管模式使用独立的比赛范围授权和预算账本，不逐版本询问。技能不公开用户凭据，认证由已登录的平台工具提供。

## 本地验证

在这个文件夹运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

测试使用合成数据和模拟平台，不会调用付费 API 或真的参赛。托管场景测试从空工作区准备数据，实际拟合小模型、生成两个可运行 Notebook、运行独立验证器和统计门禁、取模拟平台实算分数、处理新增评论、封存集评估及平台最终选择回读；同时验证预算拒绝和重启恢复。本地测试通过不等于已完成真实账号、线上桥接、GPU 或新比赛验收，也不保证奖牌和完全避免过拟合。

## 许可

本包以 [MIT](LICENSE) 许可发布。实际比赛中使用的数据、模型和其他代码各有其使用条件，不因本包许可而改变。
