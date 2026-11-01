# Local analysis tools

`scripts/analysis.py` and `scripts/metrics.py` are standard-library-only. They read explicitly named local files and print JSON. They never write a ledger, train, download data, submit, or choose an online action. Run a command's `--help` for all options. A valid analysis, including `BLOCK` or an insufficient sample result, exits 0; malformed input exits 2 with JSON on stderr. Do not use exit status alone to promote a candidate.

## File contracts

Unit CSV requires `unit,score`; optional `weight,fold,group`. Weight defaults to 1 and must be positive. Promotion normally needs fold/group columns. Each unit is one independent resampling unit unless `--cluster group` or `--cluster fold` explicitly identifies shared dependence. Unit weights control the mean of unit scores; they do not replace row weights.

Row CSV requires `row_id,unit,fold,y_true,y_pred`; optional `weight,group`. Every row ID is globally unique. One unit belongs to one fold and one group. Binary AUC/logloss use probabilities; accuracy and binary F1 use hard class labels. QWK uses ordered integer labels with integer spacing, including absent intermediate labels. Binary F1 uses positive class 1 and returns 0 when neither truth nor predictions contains a positive. Other F1 definitions use a custom scorer.

CSV duplicate IDs/headers, malformed widths, empty required fields, nonpositive weights, nonfinite scores, and pair misalignment are errors. File order may differ: pairs are joined on exact IDs. Baseline/candidate rows must have identical row IDs, truth, units, folds, groups, and weights. Baseline/candidate unit scores must have identical unit IDs, weights, folds, and groups. When row and unit files are supplied together, each unit's score is recomputed and checked against its CSV value with numerical tolerance.

Ledger JSON follows the standalone ledger schema:

```json
{
  "metric": {"name": "mae", "direction": "lower"},
  "validator": {"authority": "full", "covered_stages": ["data", "features", "training", "prediction", "scoring"]},
  "findings": [{"id": "example", "severity": "blocker", "status": "resolved"}],
  "candidates": [],
  "trust": null,
  "holdout": {"opened_at": null}
}
```

Candidate records use `id,family,parent,root,structural,oof,holdout,public,private,status,gate,artifact_sha256,gate_artifact_sha256,selection_units`. Candidate ID/family/root are nonempty strings; parent, when set, names an existing candidate. Scores are finite JSON numbers or null, never numeric strings/booleans. `selection_units` is an object mapping unit ID to finite score. Missing optional scores remain unknown. Extra ledger fields are preserved in memory, but nothing is saved. Duplicate JSON keys and nonfinite numbers, including overflowed numeric literals, are rejected. `trust:null` means no trust analysis has been computed; it does not prevent a local selection.

## Score and custom metrics

```sh
python scripts/analysis.py score --rows predictions.csv --metric rmse
```

Built-ins: MAE, RMSE, accuracy, binary F1, binary logloss, weighted tie-aware AUC, and QWK (`mae,rmse,accuracy,f1,logloss,auc,qwk`). Undefined AUC/QWK is an error; it is never silently replaced by zero. Logloss clips exact 0/1 probabilities to `1e-15` from the boundary.

A custom scorer is an explicitly trusted local Python file containing `def scorer(y_true, y_pred, weights): ...`, returning one finite scalar. No hidden imports or dependency installation are performed by the tool; loading the scorer executes that user's Python code.

```sh
python scripts/analysis.py score --rows predictions.csv --metric custom --scorer /absolute/path/scorer.py:scorer
```

The result reports the pooled metric and each unit metric separately. For RMSE, AUC, QWK and F1, pooled metrics generally differ from averages of unit metrics. A custom metric uses `metric.name="custom"` and explicit `higher` or `lower` direction in the ledger. By default every unit must support the primary metric; group-level custom/AUC scoring needs enough variation within every scored unit.

病人或单样本 unit 可能只有一个标签，不能定义该 unit 的 AUC；正确预测的单类别 unit 也不能定义 QWK。此时可**显式**指定可计算的辅助指标，不改变总体主指标：

```sh
python scripts/analysis.py score --rows predictions.csv --metric auc --unit-metric logloss
```

例如 AUC 主指标搭配二元 logloss，QWK 搭配 MAE。`metric` 与 `overall` 仍为主指标，新增 `unit_metric` 明示 unit 使用的指标，`units[].score` 为该辅助指标的实际计算结果。辅助指标须由任务明确选择；独立 CLI 不自动替换，不产生所谓“单样本 AUC”。没有 `--unit-metric` 时保持原有行为，未定义的 unit AUC/QWK 仍报错。自定义主指标可以搭配内置辅助指标；辅助指标当前仅支持内置指标，方向使用其内置定义。

## Paired promotion screen

```sh
python scripts/analysis.py gate --ledger ledger.json --candidate-id candidate-1 --baseline baseline-units.csv --candidate candidate-units.csv --baseline-rows baseline-rows.csv --candidate-rows candidate-rows.csv --artifact frozen-candidate.csv --cluster group --min-gain 0.001 --max-worst-harm 0.01 --max-tail-harm 0.005 --search-trials 4 --replication replication.json
```

Positive gain always means improvement, honoring metric direction. This percentile paired bootstrap resamples the same independent clusters for both candidates and recomputes both the weighted mean of unit score changes and the pooled row metric. It does not bootstrap two unpaired score distributions. Row weights are used for the pooled metric. Small/dependent cluster sets need a defensible resampling design; choosing `unit` does not establish independence.

Default screen requires at least 3 folds, 2 groups, positive lower CI bound, observed gain at least `--min-gain`, no worst-unit harm, and no lower-tail unit harm. Change `--min-folds`, `--min-groups`, `--ci-min-gain`, `--tail-quantile`, `--max-tail-harm`, and `--max-worst-harm` to reflect an explicit experiment policy. Unit-level tail quantiles are unweighted across units. Cluster count must be at least two. Undefined pooled bootstrap draws block promotion rather than being silently discarded. Without both row files the unit analysis still runs, but the result is `BLOCK` with `overall_rows_required`.

使用辅助 unit 指标时，score 和 gate 必须显式传入相同的 `--unit-metric`。Gate 根据该指标从原始行预测重新计算每个 unit，核对 unit CSV；总体及其每次聚类 bootstrap 始终重算 ledger 中的主指标。两个方向分别计算，例如 AUC 增大为总体改善，logloss 减小为 unit 改善；不能把辅助分数当成主指标的分解或声称其平均值等于主指标。

```sh
python scripts/analysis.py gate --ledger ledger.json --baseline baseline-logloss.csv --candidate candidate-logloss.csv --baseline-rows baseline-rows.csv --candidate-rows candidate-rows.csv --artifact candidate.csv --unit-metric logloss --min-gain 0.001 --ci-min-gain 0 --unit-min-gain 0.002 --unit-ci-min-gain 0
```

显式 `--unit-metric` 时，`--min-gain/--ci-min-gain` 只约束主指标量纲；新增 `--unit-min-gain/--unit-ci-min-gain` 各默认 0，只约束辅助指标量纲。`--max-worst-harm/--max-tail-harm` 也是 unit 指标的量纲，须按该指标设置。辅助指标退化会阻塞，即使总体 AUC 提升；总体主指标不达标也会阻塞，即使辅助指标改善。不传 `--unit-metric` 时保留旧策略：主阈值同样约束 unit，新增 unit 阈值可进一步收紧。Gate evidence 保留原 `metric` 主指标字段，并记录 `primary_metric/unit_metric` 的名字与方向、两套实际阈值、两个层次的 gain/CI 及各自原始文件 hash。未定义的总体 bootstrap 样本仍阻塞，不用辅助指标填补。

Both this screen and final selection require validator authority `full` and coverage of `data,features,training,prediction,scoring`. Override the explicit list with `--required-stages`; do not remove stages merely to obtain a pass. A `veto_only` validator can reject but cannot authorize promotion. Any unresolved `blocker,critical,high` finding blocks the screen. `--blocking-severities` is an explicit policy override.

A searched candidate (`--search-trials > 1`) requires an independent fresh replication receipt:

```json
{"fresh":true,"independent":true,"candidate_sha256":"<actual artifact SHA256>","evidence_id":"independent-run-or-split-identifier"}
```

`--candidate-id` takes the maximum of the supplied search count and that candidate's registered `swept` count; matching artifact hashes also discover registered counts. The screen cannot execute or establish the independence of a replication from booleans. The full validator must verify the receipt's source, fresh sample/split, method, results and lack of reused search information. An absent, stale or mismatched receipt blocks the screen. Adaptive search is not cured by bootstrapping the same development data.

Gate JSON uses `{decision:"PASS"|"BLOCK",validator_authority,covered_stages,artifact_sha256,evidence:{...}}`. It hashes the explicitly supplied actual artifact and every supporting CSV. A specified registered candidate must have that same artifact hash. The artifact conversion from row predictions is caller-provided and must be covered by the full validator; a SHA256 proves byte identity, not scientific validity or conversion correctness. Embed the complete result in the ledger candidate's `gate` using the state tool, and set both candidate artifact hashes to the same checked artifact hash. A later artifact change invalidates the gate. Final selection also requires `gate.evidence.metric` to exactly match the current ledger's metric name and direction; a missing or changed metric invalidates the gate.

## Absolute uncertainty and simulation priors

```sh
python scripts/analysis.py noise --units unit-scores.csv --cluster group --target-units 30 --resamples 2000
```

This estimates absolute-score uncertainty, not the uncertainty of a paired improvement. With `--cluster group`, `--target-units` means target independent **groups**. The default target is the number of observed independent clusters. Resampling the empirical distribution for a different target size assumes new clusters are exchangeable with observed clusters. The output states this assumption; it does not estimate new test populations or missing group types. The reported noise floor is half the percentile CI width, not a universal minimum useful effect. Fewer than two observed clusters gives unknown noise, even with a large target size.

A parameterized score-scale SD prior can run without observed units:

```sh
python scripts/analysis.py noise --simulated-sd 0.1 --simulated-n 30 --resamples 2000
```

This Gaussian iid mean prior is explicitly labeled `SIMULATED_PRIOR_NOT_OBSERVED_EVIDENCE`. It is not an observed metric model.

Metric-specific synthetic priors are also available, without any dataset:

```sh
python scripts/analysis.py noise --prior-metric auc --prevalence 0.2 --effect 1 --rows-per-unit 64 --correlation 0.4 --target-units 20 --resamples 500 --seed 42
```

Binary priors use Bernoulli labels and a Gaussian class margin, then sigmoid probabilities; accuracy/F1 threshold at 0.5. `--correlation` is Gaussian within-unit latent **noise** correlation, not an estimated label/error correlation. Independent units receive independent shared noise. `--effect` controls latent class separation, not an observed effect estimate. Supported binary priors: accuracy, F1, logloss, AUC. `--prior-metric qwk --classes 5` uses uniform ordinal truth and Gaussian-noised class margins with class thresholds. All assumptions, seed, valid/undefined draws and intervals are returned under `SIMULATED_ASSUMPTIONS_NOT_OBSERVED_EVIDENCE`. These synthetic distributions are sensitivity analyses. They do not establish competition class balance, calibration, group dependence or achievable scores. Empirical and prior results stay in separate fields.

## Group comparisons and shuffle null

```sh
python scripts/analysis.py groups --candidate baseline=baseline-units.csv --candidate challenger=challenger-units.csv --direction lower
```

Two or more candidates produce per-group weighted unit-score means, tied ranks, uncertainty, and pairwise group/overall sign flips. IDs and metadata must align exactly between candidates. `--units` alone gives a one-candidate absolute report. These are averages of unit scores; they do not claim pooled AUC/QWK/F1. Sign flips are descriptive alerts, not independent significance tests.

```sh
python scripts/analysis.py null --rows predictions.csv --metric auc --resamples 2000 --seed 42
```

The null shuffles predictions within each unit/fold, leaving labels, IDs and weights fixed. Its corrected empirical p-value is `(extreme + 1)/(draws + 1)`. This requires exchangeability inside each stratum; temporal/spatial dependence can invalidate it. Constant or singleton strata with no exchangeable variation produce an insufficient result. Custom metrics require `--direction higher|lower`.

## Public/local rank trust

```sh
python scripts/analysis.py trust --ledger ledger.json --local oof --min-samples 4 --min-public-units 20
```

Spearman uses average ranks for ties and explicitly returns unknown for insufficient samples or constant scores. Output separates overall, structural-candidate, and within-family correlations. Overall agreement cannot hide weak or unidentified within-family behavior. Regimes: `ALIGNED`, `WEAK`, `DIVERGENT`, `INSUFFICIENT`; unknown/small public-unit counts cannot support a stronger regime. These are descriptive correlations across searched candidates, not an independent test-set acceptance claim. Keep computed trust JSON, including local authority, candidate IDs, pair count, timestamp and candidate fingerprint, when saving through the state tool.

## Leaderboard thresholds and ties

```sh
python scripts/analysis.py leaderboard --table leaderboard.csv --thresholds user-thresholds.json --direction higher
```

Leaderboard CSV is `id,score`. Threshold JSON maps user-specified medal names to current score thresholds, e.g. `{"gold":0.95,"silver":0.91}`. The tool reports equal-score platforms with rank-start/end and score threshold eligibility. It does not hardcode medal proportions, invent tie-breaking rules or assert official awards from score thresholds alone.

## Final selection and retrospective review

```sh
python scripts/analysis.py select --ledger ledger.json --count 2 --residual-baseline frozen-baseline-units.csv
```

The first slot uses the opened holdout if `holdout.opened_at` is set, otherwise OOF. Candidates missing the authoritative score or a retrieved public score are excluded. Public score retrieval confirms that the standalone final artifact was scored; it does not give public ranking authority. Candidate status must be validated/eligible/promoted/selected. Unresolved blockers, incomplete validator coverage, stale gate/artifact identity, or a searched candidate without current replication evidence prevent eligibility.

Additional slots require different declared roots and sufficient aligned unit scores. Pearson residual correlation is calculated from `candidate.selection_units[unit] - frozen_baseline_score[unit]`, with exact shared unit sets, at least 4 units by default, and nonconstant residuals. `--max-correlation` sets the cap. A missing residual baseline or unidentified diversity leaves slots unfilled. Root/family declarations and baseline provenance still need external verification; a string field cannot establish model independence.

An explicitly enabled public second-slot hedge is supported:

```sh
python scripts/analysis.py select --ledger ledger.json --residual-baseline frozen-baseline-units.csv --public-second --public-max-local-regret 0.002
```

This retains the locally best first slot. The second slot may use public ordering among locally credible candidates within the explicit regret limit. It still requires all gates, current artifact hashes and root/residual diversity. Only current `ALIGNED`/`WEAK` trust with enough public units and paired candidates may enable this. Candidate fingerprint, local authority, paired IDs/count and timestamp must match; `--trust-max-age-hours` defaults to 24. Stale/divergent/insufficient public evidence falls back to local ordering and explains why. Insufficient public trust does not block an otherwise valid local selection. Default regret limit is zero; choosing a wider limit is an explicit policy decision.

The result is a suggestion with `candidate_ids`, scores, reasons, exclusions and unfilled slots; it does not mark a candidate selected or execute a submission.

```sh
python scripts/analysis.py postmortem --ledger ledger.json --selected candidate-a,candidate-b
```

Retrospective output includes public/private Spearman, missing private scores, best selected/unselected candidate and the signed missed gain. It does not revise past choices or use private scores in prospective selection.

An optional `--target-score NUMBER` decomposes the signed target gap into the gap from the best evaluated model to the supplied target and regret from selecting a weaker evaluated model. `total_gap = model_frontier_gap + selection_regret`, oriented so positive means shortfall. A model better than the supplied target has negative frontier gap. These are descriptive model/frontier and adaptive-selection labels across evaluated candidates, not causal attribution or a claim about unobserved models. Missing selected/private/target scores produce null components.

## Local verification

```sh
python -m unittest discover -s tests -p test_analysis.py
```

The tests create temporary synthetic CSV/JSON/scorer fixtures, run real CLI subprocesses, and check metrics, exact alignment, weighted/nonadditive pooled results, bootstrap direction/CI/harm/replication policy, uncertainty, priors, groups, shuffled nulls, rank trust, ties, stale gate identities, local/public selection and retrospective results. No competition or external service is contacted.
