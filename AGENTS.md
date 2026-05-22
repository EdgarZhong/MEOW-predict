# AGENTS.md — 开发规范与协作约定

## 一、文档职责分工

| 文档 | 职责 |
|---|---|
| `README.md` | 稳定事实：项目定位、目录结构、运行方式、关键文档入口 |
| `AGENTS.md` | 规则、流程、原则、协作约束（本文件） |
| `CLAUDE.md` | 动态：当前阶段任务看板、进度、决策 |
| `docs/实验记录.md` | 所有实验结果的唯一历史记录 |

## 二、禁止事项

- 禁止只看单次 val_corr 就下结论，必须看扩展 rolling 结果
- 禁止在特征/归一化中使用未来信息（rolling / EMA / zscore 只能用历史数据）
- 禁止用验证集真实 y 的均值、rank、分位数做还原（泄漏）
- 禁止在同一验证集上反复调参后汇报单次最高分
- 禁止把 quick/2折 结果与正式 5 折 rolling 结果直接并列比较
- 禁止删除文件，旧文件一律移到 `.archive/`
- 禁止提交 `data/`、`results/`、`*.log`、`*.out`、`*.err`

## 三、信号验收标准

每个新信号组通过 rolling 验证的最低标准：

```
rolling_corr_mean  提升 ≥ 0.003（相对当前基线）
stability_score    不下降
rolling_corr_min   不明显变差（不出现新的强负 fold）
MSE / R²           不明显恶化
```

如果某一类特征只让一个 fold 变好，其他 fold 变差，则放弃。

## 四、实验开发 SOP

1. 明确要改的一个变量（特征组 / 目标变换 / 模型 / 窗口）
2. 在 `experiments/` 下新建或复用对应阶段脚本（p1_ofi_validation.py 等）
3. 用 P0 确立的标准 rolling 口径评测
4. 把结果写入 `docs/实验记录.md`，标注 `experiment_id / date / feature_set / model / split / seed / metrics / notes`
5. 提交（遵循第六节 commit 规范）
6. 与基线比较，只改一个变量

## 五、模型优先级

```
第一优先：Ridge
第二优先：ElasticNet / HuberRegressor
第三优先：浅 ExtraTrees（max_depth≤5）/ 浅 HistGradientBoosting
第四优先：受限融合（仅当 P1-P3 有独立有效信号后）
禁止：Transformer / LSTM / 复杂 MLP / 自由 stacking（当前阶段）
```

新信号必须先在 Ridge 验证有效，再考虑上浅树。

## 六、特征工程规范

- 特征按信号族组织，每族独立评估贡献
- 特征命名前缀对应信号族：`ofi_`、`trade_impact_`、`momentum_`、`regime_`、`cross_`
- 所有滑动计算（rolling / EMA / lag）只能用 `shift(1)` 及以前的数据
- 归一化策略：rolling zscore / cross-section rank / stock-level z-score，禁止全样本归一化

## 七、三层评测体系（方案 B）

```
第一层：扩展 rolling validation  → 内部选模型的主要依据
第二层：11 月 holdout           → 复核
第三层：12 月 final holdout     → 尽量少看，模拟老师隐藏集
```

## 八、Git 提交规范

### 8.1 粒度原则

- 每次 commit 代表一个明确的变更：特征组、目标变换、模型、split 逻辑、评测逻辑
- 不混提：特征工程 + 模型调参 + 文档清理禁止一次提交

### 8.2 提交格式

```
feat:   新功能或新特征
fix:    修复 bug
exp:    实验结果记录
docs:   文档更新
refact: 重构（不改行为）
```

示例：
```
feat: add OFI multi-level features (bid/ask/total)
exp: P1 OFI rolling audit, O3 best rolling_corr_mean=0.051
fix: prevent target leakage in interval_demean normalization
```

### 8.3 不可提交的内容

- `data/`（原始数据）
- `results/`（中间结果 CSV）
- `*.log / *.out / *.err`
- `__pycache__/`

### 8.4 每次实验 commit 必须记录

`experiment_id / date / feature_set / target_type / model / split_config / seed / rolling_corr_mean / rolling_corr_std / stability_score / notes`

## 九、目录约定

```
src/              核心模块（特征、模型、评估、数据加载）
experiments/      实验入口脚本（按 P 阶段命名）
experiments/legacy/  历史脚本（可运行，不主动修改）
results/          实验结果 CSV（gitignored）
data/             原始 .h5 数据（gitignored）
docs/             文档
docs/archived/    历史快照（只读）
.archive/         废弃文件（gitignored）
```
