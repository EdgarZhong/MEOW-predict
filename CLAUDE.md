# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-22

## 当前阶段目标

**方案 B 已固定**：以扩展 rolling validation 为内部选模型的主要依据，不再只看单次 val_corr。

当前首要任务：完成 P0（扩展 rolling 评测体系建立），再依次推进 P1–P5。

## 当前最优基线

| 实验名 | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |
|---|---:|---:|---:|---:|
| R02_ridge_legacy_plus_norm_core | 0.047976 | 0.014975 | 0.032599 | 0.037494 |

来源：`results/ridge_enhance_results.csv`（5折正式 rolling）

## 任务看板（P0–P5）

### P0：扩展 rolling 评测 + 训练窗口敏感性【最高优先】

- [ ] `experiments/p0_rolling_audit.py` 实现标准三层评测：
  - 第一层：扩展 rolling（内部选模型主依据）
  - 第二层：11 月 holdout 复核
  - 第三层：12 月 final holdout（尽量少碰）
- [ ] 固定 R02 backbone，扫描 `max_train_days ∈ {4, 8, 10, 20, 40, 80, expanding}`
- [ ] 统一 rolling 指标：`rolling_corr_mean / std / min`，`rolling_mse_mean`，`rolling_r2_mean`，`daily_corr_mean`
- 完成标准：R02 可在同口径下稳定复现

### P1：OFI 动态订单流验证【等 P0 完成】

- [ ] OFI 特征已在 `src/feat_engine.py`（FeatureBuilder）中实现
- [ ] 在 P0 同口径 rolling 下验证 O1–O6 实验组
- 通过标准：`rolling_corr_mean` 提升 ≥ 0.003，`stability_score` 不下降

### P2：成交冲击 trade impact 验证【等 P0 完成】

- [ ] 在 P0 同口径下验证 T1–T4 实验组
- 通过标准：同 P1

### P3：条件动量 / 条件反转验证【等 P0 完成】

- [ ] 在 P0 同口径下验证 C1–C3 实验组
- 通过标准：同 P1

### P4：稳健模型比较【等 P1–P3 有结论】

- Ridge / ElasticNet / HuberRegressor / 浅 ExtraTrees / 浅 HistGB
- 有效信号先在线性模型验证，再上浅树

### P5：受限融合【最后】

- 只融合在 rolling 下独立有效的信号组

## 重要约束

- 所有 rolling / EMA / zscore 只能用当前及历史信息，禁止前视泄漏
- 所有调参只在 rolling 内部做
- final holdout（12月）尽量少看
- 判断标准：`rolling_corr_mean` 提升 ≥ 0.003，`stability_score` 不下降，`rolling_corr_min` 不明显变差

## 工程状态

- [x] 阶段一：目录与文档重组（2026-05-22）
- [x] 阶段二：代码解耦（FeatureBuilder 抽取到 feat_engine.py）
- [ ] P0 实验脚本完成并跑通
- [ ] R02 一致性复现（run1 / run2 / run3）
