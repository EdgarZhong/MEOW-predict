# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-22

## 当前阶段目标

**方案 B 已固定**：以扩展 rolling validation 为内部选模型的主要依据，不再只看单次 val_corr。

当前首要任务：运行 `p0_eval_protocol.py --suite ridge` 建立正式基线，再依次推进 P1–P5。

## 评测体系（已完成，P0 工程部分）

### 新增文件

| 文件 | 职责 |
|---|---|
| `src/eval_protocol.py` | 核心评测协议模块（profiles/fold构造/leaderboard） |
| `experiments/p0_eval_protocol.py` | P0 主入口，重跑所有历史实验建立基准 |

### Rolling Profiles（4 个）

| profile | mode | train | val | step | embargo |
|---|---|---|---|---|---|
| short_8d_2d | sliding | 8d | 2d | 5d | 1d |
| medium_20d_5d | sliding | 20d | 5d | 5d | 1d |
| long_40d_5d | sliding | 40d | 5d | 10d | 1d |
| expanding_40d_5d | expanding | 40d+ | 5d | 5d | 1d |

### 验收状态

- [x] fold 构造逻辑验证：train_end ≤ embargo_start < val_start ✓
- [x] max_folds=None 不截断（原默认 5 已改） ✓
- [x] fold_manifest / fold_metrics / profile_summary / leaderboard 输出结构 ✓
- [x] baseline delta + make_decision 自动判断 ✓
- [ ] **待运行**：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge` 建立正式基线

### 运行命令

```bash
# 快速验证（2折+short profile，约2分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

# Ridge baseline 建立（全量，约20-40分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge

# 含 review holdout（11月）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge --include-review-holdout
```

## 当前最优基线（旧口径，待用新协议复现）

| 实验名 | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |
|---|---:|---:|---:|---:|
| R02_ridge_legacy_plus_norm_core | 0.047976 | 0.014975 | 0.032599 | 0.037494 |

来源：`results/ridge_enhance_results.csv`（5折旧口径），**需用新协议重新复现后替换**

## 任务看板（P0–P5）

### P0：扩展 rolling 评测体系【工程完成，待运行】

- [x] `src/eval_protocol.py` 三层评测协议实现
- [x] 四个 rolling profiles（short/medium/long/expanding）
- [x] fold_manifest / fold_metrics / profile_summary / leaderboard 输出
- [x] baseline delta + make_decision 自动判断
- [x] `experiments/p0_eval_protocol.py` 主入口（quick/ridge/full 三种 suite）
- [x] `experiment_runner.py` max_folds 默认值改为 None
- [ ] 运行 `--suite ridge` 跑通并记录正式基线指标
- [ ] 用新口径更新 CLAUDE.md 的"当前最优基线"表格

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
