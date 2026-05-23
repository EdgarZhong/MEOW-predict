# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-23

## 当前阶段目标

**方案 B 已固定**：以扩展 rolling validation 为内部选模型的主要依据，不再只看单次 val_corr。

当前首要任务：PE1 特征管道重构 → 重新运行 `--suite ridge --n-workers 4` 建立正式基线 → 推进 P1–P5。

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

# Ridge baseline 建立（全量，推荐 n-workers=4，约30-50分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge --n-workers 4

# 含 review holdout（11月）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge --n-workers 4 --include-review-holdout
```

> **注意**：16 GB Mac 不得使用 `--n-workers 8`，会触发 OOM 重启（已修复，默认改为 4）。

## 当前最优基线（旧口径，待用新协议复现）

| 实验名 | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |
|---|---:|---:|---:|---:|
| R02_ridge_legacy_plus_norm_core | 0.047976 | 0.014975 | 0.032599 | 0.037494 |

来源：`results/ridge_enhance_results.csv`（5折旧口径），**需用新协议重新复现后替换**

## 任务看板（PE0 + P0–P5）

### PE0：实验平台并发改造【已完成，OOM 修复已合入】

规格文档：`docs/specs/实验平台架构设计.md`

- [x] 新建 `src/trainer.py`：`BaseTrainer` ABC + `TabularTrainer`
- [x] 新建 `src/scheduler.py`：`ParallelScheduler` + `_fold_group_worker`
- [x] 改造 `src/eval_protocol.py`：`run_full_protocol` 接入 `ParallelScheduler`
- [x] 新增 `FoldData` / `FoldResult` dataclass
- [x] Resume 机制：已完成 job 自动跳过（验证通过）
- [x] 串行/并行结果一致性验证通过
- [x] **OOM 修复**（2026-05-22）：`_fold_group_worker` 每折后清空 `_split_cache`；默认 `n_workers` 改为 4
- [ ] 验收：`--suite ridge --n-workers 4` ≤50 min（待运行完整基线确认）

**OOM 根因记录**：`ExperimentRunner._split_cache` 按 `(train_dates_tuple, max_days)` 缓存 concat 后的全量 DataFrame，rolling fold 场景下每折 key 唯一、永不命中也不释放，expanding 最后几折单条 entry ≈ 4 GB，7 折一组累计 >20 GB；8 workers 并行后总峰值超过 50 GB 导致 OOM。修复：每折结束调用 `runner._split_cache.clear()` + `gc.collect()`，保留 `_daily_feature_cache`（避免重复 IO）。

### PE1：特征管道重构【实施中，M1 已完成】

规格文档：`docs/specs/特征管道重构规格.md`
交接文档：`docs/交接文档_特征管道重构_20260523.md`

**目标**：解决 OOM 根因 + 支撑 P1–P5 快速特征迭代

- [x] 新建 `src/feature_registry.py`：FeatureRegistry 类 + 9 个 stage builder + status 字段
- [ ] 新建 `src/feature_store.py`：FeatureStore 类 + 脏检测（code_hash + FEATURE_COMMON_VERSION）+ CLI
- [ ] 新建 `src/feature_loader.py`：FeatureLoader 类 + key 对齐校验 + resolved_columns 记录
- [ ] 修改 `src/experiment_runner.py`：删 4 个 cache dict + load 系列方法，保留模型训练评估
- [ ] 修改 `src/scheduler.py`：worker 改用 FeatureLoader
- [ ] 修改 `src/trainer.py`：TabularTrainer 注入 FeatureLoader
- [ ] 修改 `src/eval_protocol.py`：运行时保存 manifest_snapshot + resolved_columns.json
- [ ] 迁移 `src/feat_engine.py`：builder 逻辑迁入 registry，旧文件归档到 .archive/
- [ ] 正确性验证：新旧管道同 spec 同日期的特征输出 allclose
- [ ] OOM 验证：`--suite ridge --n-workers 4` 全部 4 profile 完成无异常
- [ ] 全量特征首次构建：`python -m feature_store build --all` ≤ 20 min

**最新进展（2026-05-23 / M1）**：
- 已完成 `FeatureRegistry` 首版实现：9 个 stage builder、status、DAG、`resolve_groups()`、`code_hash()` 全部落地
- 已新增 `test_feature_registry.py`：验证拓扑序、downstream 闭包、group 列映射与旧 `FeatureBuilder.select_groups()` 兼容
- 当前验证结果：M1 测试通过，9 个 stage 合并后的总特征列集合与旧管道一致（462 列）

**设计决策记录（2026-05-23 拍板）**：
- target/meta 来源：FeatureLoader 从 raw h5 读取
- 动态列名：build 后扫描实际输出列写入 manifest
- TabularTrainer：保留，注入 FeatureLoader
- 设计宪法：AGENTS.md 精简版（第二节）+ NOTE.md 完整背景
- stage status：装饰器声明 promoted/candidate/archived
- 公共函数版本：FEATURE_COMMON_VERSION 手动递增
- 实验可追溯：resolved_columns.json 快照
- loader 安全：多 stage 拼接时 key 对齐校验

### P0：扩展 rolling 评测体系【工程完成，待 PE1 后运行】

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
- [x] PE0 并发平台 OOM 修复（2026-05-22）：`_split_cache` 跨折清理 + n_workers 默认 4
- [x] PE1 设计完成（2026-05-23）：规格文档 + 交接文档 + 4 个设计决策拍板 + GPT 审计反馈整合
- [ ] **PE1 实施**：特征管道重构（FeatureRegistry + FeatureStore + FeatureLoader）
- [ ] P0 实验脚本跑通（`--suite ridge --n-workers 4`，依赖 PE1）
- [ ] R02 一致性复现（run1 / run2 / run3）
