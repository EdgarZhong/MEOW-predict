# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-24

## 当前阶段目标

**方案 B 已固定**：以扩展 rolling validation 为内部选模型的主要依据，不再只看单次 val_corr。

当前首要任务：PE1 特征管道重构 → 重新运行 `--suite ridge --n-workers 4` 建立正式基线 → 推进 P1–P5。

## 当前口径收敛

### Target Mode 口径（2026-05-24 已拍板）

- 最终提交给老师评测的预测值，必须回到原始 `fret12` 尺度
- 当前允许继续推进的 target 路线：
  - `raw`
  - `interval_residual`（前提：预测阶段显式加回公共基线，最终输出 raw `fret12`）
  - `common_residual`
  - `soft_regime`（前提：最终 meta 输出直接对齐 raw `fret12`）
- 当前不再作为最终提交候选的路线：
  - `date_demean`
  - `interval_demean`
  - `rank target`
  - 任何无法在测试时不依赖真实 y 而恢复原始 `fret12` 的目标变换
- 后续如新增 target mode，必须先回答一个问题：**测试时能否仅依赖已知特征 / 已知 meta / 训练期固定参数，把预测值还原到 raw `fret12`？** 不能则不进入正式 rolling 主线

### 当前阶段执行纪律（2026-05-24 已拍板）

- `P1-P3`：只做特征组扩展，默认保持 `Ridge + raw`
- `P3.5`：允许少量高先验交互项冲刺，但不扩成大规模组合搜索
- `P4`：才进入传统模型比较和小范围调参
- `P5`：只做受限融合，不做自由 stacking
- Final Holdout 前必须有一次交付演练，确认最终输出 raw `fret12`、提交入口无泄漏、`meow.py` 可独立运行

### 评测严谨性与 alpha 口径（2026-05-24 已拍板）

本轮新增四条纪律，规则正文见 AGENTS.md，"为什么"见 NOTE.md：

- **多重比较 / 边界增益**（AGENTS §4.3）：单 profile 的 +0.003 低于一个标准误（≈0.006），边界增益（0.003–0.005）不在 Dev Rolling 直接 promote，交 11 月 Review Holdout；每阶段先固定候选集再跑，禁止边加边筛
- **profile 分层 + expanding 硬门槛**（AGENTS §4.4）：内层筛选只跑 short+medium，long 当负向 veto，**expanding 串行 + memory guard，是 promote 前必跑的硬门槛**
- **组合不可加**（AGENTS §4.5）：多个特征组叠加必须当新 spec 重跑验收，禁止用单组 delta 推算
- **Ridge alpha**（AGENTS §7.7）：进 P1 前一次性扫描确认后锁定 `alpha`（见下方 P0.5 任务），P1–P3 不逐 spec 调，per-fold 调参留 P4

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

### PE1：特征管道重构【实施中，M1-M2 已完成】

规格文档：`docs/specs/特征管道重构规格.md`
交接文档：`docs/交接文档_特征管道重构_20260523.md`

**目标**：解决 OOM 根因 + 支撑 P1–P5 快速特征迭代

- [x] 新建 `src/feature_registry.py`：FeatureRegistry 类 + 9 个 stage builder + status 字段
- [x] 新建 `src/feature_store.py`：FeatureStore 类 + 脏检测（code_hash + FEATURE_COMMON_VERSION）+ CLI
- [x] 新建 `src/feature_loader.py`：FeatureLoader 类 + key 对齐校验 + resolved_columns 记录
- [ ] 修改 `src/experiment_runner.py`：删 4 个 cache dict + load 系列方法，保留模型训练评估
- [x] 修改 `src/scheduler.py`：worker 改用 FeatureLoader
- [x] 修改 `src/trainer.py`：TabularTrainer 注入 FeatureLoader
- [ ] 修改 `src/eval_protocol.py`：运行时保存 manifest_snapshot + resolved_columns.json
- [ ] 迁移 `src/feat_engine.py`：builder 逻辑迁入 registry，旧文件归档到 .archive/
- [ ] 正确性验证：新旧管道同 spec 同日期的特征输出 allclose
- [ ] OOM 验证：`--suite ridge --n-workers 4` 全部 4 profile 完成无异常
- [ ] 全量特征首次构建：`python -m feature_store build --all` ≤ 20 min

**最新进展（2026-05-23 / M1）**：
- 已完成 `FeatureRegistry` 首版实现：9 个 stage builder、status、DAG、`resolve_groups()`、`code_hash()` 全部落地
- 已新增 `test_feature_registry.py`：验证拓扑序、downstream 闭包、group 列映射与旧 `FeatureBuilder.select_groups()` 兼容
- 当前验证结果：M1 测试通过，9 个 stage 合并后的总特征列集合与旧管道一致（462 列）

**最新进展（2026-05-23 / M2）**：
- 已完成 `FeatureStore` 首版实现：manifest、脏检测、拓扑构建、`build/status` CLI 全部落地
- 已新增 `test_feature_store.py`：覆盖首次构建、clean 检查、hash 失效传播、`common_version` 全量失效、定向 stage 重建
- 当前验证结果：`python test_feature_store.py` 通过；真实数据 smoke test 已跑通 `python -m feature_store build/status --dates 20230601,20230602`
- 当前环境没有 `pyarrow/fastparquet`，因此 `FeatureStore` 自动使用 `pickle_fallback` backend，目录结构与 `.parquet` 文件名保持不变

**最新进展（2026-05-23 / M3）**：
- 已完成 `FeatureLoader` 首版实现：无状态按组加载、raw h5 提供 `meta/target`、多 stage key 对齐校验、`last_load_info()` 记录 resolved columns / stages used
- 已新增 `test_feature_loader.py`：覆盖新旧管道数值一致性、resolved_columns 记录、错位 stage 异常检测
- 当前验证结果：`python test_feature_loader.py` 通过

**最新进展（2026-05-23 / M4）**：
- 已完成主链路切换首版：`TabularTrainer` 显式注入 `FeatureLoader`，`scheduler` worker 改为创建 `FeatureLoader` 直读 stage artifact，`eval_protocol`/`experiments/p0_eval_protocol.py` 已接入 `feature_dir`
- `ExperimentRunner` 的 `run_with_groups()` / `run_common_residual_reconstruction()` / `run_soft_regime_ensemble()` / `_evaluate_spec_on_fold()` 已统一通过 `_load_group_split()` 走 `FeatureLoader`
- 已新增 `test_m4_pipeline.py`：覆盖 `FeatureStore.build()` → `ExperimentRunner.run_with_groups()` → `TabularTrainer.run_fold()` → `scheduler._fold_group_worker()` 的最小端到端链路
- 当前验证结果：`python test_m4_pipeline.py`、`python test_feature_loader.py` 通过
- 剩余尾项：`ExperimentRunner` 的旧 `load_*` / cache 仍暂保留作兼容与回归基准，待后续瘦身清理

**最新进展（2026-05-24 / P0 OOM 复盘）**：
- 只以 `logs/特征管道改造后P0首跑.txt` 作为当前有效依据；旧日志对应旧缓存问题，不再作为本轮判断依据
- 新管道下 `short` 与 `medium` 已完整跑通；失败发生在后半程重任务并发阶段，而不是早期 profile
- 失败时真正仍在运行的重任务 group 只有 3 个：`long_40d_5d_g0`、`expanding_40d_5d_g0`、`expanding_40d_5d_g1`
- 这说明当前机器上并不是 `n_workers=4` 本身有问题，而是 **3 个 long/expanding 级别重 group 同时在飞就会 OOM**
- 调度层当前并发消费单位是 `group`，不是单个 fold/spec：worker 每次消费一个 `FoldGroup`，组内按 `fold -> spec` 串行执行
- 当前 `group` 切分规则仍是静态的：`target_group_size=7`；本次实际切分为：
  - `short_8d_2d`: 19 folds → 3 groups `[7, 7, 5]`
  - `medium_20d_5d`: 16 folds → 3 groups `[6, 6, 4]`
  - `long_40d_5d`: 6 folds → 1 group `[6]`
  - `expanding_40d_5d`: 12 folds → 2 groups `[6, 6]`
- 当前口径收敛：
  - 不优先投入复杂的“内存预算自适应调度器”设计
  - 先按简单硬限制处理：**进入 long / expanding 阶段后，总并发直接降到 2**；不再区分另一个在飞 group 是否也属于 heavy
  - 若 `2` 仍不稳，再把 `expanding` 单独按 `1` 跑
- 下一步优先级：
  1. 调整调度策略，避免 `long_g0 + expanding_g0 + expanding_g1` 同时提交；允许空闲 worker 保持空闲，不追求打满
  2. 以最小改动优先，不改训练/评测协议主体
  3. 目标是先消除 OOM，再考虑是否需要更细粒度的预算调度

**最新进展（2026-05-24 / P0 调度修正进行中）**：
- 已按最小改动原则修改 `src/scheduler.py`：group 仍是并发消费单位，但调度改为两批执行
  - `short / medium` 批次：保持原 `n_workers`
  - `long / expanding` 批次：总并发硬限制为 `2`
- 该实现刻意不引入复杂预算器，也不改 fold/spec 执行顺序；目标仅是阻止重任务阶段出现 3 个 group 同时在飞
- 已补充 `test_m4_pipeline.py` 回归用例，校验 light 批次沿用 `4` worker，而 heavy 批次降为 `2`

**最新进展（2026-05-24 / P0 内存保护）**：
- 已新增 `experiments/run_with_memory_guard.py`：通用可复用 RSS 看门狗包装器
- 当前确认口径：
  - 默认用于包裹长时间评测任务，而不是只做一次性命令
  - 当前先采用 **软阈值 12 GB 持续 30 秒** + **硬阈值 13 GB 立即击杀**
  - 击杀顺序：先 `SIGTERM`，宽限后再 `SIGKILL`
- 目标：在 `P0` 续跑 `long / expanding` 时，如果实际内存再次失控，优先杀任务，避免整机被 OOM 拖到重启

**最新进展（2026-05-24 / expanding 使用口径收敛）**：
- `expanding_40d_5d` 对最终泛化判断仍有价值，但**暂不纳入 P0-P3 日常正式并发主流程**
- 当前阶段的执行分层：
  - 日常初筛：`short + medium`
  - 候选负面 veto：`long`
  - promote 前复核 / 最终候选确认：`expanding`
- 当前证据：
  - `expanding_g0`、`expanding_g1` 单独运行时可稳定推进，单进程峰值内存未超过看门狗阈值
  - 两个 `expanding` group 放入并发池后会出现 worker 异常终止，因此现阶段不继续把它塞进正式并发流
- 当前最小修复口径：
  - 需要跑 `expanding` 时，**单独跑、串行跑**
  - 即把 `expanding` 视为慢速复核层，而不是日常主筛选层

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

### P0.5：alpha 一次性扫描与锁定【下一步即时任务，编码+实验】

P0 基线已固定在 `alpha=2.0`（`src/experiment_runner.py` 的 `fit_model`，前接 StandardScaler），但从未验证。进 P1 前先确认并锁定：

- [ ] 写一个轻量扫描入口：仅 R02 baseline、仅 short+medium 两个 profile，在 `{0.5,1,2,5,10,20}` 上跑 alpha
- [ ] 确认 2.0 是否落在平台区；若明显偏离则取平台中心作为锁定值
- [ ] 把锁定值固化进标准 ridge 路径，P1–P3 全程不再逐 spec 调 alpha
- [ ] per-fold alpha 调参推迟到 P4
- 预算：short+medium、单 spec、6 个 alpha，目标 ~15 min

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

### P3.5：少量交互项冲刺【等 P1–P3 有初步结论】

- [ ] 只做少量高先验交互项，不做组合爆炸
- [ ] 首批候选：
  - `ofi_total x spread`
  - `ofi_total x trade_activity`
  - `trade_pressure_qty x spread`
  - `lagret12 x ofi_total`
  - `lagret12 x order_pressure`
  - `trade_pressure_qty x regime_score`
- [ ] 先按 scratch 方式写在实验脚本里验证；若复用或进入正式对比，再迁入 registry

### P4：稳健模型比较【等 P1–P3 / P3.5 有结论】

- Ridge / ElasticNet / HuberRegressor / 浅 ExtraTrees / 浅 HistGB
- 有效信号先在线性模型验证，再上浅树
- [ ] 进入 P4 的前提：相对当前基线 `delta_corr >= 0.003`，稳定性不下降，且无新的强负 fold
- [ ] 当前默认不做大网格，只做小范围人工扫

### P5：受限融合【最后】

- 只融合在 rolling 下独立有效的信号组
- [ ] 进入融合池的分支必须先单独证明有效，且与主分支不是高度同质
- [ ] 若使用二层融合，必须使用 OOF prediction
- [ ] 若融合不比 backbone 更稳，提交 backbone

### 交付演练：Final Holdout 前最后检查【P5 之后】

- [ ] 最终输出确认对齐 raw `fret12`
- [ ] `meow.py` 不依赖 `data/features/`、本地缓存、不可见统计量
- [ ] 提交入口与训练评测口径一致
- [ ] 通过后才允许看 Final Holdout

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
