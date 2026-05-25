# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-25

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定。文档全部落定。**

**当前唯一主线：开跑前债务清零。** 用户硬约束（2026-05-25）：正式开 P1 跑实验前，不允许残留任何工程/口径债务。所有 specs 工程项（`docs/specs/开跑前编码指导_评测口径与提速.md`）+ PE1 收尾 + P0.5 锁超参 + 提交通道，必须全部清完并跑冒烟验证，才能开 P1。冲刺看板见下方「开跑前债务清零冲刺」，A→E 档按序推进。

**本轮新增执行约束（2026-05-25，本会话确认）**：
- `#17-#19` 全部基于当前特征现状推进，**不重建特征缓存 / 不重跑 feature build 来刷新列集**；仅在 `#19` 验收项里按看板要求执行一次全量构建计时。
- `#16` 验收已通过后，后续所有跑数默认统一走 `float32` 特征路径；若需要 `float64`，仅允许作为数值对照或排障手段，不能作为正式跑数口径。

## 当前口径收敛（指针表）

CLAUDE 不复述规则，只给指针；规则正文在 AGENTS，"为什么"在 NOTE。

| 口径 | 规则位置 |
|---|---|
| 两速评测结构（快车道 short+long+每日IC / 慢车道 expanding+Review） | AGENTS §4.2 |
| 各阶段评测口径（P1–P5） | AGENTS §4.4 |
| 多重比较 / 边界增益（+0.003 是地板不是采纳标准） | AGENTS §4.3 |
| make_decision 硬契约 + 单测 | AGENTS §4.6 |
| 单候选 per-profile 复核六步 | AGENTS §4.7 |
| 组合不可加（叠加必当新 spec 重跑） | AGENTS §4.5 |
| Target Mode 准入（最终回 raw `fret12`） | AGENTS §7.5 |
| alpha + winsorize 先扫后锁 | AGENTS §7.7 / §7.11 |
| 推理契约与交付约束 | AGENTS §十一 |

## 当前最优基线（P0 分段结果）

> 这次 P0 结果分两个 run 目录保存：`20260523_223257` 负责 short / medium / long，`20260524_220846` 负责串行补完 expanding。按范围分开记录，避免把不同 scope 混成一个单点。

| 范围 | experiment_id | protocol_corr_mean | protocol_stability_score | protocol_corr_min | decision |
|---|---|---:|---:|---:|---|
| short / medium / long aggregate | R02_ridge_legacy_plus_norm_core | 0.057463 | 0.041396 | 0.025214 | baseline |
| expanding only | R01_ridge_legacy_plus_core | 0.066095 | 0.053459 | 0.041181 | promote |

来源：`results/eval_protocol/20260523_223257`（short/medium/long）、`results/eval_protocol/20260524_220846`（expanding）。expanding 串行耗时分析见 `docs/P0运行耗时监控报告_20260525.md`。

## 任务看板

### ⭐ 开跑前债务清零冲刺【当前唯一焦点】

> **接手须知（无会话上下文也能实施）**：实施总规格 = `docs/specs/开跑前编码指导_评测口径与提速.md`（下称「编码指导」）+ `AGENTS.md`。每项已标：实施指针 / 关键约束 / 验收 / 依赖 / 状态。按 A→E 推进。依赖：#17 须等 #16 通过、#18 须等 #15 就绪。commit 纪律：一条一提交（编码指导总原则 3）。
>
> **接手第一步**：① 读 `AGENTS.md` + 本看板；② 验现状健康：`PYTHONPATH=src python tests/test_eval_protocol.py`、`PYTHONPATH=src python tests/test_experiment_runner.py`、`PYTHONPATH=src python tests/test_submission_pipeline.py`、`PYTHONPATH=src python -m unittest -v tests.test_feature_loader`、`PYTHONPATH=src python -m unittest -v tests.test_scheduler`（当前合计 18 条应全过）、`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick`（应 exit 0，run 目录含 leaderboard/manifest_snapshot/resolved_columns/config）；③ `#16` 已完成，下一步进入 **#17（D 档）**。当前工作区除 `meow/` 外，还包含 **#14、#15、#16 已完成但未提交** 的改动（见下）。

**A 档 评测口径代码化 —— ✅ 已完成（commit `2ebb5fb`）**
- ✅ #10 make_decision 硬契约 + 单测（编码指导§4 / AGENTS §4.6）
- ✅ #11 daily suite（short+long）并设默认（编码指导§1）
- ✅ #12 每日 IC 均值 + IC-IR 进 leaderboard（编码指导§5）
- ✅ #13 manifest_snapshot + resolved_columns.json 写盘
- 验证：quick suite 端到端冒烟通过 + 8 条单测（`tests/test_eval_protocol.py`）全过

**B 档 PE1 清理与归档 —— #14【✅ 已完成，待提交】**
- **完成时间**：2026-05-25 本轮接手已落地。
- **归档位置**：`.archive/20260525_b14_archive/`
- **已归档内容**：
  - `src/feat_engine.py`（FeatureBuilder 本体）
  - `experiments/legacy/run_516v3_norm_only.py`
  - `experiments/legacy/run_516v3_restricted.py`
- **代码清理结果**：
  - `src/experiment_runner.py` 已移除旧加载链：`load_split` / `load_raw_split` / `load_feature_split` / `_filter_features` / `self.builder` / `_split_cache` / `_raw_split_cache` / `_daily_feature_cache`
  - `_load_group_split` 保留为唯一正式入口；缺少 loader 时改为显式 `raise`
  - 兼容层仅保留 `self.loader` 属性，供协议层读取 `h5dir`，不再承载旧特征构建逻辑
- **验收结果**：
  - `PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick` 已在每次逐文件归档后通过，最近 run_id：`20260525_142943`
  - `PYTHONPATH=src python tests/test_eval_protocol.py` 8/8 通过
  - `src/` 内已无 `feat_engine` import；`experiments/legacy/` 仅剩 `p0_rolling_audit.py`
- **下一步**：进入 #15（winsorize 开关与分位配置）

**C 档 winsorize —— #15【✅ 已完成，待提交】**
- **完成时间**：2026-05-25 本轮接手已落地。
- **代码结果**：
  - `src/experiment_runner.py` 已接入 runner 级训练标签 winsorize 配置；默认 `enabled=True, q=(0.005, 0.995)`，且只作用于训练目标
  - `src/scheduler.py` / `src/eval_protocol.py` / `experiments/p0_eval_protocol.py` 已把该配置贯通到并行 worker 和结果 `config.json`
  - 新增单测 `tests/test_experiment_runner.py`，锁默认值 / 关闭开关 / 分位裁剪行为
- **真实数据闭环**：
  - 开启版：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_b15_quick_on_v2`
  - 关闭版：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_b15_quick_off_v1 --target-winsorize off`
  - 两轮都已成功跑通；开启版 worker 日志可见实际裁剪边界，关闭版日志明确 `Target winsorize: disabled`
- **当前观察（仅 quick 冒烟，不作采纳结论）**：
  - `R02` quick `protocol_corr_mean`：开启 0.032685，关闭 0.032791
  - 说明开关真实生效，但是否采用默认 `P0.5/P99.5` 仍留待 #18 在 short+medium 正式扫锁
- **下一步**：进入 #16（float32 验收）

**S 档 提交链收口 —— S1【✅ 已完成，已提交 `2710a1f`】**
- **完成时间**：2026-05-25 本轮接手已落地。
- **结果摘要**：
  - `meow/` 已移出嵌套仓库状态并纳入主仓库跟踪
  - `src/submission_pipeline.py` 已落地为正式提交桥接层：原始数据现算正式特征 + 复用 `experiment_runner` 训练/推理核心
  - `meow.py / feat.py / mdl.py` 已改为正式提交包装层，保持老师 `python meow.py` 入口不变
  - 提交通道显式不依赖 `data/features/` 持久化缓存；`mdl.py` 内部已把 `feature_dir` 指向不存在路径作为防误用保险丝
- **验收结果**：
  - 单测：`tests/test_submission_pipeline.py` 2/2 通过
  - 真实提交通道闭环：`cd meow && MEOW_TRAIN_START=20230601 MEOW_TRAIN_END=20230605 MEOW_EVAL_START=20230606 MEOW_EVAL_END=20230606 python meow.py` 已通过；本会话已复验通过
  - 日志可见正式特征现算、训练标签 winsorize、生效后的训练与评测摘要
- **规格文档**：`docs/specs/meow提交通道收口规格.md`
- **下一步**：回到 #16（float32 验收）

**D 档 expanding 提速**
- #16 特征 float32【✅ 已完成，待提交】（编码指导§2b；`src/feature_loader.py` / `src/experiment_runner.py` / `src/scheduler.py` / `src/eval_protocol.py`）
  - **完成时间**：2026-05-25 本轮接手已落地。
  - **代码结果**：
    - `FeatureLoader` 新增 `feature_dtype` 收口，默认特征列在加载后统一转 `float32`，并保留 `float64/None` 对照入口供数值验收复用
    - `ExperimentRunner` / `ParallelScheduler` / `EvaluationProtocolRunner` 已把 `feature_dtype` 透传到串行、并行与 `config.json`，避免主进程与 worker 口径漂移
    - 新增 `tests/test_feature_loader.py`，锁默认 `float32`、显式 `float64`、`None` 原样保留、非法 dtype 早失败四条约束
  - **硬验收结果**：
    - 同 spec 同 fold 数值对照：`R02_ridge_legacy_plus_norm_core`，`short_8d_2d` 首折（train `20230601~20230612`，val `20230614~20230615`），`val_corr_float32=0.04146936915522309`，`val_corr_float64=0.04146936915522309`，`|Δcorr|=0.0 < 1e-4`
    - 端到端冒烟：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_d16_quick_smoke_v2` 已通过
    - 回归测试：`PYTHONPATH=src python -m unittest -v tests.test_feature_loader tests.test_experiment_runner tests.test_submission_pipeline` 9/9 通过；`PYTHONPATH=src python tests/test_eval_protocol.py` 8/8 通过
  - **下一步**：进入 #17（关口提速）
- #17 关口提速（编码指导§2a/§2c；改 `src/scheduler.py` + `experiments/p0_eval_protocol.py`）：① 关口只评候选+基线 2 spec ② group 按 fold 序号成本均衡切分（不连续切）③ heavy 批次 2 worker。**#16 已验收通过，可开始实施 2 worker 路径。** 验收：2 spec / 2 worker 无 OOM、~30min。依赖 #16。
  - **本轮执行补充**：实现与验证均基于现有特征产物推进，不触发特征重建；正式关口跑默认 `feature_dtype=float32`。
  - **代码状态**：已落地 `suite=gate`（候选+基线 2 spec），并把 heavy profile 的 fold 切组改为按训练窗口成本贪心均衡到 2 组；新增 `tests/test_scheduler.py` 锁住“非连续切组 + 成本接近”约束。
  - **最小验收（已通过）**：`PYTHONPATH=src python -m unittest -v tests.test_scheduler` 1/1 通过；`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite gate --candidate-spec-id R03_ridge_legacy_plus_patch_summary --max-folds 2 --run-id 20260525_d17_gate_smoke_v1` 已通过，2 worker/2 group/4 job 正常完成，墙钟 `214.8s`，无 OOM。
  - **完整验收（未通过，已推后）**：
    - 全量 `expanding` 的 `2 spec / 2 worker` 关口验收未通过，当前这台 16GB Mac **不能采纳“双并行 expanding 尾段”** 作为正式关口跑法。
    - 最重单组（`expanding_40d_5d_g0`，fold `0/3/4/7/8/11`，最大训练窗 `95` 天）已用 `run_with_memory_guard.py` 做定点观测：`logs/memory_guard_20260525_d17_max_group.log` 记录到单组峰值 RSS `8.34 GB`。
    - 结合当前均衡切组结果，双并行尾段近似对应 `95 + 90 ≈ 185` 个训练日同时在内存中；按单组观测线性外推，当前 16GB Mac 无法稳定承受。
    - **当前阶段结论**：除“两个 `expanding` heavy group 同时跑、且落在 `90/95` 天尾段”这一最高压力场景外，这台 Mac 对其余已验证场景（如 `long` 双并行、`gate` 小规模 smoke、单个最重 group）基本可承受。
    - **处理决定**：`#17` 的“full expanding 2-worker 验收”推后，不阻塞其余工作；后续若换 32GB 级别机器，再重开该完整验收。

**E 档 跑实验收尾**
- #18 P0.5 alpha+winsorize 扫锁（AGENTS §7.7）：仅 R02、仅 short+medium；alpha {0.5,1,2,5,10,20} × clip {P0.5/P99.5, P1/P99, 不裁} 一起扫，各取平台中心锁死，固化进标准 ridge 路径；per-fold alpha 留 P4。依赖 #15。~15–20min。
  - **代码状态**：`ExperimentRunner / ParallelScheduler / EvaluationProtocolRunner / experiments/p0_eval_protocol.py / src/submission_pipeline.py` 已新增 `ridge_alpha` 正式透传，标准 ridge 主路径不再依赖硬编码 `2.0`；新增脚本 `experiments/p05_lock_ridge_alpha_and_winsorize.py` 承接批量扫描。
  - **最小验收（已通过）**：`PYTHONPATH=src python -m unittest -v tests.test_experiment_runner` 4/4 通过；`PYTHONPATH=src python experiments/p05_lock_ridge_alpha_and_winsorize.py --alphas 2 --winsor-labels clip_p005_p995 --max-folds 1 --run-id 20260525_e18_smoke_v1` 已通过，成功写出 `results/p05_alpha_winsorize/20260525_e18_smoke_v1_summary.csv`。
  - **正式扫描（已完成）**：
    - `caffeinate -i env PYTHONPATH=src python experiments/p05_lock_ridge_alpha_and_winsorize.py --run-id 20260525_e18_full_v1`
    - 汇总：`results/p05_alpha_winsorize/20260525_e18_full_v1_summary.csv`
  - **锁定结论**：
    - `winsorize` 正式默认锁为 **开启 + `P1/P99`**
    - `ridge_alpha` 继续锁为 **`2.0`**（平台区中部）
    - 依据：`clip_p01_p99` 的最佳 `protocol_corr_mean≈0.054181`，高于 `clip_p005_p995≈0.054123` 与 `clip_off≈0.053483`；而 `clip_p01_p99` 内部 `alpha=0.5~20` 的 `protocol_corr_mean` 波动仅约 `1.0e-5`
  - **回归验证**：
    - `PYTHONPATH=src python -m unittest -v tests.test_experiment_runner` 4/4 通过
    - `PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_e18_postlock_quick_v1` 已通过，日志确认默认 `target winsorize: True q=(0.010, 0.990)`、`ridge alpha: 2.0`
- #19 OOM 冒烟 + 全量构建计时：daily suite 跑一轮无 OOM/无 worker 异常；`python -m feature_store build --all` ≤20min。可叠 `run_with_memory_guard.py` + `caffeinate`。**除该计时验收本身外，不额外触发 feature build；daily 冒烟默认 `float32`。**
- #20 meow.py 提交通道（AGENTS §十一）：在一个 held-out 交易日跑通 `genFeatures→predict→forecast`，列对齐、每行有限、输出回 raw `fret12`、不依赖本地缓存/不可见统计量。**该项并入 S1 实施与验收，不再独立后置。** P1 前强制。

> 下方 PE0/PE1/P0.5 等节仅留背景；待办均已并入本冲刺看板。

### PE0 实验平台并发改造【已完成】

- 并发调度 + resume + 串并一致性 + OOM 修复均已合入。
- OOM 根因（备查）：`ExperimentRunner._split_cache` 按 fold 唯一 key 累积全量 DataFrame 不释放，expanding 末段单 entry ≈ 4GB，多 worker 总峰值 >50GB。修复：每折后 `clear()` + `gc.collect()`，n_workers 默认 4。

### PE1 特征管道重构【实施中】

规格：`docs/specs/特征管道重构规格.md`；交接：`docs/交接文档_特征管道重构_20260523.md`

- 当前状态：FeatureRegistry / FeatureStore / FeatureLoader 三件套首版 + 单测均通过（M1–M4）；主链路（trainer / scheduler / eval_protocol）已切到 FeatureLoader；9 个 stage 合并 462 列与旧管道一致。
- 调度与内存（已落地）：scheduler 改 light/heavy 两批（light 沿用 4 worker，heavy 降到 2）；`experiments/run_with_memory_guard.py` RSS 看门狗（软 12GB/30s + 硬 13GB 击杀）。16GB Mac 上 heavy 批次并发 ≤ 2。

剩余项（已并入上方冲刺看板，此处不重复列）：
- #13 manifest/resolved_columns 写盘 → ✅ 已完成
- #14 清理旧 cache/load 链 + `feat_engine` 归档（含 legacy 脚本，本会话决策）→ ✅ 已完成，待提交
- #15 训练目标 winsorize 开关 + 分位配置 + 真实数据闭环 → ✅ 已完成，待提交
- #19 OOM 冒烟 + 全量构建 ≤20min
- 「新旧管道 allclose 验证」**取消**：legacy 旧管道按 #14 归档后已无对照对象；新管道已由 P0 基线产出验证可用，无需再比。

### P0 扩展 rolling 评测体系【工程完成，基线已产出】

- 三层协议 + 4 profiles + 输出结构（fold_manifest / fold_metrics / profile_summary / leaderboard）+ baseline delta + make_decision + 主入口均完成。
- `daily` / `ridge` / `full` suite 可跑（基线见上"当前最优基线"）。

> P0.5（#18）、两速 suite 与 expanding 提速（#10 / #11 / #12 ✅ + #15 / #16 / #17）、提交通道（#20）的待办均已并入上方「开跑前债务清零冲刺」，此处不再重复，避免文档矛盾。

### P1 / P2 / P3 特征组验证【等上面就绪】

- [ ] P1：O1–O6（OFI）｜P2：T1–T4（trade impact）｜P3：C1–C3（条件动量/反转）
- 口径：快车道 short+long+每日IC 筛 → 候选过线才进 expanding 关口（AGENTS §4.4）；采纳门槛见 §4.6

### P3.5 少量交互项冲刺【P1–P3 有初步结论后】

- [ ] 仅高先验交互（`ofi_total×spread` 等 6 项，AGENTS §7.10）；scratch 起步，复用才进 registry

### P4 稳健模型比较【P3.5 后】

- [ ] Ridge / ElasticNet / Huber / 浅 ExtraTrees / 浅 HistGB；准入见 AGENTS §7.8
- [ ] expanding 只在 2–3 个决赛模型上跑（树更贵且增量技巧失效）

### P5 受限融合【最后】

- [ ] 只融合独立有效、与主分支非同质的分支；二层用 OOF；不比 backbone 稳就交 backbone（AGENTS §7.9）

### 交付演练【P5 之后、看 Final Holdout 之前】

- [ ] 输出对齐 raw `fret12`；`meow.py` 不依赖本地缓存 / 不可见统计量；提交口径与训练评测一致

## 工程状态

- [x] 目录与文档重组、代码解耦、PE0 OOM 修复、PE1 设计拍板
- [ ] PE1 实施收尾
- [ ] P0.5 + 两速 suite 落地 + expanding 提速 + 提交通道打通
- [ ] R02 一致性复现（run1 / run2 / run3）
