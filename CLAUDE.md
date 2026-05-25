# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-26

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定；开跑前债务清零（#10–#20）已全部完成。**

下一步主线：**先把 P0 落实到位（基线口径 + 定位确认），再正式开 P1（OFI 特征验证）。** P0 是在「PE1 特征管道重构之后、本顾问会话之前」跑出的，4 个 profile 全跑过；如何理解 P0 的定位、以什么口径把它作为 P1 的对照基线，是开 P1 前的待议项，确认后落档本节。

## 当前口径收敛（指针表）

CLAUDE 不复述规则，只给指针；规则正文在 AGENTS，"为什么"在 NOTE。

| 口径 | 规则位置 |
|---|---|
| 两速评测结构（快车道 short+long+每日IC / 慢车道 expanding+Review） | AGENTS §4.2 |
| 各阶段评测口径（P1–P5） | AGENTS §4.4 |
| 多重比较 / 边界增益（+0.003 是地板不是采纳标准） | AGENTS §4.3 |
| make_decision 硬契约 + 单测 | AGENTS §4.6 |
| 单候选 per-profile 复核六步 | AGENTS §4.7 |
| 三层 Holdout 纪律（Dev / Review / Final） | AGENTS §4.8 |
| 组合不可加（叠加必当新 spec 重跑） | AGENTS §4.5 |
| Target Mode 准入（最终回 raw `fret12`） | AGENTS §7.5 |
| alpha + winsorize 已扫锁（开启 + P1/P99；alpha=2.0） | AGENTS §7.7 / §7.11 |
| 跑命令前自查清单（含 macOS expanding 必须 `--n-workers 1`） | AGENTS §5.1 |
| expanding 关口串/并行运行口径 | 编码指导 §2c |
| 推理契约与交付约束 | AGENTS §十 |

## 当前最优基线（P0 分段结果，定位待确认）

> 结果分两个 run 目录保存，按范围分开记录，避免把不同 scope 混成一个单点。**P0 定位与基线口径如何落实，见「当前阶段目标」中的待议项。**

| 范围 | experiment_id | protocol_corr_mean | protocol_stability_score | protocol_corr_min | decision |
|---|---|---:|---:|---:|---|
| short / medium / long aggregate | R02_ridge_legacy_plus_norm_core | 0.057463 | 0.041396 | 0.025214 | baseline |
| expanding only | R01_ridge_legacy_plus_core | 0.066095 | 0.053459 | 0.041181 | promote |

来源：`results/eval_protocol/20260523_223257`（short/medium/long）、`20260524_220846`（expanding）。耗时分析见 `docs/P0运行耗时监控报告_20260525.md`。

## 任务看板

### 开跑前债务清零冲刺（#10–#20）【✅ 全部完成】

A–E 档 + S 档全部落地并验收；详细实施与验收留在 git 历史与 `docs/specs/开跑前编码指导_评测口径与提速.md`。结论摘要：

- **评测口径代码化**：make_decision 硬契约 + 单测、daily/gate suite、每日 IC-IR 进 leaderboard、manifest_snapshot/resolved_columns 写盘（commit `2ebb5fb`）。
- **PE1 收尾**：旧加载链 + `feat_engine` + legacy 脚本归档至 `.archive/`；`FeatureLoader` 为唯一正式特征入口。
- **口径锁定**：训练标签 winsorize = 开启 + P1/P99；ridge alpha = 2.0；特征 dtype = float32（P0.5 扫描结论，commit `adf65ad`）。
- **提交通道**：`meow.py` 正式特征现算、复用 runner 训练/推理核心、不依赖本地缓存（commit `2710a1f`）。
- **expanding 提速**：gate suite（候选 vs 基线）+ 成本均衡切组；**macOS 标准跑法 = 单 worker 串行**（编码指导 §2c / AGENTS §5.1）。
- 当前 19 条单测全过（`test_eval_protocol` 8 + `test_experiment_runner` 4 + `test_feature_loader` 4 + `test_scheduler` 1 + `test_submission_pipeline` 2）。

### P1 / P2 / P3 特征组验证【P0 落实后启动】

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

## 已完成基建（备查）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复（根因：`ExperimentRunner._split_cache` 按 fold 累积全量 DataFrame 不释放，已每折 `clear()` + `gc.collect()`）。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader 三件套 + 单测；主链路已切 FeatureLoader；9 个 stage 合并 462 列。规格见 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + 输出结构（fold_manifest / fold_metrics / profile_summary / leaderboard）+ baseline delta + make_decision + 主入口；daily/gate/ridge/quick/full suite 可跑。
