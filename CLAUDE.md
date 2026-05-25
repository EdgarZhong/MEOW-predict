# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-25

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定。** 当前主线，按顺序：

1. PE1 特征管道重构收尾（experiment_runner 瘦身 + 正确性/OOM 验证）
2. P0.5：alpha + winsorize clip 一次性扫描后锁定
3. 落地两速 suite（日常 short+long）+ expanding 提速（float32 + 2 worker）——实施规格见 `docs/specs/开跑前编码指导_评测口径与提速.md`
4. 打通 `meow.py` 提交通道（AGENTS §十一）后开 P1

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

### PE0 实验平台并发改造【已完成】

- 并发调度 + resume + 串并一致性 + OOM 修复均已合入。
- OOM 根因（备查）：`ExperimentRunner._split_cache` 按 fold 唯一 key 累积全量 DataFrame 不释放，expanding 末段单 entry ≈ 4GB，多 worker 总峰值 >50GB。修复：每折后 `clear()` + `gc.collect()`，n_workers 默认 4。

### PE1 特征管道重构【实施中】

规格：`docs/specs/特征管道重构规格.md`；交接：`docs/交接文档_特征管道重构_20260523.md`

- 当前状态：FeatureRegistry / FeatureStore / FeatureLoader 三件套首版 + 单测均通过（M1–M4）；主链路（trainer / scheduler / eval_protocol）已切到 FeatureLoader；9 个 stage 合并 462 列与旧管道一致。
- 调度与内存（已落地）：scheduler 改 light/heavy 两批（light 沿用 4 worker，heavy 降到 2）；`experiments/run_with_memory_guard.py` RSS 看门狗（软 12GB/30s + 硬 13GB 击杀）。16GB Mac 上 heavy 批次并发 ≤ 2。

剩余项：
- [ ] `experiment_runner.py` 瘦身：删旧 cache dict + load 系列方法（现暂留作回归基准）
- [ ] `eval_protocol.py`：运行时保存 manifest_snapshot + resolved_columns.json
- [ ] `feat_engine.py` builder 迁入 registry，旧文件归档 `.archive/`
- [ ] 正确性验证：新旧管道同 spec 同日期特征输出 allclose
- [ ] OOM 验证：日常 suite 完成无异常
- [ ] 全量构建 `python -m feature_store build --all` ≤ 20 min

### P0 扩展 rolling 评测体系【工程完成，基线已产出】

- 三层协议 + 4 profiles + 输出结构（fold_manifest / fold_metrics / profile_summary / leaderboard）+ baseline delta + make_decision + 主入口均完成。
- `--suite ridge` 已分段跑通（见上"当前最优基线"）。

### P0.5 alpha + winsorize 一次性扫描与锁定【下一步即时任务】

进 P1 前一次性扫定、然后锁死（P1–P3 不再逐 spec 调）：
- [ ] 轻量扫描入口：仅 R02、仅 short+medium（标定刻意用 medium 折多，AGENTS §7.7 有意例外）
- [ ] alpha 扫 `{0.5,1,2,5,10,20}`、winsorize clip 扫 `{P0.5/P99.5, P1/P99, 不裁}`，一起扫
- [ ] 各取平台中心锁定，固化进标准 ridge 路径；per-fold alpha 留 P4
- 预算：~15–20 min

### 落地两速 suite + expanding 提速【与 P0.5 并行，编码】

实施规格见 `docs/specs/开跑前编码指导_评测口径与提速.md`（两边都不直接改代码，由 coding agent 实施）：
- [ ] 日常 suite 默认 profiles = short + long（medium 移出日常，可 ad-hoc 单跑）
- [ ] expanding 关口跑法：候选 vs 基线 2 spec + float32 + 2 worker 成本均衡切分（**2 worker 须 float32 数值验收通过后才开，否则维持串行**；明确不做增量正规方程）
- [ ] make_decision 硬契约 + 3 条单测（AGENTS §4.6）
- [ ] 每日 IC、IC-IR 并进 leaderboard 主视图（零额外成本）
- [ ] 训练目标 winsorize 实现（AGENTS §7.11，与 P0.5 一起扫锁）

### 提交通道打通【P1 前强制】

- [ ] 在一个 held-out 交易日跑通 `meow.py`：`genFeatures → predict → forecast` 列对齐、每行有限（AGENTS §十一）

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
