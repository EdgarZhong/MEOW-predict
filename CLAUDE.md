# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-26

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定；开跑前债务清零（#10–#20）已全部完成；基线已用锁定口径 relock 并锁定为单一 R02。**

下一步主线：**正式开 P1（OFI 特征验证，O1–O6）。** 基线已在锁定口径（winsorize 开 + P1/P99、ridge alpha=2.0、float32）下重建为单一 R02，与 P1 候选同尺（见「当前最优基线」段）。P1/P2/P3 候选 spec 已写死在 `src/eval_protocol.py`（O/T/C 系列，均为 R02 + 新 group），底层 stage 在 PE1 已实现并注册——P1 是「候选过 rolling 关 + 对 R02 判 delta」，不是从零造特征。

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

## 当前最优基线（P0 relock，锁定口径）

> 锁定口径（训练标签 winsorize 开 + P1/P99、ridge alpha=2.0、特征 float32）重建，与 P1 候选同尺；快车道 = short+long（medium 已移出日常）。**单一基线 = R02_ridge_legacy_plus_norm_core，快慢两速统一采用。**

| 范围 | experiment_id | protocol_corr_mean | protocol_stability_score | protocol_corr_min | decision |
|---|---|---:|---:|---:|---|
| 快车道 short+long | R02_ridge_legacy_plus_norm_core | 0.057168 | 0.039703 | 0.019425 | baseline |
| expanding 关口 | R02_ridge_legacy_plus_norm_core | 0.062392 | 0.047159 | 0.033476 | baseline |

来源：`results/eval_protocol/20260526_p0relock_daily_v1`（快车道）、`20260526_p0relock_gate_v1`（expanding 关口）。

定基理由：① 快车道 R02 仍居首（领先 R01 +0.0031，且 R02 全折正、R01 short 段 corr_min=-0.046）；② expanding 关口 R01 仅领先 R02 +0.0014，落在一个标准误 ~0.006 内（噪声平手），未过 +0.003 地板；③ R02 在任、norm 特征利于外推。R01 此前 +0.003 的 expanding 边际在 winsorize 锁定后收窄为 +0.0014，不再成立。

## 任务看板

### 开跑前债务清零冲刺（#10–#20）【✅ 全部完成】

A–E 档 + S 档全部落地并验收；详细实施与验收留在 git 历史与 `docs/specs/开跑前编码指导_评测口径与提速.md`。结论摘要：

- **评测口径代码化**：make_decision 硬契约 + 单测、daily/gate suite、每日 IC-IR 进 leaderboard、manifest_snapshot/resolved_columns 写盘（commit `2ebb5fb`）。
- **PE1 收尾**：旧加载链 + `feat_engine` + legacy 脚本归档至 `.archive/`；`FeatureLoader` 为唯一正式特征入口。
- **口径锁定**：训练标签 winsorize = 开启 + P1/P99；ridge alpha = 2.0；特征 dtype = float32（P0.5 扫描结论，commit `adf65ad`）。
- **提交通道**：`meow.py` 正式特征现算、复用 runner 训练/推理核心、不依赖本地缓存（commit `2710a1f`）。
- **expanding 提速**：gate suite（候选 vs 基线）+ 成本均衡切组；**macOS 标准跑法 = 单 worker 串行**（编码指导 §2c / AGENTS §5.1）。
- 当前 23 条单测全过（`test_eval_protocol` 12 + `test_experiment_runner` 4 + `test_feature_loader` 4 + `test_scheduler` 1 + `test_submission_pipeline` 2）；另 `tests/test_feature_store.py`（PE1 M2 FeatureStore 回归，脚本式，`PYTHONPATH=src python` 单独跑）由根目录搬入归位。

### P1 / P2 / P3 特征组验证【就绪，可启动】

- [ ] P1：O1–O6（OFI）｜P2：T1–T4（trade impact）｜P3：C1–C3（条件动量/反转）
- 候选 spec 已写死在 `src/eval_protocol.py`（O 系 145–150 / T 系 152–155 / C 系 157–159），均为 R02 + 新 group
- daily 已支持 `--spec-ids` 选候选（baseline 自动并入算 delta，AGENTS §5/§5.1）
- P1 启动命令（**待运行**，run-id `20260526_p1_ofi_daily_v1`）：
  `--suite daily --spec-ids O1_R02_plus_ofi_raw O2_R02_plus_ofi_dynamic O3_R02_plus_ofi_rank O4_R02_plus_ofi_safe O5_R02_plus_ofi_raw_dynamic O6_R02_plus_all_ofi`
- 口径：快车道 short+long+每日IC 筛 → 过线候选再逐个进 expanding 关口（`--suite gate --candidate-spec-id <ID> --n-workers 1`，AGENTS §4.4）；采纳门槛见 §4.6

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
