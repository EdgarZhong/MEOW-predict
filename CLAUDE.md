# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-26

## 待办队列（下一会话执行）

> 真正跑实验在**新会话**。每项流程：看门狗+串行启动（命令模板 AGENTS §5.1 item5）→ `PYTHONPATH=src python experiments/analyze_run.py --run-id <id>` 多角度看 + 带专业判断（**不只认 decision 标签**，§4.6/§4.7）→ 落档本文件（§一记录规范）→ 测好就 commit + 同步必要文档。状态锚点：`results/eval_protocol/<run-id>/leaderboard.csv` 存在=该项完成。**自动作业止于 P4（模型选型）前**，holdout 不碰。缺的脚本/group/特征实现按需自行编码（已授权）。

| # | 任务 | 关键参数 |
|---|---|---|
| ✅Q1 | P3 条件动量（C1 本体 / C2 交互）— **判负**：C1 过拟合、C2 小而真但低于地板；C2 交互信号留作 Q3/Q4 抓手（详见 P3 结论） | run `20260526_p3_condmom_daily_v1` 完成 |
| Q2 | alpha 敏感性（验 O4·T1 是否被欠正则埋） | spec `O4_R02_plus_ofi_safe T1_R02_plus_trade_impact`，`--ridge-alpha 5` 与 `20` 两跑 |
| Q3 | P3.5 高先验交互（§7.10）：缺的交互特征/group 先建（scratch vs 注册自定） | — |
| Q4 | 跨族组合（§4.5：各族最好的并成新 spec 重跑，组合不可加） | — |
| Q5 | 综合 P1–P3.5 + 对 P4 的建议，停在 P4 前交回用户 | 纯分析+落档 |

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定；P1（OFI）、P2（成交冲击）、P3（条件动量）均已跑完判负。**

当前主线：**特征侧穷尽（P3 条件动量✅ → alpha 敏感性 → P3.5 交互 → 跨族组合），做完停在 P4 前。** 队列见上，下一步 Q2。P1/P2/P3-C1 根因一致：基线 R02 的 `norm_core` 已含横截面标准化版本，单族线性追加边际≈0 或仅过拟合（详见下方各阶段结论）。**P3 新增信号**：C2 交互列（lagret×{ofi/spread/…}）是 P1 以来最干净候选——小而真（Δ+0.0019、跨 profile 同向、daily IC 双升），证明交互项比单族线性追加更有戏，已转为 Q3/Q4 抓手。评测口径已从"自动判卷"改为"诊断仪表盘+人工多角度判断"（AGENTS §4.6）。

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

### P1 / P2 / P3 特征组验证

- [x] **P1：O1–O6（OFI）— 判负，无候选进 expanding。**
- [x] **P2：T1–T4（trade impact）— 判负，无候选进 expanding。**
- [x] **P3：C1/C2（条件动量）— 判负（C1 数据饥渴过拟合 / C2 小而真但低于地板），无候选进 expanding；C2 交互信号留作 Q3/Q4 抓手。**
- 候选 spec 已写死在 `src/eval_protocol.py`（O 系 145–150 / T 系 152–155 / C 系 157–159），均为 R02 + 新 group
- daily 已支持 `--spec-ids` 选候选（baseline 自动并入算 delta，AGENTS §5/§5.1）
- 口径：快车道 short+long+每日IC 筛 → 过线候选再逐个进 expanding 关口（`--suite gate --candidate-spec-id <ID> --n-workers 1`，AGENTS §4.4）；采纳门槛见 §4.6

**P1 结论（daily run `20260526_p1_ofi_daily_v1`，2026-05-26）：** 6 个 OFI 候选相对 R02 全部 reject，最佳仅 O4 `ofi_safe` 的 Δcorr=+0.0016（< +0.003 地板），无候选进 expanding。
- 结果路径：`results/eval_protocol/20260526_p1_ofi_daily_v1/`（leaderboard.csv 全指标）。
- 根因：基线 R02 的 `norm_core` 已含横截面标准化 OFI（`ofi_*_cs_z`，来自 cross_z）——OFI 中最稳健可外推的部分基线已吃掉；O 系仅再加原始/动态/depth 等更噪变体，边际≈0。
- ⚠️ 待清理：O6 `all_ofi`（groups=`ofi_safe+ofi_rank`）与 O4 逐位相同，因 `ofi_safe` 已是 OFI 列全集（ofi stage 全列 + cross 的 ofi cs_z/cs_rank），`ofi_rank` 冗余。O6 命名"all OFI"误导，建议后续修正或删除。

**P2 结论（daily run `20260526_p2_trade_daily_v2`，2026-05-26）：** T1–T4 成交冲击候选相对 R02 全部 reject，无候选进 expanding。
- 结果路径：`results/eval_protocol/20260526_p2_trade_daily_v2/`（leaderboard.csv 全指标）。
- **T1/T2/T4（原始+动态成交冲击）= 虚涨真伤**：corr_mean +0.001，但制造新强负折（corr_min 从 +0.019 → −0.004，short 最差折 −0.040）、全折正率 100%→84%、stability −0.0048。撞 §4.7「新强负折」红线，即便不卡地板也该毙。
- **T3（交互项 `trade_pressure×spread/×order_pressure/×ofi`）= 零增益**（Δ=+0.00003）。线性 Ridge 下这三个乘积项对基线无贡献。
- 根因（同 P1）：基线 `norm_core` 已含横截面标准化成交冲击（`trade_pressure_*_cs_z` 等）；原始/动态版本只加噪、交互版本无新信息。
- ⚠️ 待清理：T4 `trade_impact_safe` 与 T1 `trade_impact` 逐位相同（同 O4≡O6，`*_safe` 即全集），冗余待清。
- 运行口径：看门狗(软9/硬11GB) + `--n-workers 1` 串行，峰值 ~8.7GB 压住，exit 0；daily 在 16GB Mac 必须串行（AGENTS §5.1）。

**P1+P2 合并启示**：OFI 与成交冲击两族,最稳健可外推的横截面标准化版本基线(R02 norm_core)已纳入;追加原始/动态/交互变体要么零增益、要么伤稳定性。**P3 之前的预期应据此下调**——单族线性追加大概率难破 +0.003。

**P3 结论（daily run `20260526_p3_condmom_daily_v1`，2026-05-26）：** C1/C2 相对 R02 均 reject（均 < +0.003 地板），但成色完全不同，必须分开看（不只认 decision 标签，§4.6/§4.7）。
- 结果路径：`results/eval_protocol/20260526_p3_condmom_daily_v1/`（leaderboard + fold_metrics 全指标）。
- **C1（条件动量全集，~30+ 列）= 真该毙，数据饥渴型过拟合**：short 段 Δval=**−0.0031**、t=−0.41、折向 7+/12−、训练-验证 gap **+0.0364**、最坏折 −0.0188（造负折，全折正率 100%→94.7%）；long 段才转正（Δval +0.0068、gap +0.0070）。即"训练窗口短就过拟合、长才勉强有用"，对外推未知未来是危险型。corr_mean 的 +0.0018 是被 long 段拽上去的虚高。
- **C2（仅交互列 `lagret{1,3,6,12,24}×{ofi,spread,trade_pressure,vol}`）= P1 以来最干净候选，小而真**：short Δval +0.0023 / 配对 t=+2.88 / 折向 15+/4− / gap 仅 +0.0032 / 最坏折 +0.0042 / daily IC +0.0027；long Δval +0.0015 / t=+3.00 / 5+/1− / gap +0.0016 / daily IC +0.0024。**两 profile 同向正、配对 t 均 >2.8、daily IC 双升、最坏折双改善、不过拟合**——几乎过 §4.7 每一镜，唯量级输（Δ=+0.0019 < 地板）。
- **一句话洞察**：交互项这条路比"单族线性追加"更有戏。C2 证明 `lagret×{ofi/spread/...}` 乘积里有基线（norm_core 只含线性横截面标准化项）没显式建过的独立小信号；它恰是 §7.10 高先验交互（lagret12×ofi_total 等）的预览。**不送 C2 单独进 expanding**（+0.0019 标配在鲁棒第一下不值得 promote），但把交互列作为 Q3（P3.5）和 Q4（跨族组合）的核心抓手——小信号要看能否在组合里累加或被非线性放大。
- 注意：配对 t 因 short 折高度重叠（8d 训练/2d 验证）有高估，别当严格显著性；真正可信的是"跨 profile 符号一致 + daily IC 双升"这种难伪造的多镜一致。
- 运行口径：看门狗(软9/硬11GB) + `--n-workers 1` 串行，峰值 8.17GB 压住，exit 0。

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
