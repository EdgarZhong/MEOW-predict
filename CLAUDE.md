# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-26

## 待办队列（下一会话执行）

> 真正跑实验在**新会话**。每项流程：看门狗+串行启动（命令模板 AGENTS §5.1 item5）→ `PYTHONPATH=src python experiments/analyze_run.py --run-id <id>` 多角度看 + 带专业判断（**不只认 decision 标签**，§4.6/§4.7）→ 落档本文件（§一记录规范）→ 测好就 commit + 同步必要文档。状态锚点：`results/eval_protocol/<run-id>/leaderboard.csv` 存在=该项完成。**自动作业止于 P4（模型选型）前**，holdout 不碰。缺的脚本/group/特征实现按需自行编码（已授权）。

| # | 任务 | 关键参数 |
|---|---|---|
| ✅Q1 | P3 条件动量（C1 本体 / C2 交互）— **判负**：C1 过拟合、C2 小而真但低于地板；C2 交互信号留作 Q3/Q4 抓手（详见 P3 结论） | run `20260526_p3_condmom_daily_v1` 完成 |
| ✅Q2 | alpha 敏感性 — **否决**：alpha 不是杠杆，O4 跨 10× alpha 纹丝不动、T1 short 崩塌 shrinkage 救不了；alpha=2 锁定（详见 Q2 结论） | runs `20260526_q2_alpha5/20_daily_v1` 完成 |
| ✅Q3 | P3.5 高先验交互（新建 4 项）— **判负零增益**；C2(lagret×X) 仍是唯一有戏交互（详见 Q3 结论） | run `20260526_q3_p35int_daily_v1` 完成 |
| ✅Q4 | 跨族组合 X1=ofi_safe+cond_mom交互 — **三视角全过**：daily Δ+0.00305 + expanding Δ+0.0044(12/12全正,t7.35)；P1–Q4 唯一通过 expanding 硬门槛的候选，待用户定 11 月 Review（详见 Q4 结论） | daily + gate 均完成 |
| ✅Q5 | 综合 P1–Q4 + 对 P4 的建议 — **完成，特征侧收官，交回用户**（综合见 NOTE.md「P1–Q4 特征侧收官 + 对 P4 的建议」） | 纯分析+落档；交回点见下 |

> **🚩 自动作业到此为止——交回用户两个决定**（详见 NOTE.md 收官节）：
> 1. **X1 进不进 11 月 Review Holdout**（§4.8 红线，Claude 不跑）：建议 `--suite gate --candidate-spec-id X1_R02_plus_ofi_safe_condmom_interaction --include-review-holdout --n-workers 1`，11 月对得上则 X1 正式替 R02 当 backbone。
> 2. **P4 怎么开**：建议 ① 对比基线换成 X1 特征集；② 决赛重点上浅树（Q3 已证线性吃不进交互，非线性是唯一未探方向）；③ 加权 long/expanding + minimax 防 ofi_safe 在 short 过拟合。
> 另：失败 candidate `p35_interactions` 待用户定夺是否按 §7.3 归档（涉及删 spec/builder，Claude 不擅自动）。

## 当前阶段目标

**方案 B + 两速评测结构已固定；特征侧穷尽（P1–Q4）已收官，停在 P4 前交回用户。**

**特征侧总结论：单族线性加特征基本死路（P1 OFI / P2 成交冲击 / Q2 alpha 非杠杆 / Q3 §7.10 新交互零增益全否）；唯一破地板的是 Q4 的 X1 = R02 + `ofi_safe` + `conditional_momentum_interaction`**——把两个"小而真"的干净信号（OFI 长窗 +0.0027、动量×微结构交互 +0.0019）几乎可加地叠起来。X1 三视角全过：short(+0.0025)<long(+0.0036)<expanding(+0.0044) **Δ 随训练历史拉长单调增大**（鲁棒外推的正面信号）、expanding 12/12 全正、daily IC 三处皆升、不过拟合。它通过了 Claude 权限内全部关口（含 expanding 硬门槛 §4.4），是 P1–Q4 唯一站得住的候选，待用户定 11 月 Review。

根因贯穿全程：基线 R02 的 `norm_core` 已含横截面标准化版本（cross-z），同族线性追加边际≈0 或仅过拟合；线性 Ridge 吃不进乘积/交互（Q3 实证）——故 P4 的增量希望在浅树等非线性模型。评测口径全程"诊断仪表盘+人工多角度判断"，不靠 decision 标签自动判卷（AGENTS §4.6，[[feedback-no-automated-grading]]）。

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

**Q2 结论（alpha 敏感性，daily runs `20260526_q2_alpha5_daily_v1` / `20260526_q2_alpha20_daily_v1`，2026-05-26）：** 验"O4·T1 弱信号是否被正则强度埋"——**答案：不是，alpha 不是杠杆，alpha=2 锁定不动**（§7.7 平台区结论再获印证）。
- 结果路径：`results/eval_protocol/20260526_q2_alpha5_daily_v1/`、`.../20260526_q2_alpha20_daily_v1/`。
- **O4（OFI）alpha 完全不敏感**：Δcorr 在 alpha=2/5/20 三档恒为 +0.0016；short 永远 +0.0005（t≈0.6 噪声）、long 永远 +0.0027（t=2.82、6/0 全正、corr_min +0.0070、daily IC +0.0034 干净小信号）。跨 10× alpha 纹丝不动=稳定但极小的结构性增量，非正则伪影。坐实 P1 根因（基线已吃 OFI 稳健部分）。
- **T1（成交冲击）加大正则救不了 short 崩塌**：alpha=2 时 short 的 gap+0.047 / corr_min−0.046 强负折，到 alpha=5/20 几乎不动（gap+0.047 / corr_min−0.046）。10× shrinkage 压不住=问题是"特征本身在 8d 小样本抓伪相关"，非系数过大；L2 无能为力。坐实 P2"虚涨真伤"是结构性、非欠正则。
- **一句话洞察**：O4-long(+0.0027) 与 C2-交互(+0.0019) 都是"小而真但单独低于地板"的干净信号；能否在 Q4 跨族组合里累加破地板是下一个真问题。alpha 路已堵死，不再扫。
- 运行口径：两 alpha 用 `&&` 串成一个后台任务顺序跑（不并行），各带看门狗 + `--n-workers 1`，峰值 8.37GB，exit 0。

### P3.5 少量交互项冲刺【P1–P3 有初步结论后】

- [x] **Q3：§7.10 高先验交互（新建 4 项）— 判负，零增益。** C2(lagret×X) 仍是唯一有戏交互，且仍低于地板。

**Q3 结论（daily run `20260526_q3_p35int_daily_v1`，2026-05-26）：** 新建 stage `p35_interactions`（status=candidate）测 §7.10 中尚未建过的 4 个交互，合并 spec `I1_R02_plus_p35_interactions` 相对 R02 **reject，基本零增益**。
- §7.10 六项映射：`trade_pressure×spread`=trade_impact 的 `trade_pressure_x_spread`（P2/T3 已测零增益）；`lagret12×ofi_total`=cond_mom 的 `lagret12_x_ofi`（C2 已测干净）；故新建只补剩 4 项：`i_ofi_x_spread`/`i_ofi_x_trade_activity`/`i_lagret12_x_order_pressure`/`i_trade_pressure_x_regime`。
- 结果路径：`results/eval_protocol/20260526_q3_p35int_daily_v1/`。Δcorr=+0.0002；short Δ+0.0002（t=0.95、gap+0.0002）、long Δ+0.0001（t=1.16、gap≈0）。干净的零，不过拟合也不伤害。
- **一句话洞察**：否决"交互普遍有用"。C2 的 +0.0019 是特指**动量×微结构族**（lagret×{ofi,spread,trade_pressure,vol} 跨 5 窗口 20 列）；我新建的跨族交互（OFI×流动性、成交压力×regime、单窗口 lagret×order_pressure）线性下全空——源特征基线已有 cross-z 线性项，乘积无独立信号；trade_pressure 本就无用(P2)×regime 救不回。**§7.10 高先验清单大部分在线性模型下是空的，唯一有戏是 C2。**
- 决定 Q4：跨族组合只并 **C2(conditional_momentum_interaction) + O4(ofi_safe)** 两个"小而真"信号，I1 不进。
- ⚠️ 待清理（Q5 收口时）：`p35_interactions` 是失败 candidate，零增益，Q5 阶段收口时按 §7.3 归档（builder 移 `.archive`、registry 移除、spec I1 删）。当前留作实验记录。
- 运行口径：`feature_store build --all --stages p35_interactions`（144 天 4 列，peak 0.85GB）→ daily（peak 6.37GB）串成一个后台任务，各带看门狗 + `--n-workers 1`，exit 0。

### Q4 跨族组合【§4.5 组合不可加，当新 spec 重跑】

- [x] **Q4-daily：X1 = R02 + ofi_safe + conditional_momentum_interaction — 第一个破 +0.003 地板的候选（Δ=+0.00305，review）。**
- [x] **Q4-gate：X1 expanding 慢车道关口确认（Δ=+0.0044，12/12 全正，t=7.35）——三视角全过，唯一通过 §4.4 expanding 硬门槛的候选。下一步 11 月 Review Holdout 属红线，交回用户。**

**Q4 结论（daily run `20260526_q4_combo_daily_v1`，2026-05-26）：** 把两个唯一"小而真"的干净信号叠加——O4 的 `ofi_safe`（OFI 长窗 +0.0027 干净）+ C2 的 `conditional_momentum_interaction`（lagret×X，+0.0019 干净）——**首次破地板**。
- 结果路径：`results/eval_protocol/20260526_q4_combo_daily_v1/`。Δcorr=**+0.00305**（make_decision=review，边界区 [0.003,0.005)）；protocol stability 0.0441↑（基线 0.0397）、corr_min 0.0252↑（基线 0.0194）。
- per-profile（§4.7，全镜一致）：short Δ+0.0025 / t=1.91 / 15+4− / corr_min**+0.0035↑** / dailyIC+0.0037 / gap+0.0070（中等，来自 ofi_safe 列在 8d 窗的数据饥渴，但未造负折）；long Δ+0.0036 / t=3.46 / **6+0−** / corr_min**+0.0081↑** / dailyIC+0.0050 / gap+0.0015。
- **一句话洞察**：①**几乎可加**——O4 单(+0.0016)+C2 单(+0.0019)=+0.0035，组合实测 +0.00305，略低于和=§4.5 预言的 OFI 信息重叠造成的轻微次可加，但大部分信号叠上了；②质量与 T1"虚涨真伤"相反：stability 升、两 profile corr_min 双升、daily IC 双升、两 profile 同向正——**这是整个 P1–Q4 第一个、也是唯一一个站得住的候选**。
- 运行口径：daily 看门狗 + `--n-workers 1` 串行，exit 0。

**Q4-gate 结论（expanding 关口 run `20260526_q4_gate_v1`，2026-05-26）：第三视角强确认 X1。**
- 结果路径：`results/eval_protocol/20260526_q4_gate_v1/`（只跑 expanding profile，不含 holdout）。
- X1 vs R02：Δcorr=**+0.0044**（比 daily 的 +0.00305 更高）、corr_mean 0.0668 vs 0.0624、stability 0.0518↑、corr_min 0.0410↑（基线 0.0335）；expanding 12 折 **Δval+0.0044 / t=7.35 / 12+0− 全正 / gap+0.0007（几乎不过拟合）/ corr_min+0.0076↑ / dailyIC+0.0061↑**。
- **一句话洞察**：三视角 Δ 随训练历史拉长**单调增大**（short+0.0025 < long+0.0036 < expanding+0.0044），是"见效快但靠近期"危险型的**反面**——信号用更多历史预测未来时更可靠，最契合面向未知隐藏集的鲁棒性目标（[[hidden-test-unknown]]）。§4.7 六镜全过。
- **X1 已通过 Claude 权限内全部关口**（short/long 快车道 + expanding 慢车道硬门槛 §4.4），是 P1–Q4 唯一通过 expanding 的候选。make_decision=review（+0.0044<0.005 未到 auto-promote，但 §4.3 真关卡=跨视角一致性，X1 达最佳）。
- **下一步 = 11 月 Review Holdout（§4.8 红线，Claude 不跑）**：交回用户决定是否对 X1 跑 Review Holdout 复核，对得上再定 promote。
- 运行口径：gate `--n-workers 1` 串行（避免 §5.1 双并行 expanding 尾段 OOM），expanding 尾段峰值 9.82GB（<11GB 硬阈，看门狗瞬时尖峰未误杀），exit 0。

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
