# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-27

## 当前阶段：P4 开局（稳健模型比较，重心 = 树）

特征侧（P1–Q4，线性 Ridge）已收官，全量明细沉淀至 `docs/实验记录.md`。一句话结论：

> **线性特征侧已到头。** 三族（OFI / 成交冲击 / 动量-反转）单族追加全走弱，根因同一——每族最可外推的主体基线 R02 已含（OFI/成交冲击在 `norm_core` 的 cross-z、**动量在 `legacy` 的 `lagret12`**）；线性又吃不进交互（Q3 实证）。唯一破地板的是 **X1 = R02 + `ofi_safe` + `conditional_momentum_interaction`**（expanding Δ+0.0044、12 折全正、Δ 随训练历史拉长单调增大）。**冲高分主战场转树。**

## P4 开局设计要点（交接给新会话）

**【定位】** 线性阶段是"立地基"非"求最优"——评测体系 / 数据认知 / 提交通道 / 9 stage 特征工程**全可复用**；但"特征价值"依赖模型，线性上的特征贡献排序**对树不适用、要重评**。

**【树路径现状 — 已就绪，不用为树改基建】**
- `experiment_runner.py` 已实现 `ExtraTreesRegressor` / `HistGradientBoostingRegressor`（+lightgbm，缺了 fallback histgb），与 ridge 平级共存；缺失值已统一 `fillna(0)`（ExtraTrees 可用）；float32 已锁。改树不破坏任何现有线性路径。

**【核心策略 — 不重搜特征组合】**
- 手工组合搜索是线性时代被逼的（线性维度多则拟合噪声）。树自带特征选择 + 对无关列鲁棒 → **喂大集让树筛，不做 2^N 子集搜索**。
- 树特征集 ≈ **去掉手工交互**（树自建交互，C2/X1 那种乘积列对树冗余）+ **保留 cross-z/cross-rank**（树不会自动按天截面归一化，截面排名要喂）+ 原始信号族。
- **反偏颇顺带做**：树喂大集看重要性，被线性低估的特征（原始 ofi/trade、Q3 交互源）若冒头 = 线性错杀，一次回答，不重跑 P1–Q4。

**【前置研究点 — 非破坏，不动线性路径】**
1. 树特征集设计：原始族+cross-z（去手工交互） vs X1，对比。
2. 清对树有害的脏特征：`regime` 广播量 `state_spread_cs`/`state_activity_cs`（一天一值，树造重复分裂、稀释重要性）。
3. winsorize 对树重评（线性锁 P1/P99 不动；树用分裂点、机制不同，单独看是否还要 winsorize）。
4. （可选）特征重要性预探缩集 + 浅树 expanding 单折算力/内存 smoke。

**【模型选型口径 — 协议已改写，正文见 AGENTS §4.9】** 选模型 = 模型为变量，§4.1–4.8 的特征增量口径要改写（2026-05-27 拍板）：
- 候选：Ridge / ElasticNet / Huber / 浅 ExtraTrees(depth≤5) / 浅 HistGB（AGENTS §六）。
- **对照单元 = 每模型用各自最佳特征集（按类型指派、非搜索）**：树喂"去手工交互+清 regime 脏列"大集自选，ridge/EN/Huber 用 X1 集。目标 = 最准提交（不究模型归因）。
- **退役 +0.003 地板 + make_decision**（仅选模型语境）：模型互换不是特征增量，地板统计推导不适用。
- **profile 重心挪 long/expanding，short 降为过拟合诊断**（short=8 天对树不公平）；便宜初筛用 **long-only**，决赛 2–3 个才上 expanding（树贵）。
- **判官 = 均值（对齐老师 pooled corr）+ 鲁棒（最坏折/minimax）双镜头同时看、人工权衡**，不退化成单指标闸刀。
- 每模型各自小超参网格**预先钉死**（防多重比较膨胀）；winsorize 对每模型重评。
- **靶子 = X1 的 expanding corr_mean 0.0668**（ridge 上限），树要在双镜头下打过它才值得换。

## 线性收尾（已了结）

C2/O4 单独 expanding 补跑完（runs `20260527_c2_gate_v1` / `20260527_o4_gate_v1`，明细见 `docs/实验记录.md` 收官节）：两者都干净、12 折全正、最坏折升、daily IC 双升，单独 Δ 低于地板（C2 +0.00275 / O4 +0.00219）——是"小而真、低于采纳地板"，非噪声/有害。**X1 ≈ 两者之和（+0.0044 < +0.00494）→ 真互补、仅轻微重叠，靶子扎实。** 线性章节"数据真空"补齐、正式收官。

## 当前最优基线（锁定口径）

> 锁定口径：训练标签 winsorize 开 + P1/P99、ridge alpha=2.0、特征 float32（P0.5 扫锁，详见 AGENTS §7.7/§7.11）。

| 范围 | experiment_id | protocol_corr_mean | stability | corr_min | 角色 |
|---|---|---:|---:|---:|---|
| 快车道 short+long | R02_ridge_legacy_plus_norm_core | 0.057168 | 0.039703 | 0.019425 | 线性 backbone |
| expanding 关口 | R02_ridge_legacy_plus_norm_core | 0.062392 | 0.047159 | 0.033476 | 线性 backbone |
| 候选（线性最优） | X1_R02_plus_ofi_safe_condmom_interaction | 0.066766(exp) | 0.051788 | 0.041041 | 树要打败的靶子；11 月 holdout 待用户 |

来源：`results/eval_protocol/20260526_p0relock_daily_v1` / `..._p0relock_gate_v1` / `..._q4_gate_v1`。

## 口径指针表（规则正文在 AGENTS，"为什么"在 NOTE）

| 口径 | 位置 |
|---|---|
| 两速评测结构 / 各阶段口径（P4 行） | AGENTS §4.2 / §4.4 |
| 宽进严出：进 expanding 不卡数字地板，地板只管采纳 | AGENTS §4.4 / §4.6 |
| make_decision 分诊 + 人工多角度判断（不自动判卷） | AGENTS §4.6 / §4.7 |
| 多重比较 / 边界增益（+0.003 是地板不是采纳标准） | AGENTS §4.3 |
| 三层 Holdout 纪律（Dev / Review / Final，红线不碰） | AGENTS §4.8 |
| **P4 选模型协议改写（退役地板/make_decision、双镜头判官、各自最佳集、long-only 初筛）** | **AGENTS §4.9** |
| P4 模型准入 / 调参粒度 / 加权 long-expanding | AGENTS §7.8 / §7.6 / §4.4 |
| Target Mode 准入（最终回 raw `fret12`） | AGENTS §7.5 |
| 跑命令前自查（看门狗 + `--n-workers 1` 串行 + 日志写 `logs/`） | AGENTS §5.1 |
| 推理契约与交付约束 | AGENTS §十 |

## 待办队列

| # | 任务 | 状态 |
|---|---|---|
| 线性收尾 | C2/O4 单独 expanding 跑完落档：均干净、低于地板、X1 真互补 | ✅ 完成 |
| P4-1 | 树特征集设计 + 清脏列 + winsorize 重评（前置研究） | 待开 |
| P4-2 | 模型选型 daily 初筛（ridge/en/huber/树）+ minimax | 待开 |
| P4-3 | 决赛 2–3 模型 expanding | 待开 |
| 🚩红线 | X1 进 11 月 Review Holdout（§4.8，Claude 不跑） | 交回用户 |
| 清理 | 失败 stage `p35_interactions` 归档（§7.3，涉及删 spec/builder） | 待用户定 |

## 已完成基建（备查）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader 三件套 + 单测；主链路已切 FeatureLoader；9 stage 462 列。规格见 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + 输出结构 + baseline delta + make_decision；daily/gate/ridge/quick/full suite 可跑。
- 开跑前债务清零（#10–#20）全部落地，详见 git 历史与 `docs/specs/开跑前编码指导_评测口径与提速.md`。
