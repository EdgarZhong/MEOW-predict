# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-28

## 当前阶段：P4-2b 树侧精炼（LightGBM + 补深度 + 精简集，准备中）

> P4-2 初筛 v3 已完成读榜（见下）。ExtraTrees d6 双镜头第一，但深度单调递增还没到头（d4<d5<d6）、GBDT 侧只试了 sklearn HistGB（弱化版）没上老师推荐的 LightGBM。**P4-2b 补一轮 long-only 精炼再进决赛。**

### P4-2 初筛 v3 结论（2026-05-28 读榜，run `20260527_p42_modelselect_longscreen_v3`）

✅ 干净跑完（returncode=0，peak RSS 11.06GB，看门狗 soft 12 / hard 13 未触发）。10 个候选 × 6 折 long_40d_5d，0.33 采样训练。

| # | 模型 | corr_mean | corr_min | stability | daily_ic | val_corr std | gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **M_tree_d6** | **0.0708** | **0.0511** | **0.0601** | 0.0659 | 0.017 | 0.012 |
| 2 | X1 (ridge) | 0.0703 | 0.0403 | 0.0523 | 0.0507 | 0.028 | 0.008 |
| 3 | M_tree_d5 | 0.0680 | 0.0476 | 0.0573 | 0.0640 | 0.017 | 0.007 |
| 4 | R02 baseline | 0.0667 | 0.0321 | 0.0476 | 0.0457 | 0.030 | 0.006 |
| 5-7 | HistGB ×3 | 0.061–0.065 | — | — | — | — | — |
| 8-9 | Huber / EN | 0.055–0.060 | — | — | — | — | — |

**双镜头判断：**
- **树的最大优势在鲁棒性（方差压缩）**：M_tree_d6 均值仅比 X1 高 +0.0005，但 corr_min 高 +0.0108（0.051 vs 0.040），逐折 std 几乎减半（0.017 vs 0.028）。树把好折的分匀给差折——隐藏测试集分布未知，minimax 天然利好树。
- **深度单调递增还没到头**：d4(0.063) < d5(0.068) < d6(0.071)，d6 不是峰值而是截断，需补 d7/d8 看拐点。
- **d6 有轻微过拟合倾向**：gap=0.012（d5=0.007），fold 4-5 gap 达 0.032-0.035，expanding 时需关注。
- **HistGB 全线不如 ExtraTrees**（均低于 R02），但 HistGB 是 sklearn 简化版——真正的 LightGBM（GBDT 正牌选手、老师方案推荐）还没上场。
- **线性替代（EN/Huber）全败**：Ridge 已是线性族天花板。
- 结果路径：`results/eval_protocol/20260527_p42_modelselect_longscreen_v3/`

**P4-2b 精炼计划（补一轮 long-only 再进决赛）：**
1. **上 LightGBM**：安装 lightgbm + 加 M_lgbm 浅深度 spec（代码已有 `lgbm` 分支 plumbing，只差装包 + 钉 spec）
2. **补深度 d7/d8**：ExtraTrees 深度还没到头，补两个点看拐点（估算 +70 min，内存不变）
3. **精简树特征集**：按 P4-2' 重要性结论剪手工交互列（0.90%≈0）+ raw ofi（0.29%）→ 减噪

**P4-2 提速配置（不变，口径见 AGENTS §4.9 第 7 点 / §5.1）：**
- `--train-subsample-frac 0.33`（仅树族采训练行、验证集全量不动、线性自动全量）；ExtraTrees `n_jobs=8`（已钉进 M_tree spec）；`--n-workers 1` 不变；保留 300 树。
- smoke 实证：保真 Δ-0.0004、提速 ~15-30×、内存 9.4→3.2GB。
- 代码落地：`experiment_runner` 加 `train_subsample_frac` + `_subsample_train_rows`（树族门控）；CLI/scheduler 全链路透传；5 项新单测通过。

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
- 候选：Ridge / ElasticNet / Huber / 浅 ExtraTrees(depth≤5) / 浅 HistGB / **LightGBM**（AGENTS §六）。
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
| 候选（线性最优） | X1_R02_plus_ofi_safe_condmom_interaction | 0.066766(exp) | 0.051788 | 0.041041 | 树要打败的靶子；**11 月 holdout 已通过 0.07367（vs R02 0.06952）** |

来源：`results/eval_protocol/20260526_p0relock_daily_v1` / `..._p0relock_gate_v1` / `..._q4_gate_v1` / `..._x1_review_v1`（X1 review holdout）。

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
| P4-1 | 树特征集 + 清脏列 + 模型 plumbing（前置研究）：**plumbing 已落地**（见「P4-1 落地」） | ✅ 就绪 |
| P4-2 | 模型 long-only 初筛：v1 OOM→提速→v3 干净跑完（`20260527_p42_modelselect_longscreen_v3`），双镜头读榜见上 | ✅ 完成 |
| P4-2′ | 树重要性扫描（Fork A）：run `20260527_p4_tree_importance_v1` 干净跑完，结论落档 `docs/实验记录.md`（剪交互 / trade被低估 / ofi没错杀） | ✅ 完成 |
| P4-2提速 | 训练行采样 plumbing（仅树族门控 + 全链路透传）+ 5 项新单测 + smoke 实测保真/提速/内存 + 文档收敛 | ✅ 完成 |
| P4-2b | 树侧精炼：① 上 LightGBM（装包+钉 spec）② 补 ExtraTrees d7/d8 ③ 按重要性结论精简树特征集（剪手工交互+raw ofi）→ 合并跑一轮 long-only | ⏳ 准备中 |
| P4-3 | 决赛 2–3 模型 expanding（全量行 + 300 树），打 X1 expanding 0.0668 靶子 | 待开（等 P4-2b 读榜） |
| 🚩红线 | X1 进 11 月 Review Holdout（§4.8）：**已通过——X1 0.07367 vs R02 0.06952，样本外跑赢 +0.0042**（run `20260527_x1_review_v1`） | ✅ 完成 |
| 清理 | 失败 stage `p35_interactions` 已归档（builder 移 .archive、摘 stage、删 spec I1、缓存移 .archive） | ✅ 完成 |

### P4-1 落地（2026-05-27，import + 12 项 unittest 通过，未跑实验）

- **树特征集（Fork A 拍板 = 先全含交互 + 跑重要性扫描）**：`eval_protocol.P4_TREE_GROUPS` = legacy + norm_core + ofi_safe + trade_impact + conditional_momentum + lag + roll + patch_summary + cross_rank + **regime_tree**。含手工交互（让重要性扫描一次回答「树是否真用得上交互」+ 反偏颇），保留 cross-z/cross-rank（树不自做按天截面归一），清广播脏列。
- **regime_tree group**（feature_registry）：regime 11 列去掉 `state_spread_cs`/`state_activity_cs`（一天一值广播常量，树会当日期身份乱切）= 9 列；`state_vol_cs` 保留。
- **模型 plumbing**（experiment_runner）：补 `huber`；加浅树变体 `tree_shallow`(ExtraTrees depth≤5)/`histgb_shallow`；`fit_model`/`run`/`run_with_groups`/`_evaluate_spec_on_fold` 加 `model_params` 穿透（预钉网格覆盖超参）；加 `_extract_tree_importance`（collect 路径线性 coef 为 None 时回退取重要性）。
- **M 系列网格 spec**（ALL_SPECS，预钉防多重比较）：线性 `M_en_X1`/`M_huber_X1`（X1 集，ridge-on-X1=既有 X1）；浅 ExtraTrees `M_tree_d4/d5/d6`；浅 HistGB `M_histgb_d3/d4/d4_lr03`（树大集）。depth 是浅树主正则器、leaf=500 当下限非约束。
- **winsorize 对树重评**：无需改码，用运行期 `--target-winsorize` 开关做 on/off A/B（默认沿用锁定 P1/P99）。
- **P4-2 跑法（long-only 初筛车道）**：`p0_eval_protocol.py --suite daily --spec-ids X1_... M_en_X1 M_huber_X1 M_tree_d4/d5/d6 M_histgb_d3/d4/d4_lr03 --profiles long_40d_5d --n-workers 1 --train-subsample-frac 0.33`（看门狗包装、日志写 `logs/`）。⚠️ 此 P4-1 时点的初版命令**漏了 `--train-subsample-frac`**，v1 全量行×单核跑树被内存看门狗杀；提速口径见上「P4-2 提速配置」+ AGENTS §4.9 第 7 点，正式跑法以 v2 为准。

## P4-2b 交接（下一会话接手点）

**P4-2 初筛 v3 已完成读榜（结论见上），树侧还有空间：深度没到头 + LightGBM 没上场 + 特征集未精简。P4-2b 补一轮 long-only 精炼再进决赛。**

**P4-2b 要做的（代码改动 + 跑一轮）：**
1. `pip install lightgbm` + 在 `eval_protocol.py` 加 M_lgbm spec（浅深度，预钉网格）
2. 加 M_tree_d7 / M_tree_d8 spec（ExtraTrees depth=7/8，其他参数同 d4-d6）
3. 精简树特征集 `P4_TREE_GROUPS_V2`：从 P4_TREE_GROUPS 中去掉 `conditional_momentum`（手工交互，重要性 0.90%）；raw ofi 暂保留（ofi_safe 含 cross-z/rank，不是纯 raw）
4. 合并跑一轮 long-only（新 spec + 精简集 + X1/R02 对照）

**看门狗阈值**：soft 12GB / hard 13GB（v3 实测 peak 11.06GB，安全）。

**红线提醒**：11 月 Review holdout 已用于 X1 确认（§4.8），**不得再用 11 月调参/选模型**；12 月 Final holdout 全程未碰、最终一次性用。选模型只在 Dev rolling / long / expanding 上做。

## 已完成基建（备查）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader 三件套 + 单测；主链路已切 FeatureLoader；9 stage 462 列。规格见 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + 输出结构 + baseline delta + make_decision；daily/gate/ridge/quick/full suite 可跑。
- 开跑前债务清零（#10–#20）全部落地，详见 git 历史与 `docs/specs/开跑前编码指导_评测口径与提速.md`。
