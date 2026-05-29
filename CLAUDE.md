# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-29

## 当前阶段：传统已基本收口 → 主线转「交付接线 + DL」（2026-05-29）

> **传统侧已收口**：P4-3 决赛 + P5 融合拍板——`M_tree_d8` expanding 塌(0.0580 跌破基线、负折)被 reject;**锁等权 zscore 融合 [X1 + M_lgbm_d4] 当传统保底代表**(expanding mean 0.0776 / pooled 0.0763 / 最坏折 0.0491,两镜头超两成员;minimax 正解 + 零自由参数可辩护)。**传统天花板 ~0.085–0.09,冲 0.10 改押 DL。** 详见 `docs/实验记录.md` 2026-05-28。
>
> **战略转型(2026-05-29 拍板)**：传统不再冲分,**主线 = ① 交付接线收口 + ② DL(Windows 4060)**。传统的 HPO / 小波 / MLP **保留为后续方向、推迟**。
>
> **交付接线进度（2026-05-29 更新）**：两成员融合 + 并集特征 + 零缓存已接通、真实数据跑通、单测全过、commit `e005f62`@`feat/submission-blend-raw-mean`。**关键设计变更：融合口径从 zscore 锁定到 `raw_mean`**——老师精度分 = MSE+Pearson+R² 各 1/3(`meow/MEOW金融时序预测2.0.docx`)，zscore 输出 std≈1 毁 MSE/R²，raw 平均保量纲且 corr 不损(P5: raw 0.0762≈zscore 0.0763)。**剩余下一步**：内存精简(全窗口 fit ~30GB→~20GB)+ Dec 演练(上 ≥32GB 机器,本机 16GB 跑不动全窗口)+ 提交版减注释。详见待办队列「交付接线」四行 + §「交付接线收口」。
>
> **传统样本外验证已全部收口（2026-05-29）**：`M_lgbm_d4` Nov holdout **0.08967** vs R02 **0.06952**(run `20260529_p43_lgbm_d4_nov_review`)、X1 已过 0.07367。两成员各自过样本外 → 等权零参融合"可辩护默认"。**12 月 Final Holdout 全程不碰**——留给"最终唯一代表(传统 or DL)选定后"一次性确认(§4.8;Dec 演练那次即动用)。
>
> **交付接线的注意事项见下「交付接线收口 — 注意事项」节。竞赛背景 + roadmap + 详细任务见下。**

### 2026-05-29 运行记录

- `run_id=20260529_p43_lgbm_d4_nov_review`，suite=`review holdout`，候选 `M_lgbm_d4` vs 基线 `R02_ridge_legacy_plus_norm_core`
- `make_decision` 标签可忽略（review 脚本仍沿旧地板提示）；实际样本外结果：`M_lgbm_d4=0.08967`，`R02=0.06952`
- 关键数字：样本外 `delta_corr=+0.02016`，远高于“没塌”的最低要求
- 一句话洞察：`M_lgbm_d4` 的“gap 偏热”并没有在 11 月外推里兑现成塌方，说明它学到的不是只在 6–10 月局部窗口生效的假信号，而是比 R02 更强的非线性排序结构
- 结果路径：`results/eval_protocol/20260529_p43_lgbm_d4_nov_review/`
- 下一步：把 `X1 + M_lgbm_d4` 融合接进 `submission_pipeline` / `meow.py`（**已完成,融合口径锁 `raw_mean` 而非 zscore——见「交付接线收口」第 2 条:老师精度分 MSE/R²/corr 各 1/3,zscore 毁 MSE/R²**）

### 竞赛背景与算力分工（2026-05-28）

- **课程竞赛**：组内**无隐藏集**、仅讨论选**一个代表模型**交老师；老师隐藏集终判。约 **06-04 组内讨论**、老师截止下下周（~06-08~12）。
- **战略**：稳健 = 竞争武器 → 冲"**可辩护、可泛化**的分"（过 expanding + 11月 holdout），不刷 dev（理由见 NOTE「冲高分阶段」节）。
- **算力分工**：传统留 **Mac → P5**；DL 上 **Windows 4060 / PyTorch**（不用 MLX/M5），互不抢算力。
- **回退点 05-31**：DL 打不过传统 ensemble 就回退，传统 ensemble 为保底代表。

### 本周双轨 roadmap

| 日 | Mac（传统，主线） | Windows 4060（DL，上行） |
|---|---|---|
| 05-28 | P4-2b 读榜→选决赛→expanding；搭加权 ensemble 脚手架 | 装 PyTorch/CUDA；搭序列管道（防泄漏开窗+归一化）；LSTM 跑起来 |
| 05-29 | 锁最优单模型；加权 ensemble 上 rolling+expanding | LSTM 接同一 rolling 出真分；判断序列有没有料 |
| 05-30 | 传统候选过 11月 Review Holdout = 保底强分 | 有料→调/试 stack 进 DL；没料→评估 DeepLOB 或回退 |
| 05-31 | （机动）小波特征 / OOF stacking，仅明显更好才采 | **回退决策点** |
| 06-01 | 收敛单一代表模型 | 同左二选一收敛 |
| 06-02~03 | pitch 材料 + **meow.py 提交通道演练（§十）** | 若 DL 入选：打通 meow.py 提交 |
| 06-04 | **组内讨论**（带验证过的模型 + 泛化故事） | — |
| 之后 | 当选→Final Holdout(12月) 一次性确认→提交，看完不改 | — |

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

### P4-2b 读榜结论（2026-05-28，run `20260528_p42b_tree_refine_v1`）

✅ 干净跑完。4 个候选 × 6 折 `long_40d_5d`，0.33 采样训练；结果路径：`results/eval_protocol/20260528_p42b_tree_refine_v1/`

| # | 模型 | corr_mean | corr_min | stability | daily_ic | gap |
|---|---|---:|---:|---:|---:|---:|
| 1 | **M_lgbm_d4** | **0.0864** | 0.0437 | **0.0714** | **0.0763** | **0.1043** |
| 2 | M_lgbm_d6 | 0.0845 | 0.0451 | 0.0698 | 0.0745 | 0.1937 |
| 3 | **M_tree_d8** | 0.0721 | **0.0523** | 0.0610 | 0.0678 | 0.0245 |
| 4 | **M_tree_d7** | 0.0711 | 0.0499 | 0.0597 | 0.0675 | 0.0187 |
| 5 | R02 baseline | 0.0667 | 0.0321 | 0.0476 | 0.0457 | 0.0065 |

**判断：**
- **LightGBM 确实打出了最强 long-only 均值**：`M_lgbm_d4` 相对 R02 `delta_corr=+0.0197`，daily 也同步抬升，不是只赚“大盘的钱”。但增益明显集中在 fold0 / fold5，且 `train_val_gap=0.104` 很大，属于**高上限、强过热嫌疑**，不能在 long-only 直接判赢，必须进 `expanding` 让判官验真伪。
- **LightGBM 只留 d4，不留 d6**：`M_lgbm_d6` 被 `d4` 基本支配（均值更低、daily 更低、gap 更大到 0.194），说明继续加深只是在把训练端推得更热，**没有换来更稳的验证收益**。
- **ExtraTrees 深度还没塌，但拐点很近**：`d8` 比上一轮 `d6` 继续小升（0.0721 > 0.0708，`corr_min` 0.0523 > 0.0511），说明深度加到 8 还没明显坏掉；但 gap 从 0.012 → 0.025 也在上行，过拟合压力开始累积。
- **树的收益结构和 LightGBM 完全不同**：`M_tree_d8/d7` 不是靠把强折再拔高，而是**修补弱折 0/4/5、抬最差情况**，这是隐藏集完全未知时更值钱的 minimax 形状。
- **因此这轮不该只送 2 个**：按“禁止自动判卷 + 宽进严出”的 memory 口径，这里有 **两类不同的真候选**：一个是上限型 `M_lgbm_d4`，一类是鲁棒型 ExtraTrees。`d8` 是树代表；`d7` 虽然被 `d8` 小幅领先，但 gap 更低，是**保守备胎**，值得一起进 `expanding` 让累积训练历史来决定深度峰值。

**P4-3 expanding 决赛名单（拍板）：**
1. `M_lgbm_d4`
2. `M_tree_d8`
3. `M_tree_d7`

**淘汰：**
- `M_lgbm_d6`：被 `d4` 支配，且过热最严重，不值得占一个 `expanding` 名额。

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
| 候选（线性最优） | X1_R02_plus_ofi_safe_condmom_interaction | 0.066766(exp) | 0.051788 | 0.041041 | 融合成员（鲁棒锚）；**11 月 holdout 已通过 0.07367（vs R02 0.06952）** |
| 单模最优（树族） | M_lgbm_d4（lgbm,精简 V2 集） | 0.076217(exp) | 0.060816 | 0.034071 | 融合成员（上限型）；daily IC 0.0651/IR1.56；gap 0.057 偏热 |
| **传统代表（已锁）** | **BLEND 等权 zscore [X1 + M_lgbm_d4]** | **0.0776(exp)** | **0.0622** | **0.0491** | **传统保底代表**；pooled 0.0763；两镜头超两成员；Nov 由两成员各自样本外推断（X1 0.0737 已过 + lgbm_d4 Nov 待跑），不单测融合 |

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

## 交付接线收口 — 进度与口径（2026-05-29 更新）

> 现状：`meow.py` 壳层(fit/predict/eval)已接进 `src/submission_pipeline.py`、no-cache 已强制、两成员融合(X1 ridge + lgbm_d4)+并集特征已落地。**融合口径已从 zscore 切到 `raw_mean`**(见第 2 条),小窗口冒烟通过(corr 0.106 / R² 0.0065 / MSE≈0,量纲正确)。**内存精简(中档:预分配流式 + 末位释放源帧)已落地、本机小窗口验等价+冒烟通过(见第 7 条),持续峰 ~30→~20GB；剩余 = 全量峰上 ≥32GB 机器实测 + 正式 Dec 演练。**

1. **serve 端归一化必须用「新数据自算」、严禁冻结训练期先验**：若用到当日截面统计(如 zscore 诊断模式),mean/std 必须从**老师传进来的当天那批数据自身**实时算,绝不能把训练集统计量存成先验。注：默认 `raw_mean` 不做截面标准化、天然无此风险;此条对 `per_day_zscore_mean` 诊断模式仍适用。
2. **融合口径 = `raw_mean`（等权 raw 平均，已锁）**：两成员都在 raw `fret12` 上以平方损失训练,输出本就同量纲;直接等权平均 → **输出留在 `fret12` 量纲,MSE/R² 才有意义**。⚠️ **关键**:老师精度分(30) = **MSE + Pearson + R² 各占 1/3**(见 `meow/MEOW金融时序预测2.0.docx`),per-day zscore 会把输出推到 std≈1、毁掉 MSE/R²(2/3 精度分归零)。P5 实测 raw 0.0762 ≈ zscore 0.0763 → 切 raw 不损 corr。`per_day_zscore_mean` 仅留作诊断对照(`SubmissionSpec.blend_mode` 可切)。
3. **特征并集**：`DEFAULT_SUBMISSION_GROUPS` = 两成员特征集并集(433 列,一次现算、两模型各取子集)。已落地。
4. **超参 / spec 配置**：两成员 model_params / 特征组 / 融合模式集中在 `SubmissionSpec` + `DEFAULT_SUBMISSION_MEMBERS` 一处声明。已落地。
5. **特征缓存禁止复用**：从老师新 raw `.h5` 现算(`mdl.py` 把 feature_dir 指向不存在路径以暴露误用)。已落地。
6. **演练验收清单**：fit 两模型 / predict 输出形状对、**raw `fret12` 量级(已验:R²转正)** / 训练推理同一 `SubmissionFeaturePipeline` 零 skew / winsorize 只作用训练标签 / 跑一轮 `fit+eval`,**Dec 分当 sanity bonus、不当选型依据**(§4.8)。
7. **【内存】全窗口 fit 的硬事实(2026-05-29 实测)**：单日 69347 行 × 433 特征(float32)=122MB;**Jun–Nov(123 天)≈8.5M 行,lgbm 子集 numpy ≈14GB**。决策口径(用户拍板):**不降采样、一切以交付老师为准、保证管线能在 32GB 上跑**(老师机器内存更高)。32GB 下全行 lgbm(~20GB 峰)够跑;**本机 16GB 跑不了全窗口 → 正式 Dec 演练需上 ≥32GB 机器**。
   - **内存精简已落地(2026-05-29,中档方案,本机编码+小窗口验等价,全量峰待 win 验)**：
     - **消 concat 2× 尖峰**：`meow.py` 改「预分配整窗 float32 矩阵 + 逐日流式填充释放」(`MeowEngine._build_window_frames`)。行数靠 `MeowDataLoader.countDate` 只读 h5 `axis1` 轴元信息廉价拿到(~1ms/天)、不加载数据块。build 峰 ~29GB→~15GB。
     - **成员顺序训练 + 末位释放源帧**：`SubmissionModelPipeline.fit_window(holder)` 消费式接收整窗帧并交出所有权,X1 先训(小)、**lgbm 末位训练前先释放整窗源帧**;直接把预抽 numpy 喂 `ExperimentRunner._fit_model_core`(新拆出),省掉「成员级 pandas 子表 + to_numpy」两份大矩阵 + 旧 `_member_xdf` 冗余 `.copy()`。lgbm 训练**持续峰 ~28GB→~19GB**。
     - **残留**：lgbm 列抽 numpy 那一刻仍有「整窗源帧 + lgbm numpy」短暂 ~28GB 瞬时尖峰(中档不消除,真要消需激进的按成员预分配、用户已否);32GB 下可survive。**端到端持续峰 ~30GB→~20GB**,达成中档目标。
     - **不动模型/不降采样/不缩窗口**;`fit`(非破坏入口)保留供单测/复用入参。新增单测:`test_fit_window_matches_fit`(释放路径预测等价)+ `test_streaming_build_matches_naive_concat`(流式构造 vs concat 逐元素一致,真实 2 天)。全 41 单测过 + `meow.py` 小窗口(训3评1)端到端冒烟通过。

## 待办队列

| # | 任务 | 状态 |
|---|---|---|
| 线性收尾 | C2/O4 单独 expanding 跑完落档：均干净、低于地板、X1 真互补 | ✅ 完成 |
| P4-1 | 树特征集 + 清脏列 + 模型 plumbing（前置研究）：**plumbing 已落地**（见「P4-1 落地」） | ✅ 就绪 |
| P4-2 | 模型 long-only 初筛：v1 OOM→提速→v3 干净跑完（`20260527_p42_modelselect_longscreen_v3`），双镜头读榜见上 | ✅ 完成 |
| P4-2′ | 树重要性扫描（Fork A）：run `20260527_p4_tree_importance_v1` 干净跑完，结论落档 `docs/实验记录.md`（剪交互 / trade被低估 / ofi没错杀） | ✅ 完成 |
| P4-2提速 | 训练行采样 plumbing（仅树族门控 + 全链路透传）+ 5 项新单测 + smoke 实测保真/提速/内存 + 文档收敛 | ✅ 完成 |
| P4-2b | 树侧精炼 long-only：M_tree_d7/d8 + M_lgbm_d4/d6（全用精简 V2 集，剪手工交互）→ run `20260528_p42b_tree_refine_v1`；**读榜完成，决赛名单已定：M_lgbm_d4 + M_tree_d8 + M_tree_d7，M_lgbm_d6 淘汰** | ✅ 完成 |
| P4-3 读榜+决赛 | 三候选带 `--dump-oof` 跑 expanding(X1/lgbm_d4 全量、d8 0.33 采样)。**结论**：`M_lgbm_d4` 0.0762(上限型,pooled 领先)、X1 0.0668(鲁棒锚,最坏折 0.0410 最好)、**`M_tree_d8` 跌破基线 0.0580/负折 -0.0073 → reject**(ExtraTrees expanding 塌,森林平均稀释近期 regime)。详见 `docs/实验记录.md` 2026-05-28 | ✅ 完成 |
| P5-a 加权 ensemble | X1 + lgbm_d4 OOF 加权融合读榜。**等权 zscore 同时超两成员(帕累托)**：pooled 0.0763 > lgbm 0.0727、最坏折 0.0491 > X1 0.0410、均值 0.0776。加 d8 反拖累(负折污染)。权重对 pooled 几乎无影响(0.0755–0.0765)、rank 模式被否(Pearson 非 Spearman)。**拍板锁等权 zscore(X1+lgbm_d4)当传统代表**——理由 minimax 正解 + 可辩护默认,非"峰值"。详见 `docs/实验记录.md` | ✅ 完成（已锁） |
| **lgbm_d4 Nov 验证** | run `20260529_p43_lgbm_d4_nov_review` 已完成：`M_lgbm_d4` Nov holdout **0.08967** vs R02 **0.06952**，样本外明显更强、未塌；传统侧样本外验证全部收口 | ✅ 完成 |
| **交付接线（主线①）— 接线+口径** | 两成员融合(X1 ridge + M_lgbm_d4)+并集特征(433列)+零缓存已接进 `submission_pipeline`/`meow.py`，真实数据 end-to-end 跑通。**融合口径已从 zscore 锁定到 `raw_mean`**（救 MSE/R²，见「交付接线收口」第 2 条）。小窗口冒烟通过(corr 0.106/R² 0.0065/MSE≈0)。commit `e005f62`@`feat/submission-blend-raw-mean` | ✅ 完成 |
| **交付接线 — 内存精简（下一步①）** | **中档方案已落地(2026-05-29)**：① concat 2× 改预分配流式填充(`_build_window_frames`+`countDate` 读 h5 轴元信息拿行数)→build 峰 29→15GB；② `fit_window` 消费式、lgbm 末位训练前释放整窗源帧 + 直接喂 `_fit_model_core` numpy(省成员级 pandas 子表+冗余 copy)→训练持续峰 28→19GB；端到端持续峰 ~30→~20GB(达中档目标，残留 lgbm 列抽 numpy 瞬时 ~28GB 尖峰，32GB 可survive)。**不动模型/不降采样/不缩窗口**。新增 2 单测(释放等价+流式构造等价)、全 41 测过、`meow.py` 小窗口端到端冒烟过。⚠️ **全量峰待 ≥32GB 机器实测**(本机 16GB 跑不了全窗口) | ✅ 编码+基本测试完成（全量峰待 win 验） |
| **交付接线 — Dec 演练（下一步②）** | 上 ≥32GB 机器(Windows 4060 若够内存最佳)跑 `fit(Jun,Nov)+eval(Dec)`：确认两成员不 OOM + 三指标(corr/MSE/R²)都健康 + 训练推理零 skew。**Dec 只当 sanity、不回灌选型**(§4.8)；这会动用 12 月 Final Holdout 一次性确认 | 🔜 下一步 |
| **交付接线 — 提交版减注释（下一步③）** | 老师评分明确"鼓励零注释/仅必要处注释"且查重(`meow/MEOW金融时序预测2.0.docx`)。**待做**：给 `meow/`+`src/` 提交路径单独出一版精简注释（与全局"详细中文注释"规则在本提交上冲突，提交前专门处理）| 🔜 下一步 |
| **DL 分支（主线②，Windows）** | 4060/PyTorch：序列管道（防泄漏开窗+归一化）→ LSTM（特征当序列，低风险）证序列有料 → DeepLOB raw-LOB（高风险）；**过同协议 + 11月 holdout 才能当代表**；05-31 回退点。提交端注意：老师 `fit()` 会现场重训，DL 需 GPU+训练时长，或确认带预训练权重是否被规格允许 | 待开 |
| 传统后续优化（推迟，保留方向） | 战略转型后**推迟**，有空才碰、上限有限：**① lgbm HPO**（num_leaves/lr+n_est/feature_fraction/min_child/L1L2，+0.003~0.008，0 holdout，性价比最高，要往大数据规模调）；**② 小波→GBDT**（老师 roadmap，+0~0.004，新 builder+防泄漏，ROI 低）；**③ MLP**（融合成员/加权，分数≈0，纯报告贡献度）。SVM/蒸馏/增量已分诊跳过 | 🅿️ 推迟 |
| P5-b OOF stacking | **仅当**加权平均明显不够：二层小线性融合器，**必须用 OOF**（§7.9）；黑箱、难辩护，慎用 | 🅿️ 推迟 |
| 🚩红线（Review 已用） | X1 进 11 月 Review Holdout（§4.8）：**已通过——X1 0.07367 vs R02 0.06952，+0.0042**（run `20260527_x1_review_v1`）；11 月预算已耗 1 次 | ✅ 完成 |
| 🚩红线（Final 未碰） | Final Holdout（12 月）全程未碰，**只在最终代表选定后一次性确认**、看完不改提交决策（§4.8） | 未动 |
| 清理 | 失败 stage `p35_interactions` 已归档（builder 移 .archive、摘 stage、删 spec I1、缓存移 .archive） | ✅ 完成 |

### P4-1 落地（2026-05-27，import + 12 项 unittest 通过，未跑实验）

- **树特征集（Fork A 拍板 = 先全含交互 + 跑重要性扫描）**：`eval_protocol.P4_TREE_GROUPS` = legacy + norm_core + ofi_safe + trade_impact + conditional_momentum + lag + roll + patch_summary + cross_rank + **regime_tree**。含手工交互（让重要性扫描一次回答「树是否真用得上交互」+ 反偏颇），保留 cross-z/cross-rank（树不自做按天截面归一），清广播脏列。
- **regime_tree group**（feature_registry）：regime 11 列去掉 `state_spread_cs`/`state_activity_cs`（一天一值广播常量，树会当日期身份乱切）= 9 列；`state_vol_cs` 保留。
- **模型 plumbing**（experiment_runner）：补 `huber`；加浅树变体 `tree_shallow`(ExtraTrees depth≤5)/`histgb_shallow`；`fit_model`/`run`/`run_with_groups`/`_evaluate_spec_on_fold` 加 `model_params` 穿透（预钉网格覆盖超参）；加 `_extract_tree_importance`（collect 路径线性 coef 为 None 时回退取重要性）。
- **M 系列网格 spec**（ALL_SPECS，预钉防多重比较）：线性 `M_en_X1`/`M_huber_X1`（X1 集，ridge-on-X1=既有 X1）；浅 ExtraTrees `M_tree_d4/d5/d6`；浅 HistGB `M_histgb_d3/d4/d4_lr03`（树大集）。depth 是浅树主正则器、leaf=500 当下限非约束。
- **winsorize 对树重评**：无需改码，用运行期 `--target-winsorize` 开关做 on/off A/B（默认沿用锁定 P1/P99）。
- **P4-2 跑法（long-only 初筛车道）**：`p0_eval_protocol.py --suite daily --spec-ids X1_... M_en_X1 M_huber_X1 M_tree_d4/d5/d6 M_histgb_d3/d4/d4_lr03 --profiles long_40d_5d --n-workers 1 --train-subsample-frac 0.33`（看门狗包装、日志写 `logs/`）。⚠️ 此 P4-1 时点的初版命令**漏了 `--train-subsample-frac`**，v1 全量行×单核跑树被内存看门狗杀；提速口径见上「P4-2 提速配置」+ AGENTS §4.9 第 7 点，正式跑法以 v2 为准。

## P4-2b 交接（下一会话接手点）

**P4-2b 代码改动已落地、long-only 已读榜**（lightgbm 已装、M_tree_d7/d8 + M_lgbm_d4/d6 specs 已加、`P4_TREE_GROUPS_V2` 已建并剪掉 `conditional_momentum`）。**接手点 = 跑 expanding 决赛**（run `20260528_p42b_tree_refine_v1` 已完成）：

- 已拍板送 `expanding` 的 3 个候选：`M_lgbm_d4`（上限型）+ `M_tree_d8`（树主代表）+ `M_tree_d7`（低 gap 保守备胎）。
- `M_lgbm_d6` 淘汰原因：被 `d4` 支配，且 gap 0.194 过热最重。
- 之后步骤见上「本周双轨 roadmap」+「待办队列」：**expanding → 锁最优单模型 → P5 加权 ensemble → Review Holdout → 提交演练**；DL 转 Windows 并行。

**看门狗阈值**：soft 12GB / hard 13GB（V2 集 peak ~9.3GB，安全）。

**红线提醒**：11 月 Review holdout 已用于 X1 确认（§4.8），**不得再用 11 月反复调参/选模型**（≤3 次预算已耗 1）；12 月 Final holdout 全程未碰、最终一次性用。选模型只在 Dev rolling / long / expanding 上做。

## 已完成基建（备查）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader 三件套 + 单测；主链路已切 FeatureLoader；9 stage 462 列。规格见 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + 输出结构 + baseline delta + make_decision；daily/gate/ridge/quick/full suite 可跑。
- 开跑前债务清零（#10–#20）全部落地，详见 git 历史与 `docs/specs/开跑前编码指导_评测口径与提速.md`。
