# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-06-04

## 当前阶段：路线锁定 = 传统保底 + 去相关 DL 集成（相关性已验 ρ=0.45 中等相关，集成已抬升）→ 按「冲 0.10 路线图」P1→P6 执行

> **最新复盘（2026-06-02，本会话）。下方"双 Pivot / 战略 / TCN 史 / 押截面"等块为历史脉络，保留备查；当前推进口径 = 本块 +「相关性判读结果」+「冲 0.10 路线图」+「🚩 交付隐患」。**

### ⏭️ 下一会话直接从这里开始

> **🎉 2026-06-04 午间 serve 焊接端到端验证 = GO + 提交件已打包（详见 `docs/实验记录.md` 2026-06-04「serve 端到端验证」）**：DL-on-raw 已按 option B 焊进 `meow.py` 交付链——`meow/dl_serve.py`（新 `DLServe`：fit 现训 K=3 seed、predict 三 seed 平均、`fuse_traditional_with_dl` 等权融合 warmup 缺行回落传统、**防御式降级**）+ `meow/meow.py` 4 处接线（签名不变）+ 7 单测绿。**端到端实测**（`experiments/_serve_delivery_eval.py`，与 `python meow.py` 同路径，老师 `eval.py` 口径，train Jun–Nov / eval Dec1–29 全窗）：传统 0.0812 → **融合 0.0919（破 0.09）**，Δic=+0.0107、**三指标全面改善**（R² 0.00645→0.00815、MSE↓）、等权 w=0.5 实测=最优、ρ=0.522 真去相关。DL 覆盖 86.2%（warmup 缺 13.8% 回落传统、稀释全窗分；**仅覆盖行融合 ic=0.0983** > 离线单 seed 0.0946，印证 3-seed 集成增益）。**时间锚**：传统 fit ~50min（瓶颈）/ 3-seed DL ~13.5min / 评测 ~5min。**缺口① 闭合**——老师 `python meow.py` 真拿 0.0919（非纸面）。**提交件**：`submission_build/MEOW_submission.zip`（42 文件，仅 meow/+src/+config/+models/+报告，无数据/缓存，闭包 import 验证通过）。**报告**：`docs/项目报告_草稿.md`（技术章节全 + serve 实测数已填）。**剩余（待用户拍板/补）**：① cert 3-seed 离线证明（run `20260604_xsection_raw_3fold_3seed` 运行中，强化报告严谨性）② 减注释提交版（与全局规则「详细中文注释」冲突，需用户取舍）③ 报告 §7 成员分工（需团队实情）。

> **✅ 2026-06-04 凌晨 DL-on-raw × 新传统 融合判决 = GO（自主接力跑完，详见 `docs/实验记录.md` 2026-06-04 + 待办队列融合行）**：机器2 DL-on-raw（直接吃 raw 59 通道）经 `git add -f` 推来，本机用**新传统（含 rx_micro，Dec sanity 0.0812）× DL** 在三窗同口径离线融合（复刻老师 `eval.py` 三指标口径，inner-join 交集行）。**结果**：交付折 delivery 0.0826→**0.0946（破 0.09，+14.5%）**、fold1 0.0854→**0.0971**、fold2 0.1013→**0.1044**，三窗 Δic 全正、**三指标全面改善**（R²↑/MSE↓，不是只抬 Pearson）、ρ=0.45–0.54 真去相关、等权≈最优（best_w 0.40–0.45）。**多 seed**：DL 两 seed 平均后融合 **fold1 破 0.10(0.1016)、fold2 逼近 0.11(0.1098)**（seed 集成=免费杠杆）。**逐日稳健性**：融合不仅 pooled 高、更托住最坏日下行——delivery 传统最坏日 −0.069→融合 **+0.030（从亏转盈）**、三窗融合逐日胜传统 15–17/20，实证「集成=时段崩盘保险」。**这是「12% 真入口=直接建模 raw」的第一个正反馈**——raw 直接建模确实产出对传统真去相关的一臂。**下一步三选项待你拍板**：A.机器2 补 delivery seed43（最低成本，实测交付口径验 ~0.10 外推）/ B.把 DL 焊进 `meow.py` 交付链 serve（工程量大：torch 进链 + 内存/时间预算 + 签名）/ C.机器2 练更强 DL 继续压 ρ 往 0.11+ 推。**诚实边界**：这是离线融合判决（用已落盘预测算融合上限），非 serve 端到端；delivery 仅 DL seed42，~0.10 是按 fold seed 平均增益外推、未实测。

> **🚨 2026-06-02 夜 重大发现（颠覆"数据锁死"前提，优先读 `docs/原始数据盘点与盘口建模诊断.md`）**：老师说**同数据同题目有人 12%+**（pooled Pearson，同尺子，我们仅 0.0776 → 差距是方法欠债不是数据上限）。诊断证实 **raw 是富 LOB+订单流+226 步日内序列**（62 列），**"无 LOB/数据锁死/raw⊆433" 是错的**；433 严重欠用 raw（挂撤单 20 列只压成 4 个静态比率、盘口形状没建模、全序列压成统计量），**DL 一直喂 433 摘要、从没直接吃过 raw**。**冲 12% 真入口 = 直接建模 raw 盘口（DeepLOB 式，规格里规划过但因假前提没建）**，不是在 433 上抠融合那一个点。融合（Tier-1 ~0.085）降为低风险保底。**下一步待用户拍板**：是否走 raw 盘口建模主线；若走，先在 fold2（传统 0.0904）跑最小 raw 原型看能否超过，再铺三折。

1. **✅ P1 已完成（2026-06-02 本会话）= 传统 ridge NaN 已修 + 合并并 push master**：根因 = X1 ridge 成员（`StandardScaler→Ridge`）输入 float32，~1e15 量级列在 StandardScaler 居中 / X^TX 累加时数值抵消 → 病态/近常数列 → ridge cholesky 失败回退 30GB svd → 内存紧则分配失败成 NaN。**修法（已落 `src/experiment_runner.py`）= 对 StandardScaler→model 的线性 Pipeline 成员(ridge/elasticnet/huber/mlp) 在 fit/predict 前转 float64**（fit 端 `_fit_model_core` line 731、predict 端 line 762；树/提升族尺度无关保持 float32）。全窗 Dec sanity（空闲 ~20GB 紧张态实跑）：**非 NaN，Pearson=0.0775 / R²=0.00585 / MSE~2.4e-5 / 峰值 23GB（cholesky，已绕开 30GB svd）**。已 cherry-pick 到 master（`962a970`）并 push origin（master/feat 均已推）。**判读**：0.0775 是 float64 干净解（旧 0.0803 是 float32 退化 SVD 解），单 20 天窗内属噪声范围持平、R² 反升、且彻底消除内存依赖型 NaN——对交付划算；不在 Dec 单窗调参，真正传统腿判据看 P2 的 3 个选型折。
2. **⏭️ 第一刀 = P2（task #8，P1 已解锁）**：造对齐 DL 折的无泄漏传统滚动预测——对 3 个选型折 + 交付折逐折 train→predict、落 OOS 预测、按 (date,symbol,interval) 对齐 DL。这是后续集成评估 / 残差训练的地基。
3. **再 P3**（task #6，依赖 P2）：① 三档消融 + 每档 ρ/集成读数；② **残差训练（DL 目标改 y−ŷ_trad）= 压 ρ 主手段，冲 0.10 的核心**。
4. 全景 + 依赖 + 判决点 = 见本文件「### 冲 0.10 路线图」。分工：本路线（集成）本侧做；用户另开更强 DL-only。
5. **本会话产物（gitignored，本机可见）**：DL Dec 预测 `results/dl/20260602_corr_probe_dl_v1/preds/preds_validation_fold0_seed42.csv`（0.0588）；传统 lgbm Dec 预测 `results/dl/_corr_probe/trad_preds_20231201_20231229.csv`（健康 0.0803）；ridge Dec 预测 `…/trad_preds_ridge_20231201_20231229.csv`（NaN，实证用）；相关性报告 `results/dl/_corr_probe/corr_report.json`。新脚本：`experiments/dump_trad_preds_members.py`（逐成员落盘）、`experiments/analyze_dl_trad_corr.py`、`experiments/dump_trad_preds.py`、`experiments/_probe_trad_features.py`（特征量级探针，一次性）。

### 一句话现状

`20260602_sweep_xsection_v1`（截面模型 = 上一轮特化主攻）已**手动停于 6/9 折**（画像足够，详见 `results/dl/.../STOPPED_MANUALLY.md`，该目录 gitignore 不入库）。`val_corr.mean=0.0624`（6 折全正，min 0.0546，**R² 已被 rescale 兜回 ~0**），**仍在传统保底 0.0776 之下**、且只比纯 GRU-on-433（0.0585）高 6.7%。**每折 `best_epoch=1–2`** → 1–2 个 epoch 即过拟合、可学增量极少。

### 为什么三轮（TCN-raw → GRU-433 → XSECTION 特化）一直没破传统：四层根因

1. **表象**：0.0091 → 0.0585 → 0.0624 一路涨，但都没摸到 0.0776。
2. **一直在修症状、没动根子**：TCN 的"截面盲区" → 上截面模型（只解决"看不看得见"）；GRU 的 R² 微负 → 上 `λ·corr`+rescale（R² 确实修好了）。**但卡死分数的不是这两个。**
3. **本质①——同源数据 + 更灵活模型 = 没有免费午餐**：无 LOB、raw⊆433，DL 与传统抢**同一份信息**；`best_epoch=1–2` 实锤"可学增量极少"，传统线性/树早把大头吃掉 → DL 只能持平或略低。
4. **本质②——时间轴锁死**：146 天 → 只有 ~146 个时间截面，学不出**能跨时间泛化**的复杂截面函数（老师恰考跨时间泛化）。
   - **上一轮的具体失手**：截面模型整个存在理由是抓 cross-z/cross-rank 维，但 §8.2.1 决定**输入端故意不喂 cross-z**、赌 set-attention 腿自学——没学出来（0.0624≈GRU 0.0585，截面腿基本没起量）。传统 0.0776 靠的就是把 cross-z/rank 当显式特征喂进去；我们却让 DL 在 146 天里从零硬学这维 = **绑着一只手打**。当初不喂的最硬理由"怕抹掉量纲毁 R²"已被 rescale 兜底、过时。

### 双 Pivot（本会话代码已落、测试绿；run 待跑）

- **Pivot 1 — 换目标：测"DL 作去相关增强"而非"DL 单挑赢传统"。** 0.06→0.10 非调参距离，DL-alone 大概率到不了；现实 0.10 路线 = 传统 0.0776 保底 + DL **去相关一臂**集成。**关键未知 = DL↔传统预测相关性（从没测过，预测从没存盘）。** → 加 `--dump-preds` 存 DL+传统 Dec 逐票预测，算相关性 + 集成增益，**这个数决定集成有没有肉**。
- **Pivot 2 — 给截面模型松绑：cross-z 喂回输入端。** rescale 已保 R²≥0，cross-z 不再有量纲顾虑 → 做成截面模型 forward **第一层**（masked 截面 z，`cross_z` hparam 默认关）。三档消融 `纯GRU / 截面无cross-z（=上轮）/ 截面+cross-z` 验它能否破 0.06 天花板、摸过传统。

### 本会话代码改动（已落、全 DL 套件 85 tests green）

| 改动 | 文件 | 用途 |
|---|---|---|
| `--dump-preds` 逐票预测落盘 | `src/dl_trainer.py`（`_dump_fold_preds`）+ `experiments/run_dl.py`（CLI flag + `_spec` 注入 `dump_preds_dir`） | GRU/截面统一落 `date,symbol,interval,label,pred` CSV（默认关、零开销） |
| 传统侧 Dec 预测落盘 | `experiments/dump_trad_preds.py`（新） | `MeowEngine.fit(Jun–Nov)+predict(Dec)`，不碰 engine 代码 |
| DL↔传统相关性 + 集成增益分析 | `experiments/analyze_dl_trad_corr.py`（新） | join 出去相关度 / 各自 vs 标签 / 等权·最优静态权集成 vs 标签 |
| 截面 cross-z 输入归一 | `models/dl_models.py`（`_build_xsection_module` forward 第一层 masked 截面 z + `cross_z` hparam）+ `tests/test_dl_xsection.py`（4 新单测：去截面水平 / off 敏感 / mask 忽略 pad / 置换等变） | Pivot 2 消融开关；单点控制 fit/predict & GPU/CPU 两态 |

### 诚实预期

最可能现实结果 = **集成 → ~0.085–0.09**；真正 0.10 是 stretch（需 DL 够正交 **且** cross-z 有实质起量）。两 Pivot 的数出来，再定 0.10 够不够得着，还是转"承认传统天花板 ~0.085–0.09、把交付做扎实"。

### 相关性判读结果（probe 已出，2026-06-02）

DL Dec 预测（`20260602_corr_probe_dl_v1`，cross_z 关）× 传统 Dec 预测（lgbm 成员，自检 pooled 0.0803）join 20 天 120 万行，`analyze_dl_trad_corr.py` 出（落 `results/dl/_corr_probe/corr_report.json`）：

| | DL（单） | 传统-lgbm（单） | 等权集成 |
|---|---|---|---|
| daily-IC 均值 | 0.0702 | 0.0542（最坏日 −0.0724） | **0.0727** |
| pooled corr | 0.0588 | 0.0781 | **0.0805** |

- **DL↔传统去相关：pooled 0.4453 / daily-IC 0.4756 = 中等相关，不是冗余**（之前"高度相关→大概率杀死"的先验被推翻）。两指标上 DL/传统换位（DL 赢 daily-IC、传统赢 pooled）= 互补形状。
- **集成数学（两去相关信号可达相关性 `R²=(IC₁²+IC₂²−2ρ·IC₁·IC₂)/(1−ρ²)`）**：代当前 pooled 数 → 最优合并 ≈ **0.083**（实测等权 0.0805 吻合）= **当前牌的集成天花板，非 0.10**。
- **关键结论——卡 10% 的是 ρ，不是 DL 强度**：扫 ρ，若 ρ→0（完全去相关）则当前强度合并 = √(0.0781²+0.0588²) ≈ **0.098**；ρ=0.45 不变要顶到 0.10 则需 DL-alone ~0.091（比传统还高，不现实）。**→ 真正杠杆 = 把 DL 做正交（压 ρ、同时 IC 不塌），不是单纯把 DL 练猛。**
- **caveat（务必打折）**：仅 Dec 单 20 天窗；"传统"这里是 lgbm-alone（非锁定等权集成 0.0776）；"最优静态权 0.0738"是同窗挑权的过拟合上界、忽略。GO/NO-GO 必须看 3 个选型折。

### 冲 0.10 路线图（2026-06-02 锁定：传统保底 + 去相关 DL 集成）

> **路线定调**：DL-alone 到不了 0.10；现实路线 = 传统保底 + DL 去相关一臂、做集成。**两层目标**：Tier-1 = 传统+DL 等权集成稳到 **~0.083–0.085**（基本到手，只差跨折验证）；Tier-2 = 压 ρ 把集成推向 **0.09–0.10**（stretch，~20–30%）。
>
> **分工**：**本路线（集成）= 本侧负责**，稳扎稳打、工程化；**用户另开"更强 DL-only"**，不依赖本路线，若产出更强 DL 可直接当本路线 Tier-2 更强的一臂喂进 Phase 4。
>
> **GPU 是单卡串行资源** → 无 GPU 的 CPU 活（Phase 1/2/6）可与 GPU 活（Phase 3）并行排在不同会话。

| Phase | 任务 | 依赖 | 资源 | 产出 / GO-NO-GO |
|---|---|---|---|---|
| **P1 修传统腿** ✅ 完成 | ① ridge NaN 修复 = 线性 Pipeline 成员 fit/predict 转 float64（已落 `experiment_runner.py`）② 重验 Dec sanity：空闲 ~20GB 紧张态非 NaN、Pearson 0.0775 / R² 0.00585 / 峰值 23GB（绕开 30GB svd）③ 已 cherry-pick master `962a970` + push origin | — | CPU/内存 | ✅ 可靠传统预测（不再内存依赖型 NaN）。**P2 及下游已解锁** |
| **P2 传统滚动预测** | harness：对 3 个选型折 + 交付折，逐折 traditional train→predict、落逐票预测、按 (date,symbol,interval) 对齐 DL | P1 | CPU | 无泄漏传统 OOS 预测，供残差训练 / 去相关惩罚 / 折级集成评估 |
| **P3 DL 去正交（核心，最不确定）** | ① 三档消融（纯GRU/截面off/截面cross-z）：测每档 DL IC，join P2 出每档 ρ 与集成增益 ② **残差训练**：DL 目标改 y−ŷ_trad（用 P2 无泄漏传统预测）= 压 ρ 主手段 ③（可选）loss 加去相关项 | P2（②③）；①可先跑 DL 侧 | GPU 串行 | 候选 DL 配置（各带 IC + 对传统 ρ）。**冲 0.10 成败在这** |
| **P4 集成 + 验收** | ① 定融合器（等权零自由参数首选 + 量纲处理保 MSE/R²）② 3 个选型折评估集成 vs 传统单独（看最坏折）= **主裁** ③ 交付折 Dec 读数 | P2+P3 | CPU 分析 | **GO-NO-GO**：等权集成跨 3 折是否稳赢传统（最坏折正）。赢→进 P5；平→守 ~0.085 |
| **P5 交付集成** | ① DL 腿接进提交链（fit() 里连 DL 一起重训，符合"交付=方法非权重"）② 交付内存/时间预算（含 torch）③ fit/predict 签名仍对 docx | P4 赢 | CPU+GPU | 含 DL 的可交付提交链 |
| **P6 既有交付尾巴（独立并行）** | 全量内存峰实测 / 提交版减注释 / fit-predict 签名核验 | — | CPU | 与主线无依赖，任何空档插做 |

**会话分配建议**：
- **CPU 会话组**：P1 → P2，然后 P4 分析、P6 尾巴。GPU 忙时照样推进。
- **GPU 会话组**：P3 实验，一晚一族（先三档消融，再残差训练）。
- **依赖红线**：P1 不修完，P2/P3②③/P4 的传统预测都是 NaN，全堵；**所以 P1 是第一刀**。

**两个判决点**：
1. P4② Tier-1：等权（传统+DL-on-y）跨 3 折是否稳赢传统最坏折 → 否则回落传统单独交付。
2. P3② 残差后：ρ 是否真降、集成是否向 0.09 爬 → 卡在 ~0.085 就承认天花板、停止冲 0.10、把交付做扎实。

**诚实分界**：P1/P2/P4/P5/P6 是扎实工程，稳交 ~0.083–0.085；**10% 全押在 P3 能否压 ρ 又不塌 IC**——这一环不确定，别拿它当承诺。

---

> **战略（历史脉络，保留备查）**：传统侧已收口、可提交（保底代表已锁）；主线 = ① 交付接线收口（仅剩签名核验/减注释尾巴）+ ② **DL 冲 0.12**（Windows 4060 / PyTorch）。
>
> **TCN 海选结论（run `20260531_search_tcn_raw_v1`，2026-05-31）**：4 trial / 2 正 2 负；best = seq_len=32 / hidden=32 / layers=4 / val_corr=0.01653；两个 seq_len=48 全负。信号：短窗（32）优于长窗（48），管线无根性问题。
>
> **TCN expanding 认证深度分析（run `20260531_valid_tcn_raw_v1`，2026-05-31）**：3 folds × 3 seeds = 9 点，mean=0.00912 / std=0.02845 / min=-0.0384 / max=0.0603 / positive_rate=7/9。逐折明细：
>
> | 折 | 训练截止 | 评测窗口 | seed=42 | seed=43 | seed=44 |
> |---|---|---|---:|---:|---:|
> | fold 0 | Jul 28 | Aug 1–7 | +0.0097 | +0.0070 | +0.0289 |
> | fold 1 | Aug 4 | Aug 8–14 | **-0.0303** | **-0.0384** | +0.0302 |
> | fold 2 | Aug 11 | Aug 15–21 | +0.0046 | +0.0100 | **+0.0603** |
>
> **三个问题同时成立**：
> 1. **绝对量级差一个数量级**：mean=0.0091 vs 传统靶子 0.0776，差 8.5×；最好单点 0.0603 也只刚刚摸到传统 corr_min（0.0491）。
> 2. **val_R² 9/9 全负**（-0.224 到 -0.005，最坏 fold1/seed44）：MSE 比直接预测均值还烂，老师精度分中 R² 那 1/3 判零。
> 3. **fold 1 种子彩票**：同一训练集 seed 42/43 深负、seed 44 反手 +0.030，种子方差≈0.068——典型欠拟合形状，不是”这个折分布特殊”。
>
> **根因（已修正）：截面盲区，不是 max_epochs**。max_epochs=4 偏短只是症状（best_epoch 多落 2–4），真因是张量契约 `[B,L,C]` 一次只看一只票、RawChannelAdapter 又无任何截面归一 → TCN **整整看不见 cross-z/cross-rank 那一维截面 alpha**（传统 0.0776 的主力恰在此）。加 epoch 也变不出它根本看不到的截面排名。
>
> **决策（2026-06-01 多轮讨论拍板）：DL 主攻"截面建模"，TCN-on-raw 否决；评测协议重设计落 `AGENTS.md §十一`。**
> - **为什么截面**：数据无 LOB、raw 与 433 同源（无更富数据）；时间轴 146 天锁死、有效 N 极小，而**截面轴每天 309 票样本海量**——DL 唯一喂得饱、且正好是传统只用死板 cross-z/rank 糊过去的轴，就是截面。完整推演 + alpha 全图见 `NOTE.md`。
> - **冲 0.10 的腿（五条有肉）**：① 截面联合建模（主攻）② 个股时序 GRU（基线 token）③ `MSE+λ·corr` 损失对齐（免费杠杆，顶老师 Pearson 1/3）④ DL+传统集成（保底 + 时段崩盘保险）⑤ 共因子/大盘均值（便宜一臂）。**"scale 更多数据"因数据锁死已关。**
> - **交付=方法非权重**：老师现场用我们 `fit()` 重训重评 → 考方法跨时间泛化 → **rolling 每折 = 老师考试的一次复现**。故评测改为**锚定扩展 walk-forward + purge/embargo + 最坏折/bootstrap 读数**（§十一），废旧"海选单切分"。
> - **选型粒度**：结构族为主（一晚一族）+ 命令内小 HPO（结构子旋钮 + λ）+ seed 探针（3 个平均）+ 其余固定**重正则**默认；一命令两档预算（筛选→认证）。
> - **DL 头号风险 = 时段过拟合**（容量大/无先验锚/优化随机，TCN 种子方差 0.068、R² 全负为证）→ 三层防御：rolling 检测 / 重正则+多种子降低 / 集成兜底。
>
> **⬇️ 进展（2026-06-01 至 2026-06-02）**：① **协议已落代码**——`Stage.SWEEP` 一命令两档（档1 小网格×近2折×2seed 按最坏折选冠军 → 档2 冠军×5折×3seed）+ `build_dl_folds(fold_select="recent")` 自 Dec 倒贴五段不重合 + `enumerate_grid`，**34 test 全过 + 真实数据 CLI 端到端跑通**；② **截面模型方案与卡带均已落地**（因子化两腿：共享 GRU 时序腿 + set-attention 截面腿 + 零初门控残差、置换等变零身份、`MSE+λ·corr` 截面对齐；为什么不用时空联合/图网络见 `NOTE.md`「截面模型怎么设计」+ 规格 §8.2/§8.2.1；代码已并入当前工作区，待 maxfold 内存演练）；③ **GRU-on-433 基线 SWEEP 已收官**（run `20260601_sweep_gru433_v1`，冠军 seq32/h32/L1，val_corr.mean 0.0585 / min 0.0391 / R² -0.0071，完整见 `docs/实验记录.md`），实验链收敛为“优先走 `FeatureLoader/data/features` 磁盘缓存，提交链继续 raw 现算”。purge 说明：`fret12` 日内不跨夜，1 日 embargo 已等价 purge，未另搭丢行逻辑。④ **瘦内存管线 + GPU 三态搬运 + 资源监控已落地并经 GRU maxfold 演练实测**：最大折（131 训练日 / 全 309 票）单折峰值 RSS=24.1GB（本机 34GB，安全、不 OOM），冷/热两跑 `val_corr` 一致（0.0075，`max_epochs=4` sanity 探针，**不可外推正式精度**）；正式跑全程资源监控落 `resource_log.csv`（GPU 利用率 78–94% / 显存稳 ~6.4GB / 系统内存 ~30GB / 进程 RSS ~21GB），崩溃可回溯。规格落 §12。
>
> **传统保底已锁**：等权 raw_mean 融合 [X1 ridge + M_lgbm_d4]，expanding 均值 0.0776 / Dec sanity Pearson 0.0803 / R²=+0.00465。**DL 候选需在新协议的 3 个 20 交易日选型折（Nov 末倒贴）均值 + 最坏折两镜头上给出清楚优势，才考虑换代表；否则 DL 作增强叠在传统核心上、保底交付不破。**

### DL 整晚实跑命令（GRU-on-433 基线，走 SWEEP 新 rolling）

> Windows 4060 / PyTorch。脚本已**自举 `sys.path`**（免设 PYTHONPATH，Win/Mac 通吃）。数据需在本机 `data/*.h5`（Jun 1–Dec 29，144 天，已 gitignore、不随仓库同步）。`--max-symbols 0` = 全 309 票；`--device cuda` 强制上卡（无卡显式报错、不静默退 CPU）。命令为**单行**，PowerShell / cmd 直接粘。

**第 0 步 · 上卡前 5 分钟 sanity**（确认 torch+GPU+FeatureAdapter+SWEEP 链路在 Win 上通，再委托整晚）：

```bash
python experiments/run_dl.py --stage sweep --model gru --adapter feature_433 --device cuda --start 20230601 --end 20230831 --val-window 5 --step 5 --min-train-days 30 --max-folds 2 --fold-select recent --grid-seq-len 16 --grid-hidden 32 --grid-layers 1 --hparams max_epochs=2,patience=2 --seeds 42,43 --max-symbols 20 --run-id 20260601_sweep_gru433_sanity_v1 --out-dir results/dl
```

出 `results/dl/20260601_sweep_gru433_sanity_v1/summary.json`（`status: ok` + 有 `champion`）即链路通。

**第 1 步 · 整晚正式**（≈31 次 fit：档1 4 网格×近2折×2seed + 档2 冠军×5折×3seed，预算 §十一·11.6）：

```bash
python experiments/run_dl.py --stage sweep --model gru --adapter feature_433 --device cuda --start 20230601 --end 20231229 --val-window 12 --step 12 --min-train-days 40 --max-folds 5 --fold-select recent --grid-seq-len 16,32 --grid-hidden 32,64 --grid-layers 1 --hparams dropout=0.2,weight_decay=0.001,max_epochs=30,patience=5 --seeds 42,43,44 --max-symbols 0 --run-id 20260601_sweep_gru433_v1 --out-dir results/dl
```

- **折**：`recent` 自 Dec 29 倒贴 5 段 ×12 天非重合 = Oct1–Dec29，训练锚 Jun1（83→131 天，§十一·11.2）。
- **重正则**：dropout 0.2 / weight_decay 1e-3 / 早停 patience 5（§十一·11.5）；loss 暂纯 MSE（λ·corr 下一晚加）。
- **早上看** `summary.json`：`val_corr.mean`（vs 传统 0.0776）、`val_corr.min`（**最坏折，决策主镜头**）、`val_r2_mean`（盯 R² 是否逼近 0，TCN 当时 9/9 全负）、`champion`；逐折逐 seed 明细 `fold_metrics.csv`、档1 网格 `trials.csv`。
- **旋钮**：想更狠 `--grid-layers 1,2`（网格翻倍→47 fit）；想更快先 `--max-symbols 60` 抽样。
- **注意（2026-06-01 更新）**：`GRU-on-433` 实验链**不再默认每折现算**，而是优先读取 `data/features/` 磁盘缓存；因此正式跑数前先确保所需交易日的特征缓存已构建完整。提交链 `meow.py` 仍坚持 raw 现算，与实验缓存层职责分离、不冲突。若缓存未就绪，代码会显式回退旧路径；正式大跑前不应依赖该慢路径。
- **运行期资源管理（2026-06-01 实跑沉淀，权威见规格 §12）**：① GPU 喂数走三态 `_GpuWindowSource`——装得下显存才 resident（事前 `mem_get_info` 预算判断、绝不盲分配再 OOM），装不下走**预取流式**（后台线程 gather→pinned、主线程 `non_blocking` 拷卡、在途仅 3 batch，内存有界永不 OOM，实测稳态 GPU 利用率 ~79%）；② **瘦内存管线**把每折峰值从 ~24GB 压到 ~12GB（逐日缓存有界蓄水 `_day_cache_cap=32` + 预分配流式装填 + Normalizer 分块 fit + 原地白化 `own_features=True` + 三段分别现算），119 天全票折外推 ~22GB（本机 **~34GB**，安全）；③ 正式跑记 GPU 利用率/显存/RSS + 每折分阶段计时行（stderr），崩溃可回溯。纪律：**宁可慢也绝不让长跑因显存/内存溢出崩**。

### 传统侧（已收口、保底代表；全量明细见 `docs/实验记录.md`）

- **传统代表已锁**：等权 `raw_mean` 融合 **[X1 ridge + M_lgbm_d4]**（expanding 均值 0.0776 / pooled ~0.0763 / 最坏折 0.0491）。两成员各自过样本外：X1 Nov 0.0737、lgbm_d4 Nov 0.0897（vs R02 0.0695）。**传统天花板 ~0.085–0.09，冲 0.10+ 改押 DL。**
- 特征侧 P1–Q4（线性）+ P4 选模型 + P5 融合全程结论已沉淀 `docs/实验记录.md`；规则正文 AGENTS §四/§七。
- **12 月 Final 从未执行**；DL 新协议不用三层、改**锚定扩展 walk-forward**（§十一），且 Dec 进 rolling 当最近折用满（不再留权重 lockbox）。

## 交付接线收口（传统交付链，活，剩尾巴）

现状：`meow.py` 壳层（fit/predict/eval）已接 `src/submission_pipeline.py`、零缓存强制、两成员融合 + 并集特征（433 列）已落地、真实数据 end-to-end 跑通。融合口径锁 **`raw_mean`**（等权 raw 平均，保 `fret12` 量纲——老师精度分 = MSE+Pearson+R² 各 1/3，zscore 会毁 MSE/R²）。内存中档精简已落地（预分配流式 + 末位释放源帧，持续峰 ~30→~20GB），本机小窗口验等价 + 冒烟过。

口径要点：
- **serve 端归一化必须用新数据自算、严禁冻结训练期先验**（默认 `raw_mean` 不做截面标准化、天然无此风险；仅 `per_day_zscore_mean` 诊断模式适用）。
- **全量训练、无藏划分器**：`fit/eval` 给什么区间就训/评什么全量（无降采样）；train/eval 划分只在驱动入口（env 变量），老师替换驱动即可（`meow/meow.py:111/121`）。
- 特征从 raw 现算、禁用 `data/features` 缓存。

剩余尾巴（见待办队列）：① 全量内存峰上 ≥32GB 机器实测；② 提交版减注释；③ `fit/predict` 签名对照 docx 核验（另会话）。

### 🚩 交付隐患（2026-06-02 发现）：X1 ridge 成员**仅因当时空闲内存差异**就会 NaN —— ✅ 已修复（2026-06-02 本会话）

> **✅ 已修复并合并 master**：按下方「P1 修法 (a)」落地——`_fit_model_core` 对线性 Pipeline 成员 fit 前转 float64、`predict` 端对齐取 float64（`src/experiment_runner.py` line 731 / 762）。全窗 Dec sanity（空闲 ~20GB 紧张态）非 NaN、Pearson 0.0775 / R² 0.00585 / 峰值 23GB（cholesky，绕开 30GB svd）。已 cherry-pick master `962a970` + push origin。下文根因链保留备查。

**现象**：本会话在本机（34GB Win）跑传统 Dec dump，融合预测**整列 1462884 行全 NaN**；而用户此前在 **master、同一台机器**上交付演练 `fit(Jun–Nov)+eval(Dec)` 跑通、得 Pearson 0.0803 / R²=0.00465。

**根因链（已实证）**：
1. 433 提交特征里有 **~1e14–1.7e15 量级的未归一化列**（特征探针：`nonfinite=0` 但 `maxabs≈1.7e15`）；`meow.py` 窗口矩阵**硬编码 float32**（`meow/meow.py:65/84`）。
2. **标准化早就有、不是缺标准化**：X1 ridge 成员 fit 走 `_fit_model_core`（`src/experiment_runner.py:614–618`）= `StandardScaler → Ridge`，但输入 `x` 是 **float32**（line 599 注释明写"x 已是 float32 特征矩阵"）。1e15 量级列在 float32 下做 StandardScaler 居中（`x−mean`，两者都 ~1e15）= **灾难性抵消**：float32 仅 ~7 位有效数字、1e15 绝对精度才到 ~1e8，比这更小的特征信号全部丢失 → 缩放后是数值垃圾 / 近常数列。
3. 垃圾/近常数列 → X^TX 近奇异 → **ridge cholesky 失败 → sklearn 回退 svd**（栈：`sklearn/_ridge.py:309 _solve_svd`）；全窗 875万×K 经济 svd 需 **~28–30GB**（U 矩阵），只在机器近 idle（≥~30GB 空闲）装得下：master 演练 idle 时 fit 峰值 ~28GB 跑通；本会话 dump 空闲仅 ~23GB → `init_gesdd failed init`（分配失败）→ ridge 系数 NaN → 等权融合 `(NaN + lgbm)/2 = 全 NaN`。
4. **同机、同码（git diff master...HEAD 传统链零差异）、同数据 → NaN 与否仅由当时空闲内存决定**；lgbm 树模型尺度无关、整窗 fit 正常，只有 ridge 这半坏。**注意**：实验选型链的 X1（Nov 0.0737）走 FeatureLoader（可能 float64），与提交链 float32 现算的 X1 **数值口径本就不同** → 提交侧 X1 ridge 一直在 float32 垃圾缩放上解，修复后预测会变（应改善或持平，须验）。

**交付风险**：老师机器内存紧 / 有其他负载，或换更小内存机时，`python meow.py` 全窗 fit 的提交输出可能**静默变 NaN**；且 ridge 每次都在跑这个 30GB svd（本该几毫秒 cholesky），又慢又脆。

**P1 修法（改共用 `src/` 代码，顺带覆盖 master；别在跑 run 时动）**：
- **(a) 推荐——根因解 = 线性成员走 float64**：在 `_fit_model_core`（`experiment_runner.py:596`）的线性分支（ridge/elasticnet/huber，均 `StandardScaler→model`）把 `x` 转 **float64** 再进 Pipeline，`_predict_with_baseline`（`experiment_runner.py:754`）对应转 float64。居中不丢精度 → 缩放后正常 → cholesky 毫秒级成功、内存小、**彻底绕开 30GB svd**。X1 成员子集 float64 ~10–14GB，本机够。**不要**只改 `solver='cholesky'`（根因是缩放精度，不是 solver 选择）。
- 验收：改完重跑 Dec sanity（`experiments/run_submission_full_window.py` 或 `experiments/dump_trad_preds_members.py`）确认传统 ≥0.0803；**内存紧（人为占住部分 RAM）+ idle 两态各跑一次**全窗 fit，确认都非 NaN、Pearson 稳定 ~0.08。
- 合并 master 覆盖交付。

## DL 主线：地基设计要点（详见 `docs/specs/DL实验设计规格.md`）

- **固定脊柱 + 可换卡带**：协议/窗口/归一化/指标/配置 = 不可变脊柱；InputAdapter + ModelCartridge = 可换卡带。换模型（LSTM-on-features → DeepLOB-on-rawLOB）只动两块卡带、脊柱零改。
- **PyTorch 封死卡带内**（torch/nn.Module/optimizer/GPU/训练循环都在 `fit/predict`），脊柱 torch-free。
- **评测 = `AGENTS.md §十一`**：锚定扩展 walk-forward + purge/embargo（主裁严格前向）、测试段铺 Oct–Dec 不重合、最坏折+bootstrap 读数、结构为主+小 HPO+重正则、一命令两档预算。旧"海选单切分+早杀"作废。
- **配置 = 分布式声明 + 中央组装**：frozen `RunConfig`；三层拆分（枚举集中各块文件顶 / 实现注册挨着 registry / 本次选择进 RunConfig）；run_id 手工语义 + `config_fingerprint` 防漂移。
- **超参只搜结构 3 旋钮**（序列长/hidden/层数）+ 随机搜索 + 早杀。
- **冲 0.12，不验"序列是否有料"**（已是先验）。

## 🚧 并行编码分工（worktree，2026-06-02 凌晨开）

> **本会话主线 = 守护正在收官的 `20260601_sweep_gru433_v1`（14/15 fit，~00:45 出 summary）。** 趁机在仓库外侧 2 个 worktree 并行铺地基；**Agent 由用户在各 worktree 内启动**（本会话不 spawn）。

### 🚨 资源横幅（所有 worktree 编码 Agent 必读）

**✅【00:31 更新：主训练已于 00:30 自然收官（summary 已落），GPU 全空、内存释放 → 本横幅的高负载/GPU 约束即刻解除，可放开跑测试与 maxfold 演练。】**

（以下为收官前原始横幅，留档）**主训练进程 PID 19560 仍在跑（占 GPU ~6.2/8GB + 系统内存 ~29–32/34GB），预计 ~00:45 收官。在它跑完前——**

- ❌ **严禁任何高负载 / 上 GPU 的操作**：`pytest tests/test_dl_pipeline.py`（含真实 h5 + torch/CUDA 用例）、`--device cuda` 任何脚本、maxfold 内存演练、训练 / SWEEP，**一律不许跑**（抢显存/内存可能把主跑挤崩）。
- ✅ **可做**：写 / 改代码、读文件、设计、torch-free 的轻量 `python -c` import 自检（不加载数据、不碰 CUDA）。
- ⏳ **测试统一推迟**：所有单测 / 集成测试**等主跑 ~00:45 收官、GPU 空闲后再统一跑**。
- 注：worktree 不含 `data/*.h5`（gitignore，未复制）→ 天然跑不了依赖数据的测试，正好契合本横幅。

### worktree 映射（均从 `feat/dl-foundation` 切出；各根目录已留自包含 `AGENT_TASK.md`）

| WT | 目录（外侧） | 分支 | 任务 | 文件归属（互不相交，可干净 merge） |
|---|---|---|---|---|
| **WT-A** ✅ 已合并 | `../MEOW-wt-incrdump` | `feat/dl-incr-dump` | **增量落盘**（实际只动 `run_dl.py` + 测试，未碰 `dl_search.py`） | `experiments/run_dl.py`、`tests/test_dl_infrastructure.py` |
| **WT-B** ✅ 已合并 | `../MEOW-wt-xsection` | `feat/dl-xsection` | **截面数据改造 + `MSE+λ·corr` loss + 可微 Pearson/截面 IC 指标 + 测试脚手架** | `src/sequence_dataset.py`、`models/dl_models.py`、`config/model_config.py`、新增 `src/dl_losses.py`、`tests/test_dl_xsection.py` |

### WT-A — 增量落盘（防中断打水漂）✅ 已合并 `feat/dl-foundation`（2026-06-02）

- 现状（已解决）：档1 全跑完才写 `trials.csv`、档2 全跑完才写 `fold_metrics.csv`/`summary.json`；今晚档2 黑盒 5h、若中途崩 15 fit 全丢——必要性已坐实。
- 已落地：**每 trial 完成即 append `trials.csv`+flush+fsync；每折完成即 append `fold_metrics.csv`+flush+fsync**（`_IncrementalCsvWriter` 表头幂等、零行兜底、与旧 `_dump_csv` 逐字节等价）+ `progress.jsonl` 逐事件时间线 + `summary.partial.json` 每折刷新快照。`summary.json` 仅成功收官落（=「在 ⇔ 跑完」完成标记）。`--resume` 复用已落盘 trial/(seed,fold) 跳过不重算（网格确定 → trial_id 即下标重建超参；折 key=`(random_seed,fold_id)`）。infra 测试 36 通过。

### WT-B — 截面改造 + loss（主攻地基）✅ 已合并 `feat/dl-foundation`（2026-06-02）

设计权威：`NOTE.md`「截面模型怎么设计」+ 规格 §8.2 + `AGENTS.md §十一·11.5`。**本会话又焊死了 loss 口径（见 🔑）。**

- **已落地**：`CrossSectionIndexer` / `CrossSectionDataset` 按 `(date,interval)` 聚票，卡带内部用 `from_whitened` 零复制把 trainer 传入的 `SequenceDataset` 重组成截面快照；predict 后按 `argsort(flat_label_rows)` 映射回逐票窗口序，`dl_trainer.py` / 协议 / Orchestrator / SWEEP 全零改。
- **`CrossSectionCartridge`**：`ModelKind.XSECTION`，绑 `FEATURE_433`；torch = 共享 GRU 时序腿 + `nn.MultiheadAttention` 1 块（无位置 / 无 ID，带 padding mask）+ **零初门控残差** `z=h+γΔ`（γ=0 退化纯 GRU）+ Linear 头。显存源 `_GpuCrossSectionSource` 走 resident / CPU-gather 两态，`snap_batch` 控瞬时峰值。
- **🔑 loss 口径（本会话焊死，务必照做）**：
  1. `loss = 量纲项 + λ·(1 − 截面内 Pearson)`，λ∈{0,0.3} 走 SWEEP 小网格。
  2. **corr 项 = 截面内 Pearson**(同一 `(date,interval)` 跨票) = 老师 pooled corr / daily-IC 那个被打分的量。**逐票 GRU 上的批内 corr 只是粗代理 → 截面模型才让"直接优化 corr"对齐到被打分的 corr。**
  3. **【定稿覆盖，2026-06-02】目标层级 = corr 力争最大化 / R²≥0 硬底**。R²≥0 不靠调 λ，靠**永远开启的训练段全局 OLS 线性 rescale `a·ŷ+b`**(fit-on-train / apply-on-val，零泄漏，不设开关)——重标定后 **R²=corr²**，corr>0 即自动 R²≥0、且 corr 不掉。**于是 corr 与 R² 不再冲突，选 λ 只按"最大化 val corr"**，三指标照报仅供核对(不必为护栏牺牲 corr、不必 Pareto 纠结)。
  4. **不要任何开关**：损失项固定 MSE（不做 MSE/Huber 切换）；rescale 始终开。corr 项 = **数值稳定可微 截面内 Pearson**，独立成 `src/dl_losses.py` + 单测。
  5. λ·corr **先在 GRU 卡带验**(便宜、确认杠杆)，再带进截面卡带。

## 🚀 下一长跑启动方案（合并后待执行）

> 背景：WT-A 增量落盘与 WT-B 截面/loss 已并入当前 `feat/dl-foundation` 工作区；下一步先验收，再决定是否启动正式长跑。

**新会话启动顺序**：
1. ✅ **跑全测试套件**确认绿（GPU 空）：2026-06-02 已跑 `python -m pytest tests/test_dl_pipeline.py tests/test_dl_infrastructure.py tests/test_dl_xsection.py`，81 passed（仅 pytest cache 权限 warning）。
2. ✅ **maxfold 内存演练**：2026-06-02 已跑 `20260602_maxfold_xsection_v1`，最大旧窗口折 131 训练日 / 全票 / CUDA / 截面卡带通过；峰值 proc RSS 20.76GB、系统内存 33.15/33.91GB、显存 3237MiB，未 OOM。该折重于新协议 3 个 20 交易日选型折，可作为长跑前硬闸通过。
3. 启动正式长跑（见下），后台 + `resource_monitor.py` + 守护。

**长跑命令（主攻 = 截面模型 SWEEP；以合并后实际 ModelKind/CLI 为准，新会话先 `run_dl.py --help` 核对）**：

> ✅ **2026-06-02 协议修订已落代码（与当前代码一致）**：选型折改为 **Nov 末倒贴 3 段×20 交易日**，不是严格日历 Sep/Oct/Nov；当前边界为 `20230831–20230927`、`20230928–20231102`、`20231103–20231130`。**Dec 抽出做交付对齐折**——冠军定死后跑 `train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)` × 1 seed、写 `summary.json` 的 `delivery` 块、不参与排名（验交付链）。CLI 采用显式方案 A：`--end 20231130` 作为选型/训练截止，`--delivery-eval-end 20231229` 作为交付折 eval 末日；因 `--embargo 1`，delivery scoring 会跳过 20231201。

```
python experiments/run_dl.py --stage sweep --model xsection --adapter feature_433 --device cuda \
  --start 20230601 --end 20231130 --delivery-eval-end 20231229 --val-window 20 --step 20 --min-train-days 40 --max-folds 3 \
  --fold-select recent --grid-seq-len 32 --grid-hidden 32 --grid-layers 1 \
  --hparams dropout=0.2,weight_decay=0.001,max_epochs=15,patience=5,lambda_corr=0.3 \
  --seeds 42,43,44 --max-symbols 0 --run-id 20260602_sweep_xsection_v1 --out-dir results/dl
```

### 2026-06-02 正式长跑托管（已启动）

- **目标 run_id**：`20260602_sweep_xsection_v1`
- **托管方式**：不用易卡死的前台长等待；改为一个隐藏 PowerShell 监督脚本负责巡检，实际训练进程与 `resource_monitor.py` 仍各自独立后台运行。
- **巡检周期**：每 **30 分钟** 固定做一轮健康检查，落 `logs/supervisor_20260602_sweep_xsection_v1.log` 与 `results/dl/20260602_sweep_xsection_v1/supervisor_status.jsonl`。
- **检查内容**：进程是否存活、stdout/stderr/结果文件是否继续增长、近半小时资源曲线是否异常、是否出现“进程还在但日志长期不动”的静默卡死。
- **异常策略**：
  - **正常完成**：保留全部日志与资源曲线，写完成标记后停监控。
  - **资源类失败**（显存/内存 OOM、系统强杀）：先保全现场，再按降配阶梯自动重启。当前阶梯只改 `--max-symbols`，依次 `0 -> 240 -> 180 -> 120`，并为新 attempt 生成独立 `run_id` 后缀与 incident 记录。
  - **非资源类失败**：先保全现场与根因摘要，停止自动重启，留待下一次 agent 醒来后只在 GPU 空闲时修代码。
- **现场路径**：
  - 训练 stdout: `logs/<effective_run_id>.log`
  - 训练 stderr: `logs/<effective_run_id>.err`
  - 监督日志: `logs/supervisor_20260602_sweep_xsection_v1.log`
  - 资源曲线: `results/dl/<effective_run_id>/resource_log.csv`
  - 事件时间线: `results/dl/20260602_sweep_xsection_v1/supervisor_status.jsonl`

### 2026-06-02 正式长跑中的 infra 现象（仅观察，不代表已定案）

- **观察对象**：`20260602_sweep_xsection_v1`
- **当前观察到的两个现象**：
  1. **GPU 确实在跑，但常态没有打满**：以 `resource_log.csv` / `nvidia-smi` 为准，GPU 在大多数训练时段处于活跃状态，显存约稳定在 `~4.6GB / 8.2GB`，但 util 常见仅 `25%~40%`，明显不是“吃满卡”的形态。
  2. **fold 边界仍存在可见的 CPU barrier**：每折之间会出现 GPU util 掉到 0 的空窗；已完成折的 `stderr timing` 显示 `load + norm + build_ds` 仍占固定时间。就当前已落盘样本看，这部分不是总墙钟主耗时，但肉眼可见、值得继续抠。
- **当前理解（暂不下结论）**：
  - 现有“预取流式 / pinned / non_blocking 拷卡”主要覆盖 **fit 阶段内部的 batch 供给**，能减少训练中 GPU 等 batch；
  - 它**没有自动消掉** fold 级别的冷启动工作：按折读盘、fit normalizer、重建 `CrossSectionDataset` / mask / index、predict 结束后的切换等；
  - 因而当前更像是“两类问题并存”：① fold 内 GPU occupancy 仍偏低；② fold 间固定 CPU 准备仍存在。
- **下一步 infra 方向建议（后续 run 再做，不在当前正式 run 中途改）**：
  1. **先做单折吞吐 profiling**：固定模型语义，只测 batch / `snap_batch`、AMP、`torch.compile`、predict 批大小对 GPU util / samples-per-sec 的影响，先确认“是 batch 太轻、kernel 太碎，还是 host 同步点太多”。
  2. **拆开 fold 级 barrier**：把 `load / norm / build_ds / predict` 继续细分到函数级计时，确认下一刀该砍在 H5 读取、normalizer fit、截面重组，还是推理落盘。
  3. **评估可复用/可预热的折间状态**：重点看“同一 fold 跨 seed 是否在重复做相同 CPU 准备”，以及“下一折 CPU 预热”是否值得做、是否会把内存峰推高到不可接受。
  4. **再决定是否扩大模型/批尺寸**：若 profiling 证明确实是单步算量太轻，再考虑通过更大 batch 或更重宽度提高 GPU 饱和度；这一类改动属于实验语义变化，需与纯 infra 优化分开记。

### 2026-06-02 正式长跑当前中间表现（进行时，不作最终结论）

- **运行位置**：`20260602_sweep_xsection_v1` 认证阶段进行中；截至当前已完成 **3 / 9 folds**（均为 `seed=42`），任务继续跑，不中断。
- **screening 冠军读数**：`screen_corr_mean=0.0630`，`screen_corr_min=0.0546`。
- **认证阶段当前均值**（仅基于已完成 3 折）：`val_corr.mean=0.0599`，`val_corr.min=0.0546`，`val_corr.max=0.0672`，`positive_rate=1.0`，`val_r2_mean=-0.00245`，`val_r2_min=-0.00931`。
- **当前解读**：结构没崩、三折全正，但量级仍在 `~0.06` 附近，离“冲 0.10”目标差距明显；此读数只能视为阶段性画像，不能当最终判决。
- **行动口径**：本 run 继续跑完以保留完整 seed / fold 画像；**下一步主线仍是尽快改进 infra / 模型吞吐与结构表现，继续冲分**，不把当前中间数当收口结论。

- **折**：选型 3 折为 Nov 末倒贴 20 交易日窗：`20230831–20230927`、`20230928–20231102`、`20231103–20231130`；档1 筛选取最近 2 折×2seed→最坏折选冠军；档2 认证冠军×3折×3seed=9 fit；**交付折**冠军×`train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)`×1seed=1 fit（显式 `--delivery-eval-end 20231229`，只报不选）。结构固定时共 14 fit；若结构网格或 λ 网格扩大，档1 按 `grid_size×2×2` 增加。
- λ 走 `--hparams`（决策 Q1）；**rescale 永远开、cross-z 不做**（决策 Q2）；结构固定昨晚冠军 seq32/h32/L1；max_epochs 30→15（best_epoch 实测 4–7，省时不损精度）。
- **Tier-1 fallback**（若截面演练没过）：先跑 **GRU-on-433 + λ·corr + rescale**（`--model gru` + `lambda_corr=0.3`，proven 管线、低风险），验杠杆 + 把 R² 翻正，不浪费今晚。

**早上看**：选型三镜头——val_corr.mean / 最坏折 val_corr.min（决策主镜头）/ val_r2（**应被 rescale 兜在 ≥0**）；**外加交付折 `delivery` 块**（当前方案 A 严格保留 1 日 embargo，因此是 Dec4–Dec29，不是完整 Dec1–Dec29；若要和传统 0.0803 完全字面对齐，需另行决策 delivery 是否改 `embargo=0`）。目标 corr 力争 >0.10、R² 不为负。§11.9：0.0776 / 0.0803 均作为传统背景参考，正式换代表看 Nov 末倒贴的 3 个 20 交易日选型折。

**loss / rescale / cross-z 定稿**：见上「并行编码分工 §WT-B 🔑」+ 各 worktree `AGENT_TASK.md`（已与两 agent 同步）。

## 待办队列

| # | 任务 | 状态 |
|---|---|---|
| 传统全程（P0–P5 + 交付接线主体） | 特征侧 / 选模型 / 融合 / 交付融合接线 + raw_mean + 内存中档精简，全部完成、明细沉淀 `docs/实验记录.md` | ✅ 收口 |
| **DL 地基设计** | 脊柱+卡带架构 / 配置管理 / D0 交付物，定稿落 `docs/specs/DL实验设计规格.md`（**评测口径已升级至 `AGENTS.md §十一`**，原海选+expanding 两段作废） | ✅ 完成（评测口径已升级） |
| **DL D0 地基实现（主线②）** | torch-free、Mac CPU 跑通，按 spec §9 全部落地：① `src/sequence_dataset.py`（WindowIndexer 惰性[B,L,C]/不跨日/不跨票/因果对齐/warmup + Normalizer fit-on-train/可 identity + subset_by_dates）② `src/dl_protocol.py`（DLFold 三段切分/embargo/4指标逐字对齐 experiment_runner/assert_folds_causal/summarize_folds）③ `src/dl_trainer.py`（`SequenceTrainer(BaseTrainer)`，鸭子类型注入 adapter+cartridge_factory+raw_loader，产 `FoldResult`）④ `models/dl_models.py`（InputAdapter 接口+IdentityAdapter+FeatureAdapter 包装433+numpy 参考模型 ReferenceZero/Last 当泄漏探测器）+`models/registry.py`（枚举→类注册+required_adapter 校验）⑤ `config/` 6 文件（frozen dataclass+枚举顶部 + `RunConfig` 组装/校验/fingerprint）⑥ `tests/test_dl_pipeline.py`（六项验收闸 **17 test 全过**：端到端/参考模型低分/无泄漏因果/不跨日跨票/归一化只用训练统计/config 校验 + 真实 h5 FeatureAdapter）。**import 约定：src/config/models 三目录平铺，入口 `PYTHONPATH=src:config:models`** | ✅ 完成 |
| **DL 基础设施实施** | `experiments/run_dl.py`（Orchestrator：组装+冻结 RunConfig+dump JSON+SEARCH→Searcher / VALIDATION→定参认证+落 trials/fold_metrics/summary）+ `src/dl_search.py`（采样器 choice/int/uniform + overrides 收窄 + EarlyKillPolicy 钩子桩 + Searcher 排名）+ `RawChannelAdapter`（59 通道）。**seq_len 走 trainer、hidden/layers 走卡带 hparams** 边界写死在 Searcher。21 test 全过 + 真实数据 CLI smoke 跑通。早杀实现仍推后（无 epoch 可杀，等 torch 卡带回调） | ✅ 完成 |
| **README 重写** | 已重写为「DL 工程地基说明 + 代码/文档索引」（DL 规格入口、`config/`/`src/dl_*`/`models/` 结构、import 约定、依赖说明 torch[D1/D2]/psutil[演练]、DL 测试运行方式） | ✅ 完成 |
| **TCN-on-raw（已否决）** | 实现完成 + smoke 通过。海选 `20260531_search_tcn_raw_v1` best 0.01653；expanding `20260531_valid_tcn_raw_v1`：mean=0.00912 / val_R² 9/9 全负（-0.224~-0.005）/ fold1 种子方差 0.068。**真因=截面盲区**（`[B,L,C]` 一次一票、无截面归一，看不见 cross-z/rank 那维 alpha；非 max_epochs）。**结论：否决**——raw 与 433 同源无更富数据、raw-LOB 终局不存在，路线转向**截面建模** | ❌ 否决 |
| **新评测协议落 run_config** | 两档骨架已落代码：`Stage.SWEEP` 一命令两档 + `build_dl_folds(fold_select="recent")` + `enumerate_grid` + 最坏折/R² 双镜头 + `--device/--grid-*` CLI。✅ **2026-06-02 协议修订已落代码**：① 选型折改为 **Nov 末倒贴 3×20 交易日窗**（当前边界 `20230831–20230927`、`20230928–20231102`、`20231103–20231130`，非严格日历月）；② 新增显式 `--delivery-eval-end` 交付对齐折——冠军定死后 `train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)`×1seed、写 `summary.json.delivery`，同时在 `fold_metrics.csv` 追加 `profile_name=sweep_delivery` 行，但不参与档1排名/档2汇总（验交付链、杜绝据 Dec 调参）。相关测试 81 passed。 | ✅ 完成 |
| **截面模型方案（主攻）** | **方案已拍板落档**（`NOTE.md`「截面模型怎么设计」+ 规格 §8.2）：因子化两腿 = 共享 GRU 时序腿（每票日内路径编码）+ set-attention 截面腿（跨票、置换等变零身份、零初门控残差）+ per-票头；`MSE+λ·corr` 截面对齐；张量契约 `[L,C]`→整截面 `[N,L,C]`+mask（WindowIndexer 按 (date,interval) 聚票；v1 不做 cross-z，见规格 §8.2.1）。业界 STGNN/关系排序同构、为什么不用时空联合/图网络已记 | ✅ 方案完成 |
| **截面模型卡带（主攻）** | **已合并（2026-06-02）**：`CrossSectionCartridge`（`ModelKind.XSECTION`，绑 FEATURE_433）+ `CrossSectionIndexer`/`CrossSectionDataset` + `_GpuCrossSectionSource` + masked 截面 Pearson loss + 训练段 OLS rescale；接已跑通 SWEEP，下一步跑测试与 maxfold 内存演练。 | ✅ 已合并待验收 |
| **GRU-on-433 基线 + 损失对齐** | **基线认证收官**（run `20260601_sweep_gru433_v1`，纯 MSE/λ=0，13:43→00:30，详见 `docs/实验记录.md` 2026-06-02）。冠军 seq32/h32/L1。**三镜头**：val_corr.mean **0.0585**（positive **15/15**、std 0.013）/ 最坏折 **0.0391** / **R² −0.0071（14/15 微负）**。结论：**结构成立**；**R² 微负 = MSE/corr 矛盾实锤**。WT-B 已把 `MSE+λ·corr` + 训练段 OLS rescale 接入 GRU，λ 经 `--hparams lambda_corr`，下一步正式扫 λ∈{0,0.3}。 | ✅ 损失对齐已合并待验收 |
| **XSECTION v1 正式 run（已停于 6/9，判决出）** | `20260602_sweep_xsection_v1` **手动停于 6/9 cert folds**（seed 42/43 各 3 折）：`val_corr.mean=0.0624 / min=0.0546 / max=0.0723 / R²≈0`；screen 冠军 `0.0630/0.0546`。**判决**：结构没崩、R² 已修好，但量级 ≈ 纯 GRU 0.0585、**仍在传统 0.0776 之下**，`best_epoch=1–2` = 可学增量极少。继续跑 seed 44 不改方向 → 提前停，转双 Pivot（见顶部复盘）。 | ⏹️ 有意停止 |
| **Pivot 1 · DL↔传统去相关 + 集成增益（定路线）** | 代码已落：`--dump-preds`（`dl_trainer`+`run_dl`）/ `dump_trad_preds.py` / `analyze_dl_trad_corr.py`。探针 `20260602_corr_probe_dl_v1` 跑 Dec 逐票预测中 → 接传统 Dec dump → 分析。**那个数决定集成能否把 0.0776 抬到 ~0.09。** | ⏳ 探针运行中 |
| **Pivot 2 · cross-z 喂回截面输入 + 三档消融** | 代码已落：`_build_xsection_module(cross_z=)` forward 第一层 masked 截面 z + `cross_z` hparam（默认关）+ 4 新单测（全 DL 套件 85 passed）。规格 §8.2.1 已更新。**消融 `纯GRU/截面off/截面+cross-z` 待跑**（Pivot 1 出数后定是否优先）。 | 🔜 代码就绪待跑 |
| **DL infra 吞吐画像与折间 barrier 拆解** | 来自正式 run `20260602_sweep_xsection_v1` 的现象：GPU 已上卡但常态 util 偏低（常见 `25%~40%`），fold 间仍有 `load / norm / build_ds` 空窗。**当前只记现象与方向，不预设结论**：后续在 GPU 空闲时做单折 profiling（batch / `snap_batch` / AMP / `torch.compile` / predict 批大小）+ 函数级计时，判断瓶颈落在 kernel 粒度、host 同步、读盘、normalizer 还是截面重组。 | 🔬 观察中（下个 infra 小回合） |
| **正式结果同步目录（双机追踪）** | 根目录新增 `tracked_results/`，专门提交“小体量、正式、可复盘”的结果文件；首批纳入 TCN search / TCN expanding / 传统 Dec 全窗口 sanity 指标，供双机同步与后续深挖分析 | ✅ 已建立 |
| 🚩 **交付接线 — X1 ridge 仅因空闲内存就 NaN** | 根因=433 特征含 ~1e15 未归一化列 + `meow.py` float32 窗口 → ridge cholesky 失败 → 回退 svd 需 ~30GB → 内存紧则分配失败成 NaN。**已按 (a) 修复**：线性 Pipeline 成员 fit/predict 转 float64（`experiment_runner.py` line 731/762），X^TX 在 float64 累加保持正定、cholesky 毫秒成功、绕开 30GB svd。验收：空闲 ~20GB 紧张态全窗 Dec sanity 非 NaN、Pearson 0.0775 / R² 0.00585 / 峰值 23GB。已 cherry-pick master `962a970` + push origin（master/feat）。 | ✅ 完成（2026-06-02） |
| **rx_micro 新特征接入交付链（lgbm 成员）** | 49 列被欠用 raw 信号（`rx_`/`rx2_`/`rx3_`）注册为 registry stage、**仅挂 M_lgbm_d4**（X1 ridge / 既有 433 完全不变）。三折全票 lgbm **+0.0034**（0.0765→0.0799，三折全正）；Dec sanity **0.0793 / R²0.00613 / 峰值 22.67GB / 无 NaN**（vs 接入前 0.0803 / 0.00465：Dec 单窗 Pearson −0.001 属噪声、R² +0.0015）。判据看三折，Dec 只 sanity。详见 `docs/实验记录.md` 2026-06-03。**全程未碰 master。** | ✅ 接入+验证完成（feat 分支） |
| 交付接线 — 全量内存峰实测 | 已实测：含 rx_micro（**482 列**）全窗 `fit(Jun–Nov)+eval(Dec)` 峰值 RSS **22.67GB**；本机 psutil 实测物理内存 **31.6GB**（非旧记 ~34GB）。配套修复 = `_fit_model_core` 线性成员 `StandardScaler(copy=False)`+`Ridge(copy_X=False)` 砍冗余 float64 副本（峰值 ~40→22.67GB，纯内存优化不改数值），否则 ridge fit 在 31.6GB 机上 OOM | ✅ 已实测 |
| 交付接线 — Dec 全窗口演练 | 用户已在另一侧完成全窗口 `fit(Jun–Nov)+eval(Dec)`：**Pearson=0.0803 / R²=0.00465 / MSE=2.3645e-05**。结论：提交链量纲健康、端到端可跑；**Dec 只当 sanity、不回灌选型** | ✅ 完成 |
| 交付接线 — 提交版减注释 | 老师鼓励零注释/仅必要处 + 查重；给 `meow/`+`src/` 提交路径单独出精简注释版（提交前专门处理） | 🔜 下一步 |
| 交付接线 — fit/predict 签名核验 | 对照 `meow/MEOW金融时序预测2.0.docx` 确认 `MeowEngine.fit/eval/predict` 能让老师替换路径跑通（predict 当前接特征帧，老师可能按路径取数）。代码侧已确认无藏划分器、全量训练。**另起会话专办** | 🔜 遗留（另会话） |
| **run_dl 增量落盘（防中断打水漂）** | **已实现 + 已合并 `feat/dl-foundation`（merge `feat/dl-incr-dump`，2026-06-02）**：`trials.csv`/`fold_metrics.csv` 逐 trial/逐折 `append+flush+fsync`（`_IncrementalCsvWriter` 表头幂等、零行兜底、与旧 `_dump_csv` 产物逐字节等价）+ `progress.jsonl` 时间线 + `summary.partial.json` 每折快照；`summary.json` 仅成功收官落（=完成标记）；`--resume` 复用已落盘 trial/(seed,fold) 跳过不重算。infra 测试 **36 通过**（新增 `TestSweepIncrementalDump`：逐字节等价 / 人为中断保留前序 / resume 续跑无重复无遗漏） | ✅ 完成（已合并） |
| ✅ **ridge 也加 rx_micro → 融合验证（判决 GO，2026-06-03）** | 验证"给 X1 ridge 也挂 rx_micro、两腿都加新特征"后 ridge+lgbm 融合是否更好。脚本 `experiments/probe_ridge_blend_newfeat.py`（fold1+fold2 两折，task `bxg8m6gwv` 已完成，结果 `results/dl/_p3_ridge_blend/summary.json`）。**结果**：fold1(Sep28–Nov02) ridge 0.0528→0.0558 / blend_base 0.0696→blend_both 0.0722；fold2(Nov03–Nov30) ridge 0.0755→0.0802 / blend_base 0.0904→blend_both 0.0934。**判据全过**：两折都 `ridge_new>ridge_old`（+0.0030/+0.0047）且 `blend_both>blend_base`（+0.0026/+0.0030），且 `blend_both>blend_lgbm_only`（+0.00065/+0.00156，证实"只 lgbm 加"会稀释增益）。**已接入**：`submission_pipeline` 的 X1 ridge member groups 末尾加 `"rx_micro"`（两成员都吃）；单测 6 passed。**🚨 Dec 全窗 sanity OOM（task `b5lfsmjla` 失败，日志 `logs/dec_sanity_bothlegs_v3.log`）**：ridge 157列 fit 时 `StandardScaler._incremental_mean_and_var` 要分配 (8548841,157) float64 ≈10GB nan_mask 临时，叠加整窗 xtrain 16.5GB（ridge 非末位、held 给 lgbm）+ ridge f64 10.7GB，超本机物理 31.6GB → `_ArrayMemoryError`。**这是交付链隐患**（老师 ~32GB 机跑 `python meow.py` 同样会 OOM；`copy=False` 压不掉 sklearn 内部临时）。**用户选 A → 逐成员现算+fit 重构已落、OOM 解决**：`meow.py` fit 改逐成员循环（每成员单独现算自己 groups→fit→释放，峰值压到单成员级）+ `submission_pipeline` 加 `begin_fit/fit_one_member/member_specs` + `mdl.py` 转发 + `feat.py`/`meow.py` groups 参数化；单测 7 passed（含「逐成员==整窗 fit_window」等价网 `test_per_member_fit_matches_window_fit`）。**Dec 全窗 sanity 重跑通过：Pearson 0.0812 / R² 0.00645 / 峰值 22.30GB**（三版本最高：原始 0.0803、只 lgbm 0.0793、两腿 0.0812）。**已落未 commit**：`feature_registry`(build_rx_micro+注册+schema probe) / `submission_pipeline`(ridge+lgbm 都挂 rx_micro + begin_fit/fit_one_member) / `meow.py`+`mdl.py`+`feat.py`(逐成员现算+fit) / `experiment_runner`(线性成员 copy=False)。详见 `docs/实验记录.md` 2026-06-03。 | ✅ 判决 GO + 逐成员重构解决 OOM（Dec 0.0812 / 22.30GB，7 passed） |
| 🔥 **DL-on-raw × 传统 融合（机器2 DL 推来，2026-06-04 接力中）** | 第二台机器 DL-on-raw（XSECTION_RAW，吃 raw 59 通道）经 `git add -f` 推来（commit `89b315c`，run `20260603_xsection_raw_2fold_2seed_zscore`，含 fold1/fold2/delivery 逐票预测；`results/` 被 ignore 故必须 -f 强推，本机已 `git checkout origin/feat -- ` 取到工作区）。窗口映射：DL cert_fold0=传统 fold1(Sep28–Nov2)、cert_fold1=传统 fold2(Nov3–Nov30)、delivery=Dec4–Dec29。**阶段1 融合分析**（旧传统×DL seed42，`experiments/probe_blend_dl_trad.py`，落 `results/dl/_blend_dl_trad/summary.json`）：DL 真去相关 **ρ=0.42–0.51**；三窗口等权融合 = fold1 0.0808→**0.0956**、fold2 0.0979→**0.1062(破0.10)**、delivery 0.0785→**0.0935(破0.09)**；blend_raw≈blend_z(量纲一致)、≈理论上界(等权够用、不必调权)。**阶段2 跑中**（task `ba0tjn2k9`，`experiments/dump_submission_dec_preds.py`，日志 `logs/dump_trad_dec_newfeat.log`）：dump 含 rx_micro **新传统** Dec 逐票预测(fit Jun–Nov 逐成员 + predict Dec)，完成后用**新传统(0.0812)×新DL** 重算 delivery 融合 = 交付口径(理论估 ~0.095)。**接力计划**：阶段2 完成 → 改 `probe_blend_dl_trad` 的 delivery 传统路径指向 `_blend_dl_trad/trad_dec_newfeat_preds.csv` 重算 → 加厚(多 seed/权重/可选 fold1·fold2 新传统 expanding) → 写判决到 docs/CLAUDE。**诚实边界**：这是离线融合判决(用已落盘预测算融合能到多少)；把 DL 焊进交付链 serve(meow.py 连 DL 一起出预测)是下一步工程、需用户定。 **【2026-06-04 阶段2/3 已出】** 新传统 Dec dump 自检 pooled Pearson=**0.0812** ✓（与 Dec sanity 对齐）；**delivery 交付口径融合（新传统×新DL seed42，等权 raw）= Pearson 0.0946 / R² 0.00867 / MSE 1.71e-5**，比新传统单独（同交集行 0.0826）**+0.0120(+14.5%)、破 0.09**，且**三指标全面改善**（R² 0.00867>传统单腿 0.00645、MSE 1.71e-5 更低）——融合不是只抬 Pearson、是老师三项各 1/3 一起好；**等权≈最优**（当窗 best_w=0.45≈0.5）、**ρ=0.454** → 零自由参数等权方案站得住、调权反是过拟合。**【三窗同口径 + 多 seed 已出，判决 GO，详见 `docs/实验记录.md` 2026-06-04】** 三窗全新传统同口径等权融合（DL seed42）：fold1 0.0854→**0.0971**、fold2 0.1013→**0.1044**、delivery 0.0826→**0.0946**，Δic +0.0117/+0.0031/+0.0120 **全为正**，ρ=0.45–0.54 真去相关，三窗 R² 全正/MSE 健康，best_w 0.40–0.45≈0.5（等权≈最优）。**多 seed（cert 折 seed42/43）**：DL 两 seed 平均单腿 +0.007–0.010、融合后 **fold1 破 0.10(0.1016)、fold2 逼近 0.11(0.1098)** → seed 集成是免费杠杆。**判决=GO**（DL-on-raw 真去相关、三窗稳定正增益、交付口径破 0.09 且三指标全改善）。**诚实边界**：离线融合判决（非 serve 端到端）；delivery 只有 DL seed42，补 seed43 按外推可达 ~0.10（未实测）。**下一步待用户拍板**：A.机器2 补 delivery seed43（最低成本验 ~0.10）/ B.DL 焊进 `meow.py` 交付链 serve（工程量大）/ C.机器2 练更强 DL 继续压 ρ。 **【2026-06-04 严谨性审查 + 决策】** 审出 3 缺口：① **致命=交付链 serve 零 DL**（grep 实锤 meow/+submission_pipeline 无 torch/DL → 老师跑 `python meow.py` 只得传统 0.0812、拿不到 0.0946，融合现为离线纸面数）② **只 2 折**（机器2 run `n_folds=2`，缺 fold0 Aug31–Sep27）违反"三折全票"判据 ③ **DL seed 方差**（fold2 seed42 0.0818/seed43 0.0899 差 0.008）老师单 seed 重训不可复现。**用户决策= 走 B（焊 DL 进交付链，老师机器有 GPU 可行），但先补第三折、三折全稳再焊**。补折进展：**传统四折新预测已 dump 齐**（`_blend_dl_trad/trad_fold{0,1,2}_newfeat_preds.csv`+`trad_dec_newfeat_preds.csv`；fold0 train0601-0829/eval0831-0927 自检 0.0727）；**等机器2 跑 DL 3 折 run（命令仅 `--max-folds 2→3`+换 run-id，余不变）补 DL fold0**。焊接骨架已摸（mdl 薄包装委托 SubmissionModelPipeline / `SequenceTrainer(spec,adapter,cartridge_factory,raw_loader).run_fold` / run_dl 组装）：fit 加 DL 多 seed 训练、predict 加等权融合、DL 走 raw_loader 自读 raw、warmup 缺行用传统填；待三折过关 + 老师评分约束（流程/数据量/时间/提交格式/内存）定稿。 **【2026-06-04 三折判决已出 = GO】** 机器2 推来 3 折 run `20260604_xsection_raw_3fold_2seed`（commit `05079d5`），本机 `git checkout` 取 preds + 重指 `probe_blend_dl_trad.py` 到新 run、四窗全用新传统同口径。**四窗等权融合(seed42)**：fold0 Aug31–Sep27 **0.1004**(Δ+0.0132)、fold1 **0.0971**(+0.0117)、fold2 **0.1044**(+0.0031)、delivery **0.0946**(+0.0120) → **四窗 Δic 全为正、最坏折(fold2)+0.0031 仍正 = 三折全稳判据过**；R² 四窗全正(+0.009~+0.011)/MSE 健康(~1.5–2e-5)；ρ=0.45–0.55 真去相关；**新补 fold0 反而是融合最强折(破0.10)**、fold1/fold2/delivery 与上轮一字不差复现(DL 单 seed 可复现)。**seed 平均(cert seed42+43)**：cert 三折全破 0.10（fold0 0.1040/fold1 0.1016/fold2 0.1098）=免费杠杆。**严谨性缺口②(只2折)已闭合。** 剩余：缺口③ delivery 多 seed 仍未实测(写死 seed42=0.0946)、缺口① serve 链零 DL(纸面数)——二者在焊接阶段一并解决。**下一步**：① 用户问老师 5 件事(流程/数据量/时间/提交格式/内存)定 serve 约束 ② 焊 DL 进 `meow.py`(option B)。 | ✅ 三折验严谨=GO（四窗 Δic 全正、fold0 最强破0.10）；待老师约束→焊接 |
| 传统后续优化（推迟，保留方向） | lgbm HPO / 小波→GBDT / MLP——上限有限，战略转型后推迟，有空才碰 | 🅿️ 推迟 |
| 🚩红线（口径更新） | 传统 Final（12 月）三层口径未碰；**DL 新协议（§十一）改用锚定扩展 walk-forward；2026-06-02 再修订：选型只在 Nov 末倒贴 3 个 20 交易日折、Dec 抽出做交付对齐折（冠军定死后只读不回灌）→ 杜绝据 Dec 调参**（交付=方法非权重，无权重 lockbox） | 口径已改（2026-06-02 修订） |

## 已完成基建（备查，指针）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader + 单测；9 stage ~462 列。规格 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + baseline delta + make_decision（传统口径，规则见 AGENTS §四）。
- **Trainer 层**：`src/trainer.py` 的 `BaseTrainer` ABC + `TabularTrainer`，DL 的 `SequenceTrainer` 即接此扩展点。
- **交付链**：`src/submission_pipeline.py` + `meow/meow.py` + `experiments/run_submission_full_window.py`（跨平台内存采样演练）。
