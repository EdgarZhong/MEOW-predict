# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-06-01

## 当前阶段：TCN expanding 已读榜并分析 → 决策：下一步上 GRU-on-features 海选

> **战略**：传统侧已收口、可提交（保底代表已锁）；主线 = ① 交付接线收口（仅剩签名核验/减注释尾巴）+ ② **DL 冲 0.12**（Windows 4060 / PyTorch）。
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
> 2. **val_R² 9/9 全负**（-0.134 到 -0.005）：MSE 比直接预测均值还烂，老师精度分中 MSE 那 1/3 归零。
> 3. **fold 1 种子彩票**：同一训练集 seed 42/43 深负、seed 44 反手 +0.030，种子方差≈0.068——典型欠拟合形状，不是”这个折分布特殊”。
>
> **根因锁定：max_epochs=4 撞天花板**。fold_metrics 中每个点的 best_epoch 均为 2、3 或 4，全部触及 patience/max 上限，模型从未有机会真正收敛。4 epoch 让网络从 59 个原始通道从零学交互，本来就不够。
>
> **决策（2026-06-01 拍板）：下一步上 GRU-on-features 海选，不上 TCN-on-features。**
> - **不选 TCN-on-features 的理由**：TCN 架构本身没问题，换 Adapter 只回答”特征工程有没有用”这个边际问题，信息价值有限。
> - **选 GRU-on-features 的理由**：433 特征已将截面 + 时序主要信号预编码（cross-z / cross-rank / OFI / 动量），GRU 不需要从原始通道从零学交互，收敛难度大幅降低；GRU 隐藏态记忆对已含多步期信息的特征（lag / rolling）更配；GRU 比 TCN 跑得快，同等算力可搜更多 trial。
> - **epoch 预算必须放开**：max_epochs 至少给到 10，patience=3，不再重蹈 TCN 的覆辙。
>
> **⬇️ 下一步 Windows 执行命令（GRU 海选，直接拷贝）：**
> ```bash
> python experiments/run_dl.py \
>   --run-id 20260601_search_gru_feat_v1 \
>   --stage search --model gru --adapter feature_433 \
>   --start 20230601 --end 20230831 \
>   --val-window 5 --step 5 --min-train-days 30 \
>   --trials 6 --seeds 42 \
>   --hparams “device=cuda,max_epochs=10,patience=3,batch_size=256,dropout=0.1” \
>   --out-dir results/dl
> ```
> 跑完后把 `results/dl/20260601_search_gru_feat_v1/trials.csv` + `best_config.json` 提交进 `tracked_results/` 推送，读榜决定是否进 expanding。
>
> **传统保底已锁**：等权 raw_mean 融合 [X1 ridge + M_lgbm_d4]，expanding 均值 0.0776 / Dec sanity Pearson 0.0803 / R²=+0.00465。GRU expanding 结果需同时在均值 + 最坏折两个镜头上超越这个靶子，才能换代表。

### 传统侧（已收口、保底代表；全量明细见 `docs/实验记录.md`）

- **传统代表已锁**：等权 `raw_mean` 融合 **[X1 ridge + M_lgbm_d4]**（expanding 均值 0.0776 / pooled ~0.0763 / 最坏折 0.0491）。两成员各自过样本外：X1 Nov 0.0737、lgbm_d4 Nov 0.0897（vs R02 0.0695）。**传统天花板 ~0.085–0.09，冲 0.10+ 改押 DL。**
- 特征侧 P1–Q4（线性）+ P4 选模型 + P5 融合全程结论已沉淀 `docs/实验记录.md`；规则正文 AGENTS §四/§七。
- **12 月 Final 从未执行**；DL 新协议也不用三层、改 海选 + expanding。

## 交付接线收口（传统交付链，活，剩尾巴）

现状：`meow.py` 壳层（fit/predict/eval）已接 `src/submission_pipeline.py`、零缓存强制、两成员融合 + 并集特征（433 列）已落地、真实数据 end-to-end 跑通。融合口径锁 **`raw_mean`**（等权 raw 平均，保 `fret12` 量纲——老师精度分 = MSE+Pearson+R² 各 1/3，zscore 会毁 MSE/R²）。内存中档精简已落地（预分配流式 + 末位释放源帧，持续峰 ~30→~20GB），本机小窗口验等价 + 冒烟过。

口径要点：
- **serve 端归一化必须用新数据自算、严禁冻结训练期先验**（默认 `raw_mean` 不做截面标准化、天然无此风险；仅 `per_day_zscore_mean` 诊断模式适用）。
- **全量训练、无藏划分器**：`fit/eval` 给什么区间就训/评什么全量（无降采样）；train/eval 划分只在驱动入口（env 变量），老师替换驱动即可（`meow/meow.py:111/121`）。
- 特征从 raw 现算、禁用 `data/features` 缓存。

剩余尾巴（见待办队列）：① 全量内存峰上 ≥32GB 机器实测；② 提交版减注释；③ `fit/predict` 签名对照 docx 核验（另会话）。

## DL 主线：地基设计要点（详见 `docs/specs/DL实验设计规格.md`）

- **固定脊柱 + 可换卡带**：协议/窗口/归一化/指标/配置 = 不可变脊柱；InputAdapter + ModelCartridge = 可换卡带。换模型（LSTM-on-features → DeepLOB-on-rawLOB）只动两块卡带、脊柱零改。
- **PyTorch 封死卡带内**（torch/nn.Module/optimizer/GPU/训练循环都在 `fit/predict`），脊柱 torch-free。
- **评测 = 海选（单切分+早杀，便宜搜超参）+ expanding（少折 walk-forward+多种子，认证）**；防自欺内核 = config-lock + 少看 expanding。
- **配置 = 分布式声明 + 中央组装**：frozen `RunConfig`；三层拆分（枚举集中各块文件顶 / 实现注册挨着 registry / 本次选择进 RunConfig）；run_id 手工语义 + `config_fingerprint` 防漂移。
- **超参只搜结构 3 旋钮**（序列长/hidden/层数）+ 随机搜索 + 早杀。
- **冲 0.12，不验"序列是否有料"**（已是先验）。

## 待办队列

| # | 任务 | 状态 |
|---|---|---|
| 传统全程（P0–P5 + 交付接线主体） | 特征侧 / 选模型 / 融合 / 交付融合接线 + raw_mean + 内存中档精简，全部完成、明细沉淀 `docs/实验记录.md` | ✅ 收口 |
| **DL 地基设计** | 脊柱+卡带架构 / 海选+expanding 协议 / 配置管理 / D0 交付物，定稿落 `docs/specs/DL实验设计规格.md` | ✅ 完成 |
| **DL D0 地基实现（主线②）** | torch-free、Mac CPU 跑通，按 spec §9 全部落地：① `src/sequence_dataset.py`（WindowIndexer 惰性[B,L,C]/不跨日/不跨票/因果对齐/warmup + Normalizer fit-on-train/可 identity + subset_by_dates）② `src/dl_protocol.py`（DLFold 三段切分/embargo/4指标逐字对齐 experiment_runner/assert_folds_causal/summarize_folds）③ `src/dl_trainer.py`（`SequenceTrainer(BaseTrainer)`，鸭子类型注入 adapter+cartridge_factory+raw_loader，产 `FoldResult`）④ `models/dl_models.py`（InputAdapter 接口+IdentityAdapter+FeatureAdapter 包装433+numpy 参考模型 ReferenceZero/Last 当泄漏探测器）+`models/registry.py`（枚举→类注册+required_adapter 校验）⑤ `config/` 6 文件（frozen dataclass+枚举顶部 + `RunConfig` 组装/校验/fingerprint）⑥ `tests/test_dl_pipeline.py`（六项验收闸 **17 test 全过**：端到端/参考模型低分/无泄漏因果/不跨日跨票/归一化只用训练统计/config 校验 + 真实 h5 FeatureAdapter）。**import 约定：src/config/models 三目录平铺，入口 `PYTHONPATH=src:config:models`** | ✅ 完成 |
| **DL 基础设施实施** | `experiments/run_dl.py`（Orchestrator：组装+冻结 RunConfig+dump JSON+SEARCH→Searcher / VALIDATION→定参认证+落 trials/fold_metrics/summary）+ `src/dl_search.py`（采样器 choice/int/uniform + overrides 收窄 + EarlyKillPolicy 钩子桩 + Searcher 排名）+ `RawChannelAdapter`（59 通道）。**seq_len 走 trainer、hidden/layers 走卡带 hparams** 边界写死在 Searcher。21 test 全过 + 真实数据 CLI smoke 跑通。早杀实现仍推后（无 epoch 可杀，等 torch 卡带回调） | ✅ 完成 |
| **README 重写** | 已重写为「DL 工程地基说明 + 代码/文档索引」（DL 规格入口、`config/`/`src/dl_*`/`models/` 结构、import 约定、依赖说明 torch[D1/D2]/psutil[演练]、DL 测试运行方式） | ✅ 完成 |
| **TCN 卡带本体（主攻）** | 实现完成 + smoke 通过。**海选** `20260531_search_tcn_raw_v1`：4 trial 2 正 2 负，best val_corr=0.01653（seq_len=32/hidden=32/layers=4）。**expanding** `20260531_valid_tcn_raw_v1`：mean=0.00912 / std=0.02845 / min=-0.0384 / positive_rate=7/9，val_R² 9/9 全负，fold1 种子方差=0.068。**根因**：max_epochs=4 撞天花板、从 59 原始通道从零学交互 epoch 不够。**结论**：有信号无法挑战传统靶子（0.0776），当前 TCN-on-raw 路线搁置，转 GRU-on-features | ⚠️ 搁置，路线转移 |
| **GRU 海选（当前下一步）** | 卡带已实现（`GRUCartridge` + `ModelKind.GRU`，绑 `FEATURE_433`，27 测 0 fail）。**决策依据**：433 工程特征已预编码截面+时序主要信号，GRU 不需从零学交互；GRU 隐藏态记忆对含多步期特征（lag/rolling）更配；epoch 预算放开（max_epochs=10）补 TCN 的短板。**⬇️ Windows 执行命令：** `python experiments/run_dl.py --run-id 20260601_search_gru_feat_v1 --stage search --model gru --adapter feature_433 --start 20230601 --end 20230831 --val-window 5 --step 5 --min-train-days 30 --trials 6 --seeds 42 --hparams “device=cuda,max_epochs=10,patience=3,batch_size=256,dropout=0.1” --out-dir results/dl`。**结果同步**：跑完把 `trials.csv`+`best_config.json` 提交进 `tracked_results/dl/20260601_search_gru_feat_v1/` 推送，读榜后决定是否进 expanding | 🔜 当前下一步 |
| **正式结果同步目录（双机追踪）** | 根目录新增 `tracked_results/`，专门提交“小体量、正式、可复盘”的结果文件；首批纳入 TCN search / TCN expanding / 传统 Dec 全窗口 sanity 指标，供双机同步与后续深挖分析 | ✅ 已建立 |
| 交付接线 — 全量内存峰实测 | 中档精简已落地（持续峰 ~20GB），全量峰待 ≥32GB 机器实测（本机 16GB 跑不了全窗口） | 🔜 待机器 |
| 交付接线 — Dec 全窗口演练 | 用户已在另一侧完成全窗口 `fit(Jun–Nov)+eval(Dec)`：**Pearson=0.0803 / R²=0.00465 / MSE=2.3645e-05**。结论：提交链量纲健康、端到端可跑；**Dec 只当 sanity、不回灌选型** | ✅ 完成 |
| 交付接线 — 提交版减注释 | 老师鼓励零注释/仅必要处 + 查重；给 `meow/`+`src/` 提交路径单独出精简注释版（提交前专门处理） | 🔜 下一步 |
| 交付接线 — fit/predict 签名核验 | 对照 `meow/MEOW金融时序预测2.0.docx` 确认 `MeowEngine.fit/eval/predict` 能让老师替换路径跑通（predict 当前接特征帧，老师可能按路径取数）。代码侧已确认无藏划分器、全量训练。**另起会话专办** | 🔜 遗留（另会话） |
| 传统后续优化（推迟，保留方向） | lgbm HPO / 小波→GBDT / MLP——上限有限，战略转型后推迟，有空才碰 | 🅿️ 推迟 |
| 🚩红线（Final 未碰） | Final（12 月）全程未碰；DL 新协议不用三层、用 海选+expanding（见 DL 规格 §10） | 未动 |

## 已完成基建（备查，指针）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader + 单测；9 stage ~462 列。规格 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + baseline delta + make_decision（传统口径，规则见 AGENTS §四）。
- **Trainer 层**：`src/trainer.py` 的 `BaseTrainer` ABC + `TabularTrainer`，DL 的 `SequenceTrainer` 即接此扩展点。
- **交付链**：`src/submission_pipeline.py` + `meow/meow.py` + `experiments/run_submission_full_window.py`（跨平台内存采样演练）。
