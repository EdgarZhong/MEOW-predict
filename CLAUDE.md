# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-31

## 当前阶段：正式结果同步目录已建立 + TCN 首轮 expanding 认证完成 + GRU 卡带待海选

> **战略**：传统侧已收口、可提交（保底代表已锁）；主线 = ① 交付接线收口（仅剩签名核验/减注释尾巴）+ ② **DL 冲 0.12**（Windows 4060 / PyTorch）。
> **DL 地基设计已定稿** → `docs/specs/DL实验设计规格.md`（固定脊柱 + 可换卡带 / 海选+expanding 评测 / 配置管理 / D0 交付物 + §8.0 数据实情）。当前在 `feat/dl-foundation` 分支。
> **路线收敛（接卡前查真实 h5 定）**：数据**无连续 LOB**（只 4 稀疏聚合档）→ **DeepLOB 退役**；能喂的 = **~59 原始微结构通道**；**第一个猛药 = TCN-on-原始微结构**（理由见 `NOTE.md`「为什么 TCN」：架构是小杠杆、因果自带、省 GPU、归纳偏置贴订单流）。
> **TCN 首轮海选结论（2026-05-31）**：4 个 trial，2 正 2 负；best = trial 1（seq_len=32, hidden=32, num_layers=4, val_corr=0.01653）；第二正 trial（seq_len=32, hidden=128, num_layers=2, val_corr=0.01305）；两个 seq_len=48 均负（-0.00541 / -0.00453）。**关键信号**：① 未全线崩 → 管线 / 训练循环无根性问题；② seq_len=32 >> seq_len=48 → 短窗更适合原始微结构噪声域；③ val_corr 绝对值仍低，**本阶段不与传统靶子对打、只做管线 sanity**。
> **TCN 首轮 expanding 认证结论（2026-05-31）**：`run_id=20260531_valid_tcn_raw_v1`，`mean=0.00912 / std=0.02845 / min=-0.03837 / max=0.06030 / positive_rate=7/9`。判断：**TCN 有信号但方差偏大，当前还不能挑战传统代表**；下一步保留两条路：① 继续加 TCN 预算；② 开 `GRU + Feature433` 低风险对照。
> **GRU 卡带状态**：`GRUCartridge`（第二卡带，绑 `FeatureAdapter-433`，让 GRU 吃 433 工程特征）已实现 + 单测通过，等 Windows 空出来即可开启 GRU 海选，形成“架构 × 输入”对照。
> **传统全量交付演练已拿到结果（2026-05-31）**：用户已在另一侧跑完全窗口 `fit(Jun–Nov) + eval(Dec)`，指标为 **Pearson=0.0803 / R²=0.00465 / MSE=2.3645e-05**。口径：这次属于**最终 sanity 演练**，说明提交链 `raw_mean` 量纲健康、端到端可跑；**按红线不据此再回改传统代表选型**。
> **本轮基础设施补充（2026-05-31）**：新增根目录 **`tracked_results/`** 作为 git 追踪的正式结果同步树；只收“小体量、正式、可复盘”的结果文件，供双机同步与深入分析，不改现有 `results/` 忽略策略。
> **TCN 首轮海选已完成（2026-05-31 晚）**：run_id=`20260531_search_tcn_raw_v1`，命令口径：`stage=search` / `model=tcn` / `adapter=raw_channels` / `trials=4` / `seeds=42` / `device=cuda` / `max_epochs=4` / `patience=2` / `batch_size=256`。按 `single_split` 取**第一折**海选，实际窗口：**train_core=20230601–20230720（34d） / earlystop=20230721–20230728（6d） / embargo=20230731 / scoring=20230801–20230807（5d）**。运行时长约 **17 分钟**（4 个 trial 串行）。资源观察：Python RSS 约 3–5GB、GPU 显存约 2.9GB，未见内存/显存报警。
> **结果（只代表第一折，不做最终结论）**：4 个 trial 中 **2 正 2 负**；最佳为 **trial 1：`seq_len=32 / hidden=32 / layers=4 / dropout=0.1 / batch=256`，`val_corr=0.01653`**。同为 `seq_len=32` 的 trial 3 也为正（0.01305）；两个 `seq_len=48` trial 均为负（-0.00541 / -0.00453）。初步信号：**短一些的序列长（32）优于 48，TCN 至少不是全线崩**，值得进下一步更严肃的 expanding 认证再看。
> **TCN expanding 认证已完成（2026-05-31 晚）**：run_id=`20260531_valid_tcn_raw_v1`，固定使用 search 冠军 config：`seq_len=32 / hidden=32 / layers=4 / dropout=0.1 / batch=256 / device=cuda / max_epochs=4 / patience=2`；协议口径：`stage=validation` / `profile=expanding` / `max_folds=3` / `seeds=42,43,44`。共 **3 folds × 3 seeds = 9** 个评估点，运行时长约 **42 分钟**。
> **结果（2026-05-31）**：`val_corr mean=0.00912 / std=0.02845 / min=-0.03837 / max=0.06030 / positive_rate=7/9`。逐折看：fold0（20230801–20230807）3 个 seed 全小正；fold1（20230808–20230814）2 个 seed 明显负、1 个 seed 小正；fold2（20230815–20230821）3 个 seed 全正且一个 seed 达 0.0603。**判断**：TCN 不是全线无信号，但当前方差过大、最坏折为明显负值，离“可和传统代表竞争”的稳定程度还差一截；现阶段**不能据此挑战传统最佳**。
> **下一步**：先把 `tracked_results/` 制度固化并收首批结果；随后按用户选择，继续 `TCN` 加预算或开启 `GRU + feature_433` 海选。

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
| **TCN 卡带本体（主攻）** | `models/dl_models.py` 已加 `TCNCartridge`（nn.Module + 因果膨胀卷积残差块 + 训练循环 + 早停，复用 `STRUCTURE_SEARCH_SPACE`、绑 `RAW_CHANNELS`），`run_dl.py` 已打通 device 注入。**验证**：TCN 单测通过 + 真实数据 1 折 1 epoch `device=cuda` smoke 通过 + 首轮 search `20260531_search_tcn_raw_v1` 完成（4 trial 中 2 正 2 负，best `val_corr=0.01653`，`seq_len=32 hidden=32 layers=4`）+ expanding 认证 `20260531_valid_tcn_raw_v1` 完成（`mean=0.00912 / min=-0.03837 / positive_rate=7/9`，方差偏大） | ⚠️ 有信号但不稳，待决策 |
| **GRU 卡带（第二卡带，FeatureAdapter）** | `GRUCartridge`（`nn.GRU` + 取末步隐藏态 + MSE 训练循环 + 早停）已实现，绑 `AdapterKind.FEATURE_433`（复用 433 工程特征管线），`ModelKind.GRU` 枚举新增。**设计逻辑**：GRU 两门参数少、金融短序列（seq_len≈32）上与 LSTM 持平/更快；喂 433 特征而非 59 raw 通道 = 用传统特征工程给 DL 当先验，与 TCN-on-raw 形成“架构 × 输入”对照实验。**验证**：3 项单测（required_adapter / fit+predict / 不物化全窗口）本机 CPU 通过。**下一步**：Windows 空出来后开 GRU 海选（`--model gru --adapter feature_433`） | ✅ 实现完成，待海选 |
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
