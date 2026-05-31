# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-31

## 当前阶段：DL 主线已进入首轮 TCN 海选（后台运行中，待读第一批 trial）

> **战略**：传统侧已收口、可提交（保底代表已锁）；主线 = ① 交付接线收口（仅剩签名核验/减注释尾巴）+ ② **DL 冲 0.12**（Windows 4060 / PyTorch）。
> **DL 地基设计已定稿** → `docs/specs/DL实验设计规格.md`（固定脊柱 + 可换卡带 / 海选+expanding 评测 / 配置管理 / D0 交付物 + §8.0 数据实情）。当前在 `feat/dl-foundation` 分支。
> **路线收敛（接卡前查真实 h5 定）**：数据**无连续 LOB**（只 4 稀疏聚合档）→ **DeepLOB 退役**；能喂的 = **~59 原始微结构通道**；**第一个猛药 = TCN-on-原始微结构**（理由见 `NOTE.md`「为什么 TCN」：架构是小杠杆、因果自带、省 GPU、归纳偏置贴订单流）。
> **本会话新增完成（2026-05-31 晚）**：`models/dl_models.py` 已补 **`TCNCartridge`**（PyTorch nn.Module + 因果膨胀卷积残差块 + 训练循环 + earlystop），`experiments/run_dl.py` 已把 `ExecConfig.device` 真正注入卡带；`tests/test_dl_infrastructure.py` 新增 TCN 最小接线测试。**验证**：① `python -m unittest tests.test_dl_infrastructure.TestTCNCartridge -v` 通过；② 真实数据 1 折 1 epoch smoke 通过：`python experiments/run_dl.py --stage validation --model tcn --adapter raw_channels ... --hparams "device=cuda,seq_len=16,hidden_size=16,num_layers=2,max_epochs=1,..."` 成功跑完，说明 **CUDA 路径已实际走通**（若不可用会当场报错）。
> **传统全量交付演练已拿到结果（2026-05-31）**：用户已在另一侧跑完全窗口 `fit(Jun–Nov) + eval(Dec)`，指标为 **Pearson=0.0803 / R²=0.00465 / MSE=2.3645e-05**。口径：这次属于**最终 sanity 演练**，说明提交链 `raw_mean` 量纲健康、端到端可跑；**按红线不据此再回改传统代表选型**。
> **TCN 首轮海选已启动（2026-05-31 晚，后台）**：run_id=`20260531_search_tcn_raw_v1`，命令口径：`stage=search` / `model=tcn` / `adapter=raw_channels` / `trials=4` / `seeds=42` / `device=cuda` / `max_epochs=4` / `patience=2` / `batch_size=256`。当前按 `single_split` 取**第一折**海选，实际窗口：**train_core=20230601–20230720（34d） / earlystop=20230721–20230728（6d） / embargo=20230731 / scoring=20230801–20230807（5d）**。资源观察：启动后 Python RSS 约 3–5GB、GPU 显存约 2.9GB，暂未见内存/显存报警。
> **未提交**：本会话改动仍在工作区（用户决定提交时机）。**下一步 = 等 `20260531_search_tcn_raw_v1` 跑完先读 `trials.csv + best_config.json`，再决定是否直接进 expanding 认证（validation）**；脊柱/Orchestrator/Searcher/adapter 不再改，除非正式跑数暴露新问题。

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

剩余尾巴（见待办队列）：① 全量内存峰上 ≥32GB 机器实测；② Dec 全窗口演练（sanity，跑法 `python experiments/run_submission_full_window.py`，SOP 见 `docs/交付演练SOP_Windows全窗口.md`）；③ 提交版减注释；④ `fit/predict` 签名对照 docx 核验（另会话）。

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
| **DL D0 地基实现（主线②）** | torch-free、Mac CPU 跑通，按 spec §9 全部落地：① `src/sequence_dataset.py`（WindowIndexer 惰性[B,L,C]/不跨日/不跨票/因果对齐/warmup + Normalizer fit-on-train/可 identity + subset_by_dates）② `src/dl_protocol.py`（DLFold 三段切分/embargo/4指标逐字对齐 experiment_runner/assert_folds_causal/summarize_folds）③ `src/dl_trainer.py`（`SequenceTrainer(BaseTrainer)`，鸭子类型注入 adapter+cartridge_factory+raw_loader，产 `FoldResult`）④ `models/dl_models.py`（InputAdapter 接口+IdentityAdapter+FeatureAdapter 包装433+numpy 参考模型 ReferenceZero/Last 当泄漏探测器）+`models/registry.py`（枚举→类注册+required_adapter 校验）⑤ `config/` 6 文件（frozen dataclass+枚举顶部 + `RunConfig` 组装/校验/fingerprint）⑥ `tests/test_dl_pipeline.py`（六项验收闸 **17 test 全过**：端到端/参考模型低分/无泄漏因果/不跨日跨票/归一化只用训练统计/config 校验 + 真实 h5 FeatureAdapter）。**import 约定：src/config/models 三目录平铺，入口 `PYTHONPATH=src:config:models`** | ✅ 完成（本会话，未提交） |
| **DL 基础设施实施** | `experiments/run_dl.py`（Orchestrator：组装+冻结 RunConfig+dump JSON+SEARCH→Searcher / VALIDATION→定参认证+落 trials/fold_metrics/summary）+ `src/dl_search.py`（采样器 choice/int/uniform + overrides 收窄 + EarlyKillPolicy 钩子桩 + Searcher 排名）+ `RawChannelAdapter`（59 通道）。**seq_len 走 trainer、hidden/layers 走卡带 hparams** 边界写死在 Searcher。21 test 全过 + 真实数据 CLI smoke 跑通。早杀实现仍推后（无 epoch 可杀，等 torch 卡带回调） | ✅ 完成（本会话，未提交） |
| **README 重写** | 已重写为「DL 工程地基说明 + 代码/文档索引」（DL 规格入口、`config/`/`src/dl_*`/`models/` 结构、import 约定、依赖说明 torch[D1/D2]/psutil[演练]、DL 测试运行方式） | ✅ 完成（本会话，未提交） |
| **TCN 卡带本体（主攻）** | `models/dl_models.py` 已加 `TCNCartridge`（nn.Module + 因果膨胀卷积残差块 + 训练循环 + 早停，复用 `STRUCTURE_SEARCH_SPACE`、绑 `RAW_CHANNELS`），`run_dl.py` 已打通 device 注入。**验证**：TCN 单测通过 + 真实数据 1 折 1 epoch `device=cuda` smoke 通过。**当前状态**：首轮海选 `20260531_search_tcn_raw_v1` 已后台启动，待产出 `trials.csv / best_config.json` | 🔄 运行中 |
| **D1 LSTM 卡带（低风险对照，4060 后）** | LSTM 卡带（433 特征当序列，绑 `FEATURE_433`）作对照；`DEEPLOB` 已退役（数据无连续 LOB，规格 §8.0） | 待开（等卡） |
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
