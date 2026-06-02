# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-06-02

## 当前阶段：DL 路线收敛 → 押"截面" + 新评测协议落定（TCN-on-raw 否决）

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
> **传统保底已锁**：等权 raw_mean 融合 [X1 ridge + M_lgbm_d4]，expanding 均值 0.0776 / Dec sanity Pearson 0.0803 / R²=+0.00465。**DL 候选需在新协议 Sep–Nov 选型三折的均值 + 最坏折两镜头上给出清楚优势，才考虑换代表；否则 DL 作增强叠在传统核心上、保底交付不破。**

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
2. ✅ **maxfold 内存演练**：2026-06-02 已跑 `20260602_maxfold_xsection_v1`，最大旧窗口折 131 训练日 / 全票 / CUDA / 截面卡带通过；峰值 proc RSS 20.76GB、系统内存 33.15/33.91GB、显存 3237MiB，未 OOM。该折重于新协议 Sep–Nov 选型折，可作为长跑前硬闸通过。
3. 启动正式长跑（见下），后台 + `resource_monitor.py` + 守护。

**长跑命令（主攻 = 截面模型 SWEEP；以合并后实际 ModelKind/CLI 为准，新会话先 `run_dl.py --help` 核对）**：

> ✅ **2026-06-02 协议修订已落代码**：选型折改 **3 段×~20 交易日（月度 Sep–Nov，rolling 截到 Nov 末）**；**Dec 抽出做交付对齐折**——冠军定死后跑 `train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)` × 1 seed、写 `summary.json` 的 `delivery` 块、不参与排名（验交付链）。CLI 采用显式方案 A：`--end 20231130` 作为选型/训练截止，`--delivery-eval-end 20231229` 作为交付折 eval 末日；因 `--embargo 1`，delivery scoring 会跳过 20231201。

```
python experiments/run_dl.py --stage sweep --model xsection --adapter feature_433 --device cuda \
  --start 20230601 --end 20231130 --delivery-eval-end 20231229 --val-window 20 --step 20 --min-train-days 40 --max-folds 3 \
  --fold-select recent --grid-seq-len 32 --grid-hidden 32 --grid-layers 1 \
  --hparams dropout=0.2,weight_decay=0.001,max_epochs=15,patience=5,lambda_corr=0.3 \
  --seeds 42,43,44 --max-symbols 0 --run-id 20260602_sweep_xsection_v1 --out-dir results/dl
```

- **折**：选型 3 折 Sep/Oct/Nov（各 ~20 交易日、锚定扩展、Nov 末倒贴）；档1 筛选取最近 2 折(Oct,Nov)×2seed→最坏折选冠军；档2 认证冠军×3折×3seed=9 fit；**交付折**冠军×`train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)`×1seed=1 fit（显式 `--delivery-eval-end 20231229`，只报不选）。结构固定时 ≈14 fit、λ 入网格 ≈18 fit、~6–7h。
- λ 走 `--hparams`（决策 Q1）；**rescale 永远开、cross-z 不做**（决策 Q2）；结构固定昨晚冠军 seq32/h32/L1；max_epochs 30→15（best_epoch 实测 4–7，省时不损精度）。
- **Tier-1 fallback**（若截面演练没过）：先跑 **GRU-on-433 + λ·corr + rescale**（`--model gru` + `lambda_corr=0.3`，proven 管线、低风险），验杠杆 + 把 R² 翻正，不浪费今晚。

**早上看**：选型三镜头——val_corr.mean / 最坏折 val_corr.min（决策主镜头）/ val_r2（**应被 rescale 兜在 ≥0**）；**外加交付折 `delivery` 块**（当前方案 A 严格保留 1 日 embargo，因此是 Dec4–Dec29，不是完整 Dec1–Dec29；若要和传统 0.0803 完全字面对齐，需另行决策 delivery 是否改 `embargo=0`）。目标 corr 力争 >0.10、R² 不为负。§11.9：0.0776 / 0.0803 均作为传统背景参考，正式换代表看 Sep–Nov 选型三折。

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
| **新评测协议落 run_config** | 两档骨架已落代码：`Stage.SWEEP` 一命令两档 + `build_dl_folds(fold_select="recent")` + `enumerate_grid` + 最坏折/R² 双镜头 + `--device/--grid-*` CLI。✅ **2026-06-02 协议修订已落代码**：① 选型折改为 **3×~20 交易日（月度 Sep–Nov，rolling 截到 Nov 末）**；② 新增显式 `--delivery-eval-end` 交付对齐折——冠军定死后 `train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)`×1seed、写 `summary.json.delivery`，同时在 `fold_metrics.csv` 追加 `profile_name=sweep_delivery` 行，但不参与档1排名/档2汇总（验交付链、杜绝据 Dec 调参）。相关测试 81 passed。 | ✅ 完成 |
| **截面模型方案（主攻）** | **方案已拍板落档**（`NOTE.md`「截面模型怎么设计」+ 规格 §8.2）：因子化两腿 = 共享 GRU 时序腿（每票日内路径编码）+ set-attention 截面腿（跨票、置换等变零身份、零初门控残差）+ per-票头；`MSE+λ·corr` 截面对齐；张量契约 `[L,C]`→整截面 `[N,L,C]`+mask（WindowIndexer 按 (date,interval) 聚票；v1 不做 cross-z，见规格 §8.2.1）。业界 STGNN/关系排序同构、为什么不用时空联合/图网络已记 | ✅ 方案完成 |
| **截面模型卡带（主攻）** | **已合并（2026-06-02）**：`CrossSectionCartridge`（`ModelKind.XSECTION`，绑 FEATURE_433）+ `CrossSectionIndexer`/`CrossSectionDataset` + `_GpuCrossSectionSource` + masked 截面 Pearson loss + 训练段 OLS rescale；接已跑通 SWEEP，下一步跑测试与 maxfold 内存演练。 | ✅ 已合并待验收 |
| **GRU-on-433 基线 + 损失对齐** | **基线认证收官**（run `20260601_sweep_gru433_v1`，纯 MSE/λ=0，13:43→00:30，详见 `docs/实验记录.md` 2026-06-02）。冠军 seq32/h32/L1。**三镜头**：val_corr.mean **0.0585**（positive **15/15**、std 0.013）/ 最坏折 **0.0391** / **R² −0.0071（14/15 微负）**。结论：**结构成立**；**R² 微负 = MSE/corr 矛盾实锤**。WT-B 已把 `MSE+λ·corr` + 训练段 OLS rescale 接入 GRU，λ 经 `--hparams lambda_corr`，下一步正式扫 λ∈{0,0.3}。 | ✅ 损失对齐已合并待验收 |
| **正式结果同步目录（双机追踪）** | 根目录新增 `tracked_results/`，专门提交“小体量、正式、可复盘”的结果文件；首批纳入 TCN search / TCN expanding / 传统 Dec 全窗口 sanity 指标，供双机同步与后续深挖分析 | ✅ 已建立 |
| 交付接线 — 全量内存峰实测 | 中档精简已落地（持续峰 ~20GB），全量峰待实测（**本机已核实 ~34GB**，非旧记 16GB；满足 ≥32GB，可本机试） | 🔜 待实测 |
| 交付接线 — Dec 全窗口演练 | 用户已在另一侧完成全窗口 `fit(Jun–Nov)+eval(Dec)`：**Pearson=0.0803 / R²=0.00465 / MSE=2.3645e-05**。结论：提交链量纲健康、端到端可跑；**Dec 只当 sanity、不回灌选型** | ✅ 完成 |
| 交付接线 — 提交版减注释 | 老师鼓励零注释/仅必要处 + 查重；给 `meow/`+`src/` 提交路径单独出精简注释版（提交前专门处理） | 🔜 下一步 |
| 交付接线 — fit/predict 签名核验 | 对照 `meow/MEOW金融时序预测2.0.docx` 确认 `MeowEngine.fit/eval/predict` 能让老师替换路径跑通（predict 当前接特征帧，老师可能按路径取数）。代码侧已确认无藏划分器、全量训练。**另起会话专办** | 🔜 遗留（另会话） |
| **run_dl 增量落盘（防中断打水漂）** | **已实现 + 已合并 `feat/dl-foundation`（merge `feat/dl-incr-dump`，2026-06-02）**：`trials.csv`/`fold_metrics.csv` 逐 trial/逐折 `append+flush+fsync`（`_IncrementalCsvWriter` 表头幂等、零行兜底、与旧 `_dump_csv` 产物逐字节等价）+ `progress.jsonl` 时间线 + `summary.partial.json` 每折快照；`summary.json` 仅成功收官落（=完成标记）；`--resume` 复用已落盘 trial/(seed,fold) 跳过不重算。infra 测试 **36 通过**（新增 `TestSweepIncrementalDump`：逐字节等价 / 人为中断保留前序 / resume 续跑无重复无遗漏） | ✅ 完成（已合并） |
| 传统后续优化（推迟，保留方向） | lgbm HPO / 小波→GBDT / MLP——上限有限，战略转型后推迟，有空才碰 | 🅿️ 推迟 |
| 🚩红线（口径更新） | 传统 Final（12 月）三层口径未碰；**DL 新协议（§十一）改用锚定扩展 walk-forward；2026-06-02 再修订：选型只在 Sep–Nov 三月度折、Dec 抽出做交付对齐折（冠军定死后只读不回灌）→ 杜绝据 Dec 调参**（交付=方法非权重，无权重 lockbox） | 口径已改（2026-06-02 修订） |

## 已完成基建（备查，指针）

- **PE0 并发平台**：并发调度 + resume + 串并一致性 + OOM 修复。
- **PE1 特征管道**：FeatureRegistry / FeatureStore / FeatureLoader + 单测；9 stage ~462 列。规格 `docs/specs/特征管道重构规格.md`。
- **P0 评测体系**：三层协议 + 4 profiles + baseline delta + make_decision（传统口径，规则见 AGENTS §四）。
- **Trainer 层**：`src/trainer.py` 的 `BaseTrainer` ABC + `TabularTrainer`，DL 的 `SequenceTrainer` 即接此扩展点。
- **交付链**：`src/submission_pipeline.py` + `meow/meow.py` + `experiments/run_submission_full_window.py`（跨平台内存采样演练）。
