# AGENTS.md — 开发规范与协作约定

- **面临新对话开始之前，先查看项目根目录 `.codex/memory/` 下的记忆归档，再进入常规的 `AGENTS.md` / `CLAUDE.md` / `NOTE.md` 阅读流程。**

## 一、文档职责分工

| 文档 | 受众 | 职责 |
|---|---|---|
| `README.md` | 所有人 | 稳定事实：项目定位、目录结构、运行方式、关键文档入口 |
| `AGENTS.md` | Agent | 规则、流程、原则、协作约束（本文件） |
| `CLAUDE.md` | Agent | 动态：当前阶段任务看板、进度、决策、**实验记录（当前阶段、决策导向）** |
| `NOTE.md` | **用户** | 实验笔记：策略讨论、概念解释、敲定方向、待决问题 |
| `docs/实验记录.md` | 所有人 | 实验结果**永久全量档案**（命令 + 全指标明细）；当前阶段实时记录见 CLAUDE.md，阶段收口时沉淀至此 |

### 项目记忆归档约定

- 项目级外部 Agent 记忆原文归档统一放在项目根目录 `.codex/memory/`
- `.codex/memory/` 的职责是**原文保留与人工查阅**，不是替代 `README.md` / `AGENTS.md` / `CLAUDE.md` / `NOTE.md`
- 对 Codex 真正稳定可依赖的项目记忆入口，仍以项目根核心文档为准：
  - `AGENTS.md`：规则、流程、协作边界
  - `CLAUDE.md`：当前阶段进度、任务板、动态决策
  - `NOTE.md`：讨论结论、策略直觉、开放问题
  - `README.md`：稳定事实与入口索引
- 若 `.codex/memory/` 中的内容被确认继续沿用，必须再同步收敛写入对应核心文档；不要只停留在归档目录里

### NOTE.md 维护规范

`NOTE.md` 是用户的实验日志，不是 agent 的规范文件。维护规则：

- **写入时机**：每次讨论出有价值的内容（概念搞清楚了、策略敲定了、发现了新限制），由 agent 负责将结论整理落实到 NOTE.md
- **写入内容**：策略背后的"为什么"、讨论中发现的边界条件、待决的开放问题、对某类实验的直觉判断
- **不写入**：代码细节、任务状态（那是 CLAUDE.md 的职责）、已有文档能查到的内容
- **格式**：按主题分节，每节末尾可附"待讨论/未决问题"列表；底部维护版本记录表
- **agent 行为**：每次讨论出值得记录的结论后，主动追问"要落实到 NOTE.md 吗"，或直接写入后告知用户

### CLAUDE.md 实验记录维护规范

实验记录采用**分层**：CLAUDE.md 记「当前阶段」的实时实验记录（给用户日常看、决策导向、精简），`docs/实验记录.md` 是永久全量档案（命令 + 全指标明细）。两者不混写，靠"收口沉淀"衔接。

- **写入时机**：每个 run 跑完、解读完，**立即**在 CLAUDE.md 对应阶段下补一条；不留过夜、不攒批。
- **CLAUDE.md 单条必含字段**（精简、决策导向）：
  1. `run_id` + 绝对日期 + suite / 阶段
  2. 候选 vs 基线，关键决策（promote / review / reject 或 DL 的 海选/认证 结论）
  3. 关键数字：最佳候选的核心指标、是否过参考线
  4. **一句话洞察**：本条记录里最不显然、未来重推代价最高的"为什么"——这是 leaderboard CSV 里没有、必须人写的部分，是整条记录最值钱的地方
  5. 结果路径指针（`results/.../<run_id>/`）
  6. 遗留 / 清理 flag、下一步命令（若有）
- **不抄全 leaderboard**：完整指标指向 CSV，CLAUDE.md 只留"可复算的指针 + 不可复算的洞察"。
- **日期一律绝对化**："今天"→ `2026-05-31`，便于日后回读。
- **失败结论同等重要**：reject 的候选也要记（含洞察），否则后人会重复试已知无效的方向。
- **阶段收口时沉淀**：某阶段彻底定案、进入下一阶段后，把该阶段在 CLAUDE.md 的多条 run 记录**补全为明细**写入 `docs/实验记录.md`，CLAUDE.md 内压成一行"阶段结论 + 档案指针"。这样 CLAUDE.md 始终只聚焦当前阶段、不随实验累积而膨胀。

## 二、设计宪法

> 系统不是为了做漂亮的 feature store，而是为了支撑冲高分实验的快速迭代、自由试错、计算复用和特征生命周期管理。

一票否决规则——以下任一条成立则方案有问题：

- 为了试一个新特征，需要手动改多个无关模块
- 改一个特征组导致全量 144 天所有组无脑重算
- 缓存失效规则不清楚，结果不知来自新代码还是旧缓存
- 最终 meow.py 离开本地 cache 就跑不起来

不做的事：Hamilton、YAML recipe、表达式 DSL、SQLite manifest、复杂生命周期平台、column-level 状态管理。

## 三、禁止事项

- 禁止只看单次 val_corr 就下结论，必须看扩展 rolling 结果
- 禁止在特征/归一化中使用未来信息（rolling / EMA / zscore 只能用历史数据）
- 禁止用验证集真实 y 的均值、rank、分位数做还原（泄漏）
- 禁止在同一验证集上反复调参后汇报单次最高分
- 禁止把 quick/2折 结果与正式全量 rolling 结果直接并列比较
- **禁止根据最终确认数据（如交付演练那一段）的结果回头改模型再重新提交**：看了就是看了，该结果只用于确认，不作调参依据
- 禁止删除文件，旧文件一律移到 `.archive/`
- 禁止提交 `data/`、`results/`、`*.log`、`*.out`、`*.err`

## 四、信号验收标准（传统线性期口径，已收口；传统代表沿用，DL 走 `docs/specs/DL实验设计规格.md`）

> 这套是"模型固定(ridge)、特征是唯一变量"时代标定的；P4 选模型起模型成了变量，已按 §4.4 改写。完整推导与 P1–Q4 明细见 `docs/实验记录.md`。

### 4.1 两速评测结构

- **快车道（每个候选都跑）**：short(8d) + long(40d) + 每日 IC——便宜否决；short→long 斜率抓"见效快但靠近期"的信号。
- **慢车道（提交关口，每批一次）**：expanding（最像隐藏集）+ 每日 IC 拆解 + 11 月 Review。expanding 慢、且看多会过拟合，只用于不可逆的"并入 backbone"决策。
- medium 移出日常（冗余插值）。每日 IC 看"均值÷波动"分辨"大盘的钱 vs 选股的钱"。

### 4.2 宽进严出 + make_decision 是分诊不是判决

- **+0.003 是采纳参考线，不是进 expanding 的闸刀**：dev 窗 short 实际仅 ~6 个独立样本、均值标准误 ≈0.006，单 profile +0.003 比一个标准误还小，靠跨视角一致（short/long/expanding 同向 + 每日 IC 改善 + Review 对上）才能下结论。凡有真收益迹象的候选都该进 expanding，只有被多视角判定噪声/有害的才不进。
- **make_decision 只是分诊信号**（promote/review/reject 按 delta_corr 分档），不自动判卷；每个 run 出结果都要钻 per-profile 明细多角度复核（符号一致性 / 最差折 / 增益时间分布 / 大盘 vs 选股 / 每日 IC 稳定性 /（Ridge）系数符号），带判断下结论（强制，见 memory「禁止自动判卷」）。
- 缺 expanding 最高只能 review、不得 promote。
- **组合不可加**：多特征组叠加必须作为新 spec 走同一 rolling 重跑，禁止用各组单独 delta 推算合并分。

### 4.3 三层 Holdout（传统口径）

Dev（6–10月，随意调参）/ Review（11月，≤3 次、已用 1、不得据其调参）/ Final（12月，**历史从未执行**、留作最终代表选定后一次性确认）。**DL 不用三层**，改 海选 + expanding（见 DL 规格 §5/§10）。

### 4.4 P4 选模型口径（模型为变量时的改写，已完成）

退役 +0.003 地板 / make_decision（模型互换非特征增量，统计前提不成立）；profile 重心挪 long/expanding、short 降为过拟合诊断；便宜初筛用 long-only；判官 = 均值（pooled corr，对齐老师评分）+ 鲁棒（最坏折 / minimax）双镜头人工权衡，不退化成单指标闸刀；每模型各自最佳特征集（按类型指派、非搜索）、小网格预先钉死；winsorize 对每模型重评。树初筛提速：`--train-subsample-frac 0.33`（仅树族采训练行、验证全量）+ 模型内 `n_jobs`（线程级多核）+ `--n-workers 1`。明细见 `docs/实验记录.md`。

## 五、实验开发 SOP

1. 明确要改的一个变量（特征组 / 目标变换 / 模型 / 窗口）
2. 在 `src/eval_protocol.py` 的 `ALL_SPECS` 里固定本轮候选 spec（§7.1）
3. 先过快车道筛选（`--suite daily --spec-ids <本轮候选...>`，baseline 自动并入、同表算 delta；不带 `--spec-ids` 则跑默认 R 系列），**有收益迹象的候选**（跨 profile 同向、小而真，**不必过 +0.003 地板**）再进慢车道关口（`--suite gate --candidate-spec-id <ID>`），按 §4.2「宽进严出」口径评测
4. 每个 run 跑完立即在 CLAUDE.md 当前阶段记一条实时记录（字段见 §一「CLAUDE.md 实验记录维护规范」）；阶段收口时把明细（§8.4 必录字段）沉淀进 `docs/实验记录.md`
5. 与基线比较，只改一个变量；提交遵循 §八 commit 规范

### 5.1 跑命令前自查清单（每次执行 `experiments/*.py` 前逐条过）

1. **suite 选对了吗**：日常筛选 `--suite daily`（short+long+每日IC）；筛候选用 `--suite daily --spec-ids ...`（baseline 自动并入、同表算 delta，不给 `--spec-ids` 才跑默认 R 系列）；提交关口 `--suite gate --candidate-spec-id <ID>`（候选 vs 基线、只跑 expanding）。别拿 `full`/`ridge` 当日常。
2. **macOS 并行度上限（16GB）**：`gate` / `expanding` / `daily` 都必须显式 `--n-workers 1` 串行——并行 `long`/`expanding` 长窗 heavy 组会 OOM 被杀（2026-05-26 事故）。只有迁到 20GB+ 空闲内存机器才考虑放开。**树初筛要多核靠模型内 `n_jobs`（线程级、共享内存），不靠 `--n-workers`——后者恒为 1**（单 fit 已吃满核，进程级并行只翻内存无吞吐；口径见 §4.4）。
3. **候选 spec 已一次性固定**：禁止边加 spec 边重筛（§4.2 多重比较）。本轮要比的全集先定死、一起跑、一起看。
4. **已锁默认别手改**：训练标签 winsorize = 开启 + P1/P99，ridge alpha = 2.0，特征 dtype = float32（§7.2）。要做对照才显式覆写，且记录在案。
5. **【硬约束】每个 run 必须挂内存看门狗，无一例外**：凡执行 `experiments/p0_eval_protocol.py`（任意 suite），必须经 `experiments/run_with_memory_guard.py` 包装，禁止裸跑。原因：被强杀（OOM）时后台任务不发完成通知、会无声消失。标准跑法：
   ```bash
   PYTHONPATH=src caffeinate -i python experiments/run_with_memory_guard.py \
     --rss-limit-gb 9 --rss-hard-limit-gb 11 --log-file logs/memory_guard_<run>.log \
     -- python experiments/p0_eval_protocol.py --suite daily --spec-ids ... \
        --run-id <run> --n-workers 1 > logs/<run>.log 2>&1
   ```
   - **树初筛变体**：命令尾部加 `--train-subsample-frac 0.33`（仅树族采训练行、线性自动全量；`n_jobs=8` 已钉进 M_tree spec）。
   - `caffeinate -i` 防休眠；看门狗软阈 12GB / 硬阈 13GB（16GB Mac 留 OS 余量）；输出**重定向到日志文件**（禁止 `| tail`，强杀时缓冲全丢）。
   - **日志一律写项目内 `logs/`（已 gitignore），禁止写 `/tmp`**（会被系统蒸发、不可回溯）。命名：看门狗 `logs/memory_guard_<run>.log`、stdout `logs/<run>.log`。
   - 长跑用 `--run-id` 固定、`--resume` 可续。
6. **后台运行 + 定时盯岗**：长 run 后台执行；中止任务一律 `TaskStop`（不用 Bash `kill`，harness 不认）；离开期间靠定时（≤1 小时间隔）回来查任务最新状态（leaderboard 产出 / `TaskList` / 日志尾部），覆盖「完成 / 静默死亡 / 卡死」三态，不依赖完成通知。
7. **目标与输出回 raw `fret12`**：winsorize 只作用于训练标签，评测/提交一律原始 `fret12`；提交通道不依赖 `data/features/` 缓存、不依赖任何不可见统计量（§十 推理契约）。

## 六、模型优先级

```
传统（已收口、保底代表）：Ridge → ElasticNet/Huber → 浅 ExtraTrees/HistGB/LightGBM → 受限融合（加权平均；OOF stacking 仅明显更好才上）
DL（上行主线）：序列模型 LSTM（特征当序列，低风险）→ DeepLOB raw-LOB（高风险）
```

- **传统代表已锁**：等权 raw_mean 融合 [X1 ridge + lgbm_d4]，可提交的保底（明细见 CLAUDE / `docs/实验记录.md`）。
- **DL 是上行主线**，走独立协议与地基：`docs/specs/DL实验设计规格.md`。约束：① 走 DL 规格的 海选 + expanding 评测 + 防泄漏开窗，不另立标准；② 输入须平稳化 + 归一化；③ 必须能在 `meow.py` 提交（§十）；④ 直接冲 0.12，不做"序列是否有料"的存在性验证。

## 七、特征工程规范（传统线性期口径，已收口；细节见 `docs/实验记录.md`）

- 特征按信号族组织，前缀 `ofi_`/`trade_impact_`/`momentum_`/`regime_`/`cross_`；所有滑动（rolling / EMA / lag）只能用 `shift(1)` 及以前；禁止全样本归一化（用 rolling zscore / 截面 rank / 个股 z-score）。

### 7.1 新增实验 spec

在 `src/eval_protocol.py` 的 `ALL_SPECS` 加一条 dict（`experiment_id` / `type` / `model` / `target_mode` / `groups` / `notes`）；命名 `{系列}{编号}_{模型}_{特征}`（R=Ridge 基线 / O=OFI / T=TradeImpact / C=CondMomentum）。交互项先 scratch 手写，被复用或进正式 rolling 才晋级 registry stage（`@registry.stage` + `python -m feature_store build`），避免 stage 膨胀。

### 7.2 已锁默认（P0.5 扫锁，勿手改，对照才显式覆写）

- ridge `alpha=2.0`（平台区，跨 alpha 波动 ~1e-5）；训练标签 winsorize 开启 + `P1/P99`（只裁训练标签、提交永远输出原始 `fret12`、无逆变换、不构成泄漏）；特征 dtype `float32`。
- ⚠️ winsorize / alpha 是为线性 MSE 标定的，树用分裂点机制不同，P4 已对树重评（§4.4）。

### 7.3 Target Mode 准入（硬标准）

最终提交预测值**必须对应原始 `fret12` 尺度**。`raw` / `interval_residual` / `common_residual` 等，只要能在测试时仅靠已知特征 + meta + 训练期固定参数无泄漏恢复 raw `fret12`，才可进正式 rolling；`date_demean` / `interval_demean` / `rank` / `quantile` 等需测试集真实 y 还原的一律放弃（泄漏）。有争议默认按"不可提交"处理。

### 7.4 特征生命周期

- 特征组状态 scratch / candidate / promoted / archived 在 registry 声明；archived 的 builder 代码移入 `.archive/`。当前 9 stage 均 promoted。
- 特征缓存一票否决规则见 §二（为试新特征改多模块 / 改一组全量重算 / 缓存失效不清 / `meow.py` 离开 cache 跑不起来，任一成立即设计有问题）。

## 八、Git 提交规范

### 8.1 粒度原则

- 每次 commit 代表一个明确的变更：特征组、目标变换、模型、split 逻辑、评测逻辑
- 不混提：特征工程 + 模型调参 + 文档清理禁止一次提交

### 8.2 提交格式

```
feat:   新功能或新特征
fix:    修复 bug
exp:    实验结果记录
docs:   文档更新
refact: 重构（不改行为）
```

示例：
```
feat: add OFI multi-level features (bid/ask/total)
exp: P1 OFI rolling audit, O3 best rolling_corr_mean=0.051
fix: prevent target leakage in interval_demean normalization
```

### 8.3 不可提交的内容

- `data/`（原始数据）
- `results/`（中间结果 CSV）
- `*.log / *.out / *.err`
- `__pycache__/`

### 8.4 每次实验 commit 必须记录

`experiment_id / date / feature_set / target_type / model / split_config / seed / rolling_corr_mean / rolling_corr_std / stability_score / notes`

## 九、目录约定

```
src/              核心模块（特征、模型、评估、数据加载）
experiments/      实验入口脚本（按 P 阶段命名）
experiments/legacy/  历史脚本（可运行，不主动修改）
results/          实验结果 CSV（gitignored）
data/             原始 .h5 数据（gitignored）
docs/             文档
docs/archived/    历史快照（只读）
.archive/         废弃文件（gitignored）
```

## 十、推理契约与交付约束

最终提交走老师的 `meow.py`：`engine.eval(start,end)` → `genFeatures(rawData)` → `predict(xdf)` → `ydf["forecast"]=pred` → `evaluator.eval(ydf)`（也就是 `ydf[["forecast","fret12"]].corr()`）。由此固定下面几条硬约束：

**输出契约（事实，不可违反）：**

- 预测列名必须是 `forecast`，粒度是每个 `(symbol, date, interval)` 一个 float。
- 预测的行序必须和 `genFeatures` 输出的 `xdf` 严格对齐，禁止重排 / 乱序提交。
- 每行都必须输出**有限值**：老师对 NaN/inf 会 `fillna(0)`，但这个 0 会被算进相关性（不是无损丢弃），所以不能指望“反正会被忽略”——实在算不出的行也要主动给一个合理的回退值。

**先把提交通道打通，再开实验（强制）：**

- 正式迭代之前，先在一个 held-out 交易日把 `meow.py` 这条链路完整跑通（`genFeatures → predict → forecast` 列对齐、每行有限），确认契约成立。
- 之后所有实验都长在这条已验证的提交通道上。量化最常见的翻车不是模型差，而是 train/serve skew：某个特征在提交时的算法和训练时不一致，或者某个归一化依赖了测试时根本拿不到的统计量。
- `meow.py` 不能依赖 `data/features/` 本地缓存、不能依赖任何不可见的统计量；交付演练时按这几条逐条核对。
