# AGENTS.md — 开发规范与协作约定

## 一、文档职责分工

| 文档 | 受众 | 职责 |
|---|---|---|
| `README.md` | 所有人 | 稳定事实：项目定位、目录结构、运行方式、关键文档入口 |
| `AGENTS.md` | Agent | 规则、流程、原则、协作约束（本文件） |
| `CLAUDE.md` | Agent | 动态：当前阶段任务看板、进度、决策 |
| `NOTE.md` | **用户** | 实验笔记：策略讨论、概念解释、敲定方向、待决问题 |
| `docs/实验记录.md` | 所有人 | 所有实验结果的唯一历史记录 |

### NOTE.md 维护规范

`NOTE.md` 是用户的实验日志，不是 agent 的规范文件。维护规则：

- **写入时机**：每次讨论出有价值的内容（概念搞清楚了、策略敲定了、发现了新限制），由 agent 负责将结论整理落实到 NOTE.md
- **写入内容**：策略背后的"为什么"、讨论中发现的边界条件、待决的开放问题、对某类实验的直觉判断
- **不写入**：代码细节、任务状态（那是 CLAUDE.md 的职责）、已有文档能查到的内容
- **格式**：按主题分节，每节末尾可附"待讨论/未决问题"列表；底部维护版本记录表
- **agent 行为**：每次讨论出值得记录的结论后，主动追问"要落实到 NOTE.md 吗"，或直接写入后告知用户

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
- **禁止根据 final holdout（12月）的结果回头改模型再重新提交**：看了就是看了，该结果只用于最终确认，不能作为调参依据
- 禁止删除文件，旧文件一律移到 `.archive/`
- 禁止提交 `data/`、`results/`、`*.log`、`*.out`、`*.err`

## 四、信号验收标准

### 4.1 数值门槛（protocol 层）

```
protocol_corr_mean   提升 ≥ 0.003（相对当前基线）
protocol_stability_score  不下降
protocol_corr_min    不明显变差（不出现新的强负 fold）
每日 IC-IR           不下降（每日截面 IC 的均值÷波动，分辨"选股 vs 大盘"，详见 §4.7）
MSE / R²             不明显恶化（同一 winsorize 设定下比，见 §7.11）
```

如果某一类特征只让一个 fold 变好，其他 fold 变差，则放弃。

### 4.2 两速评测结构与 profile 分工

四个 rolling profile 本质是同一根轴上的四个采样点——"训练用多少历史"：short(8天) → medium(20天) → long(40天) → expanding(40天起步、滚动累积到约95天)。medium、long 是这根轴的中间插值，信息冗余；真正互补的是两端——short 折最多、最便宜，expanding 唯一模拟"用到目前为止的全部历史、往前预测"，最像老师那场隐藏测试。

据此把评测分成两速。**核心原则：搜索用便宜的快车道，提交用昂贵的慢车道；最可信的那个判官（expanding）要少看——同一窗口看多了会被过拟合，不再可信（见 §4.3）。**

| 车道 | 跑什么 | 任务 | 频率 |
|---|---|---|---|
| 快车道（筛选） | short + long + 每日 IC | 便宜地否决；靠 short→long 斜率抓"见效快但靠近期"的信号 | 每个候选都跑 |
| 慢车道（提交关口） | expanding + 每日 IC 拆解 + 11月 Review Holdout | 只用于不可逆的"并入 backbone"决策 | 每批候选定死后只跑一次 |

- **快车道为什么是 short + long**：short 出统计功效（折最多），long 出"训练历史拉长后信号还在不在"。一个特征在 short 上涨、到 long 衰减或翻负，就是最危险的"靠近期"信号，当场标记。long 只有 ~6 折、方差大，所以只读它的**方向/是否破**（否决票），不读精确量级。
- **medium 移出日常**：它是 short 与 long 中间的冗余插值，有了两端的斜率就用不上；需要时可临时单跑，不进日常 suite。
- **每日 IC 是正交视角**：池化 corr 只说"总分"，每日截面 IC 说"是不是真在同一时刻把股票排对了"。两者一起看才能分清赚的是"大盘的钱"还是"选股的钱"（§4.7 第 4 步）；它从现成预测里就能算，零成本，应与池化 corr 并列看其"均值 ÷ 波动"。
- **expanding 是慢车道的判官，不进快车道**：它最像真实考试，但慢、要单独跑（与日常 suite 分开、配 memory guard）、且看多了会被过拟合，所以留给提交关口、每批只看一次。
- 具体采纳阈值见 §4.6（make_decision 机器判定），人工逐 profile 复核方法见 §4.7。

### 4.3 多重比较与边界增益纪律

Dev Rolling 只有 ~105 个交易日，short profile 实际仅 ~5–6 个独立样本。R02 基线 `rolling_corr_std≈0.015`，对应均值标准误 ≈ 0.015/√6 ≈ **0.006**。

**因此单个 profile 上的 +0.003 低于一个标准误，本身无法与噪声区分**，必须靠跨 profile 一致性 + Review Holdout 才能下结论。

- 边界增益（`delta_corr` 0.003～0.005）**禁止只凭 Dev Rolling 直接 promote**，必须进 11 月 Review Holdout 复核后再决策
- 每个阶段开跑前**先固定候选 spec 集合**，一次性跑完一起看；禁止"边加 spec 边重筛"——反复试探等于把 Dev 窗口磨成被过拟合的调参集（garden of forking paths）
- 对同一份 Dev Rolling 比较的 spec 越多，纯靠运气过线的越多：候选集越大，promote 的一致性与门槛要求越要从严

**口径补充（2026-05-24）：真正的关卡是“跨视角一致性”，不是数字本身。** `0.003~0.005` 这个区间只是一条**安全地板**（低于它别想直接采纳），不是采纳标准。决定采不采纳的，是同一增益有没有被多个互相独立的视角同时印证：short / long / expanding 是否同向、每日 IC 是否也改善、11 月 Review Holdout 是否对得上。在测试集完全未知的前提下，宁可保守——这条规则不算太严，是恰当的。

### 4.4 各阶段评测口径（P1–P5）

§4.2 的两速结构落到每个阶段：

| 阶段 | 这一步在决策什么 | 评测口径 | 关键诊断 | 频率/成本 |
|---|---|---|---|---|
| P1–P3 筛选 | 这个特征组有没有用、稳不稳 | 快车道：short + long + 每日 IC | short→long 斜率；每日 IC | 每候选，~10–15 min |
| P1–P3 提交关口 | 把候选并入 backbone？ | 慢车道：expanding（候选 vs 基线）+ 每日 IC + 11月 Review | §4.7 六步；§4.6 硬契约 | 每批一次，~30 min |
| P3.5 交互 | 交互项收不收 | 同筛选，**额外看系数符号稳定性** | 交互极易过拟合 → §4.6 门槛比 P1–P3 更严 | 每候选 |
| P4 选模型 | 用哪个模型 | short+long 初筛；expanding 只在 2–3 个决赛模型上跑 | 树吃数据、short 对树不公平 → 加权 long/expanding；**minimax：选最差折最好的，不是均值最高的** | 决赛才上 expanding |
| P5 融合 | 融合 vs 单 backbone | 以 expanding 关口为准（只跑一次）+ 11月 Review + OOF 预测 | 不比 backbone 更稳就交 backbone；融合在 short 上的增益最像噪声 | 最少次，最高规格 |

- **筛选只产出”候选”，不产出”采纳”**：过了快车道只进候选池，真正并入 backbone 必须过慢车道。
- **expanding 是 promote 的硬门槛**：任何候选 promote 前必须单独跑过 expanding（与日常 suite 分开、配 memory guard；worker 数等提速口径见 `docs/specs/开跑前编码指导_评测口径与提速.md`），expanding 上不成立则不得 promote，**不得因为它慢而跳过**。这是 §4.6”缺 expanding 最高只能 review”的来源。
- **P4 起 expanding 更贵**：上树后训练成本高、且无法用线性增量技巧加速，所以只在少数决赛模型上跑，候选集要小。

### 4.5 组合不可加：叠加后必须重跑

P1–P3 的实验都是"R02 + 单个特征组"。即使多个组各自单独过线，也**不能假设增益相加**：

- 特征间常有信息重叠（如 OFI 与成交冲击都刻画买卖压力），合并后真实增益通常**小于**各自之和
- 维度变多在固定 alpha 下可能拟合噪声，合并模型甚至可能**差于**只加最好的单组
- 偶有互补使合并**大于**之和，但不可预先假定

**规则：任何把多个特征组叠加使用的方案，必须作为一个新 spec、走同一套 rolling 重新验收，禁止用各组单独 delta 推算合并分数。** 这是 P1–P3（单组筛选）通向 P3.5/P4/P5（组合/融合）的强制关卡。

### 4.6 promote 自动判定的硬契约（make_decision 必须满足）

§4.2 / §4.3 / §4.4 写的是判定逻辑，但 coding agent 在循环里只认 `make_decision` 返回的标签，不会回头读这些散文。凡是“散文比代码严”的地方，agent 就会往松的那头漂。所以把上面的判定固化成 `make_decision` 的硬契约，并**配单元测试**锁住：

**采纳标签判定表（make_decision 必须照此返回）：**

| 条件 | 返回 |
|---|---|
| `delta_corr < 0.003` | `reject` |
| `0.003 ≤ delta_corr < 0.005` | `review`（送 11 月复核，**禁止直接 promote**） |
| `delta_corr ≥ 0.005`，且 expanding 已单独跑过且不为负，且 short 与 expanding 同向为正、long 不显著翻负，且每日 IC 不恶化，且 `stability_score` 不降，且没有新的强负折 | `promote` |
| 其它 | `review` |

- **没跑 expanding 就不准 promote**：如果结果里没有 expanding（比如只跑了 short+long 的快车道批次），`make_decision` 最高只能返回 `review`，不得 `promote`。
- **排行榜排序键要改**：头条按 `protocol_corr_mean` 排（这才对齐老师评分，回答“这个大概能打多少分”），`stability_score` 作为并排展示的守门指标——**不要再拿 stability 当第一排序键**。但“采纳”与否仍走上表的鲁棒性门槛，不能因为排在前面就采纳。
- **量化口径（供单测落地，P0.5 可按基线 std 复核校准）**：`long 不显著翻负` = long 的 `delta_corr ≥ −0.006`（约一个均值标准误，见 §4.3）；`新的强负折` = 候选使某折 corr 由非负转负（基线该折 ≥ 0、候选该折 < 0），或某折 corr < −0.01。没有量化定义，单测断言无法落地。
- **必须有单测**：至少覆盖三条断言——“`delta_corr=0.004` 必须返回 `review`”、“缺 expanding 必须不能 promote”、“出现强负折（按上面定义）必须不能 promote”。

### 4.7 单候选 per-profile 复核方法论（promote 前必做）

`make_decision` 只是自动初筛。一个候选要 promote，必须人工（或 agent 按固定脚本）按下面的顺序钻进 per-profile 明细，从最能一票否决的看起：

1. **符号一致性**：看 short / long / expanding 三个的 `delta_corr` 方向与形态。三个都正、量级接近 = 结构性信号，最稳；**short 正、long/expanding 转负 = 这个信号“见效快但靠近期”，训练历史一拉长就失效——对“预测未知的未来”这是最危险的一类**。
2. **最差那一折**：新特征有没有制造出一个**新的强负折**（`corr_min` 明显变差）？均值升上去、最差折反而更烂，存疑。
3. **增益在时间上的分布**：把每一折的 corr 按日期排开看，增益是均匀分布，还是集中在某一段（比如只有 8 月那几折在涨）？集中 = 撞上了某段特定行情 = 脆；均匀 = 真。
4. **拆开看是“大盘的钱”还是“选股的钱”**：对比池化 corr 和每日截面 IC 的变化。池化 corr 涨、每日 IC 没涨 → 增益来自猜大盘方向，脆；每日 IC 也涨 → 真选股能力，稳。
5. **每日 IC 稳不稳**：看每日 IC 序列的“均值 ÷ 波动”，是真的更稳了，还是只把均值抬高、波动也跟着变大（虚胖）。
6. **（Ridge）系数符号稳不稳**：新特征的系数在各折符号是否一致；这折正、下折负地来回翻 = 在拟合噪声，不是信号——哪怕总分蹭高了也不算。

第 1~5 步基本都用现成输出就能看（per-profile delta、`corr_min`、每折日期、每日 IC），只是换个姿势读；只有第 6 步要额外记录每折的系数。

### 4.8 三层 Holdout 纪律（Dev / Review / Final）

§4.2 的两速结构都在第一层 Dev Rolling 内部展开；其外还有两层只读 holdout，逐层收口、绝不回头调参：

| 层级 | 数据区间 | 用途 | 频率 | 可否用于调参 |
|---|---|---|---|---|
| 第一层 Dev Rolling | 6–10月（内部切折） | 日常筛选、选模型、调参 | 随意反复 | 是 |
| 第二层 Review Holdout | 训练6–10月、验证11月 | 候选缩窄后复核 | 同一候选 ≤3 次 | **否** |
| 第三层 Final Holdout | 训练6–11月、验证12月 | 提交前一次性确认 | **只跑一次** | **严禁** |

- **Final Holdout 只跑一次**：12月是手里最像老师隐藏集的代理，可信度完全建立在"开发全程没看过"上。一旦看了结果再据此改任何东西（哪怕"感觉调一下"），它就退化成又一个被过拟合的验证集，失去模拟隐藏集的意义。
- **四步流程**：Dev Rolling 反复筛 → 只有 `promote` 的候选进 Review Holdout 复核 → 11月对得上才定方案、此后不再改 → Final Holdout 只确认、好坏都不再改提交决策。
- **什么会污染 holdout（视为该层作废）**：看了 11/12月结果后改了特征/参数/后处理再跑一次；把 holdout 结果当普通指标纳入选择循环；用 holdout 期真实 `y` 做任何归一化或残差。Review 一旦被污染，直接只依赖 Dev Rolling 决策，不要试图用 Final 补救。

## 五、实验开发 SOP

1. 明确要改的一个变量（特征组 / 目标变换 / 模型 / 窗口）
2. 在 `src/eval_protocol.py` 的 `ALL_SPECS` 里固定本轮候选 spec（§7.1）
3. 先过快车道筛选（`--suite daily`），过线候选再进慢车道关口（`--suite gate`），按 §4.4 口径评测
4. 把结果写入 `docs/实验记录.md`，标注 `experiment_id / date / feature_set / model / split / seed / metrics / notes`（§九 必录字段）
5. 与基线比较，只改一个变量；提交遵循 §八 commit 规范

### 5.1 跑命令前自查清单（每次执行 `experiments/*.py` 前逐条过）

1. **suite 选对了吗**：日常筛选 `--suite daily`（short+long+每日IC）；提交关口 `--suite gate --candidate-spec-id <ID>`（候选 vs 基线、只跑 expanding）。别拿 `full`/`ridge` 当日常。
2. **macOS 跑 expanding / gate 必须显式 `--n-workers 1`**：`gate` 入口默认会压到 2 worker，在这台 16GB Mac 上会撞「双并行 expanding 尾段」OOM 风险；标准跑法是单 worker 串行，口径见 `docs/specs/开跑前编码指导_评测口径与提速.md` §2c。只有迁到 20GB+ 空闲内存机器才开 2 worker。
3. **候选 spec 已一次性固定**：禁止边加 spec 边重筛（§4.3 多重比较）。本轮要比的全集先定死、一起跑、一起看。
4. **已锁默认别手改**：训练标签 winsorize = 开启 + P1/P99，ridge alpha = 2.0，特征 dtype = float32，都是 P0.5 扫锁结论（§7.11 / §7.7）。要做对照才显式覆写，且记录在案。
5. **长任务挂保护**：`caffeinate -i` 防休眠 + `experiments/run_with_memory_guard.py` 兜 RSS；长跑用 `--run-id` 固定、`--resume` 可续。
6. **目标与输出回 raw `fret12`**：winsorize 只作用于训练标签，评测/提交一律原始 `fret12`；提交通道不依赖 `data/features/` 缓存、不依赖任何不可见统计量（§十 推理契约）。

## 六、模型优先级

```
第一优先：Ridge
第二优先：ElasticNet / HuberRegressor
第三优先：浅 ExtraTrees（max_depth≤5）/ 浅 HistGradientBoosting
第四优先：受限融合（仅当 P1-P3 有独立有效信号后）
禁止：Transformer / LSTM / 复杂 MLP / 自由 stacking（当前阶段）
```

新信号必须先在 Ridge 验证有效，再考虑上浅树。

## 七、特征工程规范

- 特征按信号族组织，每族独立评估贡献
- 特征命名前缀对应信号族：`ofi_`、`trade_impact_`、`momentum_`、`regime_`、`cross_`
- 所有滑动计算（rolling / EMA / lag）只能用 `shift(1)` 及以前的数据
- 归一化策略：rolling zscore / cross-section rank / stock-level z-score，禁止全样本归一化

### 7.1 新增实验 spec 规程

新增实验组合 = 在 `src/eval_protocol.py` 的 `ALL_SPECS` 列表中添加一条 dict。必须包含：

```python
{
    "experiment_id": "O1_R02_plus_ofi_raw",    # 唯一 ID，前缀表示系列
    "type": "standard",                         # standard / common_residual / soft_regime
    "model": "ridge",                           # 模型名
    "target_mode": "raw",                       # 目标变换
    "groups": ["legacy", "norm_core", "ofi_raw"], # 使用的特征组
    "notes": "R02 plus raw OFI",                # 一句话说明
}
```

命名规范：`{系列}{编号}_{模型}_{特征描述}`。系列前缀：R=Ridge 基线，O=OFI，T=Trade Impact，C=Conditional Momentum。

### 7.2 交互项探索规程（手动操作）

探索阶段的交互项不注册为正式 stage：在实验脚本里直接用 pandas 临时构造（如 `ofi_total × spread`），跑 Dev Rolling 看 `delta_corr` 与跨 profile 一致性。

- 临时手写 → 过线（`delta_corr ≥ 0.003` 且跨 profile 一致）才提升为正式 stage：在 `src/feature_registry.py` 写 builder + `@registry.stage` 注册，再 `python -m feature_store build`。
- 禁止探索阶段就把临时交互项写进 registry，避免 stage 膨胀。
- **晋级红线**：一个临时交互项被第二个实验复用、或要进正式 rolling 对比，就必须移入 registry。一次性试错可手写，反复使用必须注册。

### 7.3 特征生命周期管理（手动维护）

每个特征组有四种状态，在 registry 的 `@registry.stage(status=...)` 中声明：

| 状态 | 含义 | 操作 |
|---|---|---|
| scratch | 探索中，交互项写在实验脚本里 | 不进 registry |
| candidate | 初步有信号，已注册为 stage | 进 registry，跑完整 rolling |
| promoted | 通过验收标准，进入主候选集 | 保留，参与后续融合评估 |
| archived | 失败或被替代 | 从 registry 移除，builder 代码移入 `.archive/` |

当前所有 9 个 stage（base / lag / roll / patch / ofi / trade_impact / cross / conditional_momentum / regime）状态均为 **promoted**。

特征组状态变更时，同步更新 `docs/实验记录.md` 中的对应条目，记录变更原因和关联实验 ID。

### 7.4 特征缓存一票否决规则

与 §二"设计宪法"的一票否决规则同一套，不重复列举：为试一个新特征要改多个无关模块、改一组导致全量重算、缓存失效规则不清、`meow.py` 离开本地 cache 跑不起来——任一条成立即特征管道设计有问题，必须修正。

### 7.5 Target Mode 准入规则

最终提交给老师评测的预测值，**必须对应原始 `fret12` 尺度**。因此 target mode 分两类：

**A. 可作为最终提交候选（允许继续实验）**

- `raw`：直接预测原始 `fret12`
- `interval_residual`：训练时预测残差，但**仅当**预测阶段能用测试时已知信息把公共基线严格加回，最终输出恢复为原始 `fret12`
- `common_residual`：先预测 common component，再预测 residual，最终显式重构为原始 `fret12`
- 其他未来新增 mode：只有在**不使用验证/测试真实 y 统计量**的前提下，能把预测值严格恢复到原始 `fret12`，才允许进入正式 rolling

**B. 不可作为最终提交候选（原则上放弃）**

- `date_demean`
- `interval_demean`
- `rank target` / `quantile target`
- 任何需要用验证/测试真实 y 的均值、rank、std、分位数做还原的 mode
- 任何最终输出仍停留在 residual / demean / rank / zscore 尺度的 mode

**准入判定原则：**

- 如果 target 变换后**无法在测试时仅依赖已知特征、已知 meta、训练期固定参数**恢复原始 `fret12`，则该 mode 只能作为研究分支，不能进入最终提交候选集
- 如果 target 变换可逆，但逆变换依赖验证/测试真实 y，视为泄漏，禁止使用
- 若存在争议，默认按“不可提交”处理，直到恢复路径被明确写清并代码验证通过

### 7.6 阶段推进与调参边界

P0-P5 的默认职责如下，除非用户明确改变路线，否则按此执行：

| 阶段 | 主目标 | 默认允许变动 | 默认禁止事项 |
|---|---|---|---|
| P0 | 建立统一 rolling 评测基线 | 修评测协议、补跑 profile、确认 baseline | 混入大量新特征 / 新模型 / 新 target mode |
| P1-P3 | 验证新特征组（OFI / trade impact / conditional momentum） | 变动 feature groups；模型固定 `Ridge`；target 固定 `raw` | 大规模模型调参；把弱信号带进复杂模型 |
| P3.5 | 少量跨信号族交互项冲刺 | 手工加入少量 scratch 交互列做 rolling 验证 | 大规模组合爆炸；尚未验证就写进正式 registry |
| P4 | 稳健传统模型比较 | `Ridge / ElasticNet / Huber / 浅 ExtraTrees / 浅 HistGB` 小范围调参 | 深模型；大网格搜索；用 holdout 调参 |
| P5 | 受限融合 | 少量有效分支的权重搜索 / 二层线性融合 | 自由 stacking；把弱分支也拉入融合 |
| 交付演练 | 验证最终提交路径 | raw `fret12` 输出、`meow.py` 可独立运行、无泄漏检查 | 借机再开新实验方向 |

### 7.7 各阶段允许的调参粒度

- `P1-P3`：重点是**换特征定义**，不是反复微调同一个 spec
- `P3.5`：允许少量离散型交互结构搜索，如 `ofi_total x spread`，但不允许扩成参数网格
- `P4`：才正式允许系统性模型调参，但应以小范围人工扫为主，不做大规模网格
- `P5`：允许融合权重、二层 Ridge/ElasticNet 正则强度等小规模调参
- `Review Holdout / Final Holdout`：禁止作为普通调参集使用

**Ridge alpha 口径（2026-05-24 拍板）：**

- 标准 ridge 路径当前固定 `alpha=2.0`（前接 StandardScaler，见 `src/experiment_runner.py` 的 `fit_model`）
- 进 P1 前先做一次性扫描：**仅用 R02 baseline、仅 short+medium**，在 `{0.5,1,2,5,10,20}` 上确认 2.0 落在平台区（或取平台中心），随后 **P1–P3 全程锁定该 alpha**，不逐 spec 调
  - 这次标定刻意用 short+medium、而非日常筛选的 short+long：扫平台区要的是折数多、曲线平滑，medium 的 ~16 折在这里有用；也不用 expanding（它太贵、且要少看，不适合反复评估的标定扫描）。属于 §4.2"medium 移出日常"的**有意例外**，仅限这次一次性标定
- 锁定理由：P1–P3 是单变量换特征对比，alpha 若随特征集大小变动，会把”特征是否有效”和”alpha 是否合适”混在一起，污染对比
- per-fold / per-spec 的 alpha 调参属于 **P4**，禁止提前在 P1–P3 做
- promote 关口可对候选做一次 3–4 点 alpha 敏感性抽查，确认增益不是 alpha 设错造成的假象
- **P0.5 实测锁定（2026-05-25）**：`results/p05_alpha_winsorize/20260525_e18_full_v1_summary.csv` 显示 `alpha ∈ [0.5, 20]` 基本全在平台区（最佳 winsor 档内 `protocol_corr_mean` 跨 alpha 波动约 `1.0e-5`），因此继续把标准 ridge 默认锁在 **`alpha=2.0`**，作为平台区中部值。

判断”换 spec”还是”调参”的口径：

- 换 spec：feature groups / 交互结构 / target 路线改变
- 调参：spec 主体不变，只调整模型超参数、融合权重、后处理参数

### 7.8 P4 模型比较准入标准

一个新特征分支进入 P4 前，至少满足：

- 相对当前基线 `delta_corr >= 0.003`
- `protocol_stability_score` 不下降
- `protocol_corr_min` 不明显变差
- 不出现新的强负 fold
- profile 结论尽量一致；在 expanding 未齐时，至少 `short + long` 同向为正

设计原因：

- 防止弱信号被复杂模型“化妆”
- 节省实验预算
- 面向隐藏测试集时，把泛化性前置

### 7.9 P5 融合准入标准

一个分支进入融合池前，除满足 P4 准入标准外，还应满足：

- 已在单模型下证明自己独立有效
- 与当前主分支不是高度同质，最好能在不同 profile / 时段提供补充
- 若使用二层融合，必须使用 OOF prediction，禁止直接用同折预测训练融合器
- 若最佳融合不如 backbone 稳，最终提交 backbone，不为融合而融合

### 7.10 P3.5 交互项冲刺规则

- 只允许少量高先验交互项，优先测试：
  - `ofi_total x spread`
  - `ofi_total x trade_activity`
  - `trade_pressure_qty x spread`
  - `lagret12 x ofi_total`
  - `lagret12 x order_pressure`
  - `trade_pressure_qty x regime_score`
- 交互项默认先作为 scratch 写在实验脚本里
- 任一交互项如果被第二个实验复用，或准备进入正式 rolling 对比，必须晋级为 registry stage

### 7.11 训练目标 winsorize（P0 级口径）

`fret12` 尾巴很厚（峰度约 15，超过 5σ 的极端值约占 0.5%，且没有涨跌停截断）。用 MSE 训练时，少数暴涨暴跌样本因为误差被平方，会对 Ridge 系数产生过大影响，把拟合往尾部拽——而评分用的是相关性，关心的是大多数样本排得对不对，不是极端点拟合得多准。两者方向相反，所以把训练目标 winsorize 列为 P0 级口径：

- **只裁训练目标**：按**训练集**分位数裁两侧，只作用于 `ytrain`。P0.5 扫描后当前正式默认值锁为 **`P1 / P99`**（`lower_quantile=0.01, upper_quantile=0.99`）；候选里 `P0.5 / P99.5` 作为对照保留，但不再是默认口径。
- **测试 / 提交永远输出原始 `fret12`**：winsorize 只改训练标签、不改预测尺度，不需要逆变换，所以对 target mode 准入（§7.5）无影响、也不构成泄漏。
- **和 alpha 一起锁定**：clip 候选 `{P0.5/P99.5、P1/P99、不裁}` 跟 alpha 在 P0.5 同一次 short+medium 轻扫里一起定，落平台区就锁死，P1–P3 不再逐 spec 调。
- **它对评分是有影响的，别误解**：winsorize 不改 eval 时的算法本身，但它改了训练出来的系数，系数变了预测就变了，最终的池化 Pearson 自然会变（系数更贴主体、少被尾部带偏）。不是“只稳梯度、跟 Pearson 无关”。到底好不好是经验问题（极端 `y` 是真信号还是噪声事先不知道），所以默认带上、但以 rolling 实测为准。
- **P0.5 实测锁定（2026-05-25）**：在同一份 short+medium 扫描里，`P1/P99` 整体优于 `P0.5/P99.5` 和 `不裁`：最佳 `protocol_corr_mean` 约 `0.054181`，高于 `P0.5/P99.5` 的 `0.054123`，也明显高于 `不裁` 的 `0.053483`。因此标准训练口径正式锁为 **winsorize 开启 + `P1/P99`**。

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

- P1 正式迭代之前，先在一个 held-out 交易日把 `meow.py` 这条链路完整跑通（`genFeatures → predict → forecast` 列对齐、每行有限），确认契约成立。
- 之后所有实验都长在这条已验证的提交通道上。量化最常见的翻车不是模型差，而是 train/serve skew：某个特征在提交时的算法和训练时不一致，或者某个归一化依赖了测试时根本拿不到的统计量。
- `meow.py` 不能依赖 `data/features/` 本地缓存、不能依赖任何不可见的统计量；交付演练时按这几条逐条核对。
