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
MSE / R²             不明显恶化
```

如果某一类特征只让一个 fold 变好，其他 fold 变差，则放弃。

### 4.2 Per-profile 一致性检查（protocol 分数通过后必做）

`protocol_stability_score` 只是第一道筛选器，通过后必须手动检查 per-profile 明细：

**必须核查的项目：**
- 4 个 profile 中至少 3 个的 `delta_corr > 0`（跨窗口长度一致性）
- 各 profile 的 `rolling_corr_min` 不出现强负值
- short 与 expanding 的结论方向一致（两者相反则信号对训练历史长度敏感，标记为 review）

**信号强度分级（基于 105 天数据的置信度）：**

| 情况 | 结论 |
|---|---|
| delta_corr ≥ 0.005，short + medium + expanding 三个 profile 一致 | 有信心，可 promote |
| delta_corr = 0.003～0.005，2 个以上 profile 一致 | 边界，进 Review Holdout 验证后决策 |
| delta_corr ≤ 0.003，或仅 1 个 profile 支持 | 不接受 |

**关于 long_40d_5d：** 只有 ~6 折，方差极大。它的负结论（信号在长窗口下明确有害）是有效信息；正结论不单独作为 promote 依据。

### 4.3 多重比较与边界增益纪律

Dev Rolling 只有 ~105 个交易日，short profile 实际仅 ~5–6 个独立样本。R02 基线 `rolling_corr_std≈0.015`，对应均值标准误 ≈ 0.015/√6 ≈ **0.006**。

**因此单个 profile 上的 +0.003 低于一个标准误，本身无法与噪声区分**，必须靠跨 profile 一致性 + Review Holdout 才能下结论。

- 边界增益（`delta_corr` 0.003～0.005）**禁止只凭 Dev Rolling 直接 promote**，必须进 11 月 Review Holdout 复核后再决策
- 每个阶段开跑前**先固定候选 spec 集合**，一次性跑完一起看；禁止"边加 spec 边重筛"——反复试探等于把 Dev 窗口磨成被过拟合的调参集（garden of forking paths）
- 对同一份 Dev Rolling 比较的 spec 越多，纯靠运气过线的越多：候选集越大，promote 的一致性与门槛要求越要从严

### 4.4 Profile 分层执行与 expanding 硬门槛

隐藏测试是时间外推，`expanding_40d_5d` 最接近真实部署（累积全历史 → 向前预测），是泛化判断里最有代表性的 profile；而日常筛选用的 short/medium 是滑窗，最便宜也最不像真实任务。为同时兼顾成本与代表性，按层执行：

| 层 | profiles | 用途 | 频率 |
|---|---|---|---|
| 内层筛选 | short + medium | P1–P3 日常换特征筛选 | 随意，目标 ~15–20 min |
| 负向 veto | long | 候选通过内层后，确认无"长窗口明确有害" | 候选过内层才跑 |
| promote 硬门槛 | expanding（串行 + memory guard） | 确认时间外推下信号仍成立 | promote 关口**必跑** |

- long 的正向结论不单独采信（仅 6 折），故内层**不必每轮跑 long**，避免为否决票付固定大成本
- **expanding 是 promote 的硬门槛**：任何候选 promote 前必须单独串行跑过 expanding，expanding 上不成立则不得 promote，**不得因为它慢而跳过**

### 4.5 组合不可加：叠加后必须重跑

P1–P3 的实验都是"R02 + 单个特征组"。即使多个组各自单独过线，也**不能假设增益相加**：

- 特征间常有信息重叠（如 OFI 与成交冲击都刻画买卖压力），合并后真实增益通常**小于**各自之和
- 维度变多在固定 alpha 下可能拟合噪声，合并模型甚至可能**差于**只加最好的单组
- 偶有互补使合并**大于**之和，但不可预先假定

**规则：任何把多个特征组叠加使用的方案，必须作为一个新 spec、走同一套 rolling 重新验收，禁止用各组单独 delta 推算合并分数。** 这是 P1–P3（单组筛选）通向 P3.5/P4/P5（组合/融合）的强制关卡。

## 五、实验开发 SOP

1. 明确要改的一个变量（特征组 / 目标变换 / 模型 / 窗口）
2. 在 `experiments/` 下新建或复用对应阶段脚本（p1_ofi_validation.py 等）
3. 用 P0 确立的标准 rolling 口径评测
4. 把结果写入 `docs/实验记录.md`，标注 `experiment_id / date / feature_set / model / split / seed / metrics / notes`
5. 提交（遵循第六节 commit 规范）
6. 与基线比较，只改一个变量

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

探索阶段的交互项不需要注册为正式 stage。在实验脚本中直接用 pandas 计算：

```python
# experiments/ 下的实验脚本中
xtrain["ofi_x_spread"] = xtrain["ofi_total"] * xtrain["spread"]
```

验证流程：
1. 在实验脚本中手动添加交互列，跑 Dev Rolling
2. 如果 `delta_corr ≥ 0.003` 且跨 profile 一致 → 提升为正式 stage
3. 提升方法：在 `src/feature_registry.py` 中写 builder 函数 + `@registry.stage` 注册
4. 重建特征缓存：`python -m feature_store build`

禁止在探索阶段就把临时交互项写入 registry。避免 stage 膨胀。

**晋级红线**：一个临时交互项如果被第二个实验复用，或准备进入正式 rolling 对比，就必须移入正式 registry。一次性试错可以手写，反复使用必须注册。

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

以下任一条成立，说明特征管道设计有问题，必须修正：

- 为了试一个新特征，需要手动改多个无关模块
- 改一个特征组导致全量 144 天所有组无脑重算
- 缓存失效规则不清楚，结果不知来自新代码还是旧缓存
- 最终 `meow.py` 离开本地 cache 就跑不起来

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
- 锁定理由：P1–P3 是单变量换特征对比，alpha 若随特征集大小变动，会把”特征是否有效”和”alpha 是否合适”混在一起，污染对比
- per-fold / per-spec 的 alpha 调参属于 **P4**，禁止提前在 P1–P3 做
- promote 关口可对候选做一次 3–4 点 alpha 敏感性抽查，确认增益不是 alpha 设错造成的假象

判断”换 spec”还是”调参”的口径：

- 换 spec：feature groups / 交互结构 / target 路线改变
- 调参：spec 主体不变，只调整模型超参数、融合权重、后处理参数

### 7.8 P4 模型比较准入标准

一个新特征分支进入 P4 前，至少满足：

- 相对当前基线 `delta_corr >= 0.003`
- `protocol_stability_score` 不下降
- `protocol_corr_min` 不明显变差
- 不出现新的强负 fold
- profile 结论尽量一致；在 long / expanding 未齐时，至少 `short + medium` 同向为正

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

## 八、三层评测体系（方案 B）

### 8.1 三层的目的与限制

| 层级 | 数据区间 | 用途 | 允许频率 | 结果可否用于调参 |
|---|---|---|---|---|
| 第一层：Dev Rolling | 6月～10月（内部切折） | 日常开发、特征筛选、模型选择 | 随意，反复跑 | 是 |
| 第二层：Review Holdout | 训练6~10月，验证11月 | 候选模型缩窄后复核 | 谨慎，同一候选不超过3次 | **否** |
| 第三层：Final Holdout | 训练6~11月，验证12月 | 最终提交前的一次性确认 | **只跑一次** | **严禁** |

### 8.2 为什么 Final Holdout 是"一次性"的

老师的隐藏测试集是某段你没见过的未来数据。12月是我们手里最靠近那段数据的时间区间，所以它是最好的代理。

**完整性的前提是：你在整个开发过程中从未看过它。**

一旦你看了 12月结果，就等于悄悄知道了答案的一部分。如果再根据这个结果调参——即使只是"感觉调一下"——12月的可信度就归零了，它变成了又一个被你过拟合的验证集，失去了模拟隐藏集的意义。

> 类比：期末考试的卷子提前泄题，你做了一遍，再去"复习"——这次考试的成绩不能代表你的真实水平。

### 8.3 正确的四步开发流程

```
步骤 1：Dev Rolling（反复跑）
  → 用全量 rolling protocol 筛选特征/模型/参数
  → 只有 decision=promote 的方案进入下一步

步骤 2：Review Holdout（11月，谨慎跑）
  → 对步骤1筛出的少数候选做一次复核
  → 如果 11月结果和 rolling 一致 → 有信心
  → 如果 11月结果崩了 → 重回步骤1，不能根据11月结果直接改参数

步骤 3：确定最终提交方案
  → 此时不再改任何东西

步骤 4：Final Holdout（12月，只跑一次）
  → 纯粹确认，结果好坏都不影响提交决策
  → 结果记录在 docs/实验记录.md，仅供复盘
```

### 8.4 什么行为会污染 Holdout

以下行为会让对应 holdout 层失去可信度，应当视为该层作废：

- 看了 11月/12月结果后，**修改了特征、模型参数、后处理逻辑**，然后再跑一次
- 在调参循环中**把 holdout 结果当作一个普通指标**纳入选择
- 用 holdout 期间的真实 y 做任何归一化或残差计算

如果 Review Holdout 已被污染，应当**直接跳过，只依赖 Dev Rolling 结果**做决策，不要试图用 final holdout 来补救。

## 九、Git 提交规范

### 9.1 粒度原则

- 每次 commit 代表一个明确的变更：特征组、目标变换、模型、split 逻辑、评测逻辑
- 不混提：特征工程 + 模型调参 + 文档清理禁止一次提交

### 9.2 提交格式

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

### 9.3 不可提交的内容

- `data/`（原始数据）
- `results/`（中间结果 CSV）
- `*.log / *.out / *.err`
- `__pycache__/`

### 9.4 每次实验 commit 必须记录

`experiment_id / date / feature_set / target_type / model / split_config / seed / rolling_corr_mean / rolling_corr_std / stability_score / notes`

## 十、目录约定

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
