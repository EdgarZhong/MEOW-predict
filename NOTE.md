# NOTE.md — 实验笔记（DL 主线）

> 给我自己看的研究日志：只记最近敲定的 DL 设计**为什么这么定**。规则正文在 `AGENTS.md`，任务看板在 `CLAUDE.md`，完整架构在 `docs/specs/DL实验设计规格.md`。
> 传统/线性那一整套（评测口径、特征侧 P1–Q4、转树、选模型协议）的笔记留在传统分支，本分支不再展开。

---

## 战略：直接冲 0.12，不验"序列有没有料"

老师明示有人做到过 0.12、业界 0.2–0.3 → 序列有信号是**先验事实**，再花预算去"验证存在性"是浪费。所以 DL 起手就奔 0.12，只保留"别骗自己"的评测纪律。隐藏集分布未知（甚至不确定是不是时间外推）→ **鲁棒性优先**，冲的是"可辩护、可泛化的分"，不是 dev 刷高。传统侧已基本可提交、是保底代表，DL 是上行主线（不是"打不过就回退"的赌注）。

## 架构：固定脊柱 + 可换卡带

把**评测协议/窗口切分/归一化/指标/配置**做成不可变脊柱，把**输入适配(InputAdapter)**和**模型本体(ModelCartridge)**做成可换卡带。为什么这么切：换模型（LSTM-on-features → DeepLOB-on-rawLOB）时只动两块卡带，脊柱一行不改——所有"理解原始数据语义"的活关进 InputAdapter，它一旦吐出通道布局固定的 `[n_interval, C]`，下游全是不关心含义的张量数学。

**PyTorch 封死在卡带内**（torch import / nn.Module / optimizer / GPU / 训练循环都在 `fit/predict` 里），脊柱全程 torch-free、不碰一个 tensor。这是脊柱能复用、能换模型的根。

## 评测：海选 + expanding 两段（不是三层 holdout）

DL 有两类盲区、配两类工具：① **单窗口内**靠 train vs 早停验证两条曲线就能看出过拟合（自带探测器）；② **walk-forward 多折**探测时间不稳定性（撞运气）。所以最优分配 = 单窗口看曲线 + 少量 expanding 折看稳定性，不堆很多折（DL 贵、数据有限）。

落成两段预算：**海选**（单切分 + 早杀，便宜分诊大量 config）→ **expanding 认证**（少折 walk-forward + 多种子，最终确认）。**不再用传统"三层 Dev/Review/Final"**——第三层 12 月从来没执行过，新协议已用 海选+expanding 取代分层。防自欺只留内核：config-lock（获胜超参在交付演练前冻结）、少看 expanding（看多会过拟合）、最终确认数据看完即定不回灌。

## 训练生命周期：内层归 PyTorch，外层归框架

内层 epoch 循环/optimizer/早停归卡带（PyTorch）；外层协议（折切分、算指标）归脊柱。**早停看卡带自己的"早停验证集"**（训练区尾段切出、与打分集隔离），可用代理指标——这样不会对最终上报的打分集过拟合。训练**之中无数据回流脊柱做控制**；卡带只通过 TrainRecord（逐 epoch 曲线）**单向上报**，供脊柱判过拟合 + HPO 早杀整个 trial。官方 4 指标在 `predict()` 之后由脊柱算一次。

## 超参搜索：只搜结构 3 个旋钮

只搜 **序列长 / hidden / 层数** 三个结构超参，其余冻默认；随机搜索 ≫ 网格；早杀坏 trial；训练行采样降成本；海选少种子、认证多种子。理由：结构才是主战场，省下的算力砸给"试更多结构"，而不是在一个结构上精调一堆次要旋钮。

## 配置管理：分布式声明 + 中央组装

每层声明自己职责的配置块，Orchestrator 只**组装**成一份 frozen `RunConfig` + 校验 + 冻结 + 派发。不写成 god-object（否则换模型还得回头改 Orchestrator，泛化性就废了）。三件事拆开别糊：**①合法词表(枚举)集中**在各配置块文件顶部、**②实现注册(枚举→类)**挨着实现放 registry、**③本次选了哪个值**进 RunConfig。cartridge 用 `required_adapter: AdapterKind` 类型化引用，不写裸字符串（拼错当场报错）。

**run_id 手工语义命名**（`日期_阶段_模型_意图_版本`，看得懂），哈希降级成 `config_fingerprint` 字段防"复用同名却悄悄改了配置"。纯哈希看不懂、纯手工易漂移，这样两头都占。

## 数据实情 → 路线收敛：DeepLOB 退役，改 TCN-on-原始微结构

接 PyTorch 前先查了真实 h5（`data/20230601.h5`，62 列）才定路线，不是拍脑袋：

- **没有连续 LOB**：盘口只给 4 个稀疏聚合档位（`bid0/4/9/19` + `ask` 同构），外加聚合 size（`bsize0 / bsize0_4 / bsize5_9 / bsize10_19`）与 turnover ratio（`btr0_4 / atr0_4 …`）。**不是 DeepLOB 假设的连续 10 档价量网格**——照搬 DeepLOB 的「10×2 网格 + Inception」根本没有对应输入，**DeepLOB-on-rawLOB 退役**。
- **真正能喂的 = ~59 个原始微结构通道**：价（`midpx / lastpx / OHLC` + 4 档买卖价）、量（聚合 size / turnover ratio）、订单流（主动买卖的笔数/量/额/高低/vwad，挂单、撤单同构）。
- **序列粒度**：每票每日 ~226 个 interval（日内 bar），即「序列」= 某票某日 226 步日内路径，**绝不跨日**（脊柱 WindowIndexer 已物理保证）。

→ 收敛成一句话：**第一个猛药 = TCN，喂清洗过的原始微结构通道（RawChannelAdapter），让网络自己学通道间交互**（对照 FeatureAdapter 喂 433 手工特征那条线）。

## 为什么 TCN（不是 Transformer，也不是堆 LSTM）

先把决策拆成两件正交的事：**喂什么（输入）** 和 **用什么结构（架构）**。我的判断——**输入/归一化/正则是大杠杆，架构是小杠杆**：同一份干净输入下 TCN / LSTM / GRU 的天花板差距，远小于「输入对不对、归一化漏没漏未来、正则够不够」带来的差距。所以架构就选「训练便宜 + 因果天然 + 局部模式归纳偏置对路」的那个，把省下的算力砸去搜结构、试输入。

TCN 胜出三条：
1. **因果是结构自带的**（causal dilated conv，padding 只补左侧）——不靠我手动 mask 防未来泄漏，和脊柱「窗末因果对齐」叠成双保险。
2. **训练并行、省 GPU**：卷积沿时间并行，不像 RNN 必须串行 BPTT；4060 预算紧，这条很值（省下的跑更多结构搜索）。
3. **归纳偏置对路**：膨胀卷积的多尺度局部感受野，正贴「订单流的短程脉冲 + 多尺度累积」；226 步路径用膨胀卷积几层就能覆盖全程。

Transformer 不选：注意力 O(L²)、小数据易过拟合，还得额外加因果 mask + 位置编码，这个数据规模性价比不如 TCN。LSTM 留作 **D1 低风险对照**（喂 433 特征当序列），不是主攻。经验背书：Bai/Kolter 2018《An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling》——多数序列任务 TCN ≥ LSTM/GRU。

> 落地分工：`TCN` 进 `ModelKind` 词表，卡带本体（nn.Module / 训练循环）是 torch，等 4060；现在能做的（RawChannelAdapter + Orchestrator + HPO Searcher + 全套 torch-free 测试）先做完，接卡那天只剩写 `TCNCartridge.fit/predict`。

## RawChannelAdapter 的语义归一（为什么只做最小变换）

「喂原始」≠ 裸喂，但也**不该在 adapter 里手搓 imbalance/OFI**——那是 FeatureAdapter（433 手工特征）干的活；走 raw 这条线的全部意义就是**让 TCN 自己学交互**。所以 adapter 只做「平稳化 + 量纲驯服」的最小语义层，统计白化仍交脊柱 Normalizer：

- **价 → 相对 mid**：各档买卖价 / OHLC / 成交高低 / vwad 取 `p/midpx - 1`（行内、无状态、平稳、无量纲）；成交价为 0（该 interval 无成交）时置 0（中性，避免 `(0-mid)/mid=-1` 的假信号）。
- **midpx 本身 → 日内对数收益**：按 `(symbol)` 组内 `log(midpx_t) - log(midpx_{t-1})`，首 interval=0。这是**最关键的收益路径通道**，因果（只用 ≤t）、不跨日不跨票。
- **量 / 笔数 / 额 → log1p**：size / turnover / `nTrade*` / 挂撤单量额都是非负厚尾，`log1p` 驯尾。

边界口径沿用规格 §2.3：**语义归一在 adapter，统计白化在脊柱 Normalizer**。RawChannelAdapter 全程 torch-free、行内/组内无状态变换，无跨日跨票泄漏风险，可在 Mac 上对真实 h5 直接验。

## 喂数据：DL 喂"原始" ≠ 裸喂

序列模型输入是 LOB/成交的时间路径，但必须先**平稳化**（价转收益、量转相对不平衡）+ **归一化**（前日/滚动 z-score）+ **防泄漏开窗**；归一化是最易翻车的一步。标准张量是 `[B, L, C]`，我们的"序列" = 某票某日的**日内 interval 序列**，**绝不跨日**。当前契约锁单张 `[L,C]`（覆盖 LSTM-on-features 与 DeepLOB），多粒度/跨票图结构留扩展缝（InputAdapter 输出放宽成张量字典，仅窗口器+归一化器需泛化）。**语义归一（如按 mid-price 相对化）放 InputAdapter，统计白化放脊柱 Normalizer**，边界这么分最干净。

数据要点（影响 DL 开窗/归一化，细节见传统分支）：`fret12` = 简单收益、不跨夜、厚尾（峰度~15）、每截面~300 票、主路径无结构泄漏。

## 防泄漏：numpy 参考模型当探测器

一个 torch-free 的 numpy 参考模型（恒 0 / 末 interval 线性）走**完整脊柱管线**必须打**低分**；打高分 = 脊柱漏了未来信息（窗口/归一化/标签对齐有 bug）。这是 D0 的核心验收闸。

---

## 待决 / 未决问题

- [ ] 老师测试集是否一定时间外推？（假设是、未确认——头号根约束，故鲁棒优先）
- [ ] 提交端：老师 `fit()` 现场重训，DL 需 GPU + 训练时长；带预训练权重是否被规格允许，待确认（与传统 fit/predict 签名核验一并在交付会话处理）
- [ ] 单粒度 `[L,C]` 够不够冲 0.12，还是需要多粒度/跨票（按扩展缝再说）

## 版本记录

| 日期 | 内容 |
|---|---|
| 2026-05-31 | 接 PyTorch 前查真实 h5 定路线：记录①数据实情（无连续 LOB、~59 原始微结构通道、~226 interval/日）→ DeepLOB 退役、改 **TCN-on-原始微结构**；②为什么 TCN（架构是小杠杆、输入/归一化/正则是大杠杆 + 因果自带 + 省 GPU + 归纳偏置对路，Transformer/堆 LSTM 不选的理由）；③RawChannelAdapter 最小语义归一口径（价相对 mid / midpx 日内对数收益 / 量 log1p，为什么不手搓 imbalance） |
| 2026-05-31 | NOTE 重写为 DL 主线笔记（传统/线性全套笔记留传统分支）：记录战略(直接冲0.12)、脊柱+卡带架构、海选+expanding 两段评测(替代三层holdout)、训练生命周期归属、HPO 三旋钮、配置管理(三层枚举+run_id)、喂数据(原始≠裸喂+单张张量+扩展缝)、numpy 参考模型防泄漏 —— 各项"为什么"。完整架构见 `docs/specs/DL实验设计规格.md` |
