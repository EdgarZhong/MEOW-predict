# MEOW 金融时序预测

分钟级 A 股订单簿数据的股票收益预测研究项目。目标：预测未来 12 分钟收益 `fret12`。

## 项目定位

- 数据：144 个交易日（2023-06-01 ~ 2023-12-29），每日一个 `.h5` 文件，约 2.45 GB
- 任务：截面预测，老师精度分 = MSE + Pearson + R² 各 1/3；研究侧另看 `rolling_corr_mean`（IC 均值）与 `stability_score`
- 两条主线：
  - **传统侧**（已收口、保底代表）：等权 `raw_mean` 融合 [X1 ridge + M_lgbm_d4]，可提交。
  - **DL 主线**（上行，冲 0.12 / Windows 4060 / PyTorch）：固定脊柱 + 可换卡带架构；**D0 地基已落地、全程 torch-free、Mac CPU 可测**，PyTorch 卡带等 4060 就绪后接入。
- 动态进度与决策：见 `CLAUDE.md` 与 `docs/实验记录.md`

## 目录结构

```
MEOW--predict/
├── src/                       # 核心代码（脊柱 + 传统侧实现，平铺 import）
│   ├── feature_registry.py    # 特征族注册表（@registry.stage builder）
│   ├── feature_store.py       # 特征构造 + 落盘（python -m feature_store build）
│   ├── feature_loader.py      # 特征加载（resolve_groups + manifest，默认 float32）
│   ├── eval_protocol.py       # 传统评测协议（Rolling Profiles / Leaderboard / make_decision）
│   ├── experiment_runner.py   # 实验编排（特征/模型/训练/评估核心逻辑）
│   ├── scheduler.py           # 并发调度（ProcessPoolExecutor + resume + 成本均衡切组）
│   ├── submission_pipeline.py # 提交桥接层（正式特征现算 + 复用 runner 训练/推理核心）
│   ├── trainer.py             # BaseTrainer ABC + TabularTrainer（DL 扩展点）
│   │                          # ── DL 脊柱（torch-free）──
│   ├── sequence_dataset.py    # WindowIndexer + Normalizer + SequenceDataset（惰性 [B,L,C]，不跨日/不跨票）
│   ├── dl_protocol.py         # DL 协议引擎（walk-forward 三段切分 + embargo + 4 指标 + 防泄漏检查）
│   ├── dl_trainer.py          # SequenceTrainer(BaseTrainer)：编排 adapter→indexer→normalizer→cartridge
│   ├── feat_legacy.py         # teacher 原始 6 特征（legacy 对照）
│   ├── dl.py / mdl.py / eval.py / tradingcalendar.py / log.py  # 老师原始数据/模型/评估模板 + 日历/日志
├── config/                    # DL 配置块（frozen dataclass + 枚举顶部；平铺 import）
│   ├── model_config.py        # ModelKind 枚举 + ModelConfig
│   ├── adapter_config.py      # AdapterKind 枚举 + AdapterConfig
│   ├── protocol_config.py     # Stage / ProfileKind 枚举 + ProtocolConfig
│   ├── search_config.py       # SearchConfig（n_trials / 早杀钩子 / 行采样 / search_space 收窄）
│   ├── exec_config.py         # ExecConfig（seeds / device / resume / reuse_checkpoint / out_dir）
│   └── run_config.py          # RunConfig 组装 + 跨块校验 + config_fingerprint
├── models/                    # DL 可换卡带（唯一允许 torch 的层；平铺 import）
│   ├── dl_models.py           # InputAdapter 接口 + IdentityAdapter / FeatureAdapter + numpy 参考模型
│   └── registry.py            # 枚举值 → 实现类的注册表（required_adapter 校验源）
├── experiments/               # 实验入口脚本
│   ├── p0_eval_protocol.py    # 传统 Rolling 评测基准（daily/gate/ridge/quick/full suite）
│   ├── p05_lock_ridge_alpha_and_winsorize.py # alpha + winsorize 扫锁
│   ├── run_with_memory_guard.py # 通用 RSS 内存看门狗包装器（仅 Unix）
│   ├── run_submission_full_window.py # 交付链全窗口 fit/eval + 内存峰值采样（跨平台，Windows 演练用）
│   └── legacy/                # 历史脚本（可运行、不主动改）
├── meow/                      # 老师提交目录（python meow.py 入口；正式提交通道）
├── tests/                     # 单元测试（unittest）
│   └── test_dl_pipeline.py    # DL D0 六项验收闸（端到端/参考模型低分/无泄漏/不跨日跨票/归一化/config）
├── data/                      # 原始 .h5 数据（gitignored）；data/features/ 特征缓存（gitignored）
├── results/                   # 实验结果 CSV（gitignored）
├── .archive/                  # 废弃/归档文件（gitignored，不主动追踪）
└── docs/                      # 文档（specs/ 规格、archived/ 历史快照只读）
```

## DL 工程地基（脊柱 + 卡带）

DL 主线把**协议 / 窗口 / 归一化 / 指标 / 配置**做成不可变**脊柱**，把**输入适配 + 模型本体**做成可换**卡带**；PyTorch 封死在卡带内，脊柱全程 torch-free。换模型（LSTM-on-features → DeepLOB-on-rawLOB）只动两块卡带，脊柱零改。

- **脊柱**（`src/sequence_dataset.py` / `src/dl_protocol.py` / `src/dl_trainer.py`）：
  - 序列粒度 = 「同一票同一天」的日内 interval 序列（`[B, L, C]`），**绝不跨日、绝不跨票**；标签因果对齐窗末、warmup 不足窗丢弃。
  - 评测 = 三段切分 `[ train_core | earlystop-val | embargo | scoring-val ]`；4 指标（corr/MSE/R²/daily-IC）口径与传统侧 `experiment_runner` 逐字对齐。
  - 防泄漏三道物理保证：窗口不跨界 + 标签因果对齐 + Normalizer 只用训练统计；由 numpy 参考模型（末步线性）兜底探测。
- **卡带**（`models/dl_models.py`）：
  - `InputAdapter`：`FeatureAdapter`（包装 433 特征管线，零新公式）/ `IdentityAdapter`（调试 & 合成测试）。
  - `ModelCartridge`：D0 仅含 torch-free numpy 参考模型（恒 0 / 末步线性，作防泄漏探测器与 corr 基线）；LSTM / DeepLOB 等 4060 + PyTorch 就绪后再加。
- **配置**（`config/`）：分布式声明 + 中央组装；`RunConfig` 为 frozen dataclass（= config-lock 机械实现），`assemble_run_config` 在组装期校验 required_adapter / 阶段搭配 / 数据窗口，并算 `config_fingerprint` 防漂移。

设计真相源：`docs/specs/DL实验设计规格.md`。

## 运行环境

- Python 3.13（mise 全局环境）
- 基础依赖（含 DL D0 地基，全程 torch-free）：`pip install --system numpy pandas scikit-learn tables`
- 可选：`psutil`（交付链内存采样演练 `run_submission_full_window.py` 必需）
- DL 卡带（D1/D2，待 4060）：将额外需要 `torch`（GPU / PyTorch）；D0 地基不需要

**import 约定**：`src/` / `config/` / `models/` 三个目录均为**平铺**（非包），入口需把三者都加入 path —— 运行脚本用 `PYTHONPATH=src:config:models`；`tests/test_dl_pipeline.py` 已在文件头自行插入这三个目录，直接 `python -m unittest` 即可。

## 运行方式

```bash
# 从项目根目录运行
cd MEOW--predict

# —— DL D0 地基验收单测（torch-free，秒级，Mac CPU 可跑）——
python -m unittest tests.test_dl_pipeline -v

# —— 传统侧实验 ——
# 快速验证（2折 × short profile，约 2 分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

# 日常筛选（默认）：short + long 快车道 + 每日 IC
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite daily

# 提交关口：候选 vs 基线、只跑 expanding（macOS 必须显式 --n-workers 1，见 AGENTS §5.1）
PYTHONPATH=src python experiments/p0_eval_protocol.py \
  --suite gate --candidate-spec-id <候选实验ID> --n-workers 1

# 通用内存看门狗：为任意长任务加 RSS 保护（防休眠用 caffeinate -i）
python experiments/run_with_memory_guard.py \
  --rss-limit-gb 12 --rss-limit-duration-sec 30 --rss-hard-limit-gb 13 \
  --env PYTHONPATH=src \
  -- caffeinate -i python experiments/p0_eval_protocol.py \
    --suite daily --resume --run-id <run_id>

# 正式提交通道（老师入口，正式特征现算，不依赖本地缓存）
cd meow && python meow.py

# 交付链全窗口演练 + 内存峰值核对（≥32GB 机器；Windows 步骤见 docs/交付演练SOP_Windows全窗口.md）
python experiments/run_submission_full_window.py
```

**已锁定的标准口径（勿手改，对照才显式覆写）**：训练标签 winsorize = 开启 + P1/P99；ridge alpha = 2.0；特征 dtype = float32。来源见 AGENTS §7.2。

## 关键文档

| 文档 | 说明 |
|---|---|
| `NOTE.md` | 用户实验笔记：策略讨论、概念、待决问题 |
| `CLAUDE.md` | 当前阶段任务看板、进度、决策 |
| `AGENTS.md` | 开发规范、实验 SOP、禁止事项 |
| `docs/实验记录.md` | 所有实验结果的历史记录 |
| `docs/specs/DL实验设计规格.md` | **DL 主线设计真相源**：脊柱+卡带架构 / 海选+expanding 协议 / 配置管理 / D0 交付物 |
| `docs/交付演练SOP_Windows全窗口.md` | 交付链迁移到 Windows + 全窗口 fit/eval + 内存峰值核对的演练 SOP |
| `docs/specs/高分实验总方案V2.md` | 整体方案设计 |
| `docs/specs/MEOW金融时序预测V3.3_论文启发稳健冲10_AI执行版.md` | V3.3 执行方案 |
| `docs/specs/实验平台架构设计.md` | 并发实验平台架构（trainer/scheduler/resume） |
| `docs/specs/特征管道重构规格.md` | PE1 特征三件套（registry/store/loader）设计与实现规格 |
| `docs/specs/开跑前编码指导_评测口径与提速.md` | 两速评测口径落地 + expanding 提速的编码实施清单 |
| `docs/specs/meow提交通道收口规格.md` | 提交通道（meow.py）正式特征现算与训练/推理复用规格 |
