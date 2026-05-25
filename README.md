# MEOW 金融时序预测

分钟级 A 股订单簿数据的股票收益预测研究项目。目标：预测未来 12 分钟收益 `fret12`。

## 项目定位

- 数据：144 个交易日（2023-06-01 ~ 2023-12-29），每日一个 `.h5` 文件，约 2.45 GB
- 任务：截面预测，评价指标为 `rolling_corr_mean`（IC 均值）和 `stability_score`
- 当前 rolling 结果与基线口径：见 `CLAUDE.md` 和 `docs/实验记录.md`

## 目录结构

```
MEOW--predict/
├── src/                     # 核心代码
│   ├── dl.py                # 数据加载
│   ├── feat.py              # teacher 原始 6 特征（legacy）
│   ├── feat_engine.py       # 增强特征工程（FeatureBuilder）
│   ├── eval_protocol.py     # 三层评测协议（Rolling Profiles / Leaderboard）
│   ├── trainer.py           # BaseTrainer ABC + TabularTrainer（DL 扩展点）
│   ├── scheduler.py         # 并发调度（ProcessPoolExecutor + resume）
│   ├── experiment_runner.py # 实验编排（特征/模型/评估核心逻辑）
│   ├── mdl.py               # 模型接口
│   ├── eval.py              # 评估工具
│   ├── tradingcalendar.py   # 交易日历
│   └── log.py               # 日志
├── experiments/             # 实验入口脚本
│   ├── p0_eval_protocol.py  # P0：Rolling 评测基准（主入口）
│   ├── run_with_memory_guard.py # 通用内存看门狗包装器（可复用）
│   ├── p1_ofi_validation.py # P1：OFI 特征验证
│   ├── p2_impact_validation.py
│   ├── p3_momentum_validation.py
│   └── legacy/              # 历史实验脚本（保留可运行）
├── data/                    # 原始 .h5 数据（gitignored）
├── results/                 # 实验结果 CSV（gitignored）
└── docs/                    # 文档
    ├── 实验记录.md           # 所有实验记录的唯一入口
    ├── specs/               # 规格文档
    └── archived/            # 历史快照（只读）
```

## 运行环境

- Python 3.13（mise 全局环境）
- 依赖：`pip install --system numpy pandas scikit-learn tables`

## 运行方式

```bash
# 从项目根目录运行（确保 src/ 在路径里）
cd MEOW--predict

# 快速验证（2折 × short profile，约 2 分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

# Ridge 基线建立（全量，推荐 4 并发）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge --n-workers 4

# 通用内存看门狗：为任意长任务加 RSS 保护（示例：12 GB 阈值续跑 P0）
python experiments/run_with_memory_guard.py \
  --rss-limit-gb 12 \
  --rss-limit-duration-sec 30 \
  --rss-hard-limit-gb 13 \
  --env PYTHONPATH=src \
  -- python experiments/p0_eval_protocol.py \
    --suite ridge \
    --n-workers 4 \
    --resume \
    --run-id 20260523_223257

# 旧实验脚本（仍可运行）
PYTHONPATH=src python experiments/legacy/run_516v3_restricted.py
```

## 关键文档

| 文档 | 说明 |
|---|---|
| `NOTE.md` | 用户实验笔记：策略讨论、概念、待决问题 |
| `CLAUDE.md` | 当前阶段任务看板、进度、决策 |
| `AGENTS.md` | 开发规范、实验 SOP、禁止事项 |
| `docs/实验记录.md` | 所有实验结果的历史记录 |
| `docs/P0运行耗时监控报告_20260525.md` | 本次 P0 `expanding_40d_5d` 运行的耗时监控与阶段分析报告 |
| `docs/specs/高分实验总方案V2.md` | 整体方案设计 |
| `docs/specs/MEOW金融时序预测V3.3_论文启发稳健冲10_AI执行版.md` | V3.3 执行方案 |
| `docs/specs/实验平台架构设计.md` | 并发实验平台架构（trainer/scheduler/resume） |
| `docs/specs/开跑前编码指导_评测口径与提速.md` | 两速评测口径落地 + expanding 提速的编码实施清单（面向 coding agent） |
| `experiments/run_with_memory_guard.py` | 通用内存看门狗包装器，超过 RSS 阈值自动终止任务 |
