# MEOW 金融时序预测

分钟级 A 股订单簿数据的股票收益预测研究项目。目标：预测未来 12 分钟收益 `fret12`。

## 项目定位

- 数据：144 个交易日（2023-06-01 ~ 2023-12-29），每日一个 `.h5` 文件，约 2.45 GB
- 任务：截面预测，评价指标为 `rolling_corr_mean`（IC 均值）和 `stability_score`
- 当前最优基线：`R02_ridge_legacy_plus_norm_core`，`rolling_corr_mean = 0.047976`（5折 rolling）

## 目录结构

```
MEOW--predict/
├── src/                     # 核心代码
│   ├── dl.py                # 数据加载
│   ├── feat.py              # teacher 原始 6 特征（legacy）
│   ├── feat_engine.py       # 增强特征工程（FeatureBuilder）
│   ├── mdl.py               # 模型接口
│   ├── eval.py              # 评估工具
│   ├── experiment_runner.py # 实验编排
│   ├── tradingcalendar.py   # 交易日历
│   └── log.py               # 日志
├── experiments/             # 实验入口脚本
│   ├── p0_rolling_audit.py  # P0：扩展 rolling 评测
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
PYTHONPATH=src python experiments/p0_rolling_audit.py

# 旧实验脚本（仍可运行）
python experiments/legacy/run_516v3_restricted.py
```

## 关键文档

| 文档 | 说明 |
|---|---|
| `CLAUDE.md` | 当前阶段任务看板、进度、决策 |
| `AGENTS.md` | 开发规范、实验 SOP、禁止事项 |
| `docs/实验记录.md` | 所有实验结果的历史记录 |
| `docs/specs/高分实验总方案V2.md` | 整体方案设计 |
| `docs/specs/MEOW金融时序预测V3.3_论文启发稳健冲10_AI执行版.md` | V3.3 执行方案 |
