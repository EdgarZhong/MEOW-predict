# CLAUDE.md — 当前阶段进度与任务看板

更新日期：2026-05-25

## 当前阶段目标

**方案 B（扩展 rolling 选模型）+ 两速评测结构已固定。文档全部落定。**

**当前唯一主线：开跑前债务清零。** 用户硬约束（2026-05-25）：正式开 P1 跑实验前，不允许残留任何工程/口径债务。所有 specs 工程项（`docs/specs/开跑前编码指导_评测口径与提速.md`）+ PE1 收尾 + P0.5 锁超参 + 提交通道，必须全部清完并跑冒烟验证，才能开 P1。冲刺看板见下方「开跑前债务清零冲刺」，A→E 档按序推进。

## 当前口径收敛（指针表）

CLAUDE 不复述规则，只给指针；规则正文在 AGENTS，"为什么"在 NOTE。

| 口径 | 规则位置 |
|---|---|
| 两速评测结构（快车道 short+long+每日IC / 慢车道 expanding+Review） | AGENTS §4.2 |
| 各阶段评测口径（P1–P5） | AGENTS §4.4 |
| 多重比较 / 边界增益（+0.003 是地板不是采纳标准） | AGENTS §4.3 |
| make_decision 硬契约 + 单测 | AGENTS §4.6 |
| 单候选 per-profile 复核六步 | AGENTS §4.7 |
| 组合不可加（叠加必当新 spec 重跑） | AGENTS §4.5 |
| Target Mode 准入（最终回 raw `fret12`） | AGENTS §7.5 |
| alpha + winsorize 先扫后锁 | AGENTS §7.7 / §7.11 |
| 推理契约与交付约束 | AGENTS §十一 |

## 当前最优基线（P0 分段结果）

> 这次 P0 结果分两个 run 目录保存：`20260523_223257` 负责 short / medium / long，`20260524_220846` 负责串行补完 expanding。按范围分开记录，避免把不同 scope 混成一个单点。

| 范围 | experiment_id | protocol_corr_mean | protocol_stability_score | protocol_corr_min | decision |
|---|---|---:|---:|---:|---|
| short / medium / long aggregate | R02_ridge_legacy_plus_norm_core | 0.057463 | 0.041396 | 0.025214 | baseline |
| expanding only | R01_ridge_legacy_plus_core | 0.066095 | 0.053459 | 0.041181 | promote |

来源：`results/eval_protocol/20260523_223257`（short/medium/long）、`results/eval_protocol/20260524_220846`（expanding）。expanding 串行耗时分析见 `docs/P0运行耗时监控报告_20260525.md`。

## 任务看板

### ⭐ 开跑前债务清零冲刺【当前唯一焦点】

> **接手须知（无会话上下文也能实施）**：实施总规格 = `docs/specs/开跑前编码指导_评测口径与提速.md`（下称「编码指导」）+ `AGENTS.md`。每项已标：实施指针 / 关键约束 / 验收 / 依赖 / 状态。按 A→E 推进。依赖：#17 须等 #16 通过、#18 须等 #15 就绪。commit 纪律：一条一提交（编码指导总原则 3）。
>
> **接手第一步**：① 读 `AGENTS.md` + 本看板；② 验现状健康：`PYTHONPATH=src python tests/test_eval_protocol.py`（11 条应全过）、`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick`（应 exit 0，run 目录含 leaderboard/manifest_snapshot/resolved_columns/config）；③ **不要先做 #16**，先做下方新增的 **S1「提交链收口」**，把 `meow/` 纳入主仓库并收成正式提交入口包装层。当前工作区除 `meow/` 外，还包含 **#14、#15 已完成但未提交** 的改动（见下）。

**A 档 评测口径代码化 —— ✅ 已完成（commit `2ebb5fb`）**
- ✅ #10 make_decision 硬契约 + 单测（编码指导§4 / AGENTS §4.6）
- ✅ #11 daily suite（short+long）并设默认（编码指导§1）
- ✅ #12 每日 IC 均值 + IC-IR 进 leaderboard（编码指导§5）
- ✅ #13 manifest_snapshot + resolved_columns.json 写盘
- 验证：quick suite 端到端冒烟通过 + 8 条单测（`tests/test_eval_protocol.py`）全过

**B 档 PE1 清理与归档 —— #14【✅ 已完成，待提交】**
- **完成时间**：2026-05-25 本轮接手已落地。
- **归档位置**：`.archive/20260525_b14_archive/`
- **已归档内容**：
  - `src/feat_engine.py`（FeatureBuilder 本体）
  - `experiments/legacy/run_516v3_norm_only.py`
  - `experiments/legacy/run_516v3_restricted.py`
- **代码清理结果**：
  - `src/experiment_runner.py` 已移除旧加载链：`load_split` / `load_raw_split` / `load_feature_split` / `_filter_features` / `self.builder` / `_split_cache` / `_raw_split_cache` / `_daily_feature_cache`
  - `_load_group_split` 保留为唯一正式入口；缺少 loader 时改为显式 `raise`
  - 兼容层仅保留 `self.loader` 属性，供协议层读取 `h5dir`，不再承载旧特征构建逻辑
- **验收结果**：
  - `PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick` 已在每次逐文件归档后通过，最近 run_id：`20260525_142943`
  - `PYTHONPATH=src python tests/test_eval_protocol.py` 8/8 通过
  - `src/` 内已无 `feat_engine` import；`experiments/legacy/` 仅剩 `p0_rolling_audit.py`
- **下一步**：进入 #15（winsorize 开关与分位配置）

**C 档 winsorize —— #15【✅ 已完成，待提交】**
- **完成时间**：2026-05-25 本轮接手已落地。
- **代码结果**：
  - `src/experiment_runner.py` 已接入 runner 级训练标签 winsorize 配置；默认 `enabled=True, q=(0.005, 0.995)`，且只作用于训练目标
  - `src/scheduler.py` / `src/eval_protocol.py` / `experiments/p0_eval_protocol.py` 已把该配置贯通到并行 worker 和结果 `config.json`
  - 新增单测 `tests/test_experiment_runner.py`，锁默认值 / 关闭开关 / 分位裁剪行为
- **真实数据闭环**：
  - 开启版：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_b15_quick_on_v2`
  - 关闭版：`PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick --run-id 20260525_b15_quick_off_v1 --target-winsorize off`
  - 两轮都已成功跑通；开启版 worker 日志可见实际裁剪边界，关闭版日志明确 `Target winsorize: disabled`
- **当前观察（仅 quick 冒烟，不作采纳结论）**：
  - `R02` quick `protocol_corr_mean`：开启 0.032685，关闭 0.032791
  - 说明开关真实生效，但是否采用默认 `P0.5/P99.5` 仍留待 #18 在 short+medium 正式扫锁
- **下一步**：进入 #16（float32 验收）

**S 档 提交链收口 —— S1【🔴 立刻执行，优先级高于 #16】**
- **为什么现在插队**：
  - 老师最终口令已明确为 `python meow.py`；当前实验 driver 与 `meow/` 入口不是同一条真实执行链
  - 若先继续做 `#16/#17/#18`，存在“优化的是实验系统，不是老师最终会跑的系统”的风险
- **任务目标**：把 `meow/` 收成老师正式提交入口包装层，同时保持 `src/` 为唯一核心后端；分叉仅允许发生在 driver 编排与特征物化方式。
- **实现规格**：详见 `docs/specs/meow提交通道收口规格.md`
- **验收**：
  - `meow/` 文件已被主仓库跟踪
  - `cd meow && python meow.py` 在真实数据上可直接训练 + 预测 + 评测
  - 提交链和实验链在同一切分上的特征列/训练口径/预测输出一致
  - `winsorize` 在 `meow.py` 训练链中真实生效，而不是只存在于实验 driver
- **预估工作量**：
  - 设计澄清 + 仓库归并：0.5 小时
  - 薄包装与桥接实现：2~4 小时
  - `meow.py` 端到端闭环与修 bug：1~3 小时
  - **合计：半天到 1 天；若出现 train/serve skew 或缓存耦合问题，按 1 天看**

**D 档 expanding 提速**
- #16 特征 float32（编码指导§2b；改 `src/feature_loader.py`）。**验收（硬）**：同 spec 同 fold，float32 vs float64 `protocol_corr` 差 |Δcorr|<1e-4，不达标不合入。**注意：S1 完成前暂停。**
- #17 关口提速（编码指导§2a/§2c；改 `src/scheduler.py` + `experiments/p0_eval_protocol.py`）：① 关口只评候选+基线 2 spec ② group 按 fold 序号成本均衡切分（不连续切）③ heavy 批次 2 worker。**硬前提：#16 验收通过才开 2 worker，否则维持串行**。验收：2 spec / 2 worker 无 OOM、~30min。依赖 #16。**注意：S1 完成前暂停。**

**E 档 跑实验收尾**
- #18 P0.5 alpha+winsorize 扫锁（AGENTS §7.7）：仅 R02、仅 short+medium；alpha {0.5,1,2,5,10,20} × clip {P0.5/P99.5, P1/P99, 不裁} 一起扫，各取平台中心锁死，固化进标准 ridge 路径；per-fold alpha 留 P4。依赖 #15。~15–20min。
- #19 OOM 冒烟 + 全量构建计时：daily suite 跑一轮无 OOM/无 worker 异常；`python -m feature_store build --all` ≤20min。可叠 `run_with_memory_guard.py` + `caffeinate`。
- #20 meow.py 提交通道（AGENTS §十一）：在一个 held-out 交易日跑通 `genFeatures→predict→forecast`，列对齐、每行有限、输出回 raw `fret12`、不依赖本地缓存/不可见统计量。**该项并入 S1 实施与验收，不再独立后置。** P1 前强制。

> 下方 PE0/PE1/P0.5 等节仅留背景；待办均已并入本冲刺看板。

### PE0 实验平台并发改造【已完成】

- 并发调度 + resume + 串并一致性 + OOM 修复均已合入。
- OOM 根因（备查）：`ExperimentRunner._split_cache` 按 fold 唯一 key 累积全量 DataFrame 不释放，expanding 末段单 entry ≈ 4GB，多 worker 总峰值 >50GB。修复：每折后 `clear()` + `gc.collect()`，n_workers 默认 4。

### PE1 特征管道重构【实施中】

规格：`docs/specs/特征管道重构规格.md`；交接：`docs/交接文档_特征管道重构_20260523.md`

- 当前状态：FeatureRegistry / FeatureStore / FeatureLoader 三件套首版 + 单测均通过（M1–M4）；主链路（trainer / scheduler / eval_protocol）已切到 FeatureLoader；9 个 stage 合并 462 列与旧管道一致。
- 调度与内存（已落地）：scheduler 改 light/heavy 两批（light 沿用 4 worker，heavy 降到 2）；`experiments/run_with_memory_guard.py` RSS 看门狗（软 12GB/30s + 硬 13GB 击杀）。16GB Mac 上 heavy 批次并发 ≤ 2。

剩余项（已并入上方冲刺看板，此处不重复列）：
- #13 manifest/resolved_columns 写盘 → ✅ 已完成
- #14 清理旧 cache/load 链 + `feat_engine` 归档（含 legacy 脚本，本会话决策）→ ✅ 已完成，待提交
- #15 训练目标 winsorize 开关 + 分位配置 + 真实数据闭环 → ✅ 已完成，待提交
- #19 OOM 冒烟 + 全量构建 ≤20min
- 「新旧管道 allclose 验证」**取消**：legacy 旧管道按 #14 归档后已无对照对象；新管道已由 P0 基线产出验证可用，无需再比。

### P0 扩展 rolling 评测体系【工程完成，基线已产出】

- 三层协议 + 4 profiles + 输出结构（fold_manifest / fold_metrics / profile_summary / leaderboard）+ baseline delta + make_decision + 主入口均完成。
- `daily` / `ridge` / `full` suite 可跑（基线见上"当前最优基线"）。

> P0.5（#18）、两速 suite 与 expanding 提速（#10 / #11 / #12 ✅ + #15 / #16 / #17）、提交通道（#20）的待办均已并入上方「开跑前债务清零冲刺」，此处不再重复，避免文档矛盾。

### P1 / P2 / P3 特征组验证【等上面就绪】

- [ ] P1：O1–O6（OFI）｜P2：T1–T4（trade impact）｜P3：C1–C3（条件动量/反转）
- 口径：快车道 short+long+每日IC 筛 → 候选过线才进 expanding 关口（AGENTS §4.4）；采纳门槛见 §4.6

### P3.5 少量交互项冲刺【P1–P3 有初步结论后】

- [ ] 仅高先验交互（`ofi_total×spread` 等 6 项，AGENTS §7.10）；scratch 起步，复用才进 registry

### P4 稳健模型比较【P3.5 后】

- [ ] Ridge / ElasticNet / Huber / 浅 ExtraTrees / 浅 HistGB；准入见 AGENTS §7.8
- [ ] expanding 只在 2–3 个决赛模型上跑（树更贵且增量技巧失效）

### P5 受限融合【最后】

- [ ] 只融合独立有效、与主分支非同质的分支；二层用 OOF；不比 backbone 稳就交 backbone（AGENTS §7.9）

### 交付演练【P5 之后、看 Final Holdout 之前】

- [ ] 输出对齐 raw `fret12`；`meow.py` 不依赖本地缓存 / 不可见统计量；提交口径与训练评测一致

## 工程状态

- [x] 目录与文档重组、代码解耦、PE0 OOM 修复、PE1 设计拍板
- [ ] PE1 实施收尾
- [ ] P0.5 + 两速 suite 落地 + expanding 提速 + 提交通道打通
- [ ] R02 一致性复现（run1 / run2 / run3）
