"""
DL 协议引擎 —— walk-forward 折结构 + 三段切分 + embargo + 4 指标 + 防泄漏检查（脊柱，torch-free）

这一层是 DL 主线"不可变脊柱"里负责**评测口径**的部分（规格 §5）。它只认日期和
numpy 预测，不碰任何 tensor、不 import torch，也刻意**不依赖** ``eval_protocol`` /
``experiment_runner``（那条链拖着 sklearn/lgbm 等重依赖；脊柱要轻、要独立可测）。

提供四块能力：

1. **折结构 + 三段切分**（``DLFold`` / ``build_dl_folds``）：
   每折切成 ``[ train_core | earlystop-val | embargo | scoring-val ]``——
   - ``train_core`` + ``earlystop-val`` 合起来才是"训练区"；earlystop-val 从训练区
     **尾段**切出，归卡带早停用；
   - ``scoring-val`` 归脊柱打分用，与 earlystop-val **物理隔离**；
   - ``embargo``（沿用 1 日）夹在训练区与打分区之间，切断标签前视泄漏。
   折日期生成口径与传统侧 ``eval_protocol.build_folds_for_profile`` 完全对齐
   （sliding / expanding 两模式、cursor + embargo + val_window 同算法）。

2. **4 指标**（``evaluate_predictions`` / ``evaluate_prediction_bundle``）：
   ``corr / MSE / R² / daily-IC``，**逐字对齐** ``experiment_runner`` 的实现
   （同 EPS、同 nan 处理、同 daily-corr 分组口径），保证 DL 与传统侧分数可比。

3. **防泄漏检查**（``assert_folds_causal``）：机械校验每折四段时间严格递增、
   embargo 真的隔开了训练区与打分区——是 D0 的核心验收闸之一。

4. **稳定性汇总**（``summarize_folds``）：pooled 均值 + 最坏折（minimax）双镜头，
   沿用 AGENTS §4.9"不退化成单指标闸刀"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tradingcalendar import Calendar

# 与 experiment_runner.EPS 保持一致：R² 分母兜底，避免 var(y)==0 时除零。
EPS = 1e-8
TARGET_COL = "fret12"


# ================================================================== #
# 折结构：三段切分 + embargo
# ================================================================== #

@dataclass(frozen=True)
class DLFold:
    """
    单折的完整日期切法（四段，全部交易日 int 元组，升序）。

    时间轴（严格递增、互不重叠）：

        train_core ──▶ earlystop ──▶ embargo ──▶ scoring

    - ``train_core_dates`` + ``earlystop_dates`` = 训练区（喂给卡带 fit）
    - ``earlystop_dates``: 训练区尾段，卡带早停看它（代理指标），**不打分**
    - ``embargo_dates``: 禁飞区，隔断标签前视泄漏（可为空 = embargo 0）
    - ``scoring_dates``: 脊柱唯一打分区，predict 前一次不碰
    """

    fold_id: int
    train_core_dates: Tuple[int, ...]
    earlystop_dates: Tuple[int, ...]
    embargo_dates: Tuple[int, ...]
    scoring_dates: Tuple[int, ...]

    @property
    def train_dates(self) -> Tuple[int, ...]:
        """训练区 = train_core + earlystop（喂卡带的全部训练日）。"""
        return self.train_core_dates + self.earlystop_dates

    # —— 便于落 FoldResult / 日志的边界 —— #
    @property
    def train_start(self) -> int:
        return self.train_dates[0]

    @property
    def train_end(self) -> int:
        return self.train_dates[-1]

    @property
    def val_start(self) -> int:
        return self.scoring_dates[0]

    @property
    def val_end(self) -> int:
        return self.scoring_dates[-1]


def split_train_earlystop(
    train_dates: Sequence[int],
    earlystop_frac: float,
    min_earlystop_days: int = 1,
    min_core_days: int = 10,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    把"训练区"按尾段切成 ``(train_core, earlystop)``（公共，trainer 兼容路径也复用，
    保证三段切分逻辑只此一处）。

    - earlystop 天数 = ``max(min_earlystop_days, round(n * frac))``，且至少给 core
      留 ``min_core_days`` 天；core 实在不够则 earlystop 退化为空（让卡带自己用
      固定 epoch 数，不早停）。
    """
    n = len(train_dates)
    if n == 0:
        return tuple(), tuple()
    es = max(min_earlystop_days, int(round(n * earlystop_frac)))
    es = min(es, max(0, n - min_core_days))   # 给 core 保底
    if es <= 0:
        return tuple(train_dates), tuple()
    return tuple(train_dates[:-es]), tuple(train_dates[-es:])


def build_dl_folds(
    rolling_start: int,
    rolling_end: int,
    *,
    mode: str = "expanding",
    val_window: int = 5,
    step: int = 5,
    embargo: int = 1,
    train_window: Optional[int] = None,
    min_train_days: int = 40,
    earlystop_frac: float = 0.15,
    min_earlystop_days: int = 1,
    min_core_days: int = 10,
    max_folds: Optional[int] = None,
    calendar: Optional[Calendar] = None,
) -> List[DLFold]:
    """
    生成 walk-forward 折，每折再切出 earlystop 尾段，得到四段 ``DLFold`` 列表。

    折日期算法与 ``eval_protocol.build_folds_for_profile`` 同口径：
    - ``mode="sliding"``：固定 ``train_window`` 天训练窗，按 ``step`` 滚动；
    - ``mode="expanding"``：训练集从头扩张，最少 ``min_train_days`` 天，按 ``step`` 滚动；
    - 训练区与打分区之间始终隔 ``embargo`` 天。

    海选档可传 ``max_folds=1`` 取单切分；认证档用 expanding 少折（如 3）。
    """
    cal = calendar or Calendar()
    all_dates = cal.range(rolling_start, rolling_end)
    if not all_dates:
        return []
    all_dates = list(all_dates)
    n_all = len(all_dates)
    embargo = max(0, embargo)

    raw_splits: List[Tuple[List[int], List[int]]] = []  # (train_dates, scoring_dates)

    if mode == "sliding":
        if train_window is None:
            raise ValueError("sliding 模式必须给 train_window")
        cursor = train_window + embargo            # cursor = scoring 起点索引
        while cursor + val_window <= n_all:
            train_end_idx = cursor - embargo       # 训练区右边界（不含）
            train_dates = all_dates[max(0, train_end_idx - train_window): train_end_idx]
            scoring_dates = all_dates[cursor: cursor + val_window]
            if len(train_dates) >= min_train_days and scoring_dates:
                raw_splits.append((train_dates, scoring_dates))
            cursor += step

    elif mode == "expanding":
        cursor = min_train_days                    # cursor = 训练区长度（右边界，不含）
        while cursor + embargo + val_window <= n_all:
            train_dates = all_dates[0: cursor]
            scoring_dates = all_dates[cursor + embargo: cursor + embargo + val_window]
            if len(train_dates) >= min_train_days and scoring_dates:
                raw_splits.append((train_dates, scoring_dates))
            cursor += step

    else:
        raise ValueError(f"未知 mode: {mode}，应为 'sliding' / 'expanding'")

    if max_folds is not None:
        raw_splits = raw_splits[:max_folds]

    folds: List[DLFold] = []
    for fid, (train_dates, scoring_dates) in enumerate(raw_splits):
        core, es = split_train_earlystop(train_dates, earlystop_frac, min_earlystop_days, min_core_days)
        # embargo 段 = 训练区末日与打分区首日之间那 embargo 个交易日。
        train_end_idx = all_dates.index(train_dates[-1])
        embargo_dates = tuple(all_dates[train_end_idx + 1: train_end_idx + 1 + embargo]) if embargo > 0 else tuple()
        folds.append(DLFold(
            fold_id=fid,
            train_core_dates=core,
            earlystop_dates=es,
            embargo_dates=embargo_dates,
            scoring_dates=tuple(scoring_dates),
        ))
    return folds


def assert_folds_causal(folds: Sequence[DLFold]) -> None:
    """
    机械校验每折四段时间严格递增、互不重叠、embargo 真隔开训练区与打分区。

    任一折违反即抛 ``AssertionError``——这是防泄漏的"折结构"闸（单测会调它）。
    """
    for f in folds:
        segs = [
            ("train_core", f.train_core_dates),
            ("earlystop", f.earlystop_dates),
            ("embargo", f.embargo_dates),
            ("scoring", f.scoring_dates),
        ]
        present = [(name, s) for name, s in segs if len(s) > 0]
        # 段内升序
        for name, s in present:
            assert list(s) == sorted(s), f"fold {f.fold_id} 段 {name} 未升序"
        # 段间严格递增：前段最大 < 后段最小
        for (n1, s1), (n2, s2) in zip(present, present[1:]):
            assert s1[-1] < s2[0], (
                f"fold {f.fold_id}: 段 {n1}(末={s1[-1]}) 未严格早于段 {n2}(首={s2[0]})"
            )
        # 训练区/打分区无交集
        assert not (set(f.train_dates) & set(f.scoring_dates)), (
            f"fold {f.fold_id}: 训练区与打分区有交集（泄漏）"
        )
        # earlystop 与 scoring 隔离
        assert not (set(f.earlystop_dates) & set(f.scoring_dates)), (
            f"fold {f.fold_id}: earlystop 与 scoring 有交集"
        )


# ================================================================== #
# 4 指标（逐字对齐 experiment_runner，torch-free）
# ================================================================== #

def evaluate_predictions(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    corr / MSE / R²，口径与 ``experiment_runner.evaluate_predictions`` 完全一致：

    - pred 先 ``nan_to_num``（NaN/±inf → 0），与老师评测对 NaN 的处理同源；
    - ``mse`` = 均方误差；``corr`` = Pearson（样本<2 记 0）；
    - ``r2 = 1 - Σ(p-y)² / (var(y)·n + EPS)``。
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)
    mse = float(np.mean((p - y) ** 2)) if n > 0 else 0.0
    # 恒零/常量预测时 corrcoef 会除零告警并产 nan——抑制告警，下方 isfinite 兜底为 0。
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.corrcoef(p, y)[0, 1]) if n > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    r2 = float(1.0 - np.sum((p - y) ** 2) / (np.var(y) * n + EPS)) if n > 0 else 0.0
    return {"mse": mse, "corr": corr, "r2": r2}


def evaluate_prediction_bundle(label_df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    """
    在 ``evaluate_predictions`` 基础上加 daily-IC（按 ``date`` 分组的逐日 corr 均值/方差），
    口径对齐 ``experiment_runner.evaluate_prediction_bundle``。

    ``label_df`` 需含 ``date`` 与 ``fret12`` 列，行序与 ``pred`` 严格对齐
    （即 ``SequenceDataset.label_frame()`` 的输出）。
    """
    y = label_df[TARGET_COL].to_numpy()
    metrics = evaluate_predictions(y, pred)

    tmp = label_df[["date", TARGET_COL]].copy()
    tmp["pred"] = np.asarray(pred, dtype=np.float32)
    daily_corrs: List[float] = []
    for _, group in tmp.groupby("date", sort=True):
        if group.shape[0] < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(group["pred"].to_numpy(), group[TARGET_COL].to_numpy())[0, 1]
        if np.isfinite(c):
            daily_corrs.append(float(c))
    metrics["daily_corr_mean"] = float(np.mean(daily_corrs)) if daily_corrs else 0.0
    metrics["daily_corr_std"] = float(np.std(daily_corrs)) if daily_corrs else 0.0
    metrics["n_days"] = int(tmp["date"].nunique())
    return metrics


def corr_gap(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> float:
    """train_corr - val_corr：过拟合的粗判（与逐 epoch 曲线配合，见规格 §5.4）。"""
    return float(train_metrics["corr"] - val_metrics["corr"])


# ================================================================== #
# 稳定性汇总：pooled + minimax 双镜头
# ================================================================== #

def summarize_folds(fold_metrics: Sequence[Dict[str, float]], key: str = "corr") -> Dict[str, float]:
    """
    把多折打分汇总成"均值（pooled）+ 鲁棒（最坏折）"双镜头（AGENTS §4.9）。

    返回 ``mean / std / min(最坏折) / max / n_folds / positive_rate``。判官不退化成
    单指标闸刀——均值看整体水平、最坏折看是否撞运气，人工权衡。
    """
    vals = np.array([m[key] for m in fold_metrics if np.isfinite(m.get(key, np.nan))], dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_folds": 0, "positive_rate": 0.0}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),       # 最坏折 = minimax 镜头
        "max": float(vals.max()),
        "n_folds": int(vals.size),
        "positive_rate": float(np.mean(vals > 0)),
    }
