"""
评测协议模块 - Rolling Evaluation Protocol

实现三层评测体系：
  第一层：Dev Rolling    - 模型开发、特征筛选的主要依据
  第二层：Review Holdout - 候选模型复核（11月）
  第三层：Final Holdout  - 最终提交前模拟（12月，尽量少碰）

提供四个 rolling profile 横向对比，输出统一可复现 leaderboard。

使用方式：
  from eval_protocol import EvaluationProtocolRunner, ROLLING_PROFILES, ALL_SPECS, BASELINE_ID
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from experiment_runner import ExperimentRunner, SplitConfig, RollingFold

try:
    from scheduler import ParallelScheduler
except ImportError:
    ParallelScheduler = None


# ================================================================== #
# Rolling Profile 配置
# ================================================================== #

@dataclass
class RollingProfile:
    """Rolling 评测 profile 配置"""
    profile_name: str       # 唯一名称
    val_window: int         # 验证窗口（交易日）
    step: int               # 滚动步长（交易日）
    embargo: int            # 禁飞区（交易日）
    mode: str               # "sliding" 或 "expanding"
    train_window: Optional[int] = None  # 固定训练窗口（sliding 模式）
    min_train_days: int = 10            # 最小训练天数


# 四个标准 profile
ROLLING_PROFILES: List[RollingProfile] = [
    RollingProfile(
        profile_name="short_8d_2d",
        train_window=8,
        min_train_days=8,
        val_window=2,
        step=5,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="medium_20d_5d",
        train_window=20,
        min_train_days=20,
        val_window=5,
        step=5,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="long_40d_5d",
        train_window=40,
        min_train_days=40,
        val_window=5,
        step=10,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="expanding_40d_5d",
        train_window=None,
        min_train_days=40,
        val_window=5,
        step=5,
        embargo=1,
        mode="expanding",
    ),
]

# protocol_stability_score 的加权权重
PROFILE_WEIGHTS: Dict[str, float] = {
    "short_8d_2d": 0.25,
    "medium_20d_5d": 0.35,
    "long_40d_5d": 0.25,
    "expanding_40d_5d": 0.15,
}


# ================================================================== #
# Fold Manifest（带 embargo 信息）
# ================================================================== #

@dataclass
class FoldManifestEntry:
    """单个 fold 的完整日期切法，含 embargo 区间"""
    profile_name: str
    fold_id: int
    train_start: int
    train_end: int
    embargo_start: int  # embargo 起（如 embargo=0 则等于 train_end）
    embargo_end: int    # embargo 末
    val_start: int
    val_end: int
    n_train_days: int
    n_val_days: int


# ================================================================== #
# Baseline 与历史实验 Specs
# ================================================================== #

BASELINE_ID = "R02_ridge_legacy_plus_norm_core"

BASELINE_SPEC = {
    "experiment_id": BASELINE_ID,
    "type": "standard",
    "model": "ridge",
    "target_mode": "raw",
    "groups": ["legacy", "norm_core"],
    "notes": "current stable ridge baseline",
}

# 全部历史实验（含 baseline）
ALL_SPECS: List[Dict] = [
    # R 系列：Ridge backbone 变体（最优基线对比）
    {"experiment_id": "R00_ridge_legacy", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy"], "notes": "ridge legacy only"},
    {"experiment_id": "R01_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "ridge legacy plus core features"},
    {"experiment_id": "R02_ridge_legacy_plus_norm_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core"], "notes": "current stable ridge baseline"},
    {"experiment_id": "R03_ridge_legacy_plus_patch_summary", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "patch_summary"], "notes": "ridge legacy plus patch summary"},
    {"experiment_id": "R04_ridge_legacy_plus_cross_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "cross_rank_features"], "notes": "ridge legacy plus cross-sectional ranks"},
    # B 系列：结构性 backbone
    {"experiment_id": "B6_common_residual", "type": "common_residual", "notes": "formal common residual branch"},
    {"experiment_id": "B7_soft_regime", "type": "soft_regime", "notes": "formal soft regime ensemble"},
    {"experiment_id": "B8_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "formal ridge legacy plus core"},
    # O 系列：OFI 动态订单流
    {"experiment_id": "O1_R02_plus_ofi_raw", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw"], "notes": "R02 plus raw OFI"},
    {"experiment_id": "O2_R02_plus_ofi_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_dynamic"], "notes": "R02 plus dynamic OFI"},
    {"experiment_id": "O3_R02_plus_ofi_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_rank", "ofi_raw"], "notes": "R02 plus OFI cross ranks"},
    {"experiment_id": "O4_R02_plus_ofi_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe"], "notes": "R02 plus all safe OFI"},
    {"experiment_id": "O5_R02_plus_ofi_raw_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw", "ofi_dynamic"], "notes": "R02 plus raw and dynamic OFI"},
    {"experiment_id": "O6_R02_plus_all_ofi", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe", "ofi_rank"], "notes": "R02 plus all OFI groups"},
    # T 系列：成交冲击
    {"experiment_id": "T1_R02_plus_trade_impact", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact"], "notes": "R02 plus trade impact base"},
    {"experiment_id": "T2_R02_plus_trade_impact_dyn", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_dyn"], "notes": "R02 plus trade impact dynamic"},
    {"experiment_id": "T3_R02_plus_trade_impact_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_interaction"], "notes": "R02 plus trade impact interactions"},
    {"experiment_id": "T4_R02_plus_trade_impact_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_safe"], "notes": "R02 plus all safe trade impact"},
    # C 系列：条件动量/反转
    {"experiment_id": "C1_R02_plus_conditional_momentum", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum"], "notes": "R02 plus conditional momentum base"},
    {"experiment_id": "C2_R02_plus_conditional_momentum_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum_interaction"], "notes": "R02 plus conditional momentum interactions"},
    {"experiment_id": "C3_R02_plus_conditional_momentum_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum_safe"], "notes": "R02 plus all safe conditional momentum"},
]

# Ridge baseline 子集（快速复现用）
RIDGE_SPECS: List[Dict] = [s for s in ALL_SPECS if s["experiment_id"].startswith("R")]


# ================================================================== #
# 工具函数
# ================================================================== #

def _weighted_avg(profile_scores: Dict[str, Dict], key: str) -> float:
    """按 PROFILE_WEIGHTS 对各 profile 的指定指标加权平均"""
    total_w = 0.0
    total_v = 0.0
    for pname, scores in profile_scores.items():
        w = PROFILE_WEIGHTS.get(pname, 0.0)
        v = scores.get(key, np.nan)
        if not np.isnan(float(v)) and w > 0:
            total_v += w * float(v)
            total_w += w
    return float(total_v / total_w) if total_w > 0 else np.nan


def make_decision(row: Dict, baseline: Dict) -> Tuple[str, str]:
    """
    判断实验是否值得进入下一阶段。

    返回 (decision, reason)：
      promote  - 稳定超过基线，建议进入下阶段
      review   - corr 有提升但 MSE 恶化，需人工检查
      reject   - 不建议继续
    """
    def _safe(d, k):
        v = d.get(k, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    base_corr = _safe(baseline, "protocol_corr_mean")
    base_stability = _safe(baseline, "protocol_stability_score")
    base_mse = _safe(baseline, "protocol_mse_mean")
    base_r2 = _safe(baseline, "protocol_r2_mean")

    corr = _safe(row, "protocol_corr_mean")
    stability = _safe(row, "protocol_stability_score")
    pfr = _safe(row, "protocol_positive_fold_rate")
    mse = _safe(row, "protocol_mse_mean")
    r2 = _safe(row, "protocol_r2_mean")

    if np.isnan(corr) or np.isnan(base_corr):
        return "unknown", "缺少 protocol_corr_mean"

    if corr < base_corr + 0.003:
        return "reject", f"corr 提升不足（{corr:.4f} vs 基线 {base_corr:.4f}+0.003）"

    if not np.isnan(stability) and not np.isnan(base_stability) and stability < base_stability:
        return "reject", f"stability_score 劣于基线（{stability:.4f} < {base_stability:.4f}）"

    if not np.isnan(pfr) and pfr < 0.8:
        return "reject", f"负 fold 过多（positive_fold_rate={pfr:.2f} < 0.8）"

    if not np.isnan(mse) and not np.isnan(base_mse) and abs(base_mse) > 1e-12:
        mse_delta_pct = (mse - base_mse) / abs(base_mse)
        if mse_delta_pct > 0.05:
            return "review", f"corr 提升但 MSE 恶化 {mse_delta_pct:.1%}（>5%）"

    delta_corr = corr - base_corr
    delta_stab = (stability - base_stability) if not (np.isnan(stability) or np.isnan(base_stability)) else float("nan")
    stab_str = f"+{delta_stab:.4f}" if not np.isnan(delta_stab) else "n/a"
    return "promote", f"稳定超基线（corr +{delta_corr:.4f}，stability {stab_str}）"


# ================================================================== #
# EvaluationProtocolRunner
# ================================================================== #

class EvaluationProtocolRunner:
    """
    滚动评测协议执行器。

    封装 ExperimentRunner，提供多 profile 横向对比、holdout 评测和 leaderboard 生成。
    """

    def __init__(self, experiment_runner: ExperimentRunner):
        self.runner = experiment_runner

    # ---------------------------------------------------------------- #
    # Fold 构造
    # ---------------------------------------------------------------- #

    def build_folds_for_profile(
        self,
        profile: RollingProfile,
        rolling_start: int,
        rolling_end: int,
        max_folds: Optional[int] = None,
    ) -> Tuple[List[RollingFold], List[FoldManifestEntry]]:
        """
        为指定 profile 构建 rolling folds 列表和 fold manifest。

        严格保证：train_end < embargo_start <= embargo_end < val_start
        """
        all_dates = self.runner.calendar.range(rolling_start, rolling_end)
        if not all_dates:
            return [], []

        embargo = max(0, profile.embargo)
        folds: List[RollingFold] = []
        manifest: List[FoldManifestEntry] = []
        fold_id = 0

        if profile.mode == "sliding":
            assert profile.train_window is not None, "sliding 模式必须设置 train_window"
            # cursor 指向 val_start 的索引
            cursor = profile.train_window + embargo
            while cursor + profile.val_window <= len(all_dates):
                train_end_idx = cursor - embargo        # train_dates 右边界（不含）
                train_dates = all_dates[max(0, train_end_idx - profile.train_window):train_end_idx]
                embargo_dates = all_dates[train_end_idx:train_end_idx + embargo] if embargo > 0 else []
                val_dates = all_dates[cursor:cursor + profile.val_window]

                if len(train_dates) >= profile.min_train_days and len(val_dates) > 0:
                    folds.append(RollingFold(fold_id=fold_id, train_dates=tuple(train_dates), val_dates=tuple(val_dates)))
                    manifest.append(FoldManifestEntry(
                        profile_name=profile.profile_name,
                        fold_id=fold_id,
                        train_start=train_dates[0],
                        train_end=train_dates[-1],
                        embargo_start=embargo_dates[0] if embargo_dates else train_dates[-1],
                        embargo_end=embargo_dates[-1] if embargo_dates else train_dates[-1],
                        val_start=val_dates[0],
                        val_end=val_dates[-1],
                        n_train_days=len(train_dates),
                        n_val_days=len(val_dates),
                    ))
                    fold_id += 1

                cursor += profile.step

        elif profile.mode == "expanding":
            # expanding：训练集从头扩张，最少 min_train_days 天
            cursor = profile.min_train_days   # cursor = 训练集长度（右边界，不含）
            start_idx = 0
            while cursor + embargo + profile.val_window <= len(all_dates):
                train_dates = all_dates[start_idx:cursor]
                embargo_dates = all_dates[cursor:cursor + embargo] if embargo > 0 else []
                val_dates = all_dates[cursor + embargo:cursor + embargo + profile.val_window]

                if len(train_dates) >= profile.min_train_days and len(val_dates) > 0:
                    folds.append(RollingFold(fold_id=fold_id, train_dates=tuple(train_dates), val_dates=tuple(val_dates)))
                    manifest.append(FoldManifestEntry(
                        profile_name=profile.profile_name,
                        fold_id=fold_id,
                        train_start=train_dates[0],
                        train_end=train_dates[-1],
                        embargo_start=embargo_dates[0] if embargo_dates else train_dates[-1],
                        embargo_end=embargo_dates[-1] if embargo_dates else train_dates[-1],
                        val_start=val_dates[0],
                        val_end=val_dates[-1],
                        n_train_days=len(train_dates),
                        n_val_days=len(val_dates),
                    ))
                    fold_id += 1

                cursor += profile.step

        else:
            raise ValueError(f"未知 profile.mode: {profile.mode}，应为 'sliding' 或 'expanding'")

        if max_folds is not None:
            folds = folds[:max_folds]
            manifest = manifest[:max_folds]

        return folds, manifest

    # ---------------------------------------------------------------- #
    # 单 profile 运行
    # ---------------------------------------------------------------- #

    def run_profile(
        self,
        profile: RollingProfile,
        rolling_start: int,
        rolling_end: int,
        specs: List[Dict],
        max_folds: Optional[int] = None,
    ) -> Tuple[List[FoldManifestEntry], pd.DataFrame]:
        """
        在指定 profile 下运行所有 specs，返回 (manifest, fold_metrics_df)。

        fold_metrics_df 含 profile_name / fold 日期 / 各指标列。
        """
        folds, manifest = self.build_folds_for_profile(profile, rolling_start, rolling_end, max_folds=max_folds)
        if not folds:
            return [], pd.DataFrame()

        rows = []
        for fold in folds:
            fold_split = SplitConfig(
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[-1],
                val_start=fold.val_dates[0],
                val_end=fold.val_dates[-1],
                test_start=fold.val_dates[0],
                test_end=fold.val_dates[-1],
            )
            for spec in specs:
                try:
                    bundle = self.runner._evaluate_spec_on_fold(fold_split, spec)
                    row = self.runner._fold_metric_row(
                        fold_id=fold.fold_id,
                        experiment_id=spec["experiment_id"],
                        feature_set=bundle["feature_set"],
                        target_type=bundle["target_type"],
                        model_type=bundle["model_type"],
                        postprocess_type=bundle["postprocess_type"],
                        train_metrics=bundle["train_metrics"],
                        val_metrics=bundle["val_metrics"],
                        runtime_sec=bundle["runtime_sec"],
                        notes=spec.get("notes", ""),
                    )
                    row["profile_name"] = profile.profile_name
                    row["train_start"] = fold.train_dates[0]
                    row["train_end"] = fold.train_dates[-1]
                    row["val_start"] = fold.val_dates[0]
                    row["val_end"] = fold.val_dates[-1]
                    row["n_train_days"] = len(fold.train_dates)
                    row["n_val_days"] = len(fold.val_dates)
                    rows.append(row)
                except Exception as e:
                    # 记录失败而不中断，便于调试
                    rows.append({
                        "profile_name": profile.profile_name,
                        "fold_id": fold.fold_id,
                        "experiment_id": spec["experiment_id"],
                        "train_start": fold.train_dates[0],
                        "train_end": fold.train_dates[-1],
                        "val_start": fold.val_dates[0],
                        "val_end": fold.val_dates[-1],
                        "n_train_days": len(fold.train_dates),
                        "n_val_days": len(fold.val_dates),
                        "val_corr": np.nan,
                        "val_mse": np.nan,
                        "val_r2": np.nan,
                        "notes": f"ERROR: {str(e)[:200]}",
                    })

        return manifest, pd.DataFrame(rows)

    # ---------------------------------------------------------------- #
    # Profile 汇总
    # ---------------------------------------------------------------- #

    def summarize_profile(self, fold_df: pd.DataFrame, profile_name: str) -> pd.DataFrame:
        """聚合单个 profile 下每个实验的汇总指标"""
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for experiment_id, group in fold_df.groupby("experiment_id", sort=False):
            group = group.sort_values("fold_id")
            val_corrs = group["val_corr"].dropna().tolist()
            val_mses = group["val_mse"].dropna().tolist() if "val_mse" in group.columns else []
            val_r2s = group["val_r2"].dropna().tolist() if "val_r2" in group.columns else []
            n_folds = len(group)

            row: Dict[str, Any] = {
                "profile_name": profile_name,
                "experiment_id": experiment_id,
                "model_type": group["model_type"].iloc[0] if "model_type" in group.columns else "",
                "feature_set": group["feature_set"].iloc[0] if "feature_set" in group.columns else "",
                "target_type": group["target_type"].iloc[0] if "target_type" in group.columns else "",
                "n_folds": n_folds,
            }

            if val_corrs:
                row["rolling_corr_mean"] = float(np.mean(val_corrs))
                row["rolling_corr_std"] = float(np.std(val_corrs, ddof=0)) if len(val_corrs) > 1 else 0.0
                row["rolling_corr_min"] = float(np.min(val_corrs))
                row["rolling_corr_median"] = float(np.median(val_corrs))
                row["positive_fold_rate"] = float(sum(c > 0 for c in val_corrs) / len(val_corrs))
                row["stability_score"] = row["rolling_corr_mean"] - 0.7 * row["rolling_corr_std"]
            else:
                for k in ["rolling_corr_mean", "rolling_corr_std", "rolling_corr_min",
                          "rolling_corr_median", "positive_fold_rate", "stability_score"]:
                    row[k] = np.nan

            if val_mses:
                row["rolling_mse_mean"] = float(np.mean(val_mses))
                row["rolling_mse_std"] = float(np.std(val_mses, ddof=0)) if len(val_mses) > 1 else 0.0
            else:
                row["rolling_mse_mean"] = np.nan
                row["rolling_mse_std"] = np.nan

            if val_r2s:
                row["rolling_r2_mean"] = float(np.mean(val_r2s))
                row["rolling_r2_min"] = float(np.min(val_r2s))
            else:
                row["rolling_r2_mean"] = np.nan
                row["rolling_r2_min"] = np.nan

            if "daily_corr_mean" in group.columns:
                row["daily_corr_mean"] = float(group["daily_corr_mean"].mean())
            if "daily_corr_std" in group.columns:
                row["daily_corr_std"] = float(group["daily_corr_std"].mean())
            if "train_val_corr_gap" in group.columns:
                row["train_val_corr_gap_mean"] = float(group["train_val_corr_gap"].dropna().mean()) if group["train_val_corr_gap"].notna().any() else np.nan
            if "runtime_sec" in group.columns:
                row["runtime_sec_sum"] = float(group["runtime_sec"].sum())

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------- #
    # Holdout 评测
    # ---------------------------------------------------------------- #

    def run_holdout(
        self,
        train_start: int,
        train_end: int,
        holdout_start: int,
        holdout_end: int,
        specs: List[Dict],
        holdout_name: str = "holdout",
    ) -> pd.DataFrame:
        """
        单次 holdout 评测（不参与 rolling 汇总和 protocol_stability_score 计算）。

        holdout_name: "review"（11月）或 "final"（12月）
        """
        split_config = SplitConfig(
            train_start=train_start,
            train_end=train_end,
            val_start=holdout_start,
            val_end=holdout_end,
            test_start=holdout_start,
            test_end=holdout_end,
        )
        rows = []
        for spec in specs:
            start_ts = time.time()
            try:
                bundle = self.runner._evaluate_spec_on_fold(split_config, spec)
                rows.append({
                    "holdout_name": holdout_name,
                    "experiment_id": spec["experiment_id"],
                    "model_type": bundle["model_type"],
                    "feature_set": bundle["feature_set"],
                    "target_type": bundle["target_type"],
                    "train_start": train_start,
                    "train_end": train_end,
                    "holdout_start": holdout_start,
                    "holdout_end": holdout_end,
                    "holdout_corr": bundle["val_metrics"]["corr"],
                    "holdout_mse": bundle["val_metrics"]["mse"],
                    "holdout_r2": bundle["val_metrics"]["r2"],
                    "runtime_sec": float(time.time() - start_ts),
                    "notes": spec.get("notes", ""),
                })
            except Exception as e:
                rows.append({
                    "holdout_name": holdout_name,
                    "experiment_id": spec["experiment_id"],
                    "holdout_corr": np.nan,
                    "holdout_mse": np.nan,
                    "holdout_r2": np.nan,
                    "notes": f"ERROR: {str(e)[:200]}",
                })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- #
    # Leaderboard 构建
    # ---------------------------------------------------------------- #

    def build_leaderboard(
        self,
        profile_summaries: Dict[str, pd.DataFrame],
        baseline_id: str = BASELINE_ID,
        review_holdout_df: Optional[pd.DataFrame] = None,
        final_holdout_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构建跨 profile 加权综合 leaderboard，附带 baseline delta 和自动 decision"""
        # 收集所有实验 ID
        all_ids: List[str] = []
        seen: set = set()
        for df in profile_summaries.values():
            if df.empty or "experiment_id" not in df.columns:
                continue
            for eid in df["experiment_id"].tolist():
                if eid not in seen:
                    all_ids.append(eid)
                    seen.add(eid)

        if not all_ids:
            return pd.DataFrame()

        rows = []
        for experiment_id in all_ids:
            row: Dict[str, Any] = {"experiment_id": experiment_id}
            profile_scores: Dict[str, Dict] = {}

            # 各 profile 指标
            for pname, pdf in profile_summaries.items():
                if pdf.empty or "experiment_id" not in pdf.columns:
                    continue
                exp_rows = pdf[pdf["experiment_id"] == experiment_id]
                if exp_rows.empty:
                    continue
                r = exp_rows.iloc[0]
                # 取 profile 名称前缀（short/medium/long/expanding）
                prefix = pname.split("_")[0]
                for metric in ["corr_mean", "corr_std", "corr_min", "stability_score",
                               "n_folds", "positive_fold_rate"]:
                    src_key = f"rolling_{metric}" if metric in ["corr_mean", "corr_std", "corr_min"] else metric
                    row[f"{prefix}_{metric}"] = r.get(src_key, np.nan)

                profile_scores[pname] = {
                    "corr_mean": r.get("rolling_corr_mean", np.nan),
                    "stability": r.get("stability_score", np.nan),
                    "corr_min": r.get("rolling_corr_min", np.nan),
                    "mse_mean": r.get("rolling_mse_mean", np.nan),
                    "r2_mean": r.get("rolling_r2_mean", np.nan),
                    "positive_fold_rate": r.get("positive_fold_rate", np.nan),
                }
                # 顺便从第一个匹配的 profile 取元信息
                if "model_type" not in row:
                    row["model_type"] = r.get("model_type", "")
                    row["feature_set"] = r.get("feature_set", "")
                    row["target_type"] = r.get("target_type", "")

            # 加权综合指标
            row["protocol_corr_mean"] = _weighted_avg(profile_scores, "corr_mean")
            row["protocol_stability_score"] = _weighted_avg(profile_scores, "stability")
            row["protocol_corr_min"] = _weighted_avg(profile_scores, "corr_min")
            row["protocol_mse_mean"] = _weighted_avg(profile_scores, "mse_mean")
            row["protocol_r2_mean"] = _weighted_avg(profile_scores, "r2_mean")
            row["protocol_positive_fold_rate"] = _weighted_avg(profile_scores, "positive_fold_rate")

            # holdout 结果（不参与 protocol_stability_score）
            if review_holdout_df is not None and not review_holdout_df.empty:
                rv = review_holdout_df[review_holdout_df["experiment_id"] == experiment_id]
                if not rv.empty:
                    row["review_holdout_corr"] = rv.iloc[0].get("holdout_corr", np.nan)
                    row["review_holdout_mse"] = rv.iloc[0].get("holdout_mse", np.nan)
                    row["review_holdout_r2"] = rv.iloc[0].get("holdout_r2", np.nan)

            if final_holdout_df is not None and not final_holdout_df.empty:
                fv = final_holdout_df[final_holdout_df["experiment_id"] == experiment_id]
                if not fv.empty:
                    row["final_holdout_corr"] = fv.iloc[0].get("holdout_corr", np.nan)
                    row["final_holdout_mse"] = fv.iloc[0].get("holdout_mse", np.nan)
                    row["final_holdout_r2"] = fv.iloc[0].get("holdout_r2", np.nan)

            rows.append(row)

        lb = pd.DataFrame(rows)

        # baseline delta
        baseline_rows = lb[lb["experiment_id"] == baseline_id]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            base_corr = float(baseline.get("protocol_corr_mean", np.nan))
            base_stab = float(baseline.get("protocol_stability_score", np.nan))
            base_mse = float(baseline.get("protocol_mse_mean", np.nan))
            base_r2 = float(baseline.get("protocol_r2_mean", np.nan))

            lb["baseline_delta_corr"] = lb["protocol_corr_mean"].astype(float) - base_corr
            lb["baseline_delta_stability"] = lb["protocol_stability_score"].astype(float) - base_stab
            if not np.isnan(base_mse) and abs(base_mse) > 1e-12:
                lb["baseline_delta_mse_pct"] = (lb["protocol_mse_mean"].astype(float) - base_mse) / abs(base_mse)
            lb["baseline_delta_r2"] = lb["protocol_r2_mean"].astype(float) - base_r2

        # 自动 decision
        if not baseline_rows.empty:
            baseline_dict = baseline_rows.iloc[0].to_dict()
            decisions, reasons = [], []
            for _, r in lb.iterrows():
                d, rsn = make_decision(r.to_dict(), baseline_dict)
                decisions.append(d)
                reasons.append(rsn)
            lb["decision"] = decisions
            lb["reason"] = reasons
            # 基线本身标记
            lb.loc[lb["experiment_id"] == baseline_id, "decision"] = "baseline"
            lb.loc[lb["experiment_id"] == baseline_id, "reason"] = "当前稳定基线"

        # 按 protocol_stability_score 降序排列
        if "protocol_stability_score" in lb.columns:
            lb = lb.sort_values("protocol_stability_score", ascending=False).reset_index(drop=True)

        return lb

    # ---------------------------------------------------------------- #
    # 主入口
    # ---------------------------------------------------------------- #

    def run_full_protocol(
        self,
        rolling_start: int,
        rolling_end: int,
        specs: List[Dict],
        profiles: Optional[List[RollingProfile]] = None,
        max_folds: Optional[int] = None,
        include_review_holdout: bool = False,
        review_train_start: Optional[int] = None,
        review_train_end: Optional[int] = None,
        review_holdout_start: Optional[int] = None,
        review_holdout_end: Optional[int] = None,
        include_final_holdout: bool = False,
        final_train_start: Optional[int] = None,
        final_train_end: Optional[int] = None,
        final_holdout_start: Optional[int] = None,
        final_holdout_end: Optional[int] = None,
        baseline_id: str = BASELINE_ID,
        n_workers: int = 1,
        resume: bool = False,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        主入口：运行完整三层评测协议。

        流程：
          1. 对每个 profile 运行所有 specs → fold_metrics
          2. 聚合每个 profile 的 summary
          3. 可选：review holdout（11月）
          4. 可选：final holdout（12月，尽量少跑）
          5. 构建 leaderboard（含 baseline delta 和 decision）
          6. 保存所有输出到 output_dir/<run_id>/

        返回包含所有结果 DataFrame 的字典。
        """
        if profiles is None:
            profiles = ROLLING_PROFILES
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n[Protocol] run_id={run_id}")
        print(f"[Protocol] rolling 范围：{rolling_start} ~ {rolling_end}")
        print(f"[Protocol] profiles: {[p.profile_name for p in profiles]}")
        print(f"[Protocol] specs 数量: {len(specs)}")
        if max_folds:
            print(f"[Protocol] max_folds={max_folds}（调试模式）")

        all_manifests: List[FoldManifestEntry] = []
        all_fold_metrics: List[pd.DataFrame] = []
        profile_summaries: Dict[str, pd.DataFrame] = {}
        _parallel_wrote_fold_metrics = False  # 并行模式下 scheduler 已增量落盘，末尾跳过重写

        if n_workers > 1 and ParallelScheduler is not None:
            # ── 并行路径 ─────────────────────────────────────────────────────
            # 1. 主进程统一构建所有 fold manifest
            profiles_with_folds = []
            for profile in profiles:
                folds, manifest = self.build_folds_for_profile(
                    profile, rolling_start, rolling_end, max_folds=max_folds
                )
                profiles_with_folds.append((profile, folds))
                all_manifests.extend(manifest)

            # 2. 提前创建输出目录，配置 scheduler 落盘路径（供 resume 使用）
            _run_dir_parallel = os.path.join(output_dir, run_id) if output_dir else None
            if _run_dir_parallel:
                os.makedirs(_run_dir_parallel, exist_ok=True)
            _fold_metrics_csv = (
                os.path.join(_run_dir_parallel, "fold_metrics.csv")
                if _run_dir_parallel else None
            )

            # 3. 并发执行
            h5dir = self.runner.loader.h5dir
            scheduler = ParallelScheduler(h5dir, n_workers=n_workers)
            if _fold_metrics_csv:
                scheduler.set_output_path(_fold_metrics_csv)
                _parallel_wrote_fold_metrics = True

            merged_fold_df = scheduler.run(profiles_with_folds, specs, resume=resume)

            # 4. 按 profile 拆分，分别做 summarize_profile
            if not merged_fold_df.empty:
                all_fold_metrics.append(merged_fold_df)
            for profile in profiles:
                pname = profile.profile_name
                profile_df = (
                    merged_fold_df[merged_fold_df["profile_name"] == pname].copy()
                    if not merged_fold_df.empty else pd.DataFrame()
                )
                if not profile_df.empty:
                    n_folds = profile_df["fold_id"].nunique()
                    n_exp = profile_df["experiment_id"].nunique()
                    print(f"  [Profile] {pname}: {n_folds} folds × {n_exp} experiments = {len(profile_df)} rows")
                else:
                    print(f"  [Profile] {pname}: 无有效结果")
                summary = self.summarize_profile(profile_df, pname)
                profile_summaries[pname] = summary

        else:
            # ── 串行路径（原有逻辑，完全不变）────────────────────────────────
            for profile in profiles:
                print(f"\n[Profile] {profile.profile_name} (mode={profile.mode})")
                manifest, fold_df = self.run_profile(
                    profile, rolling_start, rolling_end, specs, max_folds=max_folds
                )
                all_manifests.extend(manifest)
                if not fold_df.empty:
                    all_fold_metrics.append(fold_df)
                    n_folds = fold_df["fold_id"].nunique()
                    n_exp = fold_df["experiment_id"].nunique()
                    print(f"  → {n_folds} folds × {n_exp} experiments = {len(fold_df)} rows")
                else:
                    print("  → 无有效 fold（日期范围不足）")

                summary = self.summarize_profile(fold_df, profile.profile_name)
                profile_summaries[profile.profile_name] = summary

        # 组装汇总表
        fold_manifest_df = (
            pd.DataFrame([vars(m) for m in all_manifests])
            if all_manifests else pd.DataFrame()
        )
        fold_metrics_df = (
            pd.concat(all_fold_metrics, ignore_index=True)
            if all_fold_metrics else pd.DataFrame()
        )
        profile_summary_df = (
            pd.concat(
                [df for df in profile_summaries.values() if not df.empty],
                ignore_index=True,
            )
            if any(not df.empty for df in profile_summaries.values())
            else pd.DataFrame()
        )

        # Review holdout（11月）
        review_holdout_df: Optional[pd.DataFrame] = None
        if include_review_holdout and all(
            x is not None for x in [review_train_start, review_train_end, review_holdout_start, review_holdout_end]
        ):
            print(f"\n[Holdout] review：train {review_train_start}~{review_train_end}, holdout {review_holdout_start}~{review_holdout_end}")
            review_holdout_df = self.run_holdout(
                review_train_start, review_train_end,
                review_holdout_start, review_holdout_end,
                specs, holdout_name="review",
            )

        # Final holdout（12月）
        final_holdout_df: Optional[pd.DataFrame] = None
        if include_final_holdout and all(
            x is not None for x in [final_train_start, final_train_end, final_holdout_start, final_holdout_end]
        ):
            print(f"\n[Holdout] final：train {final_train_start}~{final_train_end}, holdout {final_holdout_start}~{final_holdout_end}")
            final_holdout_df = self.run_holdout(
                final_train_start, final_train_end,
                final_holdout_start, final_holdout_end,
                specs, holdout_name="final",
            )

        # Leaderboard
        print("\n[Protocol] 构建 leaderboard...")
        leaderboard_df = self.build_leaderboard(
            profile_summaries,
            baseline_id=baseline_id,
            review_holdout_df=review_holdout_df,
            final_holdout_df=final_holdout_df,
        )

        # 保存输出
        if output_dir:
            run_dir = os.path.join(output_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            config_data = {
                "run_id": run_id,
                "rolling_start": rolling_start,
                "rolling_end": rolling_end,
                "profiles": [vars(p) for p in profiles],
                "specs": [s["experiment_id"] for s in specs],
                "max_folds": max_folds,
                "baseline_id": baseline_id,
                "include_review_holdout": include_review_holdout,
                "include_final_holdout": include_final_holdout,
                "review_train_start": review_train_start,
                "review_train_end": review_train_end,
                "review_holdout_start": review_holdout_start,
                "review_holdout_end": review_holdout_end,
                "final_train_start": final_train_start,
                "final_train_end": final_train_end,
                "final_holdout_start": final_holdout_start,
                "final_holdout_end": final_holdout_end,
            }
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2, default=str)

            if not fold_manifest_df.empty:
                fold_manifest_df.to_csv(os.path.join(run_dir, "fold_manifest.csv"), index=False, encoding="utf-8-sig")
            if not fold_metrics_df.empty and not _parallel_wrote_fold_metrics:
                fold_metrics_df.to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False, encoding="utf-8-sig")
            if not profile_summary_df.empty:
                profile_summary_df.to_csv(os.path.join(run_dir, "profile_summary.csv"), index=False, encoding="utf-8-sig")
            if not leaderboard_df.empty:
                leaderboard_df.to_csv(os.path.join(run_dir, "leaderboard.csv"), index=False, encoding="utf-8-sig")
            if review_holdout_df is not None and not review_holdout_df.empty:
                review_holdout_df.to_csv(os.path.join(run_dir, "review_holdout.csv"), index=False, encoding="utf-8-sig")
            if final_holdout_df is not None and not final_holdout_df.empty:
                final_holdout_df.to_csv(os.path.join(run_dir, "final_holdout.csv"), index=False, encoding="utf-8-sig")

            print(f"\n[Protocol] 输出已保存至: {run_dir}")

        return {
            "run_id": run_id,
            "fold_manifest": fold_manifest_df,
            "fold_metrics": fold_metrics_df,
            "profile_summary": profile_summary_df,
            "profile_summaries": profile_summaries,
            "leaderboard": leaderboard_df,
            "review_holdout": review_holdout_df,
            "final_holdout": final_holdout_df,
        }
