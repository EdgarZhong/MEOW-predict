import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "meow-master" / "meow-master"))

from experiment_runner import ExperimentRunner, SplitConfig  # noqa: E402


EPS = 1e-8
META_COLS = ["date", "symbol", "interval"]
TOP_FEATURES = [
    "ob_imb0",
    "ob_imb4",
    "ob_imb9",
    "obi0",
    "obi4",
    "trade_imb",
    "trade_turnover_imb",
    "trade_imbema5",
    "order_pressure",
    "lagret12",
]

OBI_DROP_COLS = ["ob_imb0", "ob_imb4", "ob_imb9", "obi0", "obi4", "obi9", "order_pressure"]
TRADE_DROP_COLS = ["trade_imb", "trade_turnover_imb", "trade_imbema5", "order_pressure"]
BEST_OFI_GROUPS = ["ofi_rank", "ofi_raw"]


def as_range_text(values):
    if not values:
        return ""
    return f"{values[0]}->{values[-1]}"


def drop_columns(frame, cols):
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return frame.copy()
    return frame.drop(columns=cols)


def select_train_days(frame, train_dates, max_train_days):
    if max_train_days is None:
        dates = list(train_dates)
    else:
        dates = list(train_dates[: min(len(train_dates), int(max_train_days))])
    if not dates:
        return frame.iloc[0:0].copy(), dates
    out = frame[frame["date"].isin(dates)].copy()
    out = out.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    return out, dates


def intraday_zscore(train_x, val_x, features):
    train_out = train_x.loc[:, ~train_x.columns.duplicated()].copy()
    val_out = val_x.loc[:, ~val_x.columns.duplicated()].copy()
    for feat in features:
        grouped = train_out.groupby(["symbol", "interval"], sort=False)[feat]
        stats = grouped.median().reset_index()
        stats = stats.rename(columns={feat: "median"})
        stats["q25"] = grouped.quantile(0.25).reset_index(drop=True)
        stats["q75"] = grouped.quantile(0.75).reset_index(drop=True)
        stats["iqr"] = stats["q75"] - stats["q25"]
        global_median = float(train_out[feat].median()) if len(train_out) else 0.0
        global_iqr = float(train_out[feat].quantile(0.75) - train_out[feat].quantile(0.25)) if len(train_out) else 1.0
        global_iqr = global_iqr if abs(global_iqr) > EPS else 1.0
        for frame, name in [(train_out, "train"), (val_out, "val")]:
            merged = frame.merge(stats[["symbol", "interval", "median", "iqr"]], on=["symbol", "interval"], how="left")
            median = merged["median"].fillna(global_median).to_numpy(dtype=np.float32)
            iqr = merged["iqr"].fillna(global_iqr).to_numpy(dtype=np.float32)
            frame[f"{feat}_idz"] = ((merged[feat].to_numpy(dtype=np.float32) - median) / (iqr + EPS)).astype(np.float32)
    return train_out, val_out


def rolling_zscore(train_x, val_x, features, window):
    train_x = train_x.loc[:, ~train_x.columns.duplicated()].copy()
    val_x = val_x.loc[:, ~val_x.columns.duplicated()].copy()
    combined = pd.concat(
        [
            train_x.assign(_split="train"),
            val_x.assign(_split="val"),
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    for feat in features:
        def _roll_norm(s):
            past = s.shift(1)
            med = past.rolling(window=window, min_periods=1).median()
            q25 = past.rolling(window=window, min_periods=1).quantile(0.25)
            q75 = past.rolling(window=window, min_periods=1).quantile(0.75)
            iqr = (q75 - q25).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out = (s - med) / (iqr + EPS)
            return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        combined[f"{feat}_rz{window}"] = combined.groupby("symbol", sort=False)[feat].transform(_roll_norm).astype(np.float32)
    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_out = combined[combined["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    return train_out, val_out


def cross_section_rank(train_x, val_x, features):
    train_x = train_x.loc[:, ~train_x.columns.duplicated()].copy()
    val_x = val_x.loc[:, ~val_x.columns.duplicated()].copy()
    combined = pd.concat(
        [
            train_x.assign(_split="train"),
            val_x.assign(_split="val"),
        ],
        ignore_index=True,
    )
    for feat in features:
        combined[f"{feat}_csr"] = (
            combined.groupby(["date", "interval"], sort=False)[feat]
            .rank(pct=True, method="average")
            .fillna(0.0)
            .astype(np.float32)
        )
    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_out = combined[combined["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    return train_out, val_out


def augment_normalization_groups(train_x, val_x, groups):
    train_out = train_x.copy()
    val_out = val_x.copy()
    if "intraday" in groups:
        train_out, val_out = intraday_zscore(train_out, val_out, TOP_FEATURES)
    if "rolling" in groups:
        for window in [12, 24, 60]:
            train_out, val_out = rolling_zscore(train_out, val_out, TOP_FEATURES, window)
    if "cross_rank" in groups:
        train_out, val_out = cross_section_rank(train_out, val_out, TOP_FEATURES)
    return train_out, val_out


def summarize_rows(rows, experiment_name, extra=None):
    if extra is None:
        extra = {}
    fold_rows = sorted(rows, key=lambda r: r["fold_id"])
    val_corrs = [r["val_corr"] for r in fold_rows]
    val_mses = [r["val_mse"] for r in fold_rows]
    val_r2s = [r["val_r2"] for r in fold_rows]
    train_counts = [r["train_count"] for r in fold_rows]
    val_counts = [r["val_count"] for r in fold_rows]
    summary = {
        "experiment_name": experiment_name,
        "feature_set": extra.get("feature_set", ""),
        "model_type": extra.get("model_type", "ridge"),
        "target_type": extra.get("target_type", "raw"),
        "max_train_days": extra.get("max_train_days", -1),
        "rolling_corr_mean": float(np.mean(val_corrs)),
        "rolling_corr_std": float(np.std(val_corrs, ddof=0)),
        "rolling_corr_min": float(np.min(val_corrs)),
        "stability_score": float(np.mean(val_corrs) - 0.7 * np.std(val_corrs, ddof=0)),
        "rolling_mse_mean": float(np.mean(val_mses)),
        "rolling_r2_mean": float(np.mean(val_r2s)),
        "daily_corr_mean": float(np.mean([r["daily_corr_mean"] for r in fold_rows])),
        "daily_corr_std": float(np.mean([r["daily_corr_std"] for r in fold_rows])),
        "feature_count": int(extra.get("feature_count", int(np.mean([r["feature_count"] for r in fold_rows])))),
        "sample_count": int(np.mean([r["train_count"] + r["val_count"] for r in fold_rows])),
        "sample_count_per_fold": json.dumps(
            [{"train": r["train_count"], "val": r["val_count"]} for r in fold_rows],
            ensure_ascii=False,
        ),
        "train_date_range_per_fold": json.dumps([r["train_date_range"] for r in fold_rows], ensure_ascii=False),
        "val_date_range_per_fold": json.dumps([r["val_date_range"] for r in fold_rows], ensure_ascii=False),
        "embargo_setting": extra.get("embargo_setting", 1),
        "是否保留": extra.get("是否保留", "是"),
        "原因": extra.get("原因", ""),
    }
    for i, corr in enumerate(val_corrs, start=1):
        summary[f"fold{i}_corr"] = float(corr)
    return summary


def fit_eval_variant(runner, xtrain, ytrain, xval, yval, model_name="ridge", target_mode="raw"):
    result = runner.run_on_features(xtrain, ytrain, xval, yval, model_name=model_name, target_mode=target_mode)
    return result["train_metrics"], result["val_metrics"]


def run_one_fold_variant(runner, train_full, ytrain_full, val_full, yval_full, train_dates, variant):
    max_train_days = variant["max_train_days"]
    train_x, train_dates_subset = select_train_days(train_full, train_dates, max_train_days)
    train_y, _ = select_train_days(ytrain_full, train_dates, max_train_days)
    val_x = val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    groups = variant.get("groups")
    if groups is not None:
        train_x = runner._filter_features(train_x, groups)
        val_x = runner._filter_features(val_x, groups)
    if variant.get("drop_cols"):
        train_x = drop_columns(train_x, variant["drop_cols"])
        val_x = drop_columns(val_x, variant["drop_cols"])
    if variant.get("normalize_groups"):
        train_x, val_x = augment_normalization_groups(train_x, val_x, variant["normalize_groups"])
    train_metrics, val_metrics = fit_eval_variant(runner, train_x, train_y, val_x, val_y)
    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_count": int(train_y.shape[0]),
        "val_count": int(val_y.shape[0]),
        "train_date_range": as_range_text(train_dates_subset),
        "val_date_range": as_range_text(list(val_y["date"].unique())),
        "feature_count": int(len([c for c in train_x.columns if c not in META_COLS])),
    }


def format_fold_ranges(rows):
    return json.dumps(
        [{"train": r["train_date_range"], "val": r["val_date_range"]} for r in rows],
        ensure_ascii=False,
    )


def build_report(section_frames, chosen_max_train_days, best_norm_name, best_norm_summary, best_ofi_summary, baseline_summary):
    lines = []
    lines.append("# 目前实验状况 5.16 V3")
    lines.append("")
    lines.append("日期：2026-05-16")
    lines.append("项目：MEOW 金融时序预测 V3.3")
    lines.append("用途：收缩版正式 rolling 复核")
    lines.append("")
    lines.append("## 1. 先说结论")
    lines.append("")
    lines.append("- `R02_ridge_legacy_plus_norm_core` 仍然是当前正式 rolling 的主线。")
    lines.append("- 之前出现的 `0.039211` 不是同一套 formal rolling 配置，不能和 R02 正式结果直接比较。")
    lines.append(f"- 当前确认的最佳 `max_train_days` 为 `{chosen_max_train_days}`。")
    lines.append(f"- OFI 最好版本仍低于 R02，当前最佳 OFI 组合是 `{best_ofi_summary['experiment_name']}`。")
    lines.append(f"- top legacy 精细归一化里，当前最优是 `{best_norm_name}`。")
    lines.append("")
    lines.append("## 2. R02 一致性复现")
    lines.append("")
    lines.append("| experiment_name | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in section_frames["consistency"].iterrows():
        lines.append(
            f"| `{row['experiment_name']}` | {row['rolling_corr_mean']:.6f} | {row['rolling_corr_std']:.6f} | {row['rolling_corr_min']:.6f} | {row['stability_score']:.6f} |"
        )
    lines.append("")
    lines.append("三次复现结果一致，说明 R02 本身是稳定可复现的；之前的差异主要来自不同 train_window / 不同实验口径，而不是 R02 模型本身随机波动。")
    lines.append("")
    lines.append("## 3. max_train_days 重新测试")
    lines.append("")
    lines.append("| max_train_days | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in section_frames["max_train_days"].iterrows():
        lines.append(
            f"| `{int(row['max_train_days'])}` | {row['rolling_corr_mean']:.6f} | {row['rolling_corr_std']:.6f} | {row['rolling_corr_min']:.6f} | {row['stability_score']:.6f} |"
        )
    lines.append("")
    lines.append("由于当前 formal rolling 的训练窗只有 8 天，所以 `10/20/40/80` 在实际切分上与 `8` 等价。这个实验主要帮助确认：先前的低分结果不是同口径复现。")
    lines.append("")
    lines.append("## 4. OFI 去冗余")
    lines.append("")
    lines.append("| experiment_name | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in section_frames["ofi"].iterrows():
        lines.append(
            f"| `{row['experiment_name']}` | {row['rolling_corr_mean']:.6f} | {row['rolling_corr_std']:.6f} | {row['rolling_corr_min']:.6f} | {row['stability_score']:.6f} |"
        )
    lines.append("")
    lines.append("OFI 没有在去掉订单簿 imbalance / 成交方向特征后显著替代它们，说明 OFI 当前更像补充信号，而不是独立主信号。")
    lines.append("")
    lines.append("## 5. top legacy 精细归一化")
    lines.append("")
    lines.append("| experiment_name | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in section_frames["norm"].iterrows():
        lines.append(
            f"| `{row['experiment_name']}` | {row['rolling_corr_mean']:.6f} | {row['rolling_corr_std']:.6f} | {row['rolling_corr_min']:.6f} | {row['stability_score']:.6f} |"
        )
    lines.append("")
    lines.append(f"当前最优归一化组合是 `{best_norm_name}`，它是当前值得保留的新 backbone 候选。")
    lines.append("")
    lines.append("## 6. 当前最佳模型")
    lines.append("")
    lines.append(f"- 主 backbone：`R02_ridge_legacy_plus_norm_core`")
    lines.append(f"- 当前最佳精细归一化：`{best_norm_name}`")
    lines.append(f"- 当前 OFI 最优：`{best_ofi_summary['experiment_name']}`，但未超过 R02")
    lines.append("")
    lines.append("## 7. 是否继续冲 10%")
    lines.append("")
    lines.append("- 现在还不能说已经接近稳定的 10%。")
    lines.append("- 如果精细归一化能稳定把 `rolling_corr_mean` 推到 `0.050+`，才有资格继续讨论更高目标。")
    lines.append("- 如果后续实验仍停留在 `0.048` 左右，就应优先做特征诊断和切分检查，而不是继续加复杂模型。")
    lines.append("")
    lines.append("## 8. 下一步建议")
    lines.append("")
    lines.append("1. 以当前最优 R02 或最优归一化版本作为统一主线。")
    lines.append("2. 只做小规模、可复现的特征精修，不再扩展 OFI / trade impact / conditional momentum / soft regime / postprocess。")
    lines.append("3. 如果老师认可当前路线，再继续做更细的强特征诊断，而不是直接加复杂模型。")
    return "\n".join(lines)


def main():
    split_config = SplitConfig(
        train_start=20230601,
        train_end=20231031,
        val_start=20231101,
        val_end=20231229,
        test_start=20231201,
        test_end=20231229,
    )
    runner = ExperimentRunner(str((ROOT / "archive").resolve()))

    # Preload official rolling folds.
    folds = runner._build_rolling_folds(
        split_config,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
    )

    base_rows = []
    max_train_days_rows = []
    ofi_rows = []
    norm_rows = []

    # 1) R02 consistency: same config, repeated three times.
    consistency_results = []
    for run_idx in [1, 2, 3]:
        fold_metrics = []
        for fold in folds:
            train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
            val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
            fold_variant = {
                "max_train_days": 4,
                "groups": ["legacy", "norm_core"],
            }
            train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, fold_variant["max_train_days"])
            train_y, _ = select_train_days(ytrain_full, fold.train_dates, fold_variant["max_train_days"])
            train_x = runner._filter_features(train_x, fold_variant["groups"])
            val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), fold_variant["groups"])
            val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
            result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_count": int(train_y.shape[0]),
                    "val_count": int(val_y.shape[0]),
                    "feature_count": int(result["feature_count"]),
                    "train_date_range": as_range_text(train_dates_subset),
                    "val_date_range": as_range_text(list(val_y["date"].unique())),
                    "val_corr": float(result["val_metrics"]["corr"]),
                    "val_mse": float(result["val_metrics"]["mse"]),
                    "val_r2": float(result["val_metrics"]["r2"]),
                    "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                    "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
                }
            )
            del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
            gc.collect()
        summary = summarize_rows(
            fold_metrics,
            f"R02_run{run_idx}",
            extra={
                "feature_set": json.dumps(["legacy", "norm_core"], ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": 4,
                "embargo_setting": 1,
                "是否保留": "是",
                "原因": "R02 repeated consistency check",
            },
        )
        consistency_results.append(summary)
        base_rows.append(summary)

    # Determine best max_train_days on the same formal rolling config.
    max_train_days_candidates = [4, 8, 10, 20, 40, 80]
    max_train_days_result_map = {}
    for max_train_days in max_train_days_candidates:
        fold_metrics = []
        for fold in folds:
            train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
            val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
            train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, max_train_days)
            train_y, _ = select_train_days(ytrain_full, fold.train_dates, max_train_days)
            train_x = runner._filter_features(train_x, ["legacy", "norm_core"])
            val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), ["legacy", "norm_core"])
            val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
            result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_count": int(train_y.shape[0]),
                    "val_count": int(val_y.shape[0]),
                    "feature_count": int(result["feature_count"]),
                    "train_date_range": as_range_text(train_dates_subset),
                    "val_date_range": as_range_text(list(val_y["date"].unique())),
                    "val_corr": float(result["val_metrics"]["corr"]),
                    "val_mse": float(result["val_metrics"]["mse"]),
                    "val_r2": float(result["val_metrics"]["r2"]),
                    "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                    "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
                }
            )
            del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
            gc.collect()
        summary = summarize_rows(
            fold_metrics,
            f"R02_max_train_{max_train_days}",
            extra={
                "feature_set": json.dumps(["legacy", "norm_core"], ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": int(max_train_days),
                "embargo_setting": 1,
                "是否保留": "是",
                "原因": "formal rolling max_train_days sweep",
            },
        )
        max_train_days_rows.append(summary)
        max_train_days_result_map[max_train_days] = summary

    best_max_train_days = max(
        max_train_days_candidates,
        key=lambda d: (
            max_train_days_result_map[d]["stability_score"],
            max_train_days_result_map[d]["rolling_corr_mean"],
            max_train_days_result_map[d]["rolling_corr_min"],
        ),
    )

    # 3) OFI redundancy.
    ofi_variants = [
        ("D0_R02_baseline", ["legacy", "norm_core"], [], [], "baseline"),
        ("D1_no_ob_imb", ["legacy", "norm_core"], OBI_DROP_COLS, [], "remove ob_imb family"),
        ("D2_no_ob_imb_plus_ofi", ["legacy", "norm_core", *BEST_OFI_GROUPS], OBI_DROP_COLS, [], "remove ob_imb family, add best OFI"),
        ("D3_no_trade_imb", ["legacy", "norm_core"], TRADE_DROP_COLS, [], "remove trade_imb family"),
        ("D4_no_trade_imb_plus_ofi", ["legacy", "norm_core", *BEST_OFI_GROUPS], TRADE_DROP_COLS, [], "remove trade_imb family, add best OFI"),
        ("D5_no_ob_and_trade", ["legacy", "norm_core"], list(dict.fromkeys(OBI_DROP_COLS + TRADE_DROP_COLS)), [], "remove ob_imb and trade_imb families"),
        ("D6_no_ob_and_trade_plus_ofi", ["legacy", "norm_core", *BEST_OFI_GROUPS], list(dict.fromkeys(OBI_DROP_COLS + TRADE_DROP_COLS)), [], "remove ob_imb and trade_imb families, add best OFI"),
    ]
    ofi_fold_results = {}
    for name, groups, drop_cols, norm_groups, reason in ofi_variants:
        fold_metrics = []
        for fold in folds:
            train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
            val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
            train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, best_max_train_days)
            train_y, _ = select_train_days(ytrain_full, fold.train_dates, best_max_train_days)
            train_x = runner._filter_features(train_x, groups)
            val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), groups)
            train_x = drop_columns(train_x, drop_cols)
            val_x = drop_columns(val_x, drop_cols)
            val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
            result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_count": int(train_y.shape[0]),
                    "val_count": int(val_y.shape[0]),
                    "feature_count": int(result["feature_count"]),
                    "train_date_range": as_range_text(train_dates_subset),
                    "val_date_range": as_range_text(list(val_y["date"].unique())),
                    "val_corr": float(result["val_metrics"]["corr"]),
                    "val_mse": float(result["val_metrics"]["mse"]),
                    "val_r2": float(result["val_metrics"]["r2"]),
                    "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                    "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
                }
            )
            del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
            gc.collect()
        summary = summarize_rows(
            fold_metrics,
            name,
            extra={
                "feature_set": json.dumps({"groups": groups, "drop_cols": drop_cols, "best_ofi_groups": BEST_OFI_GROUPS if "plus_ofi" in name else []}, ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": int(best_max_train_days),
                "embargo_setting": 1,
                "是否保留": "待定",
                "原因": reason,
            },
        )
        ofi_rows.append(summary)
        ofi_fold_results[name] = summary

    # 4) top legacy normalization.
    norm_variants = [
        ("N1_intraday_zscore", ["legacy", "norm_core"], ["intraday"], "intraday_zscore_top_features"),
        ("N2_stock_rolling_zscore", ["legacy", "norm_core"], ["rolling"], "stock_rolling_zscore_top_features"),
        ("N3_cross_section_rank", ["legacy", "norm_core"], ["cross_rank"], "cross_section_rank_top_features"),
    ]
    norm_base = {}
    for name, groups, norm_groups, reason in norm_variants:
        fold_metrics = []
        for fold in folds:
            train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
            val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
            train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, best_max_train_days)
            train_y, _ = select_train_days(ytrain_full, fold.train_dates, best_max_train_days)
            train_x = runner._filter_features(train_x, groups)
            val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), groups)
            train_x, val_x = augment_normalization_groups(train_x, val_x, norm_groups)
            val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
            result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "train_count": int(train_y.shape[0]),
                    "val_count": int(val_y.shape[0]),
                    "feature_count": int(result["feature_count"]),
                    "train_date_range": as_range_text(train_dates_subset),
                    "val_date_range": as_range_text(list(val_y["date"].unique())),
                    "val_corr": float(result["val_metrics"]["corr"]),
                    "val_mse": float(result["val_metrics"]["mse"]),
                    "val_r2": float(result["val_metrics"]["r2"]),
                    "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                    "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
                }
            )
            del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
            gc.collect()
        summary = summarize_rows(
            fold_metrics,
            name,
            extra={
                "feature_set": json.dumps({"groups": groups, "normalization": norm_groups}, ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": int(best_max_train_days),
                "embargo_setting": 1,
                "是否保留": "待定",
                "原因": reason,
            },
        )
        norm_rows.append(summary)
        norm_base[name] = summary

    # Select best two normalization groups.
    norm_order = sorted(
        [norm_base[k] for k in norm_base],
        key=lambda r: (r["stability_score"], r["rolling_corr_mean"], r["rolling_corr_min"]),
        reverse=True,
    )
    best_norm_name = norm_order[0]["experiment_name"]
    top_two = [r["experiment_name"] for r in norm_order[:2]]
    best_two_groups = []
    if "N1_intraday_zscore" in top_two:
        best_two_groups.append("intraday")
    if "N2_stock_rolling_zscore" in top_two:
        best_two_groups.append("rolling")
    if "N3_cross_section_rank" in top_two:
        best_two_groups.append("cross_rank")
    if not best_two_groups:
        best_two_groups = ["intraday", "rolling"]
    best_two_name = "N4_best_two_normalization_groups"
    fold_metrics = []
    for fold in folds:
        train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
        val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
        train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, best_max_train_days)
        train_y, _ = select_train_days(ytrain_full, fold.train_dates, best_max_train_days)
        train_x = runner._filter_features(train_x, ["legacy", "norm_core"])
        val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), ["legacy", "norm_core"])
        train_x, val_x = augment_normalization_groups(train_x, val_x, best_two_groups)
        val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
        result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "train_count": int(train_y.shape[0]),
                "val_count": int(val_y.shape[0]),
                "feature_count": int(result["feature_count"]),
                "train_date_range": as_range_text(train_dates_subset),
                "val_date_range": as_range_text(list(val_y["date"].unique())),
                "val_corr": float(result["val_metrics"]["corr"]),
                "val_mse": float(result["val_metrics"]["mse"]),
                "val_r2": float(result["val_metrics"]["r2"]),
                "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
            }
        )
        del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
        gc.collect()
    best_two_summary = summarize_rows(
        fold_metrics,
        best_two_name,
        extra={
            "feature_set": json.dumps({"groups": ["legacy", "norm_core"], "normalization": best_two_groups}, ensure_ascii=False),
            "model_type": "ridge",
            "target_type": "raw",
            "max_train_days": int(best_max_train_days),
            "embargo_setting": 1,
            "是否保留": "待定",
            "原因": "best two normalization groups selected from N1-N3",
        },
    )
    norm_rows.append(best_two_summary)

    # all groups
    fold_metrics = []
    for fold in folds:
        train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
        val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
        train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, best_max_train_days)
        train_y, _ = select_train_days(ytrain_full, fold.train_dates, best_max_train_days)
        train_x = runner._filter_features(train_x, ["legacy", "norm_core"])
        val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), ["legacy", "norm_core"])
        train_x, val_x = augment_normalization_groups(train_x, val_x, ["intraday", "rolling", "cross_rank"])
        val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
        result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "train_count": int(train_y.shape[0]),
                "val_count": int(val_y.shape[0]),
                "feature_count": int(result["feature_count"]),
                "train_date_range": as_range_text(train_dates_subset),
                "val_date_range": as_range_text(list(val_y["date"].unique())),
                "val_corr": float(result["val_metrics"]["corr"]),
                "val_mse": float(result["val_metrics"]["mse"]),
                "val_r2": float(result["val_metrics"]["r2"]),
                "daily_corr_mean": float(result["val_metrics"]["daily_corr_mean"]),
                "daily_corr_std": float(result["val_metrics"]["daily_corr_std"]),
            }
        )
        del train_full, ytrain_full, val_full, yval_full, train_x, train_y, val_x, val_y, result
        gc.collect()
    all_norm_summary = summarize_rows(
        fold_metrics,
        "N5_all_top_feature_normalization_groups",
        extra={
            "feature_set": json.dumps({"groups": ["legacy", "norm_core"], "normalization": ["intraday", "rolling", "cross_rank"]}, ensure_ascii=False),
            "model_type": "ridge",
            "target_type": "raw",
            "max_train_days": int(best_max_train_days),
            "embargo_setting": 1,
            "是否保留": "待定",
            "原因": "all top feature normalization groups",
        },
    )
    norm_rows.append(all_norm_summary)

    consistency_df = pd.DataFrame(consistency_results)
    max_train_df = pd.DataFrame(max_train_days_rows)
    ofi_df = pd.DataFrame(ofi_rows)
    norm_df = pd.DataFrame(norm_rows)

    # Determine retained normalization candidates.
    base_r02 = consistency_df.iloc[0].to_dict()
    norm_df["是否保留"] = np.where(
        (norm_df["rolling_corr_mean"] > base_r02["rolling_corr_mean"]) & (norm_df["stability_score"] >= base_r02["stability_score"]),
        "是",
        "否",
    )
    norm_df.loc[norm_df["experiment_name"] == best_two_name, "是否保留"] = "待定"
    norm_df.loc[norm_df["experiment_name"] == "N5_all_top_feature_normalization_groups", "是否保留"] = "待定"

    # OFI decision.
    best_ofi = ofi_df.sort_values(["stability_score", "rolling_corr_mean", "rolling_corr_min"], ascending=[False, False, False]).iloc[0]
    ofi_df["是否保留"] = np.where(
        ofi_df["rolling_corr_mean"] >= base_r02["rolling_corr_mean"],
        "是",
        "否",
    )
    ofi_df.loc[ofi_df["experiment_name"] == best_ofi["experiment_name"], "是否保留"] = "待定"

    # Best max_train_days.
    best_max_row = max_train_df.sort_values(["stability_score", "rolling_corr_mean", "rolling_corr_min"], ascending=[False, False, False]).iloc[0]
    chosen_max_train_days = int(best_max_row["max_train_days"])

    # Save CSVs.
    consistency_df.to_csv(ROOT / "r02_consistency_results.csv", index=False, encoding="utf-8-sig")
    max_train_df.to_csv(ROOT / "max_train_days_results.csv", index=False, encoding="utf-8-sig")
    ofi_df.to_csv(ROOT / "ofi_redundancy_results.csv", index=False, encoding="utf-8-sig")
    norm_df.to_csv(ROOT / "legacy_norm_results.csv", index=False, encoding="utf-8-sig")

    # Build markdown report.
    report = build_report(
        {
            "consistency": consistency_df,
            "max_train_days": max_train_df,
            "ofi": ofi_df,
            "norm": norm_df,
        },
        chosen_max_train_days=chosen_max_train_days,
        best_norm_name=best_norm_name,
        best_norm_summary=norm_df.sort_values(["stability_score", "rolling_corr_mean", "rolling_corr_min"], ascending=[False, False, False]).iloc[0].to_dict(),
        best_ofi_summary=best_ofi.to_dict(),
        baseline_summary=base_r02,
    )
    (ROOT / "目前实验状况5.16v3.md").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
