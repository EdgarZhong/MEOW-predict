import gc
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_516v3_restricted import (  # noqa: E402
    META_COLS,
    TOP_FEATURES,
    as_range_text,
    augment_normalization_groups,
    select_train_days,
    summarize_rows,
)

sys.path.insert(0, str(ROOT / "meow-master" / "meow-master"))
from experiment_runner import ExperimentRunner, SplitConfig  # noqa: E402


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
    folds = runner._build_rolling_folds(
        split_config,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
    )

    max_train_days = 4
    initial_variants = [
        ("N1_intraday_zscore", ["intraday"], "intraday_zscore_top_features"),
        ("N2_stock_rolling_zscore", ["rolling"], "stock_rolling_zscore_top_features"),
        ("N3_cross_section_rank", ["cross_rank"], "cross_section_rank_top_features"),
        ("N5_all_top_feature_normalization_groups", ["intraday", "rolling", "cross_rank"], "all top feature normalization groups"),
    ]
    variant_fold_rows = {name: [] for name, _, _ in initial_variants}

    for fold in folds:
        train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
        val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
        train_x_base, train_dates_subset = select_train_days(train_full, fold.train_dates, max_train_days)
        train_y, _ = select_train_days(ytrain_full, fold.train_dates, max_train_days)
        train_x_base = runner._filter_features(train_x_base, ["legacy", "norm_core"])
        train_x_base = train_x_base.loc[:, ~train_x_base.columns.duplicated()].copy()
        val_x_base = runner._filter_features(
            val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True),
            ["legacy", "norm_core"],
        )
        val_x_base = val_x_base.loc[:, ~val_x_base.columns.duplicated()].copy()
        val_y = yval_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)

        for name, norm_groups, _ in initial_variants:
            train_x, val_x = augment_normalization_groups(train_x_base, val_x_base, norm_groups)
            result = runner.run_on_features(train_x, train_y, val_x, val_y, model_name="ridge", target_mode="raw")
            variant_fold_rows[name].append(
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
            del train_x, val_x, result
            gc.collect()
        del train_full, ytrain_full, val_full, yval_full, train_x_base, train_y, val_x_base, val_y
        gc.collect()

    norm_rows = []
    norm_base = {}
    for name, norm_groups, reason in initial_variants:
        summary = summarize_rows(
            variant_fold_rows[name],
            name,
            extra={
                "feature_set": json.dumps({"groups": ["legacy", "norm_core"], "normalization": norm_groups}, ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": max_train_days,
                "embargo_setting": 1,
                "是否保留": "待定",
                "原因": reason,
            },
        )
        norm_rows.append(summary)
        norm_base[name] = summary

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

    for name, groups, reason in [
        ("N4_best_two_normalization_groups", best_two_groups, "best two normalization groups selected from N1-N3"),
    ]:
        fold_metrics = []
        for fold in folds:
            train_full, ytrain_full = runner.load_feature_split(fold.train_dates, max_days=None, groups=None)
            val_full, yval_full = runner.load_feature_split(fold.val_dates, max_days=None, groups=None)
            train_x, train_dates_subset = select_train_days(train_full, fold.train_dates, max_train_days)
            train_y, _ = select_train_days(ytrain_full, fold.train_dates, max_train_days)
            train_x = runner._filter_features(train_x, ["legacy", "norm_core"])
            val_x = runner._filter_features(val_full.copy().sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True), ["legacy", "norm_core"])
            train_x, val_x = augment_normalization_groups(train_x, val_x, groups)
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
                "feature_set": json.dumps({"groups": ["legacy", "norm_core"], "normalization": groups}, ensure_ascii=False),
                "model_type": "ridge",
                "target_type": "raw",
                "max_train_days": max_train_days,
                "embargo_setting": 1,
                "是否保留": "待定",
                "原因": reason,
            },
        )
        norm_rows.append(summary)

    norm_df = pd.DataFrame(norm_rows)
    norm_df["是否保留"] = "待定"
    norm_df.loc[
        (norm_df["rolling_corr_mean"] > 0.047976) & (norm_df["stability_score"] >= 0.037494),
        "是否保留",
    ] = "是"

    norm_df.to_csv(ROOT / "legacy_norm_results.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 5.16 V3 归一化补跑")
    lines.append("")
    lines.append(f"当前固定 `max_train_days={max_train_days}`，用于复核 top legacy 精细归一化。")
    lines.append("")
    lines.append("| experiment_name | rolling_corr_mean | rolling_corr_std | rolling_corr_min | stability_score | 是否保留 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in norm_df.iterrows():
        lines.append(
            f"| `{row['experiment_name']}` | {row['rolling_corr_mean']:.6f} | {row['rolling_corr_std']:.6f} | {row['rolling_corr_min']:.6f} | {row['stability_score']:.6f} | {row['是否保留']} |"
        )
    lines.append("")
    lines.append(f"当前最优归一化组是 `{best_norm_name}`。")
    (ROOT / "目前实验状况5.16v3_norm_only.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
