import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "meow-master" / "meow-master"))

from experiment_runner import ExperimentRunner, SplitConfig


def main():
    runner = ExperimentRunner(r"D:\code\final\archive")
    split = SplitConfig(
        train_start=20230601,
        train_end=20231031,
        val_start=20231101,
        val_end=20231130,
        test_start=20231201,
        test_end=20231229,
    )
    summary, fold_rows, coef_payload = runner.run_formal_rolling_suite(
        split,
        specs=[
            {
                "experiment_id": "B8_ridge_legacy_plus_core",
                "type": "standard",
                "model": "ridge",
                "target_mode": "raw",
                "groups": ["legacy", "base", "lag", "roll", "cross"],
                "notes": "formal ridge legacy plus core",
                "collect_coefs": True,
            }
        ],
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
        max_train_days=4,
        max_val_days=2,
    )
    feature_names = [
        "ob_imb0",
        "ob_imb4",
        "ob_imb9",
        "trade_imb",
        "trade_imbema5",
        "lagret12",
        "spread",
        "mid_ret1_raw",
        "obi0",
        "obi4",
        "obi9",
        "trade_turnover_imb",
        "add_imb",
        "cxl_imb",
        "qty_add_imb",
        "qty_cxl_imb",
        "buy_vwad_gap",
        "sell_vwad_gap",
        "trade_activity",
        "order_pressure",
    ]
    coef = runner.summarize_feature_coefficients(
        coef_payload,
        feature_names,
        "B8_ridge_legacy_plus_core",
    )
    summary.to_csv(r"D:\code\final\b8_diag_summary.csv", index=False, encoding="utf-8-sig")
    coef.to_csv(r"D:\code\final\b8_legacy_coefficients.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
