import argparse
import os
from pathlib import Path
from log import log
from dl import MeowDataLoader
from feat_legacy import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from experiment_runner import ExperimentRunner, SplitConfig


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


def _default_h5dir():
    candidates = [
        Path(__file__).resolve().parents[1] / "data",
        Path.cwd() / "data",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MEOW unified entry point")
    parser.add_argument("--h5dir", type=str, default=_default_h5dir())
    parser.add_argument(
        "--mode",
        type=str,
        default="legacy",
        choices=["legacy", "suite", "eval_protocol"],
        help="legacy=原始流程; suite=实验套件; eval_protocol=新评测协议",
    )
    parser.add_argument("--suite", type=str, default=None, choices=[
        "stage0", "stage0_quick", "stage0_roll", "formal_backbone", "ridge_enhance",
        "restricted_fusion", "train_window_sensitivity", "train_window_sensitivity_quick",
        "ofi_audit", "trade_impact_audit", "conditional_momentum_audit",
        "stage1", "stage2", "ablation", "v2", "v31", "v31_quick", "v31_roll",
        # 新评测协议 suite
        "eval_protocol_quick", "eval_protocol_ridge", "eval_protocol_full",
    ])
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "elasticnet", "tree", "gbdt", "histgb", "lgbm", "mlp"])
    parser.add_argument("--target-mode", type=str, default="raw", choices=["raw", "date_demean", "interval_demean", "interval_residual"])
    parser.add_argument("--feature-groups", nargs="*", default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="eval_protocol 输出目录")
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--val-start", type=int, default=20231101)
    parser.add_argument("--val-end", type=int, default=20231130)
    parser.add_argument("--test-start", type=int, default=20231201)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-val-days", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None, help="限制每个 profile 的 fold 数（调试用）")
    parser.add_argument("--profiles", nargs="*", default=None, help="指定 rolling profile 名称")
    parser.add_argument("--include-review-holdout", action="store_true", help="运行 11月 review holdout")
    parser.add_argument("--include-final-holdout", action="store_true", help="运行 12月 final holdout（谨慎）")
    parser.add_argument("--fit-start", type=int, default=20230601)
    parser.add_argument("--fit-end", type=int, default=20231130)
    parser.add_argument("--eval-start", type=int, default=20231201)
    parser.add_argument("--eval-end", type=int, default=20231229)
    return parser


def _run_eval_protocol(args):
    """处理 eval_protocol 模式（新三层评测协议）"""
    from eval_protocol import (
        EvaluationProtocolRunner, ROLLING_PROFILES,
        ALL_SPECS, RIDGE_SPECS, BASELINE_ID,
    )

    suite = (args.suite or "eval_protocol_ridge").replace("eval_protocol_", "")
    if suite == "quick":
        specs = RIDGE_SPECS[:3]
        max_folds = args.max_folds or 2
        selected_profiles = ROLLING_PROFILES[:1]
    elif suite == "ridge":
        specs = RIDGE_SPECS
        max_folds = args.max_folds
        selected_profiles = ROLLING_PROFILES
    else:  # full
        specs = ALL_SPECS
        max_folds = args.max_folds
        selected_profiles = ROLLING_PROFILES

    if args.profiles:
        profile_map = {p.profile_name: p for p in ROLLING_PROFILES}
        override = [profile_map[n] for n in args.profiles if n in profile_map]
        if override:
            selected_profiles = override

    output_dir = args.output_dir or str(Path(__file__).resolve().parents[1] / "results" / "eval_protocol")

    runner = ExperimentRunner(args.h5dir)
    protocol = EvaluationProtocolRunner(runner)

    result = protocol.run_full_protocol(
        rolling_start=args.train_start,
        rolling_end=args.train_end,
        specs=specs,
        profiles=selected_profiles,
        max_folds=max_folds,
        include_review_holdout=args.include_review_holdout,
        review_train_start=args.train_start,
        review_train_end=args.train_end,
        review_holdout_start=args.val_start,
        review_holdout_end=args.val_end,
        include_final_holdout=args.include_final_holdout,
        final_train_start=args.train_start,
        final_train_end=args.val_end,
        final_holdout_start=args.test_start,
        final_holdout_end=args.test_end,
        baseline_id=BASELINE_ID,
        output_dir=output_dir,
    )

    lb = result["leaderboard"]
    if not lb.empty:
        display_cols = [c for c in [
            "experiment_id", "protocol_corr_mean", "protocol_stability_score",
            "protocol_corr_min", "protocol_positive_fold_rate",
            "baseline_delta_corr", "decision",
        ] if c in lb.columns]
        log.inf("\n" + "=" * 80)
        log.inf("Leaderboard（按 protocol_stability_score 降序）")
        log.inf("=" * 80)
        log.inf("\n" + lb[display_cols].to_string(index=False))

    if args.output_csv:
        lb.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        log.inf(f"Leaderboard saved to {args.output_csv}")


def main():
    args = build_arg_parser().parse_args()

    # 新评测协议模式
    if args.mode == "eval_protocol" or (args.suite and args.suite.startswith("eval_protocol")):
        _run_eval_protocol(args)
        return

    if args.mode == "suite" or args.suite:
        split_config = SplitConfig(
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        runner = ExperimentRunner(args.h5dir)
        df = runner.run_suite(
            split_config=split_config,
            suite_name=args.suite or "ablation",
            max_train_days=args.max_train_days,
            max_val_days=args.max_val_days,
        )
        log.inf("\n" + df.to_string(index=False))
        if args.output_csv:
            df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
            log.inf(f"Saved suite results to {args.output_csv}")
        return

    engine = MeowEngine(h5dir=args.h5dir, cacheDir=None)
    engine.fit(args.fit_start, args.fit_end)
    engine.eval(args.eval_start, args.eval_end)


if __name__ == "__main__":
    main()
