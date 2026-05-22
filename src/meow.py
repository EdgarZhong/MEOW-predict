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
        choices=["legacy", "suite"],
        help="legacy runs the original fit/eval flow; suite runs the experiment runner",
    )
    parser.add_argument("--suite", type=str, default=None, choices=["stage0", "stage0_quick", "stage0_roll", "formal_backbone", "ridge_enhance", "restricted_fusion", "train_window_sensitivity", "train_window_sensitivity_quick", "ofi_audit", "trade_impact_audit", "conditional_momentum_audit", "stage1", "stage2", "ablation", "v2", "v31", "v31_quick", "v31_roll"])
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "elasticnet", "tree", "gbdt", "histgb", "lgbm", "mlp"])
    parser.add_argument("--target-mode", type=str, default="raw", choices=["raw", "date_demean", "interval_demean", "interval_residual"])
    parser.add_argument("--feature-groups", nargs="*", default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--val-start", type=int, default=20231101)
    parser.add_argument("--val-end", type=int, default=20231130)
    parser.add_argument("--test-start", type=int, default=20231201)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-val-days", type=int, default=None)
    parser.add_argument("--fit-start", type=int, default=20230601)
    parser.add_argument("--fit-end", type=int, default=20231130)
    parser.add_argument("--eval-start", type=int, default=20231201)
    parser.add_argument("--eval-end", type=int, default=20231229)
    return parser


def main():
    args = build_arg_parser().parse_args()
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
