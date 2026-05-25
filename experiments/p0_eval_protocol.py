"""
P0：建立统一评测基准

目标：
  1. 在新 Rolling Evaluation Protocol 下重新评测所有历史实验（R/B/O/T/C 系列）
  2. 以 R02_ridge_legacy_plus_norm_core 为 baseline，输出跨 profile 的 leaderboard
  3. 可选加入 11月 review holdout 和 12月 final holdout

运行方式：
  # 日常筛选（默认）：short + long 快车道，~10-15 min
  cd MEOW--predict
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite daily

  # 快速验证（每个 profile 只跑 2 个 fold，只跑 Ridge 系列）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

  # Ridge 全四 profile 重建基线
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge

  # 全部历史实验重新评测（含 O/T/C 系列，耗时较长）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite full

  # 指定特定 profile + 含 review holdout
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge \\
    --profiles short_8d_2d medium_20d_5d --include-review-holdout

  # 完整协议含 final holdout（慎用，少跑 final）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite full \\
    --include-review-holdout --include-final-holdout

输出目录结构：
  results/eval_protocol/<run_id>/
    config.json
    fold_manifest.csv
    fold_metrics.csv
    profile_summary.csv
    leaderboard.csv
    review_holdout.csv    （如果 --include-review-holdout）
    final_holdout.csv     （如果 --include-final-holdout）
"""

import argparse
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from eval_protocol import (
    EvaluationProtocolRunner,
    ROLLING_PROFILES,
    ALL_SPECS,
    RIDGE_SPECS,
    BASELINE_ID,
)
from experiment_runner import ExperimentRunner


# ================================================================== #
# 默认配置
# ================================================================== #

DATA_DIR = str(PROJ_ROOT / "data")
OUTPUT_DIR = str(PROJ_ROOT / "results" / "eval_protocol")
FEATURE_DIR = str(PROJ_ROOT / "data" / "features")

# 第一层 rolling 区间（内部选模型主依据）
ROLLING_START = 20230601
ROLLING_END = 20231031

# 第二层：11月 review holdout
REVIEW_TRAIN_START = 20230601
REVIEW_TRAIN_END = 20231031
REVIEW_HOLDOUT_START = 20231101
REVIEW_HOLDOUT_END = 20231130

# 第三层：12月 final holdout（尽量少跑）
FINAL_TRAIN_START = 20230601
FINAL_TRAIN_END = 20231130
FINAL_HOLDOUT_START = 20231201
FINAL_HOLDOUT_END = 20231229


# ================================================================== #
# CLI
# ================================================================== #

def build_arg_parser():
    parser = argparse.ArgumentParser(description="P0 Rolling Evaluation Protocol")
    parser.add_argument(
        "--suite",
        type=str,
        default="daily",
        choices=["quick", "daily", "ridge", "full"],
        help="daily=日常筛选(short+long 快车道, 默认); quick=2折Ridge调试; "
             "ridge=Ridge 全四 profile 重建基线; full=全部历史实验",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="指定 profile 名称列表覆盖 suite 默认（如单独 ad-hoc 跑 medium_20d_5d / expanding_40d_5d）",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="限制每个 profile 的 fold 数（调试用，默认不限制）",
    )
    parser.add_argument(
        "--include-review-holdout",
        action="store_true",
        help="同时运行 11月 review holdout",
    )
    parser.add_argument(
        "--include-final-holdout",
        action="store_true",
        help="同时运行 12月 final holdout（谨慎使用）",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="并发 worker 进程数（默认 4，设为 1 退回串行模式；16 GB Mac 不建议超过 4）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过已完成的 job（需与上次相同的 --run-id）",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="固定 run_id（resume 时必须与上次一致）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录（默认 results/eval_protocol）",
    )
    parser.add_argument(
        "--h5dir",
        type=str,
        default=DATA_DIR,
        help="数据目录",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=FEATURE_DIR,
        help="特征缓存目录（PE1/M4 起评测主链路依赖该目录）",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    # profile 名 → 对象映射，供 daily 选取与 --profiles 覆盖
    profile_map = {p.profile_name: p for p in ROLLING_PROFILES}
    daily_profiles = [profile_map[n] for n in ("short_8d_2d", "long_40d_5d") if n in profile_map]

    # 选择要运行的 specs 与 profiles
    if args.suite == "quick":
        specs = RIDGE_SPECS[:3]           # R00/R01/R02，快速验证
        max_folds = args.max_folds or 2   # 默认只跑 2 折
        selected_profiles = ROLLING_PROFILES[:1]  # 只跑 short profile
        print("[P0] 快速调试模式：R00-R02，short_8d_2d profile，2 folds")
    elif args.suite == "daily":
        specs = RIDGE_SPECS               # 日常筛选默认 Ridge 系列
        max_folds = args.max_folds        # None = 全部 fold
        selected_profiles = daily_profiles  # 快车道：short + long（medium/expanding 不进日常）
        print("[P0] 日常筛选 suite：short + long 快车道（medium 移出日常、expanding 只在关口跑）")
    elif args.suite == "ridge":
        specs = RIDGE_SPECS               # R00-R04 完整 Ridge 系列
        max_folds = args.max_folds        # None = 全部 fold
        selected_profiles = ROLLING_PROFILES
        print("[P0] Ridge 全 profile 重建：R00-R04，四个 profiles，全量 folds")
    else:  # full
        specs = ALL_SPECS
        max_folds = args.max_folds
        selected_profiles = ROLLING_PROFILES
        print(f"[P0] 完整评测：{len(specs)} 个历史实验，四个 profiles")

    # 按用户指定过滤 profile
    if args.profiles:
        profile_map = {p.profile_name: p for p in ROLLING_PROFILES}
        selected_profiles = [profile_map[name] for name in args.profiles if name in profile_map]
        if not selected_profiles:
            print(f"[P0] 警告：指定的 profiles {args.profiles} 无效，使用全部 profiles")
            selected_profiles = ROLLING_PROFILES

    print(f"\n[P0] 数据目录: {args.h5dir}")
    print(f"[P0] 特征目录: {args.feature_dir}")
    print(f"[P0] 输出目录: {args.output_dir}")
    print(f"[P0] rolling 区间: {ROLLING_START} ~ {ROLLING_END}")
    print(f"[P0] 实验数: {len(specs)}, profiles: {[p.profile_name for p in selected_profiles]}")
    print(f"[P0] max_folds: {max_folds or '全量'}")
    print(f"[P0] n_workers: {args.n_workers}{'（串行）' if args.n_workers == 1 else '（并行）'}")
    if args.resume:
        print(f"[P0] resume 模式：跳过已完成 job")

    runner = ExperimentRunner(args.h5dir, feature_dir=args.feature_dir)
    protocol = EvaluationProtocolRunner(runner)

    result = protocol.run_full_protocol(
        rolling_start=ROLLING_START,
        rolling_end=ROLLING_END,
        specs=specs,
        profiles=selected_profiles,
        max_folds=max_folds,
        include_review_holdout=args.include_review_holdout,
        review_train_start=REVIEW_TRAIN_START,
        review_train_end=REVIEW_TRAIN_END,
        review_holdout_start=REVIEW_HOLDOUT_START,
        review_holdout_end=REVIEW_HOLDOUT_END,
        include_final_holdout=args.include_final_holdout,
        final_train_start=FINAL_TRAIN_START,
        final_train_end=FINAL_TRAIN_END,
        final_holdout_start=FINAL_HOLDOUT_START,
        final_holdout_end=FINAL_HOLDOUT_END,
        baseline_id=BASELINE_ID,
        n_workers=args.n_workers,
        resume=args.resume,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    # 打印关键结果
    lb = result["leaderboard"]
    if not lb.empty:
        display_cols = [
            "experiment_id", "protocol_corr_mean", "protocol_stability_score",
            "protocol_daily_ic_mean", "protocol_daily_ic_ir",
            "protocol_corr_min", "protocol_positive_fold_rate",
            "baseline_delta_corr", "decision",
        ]
        display_cols = [c for c in display_cols if c in lb.columns]
        print("\n" + "=" * 80)
        print("Leaderboard（按 protocol_corr_mean 降序；stability / 每日 IC-IR 为并排守门指标）")
        print("=" * 80)
        print(lb[display_cols].to_string(index=False))
    else:
        print("\n[P0] 警告：leaderboard 为空，请检查数据和配置")

    print(f"\n[P0] 完成。run_id={result['run_id']}")


if __name__ == "__main__":
    main()
