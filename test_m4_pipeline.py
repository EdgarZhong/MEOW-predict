#!/usr/bin/env python3
"""
M4 回归测试：主链路切到 FeatureLoader

覆盖点：
1. `ExperimentRunner.run_with_groups()` 已默认走 FeatureLoader，而不是旧版内存构建链路。
2. `TabularTrainer` 注入 FeatureLoader 后，可以完成单 fold 执行。
3. `scheduler._fold_group_worker()` 能基于临时 HDF + FeatureStore 产物跑出结果。

测试策略：
- 不依赖仓库真实 `data/*.h5`，在临时目录里构造最小可运行的 HDF 数据。
- 先用 `FeatureStore.build()` 生成 stage artifact，再走 M4 后的新评测主链路。
- 与旧版 `FeatureBuilder` 直接构造的特征结果做对照，确认数值口径未漂移。
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from experiment_runner import ExperimentRunner, SplitConfig
from feat_engine import FeatureBuilder
from feature_loader import FeatureLoader
from feature_registry import _make_schema_probe_raw
from feature_store import FeatureStore
import scheduler as scheduler_module
from scheduler import FoldGroup, FoldMeta, ParallelScheduler, _fold_group_worker
from trainer import FoldData, TabularTrainer


def _expected_feature_columns(groups):
    """
    生成 M4 新链路下的期望特征列顺序。

    规则与 FeatureLoader 完全一致：
    - 用 registry 解析 group
    - 按 stage 拓扑序展开
    - 对重复列按首次出现去重
    """
    from feature_registry import registry

    resolved = registry.resolve_groups(groups)
    ordered = []
    seen = set()
    for stage_name in registry.topo_order():
        for col in resolved.get(stage_name, []):
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered


def _write_hdf_dataset(h5dir: Path, dates):
    """
    写出临时 HDF 数据集。

    这里复用 schema probe raw 的结构，只把 `date` 改成目标交易日，
    这样可以保证：
    - 列结构与真实数据一致
    - lag/roll/patch/ofi 等 builder 路径都能被覆盖
    """
    template = _make_schema_probe_raw()
    base = template[template["date"] == 20230601].copy()
    h5dir.mkdir(parents=True, exist_ok=True)
    for date in dates:
        frame = base.copy()
        frame.loc[:, "date"] = int(date)
        frame.to_hdf(h5dir / f"{date}.h5", key="data", mode="w")


def _build_old_selected_frame(runner: ExperimentRunner, train_dates, val_dates, groups):
    """
    用旧版 FeatureBuilder 手动构造一份对照结果。

    这份结果不经过 FeatureLoader，只用于验证 M4 切链后训练/验证指标没有偏移。
    """
    builder = FeatureBuilder()

    def _one_side(dates):
        x_parts = []
        y_parts = []
        for date in dates:
            raw = runner.loader.loadDate(int(date))
            xdf, ydf = builder.build(raw)
            x_parts.append(builder.select_groups(xdf, groups))
            y_parts.append(ydf)
        return (
            pd.concat(x_parts, ignore_index=True),
            pd.concat(y_parts, ignore_index=True),
        )

    xtrain, ytrain = _one_side(train_dates)
    xval, yval = _one_side(val_dates)
    return xtrain, ytrain, xval, yval


def run_tests():
    checks = []

    def ok(name, cond, detail=""):
        tag = "✓ PASS" if cond else "✗ FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  [{detail}]"
        checks.append((cond, line))

    with tempfile.TemporaryDirectory(prefix="m4-pipeline-test-") as tmp:
        root = Path(tmp)
        h5dir = root / "data"
        feature_dir = root / "features"
        dates = [20230601, 20230602, 20230605]
        groups = ["legacy", "norm_core"]

        _write_hdf_dataset(h5dir, dates)

        # 先生成 stage artifact，模拟 PE1 正常预构建后的运行环境。
        store = FeatureStore(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            storage_backend="pickle_fallback",
        )
        store.build(dates=dates)

        runner = ExperimentRunner(
            str(h5dir),
            feature_dir=str(feature_dir),
        )
        split = SplitConfig(
            train_start=20230601,
            train_end=20230602,
            val_start=20230605,
            val_end=20230605,
            test_start=20230605,
            test_end=20230605,
        )

        # ── T1: run_with_groups 走新 loader，结果与旧特征口径一致 ─────────
        new_result = runner.run_with_groups(
            split_config=split,
            model_name="ridge",
            feature_groups=groups,
        )
        xtrain_old, ytrain_old, xval_old, yval_old = _build_old_selected_frame(
            runner,
            train_dates=[20230601, 20230602],
            val_dates=[20230605],
            groups=groups,
        )
        expected_feature_cols = _expected_feature_columns(groups)
        xtrain_old = xtrain_old.loc[:, ~xtrain_old.columns.duplicated()].copy()
        xval_old = xval_old.loc[:, ~xval_old.columns.duplicated()].copy()
        xtrain_old = xtrain_old[["date", "symbol", "interval"] + expected_feature_cols].copy()
        xval_old = xval_old[["date", "symbol", "interval"] + expected_feature_cols].copy()
        old_result = runner.run_on_features(
            xtrain_old,
            ytrain_old,
            xval_old,
            yval_old,
            model_name="ridge",
        )
        ok(
            "T1  新链路 feature_cols 数量与旧链路一致",
            len(new_result["feature_cols"]) == len(old_result["feature_cols"]),
            f"new={len(new_result['feature_cols'])}, old={len(old_result['feature_cols'])}",
        )
        ok(
            "T1  新链路 val 预测与旧链路 allclose",
            np.allclose(
                np.asarray(new_result["pred_val"], dtype=np.float32),
                np.asarray(old_result["pred_val"], dtype=np.float32),
                atol=1e-6,
                rtol=1e-6,
            ),
        )
        ok(
            "T1  新链路 val_corr 与旧链路一致",
            abs(float(new_result["val_metrics"]["corr"]) - float(old_result["val_metrics"]["corr"])) < 1e-8,
            f"new={new_result['val_metrics']['corr']}, old={old_result['val_metrics']['corr']}",
        )

        # ── T2: TabularTrainer 显式注入 loader 后可执行单 fold ────────────
        feature_loader = FeatureLoader(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            storage_backend="pickle_fallback",
        )
        trainer = TabularTrainer(
            {
                "experiment_id": "R02_ridge_legacy_plus_norm_core",
                "type": "standard",
                "model": "ridge",
                "target_mode": "raw",
                "groups": groups,
                "notes": "m4 trainer smoke",
            },
            runner,
            feature_loader,
        )
        fold_result = trainer.run_fold(
            FoldData(
                profile_name="m4_smoke",
                fold_id=0,
                train_dates=(20230601, 20230602),
                val_dates=(20230605,),
            )
        )
        ok("T2  TabularTrainer 返回 ok 状态", fold_result.status == "ok", fold_result.error_msg)
        ok("T2  TabularTrainer 产出有限 val_corr", np.isfinite(fold_result.val_corr), str(fold_result.val_corr))

        # ── T3: scheduler worker 能基于 FeatureLoader 主链路返回结果 ───────
        rows = _fold_group_worker(
            (
                str(h5dir),
                str(feature_dir),
                FoldGroup(
                    group_id="m4_worker_g0",
                    fold_metas=[
                        FoldMeta(
                            profile_name="m4_worker",
                            fold_id=0,
                            train_dates=(20230601, 20230602),
                            val_dates=(20230605,),
                        )
                    ],
                ),
                [
                    {
                        "experiment_id": "R02_ridge_legacy_plus_norm_core",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": groups,
                        "notes": "m4 worker smoke",
                    }
                ],
                frozenset(),
            )
        )
        ok("T3  worker 返回 1 条结果", len(rows) == 1, f"实际={len(rows)}")
        ok("T3  worker 结果状态为 ok", rows[0].get("status") == "ok", str(rows[0]))
        ok(
            "T3  worker experiment_id 正确",
            rows[0].get("experiment_id") == "R02_ridge_legacy_plus_norm_core",
            str(rows[0].get("experiment_id")),
        )

        # ── T4: ParallelScheduler.run 能覆盖 future -> group_id 映射 ──────
        class _FakeFuture:
            """极简 Future 桩，避免测试里真的拉起子进程。"""

            def __init__(self, rows):
                self._rows = rows

            def result(self):
                return self._rows

        class _FakeExecutor:
            """同步执行的执行器桩，只验证主进程编排逻辑。"""

            def __init__(self, max_workers):
                self.max_workers = max_workers
                executor_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, args):
                group = args[2]
                rows = [
                    {
                        "profile_name": group.fold_metas[0].profile_name,
                        "fold_id": group.fold_metas[0].fold_id,
                        "experiment_id": "R02_ridge_legacy_plus_norm_core",
                        "status": "ok",
                    }
                ]
                return _FakeFuture(rows)

        def _fake_as_completed(futures):
            """保持 as_completed 接口形状，按提交顺序直接返回。"""

            return list(futures)

        def _fake_build_fold_groups(profile_name, folds):
            """用固定 group 覆盖 run() 主路径，避免依赖真实 fold 构造。"""

            if profile_name == "m4_parallel":
                return [
                    FoldGroup(
                        group_id="m4_parallel_g0",
                        fold_metas=[
                            FoldMeta(
                                profile_name=profile_name,
                                fold_id=0,
                                train_dates=(20230601, 20230602),
                                val_dates=(20230605,),
                            )
                        ],
                    )
                ]
            return [
                FoldGroup(
                    group_id=f"{profile_name}_g0",
                    fold_metas=[
                        FoldMeta(
                            profile_name=profile_name,
                            fold_id=0,
                            train_dates=(20230601, 20230602),
                            val_dates=(20230605,),
                        )
                    ],
                )
            ]

        old_executor = scheduler_module.ProcessPoolExecutor
        old_as_completed = scheduler_module.as_completed
        old_build_groups = scheduler_module._build_fold_groups
        executor_workers = []
        try:
            scheduler_module.ProcessPoolExecutor = _FakeExecutor
            scheduler_module.as_completed = _fake_as_completed
            scheduler_module._build_fold_groups = _fake_build_fold_groups

            scheduler = ParallelScheduler(
                str(h5dir),
                feature_dir=str(feature_dir),
                n_workers=4,
            )
            scheduler.set_output_path(str(root / "fold_metrics.csv"))
            scheduler_df = scheduler.run(
                profiles_with_folds=[
                    (SimpleNamespace(profile_name="m4_parallel"), ["placeholder_fold"]),
                    (SimpleNamespace(profile_name="long_40d_5d"), ["placeholder_fold"]),
                ],
                specs=[
                    {
                        "experiment_id": "R02_ridge_legacy_plus_norm_core",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": groups,
                        "notes": "m4 scheduler smoke",
                    }
                ],
                resume=False,
            )
        finally:
            scheduler_module.ProcessPoolExecutor = old_executor
            scheduler_module.as_completed = old_as_completed
            scheduler_module._build_fold_groups = old_build_groups

        ok("T4  ParallelScheduler.run 返回 2 条结果", len(scheduler_df) == 2, f"实际={len(scheduler_df)}")
        ok(
            "T4  ParallelScheduler.run 保留 light/heavy 两类 profile",
            set(scheduler_df["profile_name"].tolist()) == {"m4_parallel", "long_40d_5d"},
            str(scheduler_df[["profile_name", "fold_id", "experiment_id"]].to_dict("records")),
        )
        ok(
            "T4  heavy 批次并发硬限制为 2",
            executor_workers == [4, 2],
            f"实际 executor workers={executor_workers}",
        )

    print("\n" + "=" * 60)
    for _, line in checks:
        print(line)
    print("=" * 60)

    all_pass = all(cond for cond, _ in checks)
    if all_pass:
        print("  全部通过 ✓")
    else:
        fail_count = sum(1 for cond, _ in checks if not cond)
        print(f"  {fail_count} 项失败 ✗")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
