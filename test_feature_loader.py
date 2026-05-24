#!/usr/bin/env python3
"""
M3 回归测试：FeatureLoader

覆盖点：
1. Loader 能从 FeatureStore 产物中按 group 正确加载多天数据。
2. 新管道加载结果与旧版 FeatureBuilder.select_groups 口径一致。
3. Loader 会记录最近一次 load 的 resolved_columns / stages_used 信息。
4. stage 文件被打乱行顺序时，Loader 能在加载阶段直接报错。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from feat_engine import FeatureBuilder
from feature_loader import FeatureLoader
from feature_registry import _make_schema_probe_raw, registry
from feature_store import FeatureStore, write_feature_frame


class FakeLoader:
    """返回可重复的单日 raw DataFrame，用于替代真实 H5 IO。"""

    def __init__(self, h5dir: str):
        self.h5dir = h5dir
        self._template = _make_schema_probe_raw()

    def loadDate(self, date: int) -> pd.DataFrame:
        frame = self._template[self._template["date"] == 20230601].copy()
        frame.loc[:, "date"] = int(date)
        return frame


def _touch(path: Path) -> None:
    """创建一个空文件，用于模拟本地 H5 存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _build_old_pipeline_output(loader: FakeLoader, dates, groups):
    """
    用旧版 FeatureBuilder 构造基准结果。

    这里直接复用老管道，目的是验证：
    - 新的 FeatureLoader 没有改动 group 口径
    - 多天 concat 的结果顺序与旧实现一致
    """
    builder = FeatureBuilder()
    x_parts = []
    y_parts = []
    for date in dates:
        raw = loader.loadDate(int(date))
        xdf, ydf = builder.build(raw)
        x_parts.append(builder.select_groups(xdf, groups))
        y_parts.append(ydf)
    return (
        pd.concat(x_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def _expected_feature_columns(groups):
    """
    按 PE1 新口径生成期望列顺序。

    规则与 `FeatureLoader.load()` 一致：
    - 先按 registry.resolve_groups 解析
    - 再按 stage 拓扑序展开
    - 同一列只保留第一次出现
    """
    resolved = registry.resolve_groups(groups)
    ordered = []
    seen = set()
    for stage_name in registry.topo_order():
        for col in resolved.get(stage_name, []):
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered


def run_tests():
    checks = []

    def ok(name, cond, detail=""):
        tag = "✓ PASS" if cond else "✗ FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  [{detail}]"
        checks.append((cond, line))

    with tempfile.TemporaryDirectory(prefix="feature-loader-test-") as tmp:
        root = Path(tmp)
        h5dir = root / "data"
        feature_dir = root / "features"
        dates = [20230601, 20230602]
        groups = ["legacy", "norm_core", "ofi_raw"]

        for date in dates:
            _touch(h5dir / f"{date}.h5")

        store = FeatureStore(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            registry=registry,
            loader_cls=FakeLoader,
            storage_backend="pickle_fallback",
        )
        store.build(dates=dates)

        loader = FeatureLoader(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            registry=registry,
            loader_cls=FakeLoader,
            storage_backend="pickle_fallback",
        )

        # ── T1: 与旧管道输出一致 ─────────────────────────────────────
        new_xdf, new_ydf = loader.load(dates=dates, groups=groups)
        old_xdf, old_ydf = _build_old_pipeline_output(FakeLoader(str(h5dir)), dates, groups)
        expected_feature_cols = _expected_feature_columns(groups)
        old_xdf_unique = old_xdf.loc[:, ~old_xdf.columns.duplicated()].copy()
        old_xdf_aligned = old_xdf_unique[["date", "symbol", "interval"] + expected_feature_cols].copy()

        ok("T1  xdf 列顺序符合新管道拓扑口径", list(new_xdf.columns) == list(old_xdf_aligned.columns))
        ok("T1  ydf 列顺序与旧管道一致", list(new_ydf.columns) == list(old_ydf.columns))
        ok(
            "T1  xdf 数值与旧管道 allclose",
            np.allclose(
                new_xdf.drop(columns=["date", "symbol", "interval"]).to_numpy(dtype=np.float32),
                old_xdf_aligned.drop(columns=["date", "symbol", "interval"]).to_numpy(dtype=np.float32),
                equal_nan=True,
            ),
        )
        ok("T1  ydf 与旧管道完全一致", new_ydf.equals(old_ydf))

        # ── T2: 最近一次 load 的 resolved_columns 信息可追溯 ─────────
        info = loader.last_load_info()
        expected_resolved = [
            col for col in new_xdf.columns
            if col not in ["date", "symbol", "interval"]
        ]
        ok("T2  resolved_columns 已记录", info.get("resolved_columns") == expected_resolved)
        ok(
            "T2  stages_used 覆盖 base/ofi/cross",
            {"base", "ofi", "cross"}.issubset(set((info.get("stages_used") or {}).keys())),
            str(info.get("stages_used")),
        )

        # ── T3: stage 行顺序被打乱时会立即报错 ───────────────────────
        broken_stage = store.read_stage_frame("base", 20230601).iloc[::-1].copy()
        write_feature_frame(
            broken_stage,
            store.stage_file("base", 20230601),
            backend="pickle_fallback",
        )
        misaligned_error = ""
        try:
            loader.load(dates=[20230601], groups=["base"])
        except ValueError as exc:
            misaligned_error = str(exc)
        ok(
            "T3  错位 stage 会触发对齐异常",
            "行顺序不匹配" in misaligned_error,
            misaligned_error,
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
