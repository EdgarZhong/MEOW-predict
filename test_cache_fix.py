#!/usr/bin/env python3
"""
最小缓存行为测试 — 验证并行 OOM 修复

T1  fold1 首次加载：loadDate 调用 8 次，split_cache=1，daily_cache=8
T2  折内 spec2/3 复用：loadDate 再调用 0 次（split_cache 命中）
T3  折结束清空 split_cache：split_cache=0，daily_cache 仍为 8
T4  fold2 加载（3 天重叠）：loadDate 只调用 5 次（重叠天命中 daily_cache）
T5  fold2 结束：split_cache=1，daily_cache=11

不需要真实 H5 数据，全部 mock。
"""
import gc
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))


# ── 假数据工厂 ──────────────────────────────────────────────────────────

def _fake_day(date, n=50):
    """返回 (xdf, ydf)，列结构与 ExperimentRunner.load_split 期望一致"""
    rng = np.random.default_rng(int(date) % (2**31))
    xdf = pd.DataFrame({
        "date": [date] * n,
        "symbol": ["SYM"] * n,
        "interval": range(n),
        "feat_a": rng.standard_normal(n).astype(np.float32),
        "feat_b": rng.standard_normal(n).astype(np.float32),
    })
    ydf = pd.DataFrame({
        "date": [date] * n,
        "symbol": ["SYM"] * n,
        "interval": range(n),
        "fret12": rng.standard_normal(n).astype(np.float32),
    })
    return xdf, ydf


# ── 构建带 mock 的 runner ───────────────────────────────────────────────

def make_runner():
    """
    patch MeowDataLoader / FeatureBuilder / Calendar，
    返回 (runner, io_calls)。
    io_calls 记录每次真实 loadDate 被调用的 date。
    """
    io_calls = []

    def fake_load_date(date):
        io_calls.append(date)
        return {"_date": date}          # raw 只是 tag，传给 build

    def fake_build(raw):
        return _fake_day(raw["_date"])  # 返回 (xdf, ydf)

    with patch("experiment_runner.MeowDataLoader"), \
         patch("experiment_runner.FeatureBuilder"), \
         patch("experiment_runner.Calendar"):
        from experiment_runner import ExperimentRunner
        runner = ExperimentRunner("/fake/path")

    # with 块退出后 runner.loader / runner.builder 仍是 Mock 实例，可自由替换方法
    runner.loader.loadDate = fake_load_date
    runner.builder.build   = fake_build

    return runner, io_calls


# ── 测试主逻辑 ──────────────────────────────────────────────────────────

def run_tests():
    checks = []

    def ok(name, cond, detail=""):
        tag = "✓ PASS" if cond else "✗ FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  [{detail}]"
        checks.append((cond, line))

    runner, io_calls = make_runner()

    # 日期设计：fold1=[1..8]，fold2=[6..13]，重叠 [6,7,8]（3 天），新增 [9..13]（5 天）
    FOLD1 = tuple(range(1, 9))   # 8 天
    FOLD2 = tuple(range(6, 14))  # 8 天

    # ── Fold 1 首次加载（spec 1）──────────────────────────────────────
    print("\n[Fold 1 / spec 1]  首次加载 8 天数据")
    runner.load_split(FOLD1)

    ok("T1  loadDate 调用 8 次",
       len(io_calls) == 8, f"实际={len(io_calls)}")
    ok("T1  _split_cache = 1 条",
       len(runner._split_cache) == 1, f"实际={len(runner._split_cache)}")
    ok("T1  _daily_cache = 8 条",
       len(runner._daily_feature_cache) == 8, f"实际={len(runner._daily_feature_cache)}")

    # ── 折内 spec 2 / 3（相同 train_dates，应命中 split_cache）───────
    print("[Fold 1 / spec 2-3]  复用 split_cache，不应触发 IO")
    io_calls.clear()
    runner.load_split(FOLD1)   # spec 2
    runner.load_split(FOLD1)   # spec 3

    ok("T2  折内 spec2/3 loadDate 调用 0 次（split_cache 命中）",
       len(io_calls) == 0, f"实际={len(io_calls)}")

    # ── Fold 1 结束：OOM 修复 ─────────────────────────────────────────
    print("[Fold 1 结束]  清空 split_cache（OOM 修复）")
    runner._split_cache.clear()
    runner._raw_split_cache.clear()
    gc.collect()

    ok("T3  清空后 _split_cache = 0",
       len(runner._split_cache) == 0, f"实际={len(runner._split_cache)}")
    ok("T3  清空后 _daily_cache 仍 8 条（不受影响）",
       len(runner._daily_feature_cache) == 8, f"实际={len(runner._daily_feature_cache)}")

    # ── Fold 2：3 天重叠 + 5 天新日期 ────────────────────────────────
    print("[Fold 2 / spec 1]  8 天中 3 天重叠，只应 IO 5 次")
    io_calls.clear()
    runner.load_split(FOLD2)

    ok("T4  loadDate 调用 5 次（重叠 3 天命中 daily_cache）",
       len(io_calls) == 5, f"实际={len(io_calls)}")
    ok("T5  fold2 后 _split_cache = 1 条",
       len(runner._split_cache) == 1, f"实际={len(runner._split_cache)}")
    ok("T5  fold2 后 _daily_cache = 13 条（fold1 的 8 天 + 新增 5 天）",
       len(runner._daily_feature_cache) == 13, f"实际={len(runner._daily_feature_cache)}")

    # ── 打印汇总 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for _, line in checks:
        print(line)
    print("=" * 60)

    all_pass = all(c for c, _ in checks)
    if all_pass:
        print("  全部通过 ✓")
    else:
        fail_count = sum(1 for c, _ in checks if not c)
        print(f"  {fail_count} 项失败 ✗")

    return all_pass


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
