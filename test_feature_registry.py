#!/usr/bin/env python3
"""
M1 回归测试：FeatureRegistry

验证目标：
1. DAG 拓扑序满足依赖先于下游。
2. downstream 闭包正确。
3. 新版 resolve_groups 与旧版 FeatureBuilder.select_groups 的列映射一致。
4. 9 个 stage 合并后的总列集合与旧管道 build() 的输出列集合一致。

说明：
- 这是一个独立脚本，沿用仓库现有 `test_cache_fix.py` 的风格。
- 不依赖真实 H5 数据，使用 registry 自带的 schema probe raw。
"""

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))


from feat_engine import FeatureBuilder
from feature_registry import META_COLS, _make_schema_probe_raw, registry


def _flatten_resolved_columns(groups):
    """
    将 `{stage: [cols...]}` 按 registry 的拓扑序打平成单列列表。

    这里故意保留顺序去重，便于和旧版 select_groups 的行为口径一致。
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

    raw = _make_schema_probe_raw()
    builder = FeatureBuilder()
    xdf, _ = builder.build(raw)
    feature_cols = [col for col in xdf.columns if col not in META_COLS]

    # ── T1: 拓扑序验证 ────────────────────────────────────────────────
    topo = registry.topo_order()
    pos = {name: idx for idx, name in enumerate(topo)}
    ok("T1  注册了 9 个活跃 stage", len(topo) == 9, f"实际={len(topo)}")

    expected_deps = {
        "lag": ["base"],
        "roll": ["base"],
        "patch": ["base"],
        "trade_impact": ["base", "ofi"],
        "cross": ["base", "ofi", "trade_impact"],
        "conditional_momentum": ["base", "ofi", "trade_impact"],
        "regime": ["base"],
    }
    dep_ok = True
    dep_detail = []
    for stage_name, deps in expected_deps.items():
        for dep in deps:
            if pos[dep] >= pos[stage_name]:
                dep_ok = False
                dep_detail.append(f"{dep}->{stage_name}")
    ok("T1  所有依赖都排在下游之前", dep_ok, ",".join(dep_detail) if dep_detail else "")

    # ── T2: downstream 闭包验证 ──────────────────────────────────────
    ofi_downstream = registry.downstream("ofi")
    ok(
        "T2  ofi 的下游闭包正确",
        ofi_downstream == {"trade_impact", "cross", "conditional_momentum"},
        f"实际={sorted(ofi_downstream)}",
    )
    base_downstream = registry.downstream("base")
    ok(
        "T2  base 的下游覆盖其所有衍生 stage",
        base_downstream == {"lag", "roll", "patch", "trade_impact", "cross", "conditional_momentum", "regime"},
        f"实际={sorted(base_downstream)}",
    )

    # ── T3: group 列映射兼容旧版 select_groups ───────────────────────
    group_cases = [
        ["legacy"],
        ["base"],
        ["legacy", "norm_core"],
        ["lag_short", "lag_mid", "lag_long"],
        ["roll_short", "roll_mid", "roll_long"],
        ["patch_summary"],
        ["ofi"],
        ["ofi_raw"],
        ["ofi_dynamic"],
        ["ofi_rank"],
        ["trade_impact"],
        ["trade_impact_dyn"],
        ["trade_impact_interaction"],
        ["trade_impact_safe"],
        ["conditional_momentum"],
        ["conditional_momentum_interaction"],
        ["cross_z"],
        ["cross_rank"],
        ["cross"],
        ["regime"],
    ]
    for groups in group_cases:
        old_cols = [col for col in builder.select_groups(xdf, groups).columns if col not in META_COLS]
        new_cols = _flatten_resolved_columns(groups)
        old_set = set(old_cols)
        new_set = set(new_cols)
        ok(
            f"T3  group={'+'.join(groups)} 列集合兼容",
            old_set == new_set,
            f"old={len(old_set)}, new={len(new_set)}",
        )

    # ── T4: 9 个 stage 的总输出列集合与旧管道一致 ───────────────────
    stage_outputs = {}
    sorted_raw = raw.sort_values(META_COLS, kind="mergesort").reset_index(drop=True)
    for stage_name in registry.topo_order():
        builder_fn = registry.get_builder(stage_name)
        deps = {dep: stage_outputs[dep] for dep in registry.get_deps(stage_name)}
        stage_outputs[stage_name] = builder_fn(sorted_raw, **deps)
    merged_cols = []
    for stage_name in registry.topo_order():
        merged_cols.extend(stage_outputs[stage_name].columns.tolist())
    merged_set = set(merged_cols)
    feature_set = set(feature_cols)
    ok(
        "T4  新旧总特征列集合一致",
        merged_set == feature_set,
        f"old={len(feature_set)}, new={len(merged_set)}",
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
