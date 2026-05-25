"""
正式提交桥接层

本模块的职责非常克制，只做两件事：

1. 把 `feature_registry` 里已经存在的正式特征 builder 串起来，
   支持“从原始 raw DataFrame 现算提交所需特征”。
2. 复用 `ExperimentRunner` 现有的训练 / 推理核心逻辑，
   让老师的 `meow.py` 提交通道和我们自己的实验 driver 共享同一套后端。

设计原则：
- 不引入新的特征公式；正式特征仍以 registry 为真相源。
- 不大改 experiment_runner；这里只做薄桥接。
- 不依赖 `data/features/` 持久化特征缓存；提交链必须可从原始数据现算。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiment_runner import DEFAULT_TARGET_WINSORIZE, ExperimentRunner
from feature_registry import META_COLS, TARGET_COL, FeatureRegistry, registry as default_registry
from feature_store import DEFAULT_FEATURE_DIR


# 当前正式提交默认沿用 R02 backbone。
# 后续若提交口径升级，这里只改一处即可，meow 包装层和实验闭环都会同步。
DEFAULT_SUBMISSION_GROUPS: Tuple[str, ...] = ("legacy", "norm_core")
DEFAULT_SUBMISSION_MODEL = "ridge"
DEFAULT_SUBMISSION_TARGET_MODE = "raw"


@dataclass(frozen=True)
class SubmissionSpec:
    """
    正式提交通道使用的最小规格。

    这里只保留会真正影响预测值的核心字段：
    - groups: 使用哪些正式特征组
    - model_name: 训练模型
    - target_mode: 训练目标口径
    """

    groups: Tuple[str, ...] = DEFAULT_SUBMISSION_GROUPS
    model_name: str = DEFAULT_SUBMISSION_MODEL
    target_mode: str = DEFAULT_SUBMISSION_TARGET_MODE


class SubmissionFeaturePipeline:
    """
    从原始 raw DataFrame 现算正式提交特征。

    这里故意不走 FeatureLoader / data/features：
    - 实验 driver 可以为了速度走磁盘缓存
    - 正式提交通道必须保证“拿到一份全新的 raw 数据也能现算”
    """

    def __init__(
        self,
        groups: Optional[Sequence[str]] = None,
        registry: FeatureRegistry = default_registry,
    ):
        self.groups = tuple(groups or DEFAULT_SUBMISSION_GROUPS)
        self.registry = registry
        # 记录最近一次构造时解析到的 stage / columns，供闭环核对使用。
        self._last_build_info: Dict[str, object] = {}

    def _sorted_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        统一提交链的原始排序口径。

        和训练/评测主链一致地按 `(date, symbol, interval)` 稳定排序，
        这样后续：
        - stage builder 的 lag / rolling 结果稳定
        - 特征与目标行对齐稳定
        """

        return (
            raw_df.copy()
            .sort_values(META_COLS, kind="mergesort")
            .reset_index(drop=True)
        )

    def _resolved_stage_columns(self, groups: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
        """把 group 名称解析成 `{stage_name: [columns...]}`，复用 registry 单点定义。"""
        return self.registry.resolve_groups(groups or self.groups)

    def _stage_closure(self, resolved_stage_columns: Dict[str, List[str]]) -> List[str]:
        """
        计算为本次 group 所必需的 stage 闭包。

        规则：
        - 最终需要的 stage：直接出现在 resolved_stage_columns 中的 stage
        - 同时要把这些 stage 的所有上游依赖一并纳入
        - 返回顺序仍以 registry 的稳定拓扑序为准，避免人工维护执行顺序
        """

        needed = set(resolved_stage_columns.keys())
        queue = deque(resolved_stage_columns.keys())
        while queue:
            stage_name = queue.popleft()
            for dep in self.registry.get_deps(stage_name):
                if dep in needed:
                    continue
                needed.add(dep)
                queue.append(dep)
        return [
            stage_name
            for stage_name in self.registry.topo_order(include_archived=False)
            if stage_name in needed
        ]

    def build_feature_frames(
        self,
        raw_df: pd.DataFrame,
        groups: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        从原始数据直接构造 `xdf / ydf`。

        返回口径和实验主链一致：
        - `xdf`: `date/symbol/interval + feature columns`
        - `ydf`: `date/symbol/interval + fret12`
        """

        resolved = self._resolved_stage_columns(groups=groups)
        stage_order = self._stage_closure(resolved)
        raw = self._sorted_raw(raw_df)
        x_parts: List[pd.DataFrame] = []
        y_parts: List[pd.DataFrame] = []
        resolved_columns_in_order: List[str] = [
            col
            for stage_name in stage_order
            if stage_name in resolved
            for col in resolved[stage_name]
        ]

        # 正式提交必须按“逐日现算”执行。
        # 原因：
        # 1. 当前很多 builder（如 EMA / rolling / pct_change）是按日内序列定义的
        # 2. 若把多天 raw 一次性送进去，会出现跨日串值
        # 3. 老师最终评测也会给我们新的原始数据，因此提交链应天然支持逐日重算
        for _, day_raw in raw.groupby("date", sort=True):
            day_raw = day_raw.reset_index(drop=True)
            built_outputs: Dict[str, pd.DataFrame] = {}
            for stage_name in stage_order:
                deps = {
                    dep: built_outputs[dep]
                    for dep in self.registry.get_deps(stage_name)
                }
                builder = self.registry.get_builder(stage_name)
                built_outputs[stage_name] = builder(day_raw, **deps)

            day_feature_parts: List[pd.DataFrame] = [day_raw[META_COLS].copy()]
            for stage_name in stage_order:
                if stage_name not in resolved:
                    continue
                day_feature_parts.append(
                    built_outputs[stage_name].loc[:, list(resolved[stage_name])].copy()
                )
            day_xdf = pd.concat(day_feature_parts, axis=1)
            day_xdf = day_xdf.loc[:, ~day_xdf.columns.duplicated()].copy()
            day_ydf = day_raw[META_COLS + [TARGET_COL]].copy()
            day_ydf[TARGET_COL] = (
                pd.to_numeric(day_ydf[TARGET_COL], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float32)
            )
            x_parts.append(day_xdf)
            y_parts.append(day_ydf)

        xdf = pd.concat(x_parts, ignore_index=True)
        ydf = pd.concat(y_parts, ignore_index=True)

        self._last_build_info = {
            "groups": list(groups or self.groups),
            "stage_order": list(stage_order),
            "resolved_stage_columns": {
                stage_name: list(columns)
                for stage_name, columns in resolved.items()
            },
            "resolved_columns": list(resolved_columns_in_order),
        }
        return xdf, ydf

    def feature_names(self) -> List[str]:
        """
        返回当前正式提交 spec 的特征列名。

        这里直接复用 registry 的 group 解析结果，因此老师包装层看到的列集合，
        与实验链对该 group 的理解天然保持一致。
        """

        resolved = self._resolved_stage_columns()
        return [
            col
            for stage_name in self.registry.topo_order(include_archived=False)
            if stage_name in resolved
            for col in resolved[stage_name]
        ]

    def last_build_info(self) -> Dict[str, object]:
        """返回最近一次 raw 现算的结构化信息，供提交链闭环核对。"""
        return dict(self._last_build_info)


class SubmissionModelPipeline:
    """
    正式提交训练 / 推理桥接层。

    它不重新发明训练逻辑，而是直接复用 `ExperimentRunner.fit_model()` /
    `ExperimentRunner.predict()`，这样：
    - winsorize
    - Ridge/其他模型参数
    - 目标口径
    会和实验主链保持同一份实现。
    """

    def __init__(
        self,
        h5dir: str,
        feature_dir: str = DEFAULT_FEATURE_DIR,
        spec: SubmissionSpec = SubmissionSpec(),
        target_winsorize_config: Optional[Dict[str, object]] = None,
        ridge_alpha: float = 2.0,
    ):
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        self.spec = spec
        self.runner = ExperimentRunner(
            h5dir=h5dir,
            feature_dir=feature_dir,
            target_winsorize_config=target_winsorize_config or DEFAULT_TARGET_WINSORIZE,
            ridge_alpha=ridge_alpha,
        )
        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.baseline = None

    def _normalize_meow_frame(
        self,
        frame: pd.DataFrame,
        require_target: bool = False,
    ) -> pd.DataFrame:
        """
        把 `meow/` 包装层传进来的 DataFrame 统一还原成实验主链口径。

        `meow` 样例习惯把 `(symbol, date, interval)` 设为 index；
        experiment_runner 习惯把三列保留成普通列。
        这里做一次无损归一化，避免两套入口因为 DataFrame 形态不同而漂移。
        """

        out = frame.copy()
        if not set(META_COLS).issubset(out.columns):
            if list(out.index.names) == META_COLS:
                out = out.reset_index()
            else:
                raise ValueError(
                    "输入 DataFrame 缺少正式提交所需的 meta 列，且 index 也不是标准 MultiIndex"
                )
        if require_target and TARGET_COL not in out.columns:
            raise ValueError(f"训练目标缺少列: {TARGET_COL}")
        return out

    def fit(self, xdf: pd.DataFrame, ydf: pd.DataFrame) -> None:
        """
        用正式提交 spec 训练模型。

        这里直接复用 experiment_runner 的训练核心，确保：
        - winsorize 只作用于训练标签
        - 模型参数与实验链一致
        - 特征列选择口径一致
        """

        xtrain = self._normalize_meow_frame(xdf, require_target=False)
        ytrain = self._normalize_meow_frame(ydf, require_target=True)
        self.model, self.feature_cols, self.baseline = self.runner.fit_model(
            self.spec.model_name,
            xtrain,
            ytrain,
            target_mode=self.spec.target_mode,
        )

    def predict(self, xdf: pd.DataFrame) -> np.ndarray:
        """
        用已经训练好的正式提交模型做推理。

        当前正式提交口径是 `target_mode=raw`，因此这里只需直接走 runner 的预测核心。
        若未来正式提交改成可逆 residual 路线，也仍可在这里集中处理。
        """

        if self.model is None or self.feature_cols is None:
            raise RuntimeError("模型尚未训练，不能直接调用 predict()")
        xpred = self._normalize_meow_frame(xdf, require_target=False)
        return np.asarray(
            self.runner._predict_with_baseline(
                self.model,
                xpred,
                self.feature_cols,
                ydf=None,
                baseline=self.baseline,
                target_mode=self.spec.target_mode,
            ),
            dtype=np.float32,
        )
