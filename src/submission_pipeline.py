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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiment_runner import DEFAULT_TARGET_WINSORIZE, ExperimentRunner
from feature_registry import META_COLS, TARGET_COL, FeatureRegistry, registry as default_registry
from feature_store import DEFAULT_FEATURE_DIR


@dataclass(frozen=True)
class SubmissionMemberSpec:
    """
    正式提交中“单个成员模型”的规格。

    这里把成员训练/推理真正会用到的口径都收口在一处：
    - experiment_id: 成员的人类可读 ID，便于日志与调试
    - groups: 使用哪些正式特征组
    - model_name: 训练模型
    - target_mode: 训练目标口径
    - model_params: 该成员自己的预钉模型超参
    """

    experiment_id: str
    groups: Tuple[str, ...]
    model_name: str
    target_mode: str = "raw"
    model_params: Dict[str, object] = field(default_factory=dict)


DEFAULT_SUBMISSION_MEMBERS: Tuple[SubmissionMemberSpec, ...] = (
    SubmissionMemberSpec(
        experiment_id="X1_R02_plus_ofi_safe_condmom_interaction",
        groups=("legacy", "norm_core", "ofi_safe", "conditional_momentum_interaction"),
        model_name="ridge",
        target_mode="raw",
        model_params={"alpha": 2.0},
    ),
    SubmissionMemberSpec(
        experiment_id="M_lgbm_d4",
        groups=("legacy", "norm_core", "ofi_safe", "trade_impact", "lag", "roll", "patch_summary", "cross_rank", "regime_tree"),
        model_name="lgbm",
        target_mode="raw",
        model_params={"max_depth": 4, "num_leaves": 15, "n_jobs": 8},
    ),
)


def _merge_member_groups(members: Sequence[SubmissionMemberSpec]) -> Tuple[str, ...]:
    """
    把多个成员的特征组并成一个稳定有序的并集。

    为什么要并集：
    - 提交通道需要“一次从 raw 现算全部所需特征”
    - 之后再让各成员从这份并集里各取自己的子集
    - 这样既不重复算特征，也能保证 train/serve 用同一份原始现算结果
    """

    ordered_groups: List[str] = []
    seen = set()
    for member in members:
        for group_name in member.groups:
            if group_name in seen:
                continue
            seen.add(group_name)
            ordered_groups.append(group_name)
    return tuple(ordered_groups)


# 当前正式提交默认口径 = 两成员融合：
# 1. X1 Ridge：鲁棒锚
# 2. M_lgbm_d4：上限型成员
# 外部包装层只引用这里，后续若提交口径升级，仍维持“改一处，全链同步”。
DEFAULT_SUBMISSION_GROUPS: Tuple[str, ...] = _merge_member_groups(DEFAULT_SUBMISSION_MEMBERS)


@dataclass(frozen=True)
class SubmissionSpec:
    """
    正式提交通道的整体规格。

    当前默认是“多成员融合”而非单模型，因此这里的核心字段改为：
    - members: 参与融合的成员列表
    - blend_mode: 融合方式（当前只锁 `per_day_zscore_mean`）
    """

    members: Tuple[SubmissionMemberSpec, ...] = DEFAULT_SUBMISSION_MEMBERS
    # 提交默认 raw_mean：输出留在 fret12 量纲，MSE/R² 才有意义（老师精度分 = MSE+Pearson+R² 各 1/3）。
    # per_day_zscore_mean 仅作诊断对照（会把输出推到 std≈1、毁掉 MSE/R²）。
    blend_mode: str = "raw_mean"

    def feature_groups(self) -> Tuple[str, ...]:
        """返回当前整体规格所需的特征组并集，供特征现算入口直接复用。"""
        return _merge_member_groups(self.members)


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
        spec: Optional[SubmissionSpec] = None,
        target_winsorize_config: Optional[Dict[str, object]] = None,
        ridge_alpha: float = 2.0,
    ):
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        self.spec = spec or SubmissionSpec()
        self.runner = ExperimentRunner(
            h5dir=h5dir,
            feature_dir=feature_dir,
            target_winsorize_config=target_winsorize_config or DEFAULT_TARGET_WINSORIZE,
            ridge_alpha=ridge_alpha,
        )
        # 多成员提交态：
        # - models: 每个成员训练好的 estimator
        # - member_feature_cols: 每个成员真正喂进模型的列
        # - member_input_cols: 从“并集特征表”里切子集时要取哪些列
        # - member_baselines: 仅 residual 类 target_mode 才会用到；当前 raw 路线保持兼容
        self.models: Dict[str, object] = {}
        self.member_feature_cols: Dict[str, List[str]] = {}
        self.member_input_cols: Dict[str, List[str]] = {}
        self.member_baselines: Dict[str, object] = {}

    def _member_xdf(self, xdf: pd.DataFrame, member: SubmissionMemberSpec) -> pd.DataFrame:
        """
        从“并集特征表”里切出某个成员自己的输入子集。

        这是提交链的关键点：
        - raw 侧只现算一次特征并集
        - 训练/推理时每个成员严格只看自己定义过的列
        - 避免 X1 错吃到 LightGBM 的扩展列，或反之
        """

        input_cols = self.member_input_cols.get(member.experiment_id)
        if input_cols is None:
            input_cols = SubmissionFeaturePipeline(groups=member.groups).feature_names()
            self.member_input_cols[member.experiment_id] = list(input_cols)
        return xdf[META_COLS + list(input_cols)].copy()

    def _per_day_zscore(self, frame: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        """
        对单个成员预测做“按天截面 zscore”。

        这是提交口径里最重要的一条：
        - **均值/标准差只从当前要预测的那天数据自身计算**
        - 绝不冻结训练期统计量到 serve 端
        - 这样不会产生 train/serve skew，也不会把历史窗口先验偷带到新数据
        """

        work = frame.loc[:, ["date"]].copy()
        work["pred"] = np.asarray(pred, dtype=np.float64)
        zpred = np.zeros(len(work), dtype=np.float64)
        for _, idx in work.groupby("date", sort=False).groups.items():
            day_values = work.loc[idx, "pred"].to_numpy(dtype=np.float64)
            day_std = float(np.std(day_values))
            if day_std < 1e-12:
                zpred[np.asarray(list(idx), dtype=np.int64)] = day_values - float(np.mean(day_values))
                continue
            zpred[np.asarray(list(idx), dtype=np.int64)] = (
                (day_values - float(np.mean(day_values))) / day_std
            )
        return zpred.astype(np.float32)

    def _blend_member_predictions(self, xpred: pd.DataFrame, member_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        把多个成员预测融合成最终提交分数。

        正式提交锁定 `raw_mean`（等权 raw 平均）：
        - 两个成员都在 raw `fret12` 上以平方损失训练，输出本就是同量纲的收益预测；
        - 直接等权平均 → 输出仍落在 `fret12` 量纲，MSE / R² 才有意义（老师精度分 = MSE + Pearson + R² 各占 1/3）；
        - corr 与 per-day zscore 等权几乎一致（P5 实测 raw 0.0762 ≈ zscore 0.0763），但 zscore 会把输出推到 std≈1，
          毁掉 MSE / R²，故提交端不用 zscore。
        - 仍是零自由参数、可辩护的等权融合。

        `per_day_zscore_mean` 仅保留作诊断/对照，不作提交默认。
        """

        if not member_preds:
            raise RuntimeError("没有任何成员预测，无法融合")

        blended = np.zeros(len(xpred), dtype=np.float64)
        if self.spec.blend_mode == "raw_mean":
            for pred in member_preds.values():
                blended += np.asarray(pred, dtype=np.float64)
        elif self.spec.blend_mode == "per_day_zscore_mean":
            for pred in member_preds.values():
                blended += self._per_day_zscore(xpred, pred).astype(np.float64)
        else:
            raise ValueError(f"未知融合模式: {self.spec.blend_mode}")
        blended /= float(len(member_preds))
        return blended.astype(np.float32)

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
        用正式提交 spec 训练全部成员模型。

        这里继续复用 experiment_runner 的训练核心，确保：
        - winsorize 只作用于训练标签
        - 模型参数与实验链一致
        - 但每个成员只吃自己定义过的列，而不是整张并集表
        """

        xtrain = self._normalize_meow_frame(xdf, require_target=False)
        ytrain = self._normalize_meow_frame(ydf, require_target=True)
        self.models = {}
        self.member_feature_cols = {}
        self.member_baselines = {}
        for member in self.spec.members:
            member_xtrain = self._member_xdf(xtrain, member)
            model, feature_cols, baseline = self.runner.fit_model(
                member.model_name,
                member_xtrain,
                ytrain,
                target_mode=member.target_mode,
                model_params=member.model_params,
            )
            self.models[member.experiment_id] = model
            self.member_feature_cols[member.experiment_id] = list(feature_cols)
            self.member_baselines[member.experiment_id] = baseline

    def predict(self, xdf: pd.DataFrame) -> np.ndarray:
        """
        用全部已训练成员做推理，并按正式融合口径输出最终预测。

        当前正式提交通道不再输出单模型结果，而是：
        1. 各成员各自预测
        2. 各成员按天截面 zscore
        3. 等权平均成最终 `forecast`
        """

        if not self.models:
            raise RuntimeError("模型尚未训练，不能直接调用 predict()")
        xpred = self._normalize_meow_frame(xdf, require_target=False)
        member_preds: Dict[str, np.ndarray] = {}
        for member in self.spec.members:
            member_xpred = self._member_xdf(xpred, member)
            member_preds[member.experiment_id] = np.asarray(
                self.runner._predict_with_baseline(
                    self.models[member.experiment_id],
                    member_xpred,
                    self.member_feature_cols[member.experiment_id],
                    ydf=None,
                    baseline=self.member_baselines.get(member.experiment_id),
                    target_mode=member.target_mode,
                ),
                dtype=np.float32,
            )
        return self._blend_member_predictions(xpred, member_preds)
