"""
卡带层 —— InputAdapter 接口 + 适配器实现 + ModelCartridge 接口 + numpy 参考模型

这是规格 §3 两个接口契约的落地，也是"可换卡带"那一列：

- ``InputAdapter``（输入适配卡带）：唯一理解原始数据语义的地方，吐出通道布局固定的
  ``[n_rows_day, C]`` 干净数组；下游全是不关心含义的张量数学。
    - ``IdentityAdapter``：直接把指定 raw 数值列当通道（调试 / 防泄漏合成测试）。
    - ``FeatureAdapter``：包装现有 433 特征管线（``SubmissionFeaturePipeline``），零新特征公式。
- ``ModelCartridge``（模型卡带，**唯一允许出现 torch 的地方**——但本文件只放 torch-free
  的 numpy 参考模型；真正的 LSTM/DeepLOB 卡带等 4060 + PyTorch 就绪后再加，脊柱不动）。
    - ``ReferenceZeroCartridge``：恒 0，corr 基线 sanity。
    - ``ReferenceLastCartridge``：末步通道线性回归——**防泄漏探测器**：干净因果管线下
      只能拿到窗末特征、对"依赖未来的标签"打低分；一旦窗口/对齐/归一化漏了未来信息，
      它立刻能吃到并涨分（规格 §5.5）。

全部 torch-free、Mac CPU 可跑。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

import numpy as np

from adapter_config import AdapterKind
from model_config import ModelKind
from registry import register_adapter, register_model


# ================================================================== #
# InputAdapter 接口 + 实现
# ================================================================== #

class InputAdapter(ABC):
    """
    输入适配卡带接口（规格 §3.1）。

    - ``channels``: 通道名列表，顺序即 C 维布局（固定）。
    - ``build(raw_day_df)``: 吃**一天** raw，返回 ``[n_rows_day, C]`` float32；
      行序约定 = 输入按 ``(date, symbol, interval)`` 稳定排序后的序（与
      ``sequence_dataset.build_sequence_arrays`` 的取 meta 口径一致）。
    """

    channels: List[str]

    @classmethod
    @abstractmethod
    def from_config(cls, adapter_config) -> "InputAdapter":
        ...

    @abstractmethod
    def build(self, raw_day_df) -> np.ndarray:
        ...


@register_adapter(AdapterKind.IDENTITY)
class IdentityAdapter(InputAdapter):
    """把指定 raw 数值列**原样**当通道。用于调试与防泄漏合成测试（无任何语义加工）。"""

    def __init__(self, columns):
        if not columns:
            raise ValueError("IdentityAdapter 需要非空 columns（要把哪些 raw 列当通道）")
        self.channels = list(columns)

    @classmethod
    def from_config(cls, adapter_config) -> "IdentityAdapter":
        return cls(columns=adapter_config.columns)

    def build(self, raw_day_df) -> np.ndarray:
        # 按 (symbol, interval) 稳定排序（单日内 date 恒定），对齐统一行序契约。
        day = raw_day_df.sort_values(["symbol", "interval"], kind="mergesort")
        return day[self.channels].to_numpy(dtype=np.float32)


@register_adapter(AdapterKind.FEATURE_433)
class FeatureAdapter(InputAdapter):
    """
    包装现有 433 特征管线：把正式提交特征列当通道（规格 §3.1「首个实现」）。

    复用 ``SubmissionFeaturePipeline.build_feature_frames``（逐日现算、不跨日、不依赖
    磁盘缓存），因此 train/serve 用同一套现算口径，天然无 train/serve skew。
    """

    def __init__(self, groups=None):
        # 延迟 import：FeatureAdapter 用到时才拉特征管线（src 在 path）。
        from submission_pipeline import SubmissionFeaturePipeline, DEFAULT_SUBMISSION_GROUPS
        groups = tuple(groups) if groups else DEFAULT_SUBMISSION_GROUPS
        self._pipeline = SubmissionFeaturePipeline(groups=groups)
        self.channels = list(self._pipeline.feature_names())

    @classmethod
    def from_config(cls, adapter_config) -> "FeatureAdapter":
        return cls(groups=adapter_config.groups or None)

    def build(self, raw_day_df) -> np.ndarray:
        # build_feature_frames 内部按 (date,symbol,interval) 排序后逐日算，单日块行序
        # = (symbol,interval)，与统一行序契约一致。groups=None 即用构造时锁定的 groups；
        # 再强制按 self.channels 取列、定死列顺序（C 维布局固定）。
        xdf, _ = self._pipeline.build_feature_frames(raw_day_df, groups=None)
        return xdf[self.channels].to_numpy(dtype=np.float32)


# ================================================================== #
# ModelCartridge 接口 + numpy 参考模型
# ================================================================== #

@dataclass
class TrainRecord:
    """
    卡带 ``fit`` 的单向上报（规格 §3.2/§4）：逐 epoch 曲线 + best epoch + 用时。

    供脊柱判过拟合（train vs earlystop gap / 曲线）与 HPO 早杀整个 trial。
    **单向**——脊柱只读不写，绝不反过来控制卡带内层 epoch 循环。
    """
    train_curve: List[float] = field(default_factory=list)
    earlystop_curve: List[float] = field(default_factory=list)
    best_epoch: int = 0
    n_epochs: int = 0
    fit_seconds: float = 0.0
    extra: Dict = field(default_factory=dict)


class ModelCartridge(ABC):
    """
    模型卡带接口（规格 §3.2）。唯一允许 import torch 的层（参考模型 torch-free）。

    - ``search_space``: 本模型私有的超参声明（不进 config/ 全局枚举）。
    - ``required_adapter``: 类型化引用，``None`` = 不挑适配器（参考模型）。
    - ``fit(train_ds, earlystop_ds, hparams, seed) -> TrainRecord``：内含完整训练循环 +
      早停（看 earlystop_ds，不碰 scoring）。
    - ``predict(ds) -> np.ndarray``：输出留在 ``fret12`` 量纲、与 ds 窗口顺序对齐。
    """

    search_space: ClassVar[Dict] = {}
    required_adapter: ClassVar[Optional[AdapterKind]] = None

    @classmethod
    @abstractmethod
    def from_config(cls, model_config) -> "ModelCartridge":
        ...

    @abstractmethod
    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        ...

    @abstractmethod
    def predict(self, ds) -> np.ndarray:
        ...


@register_model(ModelKind.REFERENCE_ZERO)
class ReferenceZeroCartridge(ModelCartridge):
    """恒 0 预测。corr 恒约 0，作 sanity 基线（注意：恒 0 无法探测泄漏，探测靠末步线性）。"""

    required_adapter = None   # 不挑适配器

    @classmethod
    def from_config(cls, model_config) -> "ReferenceZeroCartridge":
        return cls()

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=0.0, extra={"kind": "reference_zero"})

    def predict(self, ds) -> np.ndarray:
        return np.zeros(len(ds), dtype=np.float32)


@register_model(ModelKind.REFERENCE_LAST)
class ReferenceLastCartridge(ModelCartridge):
    """
    **防泄漏探测器**：用窗口**末步**各通道做一个 numpy 线性回归预测标签。

    机理：因果窗口下末步只含 ``t`` 时刻信息——若标签依赖未来 ``t+k``，它学不到、打低分；
    一旦 WindowIndexer 跨界 / 标签对齐错位 / Normalizer 用了 val 统计而漏进未来信息，
    它立刻能吃到并涨分。所以"它在干净管线上打低分"是 D0 的核心验收闸（规格 §5.5）。
    """

    required_adapter = None   # 不挑适配器，可在合成 identity 通道或真实 433 特征上探测

    def __init__(self):
        self._w: Optional[np.ndarray] = None   # [C]
        self._b: float = 0.0

    @classmethod
    def from_config(cls, model_config) -> "ReferenceLastCartridge":
        return cls()

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        t0 = time.time()
        X, y = train_ds.gather_all()              # X[B,L,C], y[B]
        if X.shape[0] == 0:
            self._w = np.zeros(train_ds.n_channels, dtype=np.float64)
            self._b = 0.0
            return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=time.time() - t0,
                               extra={"kind": "reference_last", "empty": True})
        x_last = X[:, -1, :].astype(np.float64)   # [B, C] 窗口末步
        # 最小二乘线性回归（含 bias），torch-free。
        A = np.concatenate([x_last, np.ones((x_last.shape[0], 1))], axis=1)  # [B, C+1]
        coef, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        self._w = coef[:-1]
        self._b = float(coef[-1])
        pred = A @ coef
        train_mse = float(np.mean((pred - y) ** 2))
        return TrainRecord(train_curve=[train_mse], n_epochs=1, fit_seconds=time.time() - t0,
                           extra={"kind": "reference_last"})

    def predict(self, ds) -> np.ndarray:
        X, _ = ds.gather_all()
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        x_last = X[:, -1, :].astype(np.float64)
        return (x_last @ self._w + self._b).astype(np.float32)
