"""
卡带层 —— InputAdapter 接口 + 适配器实现 + ModelCartridge 接口 + numpy 参考模型

这是规格 §3 两个接口契约的落地，也是"可换卡带"那一列：

- ``InputAdapter``（输入适配卡带）：唯一理解原始数据语义的地方，吐出通道布局固定的
  ``[n_rows_day, C]`` 干净数组；下游全是不关心含义的张量数学。
    - ``IdentityAdapter``：直接把指定 raw 数值列当通道（调试 / 防泄漏合成测试）。
    - ``FeatureAdapter``：包装现有 433 特征管线（``SubmissionFeaturePipeline``），零新特征公式（D1/LSTM）。
    - ``RawChannelAdapter``：~59 原始微结构通道做**最小语义归一**（价相对 mid / midpx 日内对数收益 /
      量 log1p），让网络自己学交互——刻意不手搓 imbalance/OFI（D2/TCN，规格 §3.1/§8.0）。
- ``ModelCartridge``（模型卡带，**唯一允许出现 torch 的地方**——但本文件只放 torch-free
  的 numpy 参考模型；真正的 LSTM/TCN 卡带等 4060 + PyTorch 就绪后再加，脊柱不动）。
    - ``ReferenceZeroCartridge``：恒 0，corr 基线 sanity。
    - ``ReferenceLastCartridge``：末步通道线性回归——**防泄漏探测器**：干净因果管线下
      只能拿到窗末特征、对"依赖未来的标签"打低分；一旦窗口/对齐/归一化漏了未来信息，
      它立刻能吃到并涨分（规格 §5.5）。
    - ``ReferencePoolCartridge``：窗口均值池化 + numpy 线性；**声明 ``STRUCTURE_SEARCH_SPACE``**，
      作 HPO Searcher 的 torch-free 被测对象 + 未来 TCN 卡带的 search_space 模板。

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


# —— RawChannelAdapter 的通道分桶（按 raw 列语义固定，规格 §8.0；C 维布局即下面顺序） —— #
# ① 价类：相对 mid 归一 (p/midpx - 1)，价<=0（该 interval 无成交）时置 0（中性）。
_PRICE_REL_COLS = (
    "lastpx", "open", "high", "low",
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "tradeBuyHigh", "tradeBuyLow", "buyVwad",
    "tradeSellHigh", "tradeSellLow", "sellVwad",
    "addBuyHigh", "addBuyLow", "addSellHigh", "addSellLow",
    "cxlBuyHigh", "cxlBuyLow", "cxlSellHigh", "cxlSellLow",
)
# ② 量类：log1p 驯厚尾（非负：聚合 size / 笔数 / 量 / 额）。
_LOG_VOLUME_COLS = (
    "bsize0", "asize0", "bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19",
    "nTradeBuy", "tradeBuyQty", "tradeBuyTurnover",
    "nTradeSell", "tradeSellQty", "tradeSellTurnover",
    "nAddBuy", "addBuyQty", "addBuyTurnover",
    "nAddSell", "addSellQty", "addSellTurnover",
    "nCxlBuy", "cxlBuyQty", "cxlBuyTurnover",
    "nCxlSell", "cxlSellQty", "cxlSellTurnover",
)
# ③ 比率类：已是各档 turnover ratio（量纲驯过），直接透传，交脊柱 zscore 统计白化。
_RATIO_COLS = ("btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19")
# ④ 价格路径：midpx 的日内对数收益（按 symbol 组内、首步 0、因果不跨日跨票），单独成一通道。
_MIDPX_COL = "midpx"


@register_adapter(AdapterKind.RAW_CHANNELS)
class RawChannelAdapter(InputAdapter):
    """
    把 ~59 个原始微结构通道做**最小语义归一**当通道（规格 §3.1 / §8.0；D2/TCN）。

    设计哲学：走 raw 这条线的意义就是**让网络自己学通道间交互**（对照 ``FeatureAdapter``
    喂 433 手工特征）。所以这里**只做"平稳化 + 量纲驯服"**，刻意不手搓 imbalance/OFI：

    - **价 → 相对 mid**（``_PRICE_REL_COLS``）：``p/midpx - 1``，行内无状态；价<=0（无成交）置 0。
    - **midpx → 日内对数收益**（``midpx_logret``）：按 symbol 组内 ``log(midpx_t)-log(midpx_{t-1})``，
      段首（symbol 变化）置 0——这是最关键的收益路径通道，**因果、不跨日跨票**。
    - **量 → log1p**（``_LOG_VOLUME_COLS``）：非负厚尾，``log1p`` 驯尾。
    - **比率 → 透传**（``_RATIO_COLS``）：各档 turnover ratio 已驯过量纲，交脊柱 zscore。

    通道布局固定（C 维顺序）：``midpx_logret`` → 价类(rel) → 量类(log) → 比率类。
    全程 numpy、行内/组内无状态变换，无跨日跨票泄漏风险。
    """

    EPS: float = 1e-12

    def __init__(self):
        # 通道名定死顺序即 C 维布局：路径 + 价(rel) + 量(log) + 比率。
        self.channels = (
            ["midpx_logret"]
            + [f"{c}_rel" for c in _PRICE_REL_COLS]
            + [f"{c}_log" for c in _LOG_VOLUME_COLS]
            + list(_RATIO_COLS)
        )
        self._required_raw = (_MIDPX_COL,) + _PRICE_REL_COLS + _LOG_VOLUME_COLS + _RATIO_COLS

    @classmethod
    def from_config(cls, adapter_config) -> "RawChannelAdapter":
        # RAW_CHANNELS 用固定的 59 通道，忽略 AdapterConfig.groups/columns（通道布局须固定）。
        return cls()

    def build(self, raw_day_df) -> np.ndarray:
        # 缺列即报错（通道布局须固定，不静默降级）。
        missing = [c for c in self._required_raw if c not in raw_day_df.columns]
        if missing:
            raise KeyError(f"RawChannelAdapter 缺原始列: {missing[:8]}{'...' if len(missing) > 8 else ''}")

        # 与统一行序契约一致：单日内按 (symbol, interval) 稳定排序。
        day = raw_day_df.sort_values(["symbol", "interval"], kind="mergesort").reset_index(drop=True)
        n = len(day)
        mid = day[_MIDPX_COL].to_numpy(dtype=np.float64)
        safe_mid = np.where(np.abs(mid) < self.EPS, np.nan, mid)   # 防除零

        cols: List[np.ndarray] = []

        # ④ midpx 日内对数收益（按 symbol 段内 diff，段首置 0，因果不跨票）
        syms = day["symbol"].to_numpy()
        logmid = np.log(np.where(mid > 0, mid, np.nan))
        logret = np.zeros(n, dtype=np.float64)
        if n > 1:
            logret[1:] = logmid[1:] - logmid[:-1]
        is_new_sym = np.empty(n, dtype=bool)
        is_new_sym[0] = True if n > 0 else False
        if n > 1:
            is_new_sym[1:] = syms[1:] != syms[:-1]
        logret[is_new_sym] = 0.0
        cols.append(np.nan_to_num(logret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ① 价相对 mid（价<=0 置 0）
        for c in _PRICE_REL_COLS:
            p = day[c].to_numpy(dtype=np.float64)
            rel = np.where(p > 0, p / safe_mid - 1.0, 0.0)
            cols.append(np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ② 量 log1p（负值截 0 再 log1p）
        for c in _LOG_VOLUME_COLS:
            v = day[c].to_numpy(dtype=np.float64)
            lv = np.log1p(np.where(v > 0, v, 0.0))
            cols.append(np.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ③ 比率透传
        for c in _RATIO_COLS:
            r = day[c].to_numpy(dtype=np.float64)
            cols.append(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        return np.stack(cols, axis=1).astype(np.float32)   # [n_rows_day, C]


# ================================================================== #
# ModelCartridge 接口 + numpy 参考模型
# ================================================================== #

# TCN / LSTM 等结构化序列卡带共用的"只搜 3 个结构旋钮"搜索空间（规格 §6）。
# 约定：``seq_len`` 是 trainer 级旋钮（决定开窗、要重建 SequenceDataset，由 Searcher 喂
# SequenceTrainer），``hidden_size`` / ``num_layers`` 是卡带级 hparams（进 fit）。
# 迷你 spec 三型：{"type":"choice","values":[...]} / {"type":"int","low":a,"high":b} /
# {"type":"uniform","low":a,"high":b}（见 src/dl_search.py 采样器）。
STRUCTURE_SEARCH_SPACE: Dict = {
    "seq_len":     {"type": "choice", "values": [16, 32, 48, 64]},
    "hidden_size": {"type": "choice", "values": [32, 64, 128]},
    "num_layers":  {"type": "int", "low": 1, "high": 4},
}

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


@register_model(ModelKind.REFERENCE_POOL)
class ReferencePoolCartridge(ModelCartridge):
    """
    窗口**均值池化** + numpy 线性回归（torch-free）。

    用途有三：① 比 ReferenceLast 多看整窗（池化）的 sanity 对照；② **声明
    ``STRUCTURE_SEARCH_SPACE``**，当 HPO Searcher 的 torch-free 被测对象——不同
    ``seq_len`` 会改变池化窗口与样本数、产出不同 val_corr，让搜索/排名机制可在 Mac
    上端到端验证；③ 给未来 ``TCNCartridge`` 当 search_space + fit/predict 形态模板。

    说明：本参考模型用不上 ``hidden_size`` / ``num_layers``（numpy 线性无此结构），
    仅接收并原样记进 TrainRecord——真正消费它们的是等 4060 的 torch 卡带。
    """

    search_space: ClassVar[Dict] = STRUCTURE_SEARCH_SPACE
    required_adapter = None   # 不挑适配器

    def __init__(self):
        self._w: Optional[np.ndarray] = None   # [C]
        self._b: float = 0.0

    @classmethod
    def from_config(cls, model_config) -> "ReferencePoolCartridge":
        return cls()

    @staticmethod
    def _pool(ds):
        """整窗对 L 轴均值池化：``[B, L, C] -> [B, C]``。"""
        X, y = ds.gather_all()
        if X.shape[0] == 0:
            return np.zeros((0, ds.n_channels), dtype=np.float64), y
        return X.mean(axis=1).astype(np.float64), y

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        t0 = time.time()
        Xp, y = self._pool(train_ds)
        if Xp.shape[0] == 0:
            self._w = np.zeros(train_ds.n_channels, dtype=np.float64)
            self._b = 0.0
            return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=time.time() - t0,
                               extra={"kind": "reference_pool", "empty": True})
        A = np.concatenate([Xp, np.ones((Xp.shape[0], 1))], axis=1)   # [B, C+1]
        coef, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        self._w = coef[:-1]
        self._b = float(coef[-1])
        train_mse = float(np.mean((A @ coef - y) ** 2))
        return TrainRecord(train_curve=[train_mse], n_epochs=1, fit_seconds=time.time() - t0,
                           extra={"kind": "reference_pool", "hparams": dict(hparams)})

    def predict(self, ds) -> np.ndarray:
        Xp, _ = self._pool(ds)
        if Xp.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        return (Xp @ self._w + self._b).astype(np.float32)
