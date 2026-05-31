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


def _require_torch():
    """
    按需导入 torch。

    约束：``dl_models.py`` 作为卡带注册中心会被脊柱频繁 import；若在模块顶层直接
    import torch，会把“torch-free 脊柱”这条纪律打破，也会让没装 torch 的环境在
    仅使用参考模型时直接炸掉。因此这里统一走懒加载：真正进入 TCN/LSTM 卡带时才导。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装 PyTorch，无法使用 TCN/LSTM 卡带。"
            "如只想跑 torch-free 脊柱，请继续使用 reference_* 模型。"
        ) from exc
    return torch, nn, F


def _resolve_torch_device(torch, requested: str) -> str:
    """
    解析本次训练实际使用的 device。

    口径：
    - 显式要求 ``cuda`` 但本机不可用 → 立即报错，避免用户误以为已经上卡。
    - ``auto`` / 空值 → 有卡用 cuda，否则退 cpu。
    - 其余显式值直接透传给 torch（如 ``cpu`` / ``cuda:0``）。
    """
    req = str(requested or "auto").strip().lower()
    if req in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 torch.cuda.is_available() 为 False。")
    return requested


class _CausalConvBlock:
    """
    极薄的因果卷积包装。

    这里不用 ``padding='same'``，因为 same padding 会在左右两侧同时补零；那样时间 t
    的输出会看到未来位置信息，直接破坏“因果卷积”这一条根纪律。正确做法是只在左侧补。
    """

    def __init__(self, conv, pad: int, torch):
        self.conv = conv
        self.pad = int(pad)
        self._torch = torch

    def __call__(self, x):
        if self.pad > 0:
            x = self._torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


def _build_tcn_module(input_channels: int, hidden_size: int, num_layers: int, dropout: float):
    """
    构造一个轻量 TCN 回归头。

    结构选择刻意保守：
    - 残差块 + 膨胀卷积，覆盖 226 步日内序列足够；
    - 全局平均池化后接线性头，输出单个 ``fret12`` 标量；
    - 不在这里搞花哨结构，先把“接线 + 因果 + 可训 + 可搜”打通。
    """
    torch, nn, _ = _require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int):
            super().__init__()
            kernel_size = 3
            pad = dilation * (kernel_size - 1)
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
            self.conv1_wrap = _CausalConvBlock(self.conv1, pad, torch)
            self.conv2_wrap = _CausalConvBlock(self.conv2, pad, torch)
            self.norm1 = nn.BatchNorm1d(out_ch)
            self.norm2 = nn.BatchNorm1d(out_ch)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(float(dropout))
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            residual = self.skip(x)
            out = self.conv1_wrap(x)
            out = self.norm1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.conv2_wrap(out)
            out = self.norm2(out)
            out = self.act(out)
            out = self.dropout(out)
            return out + residual

    class TCNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = int(input_channels)
            for i in range(int(num_layers)):
                dilation = 2 ** i
                layers.append(ResidualBlock(in_ch, int(hidden_size), dilation))
                in_ch = int(hidden_size)
            self.backbone = nn.ModuleList(layers)
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, x):
            # 输入约定：[B, L, C]，Conv1d 期望 [B, C, L]
            out = x.transpose(1, 2)
            for block in self.backbone:
                out = block(out)
            # 只用时间维平均池化，保持回归头极薄，避免再引入额外泄漏面。
            pooled = out.mean(dim=-1)
            return self.head(pooled).squeeze(-1)

    return TCNRegressor()

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


@register_model(ModelKind.TCN)
class TCNCartridge(ModelCartridge):
    """
    主攻卡带：TCN-on-原始微结构。

    设计边界严格贴规格：
    - **唯一 torch 层**：nn.Module / optimizer / autograd / device 全封在这里；
    - **required_adapter 固定 RAW_CHANNELS**：让组装期就拦住“TCN 却喂错输入”的配置错误；
    - **search_space 复用 STRUCTURE_SEARCH_SPACE**：序列长由 trainer 开窗，hidden/layers 在
      这里真正消费。

    训练策略先取最小可用闭环，不在第一版里把复杂度堆上去：
    - loss = MSE（与最终回归目标同量纲）；
    - early stopping 看训练区尾段的 ``earlystop_ds``，绝不碰 scoring；
    - 如果 earlystop 为空，就退化成看训练损失，至少保持接口闭环不炸。
    """

    search_space: ClassVar[Dict] = STRUCTURE_SEARCH_SPACE
    required_adapter = AdapterKind.RAW_CHANNELS

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._fallback_bias = 0.0
        self._fitted = False

    @classmethod
    def from_config(cls, model_config) -> "TCNCartridge":
        return cls()

    @staticmethod
    def _default_hparams(hparams: Dict) -> Dict:
        """
        合并本卡带的默认训练超参与搜索得到的结构超参。

        结构三旋钮（seq_len / hidden_size / num_layers）里，本卡带只真正消费后两者；
        ``seq_len`` 仍由 Searcher/Trainer 决定开窗，不在此重复处理。
        """
        merged = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 20,
            "patience": 4,
            "grad_clip": 1.0,
            "device": "auto",
        }
        merged.update(dict(hparams or {}))
        return merged

    def _eval_loss(self, ds, loss_fn) -> float:
        """按自然顺序评估一个数据集的平均损失。"""
        torch = self._torch
        if len(ds) == 0:
            return float("inf")
        total = 0.0
        count = 0
        self._model.eval()
        with torch.no_grad():
            for xb, yb in ds.iter_batches(batch_size=512, shuffle=False):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                pred = self._model(x)
                loss = loss_fn(pred, y)
                total += float(loss.item()) * len(yb)
                count += len(yb)
        return total / max(count, 1)

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        torch, nn, _ = _require_torch()
        cfg = self._default_hparams(hparams)
        self._torch = torch
        self._device = _resolve_torch_device(torch, cfg.get("device", "auto"))

        t0 = time.time()
        if len(train_ds) == 0:
            self._model = None
            self._fitted = True
            self._fallback_bias = 0.0
            return TrainRecord(
                train_curve=[0.0],
                earlystop_curve=[0.0] if len(earlystop_ds) else [],
                best_epoch=0,
                n_epochs=1,
                fit_seconds=time.time() - t0,
                extra={"kind": "tcn", "empty": True, "device": self._device},
            )

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        self._model = _build_tcn_module(
            input_channels=train_ds.n_channels,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(cfg["batch_size"])
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        grad_clip = float(cfg["grad_clip"])

        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        train_curve: List[float] = []
        early_curve: List[float] = []

        # 兜底偏置：只需要“训练标签的均值”，不需要把整份 ``X[B,L,C]`` 一次性物化进内存。
        # 这里改走 label_frame 只取 ``fret12`` 一列，避免海选单折在 40 天训练窗上额外造一个
        # 数 GB 级的窗口张量副本；训练主路径仍保持 iter_batches 流式。
        train_labels = train_ds.label_frame()["fret12"].to_numpy(dtype=np.float32)
        self._fallback_bias = float(np.mean(train_labels)) if len(train_labels) else 0.0

        for epoch in range(1, max_epochs + 1):
            self._model.train()
            total = 0.0
            count = 0
            for xb, yb in train_ds.iter_batches(batch_size=batch_size, shuffle=True, seed=int(seed) + epoch):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=grad_clip)
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)

            train_loss = total / max(count, 1)
            train_curve.append(train_loss)

            if len(earlystop_ds) > 0:
                metric = self._eval_loss(earlystop_ds, loss_fn)
                early_curve.append(metric)
            else:
                metric = train_loss

            if metric + 1e-8 < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        return TrainRecord(
            train_curve=train_curve,
            earlystop_curve=early_curve,
            best_epoch=best_epoch,
            n_epochs=len(train_curve),
            fit_seconds=time.time() - t0,
            extra={
                "kind": "tcn",
                "device": self._device,
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
            },
        )

    def predict(self, ds) -> np.ndarray:
        if len(ds) == 0:
            return np.zeros(0, dtype=np.float32)
        if not self._fitted or self._model is None:
            return np.full(len(ds), self._fallback_bias, dtype=np.float32)

        torch = self._torch
        preds: List[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for xb, _ in ds.iter_batches(batch_size=512, shuffle=False):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                pred = self._model(x).detach().cpu().numpy().astype(np.float32)
                preds.append(pred)
        return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)


def _build_gru_module(input_channels: int, hidden_size: int, num_layers: int, dropout: float):
    """
    构造轻量 GRU 回归头。

    设计选择与 TCN 对称保守：
    - nn.GRU（batch_first=True）接收 [B, L, C]，取**末步输出**（等价于最后层最后时刻隐藏态）；
    - 末步天然因果：GRU 从左到右顺序处理，t=L 时刻已积累全窗上下文，不跨越标签时刻；
    - 不加 LayerNorm / Attention 等附件——FeatureAdapter 已经过脊柱 zscore 白化，量纲一致；
    - dropout 仅在 num_layers > 1 时有效（PyTorch 限制，单层 dropout 静默忽略）。
    """
    torch, nn, _ = _require_torch()

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            gru_drop = float(dropout) if int(num_layers) > 1 else 0.0
            self.gru = nn.GRU(
                input_size=int(input_channels),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=gru_drop,
            )
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, x):
            # x: [B, L, C]；GRU 输出 (output[B,L,H], h_n[num_layers,B,H])
            # output[:, -1, :] == h_n[-1]（最后层、最后时刻），语义更直观。
            output, _ = self.gru(x)
            last = output[:, -1, :]          # [B, H]
            return self.head(last).squeeze(-1)  # [B]

    return GRURegressor()


@register_model(ModelKind.GRU)
class GRUCartridge(ModelCartridge):
    """
    第二卡带：GRU-on-433工程特征。

    与 TCNCartridge 形成"架构 × 输入"对照：
    - TCN 喂 59 个原始微结构通道，让网络自学交互；
    - GRU 喂 433 个工程特征，把传统特征工程的先验直接注入 DL。

    required_adapter 固定 FEATURE_433：
    - 用传统侧已沉淀的截面归一化特征（cross-z / cross-rank / OFI / 动量等）；
    - FeatureAdapter 复用 SubmissionFeaturePipeline，train/serve 同口径，无 skew 风险。

    训练策略与 TCNCartridge 完全一致（MSE + AdamW + 早停 + 梯度裁剪），降低对照噪声。
    """

    search_space: ClassVar[Dict] = STRUCTURE_SEARCH_SPACE
    required_adapter = AdapterKind.FEATURE_433

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._fallback_bias = 0.0
        self._fitted = False

    @classmethod
    def from_config(cls, model_config) -> "GRUCartridge":
        return cls()

    @staticmethod
    def _default_hparams(hparams: Dict) -> Dict:
        merged = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 20,
            "patience": 4,
            "grad_clip": 1.0,
            "device": "auto",
        }
        merged.update(dict(hparams or {}))
        return merged

    def _eval_loss(self, ds, loss_fn) -> float:
        torch = self._torch
        if len(ds) == 0:
            return float("inf")
        total = 0.0
        count = 0
        self._model.eval()
        with torch.no_grad():
            for xb, yb in ds.iter_batches(batch_size=512, shuffle=False):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                pred = self._model(x)
                total += float(loss_fn(pred, y).item()) * len(yb)
                count += len(yb)
        return total / max(count, 1)

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        torch, nn, _ = _require_torch()
        cfg = self._default_hparams(hparams)
        self._torch = torch
        self._device = _resolve_torch_device(torch, cfg.get("device", "auto"))

        t0 = time.time()
        if len(train_ds) == 0:
            self._model = None
            self._fitted = True
            self._fallback_bias = 0.0
            return TrainRecord(
                train_curve=[0.0],
                earlystop_curve=[0.0] if len(earlystop_ds) else [],
                best_epoch=0, n_epochs=1,
                fit_seconds=time.time() - t0,
                extra={"kind": "gru", "empty": True, "device": self._device},
            )

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        self._model = _build_gru_module(
            input_channels=train_ds.n_channels,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(cfg["batch_size"])
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        grad_clip = float(cfg["grad_clip"])

        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        train_curve: List[float] = []
        early_curve: List[float] = []

        # 兜底偏置：只取 label 列均值，不物化整窗 X[B,L,C]。
        train_labels = train_ds.label_frame()["fret12"].to_numpy(dtype=np.float32)
        self._fallback_bias = float(np.mean(train_labels)) if len(train_labels) else 0.0

        for epoch in range(1, max_epochs + 1):
            self._model.train()
            total = 0.0
            count = 0
            for xb, yb in train_ds.iter_batches(batch_size=batch_size, shuffle=True, seed=int(seed) + epoch):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=grad_clip)
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)

            train_loss = total / max(count, 1)
            train_curve.append(train_loss)

            if len(earlystop_ds) > 0:
                metric = self._eval_loss(earlystop_ds, loss_fn)
                early_curve.append(metric)
            else:
                metric = train_loss

            if metric + 1e-8 < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        return TrainRecord(
            train_curve=train_curve,
            earlystop_curve=early_curve,
            best_epoch=best_epoch,
            n_epochs=len(train_curve),
            fit_seconds=time.time() - t0,
            extra={
                "kind": "gru",
                "device": self._device,
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
            },
        )

    def predict(self, ds) -> np.ndarray:
        if len(ds) == 0:
            return np.zeros(0, dtype=np.float32)
        if not self._fitted or self._model is None:
            return np.full(len(ds), self._fallback_bias, dtype=np.float32)

        torch = self._torch
        preds: List[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for xb, _ in ds.iter_batches(batch_size=512, shuffle=False):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                pred = self._model(x).detach().cpu().numpy().astype(np.float32)
                preds.append(pred)
        return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)
