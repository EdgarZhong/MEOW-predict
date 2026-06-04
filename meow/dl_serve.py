# -*- coding: utf-8 -*-
"""
DL-on-raw serve 腿 —— 把"截面模型直接吃 raw 盘口"焊进正式提交链 `meow.py`。

为什么要它
----------
老师 `python meow.py` 用他自己的数据 `fit()` 再 `eval()` 评分（数据与我们格式一致）。
离线判决已证 DL-on-raw（XSECTION_RAW 截面卡带，直吃 raw 59 通道）与传统去相关（ρ≈0.45–0.55），
三折 + 交付窗等权融合稳赢传统、交付窗破 0.09、cert 折 seed 平均破 0.10。但那是**离线纸面数**——
serve 链里没有 DL，老师跑出来只有传统 0.0812。本模块把 DL 真正接进 fit/predict，让融合分落地。

老师约束（全部满足，见 docx）
-----------------------------
- 不许带模型缓存文件 → DL 在 `fit()` **现训**、不落任何权重文件；
- 评测数据零重合、严禁 overfit → 训练段切 15% 做 early-stop、卡带内置训练段 OLS rescale、
  归一化只用训练区统计（零泄漏）；
- 环境只保证 Python≥3.8（未承诺 torch/GPU）→ **防御式降级**：torch/CUDA/任一 DL 环节出错，
  `available=False`，`MeowEngine` 自动回落纯传统（最坏=0.0812，绝不崩、绝不 NaN）。

fit/predict 为何要拆开
----------------------
实验链的 `run_on_dl_fold` 是"训练+评测一次做完"（训练时就知道评测窗）。但 serve 端老师分两次调：
`fit(train)` 训练并把模型留在内存、`predict(eval)` 用留下的模型评后来的窗。所以这里把底层积木
（`build_sequence_arrays` / `Normalizer` / `SequenceDataset` / 卡带 `fit`/`predict`）按 fit/predict
两阶段重新编排，训练期 fit 的 `Normalizer` 存下来给 predict 复用（零泄漏）。

交付形态
--------
`fit()` 现训 **K=3 个 seed** 的 DL（seed 集成是免费杠杆、降单 seed 方差），`predict()` 取 K seed
**平均**再交给 `MeowEngine` 与传统等权融合。K / seq_len / 超参均可经环境变量覆盖（默认照搬机器2
proven 冠军 `20260604_xsection_raw_3fold_2seed`）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from log import log  # meow/ 入口已在 path 上
except Exception:  # 极端兜底：连日志模块都没有也不让它把提交链拖崩
    log = None


def _loginf(msg):
    """信息日志：优先 MeowLogger.inf，缺失/出错则退化到 print。防御代码的日志自身绝不能崩。"""
    fn = getattr(log, "inf", None)
    if callable(fn):
        try:
            fn(msg)
            return
        except Exception:
            pass
    print(msg)


def _logerr(msg):
    """错误日志：MeowLogger 无 .err，用 .red；都没有则退 .inf / print。"""
    for name in ("red", "inf"):
        fn = getattr(log, name, None)
        if callable(fn):
            try:
                fn(msg)
                return
            except Exception:
                break
    print(msg)


# —— 默认 DL 配置：照搬机器2 proven 冠军 run（summary.json.champion），不臆造 —— #
# seq_len 走 SequenceDataset（结构旋钮，独立于卡带 hparams）；其余进卡带 hparams。
_DEFAULT_SEQ_LEN = 32
_DEFAULT_HPARAMS = {
    "dropout": 0.2,
    "hidden_size": 32,
    "num_layers": 1,
    "lambda_corr": 0.3,
    "max_epochs": 15,
    "patience": 5,
    "weight_decay": 0.001,
}
_DEFAULT_SEEDS = (42, 43, 44)        # K=3 seed 平均（交付真正会用的形态）
_EARLYSTOP_FRAC = 0.15               # 训练段尾部切 15% 做 early-stop（防 overfit）
_NORMALIZER_MODE = "zscore"          # 与机器2 训练一致（run 名 *_zscore）
_FUSION_WEIGHT_DL = 0.5              # 等权融合（零自由参数，离线已证 best_w≈0.5）


def _read_seeds_env() -> tuple:
    """允许用 MEOW_DL_SEEDS=42,43 临时改 seed 数（serve 测试提速用）；非法则回默认。"""
    raw = os.environ.get("MEOW_DL_SEEDS", "").strip()
    if not raw:
        return _DEFAULT_SEEDS
    try:
        seeds = tuple(int(s) for s in raw.split(",") if s.strip() != "")
        return seeds or _DEFAULT_SEEDS
    except Exception:
        return _DEFAULT_SEEDS


def _ensure_dl_path() -> None:
    """把 src/config/models 三目录平铺加进 sys.path（DL 机理依赖三目录平铺 import）。"""
    repo = Path(__file__).resolve().parent.parent
    for sub in ("src", "config", "models"):
        p = str(repo / sub)
        if p not in sys.path:
            sys.path.append(p)


class DLServe:
    """
    DL-on-raw serve 腿。两阶段：`fit(train_dates)` 现训 K seed；`predict(eval_dates)` 出 K seed 平均。

    任一环节抛异常都被吞掉并置 `available=False`——调用方据此回落纯传统，绝不让 DL 把提交链拖崩。
    """

    def __init__(self, raw_loader, seeds=None, seq_len=_DEFAULT_SEQ_LEN, hparams=None):
        """
        - `raw_loader`: `dates -> raw DataFrame`（含 META + raw 列 + fret12）；
          serve 直接传 `MeowEngine.dloader.loadDates`，格式与机器2 训练用的 src/dl.py 一致。
        - `seeds` / `seq_len` / `hparams`: 默认照搬 proven 冠军；可显式覆盖。
        """
        self.raw_loader = raw_loader
        self.seeds = tuple(seeds) if seeds else _read_seeds_env()
        self.seq_len = int(seq_len)
        self.hparams = dict(hparams or _DEFAULT_HPARAMS)
        self.available = False          # 训练成功才置 True
        self._cartridges = []           # 每 seed 一个训练好的卡带
        self._normalizer = None         # 训练期 fit 的 Normalizer，predict 复用（零泄漏）
        self._adapter = None            # RawChannelAdapter（无状态，fit/predict 共用）
        self._device = "cpu"

    # ---- 训练阶段：现训 K seed ---- #
    def fit(self, train_dates) -> None:
        """在训练窗现训 K 个 seed 的 DL-on-raw；任何失败都置 available=False（回落传统）。"""
        train_dates = list(train_dates)
        try:
            _ensure_dl_path()
            import torch  # noqa: F401  —— import 失败即触发降级
            from sequence_dataset import (
                Normalizer,
                SequenceDataset,
                build_sequence_arrays,
                subset_by_dates,
            )
            from dl_protocol import split_train_earlystop
            from registry import build_adapter, build_cartridge
            from adapter_config import AdapterConfig, AdapterKind
            from model_config import ModelConfig, ModelKind

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            hp = dict(self.hparams)
            hp["device"] = self._device

            # RawChannelAdapter：固定 59 通道、无需 bind_data_sources（纯 raw 行内变换）。
            adapter = build_adapter(AdapterConfig(kind=AdapterKind.RAW_CHANNELS))
            self._adapter = adapter

            # 训练窗尾部切 15% 做 early-stop（与实验一致，防 overfit）。
            core_dates, es_dates = split_train_earlystop(tuple(train_dates), _EARLYSTOP_FRAC)
            _loginf(
                "[DLServe] 现算训练 raw 序列：core={} 天 / earlystop={} 天，device={}".format(
                    len(core_dates), len(es_dates), self._device
                )
            )
            core_arrays = build_sequence_arrays(self.raw_loader(list(core_dates)), adapter)
            # earlystop 段（日期与 core 互斥、无泄漏）；为空则用空切分得到形状对的空 arrays。
            es_arrays = (
                build_sequence_arrays(self.raw_loader(list(es_dates)), adapter)
                if es_dates else subset_by_dates(core_arrays, ())
            )

            # Normalizer 只用训练区（core+es）统计、分块 fit（不 concatenate、不物化整张副本）。
            normalizer = Normalizer(_NORMALIZER_MODE).fit_chunked(
                [core_arrays.features, es_arrays.features]
            )
            self._normalizer = normalizer

            cartridges = []
            for seed in self.seeds:
                # own_features=False：每 seed 各自拷贝白化，绝不原地改 core_arrays（否则下一 seed 双重白化）。
                core_ds = SequenceDataset(core_arrays, self.seq_len, normalizer, own_features=False)
                es_ds = SequenceDataset(es_arrays, self.seq_len, normalizer, own_features=False)
                cart = build_cartridge(ModelConfig(kind=ModelKind.XSECTION_RAW, hparams=hp))
                _loginf("[DLServe] 训练 DL seed={} ...".format(seed))
                cart.fit(core_ds, es_ds, hp, int(seed))
                cartridges.append(cart)

            self._cartridges = cartridges
            self.available = True
            _loginf("[DLServe] DL 现训完成：{} seed 就绪（device={}）".format(len(cartridges), self._device))
        except Exception as e:  # 任何环节失败 → 降级，绝不抛给提交链
            self.available = False
            self._cartridges = []
            _logerr("[DLServe] DL 训练失败 → 本次回落纯传统：{}".format(repr(e)[:300]))

    # ---- 推理阶段：K seed 平均 ---- #
    def predict(self, eval_dates):
        """
        返回 DataFrame(date, symbol, interval, pred_dl)（K seed 平均），供 MeowEngine 对齐融合。

        不可用 / 空 / 失败一律返回 None（调用方据此用纯传统）。
        因序列 warmup，每个 (date,symbol) 的前 seq_len-1 个 interval 没有 DL 预测（属正常），
        这些行不在返回结果里，MeowEngine 对这些行用纯传统填。
        """
        if not self.available or not self._cartridges:
            return None
        try:
            from sequence_dataset import SequenceDataset, build_sequence_arrays

            arrays = build_sequence_arrays(self.raw_loader(list(eval_dates)), self._adapter)
            if arrays.n_rows == 0:
                return None

            preds = []
            label_frame = None
            for cart in self._cartridges:
                # 每 seed 一份拷贝白化的 dataset；推理顺序=label_frame 行序（卡带 predict 已映射回窗口序）。
                ds = SequenceDataset(arrays, self.seq_len, self._normalizer, own_features=False)
                p = np.asarray(cart.predict(ds), dtype=np.float64).reshape(-1)
                preds.append(p)
                if label_frame is None:
                    label_frame = ds.label_frame()
            if not preds or label_frame is None or len(label_frame) == 0:
                return None
            # 各 seed predict 同 eval 数据、同 seq_len → 同窗口集、行序一致，可直接按列平均。
            avg = np.mean(np.column_stack(preds), axis=1)
            out = label_frame[["date", "symbol", "interval"]].copy()
            out["pred_dl"] = np.asarray(avg, dtype=np.float64)
            return out
        except Exception as e:
            _logerr("[DLServe] DL 推理失败 → 本次回落纯传统：{}".format(repr(e)[:300]))
            return None


def fuse_traditional_with_dl(xdf, trad_pred, dl_df, weight_dl: float = _FUSION_WEIGHT_DL):
    """
    把传统预测与 DL 预测按 (date,symbol,interval) 等权融合，输出对齐 `xdf` 行序的一维数组。

    - `xdf`: 含 date/symbol/interval 列的特征帧（行序即最终预测要返回的顺序）。
    - `trad_pred`: 传统预测，长度/顺序与 `xdf` 一致。
    - `dl_df`: DLServe.predict 的输出 DataFrame(date,symbol,interval,pred_dl)，可能为 None 或只覆盖部分行。
    - 融合口径 = `(1-w)*trad + w*dl`（默认 w=0.5 等权，量纲留在 fret12 保 MSE/R²）。
    - DL warmup 缺的行（dl 无预测）→ 用纯传统填，绝不引入 NaN。
    """
    trad = np.asarray(trad_pred, dtype=np.float64).reshape(-1)
    if dl_df is None or len(dl_df) == 0:
        return trad.astype(np.float32)

    # 用行号保住 xdf 原始顺序：merge 后按 _row 复原（meta 每行唯一 → 1:1 左连接）。
    key = xdf[["date", "symbol", "interval"]].copy()
    key["_row"] = np.arange(len(key), dtype=np.int64)
    merged = key.merge(dl_df, on=["date", "symbol", "interval"], how="left").sort_values("_row")
    dl_aligned = merged["pred_dl"].to_numpy(dtype=np.float64)   # 缺行为 NaN

    has_dl = np.isfinite(dl_aligned)
    fused = trad.copy()
    fused[has_dl] = (1.0 - weight_dl) * trad[has_dl] + weight_dl * dl_aligned[has_dl]
    n_dl = int(has_dl.sum())
    log.inf(
        "[DLServe] 融合完成：{}/{} 行用上 DL（其余 warmup 行纯传统），融合权重 w_dl={}".format(
            n_dl, len(trad), weight_dl
        )
    )
    return fused.astype(np.float32)
