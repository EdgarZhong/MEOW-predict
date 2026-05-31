"""
SequenceTrainer —— DL 序列模型的 Trainer 骨架（脊柱编排层，torch-free）

接 ``trainer.BaseTrainer`` 扩展点（与传统 ``TabularTrainer`` 同构），把"可换卡带"的
data pipeline + model cartridge 编排成一次完整的 fold 训练-评测，产出与 leaderboard
兼容的 ``FoldResult``。

编排顺序（规格 §2.2）：

    raw_loader(dates) → InputAdapter.build（逐日）→ SequenceArrays
      → Normalizer.fit-on-train → SequenceDataset（惰性 [B,L,C]）
      → ModelCartridge.fit(train_core, earlystop) / predict(scoring)
      → dl_protocol 4 指标 → FoldResult

解耦策略：本层**只靠鸭子类型**认 adapter（有 ``channels`` / ``build``）与 cartridge
（有 ``fit`` / ``predict``），不 import ``models/`` / ``config/`` 具体类——保持 src 层
torch-free 且不反向依赖卡带目录。具体卡带/适配器由 Orchestrator 用 registry 构造后注入。

防泄漏的三道物理保证全部落在本编排里：
1. **窗口不跨日不跨票**（WindowIndexer，在 SequenceDataset 内）；
2. **标签因果对齐窗末**（同上）；
3. **Normalizer 只用训练区 fit**（本层显式只喂 train 区特征，再套到 scoring）。
"""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from trainer import BaseTrainer, FoldData, FoldResult
from sequence_dataset import (
    Normalizer,
    SequenceArrays,
    SequenceDataset,
    build_sequence_arrays,
    subset_by_dates,
)
from dl_protocol import DLFold, corr_gap, evaluate_prediction_bundle, split_train_earlystop


class SequenceTrainer(BaseTrainer):
    """
    序列模型 Trainer。两条入口：

    - ``run_on_dl_fold(fold: DLFold)``：DL 主路径，吃 protocol 切好的四段折（含 earlystop）。
    - ``run_fold(fold_data: FoldData)``：BaseTrainer 兼容入口，用 ``earlystop_frac`` 从
      ``train_dates`` 尾段切 earlystop 再委托（让调度层可像传统 trainer 一样调它）。
    """

    def __init__(
        self,
        spec: Dict,
        adapter,
        cartridge_factory: Callable[[], object],
        raw_loader: Callable[[Sequence[int]], pd.DataFrame],
        seq_len: int,
        normalizer_mode: str = "zscore",
        hparams: Optional[Dict] = None,
        seed: int = 42,
        earlystop_frac: float = 0.15,
    ):
        """
        - ``spec``: 给 FoldResult 的元信息（experiment_id / model_type / feature_set /
          target_type / postprocess_type / notes）。
        - ``adapter``: 已构造的 InputAdapter 实例。
        - ``cartridge_factory``: 无参可调用，**每折新建**一个 ModelCartridge（避免折间状态串）。
        - ``raw_loader``: ``dates -> raw DataFrame``（含 META + raw 列 + fret12）。
        - ``seq_len`` / ``normalizer_mode`` / ``hparams`` / ``seed`` / ``earlystop_frac``: 见上。
        """
        super().__init__(spec)
        self.adapter = adapter
        self.cartridge_factory = cartridge_factory
        self.raw_loader = raw_loader
        self.seq_len = int(seq_len)
        self.normalizer_mode = normalizer_mode
        self.hparams = dict(hparams or {})
        self.seed = int(seed)
        self.earlystop_frac = float(earlystop_frac)

    # ---- 数据 ---- #
    def _build_arrays(self, dates: Sequence[int]) -> SequenceArrays:
        raw = self.raw_loader(list(dates))
        return build_sequence_arrays(raw, self.adapter)

    # ---- DL 主入口 ---- #
    def run_on_dl_fold(self, fold: DLFold, profile_name: str = "dl") -> FoldResult:
        start_ts = time.time()
        try:
            # 1) 训练区一次现算，按 date 掩码拆 core / earlystop（不重算特征）。
            train_arrays = self._build_arrays(fold.train_dates)
            # 2) Normalizer 只用训练区（core+es 都属训练区，无泄漏）fit，再套到全部。
            normalizer = Normalizer(self.normalizer_mode).fit(train_arrays.features)
            core_arrays = subset_by_dates(train_arrays, fold.train_core_dates)
            es_arrays = subset_by_dates(train_arrays, fold.earlystop_dates)  # 空 dates → 空 arrays
            score_arrays = self._build_arrays(fold.scoring_dates)

            train_core_ds = SequenceDataset(core_arrays, self.seq_len, normalizer)
            earlystop_ds = SequenceDataset(es_arrays, self.seq_len, normalizer)
            scoring_ds = SequenceDataset(score_arrays, self.seq_len, normalizer)

            # 3) 每折新建卡带，fit（earlystop 看尾段、绝不碰 scoring），predict。
            cartridge = self.cartridge_factory()
            record = cartridge.fit(train_core_ds, earlystop_ds, self.hparams, self.seed)
            pred_val = cartridge.predict(scoring_ds)
            pred_train = cartridge.predict(train_core_ds)

            # 4) 4 指标（脊柱在 predict 之后算一次；scoring 在此前一次不碰）。
            vm = evaluate_prediction_bundle(scoring_ds.label_frame(), pred_val)
            tm = evaluate_prediction_bundle(train_core_ds.label_frame(), pred_train)

            notes = self.spec.get("notes", "")
            best_epoch = getattr(record, "best_epoch", 0)
            n_epochs = getattr(record, "n_epochs", 0)
            # 曲线序列化为 JSON 字符串（保留 6 位小数，节省空间）
            train_curve_json = json.dumps(
                [round(v, 6) for v in (record.train_curve or [])]) if hasattr(record, "train_curve") else "[]"
            es_curve_json = json.dumps(
                [round(v, 6) for v in (record.earlystop_curve or [])]) if hasattr(record, "earlystop_curve") else "[]"
            return FoldResult(
                profile_name=profile_name,
                fold_id=fold.fold_id,
                experiment_id=self.spec.get("experiment_id", "dl_run"),
                feature_set=self.spec.get("feature_set", "dl_channels"),
                model_type=self.spec.get("model_type", "sequence"),
                target_type=self.spec.get("target_type", "raw"),
                postprocess_type=self.spec.get("postprocess_type", "none"),
                train_corr=float(tm["corr"]), val_corr=float(vm["corr"]),
                train_mse=float(tm["mse"]), val_mse=float(vm["mse"]),
                train_r2=float(tm["r2"]), val_r2=float(vm["r2"]),
                daily_corr_mean=float(vm["daily_corr_mean"]),
                daily_corr_std=float(vm["daily_corr_std"]),
                train_val_corr_gap=corr_gap(tm, vm),
                runtime_sec=float(time.time() - start_ts),
                train_start=fold.train_start, train_end=fold.train_end,
                val_start=fold.val_start, val_end=fold.val_end,
                n_train_days=len(fold.train_dates), n_val_days=len(fold.scoring_dates),
                status="ok", error_msg="",
                notes=notes,
                best_epoch=best_epoch,
                n_epochs=n_epochs,
                train_curve=train_curve_json,
                earlystop_curve=es_curve_json,
            )
        except Exception as e:  # 单折失败不拖垮整轮，记 error 供 resume（同 TabularTrainer）。
            nan = float("nan")
            return FoldResult(
                profile_name=profile_name,
                fold_id=fold.fold_id,
                experiment_id=self.spec.get("experiment_id", "dl_run"),
                feature_set=self.spec.get("feature_set", "dl_channels"),
                model_type=self.spec.get("model_type", "sequence"),
                target_type=self.spec.get("target_type", "raw"),
                postprocess_type=self.spec.get("postprocess_type", "none"),
                train_corr=nan, val_corr=nan, train_mse=nan, val_mse=nan,
                train_r2=nan, val_r2=nan, daily_corr_mean=nan, daily_corr_std=nan,
                train_val_corr_gap=nan, runtime_sec=float(time.time() - start_ts),
                train_start=fold.train_start, train_end=fold.train_end,
                val_start=fold.val_start, val_end=fold.val_end,
                n_train_days=len(fold.train_dates), n_val_days=len(fold.scoring_dates),
                status="error", error_msg=str(e)[:500],
                notes=self.spec.get("notes", ""),
            )

    # ---- BaseTrainer 兼容入口 ---- #
    def run_fold(self, fold_data: FoldData) -> FoldResult:
        core, es = split_train_earlystop(fold_data.train_dates, self.earlystop_frac)
        fold = DLFold(
            fold_id=fold_data.fold_id,
            train_core_dates=tuple(core),
            earlystop_dates=tuple(es),
            embargo_dates=tuple(),                 # FoldData 生成时已在 train/val 间留 embargo
            scoring_dates=tuple(fold_data.val_dates),
        )
        return self.run_on_dl_fold(fold, profile_name=fold_data.profile_name)
