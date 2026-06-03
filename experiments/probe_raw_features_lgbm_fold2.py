"""快速验证：把"被欠用的 raw 数据"摘成归一化特征加进强传统 lgbm，看 fold2 上涨不涨。

背景（2026-06-03）：诊断发现 raw 是富 LOB+订单流，但 433 特征严重欠用——挂撤单 20 列
只压成 4 个静态比率、盘口只做不平衡摘要、成交明细几乎没用。本脚本**不碰交付特征管线**，
独立验证：在 fold2（与 DL/P2 同边界）上，比

    A = lgbm(纯 433)                  ← 复现强传统 lgbm 腿
    B = lgbm(433 + 摘取的新特征)       ← 加上挂撤单/盘口/成交明细的归一化特征

的 pooled Pearson + delta。

口径对齐：
- lgbm 走 `ExperimentRunner._fit_model_core("lgbm", ...)`，参数 = 提交链 M_lgbm_d4
  （max_depth=4, num_leaves=15），target/winsorize 与交付完全一致；
- 评分 = pooled Pearson（与 meow/eval.py、experiment_runner.evaluate_predictions 同口径）；
- 折由 `build_dl_folds` 派生（与 DL/传统 P2 逐字节同边界）。

**新特征全部归一化/比率/相对量 → 跨票跨时间可比、平稳 → 泛化优先，绝不喂原始量纲值。**

用法：python experiments/probe_raw_features_lgbm_fold2.py
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# meow 置最前：保证裸 import 的 dl 解析到 meow/dl.py（带 loadDate/date 列）。
for d in ("config", "models", "src", "meow"):
    sys.path.insert(0, os.path.join(REPO, d))

from dl import MeowDataLoader                       # meow/dl.py
from submission_pipeline import SubmissionFeaturePipeline, DEFAULT_SUBMISSION_GROUPS
from experiment_runner import ExperimentRunner
from dl_protocol import build_dl_folds

META = ["date", "symbol", "interval"]
EPS = 1e-9


def _safe_div(a, b):
    return np.asarray(a, dtype=np.float64) / (np.asarray(b, dtype=np.float64) + EPS)


def build_new_features(raw: pd.DataFrame) -> pd.DataFrame:
    """从 raw 摘取被 433 欠用的归一化特征（挂撤单为主 + 盘口形状 + 成交明细）。

    全部为比率/不平衡/相对量/归一化统计 → 跨票跨时间可比、平稳。
    时序项（EMA/变化率）按 (date,symbol) 组内、日内因果、不跨日跨票。
    """
    # 新特征统一加 rx_ 前缀，避免与 433 既有列（如 add_imb/cxl_imb/buy_vwad_gap）重名。
    df = raw.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({c: df[c].to_numpy() for c in META})
    mid = df["midpx"].to_numpy(dtype=np.float64)
    safe_mid = np.where(np.abs(mid) < EPS, np.nan, mid)

    # add_imb / cxl_imb 已在 433（build_base），这里只作中间量算"新"信号，不重复输出。
    add_imb = _safe_div(df["nAddBuy"] - df["nAddSell"], df["nAddBuy"] + df["nAddSell"])
    cxl_imb = _safe_div(df["nCxlBuy"] - df["nCxlSell"], df["nCxlBuy"] + df["nCxlSell"])

    # ===== ① 挂撤单（最欠用：20 列只做过 4 比率）=====
    out["rx_cxl_rate_cnt"] = _safe_div(df["nCxlBuy"] + df["nCxlSell"], df["nAddBuy"] + df["nAddSell"])  # 撤单率（毒性）
    out["rx_cxl_rate_qty"] = _safe_div(df["cxlBuyQty"] + df["cxlSellQty"], df["addBuyQty"] + df["addSellQty"])
    out["rx_add_imb_to"] = _safe_div(df["addBuyTurnover"] - df["addSellTurnover"], df["addBuyTurnover"] + df["addSellTurnover"])
    out["rx_cxl_imb_to"] = _safe_div(df["cxlBuyTurnover"] - df["cxlSellTurnover"], df["cxlBuyTurnover"] + df["cxlSellTurnover"])
    out["rx_net_order_press"] = add_imb - cxl_imb   # 挂买多+撤买少=真实买压
    trd = df["nTradeBuy"].to_numpy(dtype=np.float64) + df["nTradeSell"].to_numpy(dtype=np.float64)
    out["rx_add_vs_trade"] = np.log1p(_safe_div(df["nAddBuy"] + df["nAddSell"], trd))
    out["rx_cxl_vs_trade"] = np.log1p(_safe_div(df["nCxlBuy"] + df["nCxlSell"], trd))

    # ===== ② 盘口形状（433 只做了不平衡摘要 obi/ofi，没建形状）=====
    asz = df["asize0"].to_numpy(dtype=np.float64); bsz = df["bsize0"].to_numpy(dtype=np.float64)
    asz04 = df["asize0_4"].to_numpy(dtype=np.float64); bsz04 = df["bsize0_4"].to_numpy(dtype=np.float64)
    asz59 = df["asize5_9"].to_numpy(dtype=np.float64); bsz59 = df["bsize5_9"].to_numpy(dtype=np.float64)
    asz1019 = df["asize10_19"].to_numpy(dtype=np.float64); bsz1019 = df["bsize10_19"].to_numpy(dtype=np.float64)
    obi0 = _safe_div(asz - bsz, asz + bsz)
    obi4 = _safe_div(asz04 - bsz04, asz04 + bsz04)
    obi9 = _safe_div(asz59 - bsz59, asz59 + bsz59)
    obi19 = _safe_div(asz1019 - bsz1019, asz1019 + bsz1019)   # 最深档（433 的 obi 没到这档）
    out["rx_obi19"] = obi19
    out["rx_obi_weighted"] = 0.4 * obi0 + 0.3 * obi4 + 0.2 * obi9 + 0.1 * obi19
    out["rx_depth_nf_bid"] = np.log1p(_safe_div(bsz04, bsz1019))   # 近/远档深度比
    out["rx_depth_nf_ask"] = np.log1p(_safe_div(asz04, asz1019))
    spr0 = df["ask0"].to_numpy(dtype=np.float64) - df["bid0"].to_numpy(dtype=np.float64)
    spr4 = df["ask4"].to_numpy(dtype=np.float64) - df["bid4"].to_numpy(dtype=np.float64)
    out["rx_spread_deep_rel"] = _safe_div(spr4 - spr0, safe_mid)   # 深档价差相对收紧

    # ===== ③ 成交明细（buyVwad/sellVwad/High/Low 几乎没用）=====
    out["rx_vwad_gap_buy"] = _safe_div(df["buyVwad"].to_numpy(dtype=np.float64) - mid, safe_mid)
    out["rx_vwad_gap_sell"] = _safe_div(df["sellVwad"].to_numpy(dtype=np.float64) - mid, safe_mid)
    out["rx_trade_range_buy"] = _safe_div(df["tradeBuyHigh"].to_numpy(dtype=np.float64) - df["tradeBuyLow"].to_numpy(dtype=np.float64), safe_mid)
    out["rx_trade_range_sell"] = _safe_div(df["tradeSellHigh"].to_numpy(dtype=np.float64) - df["tradeSellLow"].to_numpy(dtype=np.float64), safe_mid)

    # 静态项清 NaN/inf（无成交导致）
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if c not in META:
            out[c] = out[c].fillna(0.0)

    # ===== ④ 时序项（日内因果，按 (date,symbol) 组内）=====
    out["_add_imb_tmp"] = add_imb   # 临时列供组内 diff，算完即丢
    g = out.groupby(["date", "symbol"], sort=False)
    out["rx_net_press_ema5"] = g["rx_net_order_press"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_cxl_rate_ema5"] = g["rx_cxl_rate_cnt"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_obi_w_ema5"] = g["rx_obi_weighted"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_add_imb_chg"] = g["_add_imb_tmp"].transform(lambda s: s.diff()).fillna(0.0)
    out = out.drop(columns=["_add_imb_tmp"])

    # ============================================================ #
    # 第二批（rx2_）：盘口微观结构深挖 + 多时间尺度 + 波动率 + 交互
    # ============================================================ #
    # 微观价格偏离：对侧量加权 microprice 减 mid —— 经典短期方向信号（卖压大→micro 偏 bid）
    micro = _safe_div(df["bid0"].to_numpy(np.float64) * asz + df["ask0"].to_numpy(np.float64) * bsz, asz + bsz)
    out["rx2_microprice_dev"] = _safe_div(micro - mid, safe_mid)
    # 全档量不平衡 + 流动性集中度（最优档量占总深度的比）
    tot_b = bsz + bsz04 + bsz59 + bsz1019
    tot_a = asz + asz04 + asz59 + asz1019
    out["rx2_depth_imb_all"] = _safe_div(tot_b - tot_a, tot_b + tot_a)
    out["rx2_conc_bid"] = _safe_div(bsz, tot_b)
    out["rx2_conc_ask"] = _safe_div(asz, tot_a)
    out["rx2_obi_term"] = obi19 - obi0                                   # 盘口不平衡的深浅期限结构
    out["rx2_spread_rel"] = _safe_div(df["ask0"].to_numpy(np.float64) - df["bid0"].to_numpy(np.float64), safe_mid)
    # 成交不平衡（中间量）→ 与净挂撤压力的交互（订单流一致性）
    trade_imb = _safe_div(df["tradeBuyQty"].to_numpy(np.float64) - df["tradeSellQty"].to_numpy(np.float64),
                          df["tradeBuyQty"].to_numpy(np.float64) + df["tradeSellQty"].to_numpy(np.float64))
    out["rx2_press_x_tradeimb"] = out["rx_net_order_press"].to_numpy(np.float64) * trade_imb

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in [c for c in out.columns if c.startswith("rx2_")]:
        out[c] = out[c].fillna(0.0)

    # 多时间尺度时序（z-score = 当前值相对近窗的标准化异常；长窗 EMA；已实现波动）
    out["_mid_tmp"] = mid
    g = out.groupby(["date", "symbol"], sort=False)

    def _z(col, w):
        return g[col].transform(
            lambda s: (s - s.rolling(w, min_periods=2).mean()) / (s.rolling(w, min_periods=2).std(ddof=0) + EPS)
        ).fillna(0.0)

    out["rx2_net_press_z12"] = _z("rx_net_order_press", 12)
    out["rx2_obi_w_z12"] = _z("rx_obi_weighted", 12)
    out["rx2_cxl_rate_z12"] = _z("rx_cxl_rate_cnt", 12)
    out["rx2_micro_dev_z12"] = _z("rx2_microprice_dev", 12)
    out["rx2_net_press_ema12"] = g["rx_net_order_press"].transform(lambda s: s.ewm(halflife=12, adjust=False).mean()).fillna(0.0)
    # 已实现波动（mid 日内对数收益的 rolling std）+ 与撤单率交互
    out["_logret_tmp"] = g["_mid_tmp"].transform(lambda s: np.log(s.clip(lower=EPS)).diff()).fillna(0.0)
    g2 = out.groupby(["date", "symbol"], sort=False)
    out["rx2_rvol12"] = g2["_logret_tmp"].transform(lambda s: s.rolling(12, min_periods=2).std(ddof=0)).fillna(0.0)
    out["rx2_cxl_x_rvol"] = out["rx_cxl_rate_cnt"].to_numpy(np.float64) * out["rx2_rvol12"].to_numpy(np.float64)
    out = out.drop(columns=["_mid_tmp", "_logret_tmp"])

    return out.astype({c: np.float32 for c in out.columns if c not in META})


def _build_window(loader, pipeline, dates, symbols=None):
    """逐日 loadDate 拼 raw → 算 433 + 新特征 → merge on (date,symbol,interval)。

    symbols 非 None 时只保留这组票（控内存：全 309 票全窗 433 帧 concat 峰值 ~22GB OOM，
    抽样后压到一半内）。train/score 用同一组票，A/B 用同一组 → delta 仍有效。
    """
    raw = pd.concat([loader.loadDate(int(d)) for d in dates], ignore_index=True)
    if symbols is not None:
        raw = raw[raw["symbol"].isin(symbols)].reset_index(drop=True)
    x433, y = pipeline.build_feature_frames(raw)
    xnew = build_new_features(raw)
    del raw
    gc.collect()
    feat433 = [c for c in x433.columns if c not in META]
    newcols = [c for c in xnew.columns if c not in META]
    x = x433.merge(xnew, on=META, how="left")
    del x433, xnew
    gc.collect()
    return x, y, feat433, newcols


def _pooled_pearson(pred, y):
    p = np.asarray(pred, dtype=np.float64); yy = np.asarray(y, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(yy)
    return float(np.corrcoef(p[m], yy[m])[0, 1]) if m.sum() > 1 else float("nan")


def main():
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(REPO, "data"))
    t0 = time.time()

    # ---- fold2：与 DL/P2 同边界（recent 单折）----
    folds = build_dl_folds(20230601, 20231130, mode="expanding", val_window=20, step=20,
                           embargo=1, min_train_days=40, max_folds=1, fold_select="recent")
    fold = folds[0]
    print(f"[probe] fold2 训练 {fold.train_start}-{fold.train_end}（{len(fold.train_dates)}日）"
          f" → 打分 {fold.val_start}-{fold.val_end}（{len(fold.scoring_dates)}日）", flush=True)

    loader = MeowDataLoader(h5dir=h5dir)
    pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
    runner = ExperimentRunner(h5dir=h5dir)
    lgbm_params = {"max_depth": 4, "num_leaves": 15, "n_jobs": 8}   # = M_lgbm_d4

    # ---- 控内存：抽样固定一组票（train/score 同组，A/B 同组 → delta 有效）----
    max_symbols = int(os.environ.get("PROBE_MAX_SYMBOLS", "180"))
    raw0 = loader.loadDate(int(fold.train_dates[0]))
    all_syms = np.sort(raw0["symbol"].unique()); del raw0; gc.collect()
    if max_symbols and max_symbols < len(all_syms):
        symbols = set(np.random.default_rng(42).choice(all_syms, size=max_symbols, replace=False).tolist())
        print(f"[probe] 抽样 {len(symbols)}/{len(all_syms)} 票（固定 seed42，控内存；A 因此不是全票 0.0904，看 delta）", flush=True)
    else:
        symbols = None
        print(f"[probe] 全 {len(all_syms)} 票", flush=True)

    # ---- 训练窗：build → fit A(433) / B(433+new) ----
    print("[probe] 构造训练窗特征 ...", flush=True)
    xtr, ytr, feat433, newcols = _build_window(loader, pipeline, fold.train_dates, symbols=symbols)
    print(f"[probe] 训练 {len(xtr)} 行 | 433 列={len(feat433)} | 新列={len(newcols)}：{newcols}", flush=True)

    print("[probe] fit A = lgbm(纯433) ...", flush=True)
    xa = xtr[feat433].to_numpy(dtype=np.float32)
    model_a, fcols_a, base_a = runner._fit_model_core("lgbm", xa, feat433, ytr, target_mode="raw", model_params=lgbm_params)
    del xa; gc.collect()

    print("[probe] fit B = lgbm(433+新) ...", flush=True)
    feat_all = feat433 + newcols
    xb = xtr[feat_all].to_numpy(dtype=np.float32)
    model_b, fcols_b, base_b = runner._fit_model_core("lgbm", xb, feat_all, ytr, target_mode="raw", model_params=lgbm_params)
    del xb, xtr, ytr; gc.collect()

    # ---- 打分窗：build → predict A/B ----
    print("[probe] 构造打分窗特征 + predict ...", flush=True)
    xsc, ysc, _, _ = _build_window(loader, pipeline, fold.scoring_dates, symbols=symbols)
    yv = ysc["fret12"].to_numpy(dtype=np.float64)
    pred_a = runner._predict_with_baseline(model_a, xsc, fcols_a, ydf=None, baseline=base_a, target_mode="raw")
    pred_b = runner._predict_with_baseline(model_b, xsc, fcols_b, ydf=None, baseline=base_b, target_mode="raw")
    pa, pb = _pooled_pearson(pred_a, yv), _pooled_pearson(pred_b, yv)

    # ---- 新特征在 B 里的重要性（看 lgbm 到底用没用）----
    imp = np.asarray(model_b.feature_importances_, dtype=np.float64)
    imp = imp / (imp.sum() + EPS)
    new_imp = sorted([(feat_all[i], imp[i]) for i in range(len(feat_all)) if feat_all[i] in newcols],
                     key=lambda kv: -kv[1])

    print("\n" + "=" * 60, flush=True)
    print(f"fold2 打分窗 {fold.val_start}-{fold.val_end}  行={len(xsc)}", flush=True)
    print(f"  A  lgbm(纯433)      pooled Pearson = {pa:.4f}", flush=True)
    print(f"  B  lgbm(433+新)     pooled Pearson = {pb:.4f}", flush=True)
    print(f"  Δ (B - A)          = {pb - pa:+.4f}", flush=True)
    print(f"  (传统融合代表在本折 = 0.0904，仅背景参考)", flush=True)
    print(f"  新特征重要性占比 top（B 是否真用了新特征）：", flush=True)
    for name, w in new_imp[:10]:
        print(f"     {name:22s} {w*100:5.2f}%", flush=True)
    print(f"  新特征重要性合计 = {sum(w for _, w in new_imp)*100:.1f}%", flush=True)
    print(f"  总耗时 {time.time()-t0:.0f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
