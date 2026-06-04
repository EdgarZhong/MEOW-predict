#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL-on-raw × 传统模型 融合分析（机器1 用第二台机器推来的 DL 预测做）。

目的
----
第二台机器跑的 DL-on-raw（XSECTION_RAW 卡带，直接吃 raw 59 通道）在三个窗口落了逐票预测；
本脚本把它和机器1 的传统 OOS 逐票预测按 (date,symbol,interval) inner-join，
在**同一交集行**上算（**全部复刻老师 `meow/eval.py` 口径：fillna(0) 后 Pearson / R² / MSE**）：
  - 传统单独 (Pearson, R², MSE)
  - DL 单独   (Pearson, R², MSE)
  - DL↔传统去相关度 ρ（两预测之间的 Pearson）
  - 等权融合 blend_raw = 0.5*dl + 0.5*trad —— **交付口径**（零自由参数、量纲留在 fret12，保 R²/MSE）
  - blend_z = z(dl) + z(trad) —— 纯方向融合对照（对量纲差异稳健，但会破 fret12 量纲、不能直接交付）
  - 理论最优合并相关：R = sqrt((i1²+i2²-2ρ·i1·i2)/(1-ρ²))
  - 权重扫描 w·dl+(1-w)·trad 的 Pearson —— 报最优 w（**in-sample 过拟合上界、仅诊断**）

为什么三项指标都要算
--------------------
老师评分 = pooled Pearson + R² + MSE 各 1/3。只看 Pearson 会漏掉"融合后量纲是否还健康"。
DL 腿已在训练段做 OLS rescale → 量纲落在 fret12（实测 pred std≈4.4e-4，与传统 3.8e-4 同量级），
故等权 raw 融合后量纲应仍健康；但必须把 R²/MSE 也报出来核对、不能假设。

为什么要在"同一交集行"上算
--------------------------
DL 因序列 warmup 比传统少 ~26 万行（且 DL delivery 从 Dec4 起）；若各自在自己全量行上算
corr 再比，不公平。inner-join 后传统/DL/融合都在同一行集上，融合 vs 传统的增益才可直接比较。

口径分层（务必读）
------------------
- **delivery（Dec4–Dec29，交付窗）= 主判决**：传统腿用**新传统**（含 rx_micro 两腿，
  Dec sanity pooled Pearson=0.0812）× DL delivery seed42。这一行就是"交付口径"融合分。
- **fold1 / fold2（Nov）= 跨窗稳定性参考**：传统腿当前用**旧 P2 传统**（不含 rx_micro），
  仅看"融合增益是否跨窗稳定为正"；不要把它的绝对值当交付口径（口径不同已标注）。
  后续会用新传统 expanding 预测补齐这两折，届时三窗口完全同口径。

只读 CSV、不训练、不碰 GPU；内存峰值 ~单窗两文件（~几百 MB）。
结果打印成表并落 results/dl/_blend_dl_trad/summary.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DL_DIR = REPO / "results/dl/20260603_xsection_raw_2fold_2seed_zscore/preds"
TRAD_P2_DIR = REPO / "results/dl/_p2_trad_folds"          # 旧 P2 传统（不含 rx_micro）
BLEND_DIR = REPO / "results/dl/_blend_dl_trad"            # 新传统 Dec 预测落在这里
OUT_DIR = BLEND_DIR

# 每个窗口的配置：DL 预测、传统预测、传统口径标签。
# 第二台机器命名 cert fold0=Sep28–Nov02（=传统 fold1）、cert fold1=Nov03–Nov30（=传统 fold2）、
# delivery fold2=Dec04–Dec29（=传统 delivery）。delivery 只有 seed42，三窗统一用 seed42。
WINDOWS = {
    "fold1_Sep28_Nov02": {
        "dl": DL_DIR / "preds_sweep_cert_fold0_seed42.csv",
        # 优先用新传统 expanding 预测（若已 dump），否则回退旧 P2 传统。
        "trad_new": BLEND_DIR / "trad_fold1_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_fold1_20230928_20231102.csv",
    },
    "fold2_Nov03_Nov30": {
        "dl": DL_DIR / "preds_sweep_cert_fold1_seed42.csv",
        "trad_new": BLEND_DIR / "trad_fold2_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_fold2_20231103_20231130.csv",
    },
    "delivery_Dec04_Dec29": {
        "dl": DL_DIR / "preds_sweep_delivery_fold2_seed42.csv",
        # delivery 新传统 = dump_submission_dec_preds.py 产物（含 rx_micro，自检 0.0812）。
        "trad_new": BLEND_DIR / "trad_dec_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_delivery_20231204_20231229.csv",
    },
}

META = ["date", "symbol", "interval"]


def meow_metrics(p: np.ndarray, y: np.ndarray) -> tuple:
    """
    复刻老师 `meow/eval.py` 的三指标口径：
    - 先把预测/真值的 ±inf 换 NaN、再 fillna(0)（老师 eval 前置处理）。
    - Pearson = pandas .corr()（皮尔逊）。
    - R² = 1 - SS_res / var(y, ddof=1) / N（老师写法，带 ddof=1 的 var 乘 N，N 大时≈标准 R²）。
    - MSE = SS_res / N。
    返回 (pearson, r2, mse)。
    """
    p = pd.Series(np.asarray(p, dtype=np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.Series(np.asarray(y, dtype=np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(y)
    pcor = float(p.corr(y))
    sse = float(((p - y) ** 2).sum())
    yvar = float(y.var())  # pandas 默认 ddof=1
    r2 = float(1.0 - sse / yvar / n) if (yvar > 0 and n > 0) else float("nan")
    mse = float(sse / n) if n > 0 else float("nan")
    return pcor, r2, mse


def zscore(x: np.ndarray) -> np.ndarray:
    """整列标准化（减均值除标准差）；std 退化时退回仅去均值。"""
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-15:
        return x - mu
    return (x - mu) / sd


def theory_optimal(i1: float, i2: float, rho: float) -> float:
    """两去相关信号线性合并可达的相关上界 R=sqrt((i1²+i2²-2ρ i1 i2)/(1-ρ²))。"""
    denom = 1.0 - rho * rho
    if denom <= 1e-12:
        return float("nan")
    val = (i1 * i1 + i2 * i2 - 2.0 * rho * i1 * i2) / denom
    return float(np.sqrt(val)) if val > 0 else float("nan")


def weight_sweep(pdl: np.ndarray, ptr: np.ndarray, y: np.ndarray) -> dict:
    """
    扫描融合权重 w∈[0,1]（blend=w·dl+(1-w)·trad）的 Pearson，找最优 w。

    诚实声明：最优 w 是**在 delivery 当窗挑出来的 in-sample 过拟合上界**，不可作为交付权重；
    交付一律用等权（w=0.5，零自由参数）。这里仅用于回答"等权离最优有多远"。
    """
    ws = np.linspace(0.0, 1.0, 21)
    best_w, best_ic = 0.5, -1.0
    curve = []
    for w in ws:
        ic, _, _ = meow_metrics(w * pdl + (1.0 - w) * ptr, y)
        curve.append((round(float(w), 2), round(float(ic), 5)))
        if ic > best_ic:
            best_ic, best_w = ic, float(w)
    return {"best_w": round(best_w, 3), "best_ic": round(best_ic, 5), "curve": curve}


def analyze_one(name: str, cfg: dict) -> dict:
    """对单个窗口做 inner-join 并算全套读数（三指标 + 去相关 + 融合 + 权重扫描）。"""
    dl_path = cfg["dl"]
    # 传统腿优先用新传统（含 rx_micro），没有就回退旧 P2，并记录用了哪个。
    if cfg["trad_new"].exists():
        tr_path, trad_tag = cfg["trad_new"], "new(rx_micro)"
    elif cfg["trad_old"].exists():
        tr_path, trad_tag = cfg["trad_old"], "old(P2,no rx_micro)"
    else:
        return {"window": name, "error": f"传统预测均缺失: {cfg['trad_new']} / {cfg['trad_old']}"}
    if not dl_path.exists():
        return {"window": name, "error": f"DL 预测缺失: {dl_path}"}

    dl = pd.read_csv(dl_path, usecols=META + ["label", "pred"]).rename(
        columns={"pred": "pred_dl", "label": "label_dl"}
    )
    tr = pd.read_csv(tr_path, usecols=META + ["label", "pred"]).rename(
        columns={"pred": "pred_tr", "label": "label_tr"}
    )
    m = dl.merge(tr, on=META, how="inner")
    n = len(m)
    if n == 0:
        return {"window": name, "error": "inner-join 交集为空（meta 对不齐）", "trad_tag": trad_tag}

    # 真值一致性核对：两侧 label 应是同一 fret12。
    label_max_abs_diff = float(np.nanmax(np.abs(m["label_dl"].to_numpy() - m["label_tr"].to_numpy())))
    y = m["label_dl"].to_numpy(dtype=np.float64)
    pdl = m["pred_dl"].to_numpy(dtype=np.float64)
    ptr = m["pred_tr"].to_numpy(dtype=np.float64)

    ic_dl, r2_dl, mse_dl = meow_metrics(pdl, y)
    ic_tr, r2_tr, mse_tr = meow_metrics(ptr, y)
    rho, _, _ = meow_metrics(pdl, ptr)
    ic_braw, r2_braw, mse_braw = meow_metrics(0.5 * pdl + 0.5 * ptr, y)
    ic_bz, r2_bz, mse_bz = meow_metrics(zscore(pdl) + zscore(ptr), y)
    theo = theory_optimal(ic_tr, ic_dl, rho)
    sweep = weight_sweep(pdl, ptr, y)

    return {
        "window": name,
        "trad_tag": trad_tag,
        "n_join": int(n),
        "n_dl_rows": int(len(dl)),
        "n_trad_rows": int(len(tr)),
        "label_max_abs_diff": label_max_abs_diff,
        # 三指标（老师口径）
        "trad": {"pearson": ic_tr, "r2": r2_tr, "mse": mse_tr},
        "dl": {"pearson": ic_dl, "r2": r2_dl, "mse": mse_dl},
        "blend_raw_mean": {"pearson": ic_braw, "r2": r2_braw, "mse": mse_braw},  # 交付口径
        "blend_zscore": {"pearson": ic_bz, "r2": r2_bz, "mse": mse_bz},          # 仅方向对照
        # 去相关 & 增益
        "rho_dl_trad": rho,
        "theory_optimal_pearson": theo,
        "blend_raw_pearson_vs_trad": ic_braw - ic_tr,
        # 权重扫描（诊断、过拟合上界）
        "weight_sweep": sweep,
    }


def _fmt(x, nd=4):
    return ("{:>" + str(7 + nd - 4) + "." + str(nd) + "f}").format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "    n/a"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [analyze_one(name, cfg) for name, cfg in WINDOWS.items()]

    print("\n========================= DL-on-raw × 传统 融合分析（seed42，同交集行，老师口径） =========================")
    print("{:<22} {:<16} {:>9} {:>9} {:>9} {:>9} {:>6} {:>9} {:>10} {:>8}".format(
        "window", "trad_leg", "ic_trad", "ic_dl", "ic_blend", "Δic", "rho", "r2_blend", "mse_blend", "best_w"))
    for r in rows:
        if "error" in r:
            print("{:<22} ERROR: {}".format(r["window"], r["error"]))
            continue
        print("{:<22} {:<16} {:>9.4f} {:>9.4f} {:>9.4f} {:>+9.4f} {:>6.3f} {:>9.5f} {:>10.2e} {:>8.2f}".format(
            r["window"], r["trad_tag"],
            r["trad"]["pearson"], r["dl"]["pearson"], r["blend_raw_mean"]["pearson"],
            r["blend_raw_pearson_vs_trad"], r["rho_dl_trad"],
            r["blend_raw_mean"]["r2"], r["blend_raw_mean"]["mse"], r["weight_sweep"]["best_w"]))
    print("=" * 112)
    print("ic=pooled Pearson；ic_blend=等权 raw 融合(交付口径)；Δic=融合相对传统单独增益；")
    print("rho=DL↔传统去相关度；r2_blend/mse_blend=等权融合的 R²/MSE(老师口径)；best_w=当窗最优权(过拟合上界,仅诊断)。")
    print("注：fold1/fold2 若 trad_leg=old 则口径与 delivery 不同(传统腿缺 rx_micro)，只看 Δic 跨窗是否稳定为正。")

    out = OUT_DIR / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "windows": rows,
                "seed": 42,
                "metric_convention": "复刻 meow/eval.py：fillna(0) 后 Pearson / R²(1-SSres/var(ddof=1)/N) / MSE(SSres/N)",
                "delivery_is_delivery_kou_jing": "delivery 行的 trad_leg=new(rx_micro) 即交付口径；等权 raw 融合为交付方案(零自由参数)",
                "note": "DL=XSECTION_RAW(raw 59ch, 训练段 OLS rescale)；最优权 best_w 为当窗 in-sample 上界、不可作交付权重",
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n结果已落：{out}")


if __name__ == "__main__":
    main()
