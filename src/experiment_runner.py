import argparse
import gc
import os
import json
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dl import MeowDataLoader
from log import log
from tradingcalendar import Calendar

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from lightgbm import LGBMRegressor
except ImportError:  # LightGBM is optional in this workspace.
    LGBMRegressor = None


EPS = 1e-8
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass(frozen=True)
class SplitConfig:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class RollingFold:
    fold_id: int
    train_dates: tuple
    val_dates: tuple


class FeatureBuilder(object):
    def __init__(self):
        self.meta_cols = ["date", "symbol", "interval"]
        self.target_col = "fret12"

    def build(self, df):
        df = df.copy()
        df = df.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
        base = self._add_base_features(df)
        working = pd.concat([df[self.meta_cols + [self.target_col, "midpx"]], base], axis=1)
        raw_inputs = df[[
            "bid0",
            "ask0",
            "bid4",
            "ask4",
            "bid9",
            "ask9",
            "bid19",
            "ask19",
            "bsize0",
            "asize0",
            "bsize0_4",
            "asize0_4",
            "bsize5_9",
            "asize5_9",
            "bsize10_19",
            "asize10_19",
            "nTradeBuy",
            "tradeBuyQty",
            "tradeBuyTurnover",
            "nTradeSell",
            "tradeSellQty",
            "tradeSellTurnover",
            "nAddBuy",
            "addBuyQty",
            "nAddSell",
            "addSellQty",
            "nCxlBuy",
            "cxlBuyQty",
            "nCxlSell",
            "cxlSellQty",
            "buyVwad",
            "sellVwad",
        ]].copy()
        helper_base = pd.concat([working, raw_inputs], axis=1)
        lag = self._add_lag_features(helper_base)
        roll = self._add_roll_features(pd.concat([helper_base, lag], axis=1))
        patch = self._add_patch_summary_features(pd.concat([helper_base, lag, roll], axis=1))
        ofi = self._add_ofi_features(pd.concat([helper_base, lag, roll, patch], axis=1))
        trade_impact = self._add_trade_impact_features(pd.concat([helper_base, lag, roll, patch, ofi], axis=1))
        cross = self._add_cross_section_features(pd.concat([helper_base, lag, roll, patch, ofi, trade_impact], axis=1))
        conditional_momentum = self._add_conditional_momentum_features(pd.concat([helper_base, lag, roll, patch, ofi, trade_impact, cross], axis=1))
        regime = self._add_regime_features(pd.concat([helper_base, lag, roll, patch, ofi, trade_impact, cross, conditional_momentum], axis=1))
        feature_frames = [working[self.meta_cols + [self.target_col]].copy(), base, lag, roll, patch, ofi, trade_impact, cross, conditional_momentum, regime]
        out = pd.concat(feature_frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        xdf = out.drop(columns=[self.target_col]).copy()
        ydf = out[self.meta_cols + [self.target_col]].copy()
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in xdf.columns:
            if col not in self.meta_cols:
                xdf[col] = pd.to_numeric(xdf[col], errors="coerce").astype(np.float32)
        ydf[self.target_col] = pd.to_numeric(ydf[self.target_col], errors="coerce").astype(np.float32)
        return xdf, ydf

    def select_groups(self, xdf, groups):
        group_map = {
            "legacy": [
                "ob_imb0",
                "ob_imb4",
                "ob_imb9",
                "trade_imb",
                "trade_imbema5",
                "lagret12",
            ],
            "base": [
                "spread",
                "mid_ret1_raw",
                "obi0",
                "obi4",
                "obi9",
                "trade_imb",
                "trade_turnover_imb",
                "add_imb",
                "cxl_imb",
                "qty_add_imb",
                "qty_cxl_imb",
                "buy_vwad_gap",
                "sell_vwad_gap",
                "trade_activity",
                "order_pressure",
            ],
            "lag": [c for c in xdf.columns if "_lag_" in c],
            "roll": [c for c in xdf.columns if "_rm" in c or "_rs" in c],
            "roll_short": [c for c in xdf.columns if any(token in c for token in ["_rm3", "_rm5", "_rs3", "_rs5"])],
            "roll_mid": [c for c in xdf.columns if any(token in c for token in ["_rm10", "_rs10"])],
            "roll_long": [c for c in xdf.columns if any(token in c for token in ["_rm20", "_rm30", "_rs20", "_rs30"])],
            "patch": [c for c in xdf.columns if "_patch" in c],
            "patch_summary": [c for c in xdf.columns if "_patch" in c],
            "ofi": [c for c in xdf.columns if c.startswith("ofi_") or c.startswith("bid_ofi_") or c.startswith("ask_ofi_") or c.startswith("ofi_total")],
            "ofi_raw": [c for c in xdf.columns if c.startswith("bid_ofi_") or c.startswith("ask_ofi_") or c in ["ofi_0", "ofi_4", "ofi_9", "ofi_19", "ofi_total"]],
            "ofi_dynamic": [c for c in xdf.columns if any(token in c for token in ["_ema", "_sum", "_mean", "_z"]) and (c.startswith("ofi_") or c.startswith("bid_ofi_") or c.startswith("ask_ofi_"))],
            "ofi_rank": [c for c in xdf.columns if c.startswith("ofi_") and c.endswith("_cs_rank")],
            "ofi_safe": [c for c in xdf.columns if c.startswith("ofi_") or c.startswith("bid_ofi_") or c.startswith("ask_ofi_")],
            "trade_impact": [c for c in xdf.columns if c.startswith("trade_pressure_") or c.startswith("trade_intensity") or c.startswith("avg_trade_") or c.startswith("signed_trade_") or c.startswith("trade_impact_")],
            "trade_impact_dyn": [c for c in xdf.columns if any(token in c for token in ["_ema", "_sum", "_mean", "_z", "_cs_rank"]) and (c.startswith("trade_pressure_") or c.startswith("trade_intensity") or c.startswith("avg_trade_"))],
            "trade_impact_interaction": [c for c in xdf.columns if c.startswith("trade_pressure_x_") or c.startswith("trade_intensity_x_") or c.startswith("avg_trade_x_")],
            "trade_impact_safe": [c for c in xdf.columns if c.startswith("trade_pressure_") or c.startswith("trade_intensity") or c.startswith("avg_trade_") or c.startswith("signed_trade_") or c.startswith("trade_impact_") or c.startswith("trade_pressure_x_") or c.startswith("trade_intensity_x_") or c.startswith("avg_trade_x_")],
            "conditional_momentum": [c for c in xdf.columns if c.startswith("lagret") or c.startswith("momentum_") or c.startswith("reversal_") or c.startswith("conditional_")],
            "conditional_momentum_interaction": [c for c in xdf.columns if c.startswith("lagret") and ("_x_" in c or "_cond" in c or "_state" in c)],
            "conditional_momentum_safe": [c for c in xdf.columns if c.startswith("lagret") or c.startswith("momentum_") or c.startswith("reversal_") or c.startswith("conditional_")],
            "cross_z": [c for c in xdf.columns if c.endswith("_cs_z")],
            "cross_rank": [c for c in xdf.columns if c.endswith("_cs_rank")],
            "cross_rank_features": [c for c in xdf.columns if c.endswith("_cs_rank")],
            "norm_core": [c for c in xdf.columns if c in ["spread", "mid_ret1_raw", "obi0", "obi4", "obi9", "trade_imb", "trade_turnover_imb", "add_imb", "cxl_imb", "qty_add_imb", "qty_cxl_imb", "buy_vwad_gap", "sell_vwad_gap", "trade_activity", "order_pressure"] or c.endswith("_cs_z")],
            "lag_short": [c for c in xdf.columns if any(token in c for token in ["_lag_1", "_lag_3", "_lag_5"])],
            "lag_mid": [c for c in xdf.columns if any(token in c for token in ["_lag_10"])],
            "lag_long": [c for c in xdf.columns if any(token in c for token in ["_lag_20", "_lag_30"])],
            "cross": [c for c in xdf.columns if "_cs_" in c or c in ["interval_pos", "interval_norm", "is_morning", "is_afternoon"]],
            "regime": [c for c in xdf.columns if c.startswith("regime_") or c.startswith("state_")],
        }
        selected = list(self.meta_cols)
        for group in groups:
            selected.extend(group_map[group])
        selected = [c for c in selected if c in xdf.columns]
        return xdf[selected].copy(deep=False)

    def _safe_div(self, a, b):
        return a / (b.abs() + EPS)

    def _add_base_features(self, df):
        out = pd.DataFrame(index=df.index)
        out["spread"] = df["ask0"] - df["bid0"]
        out["mid_ret1_raw"] = df.groupby(["date", "symbol"], sort=False)["midpx"].pct_change().fillna(0.0)
        out["obi0"] = self._safe_div(df["asize0"] - df["bsize0"], df["asize0"] + df["bsize0"])
        out["obi4"] = self._safe_div(df["asize0_4"] - df["bsize0_4"], df["asize0_4"] + df["bsize0_4"])
        out["obi9"] = self._safe_div(df["asize5_9"] - df["bsize5_9"], df["asize5_9"] + df["bsize5_9"])
        out["trade_imb"] = self._safe_div(df["tradeBuyQty"] - df["tradeSellQty"], df["tradeBuyQty"] + df["tradeSellQty"])
        out["trade_imbema5"] = out.groupby(df["symbol"], sort=False)["trade_imb"].transform(
            lambda s: s.ewm(halflife=5, adjust=False).mean()
        ).fillna(0.0)
        out["trade_turnover_imb"] = self._safe_div(
            df["tradeBuyTurnover"] - df["tradeSellTurnover"],
            df["tradeBuyTurnover"] + df["tradeSellTurnover"],
        )
        out["add_imb"] = self._safe_div(df["nAddBuy"] - df["nAddSell"], df["nAddBuy"] + df["nAddSell"])
        out["cxl_imb"] = self._safe_div(df["nCxlBuy"] - df["nCxlSell"], df["nCxlBuy"] + df["nCxlSell"])
        out["qty_add_imb"] = self._safe_div(df["addBuyQty"] - df["addSellQty"], df["addBuyQty"] + df["addSellQty"])
        out["qty_cxl_imb"] = self._safe_div(df["cxlBuyQty"] - df["cxlSellQty"], df["cxlBuyQty"] + df["cxlSellQty"])
        out["buy_vwad_gap"] = df["buyVwad"] - df["midpx"]
        out["sell_vwad_gap"] = df["sellVwad"] - df["midpx"]
        out["bret12"] = df.groupby("symbol", sort=False)["midpx"].pct_change(12).fillna(0.0)
        cxbret = pd.DataFrame({
            "interval": df["interval"],
            "bret12": out["bret12"],
        }).groupby("interval", sort=False)[["bret12"]].transform("mean")
        out["lagret12"] = out["bret12"] - cxbret["bret12"]
        out["ob_imb0"] = out["obi0"]
        out["ob_imb4"] = out["obi4"]
        out["ob_imb9"] = out["obi9"]
        out["trade_activity"] = (
            df["nTradeBuy"].fillna(0.0)
            + df["nTradeSell"].fillna(0.0)
            + df["tradeBuyQty"].fillna(0.0)
            + df["tradeSellQty"].fillna(0.0)
        )
        out["order_pressure"] = out["obi0"] + 0.5 * out["obi4"] + 0.5 * out["trade_imb"]
        return out

    def _add_lag_features(self, df):
        out = pd.DataFrame(index=df.index)
        for lag in [1, 3, 5, 10, 20, 30]:
            group = df.groupby(["date", "symbol"], sort=False)
            out[f"mid_ret_lag_{lag}"] = group["midpx"].pct_change(lag).fillna(0.0)
            out[f"obi0_lag_{lag}"] = group["obi0"].shift(lag).fillna(0.0)
            out[f"trade_imb_lag_{lag}"] = group["trade_imb"].shift(lag).fillna(0.0)
            out[f"spread_lag_{lag}"] = group["spread"].shift(lag).fillna(0.0)
        return out

    def _add_roll_features(self, df):
        out = pd.DataFrame(index=df.index)
        base_cols = ["mid_ret1_raw", "obi0", "trade_imb", "order_pressure", "spread"]
        for window in [3, 5, 10, 20, 30]:
            for col in base_cols:
                group = df.groupby(["date", "symbol"], sort=False)[col]
                out[f"{col}_rm{window}"] = (
                    group.transform(lambda s: s.rolling(window=window, min_periods=1).mean())
                    .fillna(0.0)
                )
                out[f"{col}_rs{window}"] = (
                    group.transform(lambda s: s.rolling(window=window, min_periods=1).std(ddof=0))
                    .fillna(0.0)
                )
        return out

    def _add_patch_summary_features(self, df):
        out = pd.DataFrame(index=df.index)
        group = df.groupby(["date", "symbol"], sort=False)
        base_cols = ["mid_ret1_raw", "obi0", "trade_imb", "spread", "order_pressure"]
        patch_windows = [6, 12, 24, 60]
        for col in base_cols:
            series = group[col]
            for window in patch_windows:
                out[f"{col}_patch{window}_mean"] = series.transform(lambda s: s.rolling(window=window, min_periods=1).mean()).fillna(0.0)
                out[f"{col}_patch{window}_std"] = series.transform(lambda s: s.rolling(window=window, min_periods=1).std(ddof=0)).fillna(0.0)
                out[f"{col}_patch{window}_max"] = series.transform(lambda s: s.rolling(window=window, min_periods=1).max()).fillna(0.0)
                out[f"{col}_patch{window}_min"] = series.transform(lambda s: s.rolling(window=window, min_periods=1).min()).fillna(0.0)
                out[f"{col}_patch{window}_range"] = (out[f"{col}_patch{window}_max"] - out[f"{col}_patch{window}_min"]).fillna(0.0)
                out[f"{col}_patch{window}_slope"] = series.transform(
                    lambda s: s.diff().rolling(window=window, min_periods=1).mean()
                ).fillna(0.0)
                out[f"{col}_patch{window}_last"] = df[col].fillna(0.0)
        return out

    def _add_ofi_features(self, df):
        out = pd.DataFrame(index=df.index)
        group = df.groupby(["date", "symbol"], sort=False)
        level_specs = [
            ("0", "bid0", "ask0", "bsize0", "asize0"),
            ("4", "bid4", "ask4", "bsize0_4", "asize0_4"),
            ("9", "bid9", "ask9", "bsize5_9", "asize5_9"),
            ("19", "bid19", "ask19", "bsize10_19", "asize10_19"),
        ]
        ofi_cols = []
        depth_cols = []
        turnover = (df["tradeBuyTurnover"].fillna(0.0) + df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
        for suffix, bid_px, ask_px, bid_sz, ask_sz in level_specs:
            prev_bid_px = group[bid_px].shift(1)
            prev_ask_px = group[ask_px].shift(1)
            prev_bid_sz = group[bid_sz].shift(1)
            prev_ask_sz = group[ask_sz].shift(1)
            cur_bid_px = df[bid_px]
            cur_ask_px = df[ask_px]
            cur_bid_sz = df[bid_sz]
            cur_ask_sz = df[ask_sz]
            bid_ofi = np.where(
                cur_bid_px > prev_bid_px,
                cur_bid_sz,
                np.where(cur_bid_px == prev_bid_px, cur_bid_sz - prev_bid_sz, -prev_bid_sz),
            )
            ask_ofi = np.where(
                cur_ask_px < prev_ask_px,
                -cur_ask_sz,
                np.where(cur_ask_px == prev_ask_px, -(cur_ask_sz - prev_ask_sz), prev_ask_sz),
            )
            bid_ofi = pd.Series(bid_ofi, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            ask_ofi = pd.Series(ask_ofi, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            level_ofi = (bid_ofi + ask_ofi).astype(np.float32)
            depth = (cur_bid_sz + cur_ask_sz).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            out[f"bid_ofi_{suffix}"] = bid_ofi
            out[f"ask_ofi_{suffix}"] = ask_ofi
            out[f"ofi_{suffix}"] = level_ofi
            out[f"ofi_{suffix}_depth"] = depth
            out[f"ofi_{suffix}_div_depth"] = self._safe_div(level_ofi, depth)
            out[f"ofi_{suffix}_div_turnover"] = self._safe_div(level_ofi, turnover)
            ofi_cols.append(f"ofi_{suffix}")
            depth_cols.append(f"ofi_{suffix}_depth")
        out["ofi_total"] = out[ofi_cols].sum(axis=1)
        out["ofi_total_depth"] = out[depth_cols].sum(axis=1)
        out["ofi_abs"] = out["ofi_total"].abs()
        out["ofi_sign"] = np.sign(out["ofi_total"]).astype(np.float32)
        out["ofi_div_total_depth"] = self._safe_div(out["ofi_total"], out["ofi_total_depth"])
        out["ofi_div_turnover"] = self._safe_div(out["ofi_total"], turnover)
        symbol_group = df.groupby("symbol", sort=False)
        ofi_group = out["ofi_total"].groupby(df["symbol"], sort=False)
        for window in [3, 6, 12, 24]:
            out[f"ofi_total_ema{window}"] = ofi_group.transform(
                lambda s: s.ewm(halflife=window, adjust=False).mean()
            ).fillna(0.0)
            out[f"ofi_total_sum{window}"] = ofi_group.transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            ).fillna(0.0)
            out[f"ofi_total_mean{window}"] = ofi_group.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            out[f"ofi_total_z{window}"] = ofi_group.transform(
                lambda s: (s - s.rolling(window=window, min_periods=1).mean())
                / (s.rolling(window=window, min_periods=1).std(ddof=0) + EPS)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _add_trade_impact_features(self, df):
        out = pd.DataFrame(index=df.index)
        turnover = (df["tradeBuyTurnover"].fillna(0.0) + df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
        qty = (df["tradeBuyQty"].fillna(0.0) + df["tradeSellQty"].fillna(0.0)).astype(np.float32)
        trades = (df["nTradeBuy"].fillna(0.0) + df["nTradeSell"].fillna(0.0)).astype(np.float32)
        signed_qty = (df["tradeBuyQty"].fillna(0.0) - df["tradeSellQty"].fillna(0.0)).astype(np.float32)
        signed_turnover = (df["tradeBuyTurnover"].fillna(0.0) - df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
        out["signed_trade_qty"] = signed_qty
        out["signed_trade_turnover"] = signed_turnover
        out["trade_pressure_qty"] = self._safe_div(signed_qty, qty)
        out["trade_pressure_turnover"] = self._safe_div(signed_turnover, turnover)
        out["trade_intensity"] = trades
        out["avg_trade_size"] = self._safe_div(qty, trades)
        out["avg_trade_turnover"] = self._safe_div(turnover, trades)
        out["trade_pressure_x_spread"] = out["trade_pressure_qty"] * df["spread"].fillna(0.0)
        out["trade_pressure_x_order_pressure"] = out["trade_pressure_qty"] * df["order_pressure"].fillna(0.0)
        out["trade_pressure_x_ofi"] = out["trade_pressure_qty"] * df.get("ofi_total", 0.0)
        symbol_group = df.groupby("symbol", sort=False)
        for window in [3, 6, 12, 24]:
            for col in ["trade_pressure_qty", "trade_pressure_turnover", "trade_intensity", "avg_trade_size", "avg_trade_turnover"]:
                col_group = out[col].groupby(df["symbol"], sort=False)
                out[f"{col}_ema{window}"] = col_group.transform(lambda s: s.ewm(halflife=window, adjust=False).mean()).fillna(0.0)
                out[f"{col}_sum{window}"] = col_group.transform(lambda s: s.rolling(window=window, min_periods=1).sum()).fillna(0.0)
                out[f"{col}_mean{window}"] = col_group.transform(lambda s: s.rolling(window=window, min_periods=1).mean()).fillna(0.0)
                out[f"{col}_z{window}"] = col_group.transform(
                    lambda s: (s - s.rolling(window=window, min_periods=1).mean())
                    / (s.rolling(window=window, min_periods=1).std(ddof=0) + EPS)
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _add_conditional_momentum_features(self, df):
        out = pd.DataFrame(index=df.index)
        day_symbol = df.groupby(["date", "symbol"], sort=False)
        intraday = df.groupby(["date"], sort=False)
        for window in [1, 3, 6, 12, 24]:
            raw = day_symbol["midpx"].transform(lambda s: s.pct_change(window)).fillna(0.0)
            cx = raw.groupby(df["interval"], sort=False).transform("mean")
            lagret = raw - cx
            out[f"lagret{window}_raw"] = raw.astype(np.float32)
            out[f"lagret{window}"] = lagret.astype(np.float32)
            out[f"lagret{window}_abs"] = lagret.abs().astype(np.float32)
            out[f"lagret{window}_sign"] = np.sign(lagret).astype(np.float32)
            out[f"lagret{window}_x_trade_pressure"] = lagret * df.get("trade_pressure_qty", 0.0)
            out[f"lagret{window}_x_ofi"] = lagret * df.get("ofi_total", 0.0)
            out[f"lagret{window}_x_spread"] = lagret * df.get("spread", 0.0)
            out[f"lagret{window}_x_vol"] = lagret * day_symbol["mid_ret1_raw"].transform(lambda s: s.rolling(window=min(10, window + 2), min_periods=1).std(ddof=0)).fillna(0.0)
        out["momentum_state"] = (out["lagret12"] > 0).astype(np.float32) if "lagret12" in out.columns else 0.0
        out["reversal_state"] = (out["lagret12"] < 0).astype(np.float32) if "lagret12" in out.columns else 0.0
        out["conditional_momentum_rank"] = out["lagret12"].groupby(df["date"], sort=False).transform(lambda s: s.rank(pct=True, method="average")).fillna(0.0) if "lagret12" in out.columns else 0.0
        return out

    def _add_cross_section_features(self, df):
        out = pd.DataFrame(index=df.index)
        cross_group = df.groupby(["date", "interval"], sort=False)
        cols = [
            "midpx",
            "spread",
            "obi0",
            "obi4",
            "trade_imb",
            "order_pressure",
            "trade_activity",
            "ofi_total",
            "ofi_0",
            "ofi_4",
            "ofi_9",
            "ofi_19",
            "trade_pressure_qty",
            "trade_pressure_turnover",
            "trade_intensity",
            "avg_trade_size",
            "avg_trade_turnover",
        ]
        for col in cols:
            if col not in df.columns:
                continue
            mean = cross_group[col].transform("mean")
            std = cross_group[col].transform("std").replace(0.0, np.nan)
            out[f"{col}_cs_z"] = ((df[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[f"{col}_cs_rank"] = cross_group[col].rank(pct=True, method="average").fillna(0.0)
        out["interval_pos"] = df.groupby("date", sort=False)["interval"].rank(pct=True, method="average").fillna(0.0)
        out["interval_norm"] = df["interval"].astype(np.float32) / 1e8
        out["is_morning"] = (df["interval"] < 113000000).astype(np.float32)
        out["is_afternoon"] = (df["interval"] >= 130000000).astype(np.float32)
        return out

    def _add_regime_features(self, df):
        out = pd.DataFrame(index=df.index)
        day_symbol = df.groupby(["date", "symbol"], sort=False)
        intraday = df.groupby(["date"], sort=False)
        ret1 = df["mid_ret1_raw"].fillna(0.0)
        vol3 = day_symbol["mid_ret1_raw"].transform(lambda s: s.rolling(window=3, min_periods=1).std(ddof=0)).fillna(0.0)
        vol10 = day_symbol["mid_ret1_raw"].transform(lambda s: s.rolling(window=10, min_periods=1).std(ddof=0)).fillna(0.0)
        imbalance = df["order_pressure"].fillna(0.0)
        spread = df["spread"].fillna(0.0)
        activity = df["trade_activity"].fillna(0.0)
        cxl = df["cxl_imb"].fillna(0.0)

        vol_cs = intraday["mid_ret1_raw"].transform(lambda s: s.rolling(window=10, min_periods=1).std(ddof=0)).fillna(0.0)
        spread_cs = intraday["spread"].transform("mean").fillna(0.0)
        activity_cs = intraday["trade_activity"].transform("mean").fillna(0.0)

        regime_score = (
            0.35 * (vol10 / (vol10.groupby(df["date"], sort=False).transform("mean").replace(0.0, np.nan).fillna(1.0)))
            + 0.25 * spread
            + 0.20 * imbalance.abs()
            + 0.10 * activity
            + 0.10 * cxl.abs()
        )
        regime_score = regime_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score_q1 = regime_score.groupby(df["date"], sort=False).transform(lambda s: s.quantile(0.33))
        score_q2 = regime_score.groupby(df["date"], sort=False).transform(lambda s: s.quantile(0.66))
        out["regime_score"] = regime_score
        out["regime_low"] = (regime_score <= score_q1).astype(np.float32)
        out["regime_mid"] = ((regime_score > score_q1) & (regime_score <= score_q2)).astype(np.float32)
        out["regime_high"] = (regime_score > score_q2).astype(np.float32)
        out["state_momentum"] = ((ret1 > 0).astype(np.float32) * (vol10 < vol10.groupby(df["date"], sort=False).transform("median").fillna(0.0)).astype(np.float32)).fillna(0.0)
        out["state_reversal"] = ((ret1 < 0).astype(np.float32) * (spread > spread.groupby(df["date"], sort=False).transform("median").fillna(0.0)).astype(np.float32)).fillna(0.0)
        out["state_pressure"] = (imbalance.abs() > imbalance.groupby(df["date"], sort=False).transform("median").abs().fillna(0.0)).astype(np.float32)
        out["state_liquidity"] = (activity < activity_cs.groupby(df["date"], sort=False).transform("median").fillna(0.0)).astype(np.float32)
        out["state_vol_cs"] = vol_cs
        out["state_spread_cs"] = spread_cs
        out["state_activity_cs"] = activity_cs
        return out


class ExperimentRunner(object):
    def __init__(self, h5dir):
        self.calendar = Calendar()
        self.loader = MeowDataLoader(h5dir=h5dir)
        self.builder = FeatureBuilder()
        self._split_cache = {}
        self._raw_split_cache = {}
        self._daily_feature_cache = {}
        self._daily_raw_cache = {}

    def _cache_key(self, dates, max_days=None):
        return (tuple(dates), max_days)

    def _normalize_groups(self, groups):
        if groups is None:
            return None
        if isinstance(groups, str):
            groups = [groups]
        groups = [g.strip() for g in groups if g and g.strip()]
        if not groups or groups == ["full"]:
            return None
        return groups

    def _filter_features(self, xdf, groups):
        groups = self._normalize_groups(groups)
        if groups is None:
            return xdf
        return self.builder.select_groups(xdf, groups)

    def _regime_state(self, xdf):
        state_cols = [c for c in ["regime_low", "regime_mid", "regime_high"] if c in xdf.columns]
        if len(state_cols) != 3:
            raise ValueError("Regime state requires regime_low/regime_mid/regime_high columns")
        state_values = xdf[state_cols].to_numpy(dtype=np.float32)
        return np.argmax(state_values, axis=1)

    def _interval_baseline(self, ydf):
        baseline = (
            ydf.groupby("interval", sort=False)["fret12"]
            .mean()
            .rename("interval_mean")
            .reset_index()
        )
        return baseline

    def _attach_interval_baseline(self, ydf, baseline):
        out = ydf.merge(baseline, on="interval", how="left")
        out["interval_mean"] = out["interval_mean"].fillna(0.0)
        out["fret12_residual"] = out["fret12"] - out["interval_mean"]
        return out

    def _fit_common_component(self, ytrain):
        common_model = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("ridge", Ridge(alpha=10.0, fit_intercept=True, random_state=None)),
        ])
        common_x = ytrain[["interval"]].to_numpy()
        common_y = ytrain["fret12"].to_numpy(dtype=np.float32)
        common_model.fit(common_x, common_y)
        return common_model

    def _make_target_series(self, ydf, target_mode):
        if target_mode == "raw":
            return ydf["fret12"].to_numpy(dtype=np.float32), None
        if target_mode == "date_demean":
            series = (
                ydf.groupby("date")["fret12"]
                .transform(lambda s: s - s.mean())
                .to_numpy(dtype=np.float32)
            )
            return series, None
        if target_mode == "interval_demean":
            series = (
                ydf.groupby(["date", "interval"])["fret12"]
                .transform(lambda s: s - s.mean())
                .to_numpy(dtype=np.float32)
            )
            return series, None
        if target_mode == "interval_residual":
            baseline = self._interval_baseline(ydf)
            merged = self._attach_interval_baseline(ydf, baseline)
            return merged["fret12_residual"].to_numpy(dtype=np.float32), baseline
        raise ValueError(f"Unknown target mode: {target_mode}")

    def split_dates(self, split_config):
        train_dates = self.calendar.range(split_config.train_start, split_config.train_end)
        val_dates = self.calendar.range(split_config.val_start, split_config.val_end)
        test_dates = self.calendar.range(split_config.test_start, split_config.test_end)
        return train_dates, val_dates, test_dates

    def load_split(self, dates, max_days=None):
        cache_key = self._cache_key(dates, max_days=max_days)
        if cache_key in self._split_cache:
            xdf, ydf = self._split_cache[cache_key]
            return xdf, ydf
        if max_days is not None:
            dates = dates[:max_days]
        x_parts = []
        y_parts = []
        for date in dates:
            if date in self._daily_feature_cache:
                xdf, ydf = self._daily_feature_cache[date]
            else:
                log.inf(f"Loading and featurizing {date}...")
                raw = self.loader.loadDate(date)
                xdf, ydf = self.builder.build(raw)
                self._daily_feature_cache[date] = (xdf, ydf)
                del raw
                gc.collect()
            x_parts.append(xdf)
            y_parts.append(ydf)
        xdf = pd.concat(x_parts, ignore_index=True)
        ydf = pd.concat(y_parts, ignore_index=True)
        self._split_cache[cache_key] = (xdf, ydf)
        return xdf, ydf

    def load_raw_split(self, dates, max_days=None):
        cache_key = self._cache_key(dates, max_days=max_days)
        if cache_key in self._raw_split_cache:
            return self._raw_split_cache[cache_key]
        if max_days is not None:
            dates = dates[:max_days]
        parts = []
        for date in dates:
            if date in self._daily_raw_cache:
                parts.append(self._daily_raw_cache[date])
            else:
                log.inf(f"Loading raw {date}...")
                raw = self.loader.loadDate(date)
                self._daily_raw_cache[date] = raw
                parts.append(raw)
                gc.collect()
        raw = pd.concat(parts, ignore_index=True)
        self._raw_split_cache[cache_key] = raw
        return raw

    def _build_sequence_features(self, raw_df, lags):
        raw_df = raw_df.copy()
        raw_df = raw_df.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
        seq_cols = [
            "midpx",
            "bid0",
            "ask0",
            "bid4",
            "ask4",
            "bid9",
            "ask9",
            "bsize0",
            "asize0",
            "bsize0_4",
            "asize0_4",
            "bsize5_9",
            "asize5_9",
            "tradeBuyQty",
            "tradeSellQty",
            "tradeBuyTurnover",
            "tradeSellTurnover",
            "nAddBuy",
            "nAddSell",
            "nCxlBuy",
            "nCxlSell",
            "buyVwad",
            "sellVwad",
        ]
        out = raw_df[["date", "symbol", "interval", "fret12"]].copy()
        group = raw_df.groupby(["date", "symbol"], sort=False)
        for col in seq_cols:
            out[col] = raw_df[col].astype(np.float32)
            for lag in lags:
                out[f"{col}_seq_lag_{lag}"] = group[col].shift(lag).fillna(0.0).astype(np.float32)
        xdf = out.drop(columns=["fret12"]).copy()
        ydf = out[["date", "symbol", "interval", "fret12"]].copy()
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return xdf, ydf

    def load_feature_split(self, dates, max_days=None, groups=None):
        xdf, ydf = self.load_split(dates, max_days=max_days)
        xdf = self._filter_features(xdf, groups)
        return xdf, ydf

    def evaluate_predictions(self, ydf, pred):
        y = ydf["fret12"].to_numpy()
        p = np.asarray(pred)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        mse = mean_squared_error(y, p)
        corr = np.corrcoef(p, y)[0, 1] if len(y) > 1 else 0.0
        r2 = 1.0 - np.sum((p - y) ** 2) / (np.var(y) * len(y) + EPS)
        return {"mse": float(mse), "corr": float(corr), "r2": float(r2)}

    def evaluate_prediction_bundle(self, ydf, pred):
        metrics = self.evaluate_predictions(ydf, pred)
        daily_corrs = []
        tmp = ydf[["date", "fret12"]].copy()
        tmp["pred"] = np.asarray(pred, dtype=np.float32)
        for _, group in tmp.groupby("date", sort=True):
            if group.shape[0] < 2:
                continue
            corr = np.corrcoef(group["pred"].to_numpy(), group["fret12"].to_numpy())[0, 1]
            if np.isfinite(corr):
                daily_corrs.append(float(corr))
        metrics["daily_corr_mean"] = float(np.mean(daily_corrs)) if daily_corrs else 0.0
        metrics["daily_corr_std"] = float(np.std(daily_corrs)) if daily_corrs else 0.0
        metrics["n_days"] = int(tmp["date"].nunique())
        return metrics

    def _corr_gap(self, train_metrics, val_metrics):
        return float(train_metrics["corr"] - val_metrics["corr"])

    def _safe_rank_pct(self, series):
        return series.rank(pct=True, method="average")

    def _build_common_features(self, xdf):
        cols = [
            "spread",
            "obi0",
            "obi4",
            "trade_imb",
            "trade_turnover_imb",
            "cxl_imb",
            "trade_activity",
            "order_pressure",
            "mid_ret1_raw",
            "regime_score",
            "state_vol_cs",
            "state_spread_cs",
            "state_activity_cs",
            "interval_pos",
            "interval_norm",
            "is_morning",
            "is_afternoon",
        ]
        cols = [c for c in cols if c in xdf.columns]
        agg = xdf[["date", "interval"] + cols].groupby(["date", "interval"], sort=False)[cols].mean().reset_index()
        return agg

    def _extract_linear_coefficients(self, model, feature_cols):
        if isinstance(model, Pipeline):
            inner = model.named_steps.get("model")
            if hasattr(inner, "coef_"):
                coef = np.asarray(inner.coef_, dtype=np.float32)
            else:
                return None
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=np.float32)
        else:
            return None
        if coef.ndim > 1:
            coef = coef.ravel()
        if len(coef) != len(feature_cols):
            return None
        return pd.DataFrame({"feature": list(feature_cols), "coef": coef, "abs_coef": np.abs(coef)})

    def _fold_metric_row(self, fold_id, experiment_id, feature_set, target_type, model_type, postprocess_type, train_metrics, val_metrics, runtime_sec, notes, random_seed=42):
        return {
            "fold_id": fold_id,
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "target_type": target_type,
            "model_type": model_type,
            "postprocess_type": postprocess_type,
            "train_corr": train_metrics["corr"],
            "val_corr": val_metrics["corr"],
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "daily_corr_mean": val_metrics["daily_corr_mean"],
            "daily_corr_std": val_metrics["daily_corr_std"],
            "rolling_corr_mean": np.nan,
            "rolling_corr_std": np.nan,
            "rolling_corr_min": np.nan,
            "rolling_mse_mean": np.nan,
            "rolling_r2_mean": np.nan,
            "stability_score": np.nan,
            "train_val_corr_gap": self._corr_gap(train_metrics, val_metrics),
            "runtime_sec": float(runtime_sec),
            "random_seed": random_seed,
            "notes": notes,
        }

    def _summarize_rolling_rows(self, rows, fold_count):
        df = pd.DataFrame(rows)
        summary_rows = []
        for experiment_id, group in df.groupby("experiment_id", sort=False):
            val_corrs = group.sort_values("fold_id")["val_corr"].tolist()
            val_mses = group.sort_values("fold_id")["val_mse"].tolist()
            val_r2s = group.sort_values("fold_id")["val_r2"].tolist()
            row = {
                "experiment_name": experiment_id,
                "model_type": group["model_type"].iloc[0],
                "feature_set": group["feature_set"].iloc[0],
                "target_type": group["target_type"].iloc[0],
            }
            for idx in range(fold_count):
                row[f"fold{idx + 1}_corr"] = val_corrs[idx] if idx < len(val_corrs) else np.nan
            row["rolling_corr_mean"] = float(np.mean(val_corrs)) if val_corrs else np.nan
            row["rolling_corr_std"] = float(np.std(val_corrs, ddof=0)) if len(val_corrs) > 0 else np.nan
            row["rolling_corr_min"] = float(np.min(val_corrs)) if val_corrs else np.nan
            row["rolling_mse_mean"] = float(np.mean(val_mses)) if val_mses else np.nan
            row["rolling_r2_mean"] = float(np.mean(val_r2s)) if val_r2s else np.nan
            row["daily_corr_mean"] = float(group["daily_corr_mean"].mean())
            row["daily_corr_std"] = float(group["daily_corr_std"].mean())
            row["stability_score"] = row["rolling_corr_mean"] - 0.7 * row["rolling_corr_std"]
            row["是否进入下一阶段"] = "是"
            row["原因"] = "作为正式 rolling 候选保留"
            summary_rows.append(row)
        out = pd.DataFrame(summary_rows)
        ordered_cols = [
            "experiment_name",
            "model_type",
            "feature_set",
            "target_type",
        ] + [f"fold{i}_corr" for i in range(1, fold_count + 1)] + [
            "rolling_corr_mean",
            "rolling_corr_std",
            "rolling_corr_min",
            "stability_score",
            "rolling_mse_mean",
            "rolling_r2_mean",
            "daily_corr_mean",
            "daily_corr_std",
            "是否进入下一阶段",
            "原因",
        ]
        return out.reindex(columns=ordered_cols)

    def _make_common_targets(self, ydf):
        return (
            ydf.groupby(["date", "interval"], sort=False)["fret12"]
            .mean()
            .rename("fret12_common")
            .reset_index()
        )

    def _fit_common_model(self, xtrain, ytrain, model_name="ridge"):
        common_x = self._build_common_features(xtrain)
        common_y = self._make_common_targets(ytrain)
        common_df = common_x.merge(common_y, on=["date", "interval"], how="inner")
        common_feature_cols = [c for c in common_df.columns if c not in ["date", "interval", "fret12_common"]]
        if model_name == "hgb":
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=200,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
            model.fit(common_df[common_feature_cols].to_numpy(dtype=np.float32), common_df["fret12_common"].to_numpy(dtype=np.float32))
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0, fit_intercept=True, random_state=None)),
            ])
            model.fit(
                common_df[common_feature_cols].to_numpy(dtype=np.float32),
                common_df["fret12_common"].to_numpy(dtype=np.float32),
            )
        return model, common_feature_cols

    def _predict_common_component(self, model, feature_cols, xdf):
        common_x = self._build_common_features(xdf)
        common_x["pred_common"] = model.predict(common_x[feature_cols].to_numpy(dtype=np.float32))
        merged = xdf[["date", "interval"]].merge(
            common_x[["date", "interval", "pred_common"]],
            on=["date", "interval"],
            how="left",
        )
        return merged["pred_common"].fillna(0.0).to_numpy(dtype=np.float32)

    def make_rolling_folds(self, start_date, end_date, train_window=40, val_window=10, step=10, min_train_days=30, embargo=0):
        all_dates = self.calendar.range(start_date, end_date)
        folds = []
        fold_id = 0
        if not all_dates:
            return folds
        embargo = max(0, int(embargo))
        cursor = train_window + embargo
        while cursor + val_window <= len(all_dates):
            train_end = cursor - embargo
            train_dates = tuple(all_dates[max(0, train_end - train_window):train_end])
            val_dates = tuple(all_dates[cursor: cursor + val_window])
            if len(train_dates) >= min_train_days and len(val_dates) > 0:
                folds.append(RollingFold(fold_id=fold_id, train_dates=train_dates, val_dates=val_dates))
                fold_id += 1
            cursor += step
        if not folds and len(all_dates) >= 3:
            fallback_val = max(1, min(2, len(all_dates) // 3))
            folds.append(
                RollingFold(
                    fold_id=0,
                    train_dates=tuple(all_dates[:-fallback_val]),
                    val_dates=tuple(all_dates[-fallback_val:]),
                )
            )
        return folds

    def _normalize_by_date_interval_rank(self, frame, pred_col):
        out = frame.copy()
        rank = out.groupby(["date", "interval"], sort=False)[pred_col].transform(self._safe_rank_pct)
        centered = rank - 0.5
        scale = float(out[pred_col].std()) if out.shape[0] > 1 else 0.0
        out["pred_rank_scaled"] = centered.fillna(0.0) * scale * 2.0
        out["pred_group_demean"] = out[pred_col] - out.groupby(["date", "interval"], sort=False)[pred_col].transform("mean")
        return out

    def choose_postprocess_params(self, ydf, pred):
        frame = ydf[["date", "interval", "fret12"]].copy()
        frame["pred"] = np.asarray(pred, dtype=np.float32)
        candidates = []
        for q in [0.001, 0.005, 0.01]:
            candidates.append({"name": f"clip_{q:.3f}", "clip_q": q, "blend_alpha": 1.0, "blend_kind": "none"})
            for alpha in [0.7, 0.8, 0.9]:
                candidates.append({"name": f"clip_rank_{q:.3f}_{alpha:.1f}", "clip_q": q, "blend_alpha": alpha, "blend_kind": "rank"})
                candidates.append({"name": f"clip_neutral_{q:.3f}_{alpha:.1f}", "clip_q": q, "blend_alpha": alpha, "blend_kind": "neutral"})
        best = None
        best_metrics = None
        for params in candidates:
            post = self.apply_postprocess(frame, params)
            metrics = self.evaluate_prediction_bundle(frame[["date", "fret12"]], post)
            score = (metrics["corr"], -metrics["mse"], metrics["r2"])
            if best is None or score > best:
                best = score
                best_metrics = metrics
                best_params = params
        return best_params, best_metrics

    def apply_postprocess(self, frame, params):
        out = frame.copy()
        q = params.get("clip_q", 0.005)
        lower = float(out["pred"].quantile(q))
        upper = float(out["pred"].quantile(1.0 - q))
        out["pred"] = out["pred"].clip(lower=lower, upper=upper)
        out = self._normalize_by_date_interval_rank(out, "pred")
        alpha = float(params.get("blend_alpha", 1.0))
        blend_kind = params.get("blend_kind", "none")
        if blend_kind == "rank":
            return alpha * out["pred"].to_numpy(dtype=np.float32) + (1.0 - alpha) * out["pred_rank_scaled"].to_numpy(dtype=np.float32)
        if blend_kind == "neutral":
            return alpha * out["pred"].to_numpy(dtype=np.float32) + (1.0 - alpha) * out["pred_group_demean"].to_numpy(dtype=np.float32)
        return out["pred"].to_numpy(dtype=np.float32)

    def make_target(self, ydf, target_mode):
        series, _ = self._make_target_series(ydf, target_mode)
        return series

    def fit_model(self, model_name, xtrain, ytrain, target_mode="raw", sample_weight=None):
        feature_cols = [c for c in xtrain.columns if c not in ["date", "symbol", "interval"]]
        x = xtrain[feature_cols].to_numpy(dtype=np.float32)
        y, baseline = self._make_target_series(ytrain, target_mode=target_mode)
        if model_name == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0, fit_intercept=True, random_state=None)),
            ])
        elif model_name == "elasticnet":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.1, fit_intercept=True, max_iter=5000, random_state=42)),
            ])
        elif model_name == "tree":
            model = ExtraTreesRegressor(
                n_estimators=40,
                max_depth=16,
                min_samples_leaf=50,
                max_features=0.3,
                bootstrap=True,
                max_samples=0.7,
                random_state=42,
                n_jobs=1,
            )
        elif model_name == "tree_big":
            model = ExtraTreesRegressor(
                n_estimators=120,
                max_depth=20,
                min_samples_leaf=20,
                max_features=0.5,
                bootstrap=True,
                max_samples=0.8,
                random_state=42,
                n_jobs=1,
            )
        elif model_name in {"gbdt", "histgb"}:
            if model_name == "histgb":
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=200,
                    max_depth=8,
                    min_samples_leaf=50,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                )
            else:
                model = GradientBoostingRegressor(
                    learning_rate=0.05,
                    n_estimators=30,
                    max_depth=2,
                    min_samples_leaf=200,
                    subsample=0.8,
                    random_state=42,
                )
        elif model_name == "lgbm":
            if LGBMRegressor is None:
                raise ImportError(
                    "lightgbm is not installed. Install it or use model_name='histgb' in this workspace."
                )
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=1,
            )
        elif model_name == "mlp":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=10,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                )),
            ])
        else:
            raise ValueError(f"Unknown model: {model_name}")
        try:
            if sample_weight is None:
                model.fit(x, y)
            elif isinstance(model, Pipeline):
                model.fit(x, y, model__sample_weight=np.asarray(sample_weight, dtype=np.float32))
            else:
                model.fit(x, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
        except PermissionError:
            if model_name != "histgb":
                raise
            log.yellow("HistGradientBoosting hit a sandbox permission error, falling back to GradientBoostingRegressor.")
            model = GradientBoostingRegressor(
                learning_rate=0.05,
                n_estimators=30,
                max_depth=2,
                min_samples_leaf=200,
                subsample=0.8,
                random_state=42,
            )
            if sample_weight is None:
                model.fit(x, y)
            else:
                model.fit(x, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
        return model, feature_cols, baseline

    def predict(self, model, xdf, feature_cols):
        x = xdf[feature_cols].to_numpy(dtype=np.float32)
        return model.predict(x)

    def _predict_with_baseline(self, model, xdf, feature_cols, ydf=None, baseline=None, target_mode="raw"):
        pred = self.predict(model, xdf, feature_cols)
        if target_mode != "interval_residual":
            return pred
        if ydf is None or baseline is None:
            raise ValueError("interval_residual requires ydf and baseline")
        merged = ydf.merge(baseline, on="interval", how="left")
        common = merged["interval_mean"].fillna(0.0).to_numpy(dtype=np.float32)
        return common + pred

    def run(self, split_config, model_name, max_train_days=None, max_val_days=None, target_mode="raw"):
        return self.run_with_groups(
            split_config=split_config,
            model_name=model_name,
            feature_groups=None,
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
        )

    def run_with_groups(
        self,
        split_config,
        model_name,
        feature_groups=None,
        max_train_days=None,
        max_val_days=None,
        target_mode="raw",
    ):
        train_dates, val_dates, test_dates = self.split_dates(split_config)
        log.inf(f"Train dates: {train_dates[0]} -> {train_dates[-1]} ({len(train_dates)})")
        log.inf(f"Val dates: {val_dates[0]} -> {val_dates[-1]} ({len(val_dates)})")
        xtrain, ytrain = self.load_feature_split(train_dates, max_days=max_train_days, groups=feature_groups)
        xval, yval = self.load_feature_split(val_dates, max_days=max_val_days, groups=feature_groups)
        log.inf(f"Train shape: {xtrain.shape}, Val shape: {xval.shape}")
        model, feature_cols, baseline = self.fit_model(model_name, xtrain, ytrain, target_mode=target_mode)
        pred_train = self._predict_with_baseline(
            model,
            xtrain,
            feature_cols,
            ydf=ytrain,
            baseline=baseline,
            target_mode=target_mode,
        )
        pred_val = self._predict_with_baseline(
            model,
            xval,
            feature_cols,
            ydf=yval,
            baseline=baseline,
            target_mode=target_mode,
        )
        train_metrics = self.evaluate_prediction_bundle(ytrain, pred_train)
        val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": model,
            "feature_cols": feature_cols,
            "baseline": baseline,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_groups": self._normalize_groups(feature_groups),
            "model_name": model_name,
            "target_mode": target_mode,
        }

    def run_on_features(self, xtrain, ytrain, xval, yval, model_name, target_mode="raw"):
        log.inf(f"Train shape: {xtrain.shape}, Val shape: {xval.shape}")
        model, feature_cols, baseline = self.fit_model(model_name, xtrain, ytrain, target_mode=target_mode)
        pred_train = self._predict_with_baseline(
            model,
            xtrain,
            feature_cols,
            ydf=ytrain,
            baseline=baseline,
            target_mode=target_mode,
        )
        pred_val = self._predict_with_baseline(
            model,
            xval,
            feature_cols,
            ydf=yval,
            baseline=baseline,
            target_mode=target_mode,
        )
        train_metrics = self.evaluate_prediction_bundle(ytrain, pred_train)
        val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": model,
            "feature_cols": feature_cols,
            "feature_count": int(len(feature_cols)),
            "baseline": baseline,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def run_common_residual_reconstruction(
        self,
        split_config,
        residual_model_name="tree",
        common_model_name="ridge",
        feature_groups=None,
        max_train_days=None,
        max_val_days=None,
        lambda_grid=None,
    ):
        train_dates, val_dates, _ = self.split_dates(split_config)
        xtrain, ytrain = self.load_feature_split(train_dates, max_days=max_train_days, groups=feature_groups)
        xval, yval = self.load_feature_split(val_dates, max_days=max_val_days, groups=feature_groups)
        common_model, common_feature_cols = self._fit_common_model(xtrain, ytrain, model_name=common_model_name)
        common_train = self._predict_common_component(common_model, common_feature_cols, xtrain)
        common_val = self._predict_common_component(common_model, common_feature_cols, xval)

        ytrain_resid = ytrain.copy()
        ytrain_resid["fret12"] = ytrain_resid["fret12"].to_numpy(dtype=np.float32) - common_train
        residual_model, residual_feature_cols, _ = self.fit_model(
            residual_model_name,
            xtrain,
            ytrain_resid,
            target_mode="raw",
        )
        residual_train = self.predict(residual_model, xtrain, residual_feature_cols)
        residual_val = self.predict(residual_model, xval, residual_feature_cols)
        if lambda_grid is None:
            lambda_grid = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        best = None
        for lam in lambda_grid:
            pred_train = residual_train + lam * common_train
            pred_val = residual_val + lam * common_val
            val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best is None or score > best[0]:
                best = (
                    score,
                    lam,
                    pred_train.copy(),
                    pred_val.copy(),
                    self.evaluate_prediction_bundle(ytrain, pred_train),
                    val_metrics,
                )
        _, best_lambda, pred_train, pred_val, train_metrics, val_metrics = best
        return {
            "common_model": common_model,
            "common_feature_cols": common_feature_cols,
            "residual_model": residual_model,
            "residual_feature_cols": residual_feature_cols,
            "lambda": float(best_lambda),
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def _fit_and_predict_expert(self, spec, train_dates, val_dates):
        groups = spec.get("feature_groups")
        xtrain, ytrain = self.load_feature_split(train_dates, groups=groups)
        xval, yval = self.load_feature_split(val_dates, groups=groups)
        if spec["kind"] == "common_residual":
            split = SplitConfig(
                train_start=train_dates[0],
                train_end=train_dates[-1],
                val_start=val_dates[0],
                val_end=val_dates[-1],
                test_start=val_dates[0],
                test_end=val_dates[-1],
            )
            result = self.run_common_residual_reconstruction(
                split,
                residual_model_name=spec.get("residual_model", "tree"),
                common_model_name=spec.get("common_model", "ridge"),
                feature_groups=groups,
            )
            return {
                "pred_train": result["pred_train"],
                "pred_val": result["pred_val"],
                "train_metrics": result["train_metrics"],
                "val_metrics": result["val_metrics"],
                "xtrain": xtrain,
                "ytrain": ytrain,
                "xval": xval,
                "yval": yval,
                "result": result,
            }
        result = self.run_on_features(
            xtrain=xtrain,
            ytrain=ytrain,
            xval=xval,
            yval=yval,
            model_name=spec["model_name"],
            target_mode=spec.get("target_mode", "raw"),
        )
        return {
            "pred_train": result["pred_train"],
            "pred_val": result["pred_val"],
            "train_metrics": result["train_metrics"],
            "val_metrics": result["val_metrics"],
            "xtrain": xtrain,
            "ytrain": ytrain,
            "xval": xval,
            "yval": yval,
            "result": result,
        }

    def run_oof_multi_expert_fusion(
        self,
        split_config,
        expert_specs,
        max_train_days=None,
        max_val_days=None,
        meta_model_name="ridge",
    ):
        train_dates, val_dates, _ = self.split_dates(split_config)
        if max_train_days is not None:
            train_dates = train_dates[:max_train_days]
        if max_val_days is not None:
            val_dates = val_dates[:max_val_days]
        train_folds = self.make_rolling_folds(train_dates[0], train_dates[-1], train_window=min(40, len(train_dates) - 10), val_window=min(10, max(2, len(train_dates) // 6)), step=min(10, max(2, len(train_dates) // 6)), min_train_days=min(20, max(10, len(train_dates) // 3)))
        if not train_folds:
            raise ValueError("Not enough training dates to build OOF folds")

        full_ytrain = self.load_feature_split(train_dates, groups=None)[1]
        full_yval = self.load_feature_split(val_dates, groups=None)[1]
        meta_rows = []
        meta_targets = []
        for fold in train_folds:
            fold_frames = []
            for spec in expert_specs:
                exp = self._fit_and_predict_expert(spec, list(fold.train_dates), list(fold.val_dates))
                frame = exp["yval"][["date", "symbol", "interval", "fret12"]].copy()
                frame[spec["name"]] = np.asarray(exp["pred_val"], dtype=np.float32)
                fold_frames.append(frame)
            merged = fold_frames[0]
            for frame in fold_frames[1:]:
                merged = merged.merge(frame[["date", "symbol", "interval"] + [c for c in frame.columns if c not in ["date", "symbol", "interval", "fret12"]]], on=["date", "symbol", "interval"], how="inner")
            meta_rows.append(merged.drop(columns=["fret12"]).copy())
            meta_targets.append(merged[["date", "symbol", "interval", "fret12"]].copy())

        meta_xtrain = pd.concat(meta_rows, ignore_index=True)
        meta_ytrain = pd.concat(meta_targets, ignore_index=True)
        meta_model, meta_feature_cols, _ = self.fit_model(
            meta_model_name,
            meta_xtrain,
            meta_ytrain,
            target_mode="raw",
        )

        train_pred_frames = []
        val_pred_frames = []
        for spec in expert_specs:
            exp = self._fit_and_predict_expert(spec, train_dates, val_dates)
            train_frame = exp["ytrain"][["date", "symbol", "interval"]].copy()
            train_frame[spec["name"]] = np.asarray(exp["pred_train"], dtype=np.float32)
            val_frame = exp["yval"][["date", "symbol", "interval"]].copy()
            val_frame[spec["name"]] = np.asarray(exp["pred_val"], dtype=np.float32)
            train_pred_frames.append(train_frame)
            val_pred_frames.append(val_frame)
        full_meta_train = train_pred_frames[0]
        full_meta_val = val_pred_frames[0]
        for frame in train_pred_frames[1:]:
            full_meta_train = full_meta_train.merge(frame, on=["date", "symbol", "interval"], how="inner")
        for frame in val_pred_frames[1:]:
            full_meta_val = full_meta_val.merge(frame, on=["date", "symbol", "interval"], how="inner")

        meta_train_pred = self.predict(meta_model, full_meta_train, meta_feature_cols)
        meta_val_pred = self.predict(meta_model, full_meta_val, meta_feature_cols)
        post_params, post_train_metrics = self.choose_postprocess_params(full_ytrain, meta_train_pred)
        train_post_frame = full_ytrain[["date", "interval", "fret12"]].copy()
        train_post_frame["pred"] = np.asarray(meta_train_pred, dtype=np.float32)
        val_post_frame = full_yval[["date", "interval", "fret12"]].copy()
        val_post_frame["pred"] = np.asarray(meta_val_pred, dtype=np.float32)
        final_train_pred = self.apply_postprocess(train_post_frame, post_params)
        final_val_pred = self.apply_postprocess(val_post_frame, post_params)
        return {
            "meta_model": meta_model,
            "meta_feature_cols": meta_feature_cols,
            "postprocess_params": post_params,
            "postprocess_train_metrics": post_train_metrics,
            "pred_train": final_train_pred,
            "pred_val": final_val_pred,
            "train_metrics": self.evaluate_prediction_bundle(full_ytrain, final_train_pred),
            "val_metrics": self.evaluate_prediction_bundle(full_yval, final_val_pred),
            "raw_pred_train": meta_train_pred,
            "raw_pred_val": meta_val_pred,
            "oof_rows": meta_xtrain.shape[0],
        }

    def run_multi_scale_fusion(self, split_config, model_name="tree", target_mode="interval_residual", max_train_days=None, max_val_days=None):
        train_dates, val_dates, _ = self.split_dates(split_config)
        raw_train = self.load_raw_split(train_dates, max_days=max_train_days)
        raw_val = self.load_raw_split(val_dates, max_days=max_val_days)
        xtrain_full, ytrain = self.load_feature_split(train_dates, max_days=max_train_days, groups=None)
        xval_full, yval = self.load_feature_split(val_dates, max_days=max_val_days, groups=None)
        common_model = self._fit_common_component(ytrain)
        train_common = common_model.predict(ytrain[["interval"]].to_numpy()).astype(np.float32)
        val_common = common_model.predict(yval[["interval"]].to_numpy()).astype(np.float32)
        ytrain_resid = ytrain.copy()
        yval_resid = yval.copy()
        ytrain_resid["fret12"] = ytrain_resid["fret12"].to_numpy(dtype=np.float32) - train_common
        yval_resid["fret12"] = yval_resid["fret12"].to_numpy(dtype=np.float32) - val_common

        scales = {
            "short": ["base", "lag_short", "roll_short"],
            "mid": ["base", "lag_mid", "roll_mid"],
            "long": ["base", "lag_long", "roll_long", "cross"],
            "full": None,
        }
        preds_train = []
        preds_val = []
        model_rows = []
        for scale_name, groups in scales.items():
            xtrain = self._filter_features(xtrain_full, groups)
            xval = self._filter_features(xval_full, groups)
            log.inf(f"Training scale expert: {scale_name} with shape {xtrain.shape}")
            model, feature_cols, baseline = self.fit_model(model_name, xtrain, ytrain_resid, target_mode="raw")
            train_resid_pred = self.predict(model, xtrain, feature_cols)
            val_resid_pred = self.predict(model, xval, feature_cols)
            train_pred = train_common + train_resid_pred
            val_pred = val_common + val_resid_pred
            preds_train.append(train_resid_pred.reshape(-1, 1))
            preds_val.append(val_resid_pred.reshape(-1, 1))
            model_rows.append({
                "scale": scale_name,
                "model": model,
                "feature_cols": feature_cols,
                "baseline": baseline,
                "train_pred": train_pred,
                "val_pred": val_pred,
                "train_metrics": self.evaluate_prediction_bundle(ytrain, train_pred),
                "val_metrics": self.evaluate_prediction_bundle(yval, val_pred),
            })

        seq_scales = {
            "seq_short": [1, 2, 3],
            "seq_mid": [5, 10],
            "seq_long": [10, 20],
        }
        for scale_name, lags in seq_scales.items():
            xtrain_seq, ytrain_seq = self._build_sequence_features(raw_train, lags=lags)
            xval_seq, yval_seq = self._build_sequence_features(raw_val, lags=lags)
            log.inf(f"Training sequence expert: {scale_name} with shape {xtrain_seq.shape}")
            seq_model_name = "ridge"
            seq_ytrain = ytrain_seq.copy()
            seq_yval = yval_seq.copy()
            seq_ytrain["fret12"] = seq_ytrain["fret12"].to_numpy(dtype=np.float32) - train_common
            seq_yval["fret12"] = seq_yval["fret12"].to_numpy(dtype=np.float32) - val_common
            seq_model, seq_feature_cols, seq_baseline = self.fit_model(seq_model_name, xtrain_seq, seq_ytrain, target_mode="raw")
            seq_train_resid = self.predict(seq_model, xtrain_seq, seq_feature_cols)
            seq_val_resid = self.predict(seq_model, xval_seq, seq_feature_cols)
            seq_train_pred = train_common + seq_train_resid
            seq_val_pred = val_common + seq_val_resid
            preds_train.append(seq_train_resid.reshape(-1, 1))
            preds_val.append(seq_val_resid.reshape(-1, 1))
            model_rows.append({
                "scale": scale_name,
                "model": seq_model,
                "feature_cols": seq_feature_cols,
                "baseline": seq_baseline,
                "train_pred": seq_train_pred,
                "val_pred": seq_val_pred,
                "train_metrics": self.evaluate_prediction_bundle(ytrain_seq, seq_train_pred),
                "val_metrics": self.evaluate_prediction_bundle(yval_seq, seq_val_pred),
            })

        train_stack = np.hstack(preds_train)
        val_stack = np.hstack(preds_val)
        stack_xtrain = pd.DataFrame(train_stack, columns=[f"exp_{i}_pred" for i in range(train_stack.shape[1])])
        stack_xval = pd.DataFrame(val_stack, columns=[f"exp_{i}_pred" for i in range(val_stack.shape[1])])
        stack_ytrain = ytrain_resid.copy()
        stack_yval = yval_resid.copy()
        stack_model, stack_feature_cols, _ = self.fit_model("ridge", stack_xtrain, stack_ytrain, target_mode="raw")
        stack_train_resid = self.predict(stack_model, stack_xtrain, stack_feature_cols)
        stack_val_resid = self.predict(stack_model, stack_xval, stack_feature_cols)
        stack_train_pred = train_common + stack_train_resid
        stack_val_pred = val_common + stack_val_resid
        stack_result = {
            "model": stack_model,
            "feature_cols": stack_feature_cols,
            "baseline": common_model,
            "pred_train": stack_train_pred,
            "pred_val": stack_val_pred,
            "train_metrics": self.evaluate_prediction_bundle(ytrain, stack_train_pred),
            "val_metrics": self.evaluate_prediction_bundle(yval, stack_val_pred),
        }
        return {
            "experts": model_rows,
            "stacking": stack_result,
        }

    def run_regime_residual_fusion(self, split_config, model_name="tree", max_train_days=None, max_val_days=None, target_mode="interval_demean", residual_weight=0.3):
        train_dates, val_dates, _ = self.split_dates(split_config)
        xtrain_full, ytrain = self.load_feature_split(train_dates, max_days=max_train_days, groups=None)
        xval_full, yval = self.load_feature_split(val_dates, max_days=max_val_days, groups=None)

        general_model, feature_cols, baseline = self.fit_model(model_name, xtrain_full, ytrain, target_mode=target_mode)
        general_train_pred = self.predict(general_model, xtrain_full, feature_cols)
        general_val_pred = self.predict(general_model, xval_full, feature_cols)
        ytrain_resid = ytrain.copy()
        yval_resid = yval.copy()
        ytrain_resid["fret12"] = ytrain_resid["fret12"].to_numpy(dtype=np.float32) - general_train_pred
        yval_resid["fret12"] = yval_resid["fret12"].to_numpy(dtype=np.float32) - general_val_pred

        train_state = self._regime_state(xtrain_full)
        val_state = self._regime_state(xval_full)
        state_names = {0: "low", 1: "mid", 2: "high"}
        state_models = {}
        state_preds_train = np.zeros_like(general_train_pred, dtype=np.float32)
        state_preds_val = np.zeros_like(general_val_pred, dtype=np.float32)

        for state_id, state_name in state_names.items():
            train_mask = train_state == state_id
            val_mask = val_state == state_id
            if train_mask.sum() < 1000:
                continue
            log.inf(f"Training regime expert: {state_name} with train size {train_mask.sum()}")
            state_model, state_feature_cols, _ = self.fit_model(
                "ridge",
                xtrain_full.loc[train_mask].copy(),
                ytrain_resid.loc[train_mask].copy(),
                target_mode="raw",
            )
            train_resid_pred = self.predict(state_model, xtrain_full.loc[train_mask], state_feature_cols)
            val_resid_pred = self.predict(state_model, xval_full.loc[val_mask], state_feature_cols) if val_mask.any() else np.array([], dtype=np.float32)
            state_preds_train[train_mask] = train_resid_pred
            if val_mask.any():
                state_preds_val[val_mask] = val_resid_pred
            state_models[state_name] = {
                "model": state_model,
                "feature_cols": state_feature_cols,
                "train_mask": train_mask,
                "val_mask": val_mask,
            }

        weight_grid = np.arange(0.0, 1.0001, 0.05, dtype=np.float32)
        best_weight = float(residual_weight)
        best_score = None
        best_train_pred = None
        best_val_pred = None
        for weight in weight_grid:
            train_pred = general_train_pred + weight * state_preds_train
            val_pred = general_val_pred + weight * state_preds_val
            val_metrics = self.evaluate_predictions(yval, val_pred)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best_score is None or score > best_score:
                best_score = score
                best_weight = float(weight)
                best_train_pred = train_pred.copy()
                best_val_pred = val_pred.copy()
        train_pred = best_train_pred
        val_pred = best_val_pred
        train_metrics = self.evaluate_predictions(ytrain, train_pred)
        val_metrics = self.evaluate_predictions(yval, val_pred)
        log.inf(
            "Selected residual weight: {:.2f}".format(best_weight)
        )
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": general_model,
            "feature_cols": feature_cols,
            "baseline": baseline,
            "state_models": state_models,
            "residual_weight": best_weight,
            "pred_train": train_pred,
            "pred_val": val_pred,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def run_soft_regime_ensemble(self, split_config, max_train_days=None, max_val_days=None, target_mode="interval_demean"):
        train_dates, val_dates, _ = self.split_dates(split_config)
        xtrain_full, ytrain = self.load_feature_split(train_dates, max_days=max_train_days, groups=None)
        xval_full, yval = self.load_feature_split(val_dates, max_days=max_val_days, groups=None)

        main_result = self.run_with_groups(
            split_config=split_config,
            model_name="tree",
            feature_groups=["base", "lag", "roll", "cross"],
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
        )
        regime_result = self.run_with_groups(
            split_config=split_config,
            model_name="tree",
            feature_groups=["base", "lag", "roll", "cross", "regime"],
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
        )

        train_meta = pd.DataFrame({
            "main_pred": np.asarray(main_result["pred_train"], dtype=np.float32),
            "regime_pred": np.asarray(regime_result["pred_train"], dtype=np.float32),
            "regime_score": xtrain_full["regime_score"].to_numpy(dtype=np.float32),
            "regime_low": xtrain_full["regime_low"].to_numpy(dtype=np.float32),
            "regime_mid": xtrain_full["regime_mid"].to_numpy(dtype=np.float32),
            "regime_high": xtrain_full["regime_high"].to_numpy(dtype=np.float32),
        })
        val_meta = pd.DataFrame({
            "main_pred": np.asarray(main_result["pred_val"], dtype=np.float32),
            "regime_pred": np.asarray(regime_result["pred_val"], dtype=np.float32),
            "regime_score": xval_full["regime_score"].to_numpy(dtype=np.float32),
            "regime_low": xval_full["regime_low"].to_numpy(dtype=np.float32),
            "regime_mid": xval_full["regime_mid"].to_numpy(dtype=np.float32),
            "regime_high": xval_full["regime_high"].to_numpy(dtype=np.float32),
        })
        meta_ytrain = ytrain.copy()
        meta_yval = yval.copy()

        candidate_models = [
            (
                "ridge",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0, fit_intercept=True, random_state=None)),
                ]),
            ),
            (
                "tree",
                ExtraTreesRegressor(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=400,
                    max_features=1.0,
                    bootstrap=True,
                    max_samples=0.8,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
        best_name = None
        best_model = None
        best_pred_train = None
        best_pred_val = None
        best_metrics = None
        best_train_metrics = None
        for name, meta_model in candidate_models:
            meta_model.fit(train_meta, meta_ytrain["fret12"].to_numpy(dtype=np.float32))
            pred_train = np.asarray(meta_model.predict(train_meta), dtype=np.float32)
            pred_val = np.asarray(meta_model.predict(val_meta), dtype=np.float32)
            train_metrics = self.evaluate_prediction_bundle(meta_ytrain, pred_train)
            val_metrics = self.evaluate_prediction_bundle(meta_yval, pred_val)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best_metrics is None or score > (best_metrics["corr"], -best_metrics["mse"], best_metrics["r2"]):
                best_name = name
                best_model = meta_model
                best_pred_train = pred_train
                best_pred_val = pred_val
                best_metrics = val_metrics
                best_train_metrics = train_metrics
        post_params, post_train_metrics = self.choose_postprocess_params(meta_ytrain, best_pred_train)
        train_post_frame = meta_ytrain[["date", "interval", "fret12"]].copy()
        train_post_frame["pred"] = np.asarray(best_pred_train, dtype=np.float32)
        val_post_frame = meta_yval[["date", "interval", "fret12"]].copy()
        val_post_frame["pred"] = np.asarray(best_pred_val, dtype=np.float32)
        final_train_pred = np.asarray(best_pred_train, dtype=np.float32)
        final_val_pred = np.asarray(best_pred_val, dtype=np.float32)
        train_metrics = self.evaluate_prediction_bundle(meta_ytrain, final_train_pred)
        val_metrics = self.evaluate_prediction_bundle(meta_yval, final_val_pred)
        log.inf(
            "Soft meta model selected: {}".format(best_name)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": best_model,
            "feature_cols": list(train_meta.columns),
            "pred_train": final_train_pred,
            "pred_val": final_val_pred,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "main_result": main_result,
            "regime_result": regime_result,
            "postprocess_params": post_params,
            "postprocess_train_metrics": post_train_metrics,
        }

    def run_pair_blend_search(
        self,
        split_config,
        first_kind="common_residual",
        second_kind="soft_regime",
        max_train_days=None,
        max_val_days=None,
    ):
        candidates = {}
        if first_kind == "common_residual":
            candidates[first_kind] = self.run_common_residual_reconstruction(
                split_config=split_config,
                residual_model_name="tree",
                common_model_name="ridge",
                feature_groups=None,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
        elif first_kind == "soft_regime":
            candidates[first_kind] = self.run_soft_regime_ensemble(
                split_config=split_config,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode="interval_demean",
            )
        else:
            raise ValueError(f"Unknown first_kind: {first_kind}")

        if second_kind == "common_residual":
            candidates[second_kind] = self.run_common_residual_reconstruction(
                split_config=split_config,
                residual_model_name="tree",
                common_model_name="ridge",
                feature_groups=None,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
        elif second_kind == "soft_regime":
            candidates[second_kind] = self.run_soft_regime_ensemble(
                split_config=split_config,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode="interval_demean",
            )
        else:
            raise ValueError(f"Unknown second_kind: {second_kind}")

        train_a = np.asarray(candidates[first_kind]["pred_train"], dtype=np.float32)
        val_a = np.asarray(candidates[first_kind]["pred_val"], dtype=np.float32)
        train_b = np.asarray(candidates[second_kind]["pred_train"], dtype=np.float32)
        val_b = np.asarray(candidates[second_kind]["pred_val"], dtype=np.float32)
        ytrain = self.load_feature_split(self.split_dates(split_config)[0], max_days=max_train_days, groups=None)[1]
        yval = self.load_feature_split(self.split_dates(split_config)[1], max_days=max_val_days, groups=None)[1]

        best = None
        best_w = None
        best_train = None
        best_val = None
        for w in np.arange(0.0, 1.0001, 0.05, dtype=np.float32):
            train_pred = w * train_a + (1.0 - w) * train_b
            val_pred = w * val_a + (1.0 - w) * val_b
            val_metrics = self.evaluate_prediction_bundle(yval, val_pred)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best is None or score > best:
                best = score
                best_w = float(w)
                best_train = train_pred.copy()
                best_val = val_pred.copy()
        train_metrics = self.evaluate_prediction_bundle(ytrain, best_train)
        val_metrics = self.evaluate_prediction_bundle(yval, best_val)
        return {
            "weight_first": best_w,
            "first_kind": first_kind,
            "second_kind": second_kind,
            "pred_train": best_train,
            "pred_val": best_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "first_result": candidates[first_kind],
            "second_result": candidates[second_kind],
        }

    def run_rolling_validation_suite(
        self,
        split_config,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=4,
        max_train_days=None,
        max_val_days=None,
        embargo=1,
        specs=None,
    ):
        train_dates, _, _ = self.split_dates(split_config)
        folds = self.make_rolling_folds(
            train_dates[0],
            train_dates[-1],
            train_window=train_window,
            val_window=val_window,
            step=step,
            min_train_days=train_window,
            embargo=embargo,
        )[:max_folds]
        if not folds:
            raise ValueError("No rolling folds available")

        if specs is None:
            specs = [
                {
                    "experiment_id": "R00_baseline_ridge",
                    "type": "standard",
                    "model": "ridge",
                    "target_mode": "raw",
                    "groups": ["legacy"],
                    "notes": "rolling baseline ridge",
                },
                {
                    "experiment_id": "R01_common_residual",
                    "type": "common_residual",
                    "notes": "rolling common plus residual",
                },
                {
                    "experiment_id": "R02_soft_regime_ensemble",
                    "type": "soft_regime",
                    "notes": "rolling soft gating ensemble",
                },
                {
                    "experiment_id": "R03_pair_blend",
                    "type": "pair_blend",
                    "notes": "rolling blend of common residual and soft regime",
                },
            ]

        rows = []
        for fold in folds:
            fold_split = SplitConfig(
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[-1],
                val_start=fold.val_dates[0],
                val_end=fold.val_dates[-1],
                test_start=fold.val_dates[0],
                test_end=fold.val_dates[-1],
            )
            log.inf(
                f"Rolling fold {fold.fold_id}: train {fold.train_dates[0]} -> {fold.train_dates[-1]}, "
                f"val {fold.val_dates[0]} -> {fold.val_dates[-1]}"
            )
            for spec in specs:
                start_ts = time.time()
                if spec["type"] == "standard":
                    result = self.run_with_groups(
                        split_config=fold_split,
                        model_name=spec["model"],
                        feature_groups=spec["groups"],
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        target_mode=spec["target_mode"],
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_set = json.dumps(result["feature_groups"], ensure_ascii=False)
                    target_type = spec["target_mode"]
                    model_type = spec["model"]
                    postprocess_type = "none"
                elif spec["type"] == "common_residual":
                    result = self.run_common_residual_reconstruction(
                        split_config=fold_split,
                        residual_model_name="tree",
                        common_model_name="ridge",
                        feature_groups=None,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_set = json.dumps(["base", "lag", "roll", "cross", "common_aggregate"], ensure_ascii=False)
                    target_type = "common_plus_residual"
                    model_type = "common_residual"
                    postprocess_type = "none"
                elif spec["type"] == "soft_regime":
                    result = self.run_soft_regime_ensemble(
                        split_config=fold_split,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        target_mode="interval_demean",
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_set = json.dumps(["main_tree", "regime_tree", "regime_score"], ensure_ascii=False)
                    target_type = "interval_demean"
                    model_type = "tree_soft_regime"
                    postprocess_type = "none"
                elif spec["type"] == "pair_blend":
                    result = self.run_pair_blend_search(
                        split_config=fold_split,
                        first_kind="common_residual",
                        second_kind="soft_regime",
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_set = json.dumps(["common_residual", "soft_regime"], ensure_ascii=False)
                    target_type = "raw"
                    model_type = "pair_blend"
                    postprocess_type = json.dumps({"weight_first": result["weight_first"]}, ensure_ascii=False)
                else:
                    raise ValueError(f"Unknown rolling suite spec: {spec['type']}")

                rows.append({
                    "fold_id": fold.fold_id,
                    "experiment_id": spec["experiment_id"],
                    "feature_set": feature_set,
                    "target_type": target_type,
                    "model_type": model_type,
                    "postprocess_type": postprocess_type,
                    "train_corr": train_metrics["corr"],
                    "val_corr": val_metrics["corr"],
                    "train_mse": train_metrics["mse"],
                    "val_mse": val_metrics["mse"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "daily_corr_mean": val_metrics["daily_corr_mean"],
                    "daily_corr_std": val_metrics["daily_corr_std"],
                    "rolling_corr_mean": np.nan,
                    "rolling_corr_std": np.nan,
                    "rolling_corr_min": np.nan,
                    "stability_score": np.nan,
                    "train_val_corr_gap": self._corr_gap(train_metrics, val_metrics),
                    "runtime_sec": float(time.time() - start_ts),
                    "random_seed": 42,
                    "notes": spec["notes"],
                })

        df = pd.DataFrame(rows)
        summary_rows = []
        for experiment_id, group in df.groupby("experiment_id", sort=False):
            rolling_corr_mean = float(group["val_corr"].mean())
            rolling_corr_std = float(group["val_corr"].std(ddof=0)) if len(group) > 1 else 0.0
            rolling_corr_min = float(group["val_corr"].min())
            summary_rows.append({
                "fold_id": "summary",
                "experiment_id": experiment_id,
                "feature_set": group["feature_set"].iloc[0],
                "target_type": group["target_type"].iloc[0],
                "model_type": group["model_type"].iloc[0],
                "postprocess_type": group["postprocess_type"].iloc[0],
                "train_corr": group["train_corr"].mean(),
                "val_corr": group["val_corr"].mean(),
                "train_mse": group["train_mse"].mean(),
                "val_mse": group["val_mse"].mean(),
                "train_r2": group["train_r2"].mean(),
                "val_r2": group["val_r2"].mean(),
                "daily_corr_mean": group["daily_corr_mean"].mean(),
                "daily_corr_std": group["daily_corr_std"].mean(),
                "rolling_corr_mean": rolling_corr_mean,
                "rolling_corr_std": rolling_corr_std,
                "rolling_corr_min": rolling_corr_min,
                "stability_score": rolling_corr_mean - 0.7 * rolling_corr_std,
                "train_val_corr_gap": group["train_val_corr_gap"].mean(),
                "runtime_sec": group["runtime_sec"].sum(),
                "random_seed": 42,
                "notes": f"rolling summary over {len(group)} folds",
            })
        return pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    def run_val_blend_search(self, split_config, max_train_days=None, max_val_days=None, target_mode="interval_demean"):
        base_specs = [
            {
                "name": "main_tree",
                "model_name": "tree",
                "feature_groups": ["base", "lag", "roll", "cross"],
            },
            {
                "name": "regime_tree",
                "model_name": "tree",
                "feature_groups": ["base", "lag", "roll", "cross", "regime"],
            },
            {
                "name": "ridge_full",
                "model_name": "ridge",
                "feature_groups": None,
            },
        ]
        results = []
        for spec in base_specs:
            result = self.run_with_groups(
                split_config=split_config,
                model_name=spec["model_name"],
                feature_groups=spec["feature_groups"],
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=target_mode,
            )
            results.append((spec["name"], result))

        train_stack = np.column_stack([np.asarray(res["pred_train"], dtype=np.float32) for _, res in results])
        val_stack = np.column_stack([np.asarray(res["pred_val"], dtype=np.float32) for _, res in results])
        train_y = results[0][1]["train_metrics"]  # placeholder for shape validation
        _ = train_y
        train_target = self.load_feature_split(
            self.calendar.range(split_config.train_start, split_config.train_end),
            max_days=max_train_days,
            groups=None,
        )[1]["fret12"].to_numpy(dtype=np.float32)
        val_target = self.load_feature_split(
            self.calendar.range(split_config.val_start, split_config.val_end),
            max_days=max_val_days,
            groups=None,
        )[1]["fret12"].to_numpy(dtype=np.float32)

        best = None
        best_weights = None
        grid = np.arange(0.0, 1.0001, 0.05, dtype=np.float32)
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                if w2 > 1.0 + 1e-9:
                    continue
                weights = np.array([w0, w1, w2], dtype=np.float32)
                train_pred = train_stack @ weights
                val_pred = val_stack @ weights
                metrics = self.evaluate_predictions(pd.DataFrame({"fret12": val_target}), val_pred)
                score = (metrics["corr"], -metrics["mse"], metrics["r2"])
                if best is None or score > best:
                    best = score
                    best_weights = weights.copy()
                    best_train_pred = train_pred.copy()
                    best_val_pred = val_pred.copy()

        train_metrics = self.evaluate_predictions(pd.DataFrame({"fret12": train_target}), best_train_pred)
        val_metrics = self.evaluate_predictions(pd.DataFrame({"fret12": val_target}), best_val_pred)
        log.inf(
            "Blend weights - main_tree={:.2f}, regime_tree={:.2f}, ridge_full={:.2f}".format(
                float(best_weights[0]), float(best_weights[1]), float(best_weights[2])
            )
        )
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "base_results": results,
            "blend_weights": best_weights,
            "pred_train": best_train_pred,
            "pred_val": best_val_pred,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def _build_rolling_folds(self, split_config, train_window=8, val_window=2, step=10, max_folds=5, embargo=1):
        train_dates, _, _ = self.split_dates(split_config)
        folds = self.make_rolling_folds(
            train_dates[0],
            train_dates[-1],
            train_window=train_window,
            val_window=val_window,
            step=step,
            min_train_days=train_window,
            embargo=embargo,
        )[:max_folds]
        if not folds:
            raise ValueError("No rolling folds available")
        return folds

    def _evaluate_spec_on_fold(self, fold_split, spec, max_train_days=None, max_val_days=None):
        start_ts = time.time()
        coef_df = None
        if spec["type"] == "standard":
            result = self.run_with_groups(
                split_config=fold_split,
                model_name=spec["model"],
                feature_groups=spec["groups"],
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=spec.get("target_mode", "raw"),
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(result["feature_groups"], ensure_ascii=False)
            target_type = spec.get("target_mode", "raw")
            model_type = spec["model"]
            postprocess_type = "none"
            if spec.get("collect_coefs"):
                xtrain, ytrain = self.load_feature_split(
                    self.calendar.range(fold_split.train_start, fold_split.train_end),
                    max_days=max_train_days,
                    groups=spec["groups"],
                )
                xtrain = xtrain.loc[:, ~xtrain.columns.duplicated()].copy()
                model, feature_cols, _ = self.fit_model(spec["model"], xtrain, ytrain, target_mode=spec.get("target_mode", "raw"))
                coef_df = self._extract_linear_coefficients(model, feature_cols)
        elif spec["type"] == "common_residual":
            result = self.run_common_residual_reconstruction(
                split_config=fold_split,
                residual_model_name=spec.get("residual_model", "tree"),
                common_model_name=spec.get("common_model", "ridge"),
                feature_groups=spec.get("feature_groups"),
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(spec.get("feature_groups") or ["base", "lag", "roll", "cross", "common_aggregate"], ensure_ascii=False)
            target_type = "common_plus_residual"
            model_type = "common_residual"
            postprocess_type = "none"
        elif spec["type"] == "soft_regime":
            result = self.run_soft_regime_ensemble(
                split_config=fold_split,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=spec.get("target_mode", "interval_demean"),
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(["main_tree", "regime_tree", "regime_score"], ensure_ascii=False)
            target_type = spec.get("target_mode", "interval_demean")
            model_type = "tree_soft_regime"
            postprocess_type = "none"
        else:
            raise ValueError(f"Unknown spec type: {spec['type']}")
        return {
            "result": result,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_set": feature_set,
            "target_type": target_type,
            "model_type": model_type,
            "postprocess_type": postprocess_type,
            "runtime_sec": float(time.time() - start_ts),
            "coef_df": coef_df,
        }

    def run_formal_rolling_suite(
        self,
        split_config,
        specs,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
        max_train_days=None,
        max_val_days=None,
    ):
        folds = self._build_rolling_folds(
            split_config,
            train_window=train_window,
            val_window=val_window,
            step=step,
            max_folds=max_folds,
            embargo=embargo,
        )
        rows = []
        coef_payload = {}
        for fold in folds:
            fold_split = SplitConfig(
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[-1],
                val_start=fold.val_dates[0],
                val_end=fold.val_dates[-1],
                test_start=fold.val_dates[0],
                test_end=fold.val_dates[-1],
            )
            log.inf(
                f"Formal rolling fold {fold.fold_id}: train {fold.train_dates[0]} -> {fold.train_dates[-1]}, "
                f"val {fold.val_dates[0]} -> {fold.val_dates[-1]}"
            )
            for spec in specs:
                bundle = self._evaluate_spec_on_fold(
                    fold_split,
                    spec,
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                )
                rows.append(
                    self._fold_metric_row(
                        fold_id=fold.fold_id,
                        experiment_id=spec["experiment_id"],
                        feature_set=bundle["feature_set"],
                        target_type=bundle["target_type"],
                        model_type=bundle["model_type"],
                        postprocess_type=bundle["postprocess_type"],
                        train_metrics=bundle["train_metrics"],
                        val_metrics=bundle["val_metrics"],
                        runtime_sec=bundle["runtime_sec"],
                        notes=spec.get("notes", ""),
                    )
                )
                if bundle["coef_df"] is not None:
                    coef_payload.setdefault(spec["experiment_id"], []).append(bundle["coef_df"].assign(fold_id=fold.fold_id))
        return self._summarize_rolling_rows(rows, len(folds)), pd.DataFrame(rows), coef_payload

    def run_train_window_sensitivity_suite(
        self,
        split_config,
        spec,
        train_windows,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
        max_val_days=None,
    ):
        summary_rows = []
        fold_frames = []
        for train_days in train_windows:
            summary, fold_rows, _ = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=[spec],
                train_window=train_window,
                val_window=val_window,
                step=step,
                max_folds=max_folds,
                embargo=embargo,
                max_train_days=train_days,
                max_val_days=max_val_days,
            )
            row = summary.iloc[0].to_dict()
            row["experiment_name"] = f'{spec["experiment_id"]}_max_train_{train_days if train_days is not None else "expanding"}'
            row["max_train_days"] = -1 if train_days is None else int(train_days)
            row["base_experiment_id"] = spec["experiment_id"]
            row["reason"] = spec.get("notes", "")
            summary_rows.append(row)
            fold_rows = fold_rows.copy()
            fold_rows["max_train_days"] = -1 if train_days is None else int(train_days)
            fold_frames.append(fold_rows)
        out = pd.DataFrame(summary_rows)
        if not out.empty:
            ordered = [
                "experiment_name",
                "base_experiment_id",
                "max_train_days",
                "model_type",
                "feature_set",
                "target_type",
            ] + [c for c in out.columns if c.startswith("fold")] + [
                "rolling_corr_mean",
                "rolling_corr_std",
                "rolling_corr_min",
                "stability_score",
                "rolling_mse_mean",
                "rolling_r2_mean",
                "daily_corr_mean",
                "daily_corr_std",
                "是否进入下一阶段",
                "原因",
            ]
            out = out.reindex(columns=[c for c in ordered if c in out.columns])
        fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        return out, fold_df

    def summarize_feature_coefficients(self, coef_payload, feature_names, experiment_id):
        frames = coef_payload.get(experiment_id, [])
        if not frames:
            return pd.DataFrame()
        coef_df = pd.concat(frames, ignore_index=True)
        coef_df = coef_df[coef_df["feature"].isin(feature_names)].copy()
        if coef_df.empty:
            return pd.DataFrame()
        ranked_frames = []
        for fold_id, fold_group in coef_df.groupby("fold_id", sort=True):
            fold_group = fold_group.sort_values("abs_coef", ascending=False).reset_index(drop=True)
            fold_group["abs_rank"] = np.arange(1, len(fold_group) + 1)
            ranked_frames.append(fold_group)
        ranked = pd.concat(ranked_frames, ignore_index=True)
        summary_rows = []
        for feature, grp in ranked.groupby("feature", sort=False):
            abs_values = grp["abs_coef"].to_numpy(dtype=np.float32)
            coefs = grp["coef"].to_numpy(dtype=np.float32)
            ranks = grp["abs_rank"].to_numpy(dtype=np.float32)
            summary_rows.append({
                "feature": feature,
                "mean_coef": float(np.mean(coefs)),
                "mean_abs_coef": float(np.mean(abs_values)),
                "std_abs_coef": float(np.std(abs_values, ddof=0)),
                "mean_abs_rank": float(np.mean(ranks)),
                "sign_consistency": float(np.mean(np.sign(coefs) == np.sign(np.mean(coefs)))),
                "fold_stability": float(1.0 / (1.0 + np.std(abs_values, ddof=0))),
                "n_folds": int(grp["fold_id"].nunique()),
            })
        return pd.DataFrame(summary_rows).sort_values(["mean_abs_coef", "mean_abs_rank"], ascending=[False, True]).reset_index(drop=True)

    def choose_weighted_fusion(
        self,
        split_config,
        backbone_groups,
        max_train_days=None,
        max_val_days=None,
        train_window=8,
        val_window=2,
        step=10,
        max_folds=5,
        embargo=1,
    ):
        folds = self._build_rolling_folds(
            split_config,
            train_window=train_window,
            val_window=val_window,
            step=step,
            max_folds=max_folds,
            embargo=embargo,
        )
        candidate_weights = []
        for w0 in [0.70, 0.75, 0.80, 0.85, 0.90]:
            for w1 in [0.00, 0.05, 0.10, 0.15]:
                for w2 in [0.00, 0.05, 0.10, 0.15]:
                    if abs((w0 + w1 + w2) - 1.0) < 1e-9:
                        candidate_weights.append((w0, w1, w2))
        if not candidate_weights:
            raise ValueError("No valid fusion weights")

        rows = []
        for fold in folds:
            fold_split = SplitConfig(
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[-1],
                val_start=fold.val_dates[0],
                val_end=fold.val_dates[-1],
                test_start=fold.val_dates[0],
                test_end=fold.val_dates[-1],
            )
            backbone_bundle = self._evaluate_spec_on_fold(
                fold_split,
                {"type": "standard", "experiment_id": "backbone", "model": "ridge", "groups": backbone_groups, "target_mode": "raw", "notes": "backbone"},
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            common_bundle = self._evaluate_spec_on_fold(
                fold_split,
                {"type": "common_residual", "experiment_id": "common", "notes": "common"},
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            regime_bundle = self._evaluate_spec_on_fold(
                fold_split,
                {"type": "soft_regime", "experiment_id": "regime", "target_mode": "interval_demean", "notes": "regime"},
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            yval = self.load_feature_split(
                self.calendar.range(fold.val_dates[0], fold.val_dates[-1]),
                max_days=max_val_days,
                groups=None,
            )[1]
            backbone_pred = np.asarray(backbone_bundle["result"]["pred_val"], dtype=np.float32)
            common_pred = np.asarray(common_bundle["result"]["pred_val"], dtype=np.float32)
            regime_pred = np.asarray(regime_bundle["result"]["pred_val"], dtype=np.float32)
            for w0, w1, w2 in candidate_weights:
                pred = w0 * backbone_pred + w1 * common_pred + w2 * regime_pred
                metrics = self.evaluate_prediction_bundle(yval, pred)
                rows.append({
                    "fold_id": fold.fold_id,
                    "weight_w0": w0,
                    "weight_w1": w1,
                    "weight_w2": w2,
                    "corr": metrics["corr"],
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "daily_corr_mean": metrics["daily_corr_mean"],
                    "daily_corr_std": metrics["daily_corr_std"],
                })
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(["weight_w0", "weight_w1", "weight_w2"], sort=False)
            .agg(
                rolling_corr_mean=("corr", "mean"),
                rolling_corr_std=("corr", lambda s: float(np.std(s.to_numpy(dtype=np.float32), ddof=0)) if len(s) > 1 else 0.0),
                rolling_corr_min=("corr", "min"),
                rolling_mse_mean=("mse", "mean"),
                rolling_r2_mean=("r2", "mean"),
                daily_corr_mean=("daily_corr_mean", "mean"),
                daily_corr_std=("daily_corr_std", "mean"),
            )
            .reset_index()
        )
        summary["stability_score"] = summary["rolling_corr_mean"] - 0.7 * summary["rolling_corr_std"]
        best = summary.sort_values(["stability_score", "rolling_corr_mean", "rolling_corr_min"], ascending=[False, False, False]).iloc[0]
        return df, summary, best.to_dict()

    def run_suite(self, split_config, suite_name="ablation", max_train_days=None, max_val_days=None):
        suite_name = (suite_name or "ablation").lower()
        if suite_name == "train_window_sensitivity_quick":
            spec = {
                "experiment_id": "R02_ridge_legacy_plus_norm_core",
                "type": "standard",
                "model": "ridge",
                "target_mode": "raw",
                "groups": ["legacy", "norm_core"],
                "notes": "train window sensitivity on best ridge backbone",
                "collect_coefs": False,
            }
            summary, fold_rows = self.run_train_window_sensitivity_suite(
                split_config=split_config,
                spec=spec,
                train_windows=[4, 80],
                train_window=80,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "train_window_sensitivity":
            spec = {
                "experiment_id": "R02_ridge_legacy_plus_norm_core",
                "type": "standard",
                "model": "ridge",
                "target_mode": "raw",
                "groups": ["legacy", "norm_core"],
                "notes": "train window sensitivity on best ridge backbone",
                "collect_coefs": False,
            }
            summary, fold_rows = self.run_train_window_sensitivity_suite(
                split_config=split_config,
                spec=spec,
                train_windows=[4, 10, 20, 40, 80, None],
                train_window=80,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "ofi_audit":
            specs = [
                {"experiment_id": "O1_R02_plus_ofi_raw", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw"], "notes": "R02 plus raw OFI"},
                {"experiment_id": "O2_R02_plus_ofi_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_dynamic"], "notes": "R02 plus dynamic OFI"},
                {"experiment_id": "O3_R02_plus_ofi_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_rank", "ofi_raw"], "notes": "R02 plus OFI cross ranks"},
                {"experiment_id": "O4_R02_plus_ofi_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe"], "notes": "R02 plus all safe OFI"},
                {"experiment_id": "O5_R02_plus_ofi_raw_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw", "ofi_dynamic"], "notes": "R02 plus raw and dynamic OFI"},
                {"experiment_id": "O6_R02_plus_all_ofi", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe", "ofi_rank"], "notes": "R02 plus all OFI groups"},
            ]
            summary, fold_rows, _ = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=specs,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "trade_impact_audit":
            specs = [
                {"experiment_id": "T1_R02_plus_trade_impact", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact"], "notes": "R02 plus trade impact base"},
                {"experiment_id": "T2_R02_plus_trade_impact_dyn", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_dyn"], "notes": "R02 plus trade impact dynamic"},
                {"experiment_id": "T3_R02_plus_trade_impact_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_interaction"], "notes": "R02 plus trade impact interactions"},
                {"experiment_id": "T4_R02_plus_trade_impact_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_safe"], "notes": "R02 plus all safe trade impact"},
            ]
            summary, fold_rows, _ = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=specs,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "conditional_momentum_audit":
            specs = [
                {"experiment_id": "C1_R02_plus_conditional_momentum", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum"], "notes": "R02 plus conditional momentum base"},
                {"experiment_id": "C2_R02_plus_conditional_momentum_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum_interaction"], "notes": "R02 plus conditional momentum interactions"},
                {"experiment_id": "C3_R02_plus_conditional_momentum_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum_safe"], "notes": "R02 plus all safe conditional momentum"},
            ]
            summary, fold_rows, _ = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=specs,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "formal_backbone":
            specs = [
                {"experiment_id": "B0_ridge_legacy", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy"], "notes": "formal ridge legacy backbone", "collect_coefs": True},
                {"experiment_id": "B6_common_residual", "type": "common_residual", "notes": "formal common residual branch"},
                {"experiment_id": "B7_soft_regime", "type": "soft_regime", "notes": "formal soft regime ensemble"},
                {"experiment_id": "B8_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "formal ridge legacy plus core", "collect_coefs": True},
            ]
            summary, fold_rows, coef_payload = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=specs,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            summary.attrs["coef_payload"] = coef_payload
            return summary
        if suite_name == "ridge_enhance":
            specs = [
                {"experiment_id": "R00_ridge_legacy", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy"], "notes": "ridge legacy"},
                {"experiment_id": "R01_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "ridge legacy plus core"},
                {"experiment_id": "R02_ridge_legacy_plus_norm_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core"], "notes": "ridge legacy plus normalized core"},
                {"experiment_id": "R03_ridge_legacy_plus_patch_summary", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "patch_summary"], "notes": "ridge legacy plus patch summary"},
                {"experiment_id": "R04_ridge_legacy_plus_cross_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "cross_rank_features"], "notes": "ridge legacy plus cross-sectional ranks"},
            ]
            summary, fold_rows, _ = self.run_formal_rolling_suite(
                split_config=split_config,
                specs=specs,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                embargo=1,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
            summary.attrs["fold_rows"] = fold_rows
            return summary
        if suite_name == "restricted_fusion":
            fusion_rows = []
            for backbone_id, backbone_groups in [
                ("B0_ridge_legacy", ["legacy"]),
                ("B8_ridge_legacy_plus_core", ["legacy", "base", "lag", "roll", "cross"]),
            ]:
                per_fold_rows, summary, best = self.choose_weighted_fusion(
                    split_config=split_config,
                    backbone_groups=backbone_groups,
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                    train_window=8,
                    val_window=2,
                    step=10,
                    max_folds=5,
                    embargo=1,
                )
                best_row = summary.sort_values(["stability_score", "rolling_corr_mean", "rolling_corr_min"], ascending=[False, False, False]).iloc[0].to_dict()
                best_weights = (best_row["weight_w0"], best_row["weight_w1"], best_row["weight_w2"])
                best_fold = per_fold_rows[
                    (per_fold_rows["weight_w0"] == best_weights[0])
                    & (per_fold_rows["weight_w1"] == best_weights[1])
                    & (per_fold_rows["weight_w2"] == best_weights[2])
                ].sort_values("fold_id")
                best_row.update({
                    "experiment_name": f"{backbone_id}_plus_common_regime",
                    "model_type": "restricted_fusion",
                    "feature_set": json.dumps({"backbone": backbone_id, "common_expert": "B6_common_residual", "regime_expert": "B7_soft_regime", "weights": {
                        "w0": float(best_row["weight_w0"]),
                        "w1": float(best_row["weight_w1"]),
                        "w2": float(best_row["weight_w2"]),
                    }}, ensure_ascii=False),
                    "target_type": "raw",
                    "fold1_corr": float(best_fold.iloc[0]["corr"]) if len(best_fold) > 0 else np.nan,
                    "fold2_corr": float(best_fold.iloc[1]["corr"]) if len(best_fold) > 1 else np.nan,
                    "fold3_corr": float(best_fold.iloc[2]["corr"]) if len(best_fold) > 2 else np.nan,
                    "fold4_corr": float(best_fold.iloc[3]["corr"]) if len(best_fold) > 3 else np.nan,
                    "fold5_corr": float(best_fold.iloc[4]["corr"]) if len(best_fold) > 4 else np.nan,
                    "是否进入下一阶段": "是" if best_row["stability_score"] >= 0 else "否",
                    "原因": "受限融合候选",
                })
                fusion_rows.append(best_row)
            return pd.DataFrame(fusion_rows)
        if suite_name == "stage0":
            return self.run_rolling_validation_suite(
                split_config=split_config,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                embargo=1,
                specs=[
                    {
                        "experiment_id": "B0_ridge_legacy",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["legacy"],
                        "notes": "stage0 legacy ridge baseline",
                    },
                    {
                        "experiment_id": "B1_ridge_core",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 ridge core features",
                    },
                    {
                        "experiment_id": "B2_tree_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree core features",
                    },
                    {
                        "experiment_id": "B3_tree_residual_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree residual core features",
                    },
                    {
                        "experiment_id": "B4_hgb_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb core features",
                    },
                    {
                        "experiment_id": "B5_hgb_residual_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb residual core features",
                    },
                    {
                        "experiment_id": "B6_common_residual",
                        "type": "common_residual",
                        "notes": "stage0 common residual branch",
                    },
                    {
                        "experiment_id": "B7_soft_regime",
                        "type": "soft_regime",
                        "notes": "stage0 current soft regime ensemble",
                    },
                ],
            )
        if suite_name == "stage0_quick":
            return self.run_rolling_validation_suite(
                split_config=split_config,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=3,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                embargo=1,
                specs=[
                    {
                        "experiment_id": "B0_ridge_legacy",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["legacy"],
                        "notes": "stage0 legacy ridge baseline",
                    },
                    {
                        "experiment_id": "B1_ridge_core",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 ridge core features",
                    },
                    {
                        "experiment_id": "B2_tree_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree core features",
                    },
                    {
                        "experiment_id": "B3_tree_residual_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree residual core features",
                    },
                    {
                        "experiment_id": "B4_hgb_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb core features",
                    },
                    {
                        "experiment_id": "B5_hgb_residual_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb residual core features",
                    },
                    {
                        "experiment_id": "B6_common_residual",
                        "type": "common_residual",
                        "notes": "stage0 common residual branch",
                    },
                    {
                        "experiment_id": "B7_soft_regime",
                        "type": "soft_regime",
                        "notes": "stage0 current soft regime ensemble",
                    },
                ],
            )
        if suite_name == "stage1":
            experiments = [
                {"name": "E0_legacy_ridge", "model": "ridge", "target_mode": "raw", "groups": ["legacy"]},
                {"name": "E1_base_ridge", "model": "ridge", "target_mode": "raw", "groups": ["base"]},
                {"name": "E2_base_lag_ridge", "model": "ridge", "target_mode": "raw", "groups": ["base", "lag"]},
                {"name": "E3_base_lag_roll_ridge", "model": "ridge", "target_mode": "raw", "groups": ["base", "lag", "roll"]},
                {"name": "E4_full_ridge", "model": "ridge", "target_mode": "raw", "groups": ["base", "lag", "roll", "cross"]},
            ]
        elif suite_name == "stage2":
            experiments = [
                {"name": "E5_full_interval_demean_ridge", "model": "ridge", "target_mode": "interval_demean", "groups": None},
                {"name": "E6_full_interval_demean_tree", "model": "tree", "target_mode": "interval_demean", "groups": None},
                {"name": "E7_full_interval_demean_gbdt", "model": "gbdt", "target_mode": "interval_demean", "groups": None},
                {"name": "E8_full_interval_demean_mlp", "model": "mlp", "target_mode": "interval_demean", "groups": None},
            ]
        elif suite_name == "ablation":
            experiments = [
                {"name": "full_interval_demean_tree", "model": "tree", "target_mode": "interval_demean", "groups": None},
                {"name": "minus_lag", "model": "tree", "target_mode": "interval_demean", "groups": ["base", "roll", "cross"]},
                {"name": "minus_roll", "model": "tree", "target_mode": "interval_demean", "groups": ["base", "lag", "cross"]},
                {"name": "minus_cross", "model": "tree", "target_mode": "interval_demean", "groups": ["base", "lag", "roll"]},
                {"name": "ridge_full", "model": "ridge", "target_mode": "interval_demean", "groups": None},
            ]
        elif suite_name == "v2":
            experiments = [
                {"name": "v2_tabular_residual_tree", "model": "tree", "target_mode": "interval_residual", "groups": None},
                {"name": "v2_tabular_residual_ridge", "model": "ridge", "target_mode": "interval_residual", "groups": None},
                {"name": "v2_multiscale_fusion_tree", "kind": "fusion", "model": "tree", "target_mode": "interval_residual", "groups": None},
                {"name": "v2_regime_residual_tree", "kind": "regime", "model": "tree", "target_mode": "interval_demean", "groups": None},
                {"name": "v2_soft_regime_ensemble", "kind": "soft", "model": "tree", "target_mode": "interval_demean", "groups": None},
            ]
        elif suite_name == "v31":
            rows = []
            base_groups = ["base", "lag", "roll", "cross"]
            regime_groups = ["base", "lag", "roll", "cross", "regime"]
            protocols = [
                {
                    "name": "E00_baseline_ridge",
                    "type": "standard",
                    "model": "ridge",
                    "target_mode": "raw",
                    "groups": ["legacy"],
                    "notes": "baseline raw fret12",
                },
                {
                    "name": "E01_full_ridge_raw",
                    "type": "standard",
                    "model": "ridge",
                    "target_mode": "raw",
                    "groups": None,
                    "notes": "full features raw target",
                },
                {
                    "name": "E02_full_tree_raw",
                    "type": "standard",
                    "model": "tree",
                    "target_mode": "raw",
                    "groups": None,
                    "notes": "full features tree raw target",
                },
                {
                    "name": "E03_tree_interval_residual",
                    "type": "standard",
                    "model": "tree",
                    "target_mode": "interval_residual",
                    "groups": None,
                    "notes": "residual-only comparison, not final submission",
                },
                {
                    "name": "E04_common_residual_reconstruct",
                    "type": "common_residual",
                    "notes": "common + residual reconstruction",
                },
                {
                    "name": "E05_tree_regime_features",
                    "type": "standard",
                    "model": "tree",
                    "target_mode": "raw",
                    "groups": regime_groups,
                    "notes": "tree with regime features",
                },
                {
                    "name": "E07_oof_multi_expert_fusion",
                    "type": "fusion_v31",
                    "notes": "OOF fusion + fixed postprocess",
                },
            ]
            for exp in protocols:
                start_ts = time.time()
                if exp["type"] == "standard":
                    result = self.run_with_groups(
                        split_config=split_config,
                        model_name=exp["model"],
                        feature_groups=exp["groups"],
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        target_mode=exp["target_mode"],
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = result["feature_groups"]
                    postprocess_type = "none"
                elif exp["type"] == "common_residual":
                    result = self.run_common_residual_reconstruction(
                        split_config=split_config,
                        residual_model_name="tree",
                        common_model_name="ridge",
                        feature_groups=None,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = ["base", "lag", "roll", "cross", "common_aggregate"]
                    postprocess_type = "none"
                elif exp["type"] == "fusion_v31":
                    result = self.run_oof_multi_expert_fusion(
                        split_config=split_config,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        expert_specs=[
                            {"name": "et_raw", "kind": "base", "model_name": "tree", "target_mode": "raw", "feature_groups": None},
                            {"name": "hgb_raw", "kind": "base", "model_name": "histgb", "target_mode": "raw", "feature_groups": None},
                            {"name": "et_resid", "kind": "base", "model_name": "tree", "target_mode": "interval_residual", "feature_groups": None},
                            {"name": "ridge_resid", "kind": "base", "model_name": "ridge", "target_mode": "interval_residual", "feature_groups": None},
                            {"name": "common_resid", "kind": "common_residual", "residual_model": "tree", "common_model": "ridge", "feature_groups": None},
                            {"name": "regime_tree", "kind": "base", "model_name": "tree", "target_mode": "raw", "feature_groups": regime_groups},
                        ],
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = ["oof_fusion"]
                    postprocess_type = json.dumps(result["postprocess_params"], ensure_ascii=False)
                else:
                    raise ValueError(f"Unknown V3.1 protocol type: {exp['type']}")
                runtime_sec = time.time() - start_ts
                rows.append({
                    "experiment_id": exp["name"],
                    "feature_set": json.dumps(feature_groups, ensure_ascii=False),
                    "target_type": exp.get("target_mode", "reconstructed"),
                    "model_type": exp.get("model", exp["type"]),
                    "postprocess_type": postprocess_type,
                    "train_corr": train_metrics["corr"],
                    "val_corr": val_metrics["corr"],
                    "train_mse": train_metrics["mse"],
                    "val_mse": val_metrics["mse"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "daily_corr_mean": val_metrics["daily_corr_mean"],
                    "daily_corr_std": val_metrics["daily_corr_std"],
                    "train_val_corr_gap": self._corr_gap(train_metrics, val_metrics),
                    "runtime_sec": float(runtime_sec),
                    "random_seed": 42,
                    "notes": exp["notes"],
                })
            return pd.DataFrame(rows)
        elif suite_name == "v31_quick":
            experiments = [
                {"name": "Q00_baseline_ridge", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy"], "notes": "quick baseline"},
                {"name": "Q01_full_tree_raw", "type": "standard", "model": "tree", "target_mode": "raw", "groups": None, "notes": "quick full tree"},
                {"name": "Q02_common_residual", "type": "common_residual", "notes": "quick common + residual"},
                {"name": "Q03_tree_regime", "type": "standard", "model": "tree", "target_mode": "raw", "groups": ["base", "lag", "roll", "cross", "regime"], "notes": "quick regime tree"},
                {"name": "Q04_small_fusion", "type": "fusion_v31_quick", "notes": "quick OOF fusion"},
                {"name": "Q05_soft_regime_ensemble", "type": "soft_regime", "notes": "quick soft gating ensemble"},
            ]
            rows = []
            for exp in experiments:
                start_ts = time.time()
                if exp["type"] == "standard":
                    result = self.run_with_groups(
                        split_config=split_config,
                        model_name=exp["model"],
                        feature_groups=exp["groups"],
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        target_mode=exp["target_mode"],
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = result["feature_groups"]
                    target_type = exp["target_mode"]
                    model_type = exp["model"]
                    postprocess_type = "none"
                elif exp["type"] == "common_residual":
                    result = self.run_common_residual_reconstruction(
                        split_config=split_config,
                        residual_model_name="tree",
                        common_model_name="ridge",
                        feature_groups=None,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = ["base", "lag", "roll", "cross", "common_aggregate"]
                    target_type = "common_plus_residual"
                    model_type = "common_residual"
                    postprocess_type = "none"
                elif exp["type"] == "fusion_v31_quick":
                    result = self.run_oof_multi_expert_fusion(
                        split_config=split_config,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        expert_specs=[
                            {"name": "et_raw", "kind": "base", "model_name": "tree", "target_mode": "raw", "feature_groups": None},
                            {"name": "ridge_raw", "kind": "base", "model_name": "ridge", "target_mode": "raw", "feature_groups": None},
                            {"name": "common_resid", "kind": "common_residual", "residual_model": "tree", "common_model": "ridge", "feature_groups": None},
                        ],
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = ["oof_fusion_quick"]
                    target_type = "raw"
                    model_type = "oof_fusion"
                    postprocess_type = json.dumps(result["postprocess_params"], ensure_ascii=False)
                elif exp["type"] == "soft_regime":
                    result = self.run_soft_regime_ensemble(
                        split_config=split_config,
                        max_train_days=max_train_days,
                        max_val_days=max_val_days,
                        target_mode="interval_demean",
                    )
                    train_metrics = result["train_metrics"]
                    val_metrics = result["val_metrics"]
                    feature_groups = ["main_tree", "regime_tree", "regime_score"]
                    target_type = "interval_demean"
                    model_type = "tree_soft_regime"
                    postprocess_type = "none"
                else:
                    raise ValueError(f"Unknown V3.1 quick protocol type: {exp['type']}")
                runtime_sec = time.time() - start_ts
                rows.append({
                    "experiment_id": exp["name"],
                    "feature_set": json.dumps(feature_groups, ensure_ascii=False),
                    "target_type": target_type,
                    "model_type": model_type,
                    "postprocess_type": postprocess_type,
                    "train_corr": train_metrics["corr"],
                    "val_corr": val_metrics["corr"],
                    "train_mse": train_metrics["mse"],
                    "val_mse": val_metrics["mse"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "daily_corr_mean": val_metrics["daily_corr_mean"],
                    "daily_corr_std": val_metrics["daily_corr_std"],
                    "train_val_corr_gap": self._corr_gap(train_metrics, val_metrics),
                    "runtime_sec": float(runtime_sec),
                    "random_seed": 42,
                    "notes": exp["notes"],
                })
            return pd.DataFrame(rows)
        elif suite_name == "v31_roll":
            return self.run_rolling_validation_suite(
                split_config=split_config,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=4,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
            )
        elif suite_name == "stage0_roll":
            return self.run_rolling_validation_suite(
                split_config=split_config,
                train_window=8,
                val_window=2,
                step=10,
                max_folds=5,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                embargo=1,
                specs=[
                    {
                        "experiment_id": "B0_ridge_legacy",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["legacy"],
                        "notes": "stage0 legacy ridge baseline",
                    },
                    {
                        "experiment_id": "B1_ridge_core",
                        "type": "standard",
                        "model": "ridge",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 ridge core features",
                    },
                    {
                        "experiment_id": "B2_tree_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree core features",
                    },
                    {
                        "experiment_id": "B3_tree_residual_core",
                        "type": "standard",
                        "model": "tree",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 tree residual core features",
                    },
                    {
                        "experiment_id": "B4_hgb_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "raw",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb core features",
                    },
                    {
                        "experiment_id": "B5_hgb_residual_core",
                        "type": "standard",
                        "model": "histgb",
                        "target_mode": "interval_residual",
                        "groups": ["base", "lag", "roll", "cross"],
                        "notes": "stage0 histgb residual core features",
                    },
                    {
                        "experiment_id": "B6_common_residual",
                        "type": "common_residual",
                        "notes": "stage0 common residual branch",
                    },
                    {
                        "experiment_id": "B7_soft_regime",
                        "type": "soft_regime",
                        "notes": "stage0 current soft regime ensemble",
                    },
                ],
            )
        else:
            raise ValueError(f"Unknown suite: {suite_name}")

        rows = []
        for exp in experiments:
            log.inf(f"Running suite item: {exp['name']}")
            if exp.get("kind") == "fusion":
                fusion = self.run_multi_scale_fusion(
                    split_config=split_config,
                    model_name=exp["model"],
                    target_mode=exp["target_mode"],
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                )
                stacking = fusion["stacking"]
                rows.append({
                    "name": exp["name"],
                    "model": f"{exp['model']}_fusion",
                    "target_mode": exp["target_mode"],
                    "feature_groups": json.dumps(["base", "lag", "roll", "cross", "scale_fusion"], ensure_ascii=False),
                    "train_corr": stacking["train_metrics"]["corr"],
                    "train_r2": stacking["train_metrics"]["r2"],
                    "train_mse": stacking["train_metrics"]["mse"],
                    "val_corr": stacking["val_metrics"]["corr"],
                    "val_r2": stacking["val_metrics"]["r2"],
                    "val_mse": stacking["val_metrics"]["mse"],
                })
                continue
            if exp.get("kind") == "regime":
                result = self.run_regime_residual_fusion(
                    split_config=split_config,
                    model_name=exp["model"],
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                    target_mode=exp["target_mode"],
                )
                rows.append({
                    "name": exp["name"],
                    "model": f"{exp['model']}_regime",
                    "target_mode": exp["target_mode"],
                    "feature_groups": json.dumps(["base", "lag", "roll", "cross", "regime"], ensure_ascii=False),
                    "train_corr": result["train_metrics"]["corr"],
                    "train_r2": result["train_metrics"]["r2"],
                    "train_mse": result["train_metrics"]["mse"],
                    "val_corr": result["val_metrics"]["corr"],
                    "val_r2": result["val_metrics"]["r2"],
                    "val_mse": result["val_metrics"]["mse"],
                })
                continue
            if exp.get("kind") == "soft":
                result = self.run_soft_regime_ensemble(
                    split_config=split_config,
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                    target_mode=exp["target_mode"],
                )
                rows.append({
                    "name": exp["name"],
                    "model": "tree_soft_regime",
                    "target_mode": exp["target_mode"],
                    "feature_groups": json.dumps(["main_tree", "regime_tree", "regime_score"], ensure_ascii=False),
                    "train_corr": result["train_metrics"]["corr"],
                    "train_r2": result["train_metrics"]["r2"],
                    "train_mse": result["train_metrics"]["mse"],
                    "val_corr": result["val_metrics"]["corr"],
                    "val_r2": result["val_metrics"]["r2"],
                    "val_mse": result["val_metrics"]["mse"],
                })
                continue
            if exp.get("kind") == "blend":
                result = self.run_val_blend_search(
                    split_config=split_config,
                    max_train_days=max_train_days,
                    max_val_days=max_val_days,
                    target_mode=exp["target_mode"],
                )
                weights = result["blend_weights"]
                rows.append({
                    "name": exp["name"],
                    "model": "val_blend_search",
                    "target_mode": exp["target_mode"],
                    "feature_groups": json.dumps(
                        {
                            "main_tree": float(weights[0]),
                            "regime_tree": float(weights[1]),
                            "ridge_full": float(weights[2]),
                        },
                        ensure_ascii=False,
                    ),
                    "train_corr": result["train_metrics"]["corr"],
                    "train_r2": result["train_metrics"]["r2"],
                    "train_mse": result["train_metrics"]["mse"],
                    "val_corr": result["val_metrics"]["corr"],
                    "val_r2": result["val_metrics"]["r2"],
                    "val_mse": result["val_metrics"]["mse"],
                })
                continue

            result = self.run_with_groups(
                split_config=split_config,
                model_name=exp["model"],
                feature_groups=exp["groups"],
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=exp["target_mode"],
            )
            rows.append({
                "name": exp["name"],
                "model": exp["model"],
                "target_mode": exp["target_mode"],
                "feature_groups": json.dumps(result["feature_groups"], ensure_ascii=False),
                "train_corr": result["train_metrics"]["corr"],
                "train_r2": result["train_metrics"]["r2"],
                "train_mse": result["train_metrics"]["mse"],
                "val_corr": result["val_metrics"]["corr"],
                "val_r2": result["val_metrics"]["r2"],
                "val_mse": result["val_metrics"]["mse"],
            })
        return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="MEOW experiment runner")
    parser.add_argument("--h5dir", type=str, default=r"D:\code\final\archive")
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "elasticnet", "tree", "gbdt", "histgb", "lgbm", "mlp"])
    parser.add_argument("--target-mode", type=str, default="raw", choices=["raw", "date_demean", "interval_demean", "interval_residual"])
    parser.add_argument("--feature-groups", nargs="*", default=None, help="Feature groups to keep, e.g. base lag roll cross")
    parser.add_argument("--suite", type=str, default=None, choices=["stage0", "stage0_quick", "stage0_roll", "formal_backbone", "ridge_enhance", "restricted_fusion", "train_window_sensitivity", "ofi_audit", "trade_impact_audit", "conditional_momentum_audit", "stage1", "stage2", "ablation", "v2", "v31", "v31_quick", "v31_roll"])
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--val-start", type=int, default=20231101)
    parser.add_argument("--val-end", type=int, default=20231130)
    parser.add_argument("--test-start", type=int, default=20231201)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-val-days", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    split_config = SplitConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    runner = ExperimentRunner(args.h5dir)
    if args.suite:
        df = runner.run_suite(
            split_config=split_config,
            suite_name=args.suite,
            max_train_days=args.max_train_days,
            max_val_days=args.max_val_days,
        )
        log.inf("\n" + df.to_string(index=False))
        if args.output_csv:
            df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
            log.inf(f"Saved suite results to {args.output_csv}")
        return

    result = runner.run_with_groups(
        split_config=split_config,
        model_name=args.model,
        feature_groups=args.feature_groups,
        max_train_days=args.max_train_days,
        max_val_days=args.max_val_days,
        target_mode=args.target_mode,
    )
    if args.output_csv:
        summary = pd.DataFrame([{
            "model": args.model,
            "target_mode": args.target_mode,
            "feature_groups": json.dumps(result["feature_groups"], ensure_ascii=False),
            "train_corr": result["train_metrics"]["corr"],
            "train_r2": result["train_metrics"]["r2"],
            "train_mse": result["train_metrics"]["mse"],
            "val_corr": result["val_metrics"]["corr"],
            "val_r2": result["val_metrics"]["r2"],
            "val_mse": result["val_metrics"]["mse"],
        }])
        summary.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        log.inf(f"Saved result to {args.output_csv}")


if __name__ == "__main__":
    main()
