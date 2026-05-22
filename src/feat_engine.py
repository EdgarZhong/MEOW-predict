import numpy as np
import pandas as pd


EPS = 1e-8

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

