# -*- coding: utf-8 -*-
"""
DL 基础设施单元测试 —— RawChannelAdapter + HPO 采样器/早杀钩子 + Searcher + Orchestrator

覆盖（全程 torch-free、CPU 可跑）：
1. RawChannelAdapter：通道数 59 / 行序契约 / 价相对 mid / 量 log1p / midpx 日内对数收益因果
   （首步及跨票段首=0）/ 缺列报错 / 真实 h5 接通（无数据则 skip）。
2. 采样器 sample_hparams：choice/int/uniform 取值合法 / overrides 收窄 / 空空间 / 可复现。
3. EarlyKillPolicy：默认关恒不杀 / 开启后杀明显落后 trial、留好 trial、尊重 warmup。
4. Searcher：随机搜 + 排名 + 可复现 + overrides 收窄 seq_len + best 优于劣者。
5. Orchestrator：SEARCH 端到端（落 config/trials/best_config/summary）/ VALIDATION 端到端
   （落 fold_metrics/summary）/ config.json 带 fingerprint / 无折时 no_folds。
"""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in ("src", "config", "models"):
    sys.path.insert(0, os.path.join(REPO_ROOT, _sub))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl_models import (  # noqa: E402
    RawChannelAdapter, ReferencePoolCartridge, IdentityAdapter, STRUCTURE_SEARCH_SPACE,
    TCNCartridge, GRUCartridge,
    _PRICE_REL_COLS, _LOG_VOLUME_COLS, _RATIO_COLS,
)
from dl_search import EarlyKillPolicy, Searcher, sample_hparams  # noqa: E402
from dl_protocol import build_dl_folds  # noqa: E402
from sequence_dataset import SequenceDataset, Normalizer, build_sequence_arrays  # noqa: E402
from tradingcalendar import Calendar  # noqa: E402


# ------------------------------------------------------------------ #
# 合成数据工具
# ------------------------------------------------------------------ #

def make_raw_micro_oneday(symbol_ids=(0, 1, 2), n_interval=12, seed=0):
    """造单日原始微结构 raw（含 RawChannelAdapter 需要的全部列）。"""
    rng = np.random.default_rng(seed)
    cols = (["date", "symbol", "interval", "fret12", "midpx"]
            + list(_PRICE_REL_COLS) + list(_LOG_VOLUME_COLS) + list(_RATIO_COLS))
    rows = []
    for s in symbol_ids:
        base = 100.0 + 10.0 * s
        mids = base + np.cumsum(rng.normal(0.0, 0.1, size=n_interval))   # 随机游走中间价
        for it in range(n_interval):
            mid = float(mids[it])
            rec = {"date": 20230601, "symbol": int(s), "interval": int(it),
                   "fret12": float(rng.normal() * 0.01), "midpx": mid}
            for c in _PRICE_REL_COLS:
                rec[c] = float(mid * (1.0 + rng.normal(0.0, 0.001)))     # 价在 mid 附近
            for c in _LOG_VOLUME_COLS:
                rec[c] = float(abs(rng.normal(0.0, 1000.0)))             # 非负量
            for c in _RATIO_COLS:
                rec[c] = float(rng.uniform(0.0, 1.0))
            rows.append(rec)
    return pd.DataFrame(rows, columns=cols)


def make_synth_seq(dates, n_symbols=5, n_interval=20, label="current", seed=0):
    """通道 c0/c1 噪声；current: y=c0+小噪声（同期信号，参考模型应抓到）。dates 用真实交易日。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for s in range(n_symbols):
            c0 = rng.normal(size=n_interval)
            c1 = rng.normal(size=n_interval)
            y = c0 + 0.1 * rng.normal(size=n_interval) if label == "current" else rng.normal(size=n_interval)
            for it in range(n_interval):
                rows.append((int(d), s, it, float(c0[it]), float(c1[it]), float(y[it])))
    return pd.DataFrame(rows, columns=["date", "symbol", "interval", "c0", "c1", "fret12"])


def make_loader(df):
    return lambda dates: df[df["date"].isin([int(x) for x in dates])]


class _IntCalendar:
    """把一串整数当交易日的最小日历（Searcher 直测用，避免依赖真实日历）。"""
    def __init__(self, days):
        self._days = sorted(int(d) for d in days)

    def range(self, start, end):
        return [d for d in self._days if start <= d <= end]


# ------------------------------------------------------------------ #
# 1) RawChannelAdapter
# ------------------------------------------------------------------ #

class TestRawChannelAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = RawChannelAdapter()
        self.df = make_raw_micro_oneday(symbol_ids=(0, 1, 2), n_interval=10, seed=1)

    def test_channel_count_is_59(self):
        # 1(logret) + 26(价rel) + 26(量log) + 6(比率) = 59
        self.assertEqual(len(self.adapter.channels), 59)
        self.assertEqual(self.adapter.channels[0], "midpx_logret")

    def test_build_shape_and_finite(self):
        feats = self.adapter.build(self.df)
        self.assertEqual(feats.shape, (len(self.df), 59))
        self.assertTrue(np.isfinite(feats).all())
        self.assertEqual(feats.dtype, np.float32)

    def test_midpx_logret_causal(self):
        feats = self.adapter.build(self.df)
        day = self.df.sort_values(["symbol", "interval"], kind="mergesort").reset_index(drop=True)
        # 每个 symbol 段首（含全局首行）logret 必为 0（不跨票、首步无前值）
        firsts = day.groupby("symbol").head(1).index.tolist()
        np.testing.assert_allclose(feats[firsts, 0], 0.0, atol=1e-7)
        # 段内：logret[t] == log(mid[t]) - log(mid[t-1])
        s0 = day[day["symbol"] == 0]
        mid = s0["midpx"].to_numpy()
        expect = np.diff(np.log(mid))
        got = feats[s0.index.to_numpy()[1:], 0]
        np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-6)

    def test_price_relativized_and_volume_logged(self):
        feats = self.adapter.build(self.df)
        day = self.df.sort_values(["symbol", "interval"], kind="mergesort").reset_index(drop=True)
        mid = day["midpx"].to_numpy()
        # 价类首通道 = bid0_rel 应等于 bid0/mid - 1（channels: 0=logret, 1.. = _PRICE_REL_COLS 序）
        bid0_idx = 1 + _PRICE_REL_COLS.index("bid0")
        expect_rel = day["bid0"].to_numpy() / mid - 1.0
        np.testing.assert_allclose(feats[:, bid0_idx], expect_rel, rtol=1e-4, atol=1e-6)
        # 量类：nTradeBuy_log == log1p(nTradeBuy)
        log_start = 1 + len(_PRICE_REL_COLS)
        ntb_idx = log_start + _LOG_VOLUME_COLS.index("nTradeBuy")
        np.testing.assert_allclose(feats[:, ntb_idx], np.log1p(day["nTradeBuy"].to_numpy()),
                                   rtol=1e-4, atol=1e-6)

    def test_zero_price_maps_to_neutral_zero(self):
        df = self.df.copy()
        df.loc[df.index[:3], "tradeBuyHigh"] = 0.0      # 无成交：价=0 → 相对通道置 0（非 -1）
        feats = self.adapter.build(df)
        day = df.sort_values(["symbol", "interval"], kind="mergesort").reset_index(drop=True)
        idx = 1 + _PRICE_REL_COLS.index("tradeBuyHigh")
        zero_rows = day.index[day["tradeBuyHigh"] == 0.0].to_numpy()
        np.testing.assert_allclose(feats[zero_rows, idx], 0.0, atol=1e-7)

    def test_missing_column_raises(self):
        df = self.df.drop(columns=["bid0"])
        with self.assertRaises(KeyError):
            self.adapter.build(df)

    H5 = os.path.join(REPO_ROOT, "data", "20230601.h5")

    @unittest.skipUnless(os.path.exists(H5), "缺真实 h5 数据，跳过")
    def test_real_h5_builds(self):
        raw = pd.read_hdf(self.H5)
        keep = sorted(raw["symbol"].unique())[:5]
        raw = raw[raw["symbol"].isin(keep)].copy()
        raw["date"] = 20230601
        feats = RawChannelAdapter().build(raw)
        self.assertEqual(feats.shape, (len(raw), 59))
        self.assertTrue(np.isfinite(feats).all())


# ------------------------------------------------------------------ #
# 2) 采样器
# ------------------------------------------------------------------ #

class TestSampler(unittest.TestCase):
    def test_choice_int_uniform_in_range(self):
        rng = np.random.default_rng(0)
        space = {"a": {"type": "choice", "values": [16, 32, 64]},
                 "b": {"type": "int", "low": 1, "high": 4},
                 "c": {"type": "uniform", "low": 0.0, "high": 1.0}}
        for _ in range(200):
            hp = sample_hparams(space, rng)
            self.assertIn(hp["a"], [16, 32, 64])
            self.assertTrue(1 <= hp["b"] <= 4 and isinstance(hp["b"], int))
            self.assertTrue(0.0 <= hp["c"] <= 1.0)

    def test_overrides_narrow_knob(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            hp = sample_hparams(STRUCTURE_SEARCH_SPACE, rng,
                                overrides={"seq_len": {"type": "choice", "values": [8]}})
            self.assertEqual(hp["seq_len"], 8)

    def test_empty_space(self):
        self.assertEqual(sample_hparams({}, np.random.default_rng(0)), {})

    def test_reproducible(self):
        a = sample_hparams(STRUCTURE_SEARCH_SPACE, np.random.default_rng(7))
        b = sample_hparams(STRUCTURE_SEARCH_SPACE, np.random.default_rng(7))
        self.assertEqual(a, b)


# ------------------------------------------------------------------ #
# 3) 早杀钩子
# ------------------------------------------------------------------ #

class TestEarlyKill(unittest.TestCase):
    def test_disabled_never_kills(self):
        p = EarlyKillPolicy(enabled=False)
        self.assertFalse(p.should_kill([100.0, 100.0, 100.0, 100.0], best_so_far=0.001))

    def test_enabled_kills_hopeless(self):
        p = EarlyKillPolicy(enabled=True, warmup_epochs=2, rel_margin=0.5)
        # 落后 best(=0.1) 远超 50%（曲线最优 1.0）→ 杀
        self.assertTrue(p.should_kill([2.0, 1.5, 1.0, 1.0], best_so_far=0.1))

    def test_enabled_keeps_good(self):
        p = EarlyKillPolicy(enabled=True, warmup_epochs=2, rel_margin=0.5)
        # 与 best(=0.1) 同量级 → 不杀
        self.assertFalse(p.should_kill([0.2, 0.12, 0.11], best_so_far=0.1))

    def test_respects_warmup(self):
        p = EarlyKillPolicy(enabled=True, warmup_epochs=5, rel_margin=0.5)
        # 还没到 warmup epoch → 不杀
        self.assertFalse(p.should_kill([5.0, 5.0], best_so_far=0.1))


# ------------------------------------------------------------------ #
# 4) Searcher
# ------------------------------------------------------------------ #

class TestSearcher(unittest.TestCase):
    def _make(self, n_trials=6, sampling_seed=123):
        dates = list(range(1, 11))
        df = make_synth_seq(dates, n_symbols=6, n_interval=20, label="current", seed=3)
        folds = build_dl_folds(1, 10, mode="expanding", val_window=2, step=3,
                               min_train_days=4, earlystop_frac=0.25, max_folds=1,
                               calendar=_IntCalendar(dates))
        self.assertGreater(len(folds), 0)
        return Searcher(
            spec={"experiment_id": "search_test"}, adapter=IdentityAdapter(["c0", "c1"]),
            cartridge_factory=ReferencePoolCartridge, raw_loader=make_loader(df),
            folds=folds, search_space=STRUCTURE_SEARCH_SPACE, n_trials=n_trials,
            seeds=(42,), defaults={}, normalizer_mode="zscore",
            search_overrides={"seq_len": {"type": "choice", "values": [4, 6, 8]}},
            sampling_seed=sampling_seed,
        )

    def test_runs_and_ranks(self):
        outcome = self._make().run()
        self.assertEqual(len(outcome.trials), 6)
        self.assertIsNotNone(outcome.best)
        # best 必是 ok 且有评估的 trial 里 val_corr_mean 最大者
        ok = [t for t in outcome.trials if t.status == "ok" and t.n_evals > 0]
        self.assertAlmostEqual(outcome.best.val_corr_mean, max(t.val_corr_mean for t in ok))
        # seq_len 被 overrides 限制在 {4,6,8}
        for t in outcome.trials:
            self.assertIn(t.seq_len, [4, 6, 8])

    def test_reproducible(self):
        o1 = self._make(sampling_seed=999).run()
        o2 = self._make(sampling_seed=999).run()
        self.assertEqual([t.seq_len for t in o1.trials], [t.seq_len for t in o2.trials])
        np.testing.assert_allclose([t.val_corr_mean for t in o1.trials],
                                   [t.val_corr_mean for t in o2.trials])

    def test_best_config_dict(self):
        bc = self._make().run().best_config_dict()
        self.assertTrue(bc["found"])
        self.assertIn("seq_len", bc)
        self.assertIn("hparams", bc)


# ------------------------------------------------------------------ #
# 4.5) TCN 卡带（需要 torch，但只测最小接线）
# ------------------------------------------------------------------ #

try:
    import torch  # noqa: E402
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TestTCNCartridge(unittest.TestCase):
    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 TCN 接线测试")
    def test_required_adapter_is_raw_channels(self):
        from adapter_config import AdapterKind
        self.assertEqual(TCNCartridge.required_adapter, AdapterKind.RAW_CHANNELS)

    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 TCN 接线测试")
    def test_fit_and_predict_on_cpu(self):
        # 用多日原始微结构假数据走真实 RawChannelAdapter → SequenceDataset 链路，
        # 只验证“接线可训、可预测、输出有限值”，不在单测里追求分数。
        day1 = make_raw_micro_oneday(symbol_ids=(0, 1, 2), n_interval=12, seed=11).copy()
        day1["date"] = 20230601
        day1["fret12"] = (
            day1.groupby("symbol")["midpx"]
            .pct_change()
            .fillna(0.0)
            .astype(float)
        )
        day2 = make_raw_micro_oneday(symbol_ids=(0, 1, 2), n_interval=12, seed=12).copy()
        day2["date"] = 20230602
        day2["fret12"] = (
            day2.groupby("symbol")["midpx"]
            .pct_change()
            .fillna(0.0)
            .astype(float)
        )
        raw = pd.concat([day1, day2], ignore_index=True)

        arrays = build_sequence_arrays(raw, RawChannelAdapter())
        normalizer = Normalizer("zscore").fit(arrays.features)
        ds = SequenceDataset(arrays, seq_len=4, normalizer=normalizer)

        cart = TCNCartridge()
        record = cart.fit(
            ds,
            ds,
            hparams={
                "device": "cpu",
                "hidden_size": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "batch_size": 16,
                "max_epochs": 2,
                "patience": 2,
            },
            seed=42,
        )
        pred = cart.predict(ds)

        self.assertGreaterEqual(record.best_epoch, 1)
        self.assertEqual(record.extra["kind"], "tcn")
        self.assertEqual(pred.shape[0], len(ds))
        self.assertTrue(np.isfinite(pred).all())

    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 TCN 接线测试")
    def test_fit_does_not_materialize_all_windows(self):
        # 回归测试：fit 不能再调 gather_all() 把整份 X[B,L,C] 一次性搬进内存。
        day = make_raw_micro_oneday(symbol_ids=(0, 1), n_interval=10, seed=21).copy()
        day["date"] = 20230601
        day["fret12"] = day.groupby("symbol")["midpx"].pct_change().fillna(0.0).astype(float)
        arrays = build_sequence_arrays(day, RawChannelAdapter())
        normalizer = Normalizer("zscore").fit(arrays.features)
        ds = SequenceDataset(arrays, seq_len=4, normalizer=normalizer)

        def _boom():
            raise AssertionError("fit 不应调用 gather_all() 物化整窗")

        ds.gather_all = _boom  # type: ignore[method-assign]

        cart = TCNCartridge()
        record = cart.fit(
            ds,
            ds,
            hparams={
                "device": "cpu",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 8,
                "max_epochs": 1,
                "patience": 1,
            },
            seed=7,
        )
        self.assertGreaterEqual(record.best_epoch, 1)


# ------------------------------------------------------------------ #
# 5) Orchestrator 端到端
# ------------------------------------------------------------------ #

class TestOrchestrator(unittest.TestCase):
    def _blocks(self):
        from model_config import ModelKind, ModelConfig
        from adapter_config import AdapterKind, AdapterConfig
        from protocol_config import Stage, ProfileKind, ProtocolConfig
        from search_config import SearchConfig
        from exec_config import ExecConfig
        from run_config import assemble_run_config
        return (ModelKind, ModelConfig, AdapterKind, AdapterConfig, Stage,
                ProfileKind, ProtocolConfig, SearchConfig, ExecConfig, assemble_run_config)

    def _real_dates(self):
        # 用真实交易日（Orchestrator 内部用真实 Calendar 派生折）。
        return Calendar().range(20230601, 20230630)

    def _orch(self, rc, df):
        from run_dl import Orchestrator
        return Orchestrator(rc, raw_loader=make_loader(df),
                            adapter=IdentityAdapter(["c0", "c1"]),
                            cartridge_factory=ReferencePoolCartridge)

    def test_search_end_to_end(self):
        (MK, MC, AK, AC, St, PK, PC, SC, EC, assemble) = self._blocks()
        dates = self._real_dates()
        df = make_synth_seq(dates, n_symbols=6, n_interval=20, label="current", seed=5)
        with tempfile.TemporaryDirectory() as td:
            rc = assemble(
                "20260531_search_refpool_test_v1",
                MC(MK.REFERENCE_POOL, hparams={"seq_len": 6}),
                AC(AK.IDENTITY, columns=("c0", "c1")),
                PC(St.SEARCH, PK.SINGLE_SPLIT, dates[0], dates[-1],
                   val_window=2, step=3, min_train_days=8),
                SC(n_trials=4, search_overrides={"seq_len": {"type": "choice", "values": [4, 6]}}),
                EC(seeds=(42,), out_dir=td),
            )
            summary = self._orch(rc, df).run()
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["stage"], "search")
            self.assertTrue(summary["best"]["found"])
            run_dir = os.path.join(td, rc.run_id)
            for fn in ("config.json", "trials.csv", "best_config.json", "summary.json"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, fn)), fn)
            # config.json 带 fingerprint
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(len(cfg["config_fingerprint"]), 16)

    def test_validation_end_to_end(self):
        (MK, MC, AK, AC, St, PK, PC, SC, EC, assemble) = self._blocks()
        dates = self._real_dates()
        df = make_synth_seq(dates, n_symbols=6, n_interval=20, label="current", seed=6)
        with tempfile.TemporaryDirectory() as td:
            rc = assemble(
                "20260531_valid_refpool_test_v1",
                MC(MK.REFERENCE_POOL, hparams={"seq_len": 6}),
                AC(AK.IDENTITY, columns=("c0", "c1")),
                PC(St.VALIDATION, PK.EXPANDING, dates[0], dates[-1],
                   val_window=2, step=3, min_train_days=8, max_folds=2),
                SC(n_trials=1), EC(seeds=(42, 7), out_dir=td),
            )
            summary = self._orch(rc, df).run()
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["stage"], "validation")
            self.assertIn("val_corr", summary)
            self.assertGreater(summary["val_corr"]["mean"], 0.3)   # 同期信号应被池化线性抓到
            run_dir = os.path.join(td, rc.run_id)
            for fn in ("config.json", "fold_metrics.csv", "summary.json"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, fn)), fn)

    def test_no_folds(self):
        (MK, MC, AK, AC, St, PK, PC, SC, EC, assemble) = self._blocks()
        df = make_synth_seq(self._real_dates(), n_symbols=3, n_interval=10)
        with tempfile.TemporaryDirectory() as td:
            rc = assemble(
                "20260531_search_nofold_v1",
                MC(MK.REFERENCE_POOL, hparams={"seq_len": 5}),
                AC(AK.IDENTITY, columns=("c0", "c1")),
                PC(St.SEARCH, PK.SINGLE_SPLIT, 20230601, 20230605, min_train_days=40),
                SC(n_trials=2), EC(seeds=(42,), out_dir=td),
            )
            summary = self._orch(rc, df).run()
            self.assertEqual(summary["status"], "no_folds")


# ------------------------------------------------------------------ #
# 4.6) GRU 卡带（需要 torch，但只测最小接线）
# ------------------------------------------------------------------ #

class TestGRUCartridge(unittest.TestCase):
    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 GRU 接线测试")
    def test_required_adapter_is_feature_433(self):
        from adapter_config import AdapterKind
        self.assertEqual(GRUCartridge.required_adapter, AdapterKind.FEATURE_433)

    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 GRU 接线测试")
    def test_fit_and_predict_on_cpu(self):
        # 用合成双通道数据走 IdentityAdapter → SequenceDataset 链路，
        # 测卡带接线可训、可预测、输出有限值（不测 FeatureAdapter，后者需真实 h5）。
        dates = [20230601, 20230602]
        df = make_synth_seq(dates, n_symbols=4, n_interval=15, label="current", seed=99)
        arrays = build_sequence_arrays(df, IdentityAdapter(["c0", "c1"]))
        normalizer = Normalizer("zscore").fit(arrays.features)
        ds = SequenceDataset(arrays, seq_len=4, normalizer=normalizer)

        cart = GRUCartridge()
        record = cart.fit(
            ds, ds,
            hparams={
                "device": "cpu",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 16,
                "max_epochs": 2,
                "patience": 2,
            },
            seed=42,
        )
        pred = cart.predict(ds)

        self.assertGreaterEqual(record.best_epoch, 1)
        self.assertEqual(record.extra["kind"], "gru")
        self.assertEqual(pred.shape[0], len(ds))
        self.assertTrue(np.isfinite(pred).all())

    @unittest.skipUnless(_TORCH_AVAILABLE, "未安装 torch，跳过 GRU 接线测试")
    def test_fit_does_not_materialize_all_windows(self):
        # GRU fit 同样走 iter_batches 流式，不得调 gather_all()。
        dates = [20230601]
        df = make_synth_seq(dates, n_symbols=3, n_interval=10, label="current", seed=55)
        arrays = build_sequence_arrays(df, IdentityAdapter(["c0", "c1"]))
        normalizer = Normalizer("zscore").fit(arrays.features)
        ds = SequenceDataset(arrays, seq_len=4, normalizer=normalizer)

        def _boom():
            raise AssertionError("fit 不应调用 gather_all() 物化整窗")

        ds.gather_all = _boom  # type: ignore[method-assign]

        cart = GRUCartridge()
        record = cart.fit(
            ds, ds,
            hparams={
                "device": "cpu",
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 8,
                "max_epochs": 1,
                "patience": 1,
            },
            seed=7,
        )
        self.assertGreaterEqual(record.best_epoch, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
