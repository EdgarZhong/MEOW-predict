"""
Orchestrator —— DL run 的发起 + 组装冻结 RunConfig + 两阶段交接 + 落盘（torch-free 编排）

对应规格 §2.1 控制层级最顶层、§7 配置管理。职责**只有组装 + 派发 + 落盘**，不写任何块旋钮
（避免 god-object，规格 §7.1 反模式警告）：

1. 吃一份**已组装冻结**的 ``RunConfig``；按 registry 建 ``adapter`` + ``cartridge_factory``、
   按 ``MeowDataLoader`` 建 ``raw_loader``（测试可注入合成 loader + 参考卡带）、按 ``protocol``
   派生折（并跑 ``assert_folds_causal`` 防泄漏闸）。
2. dump ``config.json``（含 ``config_fingerprint``）到 ``out_dir/<run_id>/``，可复现可审计。
3. 按 ``stage`` 分派：
   - ``SEARCH``  → 交 ``Searcher`` 海选（落 ``trials.csv`` + ``best_config.json``）。
   - ``VALIDATION`` → 定参跑 expanding 少折 × 多种子认证（落 ``fold_metrics.csv`` + ``summary.json``）。

**内存约定**：numpy 参考卡带走 ``gather_all``（一次性物化全部窗口），全量真实数据会爆内存；
故 CLI smoke 默认用 ``--max-symbols`` 抽样降规模。真正的 TCN 卡带按 ``iter_batches`` 流式喂，
不物化、无此限制。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

# —— import 约定：src / config / models 三目录平铺（README / CLAUDE 已记） —— #
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in ("src", "config", "models"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from dl_protocol import DLFold, assert_folds_causal, build_dl_folds, summarize_folds  # noqa: E402
from dl_search import EarlyKillPolicy, Searcher  # noqa: E402
from dl_trainer import SequenceTrainer  # noqa: E402
from registry import build_adapter, build_cartridge  # noqa: E402
from protocol_config import ProfileKind, Stage  # noqa: E402


# ================================================================== #
# Orchestrator
# ================================================================== #

class Orchestrator:
    """对一份冻结的 ``RunConfig`` 发起一次 run。"""

    def __init__(
        self,
        run_config,
        *,
        raw_loader: Optional[Callable[[Sequence[int]], object]] = None,
        h5dir: str = "data",
        adapter=None,
        cartridge_factory: Optional[Callable[[], object]] = None,
    ):
        self.cfg = run_config
        # 依赖注入：默认从 registry + MeowDataLoader 构建；测试可注入合成 loader / 参考卡带。
        self.adapter = adapter if adapter is not None else build_adapter(run_config.adapter)
        self.cartridge_factory = cartridge_factory or (lambda: build_cartridge(run_config.model))
        self.raw_loader = raw_loader or self._default_raw_loader(h5dir)

    @staticmethod
    def _default_raw_loader(h5dir: str) -> Callable[[Sequence[int]], object]:
        from dl import MeowDataLoader   # src/dl.py
        loader = MeowDataLoader(h5dir)
        return lambda dates: loader.loadDates(list(dates))

    # ---- 派生折 ---- #
    def _build_folds(self) -> List[DLFold]:
        p = self.cfg.protocol
        folds = build_dl_folds(
            p.rolling_start, p.rolling_end,
            mode=p.fold_mode(), val_window=p.val_window, step=p.step, embargo=p.embargo,
            train_window=p.train_window, min_train_days=p.min_train_days,
            earlystop_frac=p.earlystop_frac, max_folds=p.effective_max_folds(),
        )
        assert_folds_causal(folds)   # 防泄漏闸：四段时间严格递增、embargo 隔开训练/打分
        return folds

    # ---- 给 SequenceTrainer / FoldResult 的元信息 ---- #
    def _spec(self) -> Dict:
        c = self.cfg
        return {
            "experiment_id": c.run_id,
            "model_type": c.model.kind.value,
            "feature_set": c.adapter.kind.value,
            "target_type": "raw",
            "postprocess_type": "none",
            "notes": f"fp={c.config_fingerprint}",
        }

    # ---- 顶层入口 ---- #
    def run(self) -> dict:
        out_dir = os.path.join(self.cfg.exec_.out_dir, self.cfg.run_id)
        os.makedirs(out_dir, exist_ok=True)
        self.cfg.dump_json(os.path.join(out_dir, "config.json"))

        folds = self._build_folds()
        if not folds:
            summary = {"run_id": self.cfg.run_id, "stage": self.cfg.protocol.stage.value,
                       "status": "no_folds", "note": "日期窗口/min_train_days 不够派生任何折"}
            _dump_json(os.path.join(out_dir, "summary.json"), summary)
            return summary

        if self.cfg.protocol.stage == Stage.SEARCH:
            return self._run_search(folds, out_dir)
        return self._run_validation(folds, out_dir)

    # ---- 海选 ---- #
    def _run_search(self, folds: Sequence[DLFold], out_dir: str) -> dict:
        # search_space 是卡带私有声明：建一个临时卡带读它。
        search_space = dict(getattr(self.cartridge_factory(), "search_space", {}) or {})
        sc = self.cfg.search
        # device 属执行层而非模型层：由 Orchestrator 在派发时注入，避免卡带自己猜。
        defaults = dict(self.cfg.model.hparams)
        defaults.setdefault("device", self.cfg.exec_.device)
        searcher = Searcher(
            spec=self._spec(), adapter=self.adapter, cartridge_factory=self.cartridge_factory,
            raw_loader=self.raw_loader, folds=folds, search_space=search_space,
            n_trials=sc.n_trials, seeds=self.cfg.exec_.seeds,
            defaults=defaults, normalizer_mode=self._normalizer_mode(),
            search_overrides=dict(sc.search_overrides), profile_name="search",
            early_kill=EarlyKillPolicy(enabled=sc.early_kill, warmup_epochs=sc.early_kill_warmup_epochs),
        )
        outcome = searcher.run()

        # 落 trials.csv（hparams 摊平进 hp_ 列，列集取并集）。
        rows = [t.to_row() for t in outcome.trials]
        _dump_csv(os.path.join(out_dir, "trials.csv"), rows)
        # 落 best_config.json（认证档据此冻结，config-lock）。
        best = outcome.best_config_dict()
        _dump_json(os.path.join(out_dir, "best_config.json"), best)

        summary = {
            "run_id": self.cfg.run_id, "stage": "search", "status": "ok",
            "n_trials": len(outcome.trials), "n_folds": len(folds),
            "best": best,
        }
        _dump_json(os.path.join(out_dir, "summary.json"), summary)
        return summary

    # ---- 认证 ---- #
    def _run_validation(self, folds: Sequence[DLFold], out_dir: str) -> dict:
        defaults = dict(self.cfg.model.hparams)
        defaults.setdefault("device", self.cfg.exec_.device)
        seq_len = int(defaults.pop("seq_len", 32))
        cart_hparams = defaults

        fold_rows: List[dict] = []
        corrs: List[float] = []
        for seed in self.cfg.exec_.seeds:
            trainer = SequenceTrainer(
                self._spec(), self.adapter, self.cartridge_factory, self.raw_loader,
                seq_len=seq_len, normalizer_mode=self._normalizer_mode(),
                hparams=cart_hparams, seed=int(seed),
            )
            for fold in folds:
                r = trainer.run_on_dl_fold(fold, profile_name="validation")
                d = r.to_dict()
                d["random_seed"] = int(seed)
                fold_rows.append(d)
                if r.status == "ok" and np.isfinite(r.val_corr):
                    corrs.append(float(r.val_corr))

        _dump_csv(os.path.join(out_dir, "fold_metrics.csv"), fold_rows)
        summ = summarize_folds([{"corr": c} for c in corrs])
        summary = {
            "run_id": self.cfg.run_id, "stage": "validation", "status": "ok",
            "seq_len": seq_len, "hparams": cart_hparams,
            "n_folds": len(folds), "n_seeds": len(self.cfg.exec_.seeds),
            "val_corr": summ,   # mean / std / min(最坏折) / max / positive_rate
        }
        _dump_json(os.path.join(out_dir, "summary.json"), summary)
        return summary

    def _normalizer_mode(self) -> str:
        # RAW_CHANNELS 已在 adapter 做语义归一，但脊柱仍跑 zscore 统计白化（职责分明，规格 §2.3）；
        # 模式可由 model.hparams["normalizer_mode"] 覆盖（如 identity）。
        return str(self.cfg.model.hparams.get("normalizer_mode", "zscore"))


# ================================================================== #
# 落盘小工具（不引 pandas，标准库即可）
# ================================================================== #

def _dump_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _dump_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        # 仍写一个空文件占位，便于 resume/审计看到 run 跑过。
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames: List[str] = []
    for r in rows:                       # 列集取并集，保插入序
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ================================================================== #
# CLI —— 组装一份 RunConfig 并 run（torch-free smoke：参考卡带 + 真实/抽样数据）
# ================================================================== #

def _build_run_config(args):
    from model_config import ModelKind, ModelConfig
    from adapter_config import AdapterKind, AdapterConfig
    from protocol_config import ProtocolConfig
    from search_config import SearchConfig
    from exec_config import ExecConfig
    from run_config import assemble_run_config

    stage = Stage(args.stage)
    profile = ProfileKind.SINGLE_SPLIT if stage == Stage.SEARCH else ProfileKind.EXPANDING
    max_folds = 1 if stage == Stage.SEARCH else args.max_folds

    model = ModelConfig(ModelKind(args.model), hparams=_parse_hparams(args.hparams))
    adapter_kind = AdapterKind(args.adapter)
    adapter = (AdapterConfig(adapter_kind, columns=tuple(args.columns.split(",")))
               if adapter_kind == AdapterKind.IDENTITY and args.columns
               else AdapterConfig(adapter_kind))
    protocol = ProtocolConfig(
        stage, profile, args.start, args.end,
        val_window=args.val_window, step=args.step, min_train_days=args.min_train_days,
        max_folds=max_folds,
    )
    search = SearchConfig(n_trials=args.trials)
    exec_ = ExecConfig(seeds=tuple(int(s) for s in args.seeds.split(",")), out_dir=args.out_dir)
    return assemble_run_config(args.run_id, model, adapter, protocol, search, exec_)


def _parse_hparams(s: str) -> dict:
    """解析 ``k=v,k2=v2``（v 尝试 int→float→str）。"""
    out: dict = {}
    if not s:
        return out
    for kv in s.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                out[k.strip()] = cast(v); break
            except ValueError:
                continue
        else:
            out[k.strip()] = v
    return out


def _wrap_max_symbols(h5dir: str, max_symbols: int):
    """包一层 raw_loader：抽前 N 个 symbol，控参考卡带 gather_all 的内存（CLI smoke 用）。"""
    from dl import MeowDataLoader
    loader = MeowDataLoader(h5dir)

    def _load(dates):
        df = loader.loadDates(list(dates))
        keep = sorted(df["symbol"].unique())[:max_symbols]
        return df[df["symbol"].isin(keep)].copy()
    return _load


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL Orchestrator（torch-free smoke）")
    ap.add_argument("--run-id", default="20260531_search_refpool_smoke_v1")
    ap.add_argument("--stage", default="search", choices=["search", "validation"])
    ap.add_argument("--model", default="reference_pool",
                    help="reference_zero / reference_last / reference_pool（tcn/lstm 等 4060 接卡后）")
    ap.add_argument("--adapter", default="raw_channels",
                    help="raw_channels / feature_433 / identity")
    ap.add_argument("--columns", default="", help="identity adapter 的列（逗号分隔）")
    ap.add_argument("--start", type=int, default=20230601)
    ap.add_argument("--end", type=int, default=20230731)
    ap.add_argument("--val-window", type=int, default=5)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=20)
    ap.add_argument("--max-folds", type=int, default=3, help="仅 validation 用")
    ap.add_argument("--trials", type=int, default=4, help="仅 search 用")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--hparams", default="", help="k=v,k2=v2（如 seq_len=32,hidden_size=64）")
    ap.add_argument("--out-dir", default="results/dl")
    ap.add_argument("--h5dir", default="data")
    ap.add_argument("--max-symbols", type=int, default=20,
                    help="抽前 N 个 symbol 控内存（参考卡带 gather_all 用）；<=0 = 不抽样")
    args = ap.parse_args(argv)

    rc = _build_run_config(args)
    raw_loader = None
    if args.max_symbols and args.max_symbols > 0:
        raw_loader = _wrap_max_symbols(args.h5dir, args.max_symbols)
    orch = Orchestrator(rc, raw_loader=raw_loader, h5dir=args.h5dir)
    summary = orch.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
