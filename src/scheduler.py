"""
调度层 — 并发执行 rolling fold 任务

ParallelScheduler:   将 (profile × fold) 任务展平，ProcessPoolExecutor 并发执行
_fold_group_worker:  模块级 worker 函数（multiprocessing 'spawn' 模式可 pickle）

并发策略：
  同一 profile 内相邻 fold 的 train_dates 高度重叠。
  将 fold 列表切成若干连续子组，同一子组内的 fold 分配到同一 worker，
  使 ExperimentRunner._daily_feature_cache 在 worker 进程内跨 fold 复用。

  M5 MacBook Air（4P+6E 核）建议 n_workers=8，留 2 核给系统和磁盘 I/O。
"""

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ================================================================== #
# 元数据结构
# ================================================================== #

@dataclass
class FoldMeta:
    """单个 fold 的轻量元数据（可跨进程 pickle）"""
    profile_name: str
    fold_id: int
    train_dates: tuple
    val_dates: tuple


@dataclass
class FoldGroup:
    """一组连续 fold，作为单个 worker 任务的输入单元"""
    group_id: str              # "{profile_name}_g{n}"，用于日志
    fold_metas: List[FoldMeta]


# ================================================================== #
# Worker 函数（模块级，multiprocessing 'spawn' 可序列化）
# ================================================================== #

def _fold_group_worker(args: tuple) -> List[dict]:
    """
    进程池 worker。

    每个 worker 在进程内创建独立 ExperimentRunner（独立 cache），
    按顺序处理组内各 fold，充分复用相邻 fold 的 _daily_feature_cache。

    args: (h5dir, fold_group, specs, completed_keys)
    返回: list[FoldResult.to_dict()]
    """
    # macOS 'spawn' 模式：确保 src/ 在 sys.path 中
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    from experiment_runner import ExperimentRunner
    from trainer import FoldData, TabularTrainer

    h5dir, fold_group, specs, completed_keys = args

    runner = ExperimentRunner(h5dir)
    results: List[dict] = []

    for meta in fold_group.fold_metas:
        fold_data = FoldData(
            profile_name=meta.profile_name,
            fold_id=meta.fold_id,
            train_dates=meta.train_dates,
            val_dates=meta.val_dates,
        )
        for spec in specs:
            key = (meta.profile_name, meta.fold_id, spec["experiment_id"])
            if key in completed_keys:
                continue
            trainer = TabularTrainer(spec, runner)
            result = trainer.run_fold(fold_data)
            results.append(result.to_dict())

    return results


# ================================================================== #
# 辅助：fold 列表切分
# ================================================================== #

def _build_fold_groups(
    profile_name: str,
    folds: list,
    target_group_size: int = 7,
) -> List[FoldGroup]:
    """
    把单个 profile 的 fold 列表切成若干**连续**子组。

    target_group_size: 每组目标 fold 数。
    相邻 fold 的 train_dates 重叠越多，同一组内的 cache 复用率越高。
    """
    n = len(folds)
    if n == 0:
        return []
    n_groups = max(1, math.ceil(n / target_group_size))
    size = math.ceil(n / n_groups)
    groups = []
    for i in range(0, n, size):
        chunk = folds[i: i + size]
        metas = [
            FoldMeta(
                profile_name=profile_name,
                fold_id=f.fold_id,
                train_dates=f.train_dates,
                val_dates=f.val_dates,
            )
            for f in chunk
        ]
        groups.append(FoldGroup(
            group_id=f"{profile_name}_g{i // size}",
            fold_metas=metas,
        ))
    return groups


# ================================================================== #
# ParallelScheduler
# ================================================================== #

class ParallelScheduler:
    """
    并发执行所有 (profile × fold × spec) 任务。

    - fold 按 profile 分组，相邻 fold 落同一 worker，最大化进程内 cache 复用。
    - 支持断点续跑：启动时读取已有 fold_metrics.csv，跳过已完成 job。
    - 实时落盘：每个 worker 完成后立即 append 到 fold_metrics.csv，
      主进程崩溃不丢已完成结果。
    """

    def __init__(
        self,
        h5dir: str,
        n_workers: int = 8,
    ):
        self.h5dir = h5dir
        self.n_workers = n_workers
        self._fold_metrics_path: Optional[str] = None

    def set_output_path(self, fold_metrics_path: str):
        """设置增量写入路径（resume 和实时落盘共用）"""
        self._fold_metrics_path = fold_metrics_path

    # ---------------------------------------------------------------- #
    # Resume 支持
    # ---------------------------------------------------------------- #

    def _load_completed_keys(self) -> FrozenSet[Tuple]:
        """
        从已有 fold_metrics.csv 读取已完成的 (profile_name, fold_id, experiment_id)。
        只有 status == "ok" 的行才算完成，error 行会被重跑。
        """
        if not self._fold_metrics_path:
            return frozenset()
        if not os.path.exists(self._fold_metrics_path):
            return frozenset()
        try:
            df = pd.read_csv(self._fold_metrics_path, encoding="utf-8-sig")
            if "status" in df.columns:
                df = df[df["status"] == "ok"]
            keys = frozenset(
                zip(
                    df["profile_name"].tolist(),
                    df["fold_id"].astype(int).tolist(),
                    df["experiment_id"].tolist(),
                )
            )
            return keys
        except Exception as e:
            print(f"[Scheduler] 读取 resume 记录失败（将从头开始）: {e}")
            return frozenset()

    # ---------------------------------------------------------------- #
    # 落盘
    # ---------------------------------------------------------------- #

    def _append_results(self, rows: List[dict]) -> None:
        """将一批 FoldResult 增量 append 到 fold_metrics.csv（主进程写，线程安全）"""
        if not self._fold_metrics_path or not rows:
            return
        df = pd.DataFrame(rows)
        write_header = not os.path.exists(self._fold_metrics_path)
        df.to_csv(
            self._fold_metrics_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )

    # ---------------------------------------------------------------- #
    # 主入口
    # ---------------------------------------------------------------- #

    def run(
        self,
        profiles_with_folds: List[Tuple],   # [(RollingProfile, List[RollingFold])]
        specs: List[dict],
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        并发执行所有 fold × spec 任务，实时落盘，支持 resume。

        返回：本次执行后 fold_metrics.csv 的完整 DataFrame
              （含之前已完成 + 本次新完成，供上层 summarize_profile 使用）。
        """
        completed_keys = self._load_completed_keys() if resume else frozenset()
        n_completed = len(completed_keys)

        # 展平所有 profile → FoldGroup 列表
        all_groups: List[FoldGroup] = []
        for profile, folds in profiles_with_folds:
            groups = _build_fold_groups(profile.profile_name, folds)
            all_groups.extend(groups)

        total_jobs = sum(len(g.fold_metas) for g in all_groups) * len(specs)
        pending_jobs = total_jobs - n_completed
        print(
            f"\n[Scheduler] {len(all_groups)} fold-groups，{total_jobs} 个 job"
            f"（{n_completed} 已完成，{pending_jobs} 待执行，n_workers={self.n_workers}）"
        )

        if pending_jobs <= 0:
            print("[Scheduler] 全部 job 已完成，直接读取历史结果。")
            return self._read_all_results()

        worker_args = [
            (self.h5dir, group, specs, completed_keys)
            for group in all_groups
        ]

        t0 = time.time()
        completed_groups = 0
        new_rows: List[dict] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(_fold_group_worker, args): args[1].group_id
                for args in worker_args
            }
            for future in as_completed(futures):
                group_id = futures[future]
                try:
                    rows = future.result()
                    self._append_results(rows)
                    new_rows.extend(rows)
                    completed_groups += 1
                    elapsed = time.time() - t0
                    ok_cnt = sum(1 for r in rows if r.get("status") == "ok")
                    err_cnt = len(rows) - ok_cnt
                    status_str = f"{ok_cnt} ok" + (f", {err_cnt} err" if err_cnt else "")
                    print(
                        f"  [✓] {group_id}  {status_str}"
                        f"  {completed_groups}/{len(all_groups)} groups"
                        f"  {elapsed:.0f}s elapsed"
                    )
                except Exception as e:
                    completed_groups += 1
                    print(f"  [✗] {group_id} worker 进程失败: {e}")

        total_elapsed = time.time() - t0
        print(f"\n[Scheduler] 完成，耗时 {total_elapsed:.1f}s，新增 {len(new_rows)} 条结果。")

        return self._read_all_results()

    def _read_all_results(self) -> pd.DataFrame:
        """读取 fold_metrics.csv 的完整内容（含历史 + 本次新增）"""
        if not self._fold_metrics_path or not os.path.exists(self._fold_metrics_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(self._fold_metrics_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[Scheduler] 读取 fold_metrics.csv 失败: {e}")
            return pd.DataFrame()
