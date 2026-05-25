# -*- coding: utf-8 -*-
"""
experiment_runner 训练标签 winsorize 单元测试。

覆盖目标：
  - 默认配置应落在 P0.5 扫描后锁定的 P1 / P99
  - 开关关闭时，不得偷偷改训练目标
  - 打开时，必须按训练集分位对目标两侧裁剪
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from experiment_runner import ExperimentRunner  # noqa: E402


class _DummyLoader:
    """最小占位 loader：测试只关心 runner 初始化，不触发真实数据读取。"""

    def __init__(self, h5dir):
        self.h5dir = h5dir


class _DummyFeatureLoader:
    """最小占位 feature loader：避免测试依赖磁盘特征目录。"""

    def __init__(self, h5dir, feature_dir, loader_cls, feature_dtype="float32"):
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        self.loader_cls = loader_cls
        self.feature_dtype = feature_dtype


class TestTargetWinsorize(unittest.TestCase):
    """锁住 #15 的核心约束，避免后续重构把训练标签口径改丢。"""

    def _make_runner(self, config=None, ridge_alpha=2.0):
        return ExperimentRunner(
            h5dir="dummy-data",
            feature_dir="dummy-features",
            loader_cls=_DummyLoader,
            feature_loader_cls=_DummyFeatureLoader,
            target_winsorize_config=config,
            ridge_alpha=ridge_alpha,
        )

    def test_default_config_matches_agents(self):
        """默认口径必须是开启 + P1 / P99。"""
        runner = self._make_runner()
        cfg = runner.get_target_winsorize_config()
        self.assertTrue(cfg["enabled"])
        self.assertAlmostEqual(cfg["lower_quantile"], 0.01)
        self.assertAlmostEqual(cfg["upper_quantile"], 0.99)
        self.assertAlmostEqual(runner.get_ridge_alpha(), 2.0)

    def test_disabled_keeps_targets_unchanged(self):
        """用户显式关闭时，训练目标应原样返回。"""
        runner = self._make_runner({"enabled": False})
        values = np.asarray([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
        clipped, info = runner._apply_target_winsorize(values)
        self.assertIsNone(info)
        np.testing.assert_allclose(clipped, values)

    def test_enabled_clips_to_train_quantiles(self):
        """打开时必须真的按训练集分位数裁剪，而不是只记录配置不生效。"""
        runner = self._make_runner({
            "enabled": True,
            "lower_quantile": 0.2,
            "upper_quantile": 0.8,
        })
        values = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
        clipped, info = runner._apply_target_winsorize(values)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["lower_bound"], -20.8, places=5)
        self.assertAlmostEqual(info["upper_bound"], 20.8, places=5)
        np.testing.assert_allclose(
            clipped,
            np.asarray([-20.8, -1.0, 0.0, 1.0, 20.8], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_invalid_ridge_alpha_fails_fast(self):
        """ridge alpha 必须是正数，避免扫参时把非法值带进真实训练。"""
        with self.assertRaises(ValueError):
            self._make_runner(ridge_alpha=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
