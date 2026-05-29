# -*- coding: utf-8 -*-
"""
正式提交通道桥接层单元测试。

覆盖目标：
1. 从 raw 现算正式提交特征时，列集合要和 spec 解析结果一致。
2. 训练 / 预测桥接层可以在不依赖磁盘特征缓存的情况下跑通。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from feature_registry import META_COLS, TARGET_COL, _make_schema_probe_raw  # noqa: E402
from submission_pipeline import (  # noqa: E402
    DEFAULT_SUBMISSION_GROUPS,
    SubmissionFeaturePipeline,
    SubmissionMemberSpec,
    SubmissionModelPipeline,
    SubmissionSpec,
)


class TestSubmissionFeaturePipeline(unittest.TestCase):
    """保证正式提交特征桥接层不会和 registry/group 解析漂移。"""

    def test_feature_names_match_built_columns(self):
        raw = _make_schema_probe_raw()
        pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
        xdf, ydf = pipeline.build_feature_frames(raw)
        self.assertEqual(list(xdf.columns[:3]), META_COLS)
        self.assertEqual(list(ydf.columns), META_COLS + [TARGET_COL])
        self.assertEqual(
            list(xdf.columns[3:]),
            pipeline.feature_names(),
        )


class TestSubmissionModelPipeline(unittest.TestCase):
    """保证正式提交训练/推理桥接层可以独立跑通。"""

    def test_fit_predict_smoke(self):
        raw = _make_schema_probe_raw()
        feature_pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
        xdf, ydf = feature_pipeline.build_feature_frames(raw)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir)
            model_pipeline.fit(xdf, ydf)
            pred = model_pipeline.predict(xdf)
            self.assertEqual(pred.shape[0], xdf.shape[0])
            self.assertTrue(np.isfinite(pred).all())

    def _blend_fixture(self, blend_mode):
        """构造一个跨两天、两成员的小样本，返回 (frame, member_preds, pipeline)。"""
        frame = _make_schema_probe_raw().iloc[:4].copy()
        frame.loc[:, "date"] = np.array([20230601, 20230601, 20230602, 20230602], dtype=int)
        frame = frame.reset_index(drop=True)
        member_a = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
        member_b = np.array([2.0, 4.0, 30.0, 10.0], dtype=np.float32)
        spec = SubmissionSpec(
            members=(
                SubmissionMemberSpec("member_a", ("legacy",), "ridge"),
                SubmissionMemberSpec("member_b", ("legacy",), "ridge"),
            ),
            blend_mode=blend_mode,
        )
        return frame, {"member_a": member_a, "member_b": member_b}, spec

    def test_blend_raw_mean_is_default(self):
        """
        提交默认融合 = raw_mean：直接等权平均、输出留在原始量纲（保 MSE/R²）。

        member_a=[1,2,10,20], member_b=[2,4,30,10] → 等权平均 [1.5,3,20,15]。
        同时断言默认 spec 的 blend_mode 就是 raw_mean，防止有人误改回 zscore。
        """
        self.assertEqual(SubmissionSpec().blend_mode, "raw_mean")
        frame, member_preds, spec = self._blend_fixture("raw_mean")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir, spec=spec)
            pred = model_pipeline._blend_member_predictions(frame, member_preds)
        expected = np.array([1.5, 3.0, 20.0, 15.0], dtype=np.float32)
        np.testing.assert_allclose(pred, expected, atol=1e-6)

    def test_predict_blends_per_day_zscore(self):
        """
        保证 per_day_zscore_mean 模式是“按天截面标准化后再平均”，而非全局 pooled zscore。

        这个模式现在只作诊断对照，但仍卡住口径里最容易漂移的点：
        - 标准化必须用“当天数据自身”
        - 不能偷用训练期统计量、不能把所有日期混在一起 pooled zscore
        """
        frame, member_preds, spec = self._blend_fixture("per_day_zscore_mean")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir, spec=spec)
            pred = model_pipeline._blend_member_predictions(frame, member_preds)
        expected = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(pred, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
