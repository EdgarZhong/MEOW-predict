# -*- coding: utf-8 -*-
"""
正式提交通道桥接层单元测试。

覆盖目标：
1. 从 raw 现算正式提交特征时，列集合要和 spec 解析结果一致。
2. 训练 / 预测桥接层可以在不依赖磁盘特征缓存的情况下跑通。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from feature_registry import META_COLS, TARGET_COL, _make_schema_probe_raw  # noqa: E402
from submission_pipeline import (  # noqa: E402
    DEFAULT_SUBMISSION_GROUPS,
    SubmissionFeaturePipeline,
    SubmissionModelPipeline,
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
        model_pipeline = SubmissionModelPipeline(h5dir="dummy-data")
        model_pipeline.fit(xdf, ydf)
        pred = model_pipeline.predict(xdf)
        self.assertEqual(pred.shape[0], xdf.shape[0])
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
