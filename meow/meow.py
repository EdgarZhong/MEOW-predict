import os
import sys
from pathlib import Path
import pandas as pd

# 把仓库 `src/` 加入搜索路径，但不改老师要求的 `python meow.py` 入口形式。
# 这样 meow 仍然是正式提交壳层，而真正的核心实现统一收口在 `src/`。
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        # 这里保留老师样例中的 `cacheDir` 参数形态，便于外部调用保持兼容。
        # 但正式提交实现不依赖持久化特征缓存，真正的核心逻辑统一走 `src/`。
        self.cacheDir = cacheDir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir, h5dir=h5dir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        log.inf("Running model fitting...")
        x_parts = []
        y_parts = []
        # 逐日构造特征，避免把几个月 raw 一次性堆进内存，也避免跨日滚动串值。
        for date in dates:
            rawData = self.dloader.loadDate(date)
            xday, yday = self.featGenerator.genFeatures(rawData)
            x_parts.append(xday)
            y_parts.append(yday)
        xdf = pd.concat(x_parts)
        ydf = pd.concat(y_parts)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        x_parts = []
        y_parts = []
        for date in dates:
            rawData = self.dloader.loadDate(date)
            xday, yday = self.featGenerator.genFeatures(rawData)
            x_parts.append(xday)
            y_parts.append(yday)
        xdf = pd.concat(x_parts)
        ydf = pd.concat(y_parts)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    # 默认优先读环境变量，便于老师或本地脚本临时覆盖数据路径；
    # 若未设置，则直接回退到仓库根目录下的 `data/`，保证在本仓库里 `python meow.py` 可直接跑。
    default_h5dir = os.environ.get("MEOW_DATA_DIR", str((THIS_DIR.parent / "data").resolve()))
    train_start = int(os.environ.get("MEOW_TRAIN_START", "20230601"))
    train_end = int(os.environ.get("MEOW_TRAIN_END", "20231130"))
    eval_start = int(os.environ.get("MEOW_EVAL_START", "20231201"))
    eval_end = int(os.environ.get("MEOW_EVAL_END", "20231229"))
    engine = MeowEngine(h5dir=default_h5dir, cacheDir=None)
    engine.fit(train_start, train_end)
    engine.eval(eval_start, eval_end)
