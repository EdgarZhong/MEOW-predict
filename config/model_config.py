"""
模型配置块 —— ModelKind(枚举) + ModelConfig(frozen)

按规格 §7.3/§7.4「三层枚举拆分」与「每块单独成文件 + 枚举进块文件顶部」：
- 文件顶部放**合法词表**（``ModelKind`` 枚举，封闭选项集，集中维护）；
- 下方放本块的 **schema**（``ModelConfig`` frozen dataclass，"本次 run 选了哪个值"）。

注意区分（规格 §6/§7.3）：
- ``search_space``（某模型私有的超参声明 + 范围/分布）是**卡带私有**，跟 ``ModelCartridge``
  走，**不在这里**（不属于全局枚举、不进 config/）；
- 这里的 ``hparams`` 是"本次 run 钉死的固定超参"（非搜索时直接用，或搜索的默认基底）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ModelKind(Enum):
    """模型卡带合法词表（封闭集；新增卡带先在这里登记一个枚举值）。"""
    REFERENCE_ZERO = "reference_zero"   # numpy 参考模型：恒 0，防泄漏探测器（应打低分）
    REFERENCE_LAST = "reference_last"   # numpy 参考模型：末步通道线性，防泄漏探测器
    LSTM = "lstm"                       # D1 卡带（特征当序列），等 4060 + torch 接入
    DEEPLOB = "deeplob"                 # D2 卡带（raw-LOB 当通道），高风险，等 4060


@dataclass(frozen=True)
class ModelConfig:
    """
    本次 run 的模型选择 + 钉死的固定超参。

    - ``kind``: 选哪个卡带（registry 据此实例化对应 ``ModelCartridge``）。
    - ``hparams``: 本次钉死的固定超参（如 ``{"seq_len": 32, "hidden_size": 64}``）；
      只搜结构 3 旋钮时，未搜的旋钮从这里取默认。frozen + MappingProxy 双保险，
      组装后不可变（config-lock 机械实现，规格 §7.5）。
    """
    kind: ModelKind
    hparams: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen dataclass 内改字段需走 object.__setattr__；把 dict 冻成只读视图。
        object.__setattr__(self, "hparams", MappingProxyType(dict(self.hparams)))

    def to_dict(self) -> dict:
        """可序列化视图（枚举→value、MappingProxy→dict），供 RunConfig dump JSON。"""
        return {"kind": self.kind.value, "hparams": dict(self.hparams)}
