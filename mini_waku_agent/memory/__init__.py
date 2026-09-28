"""记忆 —— 对外只暴露这几个名字，内部实现（db / embedding / store /
retrieval_gate / consolidation）怎么改都不影响调用方。

    Memory             门面：写 fact / chat_log，检索（带门控），蒸馏
    should_retrieve    检索门控：决定这一句要不要翻记忆（失败开闸）
    consolidate_if_due 蒸馏：把 chat_log 提炼成 fact
"""

from __future__ import annotations

from .consolidation import consolidate_if_due
from .retrieval_gate import should_retrieve
from .store import Memory

__all__ = ["Memory", "should_retrieve", "consolidate_if_due"]
