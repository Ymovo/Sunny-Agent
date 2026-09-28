"""会话 —— 对外暴露 Session 和 load_soul，内部怎么组织不管调用方的事。"""

from __future__ import annotations

from .session import Session, load_soul

__all__ = ["Session", "load_soul"]
