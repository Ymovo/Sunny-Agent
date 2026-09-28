"""循环 —— 对外暴露模型接入；真正的 agent loop 在 chat.py（含 main 装配）。"""

from __future__ import annotations

from .llm import EmptyReply, LLM

__all__ = ["LLM", "EmptyReply"]
