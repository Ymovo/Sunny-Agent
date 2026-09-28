"""agent 的工具箱 —— 每个工具一个 make_xxx() 工厂，在 build_registry() 里统一装配。

参考 waku 的 tools/__init__.py：工厂负责"造"，build_registry 只负责"装"。
以后加工具：写一个模块 + make_xxx()，然后在这里 register 一行，别处不用动。
装配发生在 loop/chat.py 的 main()（对应 waku 的 app.py 那一段）。
"""

from __future__ import annotations

from .registry import Tool, ToolRegistry

__all__ = ["Tool", "ToolRegistry", "build_registry"]


def build_registry(memory=None) -> ToolRegistry:
    """装配所有工具，返回注册表。

    memory 是鸭子类型（有 add_fact/search 即可），不给就只注册不依赖记忆的工具。
    """
    registry = ToolRegistry()

    # 长期记忆读写 —— 依赖 memory，给了才注册
    if memory is not None:
        from . import memory as memory_tools
        registry.register(memory_tools.make_save_note_tool(memory))
        registry.register(memory_tools.make_recall_tool(memory))

    # 人格（soul.md）—— 不依赖 memory，永远注册
    from . import soul
    registry.register(soul.make_update_soul_tool())

    return registry
