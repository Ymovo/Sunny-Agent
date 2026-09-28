"""工具注册表 —— agent 手里的"工具箱"。

一个工具 = 名字 + 描述 + 参数 schema + 一个能跑的函数。前三个给模型看，
模型据此决定要不要调、怎么调；真正执行的是第四个。设计参考 waku 的
tools/registry.py，只是把对外的 schema 换成 OpenAI 函数调用格式 ——
本项目走 DeepSeek 的 OpenAI 兼容端点，chat.completions.create(tools=...)
要的就是这个形状。

    registry = ToolRegistry()
    registry.register(make_save_note_tool(memory))
    registry.schemas()                       # 喂给模型的 tools= 参数
    registry.execute("save_note", {"subject": "alex", "content": "…"})

新增一个工具三步：写 make_xxx() 工厂返回 Tool → 在 tool/__init__.py 的
build_registry() 里 register 一行 → 完。registry 只认识 Tool，不认识任何
具体实现，所以换后端、加工具都不用动这里。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]     # JSON Schema：这个工具要哪些参数
    fn: Callable[..., str]         # 真正干活的函数，返回一段文本给模型看

    def to_openai(self) -> dict[str, Any]:
        """OpenAI 函数调用格式 —— chat.completions.create(tools=...) 要的形状。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """按名字登记工具；执行时兜住异常，绝不把 loop 弄崩。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """登记一个工具。同名后注册的覆盖先注册的。"""
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """给模型看的工具清单（OpenAI 的 tools= 参数）。"""
        return [tool.to_openai() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """安全地跑一次工具调用：结果或错误都当成文本返回，让模型自己看着办。

        参考 execute_tool_safely 模式 —— 工具抛异常不能把整个 loop 带崩，
        把报错变成文本喂回去，模型可以据此重试或换个说法。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.fn(**args)
        except Exception as exc:
            return f"Error running {name}: {exc}"
