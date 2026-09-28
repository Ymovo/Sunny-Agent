"""长期记忆的工具 —— 让模型自己读写 memory。

memory 是鸭子类型：只要有 add_fact(content) 和 search(query) 就够了，这一层
不 import store 的实现（和 session.py 一个约定）。参考 waku 的 notes.py：
一个工具 = 一个 make_xxx() 工厂，返回 Tool，方便后续加新工具。

save_note 对应"记住"，recall 对应"取回"—— 正好覆盖记忆的写、读两条路。
"""

from __future__ import annotations

from .registry import Tool


def make_save_note_tool(memory) -> Tool:
    def save_note(subject: str, content: str) -> str:
        subject = (subject or "").strip()
        content = (content or "").strip()
        if not content:
            return "没内容可记。"
        text = f"{subject}：{content}" if subject else content
        on, saved = memory.add_fact(text)
        return f"已记住（{on}）：{saved}"

    return Tool(
        name="save_note",
        description=(
            "把一条值得长期记住的事实写进记忆。用户主动分享关于自己、某人或某项目的"
            "偏好、习惯、关系时用，尤其是说了“记住”的时候。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "关于谁/什么，如 'alex' 或 'acme 项目'",
                },
                "content": {
                    "type": "string",
                    "description": "一句话说清这条事实",
                },
            },
            "required": ["subject", "content"],
        },
        fn=save_note,
    )


def make_recall_tool(memory) -> Tool:
    def recall(query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "给我一句要检索的话。"
        hits = memory.search(query)
        if not hits:
            return "没检索到相关记忆。"
        return "\n".join(f"- {h['text']}（{h['kind']}）" for h in hits)

    return Tool(
        name="recall",
        description=(
            "检索自己的长期记忆。回答用户问题前不确定背景、或用户问"
            "“你还记得什么”时用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的问题或关键词"},
            },
            "required": ["query"],
        },
        fn=recall,
    )
