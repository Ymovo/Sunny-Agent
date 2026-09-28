"""人格工具 —— update_soul 把用户给的长期偏好追加进 soul.md。

参考 waku 的 memory_admin.make_update_soul_tool：**只追加、不删除**——模型
不能改掉自己的诚实底线，整段重写留给人在编辑器里做。规则统一追加到
soul.md 末尾的 "## Learned rules" 小节，和默认人格分开，一眼能看出哪些
是后学的。
"""

from __future__ import annotations

from pathlib import Path

from ..setting import DEFAULT_SOUL, SOUL_FILE

from .registry import Tool

_LEARNED = "## Learned rules"


def make_update_soul_tool(soul_file: Path = SOUL_FILE) -> Tool:
    def update_soul(rule: str) -> str:
        rule = (rule or "").strip().lstrip("-").strip()
        if not rule:
            return "没内容可写。"
        path = Path(soul_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(DEFAULT_SOUL, encoding="utf-8")     # 没有就先建默认人格
        text = path.read_text(encoding="utf-8")
        if _LEARNED not in text:
            text = text.rstrip() + f"\n\n{_LEARNED}\n"
        text = text.rstrip() + f"\n- {rule}\n"
        path.write_text(text, encoding="utf-8")
        return f"记下了，以后我会：{rule}"

    return Tool(
        name="update_soul",
        description=(
            "把用户给你的持久行为规则/偏好追加进你的人格（soul.md），下次对话生效。"
            "用户说“以后要…”“记住我喜欢…”时用。只追加，不能删除已有规则。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "rule": {"type": "string", "description": "一条行为规则，用祈使句"},
            },
            "required": ["rule"],
        },
        fn=update_soul,
    )
