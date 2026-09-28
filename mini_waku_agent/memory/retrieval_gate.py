"""检索门控 —— 决定"这一句到底要不要去翻长期记忆"。

参考 waku 的 retrieval_gate.py。默认每次都检索是 (a) 慢——每次回复前多一次
搜索，(b) 更糟——不相关的记忆会带偏回答（过度解读）。所以在碰向量库之前，
先让模型回答一个窄问题：这句话需要用户的记忆吗？

    "2+2 等于几"        → 不用
    "我几点见 Alex？"   → 用，顺带给出检索词

代价：一次小模型调用。回报：只在有用时才检索。门控自己出错就"开闸"（照样
检索）—— 一条过期的记忆好过一条丢掉的记忆。
"""

from __future__ import annotations

import json

GATE_PROMPT = """\
You are a retrieval gate for a personal assistant's long-term memory.
Given the user's message, decide if answering well requires the user's stored
memories (facts about people, projects, preferences, or past events).

Reply with ONLY this JSON, nothing else:
{{"retrieve": true/false, "query": "<search keywords if true, else empty>", "reason": "<a few words>"}}

General knowledge, math, small talk, or self-contained requests → false.
Anything referencing the user's life, people, plans, or history → true.

User message: {message}"""


def should_retrieve(llm, message: str) -> tuple[bool, str, str]:
    """返回 (要不要检索, 检索词, 原因)。llm 是鸭子类型，有 chat(messages) -> str 即可。

    失败就开闸（检索）—— 宁可多带一条记忆，也别因为门坏了丢掉该有的上下文。
    """
    try:
        text = llm.chat([{"role": "user", "content": GATE_PROMPT.format(message=message)}])
        if "{" not in text:                        # 只吐了思考 / 回答被截断，不是错误
            return True, message, "gate returned no JSON — failing open"
        decision = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return (bool(decision.get("retrieve")),
                decision.get("query") or message,
                decision.get("reason") or "")
    except Exception as exc:
        return True, message, f"gate failed open ({type(exc).__name__})"
