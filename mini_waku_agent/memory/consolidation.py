"""蒸馏 —— 把 chat_log 里还没消化的记录，提炼成 fact。

参考 waku 的 consolidation.py，核心是那个"菱形"：**不是每条都蒸馏，攒够
every_n 条才跑一次**。每条都跑又贵又吵；攒一批再让模型读，上下文够了，
才抽得出值得长期记住的事实。

流程（consolidate_if_due）：
    读 chat_log 里 consolidated = 0 的行
      → 不够 every_n 条就返回 0（下次再说）
      → 让模型读这批记录，抽出耐久的事实（JSON）
      → 一条条写进 fact 表（内容哈希去重，重复的自动合并；
         source 记为 "consolidation"，和手加的事实区分开）
      → 把这批行标记 consolidated = 1

失败不丢：任何异常都返回 0，这批行保持未标记，下次接着蒸馏。

注意：chat_log 一行就是一条记录（不像 waku 一行是一条消息），所以阈值是
every_n 条，不是 every_n * 2。

用法：
    python consolidation.py            # 手动跑一次
被 loop/chat.py 在每轮对话后调用（攒够才真正触发）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mini_waku_agent/
from setting import CONSOLIDATE_EVERY_N  # noqa: E402

from store import Memory  # noqa: E402

SUMMARIZER_PROMPT = """\
你在把一个私人助手最近的记录，蒸馏成长期记忆。

从下面的记录里，抽出**值得长期记住的事实**：关于用户本人、他身边的人、项目、
偏好的东西 —— 一个月后依然成立、以后再聊到时用得上的。闲聊、一次性的琐事、
助手自己的客套话都不要。

每条事实一句话，主语写清楚（是"用户"还是别人）。

只回这个 JSON，不要任何多余文字：
{{"facts": [{{"subject": "<关于谁/什么>", "content": "<一句话>"}}]}}

记录：
{log}"""


def consolidate_if_due(memory: Memory, llm, *,
                       every_n: int = CONSOLIDATE_EVERY_N) -> int:
    """把到期的 chat_log 蒸馏进 fact，返回新写了多少条事实（0 = 没到期 / 没东西可记）。

    memory 是 store.Memory；llm 只要有 chat(messages) -> str 就行（鸭子类型）。
    """
    rows = memory.pending_chat_logs()
    if len(rows) < every_n:                 # 没攒够，下次再说
        return 0

    log = "\n".join(f"- ({row['date']}) {row['text']}" for row in rows)
    try:
        text = llm.chat([{"role": "user",
                          "content": SUMMARIZER_PROMPT.format(log=log)}])
        if "{" not in text:                 # 推理模型只吐了思考 / 回答被截断
            return 0
        distilled = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        return 0                            # 绝不丢记录：这批行保持未标记，下次再来

    written = 0
    for fact in distilled.get("facts", []):
        subject = str(fact.get("subject") or "").strip()
        content = str(fact.get("content") or "").strip()
        if not content:
            continue
        memory.add_fact(f"{subject}：{content}" if subject else content,
                        source="consolidation")
        written += 1

    # 解析成功就算处理过 —— 哪怕一条事实都没抽到，也别下次再喂一遍
    memory.mark_consolidated([row["id"] for row in rows])
    return written


def main() -> None:
    # 延迟 import：memory 层不该在模块级依赖 loop 层
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loop"))
    from llm import LLM

    memory = Memory()
    before = len(memory.pending_chat_logs())
    written = consolidate_if_due(memory, LLM())
    after = len(memory.pending_chat_logs())
    if after < before:
        print(f"蒸馏完成：处理 {before - after} 条 chat_log，新增 {written} 条事实")
    elif before < CONSOLIDATE_EVERY_N:
        print(f"没到阈值：待蒸馏 {before} 条，需要 {CONSOLIDATE_EVERY_N} 条才跑")
    else:
        print(f"这轮没蒸出东西（模型没回可解析的 JSON），{before} 条保持待处理")


if __name__ == "__main__":
    main()
