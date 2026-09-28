"""最小 agent loop —— 读一行、问一次（要工具就调工具）、说一句、记一轮。

    加一句用户的话 → 问模型（带工具）→ 模型要工具就执行并回填 → 循环到它给出回答
    → 把回答加进历史 / chat_log → 蒸馏

会话状态、system prompt、落盘在 Session（../session/session.py）；长期记忆在
../memory/；工具在 ../tool/。这个文件只负责"把循环转起来" + "把零件装起来"。

装配顺序（参考 waku 的 app.py）：llm → memory → tools → session，然后 loop。
"""

from __future__ import annotations

import json
import sys

from ..memory import Memory, consolidate_if_due
from ..session import Session
from ..setting import CONSOLIDATE_EVERY_N, MAX_ITERATIONS, MODEL
from ..tool import build_registry
from .llm import EmptyReply, LLM


def run_turn(llm: LLM, session: Session, tools, user: str) -> str:
    """一轮：system + 历史 + 这一句 → 循环（模型要工具就给工具结果）→ 最终回答。

    参考 waku 的 loop/agent.py：reason（问一次）→ act（执行工具）→ observe（回填）
    → 直到模型不再要工具（= 在对人说话），或撞上 MAX_ITERATIONS。
    """
    messages = session.build_messages(user)
    for _ in range(MAX_ITERATIONS):
        content, tool_calls, _ = llm.call(messages, tools.schemas())
        if not tool_calls:                        # 模型不再要工具 = 在对人说话
            return content

        # 把 assistant 的 tool_calls 记进工作记忆，再逐个执行、把结果回填
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": call.id, "type": "function",
                 "function": {"name": call.function.name,
                              "arguments": call.function.arguments}}
                for call in tool_calls
            ],
        })
        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}                          # 参数是坏 JSON 就当没参数，工具会兜住
            output = tools.execute(name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return "(迭代次数用尽，先把这件事拆小一点再问)"


def main() -> None:
    # 装配（参考 waku/app.py 的 Waku.__init__）：llm → memory → tools → session
    llm = LLM()
    memory = Memory(llm=llm)                       # 检索门控要用 llm 判断该不该翻记忆
    tools = build_registry(memory)                 # 工具在这里注册
    session = Session(memory=memory, model=MODEL)

    def turn(user: str) -> None:
        try:
            reply = run_turn(llm, session, tools, user)
        except EmptyReply as exc:
            # 空回复：既不当作回答，也不写进历史 / chat_log —— 免得污染记忆
            print(f"（模型这次没给出回答：{exc}）")
            print("  再说一次通常就好；老是这样就调大 MINIWAKU_MAX_TOKENS。\n")
            return
        session.add_exchange(user, reply)                # 会话历史 + 落盘
        memory.log_exchange(user, reply)                 # 这一轮也进 chat_log，供蒸馏
        print(f"agent › {reply}\n")
        # 攒够 every_n 条未蒸馏的 chat_log，就顺手提炼进 fact（不够则什么都不做）
        written = consolidate_if_due(memory, llm, every_n=CONSOLIDATE_EVERY_N)
        if written:
            print(f"（已从 chat_log 蒸馏出 {written} 条新事实写进 fact 表）\n")

    if len(sys.argv) > 1:                                 # 一次性提问
        turn(" ".join(sys.argv[1:]))
        return

    print(f"会话 {session.session_id}（已有 {len(session.history) // 2} 轮）—— /quit 退出")
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        turn(user)


if __name__ == "__main__":
    main()
