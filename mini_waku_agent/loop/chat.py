"""最小 agent loop —— 只做三件事：读一行、问一次、说一句。

    加一句用户的话 → 问模型 → 把回答也加进历史 → 重复

会话状态**不在这里**：历史、滑动窗口、system prompt（模型那行 + 人格 +
检索到的长期记忆）、落盘，全在 Session 里（../session/session.py）。
长期记忆在 ../memory/。这个文件只负责把循环转起来。

依赖：见 llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 项目是"一个目录一组脚本"的平铺结构，没有 package —— 跨目录 import 靠这两行
ROOT = Path(__file__).resolve().parent.parent        # mini_waku_agent/
sys.path[:0] = [str(ROOT / "session"), str(ROOT / "memory")]

from llm import MODEL, LLM          # noqa: E402  必须在 sys.path 之后
from session import Session         # noqa: E402
from store import Memory            # noqa: E402


def main() -> None:
    llm = LLM()
    session = Session(memory=Memory(), model=MODEL)

    def turn(user: str) -> None:
        reply = llm.chat(session.build_messages(user))   # system 在第一条
        session.add_exchange(user, reply)                # 记账 + 落盘
        print(f"agent › {reply}\n")

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
