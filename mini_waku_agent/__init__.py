"""mini_waku_agent —— 一个最小、透明、本地优先的个人助手。

四个模块，各管一段：
    session  会话 —— 历史、滑动窗口、system prompt、落盘
    memory   长期记忆 —— 语义检索（带门控）、chat_log 蒸馏成 fact
    loop     循环 —— observe → reason → act → repeat，外加模型接入
    tool     工具 —— 注册表 + make_xxx() 工厂

装配发生在 loop/chat.py 的 main()：llm → memory → tools → session → loop。
"""
