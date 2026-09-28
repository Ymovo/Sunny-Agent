"""模型接入 —— 官方 openai SDK 的薄包装。DeepSeek 走它的 OpenAI 兼容端点。

依赖：pip install openai python-dotenv
验证于 openai 2.53.0 / python-dotenv 1.2.2（2026-09-19）

端点、模型名、token 上限都在 ../setting.py。

call() 是带工具的底层：返回 (content, tool_calls, finish_reason)，空回复重试。
chat() 是纯文本包装（蒸馏用）；真正"要工具 → 执行 → 回填"的循环在 loop/chat.py，
这里只管连接和说一句话。
"""

from __future__ import annotations

import os

from openai import OpenAI

from ..setting import (
    API_KEY_ENV, BASE_URL, ENV_FILE, MAX_TOKENS, MODEL,
)


class EmptyReply(RuntimeError):
    """模型返回了空 content。

    deepseek-v4-pro 这类**推理模型**，思考过程和正式回答**共用** max_tokens。
    思考一长就把配额吃光，content 返回空串 —— 不处理的话界面看起来就是
    "模型没响应"（`agent ›` 后面什么都没有），而且这种空回复还会被记进
    历史和 chat_log。所以这里显式抛出来，交给调用方决定怎么提示。
    """


class LLM:
    """一个 client，一个 call()。网络那一层全交给 SDK。"""

    def __init__(self, model: str = MODEL) -> None:
        # .env 已由 setting.py 在 import 时加载好
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            raise SystemExit(f"缺少 {API_KEY_ENV}，请写在 {ENV_FILE}")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 连接大模型
    def _once(self, messages: list[dict],
              tools: list[dict] | None = None) -> tuple[str, list, str]:
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS}
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        return (message.content or "", message.tool_calls or [], choice.finish_reason or "")

    def call(self, messages: list[dict],
             tools: list[dict] | None = None) -> tuple[str, list, str]:
        """带工具的一次调用。空回复（既没内容也没要工具）重试一次，还空抛 EmptyReply。

        tool_calls 是 OpenAI 的形状：[{id, type, function: {name, arguments}}]，没有则 []。
        """
        content, tool_calls, finish = self._once(messages, tools)
        if tool_calls or content.strip():
            return content, tool_calls, finish
        content, tool_calls, finish = self._once(messages, tools)   # 换个采样再试，常能出内容
        if tool_calls or content.strip():
            return content, tool_calls, finish
        raise EmptyReply(
            f"连续两次空回复（finish_reason={finish}, max_tokens={MAX_TOKENS}）："
            "多半是思考过程把 token 配额吃光了 —— 调大 MINIWAKU_MAX_TOKENS，"
            "或换一个非推理模型")

    def chat(self, messages: list[dict]) -> str:
        """纯文本聊天（不带工具）：蒸馏那类一次调用。空回复重试后抛 EmptyReply。"""
        content, _, _ = self.call(messages)
        return content
