"""模型接入 —— 官方 openai SDK 的薄包装。DeepSeek 走它的 OpenAI 兼容端点。

依赖：pip install openai python-dotenv
验证于 openai 2.53.0 / python-dotenv 1.2.2（2026-09-19）

端点、模型名、token 上限都在 ../setting.py —— 这个文件只负责连接和说一句话。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mini_waku_agent/
from setting import (  # noqa: E402
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
    """一个 client，一个 chat()。网络那一层全交给 SDK。"""

    def __init__(self, model: str = MODEL) -> None:
        # .env 已由 setting.py 在 import 时加载好
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            raise SystemExit(f"缺少 {API_KEY_ENV}，请写在 {ENV_FILE}")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 连接大模型
    def _once(self, messages: list[dict]) -> tuple[str, str]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=MAX_TOKENS,
        )
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason or ""

    def chat(self, messages: list[dict]) -> str:
        """拿一次回复。空回复重试一次，还空就抛 EmptyReply —— 绝不静默返回空串。"""
        content, finish = self._once(messages)
        if content.strip():
            return content
        content, finish = self._once(messages)      # 换个采样再试，常能出内容
        if content.strip():
            return content
        raise EmptyReply(
            f"连续两次空回复（finish_reason={finish}, max_tokens={MAX_TOKENS}）："
            "多半是思考过程把 token 配额吃光了 —— 调大 MINIWAKU_MAX_TOKENS，"
            "或换一个非推理模型")
