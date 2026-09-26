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
    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
