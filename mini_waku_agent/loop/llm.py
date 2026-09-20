"""模型接入 —— 官方 openai SDK 的薄包装。DeepSeek 走它的 OpenAI 兼容端点。

依赖：pip install openai python-dotenv
验证于 openai 2.53.0 / python-dotenv 1.2.2（2026-09-19）
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"   # mini_waku_agent/.env


class LLM:
    """一个 client，一个 chat()。网络那一层全交给 SDK。"""

    def __init__(self, model: str = MODEL) -> None:
        load_dotenv(ENV_FILE)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(f"缺少 DEEPSEEK_API_KEY，请写在 {ENV_FILE}")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 连接大模型
    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=512,
        )
        return response.choices[0].message.content or ""
