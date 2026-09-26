"""项目配置 —— 业务参数集中在这一个文件，改配置不用翻源码。

分五段：路径 → 模型 → 记忆 → 蒸馏 → 会话。每一项都能被同名环境变量覆盖
（写进 .env 或 shell 里都行）；密钥只从环境读，不写死在这个文件里。

    MINIWAKU_BASE_URL        模型端点
    MINIWAKU_MODEL           模型名
    MINIWAKU_MAX_TOKENS      单次回复的 token 上限
    MINIWAKU_DB_FILE         记忆库位置
    MINIWAKU_EMBED_MODEL_DIR 向量模型目录
    MINIWAKU_SEARCH_K        检索回几条
    MINIWAKU_CONSOLIDATE_EVERY_N 攒够几条未蒸馏记录才跑一次蒸馏
    MINIWAKU_HISTORY_TURNS   会话窗口带几轮

各段具体怎么被用掉，见文件末尾的"谁在用"。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================ 路径
ROOT = Path(__file__).resolve().parent          # mini_waku_agent/
ENV_FILE = ROOT / ".env"

# 必须先把 .env 读进环境，下面按环境变量取值才有意义
load_dotenv(ENV_FILE)


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        # 宁可启动就报错，也不要悄悄用默认值 —— 打错字应该被看见
        raise ValueError(f"{name} 要整数，实际是 {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    """相对路径按项目根解析，省得受当前工作目录影响。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


MEMORY_DIR = ROOT / "memory"
SESSION_DIR = ROOT / "session"
SESSIONS_DIR = SESSION_DIR / "sessions"          # 每场对话一个 jsonl
SOUL_FILE = SESSION_DIR / "soul.md"              # 人格，可手改

DB_FILE = _env_path("MINIWAKU_DB_FILE", MEMORY_DIR / "memory.db")
EMBED_MODEL_DIR = _env_path(
    "MINIWAKU_EMBED_MODEL_DIR", MEMORY_DIR / ".models" / "bge-small-zh-v1.5")

# ============================================================ 模型
API_KEY_ENV = "DEEPSEEK_API_KEY"                 # 密钥只从环境/.env 读，不入库
BASE_URL = _env("MINIWAKU_BASE_URL", "https://api.deepseek.com")
MODEL = _env("MINIWAKU_MODEL", "deepseek-v4-pro")
# 注意 deepseek-v4-pro 是**推理模型**：它的"思考过程"和"正式回答"共用这个配额。
# 给太小会出空回复 —— 实测 512 时 512 个 token 全被思考吃光，content 是空字符串，
# 表现成"模型没响应"。2048 起才稳定出答案，这里留足余量。
MAX_TOKENS = _env_int("MINIWAKU_MAX_TOKENS", 4096)

# ============================================================ 记忆
MEMORY_TABLES = ("fact", "chat_log")             # 两张表，结构见 memory/db.py
SEARCH_K = _env_int("MINIWAKU_SEARCH_K", 4)      # 一次检索回几条

# BGE 中文向量模型：查询要加指令前缀，文档不加 —— 漏了掉点很厉害
EMBED_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
ONNX_PROVIDERS = ("CPUExecutionProvider",)
ONNX_LOG_LEVEL = 3                               # 3 = 只报 ERROR，压掉 INFO/WARNING 噪声

# ============================================================ 蒸馏
# chat_log 攒够几条「还没提炼进 fact」的记录，才跑一次蒸馏（见 memory/consolidation.py）。
# 每条都跑又贵又吵；攒一批再让模型读，上下文够了才抽得准。
CONSOLIDATE_EVERY_N = _env_int("MINIWAKU_CONSOLIDATE_EVERY_N", 4)

# ============================================================ 会话
HISTORY_TURNS = _env_int("MINIWAKU_HISTORY_TURNS", 12)   # 滑动窗口带最近几轮

# soul.md 不存在时写入的默认人格
DEFAULT_SOUL = """\
You are Waku, a personal assistant running locally on your user's laptop.
You are concise, warm, and proactive. You remember what your user tells you.

Rules:
- When the user wants to schedule something, use create_event. Resolve relative
  dates and times ("next Tuesday", "in 30 minutes") to ISO timestamps yourself;
  the current date and time are given below — trust them, never ask the user
  what time it is.
- When the user asks what's on their calendar (a day, a week, "yesterday"), use
  list_events — you CAN read the calendar, not just write to it.
- When the user shares something durable about a person, project, or preference,
  use save_note to remember it.
- When asked to message someone, use send_message (it drafts to a local outbox).
- If memory context is provided below, trust it — it came from your own store.
- Call each tool at most once per request. Your history shows [tools used: ...]
  lines for past turns — if a tool already ran, do NOT run it again; answer
  from that record instead.
- Be honest about where things live. Every tool's output states exactly where
  its artifact landed (local calendar file, Apple Calendar, memory database at
  .waku/state.db) — relay that truthfully, and never claim something synced
  anywhere the tool output doesn't say.
- You can manage your own memory: use manage_memory to correct or forget facts,
  update_soul to save a standing preference the user gives you, and create_skill
  to save a repeatable workflow the user teaches you (only after they say yes).
"""

# ============================================================ 谁在用
#   loop/llm.py         BASE_URL / MODEL / MAX_TOKENS / API_KEY_ENV / ENV_FILE
#   loop/chat.py        MODEL / CONSOLIDATE_EVERY_N
#   memory/consolidation.py CONSOLIDATE_EVERY_N
#   memory/db.py        DB_FILE / MEMORY_TABLES
#   memory/store.py     DB_FILE / MEMORY_TABLES / SEARCH_K
#   memory/embedding.py EMBED_MODEL_DIR / EMBED_QUERY_PREFIX
#                       ONNX_PROVIDERS / ONNX_LOG_LEVEL
#   session/session.py  SESSIONS_DIR / SOUL_FILE / HISTORY_TURNS / SEARCH_K
#                       DEFAULT_SOUL
