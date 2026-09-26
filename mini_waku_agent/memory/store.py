"""记忆库 —— SQLite 是唯一源。db.py 管存储，embedding.py 管文本→向量。

依赖：无（存储是 SQLite，见 db.py；向量是 onnxruntime，见 embedding.py）
验证于 onnxruntime 1.30.0（2026-09-19）

两张表（表名与库文件位置见 ../setting.py），写进来就是真相，没有中间文件：
    fact     事实记忆（关于人的、不过期的东西：偏好、习惯、关系）
    chat_log 聊天记录（发生过的事，带日期）

写：直接写进 SQLite 表（add_fact / add_chat_log / log_exchange）。
读：查询转向量 → 两张表各检索 top-k → 按距离合并排序。
fact.source 记来源：manual（手加）/ consolidation（蒸馏）。

三个文件的分工：
    embedding.py  文本 → 512 维向量（BGE 中文模型）
    db.py         向量库的 CRUD（SQLite）
    store.py      这一层：对外给 add / search，语义在文本、存储在下层

用法：
    python store.py add-fact "用户叫小明"
    python store.py add-chat-log "聊了记忆系统的设计"
    python store.py search "用户叫什么"
    python store.py list
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mini_waku_agent/
from setting import DB_FILE, MEMORY_TABLES, SEARCH_K  # noqa: E402

from db import VectorDB
from embedding import embed


def _id(kind: str, text: str) -> str:
    """内容决定 id：同一条记忆重复写入只会覆盖，不会变成两条。"""
    return f"{kind}-{hashlib.sha1(text.encode()).hexdigest()[:16]}"


class Memory:
    def __init__(self, path: Path = DB_FILE) -> None:
        self.db = VectorDB(path)

    # ---- 写：直接进 SQLite 表 --------------------------------------------
    def add_fact(self, content: str, on: str | None = None,
                 source: str = "manual") -> tuple[str, str]:
        """source 记这条事实哪来的："manual"（手加）或 "consolidation"（蒸馏）。"""
        on = on or date.today().isoformat()
        self.db.add("fact", _id("fact", content), content, embed([content])[0], on, source)
        return on, content

    def add_chat_log(self, content: str, on: str | None = None) -> tuple[str, str]:
        on = on or date.today().isoformat()
        self.db.add("chat_log", _id("chat_log", content), content, embed([content])[0], on)
        return on, content

    def log_exchange(self, user_message: str, reply: str,
                     on: str | None = None) -> tuple[str, str]:
        """把一轮对话记进 chat_log —— 之后会被蒸馏成 fact（见 consolidation.py）。"""
        return self.add_chat_log(f"用户：{user_message}\n助手：{reply}", on)

    # ---- 蒸馏：chat_log → fact（逻辑在 consolidation.py）-------------------
    def pending_chat_logs(self) -> list[dict]:
        """chat_log 里还没蒸馏进 fact 的记录。"""
        return self.db.pending("chat_log")

    def mark_consolidated(self, ids: list[str]) -> int:
        """把这些 chat_log 标成已蒸馏，返回改了几条。"""
        return self.db.mark_consolidated("chat_log", ids)

    # ---- 读：向量检索 ----------------------------------------------------
    def search(self, query: str, k: int = SEARCH_K) -> list[dict]:
        vector = embed([query], is_query=True)[0]
        hits = []
        for kind in MEMORY_TABLES:
            for record in self.db.search(kind, vector, k):
                hits.append({"kind": kind, "date": record["date"],
                             "text": record["text"], "distance": record["distance"]})
        hits.sort(key=lambda hit: hit["distance"])
        return hits[:k]


def main() -> None:
    args = sys.argv[1:]
    memory = Memory()
    if not args:
        print(__doc__)
    elif args[0] == "add-fact":
        on, content = memory.add_fact(" ".join(args[1:]))
        print(f"已写入 fact 表（{on}）：{content}")
    elif args[0] == "add-chat-log":
        on, content = memory.add_chat_log(" ".join(args[1:]))
        print(f"已写入 chat_log 表（{on}）：{content}")
    elif args[0] == "search":
        for hit in memory.search(" ".join(args[1:])):
            print(f"[{hit['kind']:<8}] {hit['date']}  {hit['text']}   (距离 {hit['distance']:.3f})")
    elif args[0] == "list":
        for kind in MEMORY_TABLES:
            print(f"--- {kind}（{memory.db.count(kind)} 条）")
            for record in memory.db.list(kind):
                tag = f"[{record['source']}] " if record.get("source") else ""
                text = " ".join(record["text"].split())      # 折掉换行，一行显示
                print(f"  {record['id']}  {record['date']}  {tag}{text}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
