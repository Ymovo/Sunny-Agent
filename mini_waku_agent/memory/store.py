"""记忆库 —— md 是源，db.py 管向量库，embedding.py 管文本→向量。

依赖：pip install chromadb
验证于 chromadb 1.5.9 / onnxruntime 1.30.0（2026-09-19）

    semantic/FACTS.md     事实记忆，一条一行：- (YYYY-MM-DD) 内容
    episodic/EPISODES.md  情景记忆，一条一行：- YYYY-MM-DD 发生了什么

写：先 append 到 md，再写进向量库。
读：把查询转向量，查库，拿回最相关的几条。
md 可以手改 —— reindex() 按 md 重建整个索引，向量库删了也不丢东西。

三个文件的分工：
    embedding.py  文本 → 512 维向量（BGE 中文模型）
    db.py         向量库的 CRUD（ChromaDB）
    store.py      这一层：md 是源，负责把两边对齐 + 对外给 add/search

用法：
    python store.py reindex
    python store.py add-fact "用户叫小明"
    python store.py add-episode "聊了记忆系统的设计"
    python store.py search "用户叫什么"
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import date
from pathlib import Path

from db import DB, VectorDB
from embedding import embed

ROOT = Path(__file__).resolve().parent          # mini_waku_agent/memory
FACTS = ROOT / "semantic" / "FACTS.md"
EPISODES = ROOT / "episodic" / "EPISODES.md"

FACT_RE = re.compile(r"^-\s*\((\d{4}-\d{2}-\d{2})\)\s*(.+)$")
EPISODE_RE = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2})\s+(.+)$")


def _read_md(path: Path, pattern: re.Pattern) -> list[tuple[str, str]]:
    """读出 [(日期, 内容)]。标题、注释、格式不对的行一律忽略。

    按内容去重：同一条记忆在 md 里出现两次（比如手滑贴了两遍），
    索引里也只应该有一条 —— 否则内容哈希 id 会撞车，Chroma 直接报
    DuplicateIDError。
    """
    if not path.exists():
        return []
    seen: set[str] = set()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        found = pattern.match(line.strip())
        if not found:
            continue
        content = found.group(2).strip()
        if content in seen:
            continue
        seen.add(content)
        rows.append((found.group(1), content))
    return rows


def _id(kind: str, text: str) -> str:
    """内容决定 id：同一条记忆重复写入只会覆盖，不会变成两条。"""
    return f"{kind}-{hashlib.sha1(text.encode()).hexdigest()[:16]}"


class Memory:
    def __init__(self, path: Path = DB) -> None:
        self.db = VectorDB(path)

    # ---- 写：md 先落盘，向量库跟上 ---------------------------------------
    @staticmethod
    def _append(path: Path, line: str) -> bool:
        """已经有了就不重复写。返回是否真的写进去了。"""
        if path.exists() and line in path.read_text(encoding="utf-8").splitlines():
            return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True

    def add_fact(self, content: str, on: str | None = None) -> tuple[str, str]:
        on = on or date.today().isoformat()
        self._append(FACTS, f"- ({on}) {content}")
        self.db.add("fact", _id("fact", content), content, embed([content])[0], {"date": on})
        return on, content

    def add_episode(self, content: str, on: str | None = None) -> tuple[str, str]:
        on = on or date.today().isoformat()
        self._append(EPISODES, f"- {on} {content}")
        self.db.add("episode", _id("episode", content), content, embed([content])[0], {"date": on})
        return on, content

    # ---- 索引：md → 向量库（清空重建，所以手改 md 也不会漂）---------------
    def reindex(self) -> tuple[int, int]:
        return (self._load("fact", FACTS, FACT_RE),
                self._load("episode", EPISODES, EPISODE_RE))

    def _load(self, kind: str, path: Path, pattern: re.Pattern) -> int:
        rows = _read_md(path, pattern)
        self.db.clear(kind)
        if not rows:
            return 0
        vectors = embed([text for _, text in rows])          # 文档不加查询前缀
        return self.db.add_many(kind, [
            (_id(kind, text), text, vector, {"date": on})
            for (on, text), vector in zip(rows, vectors)
        ])

    # ---- 读：向量检索 ----------------------------------------------------
    def search(self, query: str, k: int = 4) -> list[dict]:
        vector = embed([query], is_query=True)[0]
        hits = []
        for kind in ("fact", "episode"):
            for record in self.db.search(kind, vector, k):
                hits.append({"kind": kind, "date": record["metadata"].get("date"),
                             "text": record["text"], "distance": record["distance"]})
        hits.sort(key=lambda hit: hit["distance"])
        return hits[:k]


def main() -> None:
    args = sys.argv[1:]
    memory = Memory()
    if not args or args[0] == "reindex":
        facts, episodes = memory.reindex()
        print(f"索引完成：{facts} 条事实，{episodes} 条情节 → {memory.db.path}")
    elif args[0] == "add-fact":
        on, content = memory.add_fact(" ".join(args[1:]))
        print(f"已写入 FACTS.md（{on}）：{content}")
    elif args[0] == "add-episode":
        on, content = memory.add_episode(" ".join(args[1:]))
        print(f"已写入 EPISODES.md（{on}）：{content}")
    elif args[0] == "search":
        for hit in memory.search(" ".join(args[1:])):
            print(f"[{hit['kind']:<7}] {hit['date']}  {hit['text']}   (距离 {hit['distance']:.3f})")
    elif args[0] == "list":
        for kind in ("fact", "episode"):
            print(f"--- {kind}（{memory.db.count(kind)} 条）")
            for record in memory.db.list(kind):
                print(f"  {record['id']}  {record['metadata'].get('date')}  {record['text']}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
