"""向量库 —— SQLite 本地存储。只管向量怎么存、怎么查，不认识文本。

依赖：无（sqlite3 是标准库）；向量运算复用 numpy（embedding.py 已经引入）

库文件位置与表名在 ../setting.py。

两张表，分别对应两类记忆：
    fact     事实记忆（关于人的、不过期的东西）
    chat_log 聊天记录（发生过的事，带日期）

这两张表就是记忆的**源数据**：写进来就是真相，没有别的中间文件。
（表名拼进 SQL 前都经过白名单校验，无注入风险。）

    共有的列：
        id         TEXT   -- 内容哈希，内容决定 id
        text       TEXT   -- 原文
        embedding  BLOB   -- 512 个 float32
        date       TEXT   -- YYYY-MM-DD
    各自的附加列：
        fact.source             -- 这条事实哪来的："manual" / "consolidation"
        chat_log.consolidated   -- 是否已蒸馏进 fact：0 / 1（见 consolidation.py）

向量检索：SQLite 没有原生向量索引，所以直接在 Python 里暴力算平方欧氏距离。
512 维、几百条规模下是毫秒级，完全够用；真要上规模再换专门的向量库，
这一层接口不变。

    db = VectorDB()
    db.add("fact", "fact-abc", "用户叫小明", vector, "2026-09-19", source="manual")
    db.get("fact", "fact-abc")          db.list("fact")     db.count("fact")
    db.search("fact", query_vector, k=4)
    db.update("fact", "fact-abc", text="...", embedding=[...], date="...")
    db.delete("fact", "fact-abc")       db.clear("fact")

文本 → 向量是 embedding.py 的事，这一层只收向量。所以换模型、换维度，
db.py 一个字都不用改。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mini_waku_agent/
from setting import DB_FILE, MEMORY_TABLES  # noqa: E402

_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id        TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL,
    date      TEXT NOT NULL DEFAULT ''
);
"""

# 每张表在自己基础结构之外额外的列；老库缺列就在 __init__ 里自动补上。
#   fact.source            这条事实哪来的：manual（手加/命令行）/ consolidation（蒸馏）
#   chat_log.consolidated  是否已蒸馏进 fact（0/1），逻辑见 consolidation.py
_EXTRA_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "fact": (("source", "TEXT NOT NULL DEFAULT 'manual'"),),
    "chat_log": (("consolidated", "INTEGER NOT NULL DEFAULT 0"),),
}

# 读取时按这个顺序挑出表里实际存在的列（id 之外）
_RECORD_COLUMNS = ("text", "date", "source", "consolidated")


class VectorDB:
    def __init__(self, path: Path = DB_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        for table in MEMORY_TABLES:
            self._conn.execute(_SCHEMA.format(table=table))
            for column, decl in _EXTRA_COLUMNS.get(table, ()):
                if column not in self._columns(table):        # 老库补列
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        self._conn.commit()

    # ---- 生命周期 --------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VectorDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 内部工具 --------------------------------------------------------
    @staticmethod
    def _check(kind: str) -> None:
        if kind not in MEMORY_TABLES:
            raise ValueError(f"未知表 {kind!r}，只有 {', '.join(MEMORY_TABLES)}")

    @staticmethod
    def _pack(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _unpack(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def _columns(self, table: str) -> set[str]:
        return {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}

    def _record_columns(self, kind: str) -> list[str]:
        """这张表实际有的业务列（id 之外），顺序固定，便于 get/list/search 复用。"""
        available = self._columns(kind)
        return [c for c in _RECORD_COLUMNS if c in available]

    @staticmethod
    def _row_to_record(columns: list[str], row: Sequence[Any]) -> dict[str, Any]:
        return {"id": row[0], **dict(zip(columns, row[1:]))}

    # ---- Create / Update：upsert，id 在就是更新、不在就是新建 -------------
    def add(self, kind: str, record_id: str, text: str,
            embedding: list[float], date: str = "", source: str = "") -> None:
        self.add_many(kind, [(record_id, text, embedding, date, source)])

    def add_many(self, kind: str,
                 records: Iterable[tuple[str, str, list[float], str, str]]) -> int:
        """批量写。records = [(id, text, embedding, date, source), ...]，返回写了几条。

        同一条记录重复写入只会覆盖（主键 upsert），不会变成两条。
        source 只有 fact 表有这一列，别的表会自动忽略。
        """
        self._check(kind)
        has_source = "source" in self._columns(kind)
        columns = ["id", "text", "embedding", "date"] + (["source"] if has_source else [])
        updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "id")
        rows = []
        for record_id, text, embedding, date, source in records:
            row = [record_id, text, self._pack(embedding), date or ""]
            if has_source:
                row.append(source or "")
            rows.append(row)
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT INTO {kind} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ---- Read -----------------------------------------------------------
    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        self._check(kind)
        columns = self._record_columns(kind)
        row = self._conn.execute(
            f"SELECT id, {', '.join(columns)} FROM {kind} WHERE id = ?", (record_id,)
        ).fetchone()
        return None if row is None else self._row_to_record(columns, row)

    def list(self, kind: str) -> list[dict[str, Any]]:
        self._check(kind)
        columns = self._record_columns(kind)
        rows = self._conn.execute(
            f"SELECT id, {', '.join(columns)} FROM {kind} ORDER BY rowid"
        ).fetchall()
        return [self._row_to_record(columns, row) for row in rows]

    def count(self, kind: str) -> int:
        self._check(kind)
        return self._conn.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()[0]

    def search(self, kind: str, embedding: list[float], k: int = 4) -> list[dict[str, Any]]:
        """向量检索：平方欧氏距离升序，取前 k 条。

        embedding 已 L2 归一化（见 embedding.py），平方欧氏距离和余弦距离
        是单调关系，排序结果相同。
        """
        self._check(kind)
        columns = self._record_columns(kind)
        rows = self._conn.execute(
            f"SELECT id, {', '.join(columns)}, embedding FROM {kind}"
        ).fetchall()
        if not rows:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        matrix = np.vstack([self._unpack(row[-1]) for row in rows])
        distances = np.sum((matrix - query) ** 2, axis=1)
        top = np.argsort(distances)[:min(k, len(rows))]
        hits = []
        for i in top:
            record = {"id": rows[i][0], **dict(zip(columns, rows[i][1:-1]))}
            record["distance"] = float(distances[i])
            hits.append(record)
        return hits

    # ---- Update：只改传进来的字段 -----------------------------------------
    def update(self, kind: str, record_id: str, *, text: str | None = None,
               embedding: list[float] | None = None,
               date: str | None = None) -> bool:
        """返回是否改到了东西。

        改 text 必须同时给 embedding —— 否则向量和文本就对不上了，
        检索会拿着旧向量去匹配新文本，悄悄错位。
        """
        self._check(kind)
        if self.get(kind, record_id) is None:
            return False
        if text is not None and embedding is None:
            raise ValueError("改 text 必须同时给新的 embedding，否则向量和文本对不上")
        sets: list[str] = []
        params: list[Any] = []
        if text is not None:
            sets.append("text = ?")
            params.append(text)
        if embedding is not None:
            sets.append("embedding = ?")
            params.append(self._pack(embedding))
        if date is not None:
            sets.append("date = ?")
            params.append(date)
        if sets:
            params.append(record_id)
            self._conn.execute(
                f"UPDATE {kind} SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()
        return True

    # ---- Delete ---------------------------------------------------------
    def delete(self, kind: str, record_id: str) -> bool:
        self._check(kind)
        cursor = self._conn.execute(f"DELETE FROM {kind} WHERE id = ?", (record_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self, kind: str) -> int:
        """清空一张表，返回删掉几条。"""
        self._check(kind)
        cursor = self._conn.execute(f"DELETE FROM {kind}")
        self._conn.commit()
        return cursor.rowcount

    # ---- 蒸馏：待处理 / 标记已处理 ---------------------------------------
    # 只有带 consolidated 列的表（chat_log）能用，逻辑在 consolidation.py。
    def pending(self, kind: str) -> list[dict[str, Any]]:
        """还没蒸馏过的行（consolidated = 0），按写入顺序返回。"""
        self._check(kind)
        if "consolidated" not in self._columns(kind):
            raise ValueError(f"{kind} 表没有 consolidated 列，取不了待蒸馏行")
        columns = self._record_columns(kind)
        rows = self._conn.execute(
            f"SELECT id, {', '.join(columns)} FROM {kind} "
            "WHERE consolidated = 0 ORDER BY rowid"
        ).fetchall()
        return [self._row_to_record(columns, row) for row in rows]

    def mark_consolidated(self, kind: str, ids: Sequence[str]) -> int:
        """把这些行标成已蒸馏，返回实际改了几行。"""
        self._check(kind)
        ids = list(ids)
        if not ids:
            return 0
        cursor = self._conn.execute(
            f"UPDATE {kind} SET consolidated = 1 WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        )
        self._conn.commit()
        return cursor.rowcount
