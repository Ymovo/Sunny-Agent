"""向量库 —— ChromaDB 的 CRUD 封装。只管向量怎么存、怎么查，不认识文本。

依赖：pip install chromadb
验证于 chromadb 1.5.9（2026-09-19）

两个集合：fact（事实记忆）、episode（情景记忆）。
每条记录 = id + 文本 + 向量 + 元数据（现在只放 date）。

    db = VectorDB()
    db.add("fact", "fact-abc", "用户叫小明", vector, {"date": "2026-09-19"})
    db.get("fact", "fact-abc")          db.list("fact")     db.count("fact")
    db.search("fact", query_vector, k=4)
    db.update("fact", "fact-abc", text="...", embedding=[...], metadata={...})
    db.delete("fact", "fact-abc")       db.clear("fact")

文本 → 向量是 embedding.py 的事，这一层只收向量。所以换模型、换维度，
db.py 一个字都不用改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import chromadb

DB = Path(__file__).resolve().parent / ".chroma"     # 删掉能重建（md 才是源）
COLLECTIONS = ("fact", "episode")


class VectorDB:
    def __init__(self, path: Path = DB) -> None:
        self.path = Path(path)
        # 本地优先：关掉匿名遥测
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=chromadb.Settings(anonymized_telemetry=False))
        self._collections = {name: self.client.get_or_create_collection(name)
                             for name in COLLECTIONS}

    def _collection(self, kind: str):
        if kind not in self._collections:
            raise ValueError(f"未知集合 {kind!r}，只有 {', '.join(COLLECTIONS)}")
        return self._collections[kind]

    # ---- Create / Update：upsert，id 在就是更新、不在就是新建 -------------
    def add(self, kind: str, record_id: str, text: str,
            embedding: list[float], metadata: dict | None = None) -> None:
        self.add_many(kind, [(record_id, text, embedding, metadata)])

    def add_many(self, kind: str,
                 records: Iterable[tuple[str, str, list[float], dict | None]]) -> int:
        """批量写。records = [(id, text, embedding, metadata), ...]

        Chroma 不接受空的 metadata dict，所以一条都没有元数据时干脆不传这个字段；
        一批里有的有、有的没有则直接报错（不然会静默丢元数据）。
        """
        rows = list(records)
        if not rows:
            return 0
        metadatas = [row[3] or {} for row in rows]
        payload: dict[str, list] = {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "embeddings": [row[2] for row in rows],
        }
        if any(metadatas):
            if not all(metadatas):
                raise ValueError("同一批里 metadata 要么每条都有、要么都没有，Chroma 不收空 dict")
            payload["metadatas"] = metadatas
        self._collection(kind).upsert(**payload)
        return len(rows)

    # ---- Read -----------------------------------------------------------
    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        found = self._collection(kind).get(ids=[record_id])
        if not found["ids"]:
            return None
        return _to_record(found["ids"][0], found["documents"][0], found["metadatas"][0])

    def list(self, kind: str) -> list[dict[str, Any]]:
        found = self._collection(kind).get()
        return [_to_record(i, d, m) for i, d, m in
                zip(found["ids"], found["documents"], found["metadatas"])]

    def count(self, kind: str) -> int:
        return self._collection(kind).count()

    def search(self, kind: str, embedding: list[float], k: int = 4) -> list[dict[str, Any]]:
        collection = self._collection(kind)
        total = collection.count()
        if not total:
            return []
        found = collection.query(query_embeddings=[embedding], n_results=min(k, total))
        return [_to_record(i, d, m, dist) for i, d, m, dist in zip(
            found["ids"][0], found["documents"][0],
            found["metadatas"][0], found["distances"][0])]

    # ---- Update：只改传进来的字段 -----------------------------------------
    def update(self, kind: str, record_id: str, *, text: str | None = None,
               embedding: list[float] | None = None,
               metadata: dict | None = None) -> bool:
        """返回是否改到了东西。

        改 text 必须同时给 embedding —— 否则 Chroma 会拿它自带的英文模型去
        现算，那就是我们特意绕开的那 79MB 下载。
        """
        if self.get(kind, record_id) is None:
            return False
        if text is not None and embedding is None:
            raise ValueError("改 text 必须同时给新的 embedding，否则会触发 Chroma 自带模型")
        if metadata is not None and not metadata:
            raise ValueError("Chroma 不收空 metadata；要清空元数据只能删了重写")
        patch: dict[str, list] = {}
        if text is not None:
            patch["documents"] = [text]
        if embedding is not None:
            patch["embeddings"] = [embedding]
        if metadata is not None:
            patch["metadatas"] = [metadata]
        if patch:
            self._collection(kind).update(ids=[record_id], **patch)
        return True

    # ---- Delete ---------------------------------------------------------
    def delete(self, kind: str, record_id: str) -> bool:
        if self.get(kind, record_id) is None:
            return False
        self._collection(kind).delete(ids=[record_id])
        return True

    def clear(self, kind: str) -> int:
        """清空一个集合，返回删掉几条。"""
        collection = self._collection(kind)
        total = collection.count()
        if total:
            collection.delete(ids=collection.get()["ids"])
        return total


def _to_record(record_id: str, text: str, metadata: dict | None,
               distance: float | None = None) -> dict[str, Any]:
    record = {"id": record_id, "text": text, "metadata": metadata or {}}
    if distance is not None:
        record["distance"] = distance
    return record
