"""把中文文本变成向量 —— BAAI/bge-small-zh-v1.5（智源），本地 ONNX，不装 torch。

依赖：onnxruntime、tokenizers、numpy（装 chromadb 时已经带进来了）
模型：.models/bge-small-zh-v1.5/{tokenizer.json, onnx/model.onnx}
验证于 onnxruntime 1.30.0 / tokenizers 0.23.2（2026-09-19）

BGE 有两个必须照做的细节，漏了掉点很厉害：
  1. 池化用 **CLS**（取第 0 个 token 的向量），不是平均池化
  2. **查询**要加指令前缀，文档不加 —— 这是它训练时的约定

所以 embed() 有 is_query 这个开关：存记忆时别开，检索时开。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(os.getenv(
    "MINIWAKU_EMBED_MODEL_DIR",
    Path(__file__).resolve().parent / ".models" / "bge-small-zh-v1.5"))
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_tokenizer: Tokenizer | None = None
_session: ort.InferenceSession | None = None


def _load() -> tuple[Tokenizer, ort.InferenceSession]:
    """懒加载：第一次用的时候才读模型（进程内只读一次）。"""
    global _tokenizer, _session
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    if _session is None:
        options = ort.SessionOptions()
        options.log_severity_level = 3          # 关掉 onnxruntime 的 INFO/WARNING 噪声
        _session = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"),
                                        sess_options=options,
                                        providers=["CPUExecutionProvider"])
    return _tokenizer, _session


def embed(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """文本 → 512 维单位向量（已 L2 归一化，所以余弦相似度就是点积）。"""
    tokenizer, session = _load()
    if is_query:
        texts = [QUERY_PREFIX + text for text in texts]

    # encode_batch 只切 token、不加 padding，所以下面自己补齐到同一长度
    encodings = tokenizer.encode_batch(texts)
    width = max(len(enc.ids) for enc in encodings)
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    input_ids = np.full((len(texts), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(texts), width), dtype=np.int64)
    for row, enc in enumerate(encodings):
        size = len(enc.ids)
        input_ids[row, :size] = enc.ids
        attention[row, :size] = enc.attention_mask

    hidden = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": np.zeros_like(input_ids),
    })[0]

    vectors = hidden[:, 0]                                    # CLS 池化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-9)).tolist()
