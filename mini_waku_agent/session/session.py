"""会话 —— 管一场对话：历史、滑动窗口、system prompt、落盘、多会话。

会话就是一个 id + 一个 jsonl 文件，一行一轮：

    session/sessions/default.jsonl    {"ts": "...", "user": "...", "reply": "..."}

    s = Session(memory=Memory(), model="deepseek-v4-pro")   # memory 可以不给
    reply = llm.chat(s.build_messages("用户问的话"))   # system + 窗口 + 这一句
    s.add_exchange("用户问的话", reply)                # 记账 + 落盘

    s.new("work")            # 新开一个会话
    s.switch("default")      # 切回旧会话（历史从磁盘装回来）
    s.sessions()             # 有哪些会话

目录、人格文件、窗口轮数、检索条数都在 ../setting.py。

和 memory 的分工：

    session   这一段对话说过什么 —— 工作记忆，长了会被窗口截掉
    memory    跨会话的长期事实 / 情节 —— 不丢，靠检索取回

所以两者是互补的：会话负责"刚才聊到哪"，memory 负责"我一直记得什么"。
memory 是鸭子类型，只要有 search(query, k=4) -> list[dict] 就能用，
这一层不 import 任何存储实现。

用法（看有哪些会话 / 看某个会话）：
    python session.py
    python session.py show default
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mini_waku_agent/
from setting import (  # noqa: E402
    DEFAULT_SOUL, HISTORY_TURNS, SEARCH_K, SESSIONS_DIR, SOUL_FILE,
)


def _safe(session_id: str) -> str:
    """会话 id 会变成文件名，先洗一遍（别让人用 ../ 跑到别的目录去）。"""
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", session_id).strip("_") or "default"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_soul() -> str:
    """读 soul.md 作为人格。文件不存在就创建，写入 DEFAULT_SOUL。

    每次 build_system 都重新读一遍 —— 直接编辑 soul.md 就能改人格，
    不用重启程序。
    """
    if not SOUL_FILE.exists():
        SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
    return SOUL_FILE.read_text(encoding="utf-8").strip()


class Session:
    def __init__(self, memory=None, model: str = "", session_id: str = "default",
                 turns: int = HISTORY_TURNS, max_distance: float | None = None,
                 path: Path = SESSIONS_DIR) -> None:
        self.memory = memory
        self.model = model
        self.turns = turns
        # 默认不过滤。实测同一个意图换个说法，距离能从 0.97 掉到 1.31
        # （"用户叫什么名字" vs "我叫什么名字？"），任何固定阈值都会误杀。
        # 想收紧再自己传一个值，别指望有一个通用数字。
        self.max_distance = max_distance
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.session_id = ""
        self.history: list[dict] = []
        self.switch(session_id)

    # ---- 会话生命周期 ----------------------------------------------------
    def file(self, session_id: str | None = None) -> Path:
        return self.path / f"{_safe(session_id or self.session_id)}.jsonl"

    def switch(self, session_id: str) -> None:
        """切到某个会话，历史从磁盘装回来。不存在就是新会话。"""
        self.session_id = _safe(session_id)
        self.history = self._load()

    def new(self, session_id: str) -> None:
        """新开一个。名字被占了就报错 —— 免得把旧对话悄悄覆盖了。"""
        if self.file(session_id).exists():
            raise ValueError(f"会话 {session_id!r} 已存在，换个名字或直接 switch()")
        self.switch(session_id)

    def sessions(self) -> list[dict]:
        found = []
        for file in sorted(self.path.glob("*.jsonl")):
            rows = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            found.append({"id": file.stem, "turns": len(rows),
                          "updated": datetime.fromtimestamp(file.stat().st_mtime)
                          .strftime("%Y-%m-%d %H:%M")})
        return found

    # ---- 拼这一轮要发的东西 ----------------------------------------------
    def build_system(self, user_message: str) -> str:
        """system prompt：我是谁 + 怎么说话 + 这次相关的长期记忆。"""
        parts = []
        if self.model:
            parts.append(f"你运行在 {self.model} 上。")
        parts.append(load_soul())
        if self.memory is not None:
            hits = self.memory.search(user_message, k=SEARCH_K)
            if self.max_distance is not None:
                hits = [hit for hit in hits
                        if hit.get("distance", 0.0) <= self.max_distance]
            if hits:
                lines = "\n".join(f"- {hit['text']}" for hit in hits)
                # 措辞要肯定。写成"无关就忽略"这种留退路的话，模型会整段无视 —— 
                # 实测把名字喂给它，它照样回答"我不知道你的名字"。
                parts.append("下面这些是你自己长期记忆里的事实，可以直接采信并用来回答：\n" + lines)
        return "\n\n".join(parts)

    def build_messages(self, user_message: str, *, with_system: bool = True) -> list[dict]:
        """这一轮真正发给模型的东西：system + 最近 turns 轮 + 这一句。

        默认把 system 放进第一条 —— 我们用的是 OpenAI 风格的接口，system 是
        messages 里的一条，不是单独的参数。忘了带它，模型就看不到记忆，
        于是明明记得你叫小明，还是会回答"我不知道你的名字"。
        Anthropic 那种把 system 单独传的接口就 with_system=False。
        """
        messages: list[dict] = []
        if with_system:
            messages.append({"role": "system", "content": self.build_system(user_message)})
        return messages + self.history[-self.turns * 2:] + [
            {"role": "user", "content": user_message}]

    def add_exchange(self, user_message: str, reply: str) -> None:
        """记一轮。内存一份、磁盘一份 —— 内存的会被窗口截掉，磁盘的不会。"""
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        with self.file().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _now(), "user": user_message, "reply": reply},
                                    ensure_ascii=False) + "\n")

    # ---- 落盘 ------------------------------------------------------------
    def _load(self) -> list[dict]:
        path = self.file()
        if not path.exists():
            return []
        history = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue              # 上次写了一半的行，跳过就好，别让它拖垮启动
            history.append({"role": "user", "content": row.get("user", "")})
            history.append({"role": "assistant", "content": row.get("reply", "")})
        return history


def main() -> None:
    session = Session()
    args = sys.argv[1:]
    if args and args[0] == "show":
        target = args[1] if len(args) > 1 else "default"
        session.switch(target)
        print(f"=== {session.session_id}（{len(session.history) // 2} 轮）")
        for i in range(0, len(session.history), 2):
            print(f"you   › {session.history[i]['content']}")
            print(f"agent › {session.history[i + 1]['content']}\n")
        return
    print(f"会话目录：{session.path}")
    for row in session.sessions():
        print(f"  {row['id']:<20} {row['turns']:>3} 轮   {row['updated']}")


if __name__ == "__main__":
    main()
