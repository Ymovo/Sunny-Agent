# loop —— 最小 agent loop

能聊天就够了。这个目录只负责"转起来"：读一行、问一次、说一句。

```text
chat.py   循环：加一句 → 问模型 → 记一句
llm.py    openai SDK 的薄包装（DeepSeek = OpenAI 兼容端点）
```

会话状态不在这里：历史、滑动窗口、system prompt、落盘都在 `../session/`，
长期事实/情节在 `../memory/`。

## 依赖

```bash
pip install openai python-dotenv onnxruntime tokenizers numpy
```

（`onnxruntime` / `tokenizers` / `numpy` 用来跑本地的中文 embedding 模型）

## 跑

```bash
python -m mini_waku_agent.loop.chat                        # 交互式，/quit 退出
python -m mini_waku_agent.loop.chat "你好，介绍一下你自己"
```

key 写在 `mini_waku_agent/.env`：

```text
DEEPSEEK_API_KEY=sk-...
```

## 一轮的顺序

```python
reply = run_turn(llm, session, tools, user)   # system + 窗口 + 工具回环
session.add_exchange(user, reply)             # 会话历史 + 落盘
memory.log_exchange(user, reply)              # 也进 chat_log，供蒸馏
```

`chat.py` 里没有 system prompt —— 它由 `Session.build_system()` 拼：
「你运行在哪个模型上」+ 人格 + 这次检索到的长期记忆。想改人格，去改
`session/soul.md`（首次运行会自动用 `DEFAULT_SOUL` 生成），不用动循环。

## 为什么这么写

循环只认 `llm.call(messages, tools) -> (content, tool_calls, finish_reason)`。
client、连接复用、重试、超时、错误映射全部交给 SDK，关在 `llm.py` 里 ——
换模型或换 provider 只动那一个文件的 `BASE_URL` / `MODEL`。
