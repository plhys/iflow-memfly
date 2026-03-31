# iFlow MemFly — 记忆飞轮

**记忆飞轮** — iFlow 的记忆觉醒项目。

iFlow MemFly 提出并实现了「记忆飞轮」范式：记忆不是被检索的外部资源，而是在思考发生之前就已存在的认知状态。一个守护进程自动从对话中提取、分层、存储记忆，并在每次新会话启动时将其注入上下文 — AI 睁开眼睛的瞬间，就已经知道自己是谁、昨天做了什么、下一步该干什么。

> 1 进程，2 线程，64MB 内存。46 个测试全部通过。4360 行代码。

## 为什么需要记忆飞轮

### 两种范式的根本分歧

当前所有 AI 记忆系统都面临同一个问题：如何让 AI 在新会话中「记得」之前发生过什么。对这个问题的回答，分裂出两条截然不同的路径：

**检索范式（Retrieval Paradigm）**

mem0、Zep、各类 RAG 系统走的都是这条路：记忆是外部资源，需要时去查。

```
用户提问 → AI 意识到需要记忆 → 调用工具搜索 → 获取结果 → 继续回答
```

这就像一个图书馆：你得先知道要找什么书，走过去，翻开，才能获得信息。问题是 — AI 怎么知道自己该找什么？在新会话的第一个 token 生成之前，它对之前的一切一无所知。它甚至不知道自己应该去搜索。

**预装范式（Preloaded Context Paradigm）**

iFlow MemFly 走的是另一条路：记忆是意识的一部分，在思考发生之前就已存在。

```
守护进程自动注入记忆 → 新会话启动 → AI 已经拥有完整认知状态 → 直接开始工作
```

这就像你早上醒来时脑子里已有的东西：你不需要「搜索」自己的名字，不需要「查询」昨天做了什么 — 这些信息就在那里，构成了你的意识起点。

### 「飞轮」隐喻的三重含义

为什么叫「飞轮」而不叫「数据库」或「知识库」？

1. **自转**：守护进程自动运行，无需任何人推动。对话发生 → 记忆自动产生 → 自动注入下次会话。没有手动步骤。
2. **惯性**：对话越多 → 记忆越丰富 → AI 表现越好 → 用户越愿意深入对话。这是一个正反馈飞轮，转得越久势能越大。
3. **动力源**：记忆不是被动的仓库，而是驱动 AI 认知的主动引擎。它决定了 AI 「是谁」「知道什么」「接下来该做什么」。

### 三个理论层次

**上下文即身份（Context as Identity）**

AI 没有持久的自我。它的「自我」完全由上下文窗口中的内容决定。同一个模型，注入不同的记忆，就是不同的「人」。iFlow MemFly 自动从历史交互中构建这个身份 — 这不是功能，这是存在论。

**状态恢复 vs 信息检索（State Restoration vs Information Recall）**

```
信息检索：f(query) → relevant_facts     # 数据库操作
状态恢复：f(last_session) → cognitive_state  # 意识重建
```

检索范式做的是前者：给一个查询，返回相关事实。预装范式做的是后者：从上次会话的完整状态中重建认知起点。区别在于 — 信息检索回答「我知道什么」，状态恢复回答「我是谁、我在做什么、我做到哪了」。

**记忆生命周期流水线（Memory Lifecycle Pipeline）**

```
感知 → 提取 → 分层 → 存储 → 衰减 → 注入 → 消费 → 新对话（闭环）
```

记忆不是静态数据，而是有生命周期的活体。它从对话中被感知，经 LLM 提取和分层，存入数据库，随时间衰减，被注入新会话，被 AI 消费，然后在新对话中产生新的记忆 — 完成闭环。

## 与其他记忆系统的对比

| 维度 | iFlow MemFly | mem0 | Zep | Letta (MemGPT) | OpenViking |
|------|-------------|------|-----|-----------------|------------|
| **范式** | 预装注入 | 检索调用 | 检索调用 | 检索调用 | 检索调用 |
| **新会话启动时** | 记忆已在上下文中 | 需主动搜索 | 需主动搜索 | 需主动搜索 | 需主动搜索 |
| **AI 需要做什么** | 零操作 | 调用 tool | 调用 API | 调用 tool | 调用 tool |
| **第一个 token 的认知** | 完整状态 | 空白 | 空白 | 空白 | 空白 |
| **记忆产生方式** | 守护进程自动 | 显式调用 | 自动提取 | 显式调用 | 显式调用 |
| **部署复杂度** | 单进程 64MB | 需要服务端 | 需要服务端 | 需要服务端 | 需要服务端 |
| **向量搜索** | sqlite-vec | Qdrant | 内置 | 内置 | 内置 |
| **意识连续性** | 原生支持 | 不支持 | 不支持 | 部分支持 | 不支持 |

核心差异只有一个：**别人的 AI 醒来时是空白的，我们的 AI 醒来时已经知道自己是谁。**

## 记忆飞轮的七种产物

守护进程从每批对话中自动生成七种记忆产物，无需任何手动操作：

| # | 产物 | 方式 | 存储位置 | 用途 |
|---|------|------|----------|------|
| 1 | **L3 原始记录** | 纯本地清洗，不调用 LLM | 日期 .md 文件 | 完整对话兜底回溯 |
| 2 | **L1 索引短句** | LLM 生成 ≤40 字 | index.md | 快速定位「那天聊了什么」 |
| 3 | **L2 结构化摘要** | LLM 生成 Markdown | 日期 .md 文件 | 还原决策过程和技术方案 |
| 4 | **分类记忆** | LLM 提取 6 类事实 | SQLite + 向量 | 身份/偏好/纠正/实体/事件/经验 |
| 5 | **氛围快照** | LLM 第一人称回忆 | SQLite | 对话情感和隐性共识 |
| 6 | **状态快照** | LLM 结构化检查点 | SQLite | 目标/进度/决策/下一步/关键上下文 |
| 7 | **每日回顾** | LLM 从 L3 事件提取 | AGENTS.md | 第一人称工作日志 |

**关键设计**：L3 原始记录在所有 LLM 调用之前写入。即使 LLM 失败，原始数据已持久化，L1/L2 可在后续处理中补上。状态推进（commit_progress）仅在 L3 成功后执行。

### 六类分类记忆

| 分类 | 说明 | 示例 | 归档策略 |
|------|------|------|----------|
| **identity** | 用户/AI 身份信息 | "用户名是 alice" | 永不归档 |
| **preference** | 用户偏好和习惯 | "不要 emoji" | 永不归档 |
| **correction** | 用户纠正 AI 的错误 | "不对，X 不是 Y" | 永不归档 |
| **entity** | 有名字的事物及属性 | "Redis 端口是 6379" | 按热度衰减 |
| **event** | 发生过的事和决策 | "放弃方案 A 改用方案 B" | 按热度衰减 |
| **insight** | 踩坑教训和注意事项 | "服务器重启后符号链接会丢" | 按热度衰减 |

### 热度评分与遗忘机制

```
hotness = sigmoid(log1p(access_count)) * exp(-0.693 * age_days / 14)
```

- 访问越多越热：被检索命中的记忆自动 +1 访问计数
- 14 天半衰期：两周不被访问的记忆热度减半
- 自动归档：热度低于阈值且超过 7 天的记忆被归档（identity/preference/correction 永不归档）
- 注入上限 80 条：identity 和 preference 优先，其余按热度排名

## AGENTS.md 自动注入

这是记忆飞轮的核心输出 — 守护进程自动将记忆写入 AGENTS.md，AI 在新会话启动时自动加载：

```
AGENTS.md 注入内容（自动生成，按顺序）：
├── 开机自检指令        守护进程健康检查 + search_memory 可用性验证
├── 时间锚点           当前时间 + 上次对话距今时长
├── 最近对话索引        最近 50 条 L1 索引短句
├── 上次工作回顾        第一人称每日工作日志
├── 上次对话记忆        第一人称氛围快照
├── 状态快照           结构化检查点（目标/进度/决策/下一步）
└── 分类记忆           按类别分组的高热度事实（≤80 条）
```

AI 不需要调用任何工具就能获得这些信息。它们在第一个 token 生成之前就已经存在于上下文中。

## 架构

```
SessionWatcher (10s 轮询)
    │
    ▼
MemoryDaemon._on_session_change()
    │
    ├─ Indexer.get_new_messages()         增量读取新消息
    │      ├─ SessionParser.parse_acp()   ACP session (JSON)
    │      └─ SessionParser.parse_cli()   CLI session (JSONL)
    │
    ├─ Indexer.write_cleaned_messages()   → L3 原始记录
    ├─ Indexer.commit_progress()          ← 仅在 L3 成功后推进
    │
    ├─ Summarizer.generate_index_line()   → L1 索引短句
    ├─ Summarizer.generate_summary()      → L2 结构化摘要
    ├─ Summarizer.generate_classified()   → SQLite 分类记忆
    ├─ Summarizer.generate_atmosphere()   → SQLite 氛围快照
    ├─ Summarizer.generate_state()        → SQLite 状态快照
    └─ Summarizer.generate_daily_recap()  → AGENTS.md 每日回顾
           │
           ▼
       MemoryInjector.inject()            → AGENTS.md 自动更新
```

## 项目结构

目录结构本身体现了记忆生命周期：`core/`（产生）→ `store/`（存储）→ `serve/`（使用）。

```
iflow_memory/                    4360 行
├── __init__.py                  版本号
├── __main__.py          530行   CLI 入口 + 守护进程启动
├── config.py            149行   配置管理
├── guard.py             102行   健康检查 + 错误边界
│
├── core/                        记忆的产生
│   ├── daemon.py        348行   守护进程主循环 + 消息管道
│   ├── watcher.py        96行   文件变更监听
│   ├── indexer.py       305行   L1/L3 索引写入
│   └── summarizer.py    547行   LLM 摘要生成（7 个 prompt）
│
├── store/                       记忆的存储
│   ├── db.py            870行   SQLite + FTS5 + sqlite-vec
│   └── embed.py         184行   向量嵌入客户端
│
└── serve/                       记忆的使用
    ├── injector.py      446行   AGENTS.md 注入器
    ├── mcp_server.py    258行   MCP stdio 接口
    ├── web.py           519行   Web 管理 + MCP HTTP
    └── ui.html                  前端页面
```

## 存储能力

- **SQLite + FTS5 全文检索**：中文分词，支持关键词 + 分类组合查询
- **sqlite-vec 向量索引**：1024 维 bge-m3 嵌入，语义相似度搜索
- **热度评分**：访问频率 + 时间衰减，自动排序
- **冷记忆归档**：低热度记忆自动归档，不污染活跃集
- **写入去重**：相同 category+text 不重复插入

## 服务能力

- **AGENTS.md 自动注入**：开机自检、时间锚点、50 条 L1 索引、工作回顾、氛围快照、状态快照、分类记忆
- **MCP HTTP 端点**：`search_memory`、`save_memory`、`get_recent_context`
- **MCP stdio 模式**：兼容 iFlow CLI 原生 MCP 注册
- **Web 管理后台**：记忆浏览、搜索、统计、功能开关

## 安装

```bash
pip install .

# 如需 Web 管理后台
pip install ".[web]"
```

依赖：Python >= 3.10，httpx >= 0.25.0

## 使用

```bash
# 启动守护进程
iflow-memory start

# 查看运行状态
iflow-memory status

# 搜索记忆
iflow-memory query "部署方案"
iflow-memory query "SSH" -c entity -n 5

# 手动触发注入
iflow-memory inject

# 查看功能开关
iflow-memory features list
iflow-memory features enable atmosphere
iflow-memory features disable summary

# 模型管理
iflow-memory model list
iflow-memory model use memory-worker

# 切换处理策略
iflow-memory strategy interval --interval 300

# 补跑历史数据
iflow-memory backfill

# 启动 Web 管理后台
iflow-memory web --port 8765
```

## MCP 集成

iFlow MemFly 同时支持 MCP stdio 和 HTTP 两种模式。

**stdio 模式**（注册到 iFlow CLI）：

```bash
iflow mcp add-json iflow-memory \
  '{"command":"python3","args":["-m","iflow_memory.mcp_server"],"type":"stdio","trust":true}' \
  --scope user
```

**HTTP 模式**（守护进程内嵌，启动即可用）：

```
MCP endpoint: http://127.0.0.1:18765/mcp
```

提供三个 tool：
- `search_memory` — 搜索记忆库（关键词 + 分类 + 向量）
- `save_memory` — 手动存入记忆
- `get_recent_context` — 获取最近对话索引摘要

## 配置

配置文件位于 `~/.iflow-memory/config.json`，数据存储在 `~/.iflow-memory/data/`：

```json
{
  "memory_dir": "~/.iflow-memory/data",
  "strategy": "interval",
  "interval_seconds": 300,
  "model_mode": "custom",
  "active_preset": "memory-worker",
  "agents_md_paths": ["~/.iflow/AGENTS.md"],
  "model_presets": {
    "memory-worker": {
      "base_url": "https://your-api-endpoint/v1",
      "model": "your-model-name"
    }
  }
}
```

**处理策略**：
- `interval`：每 N 秒处理一批积累的消息（默认 300 秒，推荐）
- `on_compress`：每次 session 变化立即处理

**功能开关**：5 个 LLM 功能可独立开关 — `index_line` / `summary` / `classify` / `atmosphere` / `daily_recap`。支持命令行、Web API、首次安装引导三种控制方式。

## 数据存储

```
~/.iflow-memory/data/
├── memories.db              SQLite + FTS5 + sqlite-vec 数据库
├── index.md                 L1 索引（按日期分段）
├── 2026-03-29.md            L3 原始记录 + L2 结构化摘要
├── .indexer-state.json      增量处理状态
└── iflow-memory.pid         守护进程 PID 文件
```

## 测试

```bash
python -m pytest tests/ -v
```

46 个测试用例，覆盖：
- 热度评分计算（边界值、半衰期、异常输入）
- SQLite CRUD（增删查归档、FTS5 搜索、访问计数）
- Session 解析（ACP/CLI 格式、系统标签清洗、tool call 过滤）
- 状态管理（增量处理、失败重试、跨实例持久化）
- JSON 解析容错（代码块、尾逗号、截断修复、正则兜底）

## Changelog

### v1.3.2

**项目正式定名**
- 正式名称确定为 **iFlow MemFly**（中文：iFlow 记忆飞轮）
- 项目定位更新为「记忆飞轮 — iFlow 的记忆觉醒项目」
- 统一更新所有源码、Web UI、MCP server、CLI 中的显示名称
- Python 包名 `iflow-memory`、配置目录 `~/.iflow-memory/` 保持不变，无需迁移

### v1.3.1

**新功能**
- `web_port` 可在配置文件中自定义（默认 18765），不再硬编码
- 对话空闲 60 秒自动 flush pending 消息，防止对话结束后记忆丢失
- 新增 `save_memory` MCP tool 和 `POST /api/flush` Web API，支持手动触发记忆保存

**Bug 修复**
- 修复 features API 缺少 `state_snapshot`，导致无法通过 Web 开关状态快照
- 修复 `_call_llm` 中 API 返回非标准格式（KeyError/IndexError）不会重试的问题
- 修复 `MemoryStore.close()` 不幂等，重复调用会抛异常
- 修复状态快照生成时 LLM 返回 list 类型字段导致 `'list' object has no attribute 'strip'` 崩溃
- 修复 Watcher 因符号链接导致同一 session 文件被双重处理
- 修复注入模板中 `/tmp` 路径硬编码，改为 `tempfile.gettempdir()` 动态获取

**改进**
- `import time` 移至文件顶部，消除函数内重复导入
- `hybrid_search` 中变量命名统一，提升可读性
- 分类记忆提取 prompt 优化：insight 要求包含因果关系，过滤口头语和操作碎片
- 最短记忆长度阈值从 4 提升到 12，减少垃圾记忆
- Jaccard 去重阈值从 0.85 降到 0.80，更积极清理近似重复

### v1.3.0

**重构**
- 目录结构重组为 `core/`（记忆产生）、`store/`（记忆存储）、`serve/`（记忆使用）三层，体现记忆生命周期
- 代码审查清理，统一风格和命名

**新功能**
- **状态快照（State Snapshots）**：每批消息处理后生成结构化检查点 — 目标/进度/决策/下一步/关键上下文，注入 AGENTS.md 供新会话恢复工作状态

**Bug 修复**
- 修复氛围快照中 `**` 标记导致内容被 Markdown 解析截断
- 修复 `add_atmosphere` 默认跳过同 session_id 导致新快照无法覆盖旧快照，改为 UPDATE 策略
- 修复 `created_at` 使用 UTC 时间但本地为 UTC+8 导致相对时间计算错误
- 修复时区硬编码问题，改为自动读取系统时区
- 修复 LLM 处理日志时丢失原始时间戳导致时间线错乱

### v1.2.0

**新功能**
- **对话续接机制**：自动生成上次对话的工作回顾，注入 AGENTS.md。支持跨天续接（往回搜最多 7 天），标题带日期和距今天数。每天只保留一份，自动替换不累加。
- **对话氛围快照**：每批消息处理后，LLM 自动提取 5 个维度的软性记忆 — 情感状态、对话节奏、隐性共识、里程碑时刻、未了事项。存入 SQLite `atmosphere_snapshots` 表，注入时附带低权重引导语。
- **功能开关系统**：5 个 LLM 功能可独立开关，支持命令行、Web API、首次安装引导三种控制方式，4 种预设方案（全开/精简/最小/自定义）。
- **Watcher 多工作目录**：监听 `~/.iflow/projects/` 下所有子目录，不管用户在哪个工作目录启动 iFlow CLI，session 文件都能被捕获。

**Bug 修复**
- 修复 `cli_sessions_dir` 硬编码导致其他工作目录的 session 无法被捕获
- 修复 `get_latest_atmosphere` 返回类型不匹配

**数据库**
- SQLite schema 升级到 v2，新增 `atmosphere_snapshots` 表（自动迁移）

### v1.1.0

**新功能**
- 新增 `correction` 分类：识别用户纠正 AI 错误的对话，永不归档，始终注入
- 新增 MCP stdio server，暴露 `search_memory` 和 `get_recent_context` 两个 tool
- AGENTS.md 注入时附带最近对话索引
- FTS5 搜索无结果时自动 fallback 到 LIKE 模糊匹配
- `agents_md_paths` 可在 config.json 中配置

**Bug 修复**
- 修复 interval 模式下 `_pending` 消息重复积累
- 修复 FTS5 查询特殊字符导致 SQL 报错
- 修复 `_update_agents_md` 中 `re.sub` 反斜杠转义崩溃
- 修复 follow 模式读到错误的 apiKey
- 修复首次运行判断逻辑
- 写入 SQLite 前检查去重

**改进**
- 默认 `memory_dir` 改为 `~/.iflow-memory/data/`，使用独立数据目录
- Web 后台默认监听 `127.0.0.1`
- 命令行输出抑制非 verbose 模式下的日志
- 版本号统一从 `__version__` 读取

### v1.0.0

首次发布。三层记忆架构 + 6 分类 SQLite 存储 + FTS5 全文检索 + AGENTS.md 自动注入 + Web 管理后台。

## License

Copyright (c) 2026 李不是狼

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for details.
