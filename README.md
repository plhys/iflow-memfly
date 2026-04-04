# iFlow MemFly — 记忆飞轮

> 别人的 AI 醒来时是空白的，我们的 AI 醒来时已经知道自己是谁。

给 AI CLI 助手加上持久记忆的守护进程。自动从对话中提取记忆，新会话启动时注入上下文，让 AI 跨会话保持连续的认知状态。

1 进程，64MB 内存，Python 3.10+。无需外部数据库。

---

## 最简上手方式

把这个仓库地址发给你的 AI 助手（iFlow CLI / Claude Code / Cursor 等），告诉它：

> **"这是 iFlow MemFly 记忆飞轮项目，帮我部署这个项目并进行评测。"**
>
> 仓库地址：`https://github.com/plhys/iflow-memfly`

AI 助手会自己 clone、安装、配置、启动。你只需要等它跑完就行。

---

## 功能全貌

### 记忆提取与分类

守护进程自动监听 AI 的 session 文件，检测到新对话后，通过 LLM 提取 **6 类结构化记忆**：

| 分类 | 说明 | 示例 |
|------|------|------|
| **身份 (identity)** | 用户和 AI 的身份信息 | "用户的 GitHub 用户名是 plhys" |
| **偏好 (preference)** | 用户的习惯和偏好 | "用户偏好用中文交流" |
| **纠正 (correction)** | 用户纠正 AI 的错误 | "用户指出 AI 搞混了两个项目" |
| **实体 (entity)** | 项目、服务、工具等实体知识 | "A 计划运行在端口 18788" |
| **事件 (event)** | 发生过的重要事件 | "v1.3.4 已推送到 GitHub" |
| **经验 (insight)** | 踩坑教训和最佳实践 | "daemon 重启后需要重新注入记忆" |

每条记忆自动去重，不会重复存储相同内容。

### 三层记忆架构

```
L1 索引层    index.md          每条对话一句话摘要（≤40 字），快速定位
L2 摘要层    2026-04-02.md     LLM 生成的结构化摘要，保留关键细节
L3 原始层    2026-04-02.md     完整的对话原文，按时间戳归档
```

三层各司其职：L1 用于注入上下文（轻量），L2 用于回顾和检索，L3 用于追溯原始对话。

### AGENTS.md 自动注入（预装范式）

这是记忆飞轮的核心机制。守护进程自动将以下信息写入 `AGENTS.md`：

- **分类记忆**：身份、偏好、纠正、实体、事件、经验
- **最近对话索引**：最近 10 条对话的时间和摘要
- **工作状态存档**：上次对话的目标、进度、决策、下一步
- **氛围快照**：上次对话结束时的情绪和互动氛围
- **每日简报**：当天记忆的总结概览
- **时间感知**：当前时间和上次对话的时间间隔

AI 启动新会话时，`AGENTS.md` 已经包含了所有这些信息。AI 不需要主动"回忆"，记忆已经在那里了。

### 对话续接

每次对话结束时，守护进程自动生成：

- **工作回顾**：这次对话做了什么、做到哪了、下一步是什么
- **氛围快照**：对话的情绪基调和互动风格
- **状态快照**：当前工作的技术状态（正在改的文件、遇到的问题等）

下次对话开始时，这些信息自动注入，AI 可以无缝接续上次的工作。

### 影子记录

AI 对话存在上下文压缩的问题——当对话太长时，早期的消息会被丢弃。守护进程在正常处理消息的同时，维护一份 **滚动 6 小时的影子副本**。当检测到 session 文件被重写（消息数突然减少）时，自动从影子记录中恢复未处理的消息。

不丢消息，不漏记忆。

### 知识图谱

记忆不是孤立的。每条新记忆写入后，系统自动通过 **向量相似度** 寻找相关的已有记忆，建立关联链接。向量搜索不可用时，自动降级为 **关键词匹配**。

你可以从任意一条记忆出发，沿着关联链接扩展，发现相关的知识网络。

### 每日简报

每天自动生成一份记忆简报，总结当天的对话要点和重要事件。简报由 LLM 生成，如果 LLM 不可用则自动降级为模板生成。简报会注入到 AGENTS.md，让 AI 在新会话中快速了解"昨天发生了什么"。

### 搜索与检索

两种搜索方式并存：

- **向量搜索**：基于语义相似度，用 sqlite-vec + embedding 实现，能找到意思相近但用词不同的记忆
- **全文检索**：基于关键词匹配，用 SQLite FTS5 实现，精确查找包含特定词语的记忆。FTS5 查询失败时自动降级为 LIKE 模糊匹配

### MCP 工具

通过 MCP 协议对外提供三个工具，AI 助手可以在对话中直接调用：

- `search_memory`：搜索历史记忆（支持关键词过滤和分类过滤）
- `save_memory`：手动保存一条记忆
- `get_recent_context`：获取最近对话的索引摘要

支持 **stdio** 和 **HTTP** 两种模式，兼容 iFlow CLI、Claude Code 等主流 AI 工具。

### Web 管理后台

内置 Web 界面，可以浏览、搜索、管理所有记忆。

### 数据安全

- **原子写入**：所有文件写入使用临时文件 + `os.replace()`，进程崩溃或磁盘满不会损坏文件
- **自动备份**：每次写入 AGENTS.md 前自动创建 `.bak` 备份
- **事务完整性**：数据库写入使用单一事务，不会出现半写入状态

---

## 设计理念

### 预装范式

AI 记忆系统有两条路径：一种是需要时去查（检索范式），另一种是在思考发生之前就已存在（预装范式）。

检索范式的流程是：用户提问 → AI 意识到需要记忆 → 调用工具搜索 → 获取结果 → 继续回答。问题在于，新会话的第一个 token 生成之前，AI 对之前的一切一无所知，它甚至不知道自己应该去搜索。

iFlow MemFly 走预装范式：守护进程自动将记忆写入上下文 → 新会话启动 → AI 已经拥有完整认知状态 → 直接开始工作。就像你早上醒来时脑子里已有的东西——你不需要「搜索」自己的名字，不需要「查询」昨天做了什么。

### 为什么叫「飞轮」

1. **自转**：守护进程自动运行，无需任何人推动。对话发生 → 记忆自动产生 → 自动注入下次会话。没有手动步骤。
2. **惯性**：对话越多 → 记忆越丰富 → AI 表现越好 → 用户越愿意深入对话。正反馈循环，转得越久势能越大。
3. **动力源**：记忆不是被动的仓库，而是驱动 AI 认知的主动引擎。它决定了 AI「是谁」「知道什么」「接下来该做什么」。

### 上下文即身份

AI 没有持久的自我。它的「自我」完全由上下文窗口中的内容决定。同一个模型，注入不同的记忆，就是不同的「人」。iFlow MemFly 自动从历史交互中构建这个身份——不是给 AI 贴标签，而是让它在每次醒来时，自然地成为「那个一直陪你工作的伙伴」。

### 状态恢复 vs 信息检索

```
信息检索：f(query) → relevant_facts     # 回答「我知道什么」
状态恢复：f(last_session) → cognitive_state  # 回答「我是谁、我在做什么、我做到哪了」
```

大多数记忆系统解决的是信息检索问题。iFlow MemFly 解决的是状态恢复问题——让 AI 不只是「知道一些事」，而是「回到上次离开的状态」。

### 记忆生命周期

```
感知 → 提取 → 分层 → 存储 → 关联 → 衰减 → 注入 → 消费 → 新对话（闭环）
```

记忆不是静态数据，而是有生命周期的活体。它从对话中被感知，经 LLM 提取和分层，存入数据库，与已有记忆建立关联，随时间衰减，被注入新会话，被 AI 消费，然后在新对话中产生新的记忆——完成闭环。

---

## 安装

```bash
pip install .

# 如需 Web 管理后台
pip install ".[web]"
```

依赖：Python >= 3.10，httpx >= 0.25.0

## 快速开始

```bash
# 启动守护进程（后台运行，自动监听 session 文件）
iflow-memory start

# 查看运行状态
iflow-memory status

# 搜索记忆
iflow-memory query "部署方案"
iflow-memory query "SSH" -c entity -n 5

# 手动触发注入
iflow-memory inject
```

## 功能开关

iFlow MemFly 的每个 LLM 功能都可以独立开关。不需要的功能关掉就行，省 token 也省算力。

### 可控功能列表

| 功能 | 标识 | 说明 | 默认 |
|------|------|------|------|
| 索引生成 | `index_line` | 每条对话生成一句话索引 | 开 |
| 结构化摘要 | `summary` | 生成 L2 层摘要 | 开 |
| 记忆分类 | `classify` | 提取 6 类结构化记忆 | 开 |
| 氛围快照 | `atmosphere` | 记录对话氛围和情绪 | 开 |
| 每日回顾 | `daily_recap` | 生成当日工作回顾 | 开 |
| 知识图谱 | `knowledge_graph` | 记忆自动建立关联 | 开 |
| 每日简报 | `daily_briefing` | 生成每日记忆简报 | 开 |

### 命令行操作

```bash
# 查看所有功能的开关状态
iflow-memory features list

# 关掉某个功能
iflow-memory features disable atmosphere

# 开启某个功能
iflow-memory features enable atmosphere
```

### 让 AI 自己管理

你不需要记住这些命令。直接告诉你的 AI 助手就行：

> "帮我关掉记忆飞轮的氛围快照功能"
>
> "把每日简报功能打开"
>
> "看看记忆飞轮现在哪些功能是开着的"

AI 助手会自己执行对应的命令。

---

## 更新

### 让 AI 帮你更新（推荐）

直接告诉你的 AI 助手：

> "帮我更新一下记忆飞轮"

AI 会自动执行以下步骤：
1. 进入项目目录
2. `git pull` 拉取最新代码
3. `pip install .` 重新安装
4. 重启守护进程

你什么都不用管。

### 手动更新

```bash
cd /path/to/iflow-memfly
git pull origin main
pip install .

# 重启守护进程
iflow-memory stop
iflow-memory start
```

---

## 配置

配置文件位于 `~/.iflow-memory/config.json`：

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
- `interval`（推荐）：每 N 秒处理一批消息，默认 300 秒
- `on_compress`：每次 session 变化立即处理

## MCP 集成

支持 stdio 和 HTTP 两种模式。

**stdio 模式**（注册到 iFlow CLI）：

```bash
iflow mcp add-json iflow-memory \
  '{"command":"python3","args":["-m","iflow_memory.mcp_server"],"type":"stdio","trust":true}' \
  --scope user
```

**HTTP 模式**（守护进程内嵌）：

```
MCP endpoint: http://127.0.0.1:18765/mcp
```

提供三个 tool：`search_memory`、`save_memory`、`get_recent_context`。

## 项目结构

```
iflow_memory/
├── __main__.py          CLI 入口 + 守护进程启动
├── config.py            配置管理
├── guard.py             健康检查
├── core/                记忆产生
│   ├── daemon.py        守护进程主循环
│   ├── watcher.py       文件变更监听
│   ├── indexer.py       索引写入 + 影子记录
│   ├── summarizer.py    LLM 摘要生成
│   └── briefing.py      每日简报生成
├── store/               记忆存储
│   ├── db.py            SQLite + FTS5 + sqlite-vec + 知识图谱
│   └── embed.py         向量嵌入
└── serve/               记忆使用
    ├── injector.py      AGENTS.md 注入
    ├── mcp_server.py    MCP stdio
    └── web.py           Web 管理后台
```

## 存储

```
~/.iflow-memory/data/
├── memories.db              SQLite 数据库（FTS5 全文检索 + sqlite-vec 向量 + 知识图谱）
├── index.md                 L1 索引
├── 2026-04-02.md            L3 原始记录 + L2 摘要
├── briefing-2026-04-02.md   每日简报
├── .shadow/                 影子记录（滚动 6 小时）
├── .indexer-state.json      增量处理状态
└── iflow-memory.pid         PID 文件
```

## 测试

```bash
python -m pytest tests/ -v
```

## Changelog

### v2.0.1

**Bug 修复**

- **LLM 重试队列**：修复 feature-flag 全关闭时仍入队的问题，限制最大重试 3 次；失败记忆不再静默丢弃，加入重试队列等待重试
- **Prompt Injection 防御**：增强注入检测，防止恶意记忆污染
- **delete_memory 工具**：补全 MCP delete_memory 工具实现
- **冲突检测**：新增记忆冲突检测机制，避免重复和矛盾记忆
- **提取质量增强**：优化 LLM 记忆提取 prompt，提升提取准确率
- **Embedding 向量去重**：新增向量去重逻辑，减少冗余存储
- **Embedding 补算**：新增 embedding 补算机制，确保历史记忆可被向量搜索
- **Hotness 半衰期按类别区分**：不同类别记忆使用不同的半衰期计算方式
- **动态注入优化**：优化上下文感知记忆注入逻辑
- **L3 对话原文可搜索**：L3 原文接入 FTS5 索引，支持全文检索

**搜索增强**

- FTS5 查询增加 OR fallback 策略
- 新增 `date_from` 参数支持日期过滤
- 优化 MCP tool 描述，提升可用性

**性能优化**

- Hotness 排序优化，注入内容精简（80→40 条，15→8 条）
- Summarizer 每批处理从 3 条提升到 5 条
- Correction/Identity 分类豁免 hotness 排序
- 搜索结果 limit 从 10 提升到 50

**代码重构**

- 统一 MCP tool 定义到 shared tools.py
- Source tracing：新增 source_file 和 source_line 字段，记录记忆来源
- save_memory MCP handler 修复 + source_file 回填

### v2.0.0

**新功能**

- **知识图谱**：记忆写入时自动建立关联网络（`memory_links` 表），支持向量相似度 + 关键词双路径匹配，提供关系查询、图扩展、全图导出等 6 个 API
- **每日简报**：维护周期自动生成当日记忆简报（LLM 生成 + 模板 fallback），注入 AGENTS.md 供新会话快速恢复上下文
- **LLM 深度整合（做梦）**：空闲时调用 LLM 对记忆进行合并、淘汰、升级操作，支持时间感知和类别敏感度
- **记忆分区（scope）**：新增 `scope` 列（schema v6），支持 global / private 分区，所有查询方法均支持 scope 过滤
- **密钥自动脱敏**：入库前自动检测并替换 hex 格式密钥，防止敏感信息泄露到记忆库

**Bug 修复**

- `store.add()` 返回 `tuple[int, bool]`，消除 `is_new` 检测时的双重全表扫描（性能瓶颈）
- `briefing.py` 改用公开方法 `get_memories_by_date()` / `get_state_snapshot_by_date()` 替代直接访问 `store._conn`
- `_create_links_by_keywords` 移除冗余 `commit()`
- `BriefingGenerator` 复用外部 `Summarizer` 实例，避免重复创建 httpx 客户端
- briefing + summarizer 文件写入改为原子操作（`tempfile` + `os.replace()`）
- `hex_secret` 正则增加 negative lookbehind，避免 git commit hash 误判
- `web.py` + `__main__.py` 补全 `knowledge_graph` / `daily_briefing` / `llm_dream` 功能开关
- `pyproject.toml` 版本号同步至 2.0.0

### v1.3.4

- **知识图谱关联**：新增 `memory_links` 表（schema v5），记忆写入后自动通过向量相似度建立关联，支持关键词降级匹配；提供创建、查询、扩展、导出、统计 6 个方法
- **每日简报生成**：新增 `BriefingGenerator`，在维护周期自动生成当日记忆简报（LLM 生成 + 模板 fallback 双路径），简报自动注入 AGENTS.md
- **数据安全加固**：injector 写入 AGENTS.md 改为原子操作（临时文件 + `os.replace()`），写入前自动创建 `.bak` 备份，进程崩溃或磁盘满不再导致文件损坏
- **事务完整性**：db.py `add()` 将 memories 表和 vec 表写入合并为单一事务，消除中间崩溃导致的数据不一致
- **重试覆盖扩大**：summarizer `_call_llm` 异常捕获从 4 个具体子类改为 `httpx.TransportError`，覆盖所有瞬态网络错误
- **防御性修复**：`_call_llm` 末尾 `raise last_error` 增加 `None` 兜底；indexer `_save_state()` 改为原子写入；记忆文本中的 SECTION_MARKER 自动转义防止注入格式混乱

### v1.3.3

- 新增影子记录机制：守护进程在处理消息的同时维护滚动 6 小时的影子副本
- 检测到 session 文件被重写（消息数减少）时，自动从影子记录恢复丢失的消息
- 恢复时通过内容哈希去重，排除已处理过的消息和当前文件中仍存在的消息

### v1.3.2

- 项目正式定名为 iFlow MemFly（记忆飞轮）
- 统一更新所有源码、Web UI、MCP server、CLI 中的显示名称

### v1.3.1

- `web_port` 可在配置文件中自定义
- 对话空闲 60 秒自动 flush pending 消息
- 新增 `save_memory` MCP tool 和 `POST /api/flush` Web API
- 修复多个 bug：features API 缺少 state_snapshot、LLM 返回非标准格式不重试、MemoryStore.close() 不幂等、状态快照 list 类型崩溃、符号链接双重处理、/tmp 路径硬编码
- 分类记忆 prompt 优化，最短记忆长度阈值提升到 12

### v1.3.0

- 目录结构重组为 core/store/serve 三层
- 新增状态快照功能
- 修复氛围快照、时区、时间戳等多个 bug

### v1.2.0

- 对话续接机制：自动生成工作回顾注入 AGENTS.md
- 对话氛围快照
- 功能开关系统
- Watcher 多工作目录支持

### v1.1.0

- 新增 correction 分类
- MCP stdio server
- AGENTS.md 注入最近对话索引
- FTS5 搜索 fallback 到 LIKE

### v1.0.0

首次发布。三层记忆架构 + SQLite 存储 + AGENTS.md 自动注入。

## License

Copyright (c) 2026 李不是狼. GPL-3.0-or-later. See [LICENSE](LICENSE).
