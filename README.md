# iFlow MemFly — 记忆飞轮

给 AI CLI 助手加上持久记忆的守护进程。自动从对话中提取记忆，新会话启动时注入上下文，让 AI 记得之前聊过什么。

> 1 进程，64MB 内存，Python 3.10+。

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

## 它做什么

守护进程每 10 秒扫描一次 session 文件，检测到新对话后：

1. **清洗原始对话** → 写入当日 `.md` 文件（L3 原始记录）
2. **LLM 生成索引短句** → 写入 `index.md`（L1，≤40 字）
3. **LLM 生成结构化摘要** → 追加到当日 `.md`（L2）
4. **LLM 提取分类记忆** → 写入 SQLite（6 类：身份/偏好/纠正/实体/事件/经验）
5. **LLM 生成氛围快照和状态快照** → 写入 SQLite
6. **自动注入 AGENTS.md** → 新会话启动时 AI 直接读到这些信息

核心思路：记忆在新会话启动前就已经写入上下文，AI 不需要主动调用工具去"回忆"。

### 影子记录机制

守护进程在正常处理消息的同时，维护一份滚动 6 小时的影子记录。当检测到 session 文件被重写（上下文压缩或溢出导致消息丢失）时，自动从影子记录中恢复未处理的消息，防止记忆丢失。

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

**功能开关**：5 个 LLM 功能可独立开关 — `index_line` / `summary` / `classify` / `atmosphere` / `daily_recap`。

```bash
iflow-memory features list
iflow-memory features enable atmosphere
iflow-memory features disable summary
```

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
│   └── summarizer.py    LLM 摘要生成
├── store/               记忆存储
│   ├── db.py            SQLite + FTS5 + sqlite-vec
│   └── embed.py         向量嵌入
└── serve/               记忆使用
    ├── injector.py      AGENTS.md 注入
    ├── mcp_server.py    MCP stdio
    └── web.py           Web 管理后台
```

## 存储

```
~/.iflow-memory/data/
├── memories.db              SQLite 数据库（FTS5 全文检索 + sqlite-vec 向量）
├── index.md                 L1 索引
├── 2026-03-29.md            L3 原始记录 + L2 摘要
├── .shadow/                 影子记录（滚动 6 小时）
├── .indexer-state.json      增量处理状态
└── iflow-memory.pid         PID 文件
```

## 测试

```bash
python -m pytest tests/ -v
```

## Changelog

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