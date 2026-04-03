"""MCP stdio server for iFlow MemFly — exposes memory search as tools.

Implements the Model Context Protocol (JSON-RPC over stdin/stdout) with
two tools: search_memory and get_recent_context.

Usage:
    python3 -m iflow_memory.mcp_server
"""

import asyncio
import json
import sys
from pathlib import Path

from .. import __version__
from ..config import load_config
from ..store.embed import Embedder
from ..guard import error_boundary
from ..store.db import MemoryStore


def _make_response(id, result):
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _make_error(id, code, message):
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _handle_initialize(msg):
    return _make_response(msg["id"], {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "iflow-memfly", "version": __version__},
    })


def _handle_tools_list(msg):
    tools = [
        {
            "name": "search_memory",
            "description": "搜索 iFlow MemFly 记忆库。查找历史信息的首选工具，优先于 grep 或 read_file。支持向量语义搜索 + 全文检索混合排序，结果按相关度从高到低排列。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词：提取 2-4 个核心名词，不要用完整句子。示例：用户问'今天上午的评测报告' → query='评测报告'",
                    },
                    "category": {
                        "type": "string",
                        "description": "按分类过滤。identity=身份, preference=偏好, entity=知识/事实, event=事件/操作记录, insight=经验教训, correction=纠正。不确定时不填",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量，默认 10",
                        "default": 10,
                    },
                    "date_from": {
                        "type": "string",
                        "description": "日期过滤，格式 YYYY-MM-DD。用户提到时间词时必须填写：今天→当天日期，昨天→前一天，上周→7天前。示例：'今天上午' → date_from='2026-04-03'",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_recent_context",
            "description": "获取最近几次对话的索引摘要，了解之前聊了什么、做到哪了。当需要接续上次对话的上下文时使用。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "返回最近多少条索引，默认 10",
                        "default": 10,
                    },
                },
            },
        },
        {
            "name": "save_memory",
            "description": "立即保存当前对话的记忆。当对话即将结束、或用户要求保存记忆时使用。会立即处理所有待处理的消息并更新 AGENTS.md。",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]
    return _make_response(msg["id"], {"tools": tools})


@error_boundary
def _handle_tool_call(msg, store: MemoryStore, memory_dir: Path, embedder: Embedder | None = None, loop: asyncio.AbstractEventLoop | None = None, *, web_port: int = 18765):
    params = msg.get("params", {})
    tool_name = params.get("name", "")
    args = params.get("arguments", {})

    if tool_name == "search_memory":
        query = args.get("query", "")
        category = args.get("category")
        limit = args.get("limit", 10)
        date_from = args.get("date_from")

        if not query:
            return _make_response(msg["id"], {
                "content": [{"type": "text", "text": "请提供搜索关键词"}],
                "isError": True,
            })

        # 生成查询向量用于 hybrid_search
        query_embedding = None
        if embedder and embedder.available and loop:
            try:
                query_embedding = loop.run_until_complete(embedder.embed(query))
            except Exception:
                pass  # 降级到纯 FTS5

        results = store.hybrid_search(
            query, query_embedding=query_embedding,
            category=category, limit=limit,
            date_from=date_from,
        )
        if not results:
            text = f"未找到与「{query}」相关的记忆。建议：换同义词重试，或去掉 category 过滤扩大范围"
        else:
            lines = [f"找到 {len(results)} 条相关记忆（按相关度排序）："]
            for i, r in enumerate(results, 1):
                lines.append(f"#{i} [{r['category']}] {r['text']}")
                lines.append(f"  创建: {r['created_at'][:10]} | 访问: {r['access_count']}次")
            text = "\n".join(lines)

        return _make_response(msg["id"], {
            "content": [{"type": "text", "text": text}],
        })

    elif tool_name == "get_recent_context":
        n = args.get("lines", 10)
        index_file = memory_dir / "index.md"
        if not index_file.exists():
            text = "暂无对话索引"
        else:
            with open(index_file) as f:
                all_lines = f.readlines()
            # Skip header, take recent entries
            entries = [l.rstrip() for l in all_lines if l.startswith("- ")]
            recent = entries[:n]
            text = "\n".join(recent) if recent else "暂无对话索引"

        return _make_response(msg["id"], {
            "content": [{"type": "text", "text": text}],
        })

    elif tool_name == "save_memory":
        import urllib.request
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{web_port}/api/flush",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            flushed = result.get("flushed", 0)
            injected = result.get("injected", 0)
            text = f"记忆已保存。处理 {flushed} 个 session，更新 {injected} 个文件。"
        except Exception as e:
            text = f"保存失败（daemon 可能未运行）: {e}"

        return _make_response(msg["id"], {
            "content": [{"type": "text", "text": text}],
        })

    else:
        return _make_response(msg["id"], {
            "content": [{"type": "text", "text": f"未知工具: {tool_name}"}],
            "isError": True,
        })


def main():
    config = load_config()
    memory_dir = Path(config.memory_dir)
    db_path = memory_dir / "memories.db"
    embed_dim = config.embed_dim or 0
    store = MemoryStore(db_path, embed_dim=embed_dim)

    # 创建持久 event loop（避免 asyncio.run 多次调用导致 loop closed）
    loop = asyncio.new_event_loop()

    # 初始化 Embedder 用于查询时生成向量
    embedder = Embedder(config)
    try:
        loop.run_until_complete(embedder.init())
    except Exception:
        embedder = None  # 降级到纯 FTS5

    # Suppress all logging to stderr to keep stdout clean for JSON-RPC
    import logging
    import os
    logging.disable(logging.CRITICAL)

    # Check if daemon is running — warn via stderr (won't pollute JSON-RPC on stdout)
    pid_file = memory_dir / "iflow-memory.pid"
    daemon_running = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            daemon_running = True
        except (ProcessLookupError, ValueError):
            pass
        except PermissionError:
            daemon_running = True  # exists but can't check, assume running
    if not daemon_running:
        print(
            "[iflow-memory] WARNING: daemon 未运行，记忆不会自动更新。"
            "启动: iflow-memory start",
            file=sys.stderr,
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            resp = _handle_initialize(msg)
        elif method == "notifications/initialized":
            continue  # notification, no response needed
        elif method == "tools/list":
            resp = _handle_tools_list(msg)
        elif method == "tools/call":
            resp = _handle_tool_call(msg, store, memory_dir, embedder, loop, web_port=config.web_port)
        elif method == "ping":
            resp = _make_response(msg_id, {})
        elif msg_id is not None:
            resp = _make_error(msg_id, -32601, f"Method not found: {method}")
        else:
            continue  # unknown notification, ignore

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    if embedder:
        try:
            loop.run_until_complete(embedder.close())
        except Exception:
            pass
    loop.close()
    store.close()


if __name__ == "__main__":
    main()
