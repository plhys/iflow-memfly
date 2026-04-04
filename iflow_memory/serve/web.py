"""iFlow MemFly Web UI — FastAPI backend + HTML frontend."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import __version__
from ..config import MemoryConfig, load_config, save_config, DEFAULT_CONFIG_FILE
from ..store.embed import Embedder
from ..core.indexer import STRIP_PATTERNS, TOOL_PART_TYPES
from ..store.db import MemoryStore

logger = logging.getLogger("iflow-memory")


def _check_daemon_status(config: MemoryConfig) -> None:
    """检测 daemon 是否在运行，未运行时输出 warning。"""
    pid_file = Path(config.memory_dir) / "iflow-memory.pid"
    if not pid_file.exists():
        logger.warning("Daemon 未运行（PID 文件不存在）。记忆索引不会自动更新。启动: iflow-memory start")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        logger.warning("Daemon 未运行（进程已退出）。记忆索引不会自动更新。启动: iflow-memory start")
    except PermissionError:
        pass  # 进程存在但无权检查，不告警


def create_app(
    config_path: str = None,
    store: Optional[MemoryStore] = None,
    embedder: Optional[Embedder] = None,
) -> FastAPI:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
    config = load_config(cfg_path)
    app = FastAPI(title="iFlow MemFly")

    # 保存 store/embedder 引用供 MCP 端点使用
    app.state.memory_store = store
    app.state.embedder = embedder
    app.state.memory_dir = Path(config.memory_dir)

    # 启动时检测 daemon 是否在运行
    _check_daemon_status(config)

    def _reload():
        nonlocal config
        config = load_config(cfg_path)

    def _save():
        save_config(config, cfg_path)

    mem = Path(config.memory_dir)
    index_file = mem / "index.md"
    state_file = mem / ".indexer-state.json"

    @app.get("/", response_class=HTMLResponse)
    async def home():
        html_path = Path(__file__).parent / "ui.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>ui.html not found</h1>"

    @app.post("/api/flush")
    async def flush_memory():
        """立即处理所有 pending 消息并更新 AGENTS.md。"""
        daemon = getattr(app.state, "daemon", None)
        if not daemon:
            return JSONResponse({"error": "daemon 未连接"}, status_code=503)
        result = await daemon.flush_now()
        return JSONResponse(result)

    @app.get("/api/status")
    async def status():
        _reload()
        idx_lines = 0
        if index_file.exists():
            idx_lines = index_file.read_text(encoding="utf-8").count("\n")
        summaries = [f for f in mem.glob("*.md") if f.name != "index.md"]
        model = config.get_active_model()
        return {
            "strategy": config.strategy,
            "model_mode": config.model_mode,
            "active_model": model.model,
            "index_lines": idx_lines,
            "summary_count": len(summaries),
            "memory_dir": str(mem),
        }

    @app.get("/api/index")
    async def get_index(lines: int = Query(50, ge=1, le=500)):
        if not index_file.exists():
            return {"content": "", "total_lines": 0}
        all_lines = index_file.read_text(encoding="utf-8").splitlines()
        if len(all_lines) <= lines + 2:
            content = "\n".join(all_lines)
        else:
            content = "\n".join(all_lines[:2] + all_lines[2:2 + lines])
        return {"content": content, "total_lines": len(all_lines)}

    @app.get("/api/index/search")
    async def search_index(q: str = Query(..., min_length=1)):
        if not index_file.exists():
            return {"results": []}
        results = []
        for i, line in enumerate(index_file.read_text(encoding="utf-8").splitlines()):
            if q.lower() in line.lower():
                results.append({"line": i + 1, "text": line})
        return {"results": results, "query": q}

    @app.get("/api/summaries")
    async def list_summaries():
        files = []
        for f in sorted(mem.glob("*.md"), reverse=True):
            if f.name == "index.md":
                continue
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
                "lines": f.read_text(encoding="utf-8").count("\n"),
            })
        return {"files": files}

    @app.get("/api/summary/{filename}")
    async def get_summary(filename: str, line: int = 1, limit: int = 100):
        fp = mem / filename
        if not fp.exists() or not fp.suffix == ".md":
            return JSONResponse({"error": "not found"}, 404)
        all_lines = fp.read_text(encoding="utf-8").splitlines()
        start = max(0, line - 1)
        end = min(len(all_lines), start + limit)
        return {
            "filename": filename,
            "content": "\n".join(all_lines[start:end]),
            "start_line": start + 1,
            "end_line": end,
            "total_lines": len(all_lines),
        }

    @app.post("/api/model/mode")
    async def set_model_mode(body: dict):
        mode = body.get("mode", "follow")
        if mode not in ("follow", "custom"):
            return JSONResponse({"error": "invalid mode"}, 400)
        config.model_mode = mode
        if mode == "custom":
            config.active_preset = body.get("preset", "default")
        _save()
        return {"ok": True, "mode": mode}

    @app.get("/api/model/presets")
    async def list_presets():
        _reload()
        return {
            "mode": config.model_mode,
            "active": config.active_preset,
            "presets": config.model_presets,
        }

    @app.get("/api/model/current")
    async def current_model():
        _reload()
        model = config.get_active_model()
        return {
            "mode": config.model_mode,
            "name": model.name,
            "model": model.model,
            "base_url": model.base_url,
            "has_api_key": bool(model.api_key),
        }

    @app.post("/api/model/preset")
    async def upsert_preset(body: dict):
        name = body.get("name")
        if not name:
            return JSONResponse({"error": "name required"}, 400)
        config.model_presets[name] = {
            "name": name,
            "base_url": body.get("base_url", ""),
            "api_key": body.get("api_key", ""),
            "model": body.get("model", ""),
        }
        _save()
        return {"ok": True}

    @app.delete("/api/model/preset/{name}")
    async def delete_preset(name: str):
        if name == "default":
            return JSONResponse({"error": "cannot delete default"}, 400)
        config.model_presets.pop(name, None)
        _save()
        return {"ok": True}

    @app.post("/api/strategy")
    async def set_strategy(body: dict):
        valid = ("interval", "on_compress", "idle", "realtime", "manual")
        s = body.get("strategy", "interval")
        if s not in valid:
            return JSONResponse({"error": "invalid strategy"}, 400)
        config.strategy = s
        _save()
        return {"ok": True, "strategy": s}

    @app.post("/api/reindex")
    async def reindex():
        """真正重建索引：备份旧索引 → 清除状态 → 重新扫描记忆文件生成索引。"""
        # 备份旧索引
        if index_file.exists():
            backup = mem / ".index.md.bak"
            shutil.copy2(index_file, backup)

        # 清除增量状态
        if state_file.exists():
            state_file.unlink()

        # 扫描所有记忆 md 文件，重建索引
        md_files = sorted(mem.glob("*.md"), reverse=True)
        entries_by_date = {}
        for f in md_files:
            if f.name == "index.md":
                continue
            # 从文件名提取日期
            name = f.stem
            date_str = name[:10] if len(name) >= 10 and name[:4].isdigit() else None
            if not date_str:
                continue
            lines = f.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            # 提取摘要：用文件的 h1 标题或前几行有意义的内容
            summary = ""
            for line in lines[:10]:
                line = line.strip()
                if line.startswith("# "):
                    summary = line[2:].strip()
                    break
                if line.startswith("## ") and not summary:
                    summary = line[3:].strip()
            if not summary:
                summary = f.name
            if date_str not in entries_by_date:
                entries_by_date[date_str] = []
            entries_by_date[date_str].append(
                f"- {summary} ({total}行) → {f.name}:1"
            )

        # 写入新索引
        content = "# iFlow MemFly Index\n"
        for date in sorted(entries_by_date.keys(), reverse=True):
            content += f"\n## {date}\n"
            for entry in entries_by_date[date]:
                content += entry + "\n"
        index_file.write_text(content, encoding="utf-8")

        total_entries = sum(len(v) for v in entries_by_date.values())
        return {
            "ok": True,
            "message": f"索引重建完成：{total_entries} 条记录，{len(entries_by_date)} 个日期",
        }

    @app.get("/api/clean-rules")
    async def get_clean_rules():
        """返回当前清洗规则列表。"""
        rules = []
        for i, p in enumerate(STRIP_PATTERNS):
            rules.append({
                "id": i,
                "pattern": p.pattern,
                "type": "regex",
                "description": _describe_pattern(p.pattern),
            })
        rules.append({
            "id": len(STRIP_PATTERNS),
            "pattern": str(TOOL_PART_TYPES),
            "type": "part_type",
            "description": "过滤 tool call 相关的消息部分（functionCall, functionResponse, tool_use, tool_result）",
        })
        return {"rules": rules, "truncate_length": 500}

    @app.get("/api/features")
    async def get_features():
        """获取功能开关状态。"""
        _reload()
        feature_info = {
            "index_line": {"name": "L1 索引短句", "cost": "轻量", "desc": "每批消息生成一句话索引"},
            "summary": {"name": "L2 结构化摘要", "cost": "中等", "desc": "生成结构化的对话摘要"},
            "classify": {"name": "分类记忆提取", "cost": "中等", "desc": "提取身份/偏好/知识等分类记忆"},
            "atmosphere": {"name": "对话氛围快照", "cost": "中等", "desc": "记录情感状态/对话节奏/隐性共识"},
            "state_snapshot": {"name": "状态快照", "cost": "中等", "desc": "结构化工作状态检查点（目标/进度/决策/下一步）"},
            "daily_recap": {"name": "每日工作回顾", "cost": "较高", "desc": "生成前一天的完整工作回顾"},
            "vector_search": {"name": "深度回忆", "cost": "视后端", "desc": "向量搜索（需配置 embed_backend）"},
            "knowledge_graph": {"name": "知识图谱", "cost": "轻量", "desc": "为记忆建立关联链接，注入时做图谱扩展"},
            "daily_briefing": {"name": "每日简报", "cost": "中等", "desc": "每天生成一段精炼的工作回顾简报"},
            "llm_dream": {"name": "LLM 深度整合", "cost": "较高", "desc": "用 LLM 分析记忆库，合并重复/过时条目（默认关闭）"},
        }
        features = []
        for key, info in feature_info.items():
            features.append({
                "key": key,
                "enabled": config.features.get(key, True),
                **info,
            })
        return {"features": features}

    @app.post("/api/features")
    async def set_features(body: dict):
        """更新功能开关。body: {"key": "atmosphere", "enabled": false}"""
        key = body.get("key")
        enabled = body.get("enabled")
        valid_keys = {"index_line", "summary", "classify", "atmosphere", "state_snapshot", "daily_recap", "vector_search", "knowledge_graph", "daily_briefing", "llm_dream"}
        if key not in valid_keys:
            return JSONResponse({"error": f"未知功能: {key}"}, 400)
        if not isinstance(enabled, bool):
            return JSONResponse({"error": "enabled 必须是 true/false"}, 400)
        config.features[key] = enabled
        _save()
        return {"ok": True, "key": key, "enabled": enabled}

    # ── MCP Streamable HTTP endpoint ──────────────────────────────

    from .tools import MCP_TOOLS as _MCP_TOOLS

    def _mcp_response(msg_id, result):
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _mcp_error(msg_id, code, message):
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    async def _handle_mcp_tool_call(params: dict) -> dict:
        """Handle a tools/call request using the shared store and embedder."""
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        store_ = app.state.memory_store
        embedder_ = app.state.embedder
        mem_dir = app.state.memory_dir

        if tool_name == "search_memory":
            query = args.get("query", "")
            category = args.get("category")
            limit = args.get("limit", 10)
            date_from = args.get("date_from")
            if not query:
                return {"content": [{"type": "text", "text": "请提供搜索关键词"}], "isError": True}

            query_embedding = None
            if embedder_ and embedder_.available:
                try:
                    query_embedding = await embedder_.embed(query)
                except Exception:
                    pass

            if store_ is None:
                return {"content": [{"type": "text", "text": "记忆存储未初始化"}], "isError": True}

            results = store_.hybrid_search(
                query, query_embedding=query_embedding,
                category=category, limit=limit,
                date_from=date_from,
            )
            if not results:
                text = f"未找到与「{query}」相关的记忆。建议：换同义词重试，或去掉 category/date_from 过滤扩大范围"
            else:
                lines = [f"找到 {len(results)} 条相关记忆（按相关度排序）："]
                for i, r in enumerate(results, 1):
                    lines.append(f"#{i} [id:{r['id']}] [{r['category']}] {r['text']}")
                    meta = f"  创建: {r['created_at'][:10]} | 访问: {r['access_count']}次"
                    if r.get('source_file'):
                        src = r['source_file']
                        if r.get('source_line') is not None:
                            src += f":{r['source_line']}"
                        meta += f" | 来源: {src}"
                    lines.append(meta)
                text = "\n".join(lines)

            return {"content": [{"type": "text", "text": text}]}

        elif tool_name == "get_recent_context":
            n = args.get("lines", 10)
            index_file = mem_dir / "index.md"
            if not index_file.exists():
                text = "暂无对话索引"
            else:
                all_lines = index_file.read_text(encoding="utf-8").splitlines()
                entries = [l for l in all_lines if l.startswith("- ")]
                recent = entries[:n]
                text = "\n".join(recent) if recent else "暂无对话索引"
            return {"content": [{"type": "text", "text": text}]}

        elif tool_name == "delete_memory":
            ids = args.get("ids", [])
            if not ids:
                return {"content": [{"type": "text", "text": "请提供要删除的记忆 ID"}], "isError": True}
            if store_ is None:
                return {"content": [{"type": "text", "text": "记忆存储未初始化"}], "isError": True}
            count = store_.archive_by_ids(ids)
            text = f"已归档 {count} 条记忆（ID: {ids}）"
            return {"content": [{"type": "text", "text": text}]}

        elif tool_name == "save_memory":
            daemon = getattr(app.state, "daemon", None)
            if daemon:
                try:
                    result = await daemon.flush_now()
                    text = f"记忆已保存。处理了 {result.get('processed', 0)} 条消息，AGENTS.md 已更新。"
                except Exception as e:
                    text = f"记忆保存失败: {e}"
            else:
                text = "daemon 未连接，无法立即保存。记忆将在下次 daemon 处理周期自动保存。"
            return {"content": [{"type": "text", "text": text}]}

        else:
            return {"content": [{"type": "text", "text": f"未知工具: {tool_name}"}], "isError": True}

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        """MCP Streamable HTTP endpoint — JSON-RPC over HTTP POST."""
        body = await request.json()

        # Handle batch (array) or single message
        is_batch = isinstance(body, list)
        messages = body if is_batch else [body]

        responses = []
        for msg in messages:
            method = msg.get("method", "")
            msg_id = msg.get("id")

            if method == "initialize":
                resp = _mcp_response(msg_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "iflow-memfly", "version": __version__},
                })
                responses.append(resp)

            elif method == "notifications/initialized":
                continue  # notification, no response

            elif method == "tools/list":
                responses.append(_mcp_response(msg_id, {"tools": _MCP_TOOLS}))

            elif method == "tools/call":
                params = msg.get("params", {})
                try:
                    result = await _handle_mcp_tool_call(params)
                except Exception as e:
                    result = {"content": [{"type": "text", "text": f"内部错误: {e}"}], "isError": True}
                responses.append(_mcp_response(msg_id, result))

            elif method == "ping":
                responses.append(_mcp_response(msg_id, {}))

            elif msg_id is not None:
                responses.append(_mcp_error(msg_id, -32601, f"Method not found: {method}"))
            # else: unknown notification, ignore

        if not responses:
            return JSONResponse(status_code=202, content=None)

        if is_batch:
            return JSONResponse(content=responses)
        elif len(responses) == 1:
            return JSONResponse(content=responses[0])
        else:
            return JSONResponse(content=responses)

    @app.get("/mcp")
    async def mcp_sse_stream():
        """MCP GET endpoint — not supported, we don't send server-initiated messages."""
        return JSONResponse(status_code=405, content={"error": "Server-initiated SSE not supported"})

    @app.delete("/mcp")
    async def mcp_session_delete():
        """MCP session termination — no-op since we're stateless."""
        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        store_ = app.state.memory_store
        return {
            "status": "ok",
            "version": __version__,
            "store_available": store_ is not None,
            "embedder_available": app.state.embedder is not None and app.state.embedder.available,
        }

    return app


def _describe_pattern(pattern: str) -> str:
    """为正则模式生成人类可读描述。"""
    desc_map = {
        "system-reminder": "过滤 <system-reminder> 系统提示标签",
        "environment_details": "过滤 <environment_details> 环境信息标签",
        "history_context": "过滤 <history_context> 历史上下文标签",
        "language": "过滤 [language] 语言标记块",
        "context": "过滤 <context> 上下文标签",
    }
    for key, desc in desc_map.items():
        if key in pattern:
            return desc
    return f"正则过滤: {pattern[:60]}..."