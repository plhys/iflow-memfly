"""Shared MCP tool definitions for iFlow MemFly.

Single source of truth — imported by both web.py (HTTP) and mcp_server.py (stdio).
"""

MCP_TOOLS = [
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
