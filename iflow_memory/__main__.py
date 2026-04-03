"""iFlow MemFly CLI entry point.

Thin CLI layer that delegates to core/, store/, serve/ subpackages.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

from . import __version__
from .config import MemoryConfig, load_config, save_config, DEFAULT_CONFIG_FILE
from .core.daemon import MemoryDaemon
from .store.db import MemoryStore
from .store.embed import Embedder
from .serve.injector import MemoryInjector

logger = logging.getLogger("iflow-memory")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _first_run_setup(config_path) -> None:
    """首次安装引导：提示用户选择开启哪些功能。"""
    print("\n╭─────────────────────────────────────╮")
    print("│   iFlow MemFly 首次安装配置向导     │")
    print("╰─────────────────────────────────────╯\n")
    print("以下功能需要调用 LLM，会消耗 token。")
    print("你可以根据 token 预算选择开启哪些功能。\n")

    features_list = [
        ("index_line", "L1 索引短句", "轻量", "每批消息生成一句话索引（推荐开启）"),
        ("summary", "L2 结构化摘要", "中等", "生成结构化的对话摘要"),
        ("classify", "分类记忆提取", "中等", "提取身份/偏好/知识等分类记忆（推荐开启）"),
        ("atmosphere", "对话氛围快照", "中等", "记录情感状态/对话节奏/隐性共识"),
        ("state_snapshot", "状态快照", "中等", "结构化工作状态检查点（目标/进度/决策/下一步）"),
        ("daily_recap", "每日工作回顾", "较高", "生成前一天的完整工作回顾"),
        ("vector_search", "深度回忆", "视后端", "向量搜索（需配置 embed_backend）"),
        ("knowledge_graph", "知识图谱", "轻量", "为记忆建立关联链接"),
        ("daily_briefing", "每日简报", "中等", "每天生成工作回顾简报"),
        ("llm_dream", "LLM 深度整合", "较高", "用 LLM 合并重复/过时记忆（默认关闭）"),
    ]

    print("可选方案：")
    print("  1) 全部开启（推荐，token 充裕时）")
    print("  2) 精简模式（仅 L1 索引 + 分类记忆，token 节约）")
    print("  3) 最小模式（关闭所有 LLM 功能，仅保留原始记录）")
    print("  4) 自定义选择\n")

    choice = input("请选择 [1/2/3/4]（默认 1）: ").strip() or "1"

    config = MemoryConfig()

    if choice == "1":
        print("\n已选择：全部开启")
    elif choice == "2":
        config.features = {
            "index_line": True,
            "summary": False,
            "classify": True,
            "atmosphere": False,
            "state_snapshot": False,
            "daily_recap": False,
            "vector_search": True,
        }
        print("\n已选择：精简模式（L1 索引 + 分类记忆）")
    elif choice == "3":
        config.features = {k: False for k, _, _, _ in features_list}
        print("\n已选择：最小模式（所有 LLM 功能关闭）")
    elif choice == "4":
        print("\n逐项选择（y=开启, n=关闭）：\n")
        for key, name, cost, desc in features_list:
            default = "y"
            answer = input(f"  {name} [{cost}] {desc} (y/n, 默认 {default}): ").strip().lower() or default
            config.features[key] = answer in ("y", "yes")
        print()
    else:
        print("\n无效输入，使用默认配置（全部开启）")

    save_config(config, Path(config_path))
    print(f"配置已保存到 {config_path}")
    print("后续可通过以下方式修改：")
    print("  命令行: iflow-memory features list / enable / disable")
    print("  网页UI: iflow-memory web\n")


def cmd_start(args: argparse.Namespace) -> None:
    """启动守护进程。"""
    setup_logging(args.verbose)

    # 首次安装检测：配置文件不存在时交互式引导
    config_file = args.config or DEFAULT_CONFIG_FILE
    if not Path(config_file).exists():
        _first_run_setup(config_file)

    config = load_config(args.config)

    pid_file = Path(config.memory_dir) / "iflow-memory.pid"

    # 检查是否已有实例在运行
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)  # 检查进程是否存活
            logger.error(f"Another instance is already running (PID {old_pid})")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # 进程已死或 PID 文件损坏，可以继续
        except PermissionError:
            logger.error(f"Process {old_pid} exists but no permission to check")
            sys.exit(1)

    # 写入 PID 文件
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    async def _run():
        daemon = MemoryDaemon(config)
        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        def handle_signal():
            logger.info("Received shutdown signal...")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        # 初始化 embedder + 健康自检
        await daemon.start()

        # 首次运行：处理最近活跃的 session
        state_file = Path(config.memory_dir) / ".indexer-state.json"
        is_first_run = True
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state_data = json.load(f)
                if state_data.get("processed"):
                    is_first_run = False
            except (json.JSONDecodeError, OSError):
                pass  # 文件损坏，视为首次运行
        if is_first_run:
            logger.info("First run: processing recent sessions...")
            await daemon.process_recent_sessions()

        # 启动时检查最近对话的回顾（往回搜最多 7 天）
        if config.features.get("daily_recap", True):
            memory_dir = Path(config.memory_dir)
            today = datetime.now().date()
            for days_ago in range(1, 8):
                target_date = today - timedelta(days=days_ago)
                date_str = target_date.strftime("%Y-%m-%d")
                recap_cache = memory_dir / f"recap-{date_str}.md"
                l3_file = memory_dir / f"{date_str}.md"
                if recap_cache.exists():
                    break  # 已有 recap，不需要再往前找
                if l3_file.exists():
                    logger.info(f"Generating recap for {date_str} ({days_ago} day(s) ago)")
                    try:
                        recap = await daemon.summarizer.generate_daily_recap(memory_dir, date_str)
                        if recap:
                            daemon.injector.inject()
                            logger.info(f"Recap for {date_str} generated and injected")
                    except Exception as e:
                        logger.warning(f"Failed to generate recap for {date_str}: {e}")
                break  # 只生成最近一天的

        # 启动 watcher 和 interval loop 作为后台任务
        tasks = [asyncio.create_task(daemon.watcher.start())]
        if config.strategy == "interval":
            tasks.append(asyncio.create_task(daemon.interval_loop()))

        # 启动内嵌 Web + MCP 服务
        web_port = config.web_port
        from .serve.web import create_app
        import uvicorn
        web_app = create_app(
            config_path=str(config_file),
            store=daemon.store,
            embedder=daemon.embedder,
        )
        web_app.state.daemon = daemon
        uvi_config = uvicorn.Config(
            web_app, host="127.0.0.1", port=web_port,
            log_level="warning", access_log=False,
        )
        uvi_server = uvicorn.Server(uvi_config)
        tasks.append(asyncio.create_task(uvi_server.serve()))

        logger.info(f"iFlow MemFly v{__version__} starting, strategy={config.strategy}")
        logger.info(f"Memory dir: {config.memory_dir}")
        logger.info(f"Model mode: {config.model_mode}")
        logger.info(f"MCP endpoint: http://127.0.0.1:{web_port}/mcp")

        await stop_event.wait()
        uvi_server.should_exit = True
        await daemon.stop()
        for t in tasks:
            t.cancel()

    try:
        asyncio.run(_run())
    finally:
        pid_file.unlink(missing_ok=True)


def cmd_model(args: argparse.Namespace) -> None:
    """模型管理命令。"""
    config = load_config(args.config)

    if not args.model_action or args.model_action == "list":
        print(f"当前模式: {config.model_mode}")
        if config.model_mode == "custom":
            print(f"活跃预设: {config.active_preset}")
        print("\n预设列表:")
        for name, preset in config.model_presets.items():
            marker = " ←" if name == config.active_preset else ""
            print(f"  {name}: {preset.get('model', '?')}{marker}")

    elif args.model_action == "use":
        name = args.name
        if name == "follow":
            config.model_mode = "follow"
            print("已切换到跟随模式（使用当前对话模型）")
        elif name in config.model_presets:
            config.model_mode = "custom"
            config.active_preset = name
            print(f"已切换到预设: {name} ({config.model_presets[name].get('model', '?')})")
        else:
            print(f"预设 '{name}' 不存在。可用: {', '.join(config.model_presets.keys())}, follow")
            return
        save_config(config, args.config)

    elif args.model_action == "add":
        config.model_presets[args.name] = {
            "name": args.name,
            "base_url": args.base_url or "",
            "api_key": args.api_key or "",
            "model": args.model_name or "glm-5",
        }
        save_config(config, args.config)
        print(f"已添加预设: {args.name}")


def cmd_strategy(args: argparse.Namespace) -> None:
    """策略切换命令。"""
    config = load_config(args.config)
    valid = ("on_compress", "interval", "idle")
    if args.strategy_name not in valid:
        print(f"无效策略。可选: {', '.join(valid)}")
        return
    config.strategy = args.strategy_name
    if args.interval:
        config.interval_seconds = args.interval
    save_config(config, args.config)
    print(f"策略已切换为: {args.strategy_name}")


def cmd_status(args: argparse.Namespace) -> None:
    """查看状态。"""
    logging.disable(logging.WARNING)
    config = load_config(args.config)
    memory_dir = Path(config.memory_dir)
    index_file = memory_dir / "index.md"

    print("iFlow MemFly 状态")
    print(f"  记忆目录: {config.memory_dir}")
    print(f"  策略: {config.strategy}")
    print(f"  模型模式: {config.model_mode}")
    if config.model_mode == "custom":
        print(f"  活跃预设: {config.active_preset}")

    if index_file.exists():
        with open(index_file) as f:
            entry_count = sum(1 for l in f if l.startswith("- "))
        print(f"  索引条目: {entry_count}")
    else:
        print("  索引: 未创建")

    summaries = list(memory_dir.glob("????-??-??*.md"))
    print(f"  摘要文件: {len(summaries)}")

    db_path = memory_dir / "memories.db"
    if db_path.exists():
        with MemoryStore(db_path) as store:
            s = store.stats()
            print(f"  SQLite 记忆: {s['total']} 条 (归档: {s['archived']})")
            for cat, cnt in s.get('by_category', {}).items():
                print(f"    {cat}: {cnt}")


def cmd_web(args: argparse.Namespace) -> None:
    """启动 Web 管理后台。"""
    import uvicorn
    from .serve.web import create_app
    config_path = str(args.config) if args.config else None
    app = create_app(config_path)
    print(f"iFlow MemFly Web UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_inject(args: argparse.Namespace) -> None:
    """手动触发记忆注入到 AGENTS.md。"""
    if args.verbose:
        setup_logging(True)
    else:
        logging.disable(logging.WARNING)
    config = load_config(args.config)
    db_path = Path(config.memory_dir) / "memories.db"
    with MemoryStore(db_path) as store:
        injector = MemoryInjector(store, agents_md_paths=config.agents_md_paths)
        result = injector.inject()
        print(f"已更新 {len(result['updated'])} 个文件，注入 {result['memories_count']} 条记忆")
        for p in result['updated']:
            print(f"  {p}")


# 功能描述映射
FEATURE_INFO = {
    "index_line": ("L1 索引短句", "轻量", "每批消息生成一句话索引"),
    "summary": ("L2 结构化摘要", "中等", "生成结构化的对话摘要"),
    "classify": ("分类记忆提取", "中等", "提取身份/偏好/知识等分类记忆"),
    "atmosphere": ("对话氛围快照", "中等", "记录情感状态/对话节奏/隐性共识"),
    "state_snapshot": ("状态快照", "中等", "结构化工作状态检查点（目标/进度/决策/下一步）"),
    "daily_recap": ("每日工作回顾", "较高", "生成前一天的完整工作回顾"),
    "vector_search": ("深度回忆", "视后端", "向量搜索（需配置 embed_backend）"),
    "knowledge_graph": ("知识图谱", "轻量", "为记忆建立关联链接"),
    "daily_briefing": ("每日简报", "中等", "每天生成工作回顾简报"),
    "llm_dream": ("LLM 深度整合", "较高", "用 LLM 合并重复/过时记忆（默认关闭）"),
}


def cmd_features(args: argparse.Namespace) -> None:
    """功能开关管理。"""
    config = load_config(args.config)

    if not args.features_action or args.features_action == "list":
        print("iFlow MemFly 功能开关\n")
        print(f"  {'功能':<16} {'状态':<6} {'token消耗':<8} 说明")
        print(f"  {'─'*16} {'─'*6} {'─'*8} {'─'*30}")
        for key, (name, cost, desc) in FEATURE_INFO.items():
            enabled = config.features.get(key, True)
            status = "✓ 开启" if enabled else "✗ 关闭"
            print(f"  {key:<16} {status:<6} {cost:<8} {desc}")
        print(f"\n用法: iflow-memory features enable <名称>")
        print(f"      iflow-memory features disable <名称>")

    elif args.features_action == "enable":
        name = args.feature_name
        if name == "all":
            for key in FEATURE_INFO:
                config.features[key] = True
            save_config(config, args.config)
            print("已开启所有功能")
        elif name in FEATURE_INFO:
            config.features[name] = True
            save_config(config, args.config)
            print(f"已开启: {FEATURE_INFO[name][0]} ({name})")
        else:
            print(f"未知功能: {name}。可用: {', '.join(FEATURE_INFO.keys())}, all")

    elif args.features_action == "disable":
        name = args.feature_name
        if name == "all":
            for key in FEATURE_INFO:
                config.features[key] = False
            save_config(config, args.config)
            print("已关闭所有 LLM 功能（仅保留 L3 原始记录和本地存储）")
        elif name in FEATURE_INFO:
            config.features[name] = False
            save_config(config, args.config)
            print(f"已关闭: {FEATURE_INFO[name][0]} ({name})")
        else:
            print(f"未知功能: {name}。可用: {', '.join(FEATURE_INFO.keys())}, all")


def cmd_backfill(args: argparse.Namespace) -> None:
    """为存量记忆补填 embedding 向量。"""
    setup_logging(args.verbose)
    config = load_config(args.config)

    if config.embed_backend == "off":
        print("当前 embed_backend=off，请先配置 embedding 后端")
        print("  编辑 ~/.iflow-memory/config.json 设置 embed_backend 为 'onnx' 或 'api'")
        return

    batch_size = args.batch_size

    async def _run():
        embedder = Embedder(config)
        ok = await embedder.init()
        if not ok:
            print("Embedding 初始化失败，无法 backfill")
            return

        db_path = Path(config.memory_dir) / "memories.db"
        store = MemoryStore(db_path, embed_dim=embedder.dimension)

        total_filled = 0
        while True:
            memories = store.get_memories_without_embedding(limit=batch_size)
            if not memories:
                break

            texts = [m["text"] for m in memories]
            embeddings = await embedder.embed_batch(texts)
            if not embeddings:
                print("Embedding 生成失败，中止")
                break

            for mem, emb in zip(memories, embeddings):
                if store.update_embedding(mem["id"], emb):
                    total_filled += 1

            print(f"  已补填 {total_filled} 条...")

        print(f"\nBackfill 完成: 共补填 {total_filled} 条记忆的 embedding")
        store.close()
        await embedder.close()

    asyncio.run(_run())


def cmd_query(args: argparse.Namespace) -> None:
    """搜索记忆。"""
    if args.verbose:
        setup_logging(True)
    else:
        logging.disable(logging.WARNING)
    config = load_config(args.config)
    db_path = Path(config.memory_dir) / "memories.db"
    with MemoryStore(db_path) as store:
        results = store.search(args.query, category=args.category, limit=args.limit)
        if not results:
            print("未找到相关记忆")
        else:
            for r in results:
                print(f"[{r['category']}] {r['text']}")
                meta = f"  创建: {r['created_at'][:10]} | 访问: {r['access_count']}次"
                if r.get('source_file'):
                    src = r['source_file']
                    if r.get('source_line') is not None:
                        src += f":{r['source_line']}"
                    meta += f" | 来源: {src}"
                print(meta)
                print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iflow-memory",
        description="iFlow MemFly — 记忆飞轮，iFlow 的记忆觉醒项目",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"配置文件路径 (默认: {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # start
    sub.add_parser("start", help="启动守护进程").set_defaults(func=cmd_start)

    # model
    model_parser = sub.add_parser("model", help="模型管理")
    model_parser.set_defaults(func=cmd_model)
    model_sub = model_parser.add_subparsers(dest="model_action")
    model_sub.add_parser("list", help="列出预设")
    use_parser = model_sub.add_parser("use", help="切换预设")
    use_parser.add_argument("name", help="预设名称或 'follow'")
    add_parser = model_sub.add_parser("add", help="添加预设")
    add_parser.add_argument("name", help="预设名称")
    add_parser.add_argument("--base-url", dest="base_url")
    add_parser.add_argument("--api-key", dest="api_key")
    add_parser.add_argument("--model", dest="model_name")

    # strategy
    strat_parser = sub.add_parser("strategy", help="切换策略")
    strat_parser.set_defaults(func=cmd_strategy)
    strat_parser.add_argument("strategy_name", choices=["on_compress", "interval", "idle"])
    strat_parser.add_argument("--interval", type=int, help="interval 模式的间隔秒数")

    # status
    sub.add_parser("status", help="查看状态").set_defaults(func=cmd_status)

    # features
    feat_parser = sub.add_parser("features", help="功能开关管理")
    feat_parser.set_defaults(func=cmd_features)
    feat_sub = feat_parser.add_subparsers(dest="features_action")
    feat_sub.add_parser("list", help="列出所有功能开关")
    feat_enable = feat_sub.add_parser("enable", help="开启功能")
    feat_enable.add_argument("feature_name", help="功能名称或 'all'")
    feat_disable = feat_sub.add_parser("disable", help="关闭功能")
    feat_disable.add_argument("feature_name", help="功能名称或 'all'")

    # inject
    sub.add_parser("inject", help="手动注入记忆到 AGENTS.md").set_defaults(func=cmd_inject)

    # query
    query_parser = sub.add_parser("query", help="搜索记忆")
    query_parser.set_defaults(func=cmd_query)
    query_parser.add_argument("query", help="搜索关键词")
    query_parser.add_argument("-c", "--category", help="按分类过滤")
    query_parser.add_argument("-n", "--limit", type=int, default=10, help="结果数量")

    # backfill
    backfill_parser = sub.add_parser("backfill", help="为存量记忆补填 embedding 向量")
    backfill_parser.set_defaults(func=cmd_backfill)
    backfill_parser.add_argument("--batch-size", type=int, default=50, dest="batch_size",
                                  help="每批处理数量 (默认: 50)")

    # web
    web_parser = sub.add_parser("web", help="启动 Web 管理后台")
    web_parser.set_defaults(func=cmd_web)
    web_parser.add_argument("--port", type=int, default=8765, help="端口 (默认: 8765)")
    web_parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")

    args = parser.parse_args()

    func = getattr(args, "func", None)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
