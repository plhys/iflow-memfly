"""MemoryDaemon — 记忆索引守护进程核心。

监听对话 session 文件变化，驱动三层记忆管线：
  L3 原始记录 → L1 索引 → L2 摘要 → 分类记忆 → 氛围快照 → 状态快照
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import MemoryConfig
from ..guard import daemon_health_check
from ..store.db import MemoryStore
from ..store.embed import Embedder
from .indexer import Indexer
from .summarizer import Summarizer
from .watcher import SessionWatcher
from ..serve.injector import MemoryInjector

logger = logging.getLogger("iflow-memory")


class MemoryDaemon:
    """记忆索引守护进程。"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.embedder: Optional[Embedder] = None
        db_path = Path(config.memory_dir) / "memories.db"
        self.store = MemoryStore(db_path, embed_dim=0)
        self.indexer = Indexer(Path(config.memory_dir), store=self.store)
        self.summarizer = Summarizer(config)
        self.injector = MemoryInjector(self.store, agents_md_paths=config.agents_md_paths)
        self.watcher = SessionWatcher(
            acp_dir=Path(config.acp_sessions_dir),
            cli_dir=Path(config.cli_sessions_dir),
            on_change=self._on_session_change,
            poll_interval=10.0,
        )
        self._pending: dict[str, tuple[list[dict], Path, str, int]] = {}
        self._process_task: asyncio.Task | None = None
        self._last_activity: dict[str, float] = {}  # session_key -> timestamp
        self._idle_flushed: set[str] = set()  # 已 idle-flush 的 session

    async def start(self) -> None:
        """初始化守护进程：健康自检 + embedder 初始化。"""
        health = daemon_health_check(self.config)
        if not health["db_ok"]:
            logger.error("[记忆守护] DB 不可写，请检查磁盘空间和权限")
        if not health["disk_ok"]:
            logger.warning("[记忆守护] 磁盘空间不足")

        if self.config.features.get("vector_search", True) and self.config.embed_backend != "off":
            self.embedder = Embedder(self.config)
            embed_ok = await self.embedder.init()
            if embed_ok and self.embedder.dimension > 0:
                self.store.close()
                db_path = Path(self.config.memory_dir) / "memories.db"
                self.store = MemoryStore(db_path, embed_dim=self.embedder.dimension)
                self.indexer.store = self.store
                self.injector.store = self.store
                logger.info(f"[深度回忆] 向量搜索已启用，维度={self.embedder.dimension}")
            else:
                logger.info("[深度回忆] embedding 不可用，使用纯 FTS5 搜索")
                self.embedder = None
        else:
            logger.info("[深度回忆] 向量搜索已关闭")

        from .. import __version__
        logger.info(f"iFlow MemFly v{__version__} ready")

    async def stop(self) -> None:
        """停止守护进程。"""
        self.watcher.stop()
        if self._process_task:
            self._process_task.cancel()
        self.store.close()
        await self.summarizer.close()
        if self.embedder:
            await self.embedder.close()
        logger.info("iFlow MemFly stopped")

    async def _on_session_change(self, path: Path, meta: dict) -> None:
        """session 文件变化回调。"""
        try:
            source = meta["source"]
            new_msgs, total_count = self.indexer.get_new_messages(path, source)
            if not new_msgs:
                return

            key = str(path)
            self._last_activity[key] = time.time()
            self._idle_flushed.discard(key)

            logger.info(f"New messages from {source}: {path.name} (+{len(new_msgs)})")

            if self.config.strategy == "on_compress":
                await self._process_messages(new_msgs, path, source, total_count)
            else:
                self._pending[str(path)] = (new_msgs, path, source, total_count)
        except Exception as e:
            logger.error(f"[记忆守护] session 变化处理异常: {e}")

    async def interval_loop(self) -> None:
        """定时处理攒起来的消息。"""
        cycle = 0
        idle_threshold = 60  # 秒
        while True:
            await asyncio.sleep(self.config.interval_seconds)

            # Idle flush: session 超过 60 秒无新消息，立即处理 pending
            now = time.time()
            idle_sessions = []
            for key, last_ts in list(self._last_activity.items()):
                if (now - last_ts >= idle_threshold
                        and key in self._pending
                        and key not in self._idle_flushed):
                    idle_sessions.append(key)
            if idle_sessions:
                logger.info(f"[idle flush] {len(idle_sessions)} 个 session 已静默，立即处理")
                self._idle_flushed.update(idle_sessions)
                await self._flush_pending()
            elif self._pending:
                await self._flush_pending()

            cycle += 1
            if cycle % 10 == 0:
                await self._maintenance()

    async def _maintenance(self) -> None:
        """周期性维护：归档冷记忆、检查手写区大小、生成对话回顾。"""
        archived = self.store.archive_cold(min_age_days=5)
        if archived:
            logger.info(f"[维护] 归档 {archived} 条冷记忆")
            self.injector.inject()

        for md_path in self.injector.agents_md_paths:
            try:
                content = md_path.read_text(encoding="utf-8")
                marker = "### 记忆系统（自动生成，勿手动编辑）"
                idx = content.find(marker)
                manual_size = idx if idx > 0 else len(content)
                if manual_size > 3072:
                    logger.warning(
                        f"[维护] {md_path.name} 手写部分 {manual_size} 字节，"
                        f"超过 3KB 阈值，建议精简"
                    )
            except OSError:
                pass

        if self.config.features.get("daily_recap", True):
            await self._generate_missing_recap()

        self.store.checkpoint()

    async def _generate_missing_recap(self) -> None:
        """生成最近缺失的对话回顾。"""
        memory_dir = Path(self.config.memory_dir)
        today = datetime.now().date()
        for days_ago in range(1, 8):
            target_date = today - timedelta(days=days_ago)
            date_str = target_date.strftime("%Y-%m-%d")
            recap_cache = memory_dir / f"recap-{date_str}.md"
            l3_file = memory_dir / f"{date_str}.md"
            if recap_cache.exists():
                break
            if l3_file.exists():
                logger.info(f"[维护] 生成对话回顾: {date_str}")
                try:
                    recap = await self.summarizer.generate_daily_recap(memory_dir, date_str)
                    if recap:
                        self.injector.inject()
                        logger.info(f"[维护] {date_str} 回顾已生成并注入")
                except Exception as e:
                    logger.error(f"[维护] 回顾生成失败: {e}")
            break

    async def _flush_pending(self) -> None:
        """处理所有待处理的消息。"""
        if not self._pending:
            return
        batch, self._pending = self._pending, {}
        for messages, session_path, source, total_count in batch.values():
            await self._process_messages(messages, session_path, source, total_count)

    async def flush_now(self) -> dict:
        """手动触发：立即处理所有 pending 消息并注入。

        Returns:
            dict with keys: flushed (int), injected (int)
        """
        pending_count = len(self._pending)
        if pending_count > 0:
            logger.info(f"[手动 flush] 处理 {pending_count} 个 pending session")
            await self._flush_pending()
        result = self.injector.inject()
        injected = len(result.get("updated", []))
        logger.info(f"[手动 flush] 完成，处理 {pending_count} 个 session，注入 {injected} 个文件")
        return {"flushed": pending_count, "injected": injected}

    async def process_recent_sessions(self) -> None:
        """首次运行：处理最近活跃的 session 文件。"""
        all_files = []
        acp_dir = Path(self.config.acp_sessions_dir)
        cli_dir = Path(self.config.cli_sessions_dir)

        if acp_dir.exists():
            for f in acp_dir.glob("*.json"):
                all_files.append((f, f.stat().st_mtime, "acp"))
        if cli_dir.exists():
            for f in cli_dir.glob("*/session-*.jsonl"):
                all_files.append((f, f.stat().st_mtime, "cli"))

        all_files.sort(key=lambda x: x[1], reverse=True)
        for path, _, source in all_files[:5]:
            new_msgs, total_count = self.indexer.get_new_messages(path, source)
            if new_msgs:
                logger.info(f"Initial processing: {path.name} ({len(new_msgs)} msgs)")
                await self._process_messages(new_msgs, path, source, total_count)

    async def _process_messages(
        self, messages: list[dict], session_path: Path,
        source: str, total_count: int,
    ) -> None:
        """处理一批消息：三层记忆流程。

        L3 成功后推进状态，后续 LLM 失败不阻塞——原始数据已持久化。
        """
        features = self.config.features

        # L3：写入清洗后的对话记录
        try:
            result = self.indexer.write_cleaned_messages(messages, session_path)
        except Exception as e:
            logger.error(f"[记忆守护] L3 写入异常: {e}")
            return
        if not result:
            return
        target_file, start_line = result
        self.indexer.commit_progress(session_path, total_count)

        # L1：索引短句
        if features.get("index_line", True):
            try:
                index_line = await self.summarizer.generate_index_line(messages)
                if index_line:
                    self.indexer.update_index(index_line, target_file, start_line, source=source)
                    logger.info(f"[L1 索引] {index_line[:60]}")
            except Exception as e:
                logger.error(f"[记忆守护] L1 索引生成异常: {e}")

        # L2：结构化摘要
        if features.get("summary", True):
            try:
                structured = await self.summarizer.generate_summary(messages)
                if structured:
                    self.indexer.append_structured_summary(structured, target_file)
                    logger.info(f"[L2 摘要] {len(structured)} 字符")
            except Exception as e:
                logger.error(f"[记忆守护] L2 摘要生成异常: {e}")

        # 分类记忆 → SQLite（带 embedding）
        if features.get("classify", True):
            try:
                classified = await self.summarizer.generate_classified_memories(messages)
                if classified:
                    embeddings = None
                    if self.embedder and self.embedder.available:
                        try:
                            texts = [m["text"] for m in classified]
                            embeddings = await self.embedder.embed_batch(texts)
                        except Exception as e:
                            logger.warning(f"[深度回忆] embedding 生成失败，降级写入: {e}")

                    count = self._write_memories_with_embeddings(
                        classified, session_path, embeddings
                    )
                    logger.info(f"[分类记忆] 提取 {len(classified)} 条，写入 {count} 条")
                    if count > 0:
                        result = self.injector.inject()
                        logger.info(f"[记忆注入] 更新 {len(result['updated'])} 个文件，{result['memories_count']} 条记忆")
            except Exception as e:
                logger.error(f"[记忆守护] 分类记忆提取异常: {e}")

        # 氛围快照 → SQLite
        if features.get("atmosphere", True):
            try:
                atmosphere = await self.summarizer.generate_atmosphere_snapshot(messages)
                if atmosphere:
                    today = datetime.now().strftime("%Y-%m-%d")
                    self.store.add_atmosphere(
                        session_id=session_path.stem,
                        date=today,
                        snapshot=atmosphere,
                    )
                    logger.info(f"[氛围快照] {atmosphere[:80]}...")
            except Exception as e:
                logger.error(f"[记忆守护] 氛围快照生成异常: {e}")

        # 状态快照 → SQLite
        if features.get("state_snapshot", True):
            try:
                state = await self.summarizer.generate_state_snapshot(messages)
                if state:
                    today = datetime.now().strftime("%Y-%m-%d")
                    self.store.add_state_snapshot(
                        session_id=session_path.stem,
                        date=today,
                        goal=state["goal"],
                        progress=state["progress"],
                        decisions=state["decisions"],
                        next_steps=state["next_steps"],
                        critical_context=state["critical_context"],
                    )
                    logger.info(f"[状态快照] goal={state['goal'][:60]}...")
                    result = self.injector.inject()
                    logger.info(f"[记忆注入] 状态快照触发更新 {len(result['updated'])} 个文件")
            except Exception as e:
                logger.error(f"[记忆守护] 状态快照生成异常: {e}")

    def _write_memories_with_embeddings(
        self, memories: list[dict], session_path: Path,
        embeddings: Optional[list[list[float]]] = None,
    ) -> int:
        """写入分类记忆，附带 embedding。"""
        if self.store is None:
            return 0
        count = 0
        for i, mem in enumerate(memories):
            category = mem.get("category", "")
            text = mem.get("text", "")
            if not category or not text:
                continue
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            try:
                self.store.add(
                    category=category,
                    text=text,
                    source_session=session_path.name,
                    embedding=embedding,
                )
                count += 1
            except Exception as e:
                logger.error(f"[记忆守护] 记忆写入失败: {e}")
        return count
