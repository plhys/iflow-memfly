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
from .briefing import BriefingGenerator
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
        self.briefing = BriefingGenerator(config, self.store, summarizer=self.summarizer)
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
        self._llm_retry_queue: list[tuple[list[dict], Path, str]] = []  # LLM 失败重试队列

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
                self.briefing.store = self.store
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
        await self.briefing.close()
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
        """周期性维护：归档冷记忆、做梦整合、检查手写区大小、生成对话回顾。"""
        archived = self.store.archive_cold(min_age_days=5)
        if archived:
            logger.info(f"[维护] 归档 {archived} 条冷记忆")
            self.injector.inject()

        # 做梦整合：合并高度相似的记忆，减少冗余
        # 灵感来源：Claude Code autoDream
        # 两种模式：Jaccard 快速去重（默认）或 LLM 深度整合（开关控制）
        if self.config.features.get("llm_dream", False):
            consolidated = await self._dream_consolidate_llm()
            if consolidated:
                logger.info(f"[做梦整合·LLM] 执行 {consolidated} 个整合动作")
                self.injector.inject()
        else:
            consolidated = self._dream_consolidate()
            if consolidated:
                logger.info(f"[做梦整合] 合并 {consolidated} 组重复记忆")
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

        # 每日简报生成
        if self.config.features.get("daily_briefing", True):
            await self._generate_daily_briefing()

        self.store.checkpoint()

    def _dream_consolidate(self, similarity_threshold: float = 0.75) -> int:
        """做梦整合：扫描活跃记忆，合并高度相似的条目。

        策略：
        - 同类别内，Jaccard bigram 相似度 >= threshold 的记忆视为重复
        - 保留 access_count 更高的那条（更常被引用 = 更重要）
        - 被合并的记忆归档，不删除（可恢复）
        - 每次最多处理 50 组，避免一次性改太多

        这不是简单的去重——它模拟了人类睡眠时大脑整理记忆的过程：
        把碎片化的相似记忆归并为一条更有代表性的记忆。
        """
        from ..store.db import _normalize_text, _jaccard_similarity

        consolidated = 0
        max_groups = 50

        for category in ("entity", "event", "insight", "correction"):
            if consolidated >= max_groups:
                break

            rows = self.store.get_by_category(category, limit=200)
            if len(rows) < 2:
                continue

            # 建立归一化文本索引
            normed = [(r, _normalize_text(r["text"])) for r in rows]
            merged_ids: set[int] = set()

            for i in range(len(normed)):
                if normed[i][0]["id"] in merged_ids:
                    continue
                for j in range(i + 1, len(normed)):
                    if normed[j][0]["id"] in merged_ids:
                        continue
                    sim = _jaccard_similarity(normed[i][1], normed[j][1])
                    if sim >= similarity_threshold:
                        # 保留 access_count 更高的
                        keep, discard = normed[i][0], normed[j][0]
                        if discard["access_count"] > keep["access_count"]:
                            keep, discard = discard, keep
                        merged_ids.add(discard["id"])
                        consolidated += 1
                        logger.debug(
                            f"[做梦整合] 合并 #{discard['id']} -> #{keep['id']} "
                            f"(sim={sim:.2f})"
                        )
                        if consolidated >= max_groups:
                            break

            # 批量归档被合并的记忆（通过公共 API，不直接操作 _conn）
            if merged_ids:
                self.store.archive_by_ids(list(merged_ids))

        return consolidated

    async def _dream_consolidate_llm(self) -> int:
        """LLM 深度记忆整合：用 LLM 分析同类别记忆，执行合并/归档/升级。

        比 Jaccard 更智能：能理解语义相似性（如"深色主题"="暗色模式"），
        能识别过时记忆（旧版本号 vs 新版本号），能合并碎片为完整描述。

        代价：每个类别消耗约 2000-5000 token，总计约 1-3 万 token。
        因此默认关闭，由用户主动开启。
        """
        total_actions = 0

        for category in ("identity", "preference", "entity", "event", "insight", "correction"):
            rows = self.store.get_by_category(category, limit=200)
            if len(rows) < 2:
                continue

            try:
                actions = await self.summarizer.consolidate_memories(category, rows)
            except Exception as e:
                logger.error(f"[做梦整合·LLM] {category} 整合异常: {e}")
                continue

            if not actions:
                continue

            # 执行 actions
            all_discard_ids: list[int] = []
            for action in actions:
                action_type = action["type"]
                discard_ids = action["discard_ids"]
                reason = action.get("reason", "")

                if action_type == "merge":
                    keep_id = action["keep_id"]
                    logger.info(
                        f"[做梦整合·LLM] {category} 合并: "
                        f"保留 #{keep_id}, 归档 {discard_ids} — {reason}"
                    )
                    all_discard_ids.extend(discard_ids)

                elif action_type == "obsolete":
                    logger.info(
                        f"[做梦整合·LLM] {category} 过时: "
                        f"归档 {discard_ids} — {reason}"
                    )
                    all_discard_ids.extend(discard_ids)

                elif action_type == "upgrade":
                    new_text = action["new_text"]
                    logger.info(
                        f"[做梦整合·LLM] {category} 升级: "
                        f"归档 {discard_ids}, 新建 '{new_text[:60]}' — {reason}"
                    )
                    # 先写入新记忆，再归档旧的
                    try:
                        self.store.add(
                            category=category,
                            text=new_text,
                            source_session="dream_consolidate",
                        )  # return value unused here
                    except Exception as e:
                        logger.error(f"[做梦整合·LLM] 新记忆写入失败: {e}")
                        continue
                    all_discard_ids.extend(discard_ids)

                total_actions += 1

            # 批量归档
            if all_discard_ids:
                archived = self.store.archive_by_ids(all_discard_ids)
                logger.info(f"[做梦整合·LLM] {category}: 归档 {archived} 条记忆")

        return total_actions

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

    async def _generate_daily_briefing(self) -> None:
        """生成今日简报（如果尚未生成）。"""
        today = datetime.now().strftime("%Y-%m-%d")
        memory_dir = Path(self.config.memory_dir)
        briefing_file = memory_dir / f"briefing-{today}.md"
        if briefing_file.exists():
            return

        # 检查今天是否有对话记录（index.md 中有当天条目）
        index_file = memory_dir / "index.md"
        if not index_file.exists():
            return
        has_today = False
        try:
            with open(index_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip() == f"## {today}":
                        has_today = True
                        break
        except OSError:
            return
        if not has_today:
            return

        logger.info(f"[每日简报] 开始生成 {today} 简报")
        try:
            briefing = await self.briefing.generate_daily_briefing(today)
            if briefing:
                self.injector.inject()
                logger.info(f"[每日简报] {today} 简报已生成并注入")
        except Exception as e:
            logger.error(f"[每日简报] 生成失败: {e}")

    async def _flush_pending(self) -> None:
        """处理所有待处理的消息。"""
        if not self._pending:
            pass
        else:
            batch, self._pending = self._pending, {}
            for messages, session_path, source, total_count in batch.values():
                await self._process_messages(messages, session_path, source, total_count)

        # 处理 LLM 重试队列（L3 已写入，只重试 LLM 步骤）
        if self._llm_retry_queue:
            retry_batch = self._llm_retry_queue[:]
            self._llm_retry_queue.clear()
            logger.info(f"[LLM 重试] 处理 {len(retry_batch)} 个待重试 session")
            for messages, session_path, source in retry_batch:
                await self._retry_llm_steps(messages, session_path, source)

    async def _retry_llm_steps(
        self, messages: list[dict], session_path: Path, source: str,
    ) -> None:
        """重试 LLM 步骤（L3 已写入，只做分类记忆提取）。"""
        features = self.config.features
        llm_ok = 0

        # 只重试分类记忆（最重要的 LLM 步骤）
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
                            logger.warning(f"[LLM 重试] embedding 失败: {e}")
                    count = self._write_memories_with_embeddings(
                        classified, session_path, embeddings,
                    )
                    logger.info(f"[LLM 重试] 成功提取 {len(classified)} 条，写入 {count} 条")
                    llm_ok += 1
                    self.injector.inject()
            except Exception as e:
                logger.error(f"[LLM 重试] 分类记忆提取异常: {e}")

        if llm_ok == 0 and len(self._llm_retry_queue) < 20:
            self._llm_retry_queue.append((messages, session_path, source))
            logger.warning(f"[LLM 重试] 仍然失败，重新入队")

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

    @staticmethod
    def _is_heartbeat_only(messages: list[dict]) -> bool:
        """判断一批消息是否全部是心跳消息（无实质内容）。"""
        for msg in messages:
            text = msg.get("text", "").strip()
            if not text:
                continue
            upper = text.upper()
            # 用户侧：包含 HEARTBEAT 指令
            if msg["role"] in ("user", "human") and "HEARTBEAT" in upper:
                continue
            # AI 侧：仅回复 HEARTBEAT_OK
            if msg["role"] in ("model", "assistant") and upper == "HEARTBEAT_OK":
                continue
            # 有任何非心跳内容，就不是纯心跳
            return False
        return True

    async def _process_messages(
        self, messages: list[dict], session_path: Path,
        source: str, total_count: int,
    ) -> None:
        """处理一批消息：三层记忆流程。

        L3 成功后推进状态，后续 LLM 失败不阻塞——原始数据已持久化。
        """
        features = self.config.features

        # 心跳过滤：纯心跳消息只做 L3 记录，跳过所有 LLM 处理
        if self._is_heartbeat_only(messages):
            try:
                result = self.indexer.write_cleaned_messages(messages, session_path)
                if result:
                    self.indexer.commit_progress(session_path, total_count)
                    logger.debug(f"[心跳过滤] 跳过 LLM 处理: {session_path.name}")
            except Exception as e:
                logger.error(f"[记忆守护] 心跳 L3 写入异常: {e}")
            return

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

        # LLM 步骤：跟踪成功数，全部失败则加入重试队列
        llm_ok = 0

        # L1：索引短句
        if features.get("index_line", True):
            try:
                index_line = await self.summarizer.generate_index_line(messages)
                if index_line:
                    self.indexer.update_index(index_line, target_file, start_line, source=source)
                    logger.info(f"[L1 索引] {index_line[:60]}")
                    llm_ok += 1
            except Exception as e:
                logger.error(f"[记忆守护] L1 索引生成异常: {e}")

        # L2：结构化摘要
        if features.get("summary", True):
            try:
                structured = await self.summarizer.generate_summary(messages)
                if structured:
                    self.indexer.append_structured_summary(structured, target_file)
                    logger.info(f"[L2 摘要] {len(structured)} 字符")
                    llm_ok += 1
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
                        classified, session_path, embeddings,
                        source_file=target_file.name,
                        source_line=start_line,
                    )
                    logger.info(f"[分类记忆] 提取 {len(classified)} 条，写入 {count} 条")
                    llm_ok += 1
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
                    llm_ok += 1
                    result = self.injector.inject()
                    logger.info(f"[记忆注入] 状态快照触发更新 {len(result['updated'])} 个文件")
            except Exception as e:
                logger.error(f"[记忆守护] 状态快照生成异常: {e}")

        # LLM 全部失败时加入重试队列（L3 已写入，只需重试 LLM 步骤）
        if llm_ok == 0:
            if len(self._llm_retry_queue) < 20:  # 防止无限堆积
                self._llm_retry_queue.append((messages, session_path, source))
                logger.warning(
                    f"[记忆守护] LLM 全部失败，加入重试队列 "
                    f"(队列长度: {len(self._llm_retry_queue)})"
                )

    def _write_memories_with_embeddings(
        self, memories: list[dict], session_path: Path,
        embeddings: Optional[list[list[float]]] = None,
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
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
                row_id, is_new = self.store.add(
                    category=category,
                    text=text,
                    source_session=session_path.name,
                    embedding=embedding,
                    source_file=source_file,
                    source_line=source_line,
                )
                count += 1
                # 知识图谱：仅为新建的记忆创建关联
                if is_new and self.config.features.get("knowledge_graph", True):
                    try:
                        self.store.create_links_for_memory(
                            row_id, embedding=embedding,
                        )
                    except Exception as e:
                        logger.warning(f"[知识图谱] 链接创建失败: {e}")
            except Exception as e:
                logger.error(f"[记忆守护] 记忆写入失败: {e}")
        return count
