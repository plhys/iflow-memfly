"""Session file watcher — monitors ACP and CLI session files for changes."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Callable, Awaitable, Optional

logger = logging.getLogger("iflow-memory")


class SessionWatcher:
    """轻量级文件变化监听，基于 polling（不依赖 inotify）。"""

    def __init__(
        self,
        acp_dir: Path,
        cli_dir: Path,
        on_change: Callable[[Path, dict], Awaitable[None]],
        poll_interval: float = 10.0,
    ):
        self.acp_dir = Path(acp_dir)
        self.cli_dir = Path(cli_dir)
        self.on_change = on_change
        self.poll_interval = poll_interval
        self._file_states: dict[str, float] = {}  # path -> last mtime
        self._running = False

    async def start(self) -> None:
        """启动监听循环。"""
        self._running = True
        # 初始化文件状态快照
        self._snapshot_all()
        logger.info(
            f"Watcher started: acp={self.acp_dir}, cli={self.cli_dir}, "
            f"poll={self.poll_interval}s, tracking {len(self._file_states)} files"
        )
        while self._running:
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"Watcher poll error: {e}")
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Watcher stopped")

    def _snapshot_all(self) -> None:
        """扫描所有 session 文件，记录 mtime。"""
        for path in self._iter_session_files():
            key = str(path)
            try:
                self._file_states[key] = path.stat().st_mtime
            except OSError:
                pass

    def _iter_session_files(self):
        """遍历所有 session 文件（符号链接去重）。"""
        seen: set[str] = set()
        # ACP sessions: *.json
        if self.acp_dir.exists():
            for f in self.acp_dir.glob("*.json"):
                real = str(f.resolve())
                if real not in seen:
                    seen.add(real)
                    yield f
        # CLI sessions: session-*.jsonl (遍历所有工作目录子目录)
        if self.cli_dir.exists():
            for f in self.cli_dir.glob("*/session-*.jsonl"):
                real = str(f.resolve())
                if real not in seen:
                    seen.add(real)
                    yield f

    async def _poll(self) -> None:
        """检查文件变化。"""
        current_files: dict[str, float] = {}
        for path in self._iter_session_files():
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            current_files[key] = mtime
            old_mtime = self._file_states.get(key)
            if old_mtime is None or mtime > old_mtime:
                # 新文件或已修改
                source = "acp" if self.acp_dir in path.parents or path.parent == self.acp_dir else "cli"
                try:
                    await self.on_change(path, {"source": source, "is_new": old_mtime is None})
                except Exception as e:
                    logger.error(f"on_change callback error for {path}: {e}")

        self._file_states = current_files
