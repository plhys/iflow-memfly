"""MemoryInjector — automatically updates AGENTS.md with top memories from SQLite.

Reads the highest-ranked memories from MemoryStore and injects a structured
"记忆系统" section into one or more AGENTS.md files. This section is regenerated
on each inject() call, replacing any previous version.

The injected section is delimited by a known heading marker so it can be
cleanly replaced on subsequent runs without touching hand-written content.
"""

import logging
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..store.db import MemoryStore

logger = logging.getLogger("iflow-memory")

_tmp_dir = tempfile.gettempdir()

# Heading that marks the start of the auto-generated section
SECTION_MARKER = "### 记忆系统（自动生成，勿手动编辑）"

# Display names for each category (in injection order)
CATEGORY_DISPLAY: list[tuple[str, str]] = [
    ("identity", "身份"),
    ("preference", "偏好"),
    ("correction", "纠正"),
    ("entity", "知识"),
    ("event", "事件"),
    ("insight", "经验"),
]

# Default AGENTS.md paths
DEFAULT_AGENTS_PATHS: list[str] = [
    str(Path.home() / ".iflow" / "AGENTS.md"),
]


class MemoryInjector:
    """Injects top memories from MemoryStore into AGENTS.md files."""

    def __init__(
        self,
        store: MemoryStore,
        agents_md_paths: list[str | Path] | None = None,
    ):
        """
        Args:
            store: MemoryStore instance to read memories from.
            agents_md_paths: list of AGENTS.md file paths to update.
                Default: [~/.iflow/AGENTS.md]
        """
        self.store = store
        if agents_md_paths is None:
            self.agents_md_paths = [Path(p) for p in DEFAULT_AGENTS_PATHS]
        else:
            self.agents_md_paths = [Path(p) for p in agents_md_paths]

    def inject(self) -> dict:
        """Update all AGENTS.md files with current top memories.

        Returns:
            dict with keys:
                "updated": list of path strings that were successfully updated
                "memories_count": number of memories included in the section
        """
        updated: list[str] = []
        memories_count = 0
        for path in self.agents_md_paths:
            channel = self._channel_for_path(path)
            section = self._build_memory_section(channel_filter=channel)
            memories_count = max(memories_count, sum(
                1 for line in section.splitlines() if line.startswith("- ")
            ))
            if self._update_agents_md(path, section):
                updated.append(str(path))

        result = {"updated": updated, "memories_count": memories_count}
        logger.info(
            f"Inject complete: {len(updated)} file(s) updated, "
            f"{memories_count} memories injected"
        )
        return result

    @staticmethod
    def _channel_for_path(path: Path) -> str:
        """Determine channel filter based on AGENTS.md path.

        - Paths under .iflow-bot/ → "acp" (feishu gateway)
        - Paths under .iflow/ (CLI) → "cli"
        - Otherwise → "" (no filter)
        """
        parts = str(path)
        if ".iflow-bot" in parts:
            return "acp"
        elif "/.iflow/" in parts or parts.endswith("/.iflow/AGENTS.md"):
            return "cli"
        return ""

    def _build_memory_section(self, channel_filter: str = "") -> str:
        """Build the memory section text from store.get_top_memories().

        Returns a markdown string starting with the SECTION_MARKER heading,
        containing categorized bullet lists of memories.
        """
        memories = self.store.get_top_memories()
        stats = self.store.stats()

        now = datetime.now().astimezone()
        tz_local = now.tzinfo
        now_str = now.strftime("%Y-%m-%d %H:%M")
        total = stats.get("total", 0)
        archived = stats.get("archived", 0)

        lines: list[str] = []

        # Header
        lines.append(SECTION_MARKER)
        lines.append(
            f"> 最后更新: {now_str} | 记忆总数: {total + archived} | 活跃: {total}"
        )
        lines.append("")
        lines.append("> **开机自检指令**：新 session 启动时，在首次回复前执行以下步骤：")
        lines.append("> 1. 运行 `ps aux | grep 'iflow_memory start' | grep -v grep && echo 'daemon=OK' || echo 'daemon=DOWN'` 检查 daemon 进程")
        lines.append("> 2. 用 `search_memory` 搜索一个关键词（如「最近对话」），确认向量搜索可用")
        lines.append(f"> 3. 如果 daemon=DOWN，运行 `nohup python3 -m iflow_memory start > {_tmp_dir}/iflow-memory.log 2>&1 &` 恢复")
        lines.append("> 4. 如果 search_memory 返回空或报错，检查 embedding API 和 sqlite-vec 状态")
        lines.append("> 5. 自检通过后，用 `search_memory` 检索最近对话的关键上下文，接续之前的工作")
        lines.append("> 不要等用户提醒，不要跳过自检。")
        lines.append("")

        # 时间感知锚点：让 AI 知道"现在几点"和"距上次对话多久"
        time_anchor = self._build_time_anchor(now)
        if time_anchor:
            lines.append(time_anchor)
            lines.append("")

        # Recent conversation context from index.md
        recent = self._get_recent_index(50, channel_filter=channel_filter)
        if recent:
            lines.append("**最近对话**")
            for entry in recent:
                if entry.startswith("## "):
                    # Date heading — render as sub-label
                    lines.append(f"*{entry.lstrip('# ').strip()}*")
                else:
                    lines.append(f"- {entry}")
            lines.append("")

        # 上次对话回顾（找最近有 recap 的日期，不固定昨天）
        recap_result = self._get_last_recap()
        if recap_result:
            recap_text, recap_date, days_ago = recap_result
            if days_ago == 1:
                label = f"**我的上次工作回忆（{recap_date}，昨天）**"
            else:
                label = f"**我的上次工作回忆（{recap_date}，{days_ago}天前）**"
            lines.append(label)
            for recap_line in recap_text.splitlines():
                if recap_line.strip():
                    lines.append(recap_line)
            lines.append("")

        # 对话氛围记忆（最近一次）
        atmosphere = self._get_last_atmosphere()
        if atmosphere:
            atm_text, atm_date, atm_created_at = atmosphere
            # 尝试从 created_at 提取精确时间（转换为本地时区）
            atm_time = ""
            if atm_created_at:
                try:
                    atm_dt = datetime.fromisoformat(atm_created_at)
                    atm_local = atm_dt.astimezone(tz_local)
                    atm_time = atm_local.strftime("%H:%M")
                    atm_date = atm_local.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
            if atm_time:
                lines.append(f"**我的上次对话记忆（{atm_date} {atm_time}）**")
            else:
                lines.append(f"**我的上次对话记忆（{atm_date}）**")
            for atm_line in atm_text.splitlines():
                if atm_line.strip():
                    lines.append(atm_line)
            lines.append("")

        # 状态快照（结构化上下文检查点）
        state = self._get_last_state_snapshot()
        if state:
            state_date = state.get("date", "")
            state_created = state.get("created_at", "")
            # 尝试从 created_at 提取精确时间
            state_time = ""
            if state_created:
                try:
                    state_dt = datetime.fromisoformat(state_created)
                    state_local = state_dt.astimezone(tz_local)
                    state_time = state_local.strftime("%H:%M")
                    state_date = state_local.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
            if state_time:
                lines.append(f"**上次工作状态存档（{state_date} {state_time}）**")
            else:
                lines.append(f"**上次工作状态存档（{state_date}）**")
            lines.append(f"- 目标：{state.get('goal', '无')}")
            lines.append(f"- 进度：{state.get('progress', '无')}")
            lines.append(f"- 决策：{state.get('decisions', '无')}")
            lines.append(f"- 下一步：{state.get('next_steps', '无')}")
            lines.append(f"- 关键上下文：{state.get('critical_context', '无')}")
            lines.append("")

        if not memories:
            lines.append("*暂无记忆*")
            lines.append("")
            return "\n".join(lines)

        # Group memories by category
        grouped: dict[str, list[str]] = defaultdict(list)
        for mem in memories:
            grouped[mem["category"]].append(mem["text"])

        # Emit each category in display order, skipping empty ones
        for cat_key, cat_name in CATEGORY_DISPLAY:
            texts = grouped.get(cat_key)
            if not texts:
                continue
            lines.append(f"**{cat_name}**")
            for text in texts:
                # Ensure single-line: collapse any internal newlines
                clean = text.replace("\n", " ").strip()
                lines.append(f"- {clean}")
            lines.append("")

        return "\n".join(lines)

    def _get_recent_index(self, count: int = 3, channel_filter: str = "") -> list[str]:
        """Read the most recent L1 index entries from index.md.

        Returns a list of entry strings. Date headings (## YYYY-MM-DD) are
        preserved as-is so cross-day entries are visually separated.

        Args:
            count: max number of entries to return.
            channel_filter: if set ("cli" or "acp"), only return entries
                matching that tag. Entries without a tag are included for all.
        """
        # Locate index.md via store's db_path parent (= memory_dir)
        memory_dir = self.store.db_path.parent
        index_file = memory_dir / "index.md"
        if not index_file.exists():
            return []
        try:
            raw_lines: list[str] = []
            entry_count = 0
            with open(index_file) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("## "):
                        # Date heading — include it but don't count toward limit
                        raw_lines.append(stripped)
                    elif stripped.startswith("- "):
                        # Check channel tag filter
                        if channel_filter:
                            has_own_tag = stripped.endswith(f"[{channel_filter}]")
                            has_no_tag = not stripped.endswith("[cli]") and not stripped.endswith("[acp]")
                            if not has_own_tag and not has_no_tag:
                                continue
                        # Strip the channel tag before output
                        display = stripped.lstrip("- ")
                        for tag in (" [cli]", " [acp]"):
                            if display.endswith(tag):
                                display = display[: -len(tag)]
                                break
                        raw_lines.append(display)
                        entry_count += 1
                        if entry_count >= count:
                            break
            # Remove trailing date heading if no entries follow it
            while raw_lines and raw_lines[-1].startswith("## "):
                raw_lines.pop()
            return raw_lines
        except OSError:
            return []

    def _build_time_anchor(self, now: datetime) -> str | None:
        """构建时间感知锚点，让 AI 知道"现在几点"和"距上次对话多久"。

        Returns:
            一段 blockquote 格式的时间锚点文本，或 None。
        """
        now_local_str = now.strftime("%Y-%m-%d %H:%M")

        # 从最近的氛围快照获取上次对话时间
        rows = self.store.get_latest_atmosphere(limit=1)
        if not rows:
            return f"> **时间感知**：现在是 {now_local_str}。没有找到上次对话记录。"

        created_at_str = rows[0].get("created_at", "")
        if not created_at_str:
            return f"> **时间感知**：现在是 {now_local_str}。"

        try:
            last_dt = datetime.fromisoformat(created_at_str)
            # 确保时区一致
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=now.tzinfo)
            delta = now - last_dt
            total_seconds = int(delta.total_seconds())

            if total_seconds < 0:
                gap_desc = "刚刚"
            elif total_seconds < 60:
                gap_desc = f"{total_seconds}秒前"
            elif total_seconds < 3600:
                minutes = total_seconds // 60
                gap_desc = f"{minutes}分钟前"
            elif total_seconds < 86400:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                if minutes > 0:
                    gap_desc = f"{hours}小时{minutes}分钟前"
                else:
                    gap_desc = f"{hours}小时前"
            else:
                days = total_seconds // 86400
                hours = (total_seconds % 86400) // 3600
                if hours > 0:
                    gap_desc = f"{days}天{hours}小时前"
                else:
                    gap_desc = f"{days}天前"

            last_time_str = last_dt.astimezone(now.tzinfo).strftime("%Y-%m-%d %H:%M")
            return (
                f"> **时间感知**：现在是 {now_local_str}。"
                f"上次对话结束于 {last_time_str}（{gap_desc}）。"
            )
        except (ValueError, TypeError):
            return f"> **时间感知**：现在是 {now_local_str}。"

    def _get_last_recap(self) -> tuple[str, str, int] | None:
        """查找最近一次有对话的工作回顾（往回搜最多 7 天）。

        Returns:
            (recap_text, date_str, days_ago) 或 None
        """
        memory_dir = self.store.db_path.parent
        today = datetime.now().date()
        for days_ago in range(1, 8):
            target_date = today - timedelta(days=days_ago)
            date_str = target_date.strftime("%Y-%m-%d")
            recap_file = memory_dir / f"recap-{date_str}.md"
            if recap_file.exists():
                try:
                    text = recap_file.read_text(encoding="utf-8").strip()
                    if text:
                        return (text, date_str, days_ago)
                except OSError:
                    continue
        return None

    def _get_last_atmosphere(self) -> tuple[str, str, str] | None:
        """读取最近一条对话氛围快照。

        Returns:
            (snapshot_text, date_str, created_at_iso) 或 None
        """
        rows = self.store.get_latest_atmosphere()
        if rows:
            row = rows[0]
            return (row["snapshot"], row["date"], row.get("created_at", ""))
        return None

    def _get_last_state_snapshot(self) -> dict | None:
        """读取最近一条状态快照。

        Returns:
            dict with keys: goal, progress, decisions, next_steps, critical_context, date, created_at.
            None if not available.
        """
        rows = self.store.get_latest_state_snapshot(limit=1)
        if rows:
            return rows[0]
        return None

    def _update_agents_md(self, path: Path, section: str) -> bool:
        """Update a single AGENTS.md file with the new memory section.

        Logic:
        1. Read the file content.
        2. Look for the SECTION_MARKER heading.
        3. If found: replace from that marker to the next ### heading (or EOF).
        4. If not found: append the section at the end.
        5. Write back.

        Returns True if the file was updated, False if skipped.
        """
        if not path.exists():
            logger.debug(f"AGENTS.md not found, skipping: {path}")
            return False

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to read {path}: {e}")
            return False

        # Ensure section ends with a single trailing newline
        section_text = section.rstrip("\n") + "\n"

        # Pattern: from our marker to the next ### heading or end of file
        # re.DOTALL so . matches newlines
        marker_escaped = re.escape(SECTION_MARKER)
        pattern = re.compile(
            rf"({marker_escaped}).*?(?=\n###\s|\Z)",
            re.DOTALL,
        )

        match = pattern.search(content)
        if match:
            # Replace existing section using string slicing (not re.sub)
            # to avoid backslash escape issues in section_text
            new_content = content[:match.start()] + section_text + content[match.end():]
            logger.info(f"Replaced existing memory section in {path}")
        else:
            # Append at end
            # Ensure there's a blank line before the new section
            if content and not content.endswith("\n\n"):
                if content.endswith("\n"):
                    content += "\n"
                else:
                    content += "\n\n"
            new_content = content + section_text
            logger.info(f"Appended new memory section to {path}")

        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to write {path}: {e}")
            return False

        return True
