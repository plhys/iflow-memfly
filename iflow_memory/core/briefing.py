"""BriefingGenerator — 每日简报生成器。

从当天的 L1 索引、分类记忆和状态快照中提取信息，
生成一段 300-500 字的精炼工作回顾简报。
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import MemoryConfig
from ..store.db import MemoryStore
from .summarizer import Summarizer

logger = logging.getLogger("iflow-memory")

# 简报生成 prompt
BRIEFING_PROMPT = """请根据以下信息，生成一段今日工作简报。

要求：
1. 用第一人称"我"来写，像在回忆今天做了什么
2. 按主题分组，不要按时间逐条罗列
3. 包含：今天做了什么、关键决策和进展、遗留问题和下一步
4. 语气自然简洁，不要啰嗦，不要用"首先""其次"这种格式
5. 总长度控制在 300-500 字
6. 不要用 ## 标题，用 **加粗** 作为分段标题
7. 如果信息很少，就简短写，不要硬凑

今日对话索引：
{index_entries}

今日记忆条目：
{memories}

当前状态：
{state}"""


class BriefingGenerator:
    """每日简报生成器。"""

    def __init__(self, config: MemoryConfig, store: MemoryStore):
        self.config = config
        self.store = store
        self.summarizer = Summarizer(config)
        self._memory_dir = Path(config.memory_dir)

    async def generate_daily_briefing(self, date_str: str | None = None) -> Optional[str]:
        """生成指定日期的每日简报。默认为今天。

        数据来源：
        1. 当天的 index.md 条目（L1 索引）
        2. 当天的分类记忆（从 SQLite memories 表按日期筛选）
        3. 当天的状态快照（最后一条）

        输出格式：一段 300-500 字的精炼中文简报，第一人称，包含：
        - 今天做了什么（按主题分组，不是按时间罗列）
        - 关键决策和进展
        - 遗留问题和下一步

        如果没有 LLM API 可用，使用模板拼接作为 fallback。
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        briefing_file = self._memory_dir / f"briefing-{date_str}.md"
        if briefing_file.exists():
            text = briefing_file.read_text(encoding="utf-8").strip()
            if text:
                logger.debug(f"[每日简报] 已存在: {briefing_file}")
                return text

        # 收集数据
        index_entries = self._get_today_index_entries(date_str)
        memories = self._get_today_memories(date_str)
        state = self._get_today_state(date_str)

        # 没有任何数据就不生成
        if not index_entries and not memories:
            logger.debug(f"[每日简报] {date_str} 无数据，跳过")
            return None

        # 尝试 LLM 生成
        context = self._build_context(index_entries, memories, state)
        briefing = await self._generate_with_llm(context)

        # LLM 失败则 fallback
        if not briefing:
            briefing = self._generate_fallback(index_entries, memories, state)

        if briefing:
            briefing = briefing.strip()
            briefing_file.write_text(briefing, encoding="utf-8")
            logger.info(f"[每日简报] 生成完成: {briefing_file} ({len(briefing)} 字符)")

        return briefing

    async def _generate_with_llm(self, context: dict[str, str]) -> Optional[str]:
        """调用 LLM 生成简报。复用 Summarizer 的 _call_llm 机制。"""
        preset = self.config.get_active_model()
        try:
            prompt = BRIEFING_PROMPT.replace("{index_entries}", context.get("index_entries", "（无）"))
            prompt = prompt.replace("{memories}", context.get("memories", "（无）"))
            prompt = prompt.replace("{state}", context.get("state", "（无）"))
            return await self.summarizer._call_llm(preset, prompt, max_tokens=1000)
        except Exception as e:
            logger.warning(f"[每日简报] LLM 生成失败，使用 fallback: {e}")
            return None

    def _generate_fallback(
        self,
        index_entries: list[str],
        memories: list[dict],
        state: Optional[dict],
    ) -> str:
        """无 LLM 时的模板拼接 fallback。"""
        lines: list[str] = []

        if index_entries:
            lines.append("**今日对话**")
            # 按主题去重，取前 10 条
            seen: set[str] = set()
            count = 0
            for entry in index_entries:
                # 去掉时间前缀和文件引用
                clean = re.sub(r"^\d{2}:\d{2}\s+", "", entry)
                clean = re.sub(r"\s*→\s*\S+$", "", clean).strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    lines.append(f"- {clean}")
                    count += 1
                    if count >= 10:
                        break
            lines.append("")

        if memories:
            # 按分类分组
            grouped: dict[str, list[str]] = {}
            for mem in memories:
                cat = mem.get("category", "其他")
                text = mem.get("text", "")
                if text:
                    grouped.setdefault(cat, []).append(text)

            cat_names = {
                "identity": "身份", "preference": "偏好",
                "entity": "知识", "event": "事件",
                "insight": "经验", "correction": "纠正",
            }
            for cat, texts in grouped.items():
                display = cat_names.get(cat, cat)
                lines.append(f"**今日{display}**")
                for t in texts[:5]:
                    lines.append(f"- {t}")
                lines.append("")

        if state:
            goal = state.get("goal", "")
            next_steps = state.get("next_steps", "")
            if goal and goal != "无":
                lines.append(f"**当前目标**：{goal}")
            if next_steps and next_steps != "无":
                lines.append(f"**下一步**：{next_steps}")
            if goal or next_steps:
                lines.append("")

        return "\n".join(lines).strip() if lines else ""

    def _get_today_index_entries(self, date_str: str) -> list[str]:
        """从 index.md 读取指定日期的条目。"""
        index_file = self._memory_dir / "index.md"
        if not index_file.exists():
            return []

        entries: list[str] = []
        in_target_date = False

        try:
            with open(index_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("## "):
                        heading_date = stripped.lstrip("# ").strip()
                        in_target_date = heading_date == date_str
                        continue
                    if in_target_date and stripped.startswith("- "):
                        entries.append(stripped.lstrip("- "))
        except OSError:
            pass

        return entries

    def _get_today_memories(self, date_str: str) -> list[dict]:
        """从 SQLite 读取指定日期创建的记忆。"""
        try:
            rows = self.store._conn.execute(
                """SELECT category, text, created_at
                   FROM memories
                   WHERE archived = 0
                     AND created_at LIKE ?
                   ORDER BY created_at DESC
                   LIMIT 50""",
                (f"{date_str}%",),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[每日简报] 读取记忆失败: {e}")
            return []

    def _get_today_state(self, date_str: str) -> Optional[dict]:
        """获取指定日期最新的状态快照。"""
        try:
            rows = self.store._conn.execute(
                """SELECT goal, progress, decisions, next_steps, critical_context
                   FROM state_snapshots
                   WHERE date = ?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (date_str,),
            ).fetchall()
            if rows:
                return dict(rows[0])
        except Exception as e:
            logger.warning(f"[每日简报] 读取状态快照失败: {e}")
        return None

    def _build_context(
        self,
        index_entries: list[str],
        memories: list[dict],
        state: Optional[dict],
    ) -> dict[str, str]:
        """拼接 LLM prompt 的上下文。"""
        parts: dict[str, str] = {}

        if index_entries:
            parts["index_entries"] = "\n".join(
                f"- {e}" for e in index_entries[:30]
            )
        else:
            parts["index_entries"] = "（无）"

        if memories:
            mem_lines = []
            for m in memories[:20]:
                mem_lines.append(f"[{m.get('category', '?')}] {m.get('text', '')}")
            parts["memories"] = "\n".join(mem_lines)
        else:
            parts["memories"] = "（无）"

        if state:
            state_lines = [
                f"目标：{state.get('goal', '无')}",
                f"进度：{state.get('progress', '无')}",
                f"决策：{state.get('decisions', '无')}",
                f"下一步：{state.get('next_steps', '无')}",
            ]
            parts["state"] = "\n".join(state_lines)
        else:
            parts["state"] = "（无）"

        return parts

    async def close(self) -> None:
        """关闭资源。"""
        await self.summarizer.close()
