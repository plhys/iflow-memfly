"""Indexer — cleans raw session data and generates memory index."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iflow-memory")

# 清洗规则：跳过这些内容模式
STRIP_PATTERNS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<environment_details>.*?</environment_details>", re.DOTALL),
    re.compile(r"<history_context>.*?</history_context>", re.DOTALL),
    re.compile(r"\[language\].*?(?=\n\n|\Z)", re.DOTALL),
    re.compile(r"<context>.*?</context>", re.DOTALL),
]

# tool call 相关的 part 类型
TOOL_PART_TYPES = {"functionCall", "functionResponse", "tool_use", "tool_result"}


class SessionParser:
    """解析不同格式的 session 文件。"""

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗系统标签，返回清理后的文本。"""
        for pattern in STRIP_PATTERNS:
            text = pattern.sub("", text)
        return text.strip()

    @staticmethod
    def _extract_content_text(content) -> str:
        """从 content（str 或 list of blocks）中提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        return ""

    @staticmethod
    def parse_acp(path: Path) -> list[dict]:
        """解析 ACP session JSON，返回 [{role, text}]。"""
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse ACP session {path}: {e}")
            return []

        messages = []
        for msg in data.get("chatHistory", []):
            role = msg.get("role", "")
            if role == "system":
                continue
            parts = msg.get("parts", [])
            text_parts = []
            for p in parts:
                if isinstance(p, dict):
                    if any(k in p for k in TOOL_PART_TYPES):
                        continue
                    if "text" in p:
                        text_parts.append(p["text"])
                elif isinstance(p, str):
                    text_parts.append(p)
            text = SessionParser._clean_text("".join(text_parts))
            if text:
                messages.append({"role": role, "text": text})
        return messages

    @staticmethod
    def parse_cli(path: Path) -> list[dict]:
        """解析 CLI session JSONL，返回 [{role, text}]。"""
        messages = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg_type = entry.get("type", "")
                    if msg_type not in ("human", "assistant"):
                        continue
                    role = "user" if msg_type == "human" else "model"
                    raw = entry.get("message", {}).get("content", "")
                    text = SessionParser._extract_content_text(raw).strip()
                    if not text:
                        continue
                    if role == "model":
                        text = SessionParser._clean_text(text)
                    if text:
                        messages.append({"role": role, "text": text})
        except OSError as e:
            logger.warning(f"Failed to parse CLI session {path}: {e}")
        return messages


class Indexer:
    """管理记忆索引文件。"""

    def __init__(self, memory_dir: Path, store=None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.memory_dir / "index.md"
        self.store = store  # Optional MemoryStore instance for SQLite dual-write
        # 增量状态：记录每个 session 文件已处理到第几条消息
        self._state_file = self.memory_dir / ".indexer-state.json"
        self._state: dict = self._load_state()

    def _load_state(self) -> dict:
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"processed": {}}

    def _save_state(self) -> None:
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, ensure_ascii=False)

    def get_new_messages(self, path: Path, source: str) -> tuple[list[dict], int]:
        """获取 session 文件中未处理的新消息（不推进状态）。

        Returns:
            (new_messages, total_count) — 新消息列表和文件中的总消息数。
            调用方处理成功后需调用 commit_progress(path, total_count) 来推进状态。
        """
        key = str(path)
        if source == "acp":
            all_msgs = SessionParser.parse_acp(path)
        else:
            all_msgs = SessionParser.parse_cli(path)

        processed_count = self._state["processed"].get(key, 0)
        return all_msgs[processed_count:], len(all_msgs)

    def commit_progress(self, path: Path, total_count: int) -> None:
        """推进指定 session 文件的处理进度。

        Args:
            path: session 文件路径
            total_count: get_new_messages 返回的总消息数
        """
        self._state["processed"][str(path)] = total_count
        self._save_state()

    def write_classified_memories(
        self, memories: list[dict], session_path: Path
    ) -> int:
        """Write classified memories to SQLite store (dual-write).

        Args:
            memories: list of {"category": str, "text": str} from summarizer
            session_path: source session file path

        Returns:
            Number of memories successfully written.
        """
        if self.store is None:
            return 0

        count = 0
        for mem in memories:
            category = mem.get("category", "")
            text = mem.get("text", "")
            if not category or not text:
                logger.warning("Skipping memory with missing category or text")
                continue
            try:
                self.store.add(
                    category=category,
                    text=text,
                    source_session=session_path.name,
                )
                logger.info(
                    f"SQLite write: [{category}] {text[:80]}{'...' if len(text) > 80 else ''}"
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to write memory to SQLite: {e}")
        return count

    def write_cleaned_messages(
        self, messages: list[dict], session_path: Path
    ) -> Optional[tuple[Path, int]]:
        """将清洗后的消息追加到当日摘要文件，返回 (文件路径, 起始行号)。"""
        if not messages:
            return None

        today = datetime.now().strftime("%Y-%m-%d")
        summary_file = self.memory_dir / f"{today}.md"
        is_new = not summary_file.exists()

        # 计算当前行数（用于索引行号定位）
        start_line = 1
        if summary_file.exists():
            with open(summary_file) as f:
                start_line = sum(1 for _ in f) + 1

        # 格式化消息
        ts = datetime.now().strftime("%H:%M")
        lines = [f"\n## [{ts}] {session_path.stem}\n"]
        for msg in messages:
            role_label = "用户" if msg["role"] in ("user", "human") else "AI"
            # 截断过长的单条消息
            text = msg["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"**{role_label}**: {text}\n")

        with open(summary_file, "a", encoding="utf-8") as f:
            if is_new:
                f.write(f"# {today} 对话记录\n")
                start_line = 2  # 标题占第1行
            f.writelines(lines)

        return summary_file, start_line

    def update_index(
        self,
        summary: str,
        target_file: Path,
        line_number: int,
        source: str = "",
    ) -> None:
        """更新 index.md 索引文件。新条目插入到对应日期段的最前面。"""
        today = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%H:%M")
        rel_path = target_file.name
        tag = f" [{source}]" if source else ""
        entry = f"- {ts} {summary} → {rel_path}:{line_number}{tag}\n"

        if not self.index_file.exists():
            with open(self.index_file, "w", encoding="utf-8") as f:
                f.write(f"# iFlow MemFly Index\n\n## {today}\n{entry}")
            return

        with open(self.index_file, "r", encoding="utf-8") as f:
            content = f.read()

        date_header = f"## {today}"
        if date_header in content:
            # 在日期标题后插入新条目
            content = content.replace(
                date_header + "\n",
                date_header + "\n" + entry,
            )
        else:
            # 新日期段，插入到标题后面
            header_end = content.find("\n\n")
            if header_end == -1:
                content += f"\n\n{date_header}\n{entry}"
            else:
                content = (
                    content[: header_end + 2]
                    + f"{date_header}\n{entry}\n"
                    + content[header_end + 2:]
                )

        with open(self.index_file, "w", encoding="utf-8") as f:
            f.write(content)

    def append_structured_summary(
        self, summary: str, target_file: Path
    ) -> None:
        """第2层：将结构化摘要追加到摘要文件末尾的专用区域。"""
        separator = "\n---\n\n## 结构化摘要\n\n"
        ts = datetime.now().strftime("%H:%M")

        if not target_file.exists():
            return

        content = target_file.read_text(encoding="utf-8")
        entry = f"#### [{ts}] 摘要\n{summary}\n\n"

        if "## 结构化摘要" in content:
            # 已有摘要区域，追加到末尾
            content += entry
        else:
            # 首次添加摘要区域
            content += separator + entry

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)

    def get_recent_index(self, lines: int = 50) -> str:
        """读取索引最近 N 行。"""
        if not self.index_file.exists():
            return ""
        with open(self.index_file) as f:
            all_lines = f.readlines()
        # 标题行 + 最近的条目
        if len(all_lines) <= lines + 2:
            return "".join(all_lines)
        return "".join(all_lines[:2] + all_lines[2 : 2 + lines])