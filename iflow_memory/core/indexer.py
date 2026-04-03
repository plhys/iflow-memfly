"""Indexer — cleans raw session data and generates memory index."""

import hashlib
import json
import logging
import os
import re
import time
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

    SHADOW_RETAIN_SECONDS = 6 * 3600  # 影子记录保留 6 小时

    def __init__(self, memory_dir: Path, store=None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.memory_dir / "index.md"
        self.store = store  # Optional MemoryStore instance for SQLite dual-write
        # 增量状态：记录每个 session 文件已处理到第几条消息
        self._state_file = self.memory_dir / ".indexer-state.json"
        self._state: dict = self._load_state()
        # 影子记录目录
        self._shadow_dir = self.memory_dir / ".shadow"
        self._shadow_dir.mkdir(exist_ok=True)

    def _load_state(self) -> dict:
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"processed": {}, "shadow_committed": {}}

    def _save_state(self) -> None:
        # Atomic write: temp file + os.replace() to prevent corruption on crash
        tmp_path = str(self._state_file) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._state, f, ensure_ascii=False)
        os.replace(tmp_path, str(self._state_file))

    def get_new_messages(self, path: Path, source: str) -> tuple[list[dict], int]:
        """获取 session 文件中未处理的新消息（不推进状态）。

        内置影子记录机制：
        - 每次读到新消息，同步写入影子文件（滚动保留 6 小时）
        - 检测到文件被重写（total < processed）时，从影子文件恢复丢失消息

        Returns:
            (new_messages, total_count) — 新消息列表和文件中的总消息数。
            调用方处理成功后需调用 commit_progress(path, total_count) 来推进状态。
        """
        key = str(path)
        if source == "acp":
            all_msgs = SessionParser.parse_acp(path)
        else:
            all_msgs = SessionParser.parse_cli(path)

        total_count = len(all_msgs)
        processed_count = self._state["processed"].get(key, 0)

        if total_count >= processed_count:
            # --- 正常路径：文件只增不减 ---
            new_msgs = all_msgs[processed_count:]
            if new_msgs:
                self._shadow_append(key, new_msgs)
                self._shadow_cleanup(key)
                # 记录已处理消息的哈希，供恢复时去重
                committed = self._state.setdefault("shadow_committed", {}).setdefault(key, [])
                for m in new_msgs:
                    committed.append(self._msg_hash(m))
                # 只保留最近 500 个哈希，防止无限膨胀
                if len(committed) > 500:
                    self._state["shadow_committed"][key] = committed[-500:]
            return new_msgs, total_count
        else:
            # --- 异常路径：文件被重写，消息数变少（压缩/爆掉） ---
            logger.warning(
                f"Session file shrunk: {path.name} "
                f"(was {processed_count}, now {total_count}). "
                f"Attempting shadow recovery."
            )
            recovered = self._shadow_recover(key, all_msgs)
            if recovered:
                logger.info(
                    f"Shadow recovery: found {len(recovered)} lost messages "
                    f"for {path.name}"
                )
            # 重置计数到当前文件长度，后续正常追踪
            return recovered, total_count

    def commit_progress(self, path: Path, total_count: int) -> None:
        """推进指定 session 文件的处理进度。

        Args:
            path: session 文件路径
            total_count: get_new_messages 返回的总消息数
        """
        self._state["processed"][str(path)] = total_count
        self._save_state()

    # ---- 影子记录（Shadow Record）内部方法 ----

    def _shadow_path(self, key: str) -> Path:
        """根据 session 文件路径生成对应的影子文件路径。"""
        name_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        return self._shadow_dir / f"{name_hash}.jsonl"

    @staticmethod
    def _msg_hash(msg: dict) -> str:
        """计算消息的内容哈希（role + text 前 200 字符）。"""
        raw = f"{msg.get('role', '')}:{msg.get('text', '')[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _shadow_append(self, key: str, messages: list[dict]) -> None:
        """将新消息追加写入影子文件，每条带时间戳和哈希。"""
        shadow_file = self._shadow_path(key)
        now = time.time()
        try:
            with open(shadow_file, "a", encoding="utf-8") as f:
                for msg in messages:
                    record = {
                        "ts": now,
                        "hash": self._msg_hash(msg),
                        "role": msg["role"],
                        "text": msg["text"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Shadow append failed for {key}: {e}")

    def _shadow_cleanup(self, key: str) -> None:
        """清理影子文件中超过保留时间的旧记录。"""
        shadow_file = self._shadow_path(key)
        if not shadow_file.exists():
            return
        cutoff = time.time() - self.SHADOW_RETAIN_SECONDS
        try:
            kept = []
            with open(shadow_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("ts", 0) >= cutoff:
                        kept.append(line)
            with open(shadow_file, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
        except OSError as e:
            logger.warning(f"Shadow cleanup failed for {key}: {e}")

    def _shadow_recover(self, key: str, current_msgs: list[dict]) -> list[dict]:
        """从影子文件中恢复当前 session 文件中已丢失的消息。

        对比影子记录和当前文件内容的哈希，排除已处理过的消息，
        只返回真正丢失且未被处理的部分。
        """
        shadow_file = self._shadow_path(key)
        if not shadow_file.exists():
            logger.warning(f"No shadow file for recovery: {key}")
            return []

        # 当前文件中所有消息的哈希集合
        current_hashes = {self._msg_hash(m) for m in current_msgs}
        # 之前已成功处理过的消息哈希集合
        committed_hashes = set(
            self._state.get("shadow_committed", {}).get(key, [])
        )
        skip_hashes = current_hashes | committed_hashes

        recovered = []
        seen = set()  # 影子文件内部去重
        try:
            with open(shadow_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = rec.get("hash")
                    if h not in skip_hashes and h not in seen:
                        seen.add(h)
                        recovered.append({
                            "role": rec["role"],
                            "text": rec["text"],
                        })
        except OSError as e:
            logger.warning(f"Shadow recovery read failed for {key}: {e}")
            return []

        return recovered

    def write_classified_memories(
        self, memories: list[dict], session_path: Path,
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
    ) -> int:
        """Write classified memories to SQLite store (dual-write).

        Args:
            memories: list of {"category": str, "text": str} from summarizer
            session_path: source session file path
            source_file: timeline 文件路径（如 2026-04-03.md），用于来源追踪
            source_line: 在 timeline 文件中的起始行号

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
                    source_file=source_file,
                    source_line=source_line,
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
            # 截断过长的单条消息（L3 兜底层，保留更多原始内容）
            text = msg["text"]
            if len(text) > 2000:
                text = text[:2000] + "..."
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