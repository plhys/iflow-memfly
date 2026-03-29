"""Tests for Indexer: session parsing, state management, classified memory writing."""

import json
import tempfile
from pathlib import Path

import pytest

from iflow_memory.core.indexer import Indexer, SessionParser
from iflow_memory.store.db import MemoryStore


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temp memory directory."""
    return tmp_path / "memory"


@pytest.fixture
def store(tmp_path):
    """Create a fresh MemoryStore."""
    db = tmp_path / "test.db"
    s = MemoryStore(db)
    yield s
    s.close()


@pytest.fixture
def indexer(memory_dir, store):
    """Create an Indexer with a store."""
    return Indexer(memory_dir, store=store)


# ── SessionParser ──────────────────────────────────────────────

class TestSessionParserACP:
    def test_parse_basic(self, tmp_path):
        """基本 ACP session 解析。"""
        session = {
            "chatHistory": [
                {"role": "user", "parts": [{"text": "你好"}]},
                {"role": "model", "parts": [{"text": "你好主任"}]},
            ]
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session))

        msgs = SessionParser.parse_acp(path)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "text": "你好"}
        assert msgs[1] == {"role": "model", "text": "你好主任"}

    def test_skip_system_messages(self, tmp_path):
        """system 消息应被跳过。"""
        session = {
            "chatHistory": [
                {"role": "system", "parts": [{"text": "system prompt"}]},
                {"role": "user", "parts": [{"text": "hello"}]},
            ]
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session))

        msgs = SessionParser.parse_acp(path)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_skip_tool_calls(self, tmp_path):
        """tool call parts 应被过滤。"""
        session = {
            "chatHistory": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "read_file", "args": {}}},
                        {"text": "文件内容如下"},
                    ],
                },
            ]
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session))

        msgs = SessionParser.parse_acp(path)
        assert len(msgs) == 1
        assert "文件内容" in msgs[0]["text"]
        assert "functionCall" not in msgs[0]["text"]

    def test_strip_system_reminder(self, tmp_path):
        """<system-reminder> 标签应被清洗。"""
        session = {
            "chatHistory": [
                {
                    "role": "user",
                    "parts": [{"text": "<system-reminder>ignore this</system-reminder>真正的问题"}],
                },
            ]
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(session))

        msgs = SessionParser.parse_acp(path)
        assert len(msgs) == 1
        assert "system-reminder" not in msgs[0]["text"]
        assert "真正的问题" in msgs[0]["text"]

    def test_invalid_json(self, tmp_path):
        """损坏的 JSON 应返回空列表。"""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        assert SessionParser.parse_acp(path) == []

    def test_missing_file(self, tmp_path):
        """不存在的文件应返回空列表。"""
        path = tmp_path / "nonexistent.json"
        assert SessionParser.parse_acp(path) == []


class TestSessionParserCLI:
    def test_parse_basic(self, tmp_path):
        """基本 CLI session JSONL 解析。"""
        path = tmp_path / "session-test.jsonl"
        lines = [
            json.dumps({"type": "human", "message": {"content": "你好"}}),
            json.dumps({"type": "assistant", "message": {"content": "你好主任"}}),
        ]
        path.write_text("\n".join(lines))

        msgs = SessionParser.parse_cli(path)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "text": "你好"}
        assert msgs[1] == {"role": "model", "text": "你好主任"}

    def test_skip_non_message_types(self, tmp_path):
        """非 human/assistant 类型应跳过。"""
        path = tmp_path / "session-test.jsonl"
        lines = [
            json.dumps({"type": "summary", "message": {"content": "skip me"}}),
            json.dumps({"type": "human", "message": {"content": "keep me"}}),
        ]
        path.write_text("\n".join(lines))

        msgs = SessionParser.parse_cli(path)
        assert len(msgs) == 1
        assert msgs[0]["text"] == "keep me"

    def test_content_as_list(self, tmp_path):
        """content 为 list of blocks 时应提取 text 类型。"""
        path = tmp_path / "session-test.jsonl"
        content = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "read_file"},
            {"type": "text", "text": " world"},
        ]
        line = json.dumps({"type": "human", "message": {"content": content}})
        path.write_text(line)

        msgs = SessionParser.parse_cli(path)
        assert len(msgs) == 1
        assert "hello" in msgs[0]["text"]
        assert "world" in msgs[0]["text"]

    def test_empty_lines_skipped(self, tmp_path):
        """空行应被跳过。"""
        path = tmp_path / "session-test.jsonl"
        path.write_text('\n\n{"type": "human", "message": {"content": "test"}}\n\n')

        msgs = SessionParser.parse_cli(path)
        assert len(msgs) == 1


# ── Indexer state management ───────────────────────────────────

class TestIndexerState:
    def test_get_new_messages_returns_only_new(self, indexer, tmp_path):
        """get_new_messages 应只返回未处理的消息。"""
        session = {
            "chatHistory": [
                {"role": "user", "parts": [{"text": "first"}]},
                {"role": "model", "parts": [{"text": "response 1"}]},
            ]
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(session))

        msgs, total = indexer.get_new_messages(path, "acp")
        assert len(msgs) == 2
        assert total == 2

        # commit 后再取，应该没有新消息
        indexer.commit_progress(path, total)
        msgs2, total2 = indexer.get_new_messages(path, "acp")
        assert len(msgs2) == 0
        assert total2 == 2

    def test_incremental_processing(self, indexer, tmp_path):
        """追加消息后应只返回增量。"""
        path = tmp_path / "s1.json"

        # 第一轮：2 条消息
        session = {
            "chatHistory": [
                {"role": "user", "parts": [{"text": "first"}]},
                {"role": "model", "parts": [{"text": "response 1"}]},
            ]
        }
        path.write_text(json.dumps(session))
        msgs, total = indexer.get_new_messages(path, "acp")
        assert len(msgs) == 2
        indexer.commit_progress(path, total)

        # 第二轮：追加 2 条
        session["chatHistory"].extend([
            {"role": "user", "parts": [{"text": "second"}]},
            {"role": "model", "parts": [{"text": "response 2"}]},
        ])
        path.write_text(json.dumps(session))
        msgs2, total2 = indexer.get_new_messages(path, "acp")
        assert len(msgs2) == 2
        assert total2 == 4
        assert msgs2[0]["text"] == "second"

    def test_no_commit_means_retry(self, indexer, tmp_path):
        """不 commit 的话，下次应重新返回同一批消息。"""
        session = {
            "chatHistory": [
                {"role": "user", "parts": [{"text": "important"}]},
            ]
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(session))

        msgs1, _ = indexer.get_new_messages(path, "acp")
        assert len(msgs1) == 1
        # 不 commit

        msgs2, _ = indexer.get_new_messages(path, "acp")
        assert len(msgs2) == 1  # 同样的消息再次返回
        assert msgs2[0]["text"] == "important"

    def test_state_persists_across_instances(self, memory_dir, store, tmp_path):
        """状态应持久化，新 Indexer 实例能读到。"""
        session = {
            "chatHistory": [
                {"role": "user", "parts": [{"text": "test"}]},
            ]
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(session))

        idx1 = Indexer(memory_dir, store=store)
        msgs, total = idx1.get_new_messages(path, "acp")
        assert len(msgs) == 1
        idx1.commit_progress(path, total)

        # 新实例
        idx2 = Indexer(memory_dir, store=store)
        msgs2, _ = idx2.get_new_messages(path, "acp")
        assert len(msgs2) == 0  # 已处理


# ── Indexer write operations ───────────────────────────────────

class TestIndexerWrite:
    def test_write_classified_memories(self, indexer, tmp_path):
        """write_classified_memories 应写入 SQLite。"""
        memories = [
            {"category": "entity", "text": "Redis 端口是 6379"},
            {"category": "insight", "text": "服务器重启后需要恢复符号链接"},
        ]
        session_path = tmp_path / "session.json"
        count = indexer.write_classified_memories(memories, session_path)
        assert count == 2

    def test_write_classified_skips_invalid(self, indexer, tmp_path):
        """无效记忆应被跳过。"""
        memories = [
            {"category": "", "text": "no category"},
            {"category": "entity", "text": ""},
            {"category": "entity", "text": "valid one"},
        ]
        session_path = tmp_path / "session.json"
        count = indexer.write_classified_memories(memories, session_path)
        assert count == 1

    def test_write_cleaned_messages(self, indexer, tmp_path):
        """write_cleaned_messages 应创建当日文件。"""
        messages = [
            {"role": "user", "text": "你好"},
            {"role": "model", "text": "你好主任"},
        ]
        session_path = tmp_path / "session.json"
        result = indexer.write_cleaned_messages(messages, session_path)
        assert result is not None
        target_file, start_line = result
        assert target_file.exists()
        content = target_file.read_text()
        assert "你好" in content
        assert "你好主任" in content

    def test_update_index(self, indexer):
        """update_index 应创建/更新 index.md。"""
        target = indexer.memory_dir / "2026-03-27.md"
        target.touch()
        indexer.update_index("测试索引条目", target, 1)
        assert indexer.index_file.exists()
        content = indexer.index_file.read_text()
        assert "测试索引条目" in content
