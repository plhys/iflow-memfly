"""Tests for Summarizer JSON parsing."""

import pytest

from iflow_memory.core.summarizer import _parse_json_response


class TestParseJsonResponse:
    def test_clean_json(self):
        """干净的 JSON 应直接解析。"""
        text = '{"memories": [{"category": "entity", "text": "test"}]}'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result["memories"]) == 1
        assert result["memories"][0]["category"] == "entity"

    def test_json_in_code_block(self):
        """```json 代码块包裹的 JSON。"""
        text = '```json\n{"memories": [{"category": "identity", "text": "用户名 alice"}]}\n```'
        result = _parse_json_response(text)
        assert result is not None
        assert result["memories"][0]["text"] == "用户名 alice"

    def test_json_with_trailing_comma(self):
        """尾部逗号应被清理。"""
        text = '{"memories": [{"category": "entity", "text": "test"},]}'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result["memories"]) == 1

    def test_json_with_preamble(self):
        """JSON 前面有废话应能提取。"""
        text = '好的，以下是提取的记忆：\n{"memories": [{"category": "event", "text": "完成了部署"}]}'
        result = _parse_json_response(text)
        assert result is not None
        assert result["memories"][0]["category"] == "event"

    def test_truncated_json(self):
        """截断的 JSON 应尽力修复。"""
        text = '{"memories": [{"category": "entity", "text": "first"}, {"category": "insight", "text": "second"'
        result = _parse_json_response(text)
        # 至少能提取到第一条完整的
        assert result is not None
        memories = result.get("memories", [])
        assert len(memories) >= 1

    def test_empty_memories(self):
        """空记忆列表。"""
        text = '{"memories": []}'
        result = _parse_json_response(text)
        assert result is not None
        assert result["memories"] == []

    def test_empty_input(self):
        """空输入应返回 None。"""
        assert _parse_json_response("") is None
        assert _parse_json_response("   ") is None
        assert _parse_json_response(None) is None

    def test_no_json_at_all(self):
        """完全没有 JSON 应返回 None。"""
        assert _parse_json_response("这段话里没有任何 JSON") is None

    def test_regex_fallback(self):
        """严重损坏的 JSON 应走正则兜底。"""
        text = '{"memories": [{"category": "entity", "text": "Redis port 6379"}, BROKEN STUFF HERE'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result["memories"]) >= 1
        assert "Redis" in result["memories"][0]["text"]

    def test_multiple_memories(self):
        """多条记忆正常解析。"""
        text = '''{"memories": [
            {"category": "identity", "text": "用户名是 alice"},
            {"category": "preference", "text": "不要 emoji"},
            {"category": "entity", "text": "Redis 端口 6379"}
        ]}'''
        result = _parse_json_response(text)
        assert result is not None
        assert len(result["memories"]) == 3

    def test_code_block_without_json_tag(self):
        """没有 json 标签的代码块。"""
        text = '```\n{"memories": [{"category": "insight", "text": "记住这个"}]}\n```'
        result = _parse_json_response(text)
        assert result is not None
        assert result["memories"][0]["category"] == "insight"
