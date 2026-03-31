"""Tests for MemoryStore and hotness_score."""

import math
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from iflow_memory.store.db import MemoryStore, hotness_score, VALID_CATEGORIES


@pytest.fixture
def store(tmp_path):
    """Create a fresh MemoryStore in a temp directory."""
    db = tmp_path / "test.db"
    s = MemoryStore(db)
    yield s
    s.close()


# ── hotness_score ──────────────────────────────────────────────

class TestHotnessScore:
    def test_brand_new_memory(self):
        """刚创建的记忆（0 次访问），hotness 应在 0.4~0.6 之间。"""
        now = datetime.now(timezone.utc).isoformat()
        score = hotness_score(0, now)
        assert 0.4 < score < 0.6

    def test_more_access_higher_score(self):
        """访问次数越多，同龄记忆的 hotness 越高。"""
        now = datetime.now(timezone.utc).isoformat()
        s0 = hotness_score(0, now)
        s5 = hotness_score(5, now)
        s50 = hotness_score(50, now)
        assert s0 < s5 < s50

    def test_older_memory_lower_score(self):
        """同样访问次数，越老的记忆 hotness 越低。"""
        now = datetime.now(timezone.utc)
        fresh = now.isoformat()
        week_old = (now - timedelta(days=7)).isoformat()
        month_old = (now - timedelta(days=30)).isoformat()
        s_fresh = hotness_score(3, fresh)
        s_week = hotness_score(3, week_old)
        s_month = hotness_score(3, month_old)
        assert s_fresh > s_week > s_month

    def test_14_day_halflife(self):
        """14 天后 decay 因子应约为 0.5。"""
        now = datetime.now(timezone.utc)
        two_weeks_ago = (now - timedelta(days=14)).isoformat()
        # access_count=0 时 popularity = sigmoid(0) = 0.5
        score = hotness_score(0, two_weeks_ago)
        expected = 0.5 * 0.5  # popularity(0.5) * decay(0.5)
        assert abs(score - expected) < 0.02

    def test_invalid_date_fallback(self):
        """无效日期应回退到 30 天老（不崩溃）。"""
        score = hotness_score(0, "not-a-date")
        assert 0 < score < 0.5  # 30 天老 + 0 次访问，应该很低

    def test_none_date_fallback(self):
        """None 日期应回退（不崩溃）。"""
        score = hotness_score(0, None)
        assert 0 < score < 0.5


# ── MemoryStore CRUD ───────────────────────────────────────────

class TestMemoryStoreCRUD:
    def test_add_and_search(self, store):
        """添加记忆后能搜到。"""
        store.add("entity", "Redis 服务端口是 6379")
        results = store.search("Redis")
        assert len(results) >= 1
        assert "6379" in results[0]["text"]

    def test_add_invalid_category(self, store):
        """无效分类应抛 ValueError。"""
        with pytest.raises(ValueError, match="Invalid category"):
            store.add("invalid_cat", "some text")

    def test_add_empty_text(self, store):
        """空文本应抛 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add("entity", "   ")

    def test_search_empty_query(self, store):
        """空查询应返回空列表。"""
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_increments_access_count(self, store):
        """搜索命中应自动增加 access_count。"""
        store.add("entity", "unique test keyword xyzzy")
        # 第一次搜索：返回的快照是 access_count=0（UPDATE 在 SELECT 之后）
        results = store.search("xyzzy")
        assert len(results) == 1
        assert results[0]["access_count"] == 0
        # 但 DB 里已经 +1 了，再搜一次验证
        results2 = store.search("xyzzy")
        assert results2[0]["access_count"] == 1

    def test_get_by_category(self, store):
        """按分类查询。"""
        store.add("identity", "用户名是 alice")
        store.add("preference", "不要 emoji")
        store.add("identity", "AI 昵称是 assistant")

        ids = store.get_by_category("identity")
        assert len(ids) == 2
        assert all(r["category"] == "identity" for r in ids)

    def test_archive_cold(self, store):
        """archive_cold 应归档低热度记忆。"""
        # 插入一条 8 天前的记忆（0 次访问）
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        store._conn.execute(
            "INSERT INTO memories (category, text, created_at, updated_at, access_count) VALUES (?, ?, ?, ?, ?)",
            ("event", "old event nobody cares about", old_time, old_time, 0),
        )
        store._conn.commit()
        # 手动同步 FTS
        row = store._conn.execute("SELECT id, text, category FROM memories ORDER BY id DESC LIMIT 1").fetchone()
        store._conn.execute(
            "INSERT INTO memories_fts(rowid, text, category) VALUES (?, ?, ?)",
            (row["id"], row["text"], row["category"]),
        )
        store._conn.commit()

        count = store.archive_cold(threshold=0.5, min_age_days=7)
        assert count >= 1

        # 归档后搜不到
        results = store.get_by_category("event", include_archived=False)
        assert all("old event" not in r["text"] for r in results)

    def test_archive_preserves_identity(self, store):
        """archive_cold 不应归档 identity 和 preference。"""
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        store._conn.execute(
            "INSERT INTO memories (category, text, created_at, updated_at, access_count) VALUES (?, ?, ?, ?, ?)",
            ("identity", "important identity fact", old_time, old_time, 0),
        )
        store._conn.commit()

        count = store.archive_cold(threshold=0.99, min_age_days=1)
        assert count == 0  # identity 不应被归档

    def test_stats(self, store):
        """stats 应返回正确的统计（考虑种子记忆基线）。"""
        baseline = store.stats()
        base_total = baseline["total"]
        base_entity = baseline["by_category"].get("entity", 0)
        base_insight = baseline["by_category"].get("insight", 0)

        store.add("entity", "test entity one")
        store.add("entity", "test entity two")
        store.add("insight", "test insight")

        s = store.stats()
        assert s["total"] == base_total + 3
        assert s["archived"] == 0
        assert s["by_category"]["entity"] == base_entity + 2
        assert s["by_category"]["insight"] == base_insight + 1

    def test_checkpoint(self, store):
        """checkpoint 不应崩溃。"""
        store.add("entity", "some data for checkpoint test")
        store.checkpoint()  # 不抛异常就算通过

    def test_context_manager(self, tmp_path):
        """with 语句应正常工作。"""
        db = tmp_path / "ctx.db"
        with MemoryStore(db) as s:
            s.add("entity", "context manager test")
            results = s.search("context manager")
            assert len(results) >= 1
