"""MemoryStore — SQLite + FTS5 storage layer for classified memories.

Provides persistent storage with full-text search for the iFlow MemFly system.
Schema uses WAL mode for concurrent reads and FTS5 triggers for index consistency.
"""

import logging
import math
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iflow-memory")

VALID_CATEGORIES = frozenset({"identity", "preference", "entity", "event", "insight", "correction"})

# Fuzzy dedup threshold — Jaccard similarity above this means duplicate
_DEDUP_SIMILARITY_THRESHOLD = 0.80


def _normalize_text(text: str) -> str:
    """标准化文本用于去重比较：去空格差异、统一标点、转小写。"""
    t = text.strip().lower()
    # 统一空白字符
    t = re.sub(r"\s+", " ", t)
    # 去掉中文与 ASCII 之间的空格（如 "AI 响应" vs "AI响应"）
    t = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[A-Za-z0-9])", "", t)
    t = re.sub(r"(?<=[A-Za-z0-9]) (?=[\u4e00-\u9fff])", "", t)
    # 统一中英文标点
    t = t.replace("：", ":").replace("，", ",").replace("。", ".").replace("；", ";")
    t = t.replace("（", "(").replace("）", ")").replace("'", "'").replace("'", "'")
    t = t.replace(""", '"').replace(""", '"')
    return t


def _jaccard_similarity(a: str, b: str) -> float:
    """计算两个文本的字符 bigram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    bigrams_a = {a[i:i+2] for i in range(len(a) - 1)}
    bigrams_b = {b[i:i+2] for i in range(len(b) - 1)}
    if not bigrams_a or not bigrams_b:
        return 0.0
    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    return intersection / union if union else 0.0


def _cjk_ratio(text: str) -> float:
    """计算文本中 CJK 字符的比例。"""
    if not text:
        return 0.0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk / len(text)


# Current schema version — bump this when adding migrations
SCHEMA_VERSION = 4


def _serialize_f32(vec: list[float]) -> bytes:
    """将 float 列表序列化为 little-endian float32 bytes（sqlite-vec 格式）。"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes) -> list[float]:
    """从 bytes 反序列化为 float 列表。"""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def hotness_score(access_count: int, updated_at: str) -> float:
    """Calculate hotness score for memory ranking.

    Formula: sigmoid(log1p(access_count)) * exp(-0.693 * age_days / 14)

    The sigmoid of log1p(access_count) gives a 0–1 popularity factor that
    grows quickly for the first few accesses then saturates.
    The exponential decay halves the score every 14 days since last update.
    """
    # Popularity factor
    popularity = 1.0 / (1.0 + math.exp(-math.log1p(access_count)))

    # Age decay
    try:
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0
    except (ValueError, TypeError):
        age_days = 30.0  # fallback: treat as old

    decay = math.exp(-0.693 * age_days / 14.0)

    return popularity * decay


class MemoryStore:
    """SQLite + FTS5 storage for classified memories."""

    def __init__(self, db_path: str | Path, embed_dim: int = 0):
        """Open or create the database and run any pending migrations.

        Args:
            db_path: SQLite database file path.
            embed_dim: embedding 向量维度。0 表示不启用向量搜索。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self._vec_enabled = False

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA wal_autocheckpoint=100")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # 加载 sqlite-vec 扩展（如果可用）
        if embed_dim > 0:
            self._vec_enabled = self._load_sqlite_vec()

        self._migrate()

        # 确保 vec 表存在（schema 可能已是 v3 但当时 vec 未启用）
        if self._vec_enabled and self.embed_dim > 0:
            self._ensure_vec_table()

        logger.info(f"MemoryStore opened: {self.db_path} (vec={'on' if self._vec_enabled else 'off'})")

    def _load_sqlite_vec(self) -> bool:
        """尝试加载 sqlite-vec 扩展。"""
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except ImportError:
            logger.warning("[深度回忆] sqlite-vec 未安装，向量搜索不可用")
            return False
        except Exception as e:
            logger.warning(f"[深度回忆] sqlite-vec 加载失败: {e}")
            return False

    def _ensure_vec_table(self) -> None:
        """确保 memories_vec 虚拟表存在（schema 可能已是 v3 但当时 vec 未启用）。"""
        try:
            self._conn.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                    USING vec0(embedding float[{self.embed_dim}])"""
            )
            self._conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"[深度回忆] 创建 vec 表失败: {e}")
            self._vec_enabled = False

    # ------------------------------------------------------------------
    # Schema & migrations
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Run schema creation and incremental migrations based on user_version."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]

        if version < 1:
            self._migrate_v1()
        if version < 2:
            self._migrate_v2()
        if version < 3:
            self._migrate_v3()
        if version < 4:
            self._migrate_v4()

        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _migrate_v1(self) -> None:
        """Initial schema: memories table + FTS5 virtual table + sync triggers."""
        logger.info("Running migration v1: creating memories + FTS5 tables")
        cur = self._conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                category        TEXT    NOT NULL,
                text            TEXT    NOT NULL,
                source_session  TEXT    DEFAULT '',
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                access_count    INTEGER DEFAULT 0,
                archived        INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_updated
                ON memories(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_archived
                ON memories(archived);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text,
                category,
                content=memories,
                content_rowid=id,
                tokenize='trigram'
            );

            -- Triggers to keep FTS5 in sync with the memories table

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text, category)
                VALUES (new.id, new.text, new.category);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, category)
                VALUES ('delete', old.id, old.text, old.category);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, category)
                VALUES ('delete', old.id, old.text, old.category);
                INSERT INTO memories_fts(rowid, text, category)
                VALUES (new.id, new.text, new.category);
            END;
        """)

        self._conn.commit()

    def _migrate_v2(self) -> None:
        """Migration v2: atmosphere_snapshots table for conversation context."""
        logger.info("Running migration v2: creating atmosphere_snapshots table")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS atmosphere_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                snapshot         TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_atmosphere_date
                ON atmosphere_snapshots(date DESC);
            CREATE INDEX IF NOT EXISTS idx_atmosphere_session
                ON atmosphere_snapshots(session_id);
        """)
        self._conn.commit()

    def _migrate_v3(self) -> None:
        """Migration v3: embedding column + sqlite-vec virtual table for vector search."""
        logger.info("Running migration v3: adding embedding support")

        # 添加 embedding 列（BLOB，可为 NULL）
        try:
            self._conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # 列已存在

        # 创建 sqlite-vec 虚拟表（仅当 vec 扩展可用且维度已知时）
        if self._vec_enabled and self.embed_dim > 0:
            try:
                self._conn.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                        USING vec0(embedding float[{self.embed_dim}])"""
                )
                logger.info(f"Created vec0 table: memories_vec (dim={self.embed_dim})")
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to create vec0 table: {e}")
                self._vec_enabled = False

        self._conn.commit()

    def _migrate_v4(self) -> None:
        """Migration v4: state_snapshots table for structured state checkpoints."""
        logger.info("Running migration v4: creating state_snapshots table")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS state_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                goal            TEXT    NOT NULL DEFAULT '',
                progress        TEXT    NOT NULL DEFAULT '',
                decisions       TEXT    NOT NULL DEFAULT '',
                next_steps      TEXT    NOT NULL DEFAULT '',
                critical_context TEXT   NOT NULL DEFAULT '',
                created_at      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_state_date
                ON state_snapshots(date DESC);
            CREATE INDEX IF NOT EXISTS idx_state_session
                ON state_snapshots(session_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # State snapshot operations
    # ------------------------------------------------------------------

    def add_state_snapshot(
        self, session_id: str, date: str,
        goal: str, progress: str, decisions: str,
        next_steps: str, critical_context: str,
    ) -> int:
        """存储一条状态快照。同一 session 会覆盖更新。"""
        now = datetime.now(timezone.utc).isoformat()
        existing = self._conn.execute(
            "SELECT id FROM state_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing:
            self._conn.execute(
                """UPDATE state_snapshots
                   SET goal=?, progress=?, decisions=?, next_steps=?,
                       critical_context=?, created_at=?
                   WHERE id=?""",
                (goal, progress, decisions, next_steps, critical_context, now, existing[0]),
            )
            self._conn.commit()
            return existing[0]

        cur = self._conn.execute(
            """INSERT INTO state_snapshots
               (session_id, date, goal, progress, decisions, next_steps, critical_context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, date, goal, progress, decisions, next_steps, critical_context, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_latest_state_snapshot(self, limit: int = 1) -> list[dict]:
        """获取最近的状态快照。"""
        rows = self._conn.execute(
            """SELECT id, session_id, date, goal, progress, decisions,
                      next_steps, critical_context, created_at
               FROM state_snapshots
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Atmosphere snapshot operations
    # ------------------------------------------------------------------

    def add_atmosphere(self, session_id: str, date: str, snapshot: str) -> int:
        """存储一条对话氛围快照。同一 session 会覆盖更新。"""
        now = datetime.now(timezone.utc).isoformat()
        existing = self._conn.execute(
            "SELECT id FROM atmosphere_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE atmosphere_snapshots SET snapshot = ?, created_at = ? WHERE id = ?",
                (snapshot, now, existing[0]),
            )
            self._conn.commit()
            return existing[0]

        cur = self._conn.execute(
            """INSERT INTO atmosphere_snapshots (session_id, date, snapshot, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, date, snapshot, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_latest_atmosphere(self, limit: int = 1) -> list[dict]:
        """获取最近的氛围快照。"""
        rows = self._conn.execute(
            """SELECT id, session_id, date, snapshot, created_at
               FROM atmosphere_snapshots
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_atmosphere_by_date(self, date: str) -> list[dict]:
        """获取指定日期的所有氛围快照。"""
        rows = self._conn.execute(
            """SELECT id, session_id, date, snapshot, created_at
               FROM atmosphere_snapshots
               WHERE date = ?
               ORDER BY created_at DESC""",
            (date,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, category: str, text: str, source_session: str = "",
            embedding: Optional[list[float]] = None) -> int:
        """Insert a new memory, or return existing id if duplicate.

        Raises ValueError if category is invalid or text is empty.
        Returns the row id (new or existing).
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty")

        # Dedup: check for existing active memory with same text + category
        existing = self._conn.execute(
            "SELECT id FROM memories WHERE category = ? AND text = ? AND archived = 0",
            (category, text),
        ).fetchone()
        if existing:
            self.mark_accessed([existing[0]])
            logger.debug(f"Dedup: memory #{existing[0]} already exists, skipped")
            return existing[0]

        # Fuzzy dedup: normalized text match + Jaccard similarity
        norm_new = _normalize_text(text)
        candidates = self._conn.execute(
            "SELECT id, text FROM memories WHERE category = ? AND archived = 0",
            (category,),
        ).fetchall()
        for row in candidates:
            norm_old = _normalize_text(row["text"])
            if norm_old == norm_new:
                self.mark_accessed([row["id"]])
                logger.debug(f"Dedup (normalized): memory #{row['id']} matches, skipped")
                return row["id"]
            if _jaccard_similarity(norm_old, norm_new) >= _DEDUP_SIMILARITY_THRESHOLD:
                self.mark_accessed([row["id"]])
                logger.info(
                    f"Dedup (fuzzy): memory #{row['id']} similar, skipped. "
                    f"Existing: {row['text'][:50]}... | New: {text[:50]}..."
                )
                return row["id"]

        now = datetime.now(timezone.utc).isoformat()
        embed_blob = _serialize_f32(embedding) if embedding else None
        cur = self._conn.execute(
            """INSERT INTO memories (category, text, source_session, created_at, updated_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (category, text, source_session, now, now, embed_blob),
        )
        self._conn.commit()
        row_id = cur.lastrowid

        # 同步写入向量索引
        if embedding and self._vec_enabled:
            try:
                self._conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize_f32(embedding)),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning(f"Failed to insert vec for memory #{row_id}: {e}")

        logger.debug(f"Added memory #{row_id} [{category}]: {text[:60]}")
        return row_id

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search across memories with LIKE fallback.

        First tries FTS5 MATCH (trigram). If no results, falls back to
        LIKE query which handles short keywords better.
        Optionally filter by category. Returns list of dicts with
        id, category, text, created_at, access_count, rank.
        Increments access_count for all returned results.
        """
        query = query.strip()
        if not query:
            return []

        # Phase 1: FTS5 MATCH (escape special chars by quoting)
        fts_query = '"' + query.replace('"', '""') + '"'
        sql = """
            SELECT m.id, m.category, m.text, m.created_at,
                   m.access_count, rank
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memories_fts MATCH ?
              AND m.archived = 0
        """
        params: list = [fts_query]
        if category:
            sql += "  AND m.category = ?\n"
            params.append(category)
        sql += "ORDER BY rank\nLIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []

        # Phase 2: LIKE fallback when FTS5 returns nothing
        if not rows:
            sql = """
                SELECT id, category, text, created_at,
                       access_count, 0 as rank
                FROM memories
                WHERE archived = 0
                  AND text LIKE ?
            """
            params = [f"%{query}%"]
            if category:
                sql += "  AND category = ?\n"
                params.append(category)
            sql += "ORDER BY updated_at DESC\nLIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()

        results = [dict(r) for r in rows]

        if results:
            self.mark_accessed([r["id"] for r in results])

        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """深度回忆 — FTS5 + Vector RRF 融合搜索。

        当 query_embedding 为 None 或向量搜索不可用时，退化为纯 FTS5 搜索。
        RRF 公式: score = sum(w_i / (k + rank_i))，k=60。
        CJK 自适应权重: 中文 → vector 0.8 / fts 0.2; 英文 → 0.5 / 0.5。
        """
        query = query.strip()
        if not query:
            return []

        # 判断是否走混合搜索
        use_vec = (
            query_embedding is not None
            and self._vec_enabled
            and self.embed_dim > 0
        )

        if not use_vec:
            return self.search(query, category=category, limit=limit)

        # CJK 自适应权重
        cjk_ratio = _cjk_ratio(query)
        if cjk_ratio > 0.3:
            w_vec, w_fts = 0.8, 0.2
        else:
            w_vec, w_fts = 0.5, 0.5

        k = 60  # RRF 常数

        # === FTS5 路 ===
        fts_results = self.search(query, category=category, limit=limit * 2)
        fts_ranks: dict[int, int] = {}
        for rank, r in enumerate(fts_results, 1):
            fts_ranks[r["id"]] = rank

        # === Vector 路 ===
        vec_ranks: dict[int, int] = {}
        vec_limit = limit * 3  # 多取一些候选
        try:
            vec_sql = """
                SELECT rowid, distance
                FROM memories_vec
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            vec_rows = self._conn.execute(
                vec_sql, (_serialize_f32(query_embedding), vec_limit)
            ).fetchall()

            # 如果有 category 过滤，需要关联 memories 表
            if category:
                active = []
                for vr in vec_rows:
                    mem = self._conn.execute(
                        "SELECT id FROM memories WHERE id = ? AND category = ? AND archived = 0",
                        (vr[0], category),
                    ).fetchone()
                    if mem:
                        active.append(vr)
                vec_rows = active
            else:
                # 过滤已归档的
                active = []
                for vr in vec_rows:
                    mem = self._conn.execute(
                        "SELECT id FROM memories WHERE id = ? AND archived = 0",
                        (vr[0],),
                    ).fetchone()
                    if mem:
                        active.append(vr)
                vec_rows = active

            for rank, vr in enumerate(vec_rows, 1):
                vec_ranks[vr[0]] = rank

        except Exception as e:
            logger.warning(f"[深度回忆] 向量搜索失败，降级为 FTS5: {e}")
            return fts_results[:limit]

        # === RRF 融合 ===
        all_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
        scored: list[tuple[int, float]] = []
        for mem_id in all_ids:
            score = 0.0
            if mem_id in fts_ranks:
                score += w_fts / (k + fts_ranks[mem_id])
            if mem_id in vec_ranks:
                score += w_vec / (k + vec_ranks[mem_id])
            scored.append((mem_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scored[:limit]]

        if not top_ids:
            return []

        # 取完整记忆信息
        placeholders = ",".join("?" for _ in top_ids)
        rows = self._conn.execute(
            f"""SELECT id, category, text, created_at, access_count
                FROM memories
                WHERE id IN ({placeholders}) AND archived = 0""",
            top_ids,
        ).fetchall()

        # 按 RRF 分数排序
        id_to_score = dict(scored[:limit])
        results = [dict(r) for r in rows]
        results.sort(key=lambda r: id_to_score.get(r["id"], 0), reverse=True)

        # 补充 rank 字段
        for r in results:
            r["rank"] = -id_to_score.get(r["id"], 0)  # 负数以兼容 FTS5 的 rank 语义

        if results:
            self.mark_accessed([r["id"] for r in results])

        return results

    def update_embedding(self, memory_id: int, embedding: list[float]) -> bool:
        """更新指定记忆的 embedding（用于 backfill）。"""
        embed_blob = _serialize_f32(embedding)
        try:
            self._conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (embed_blob, memory_id),
            )
            if self._vec_enabled:
                # vec0 不支持 UPDATE，先删后插
                self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
                self._conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (memory_id, embed_blob),
                )
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update embedding for memory #{memory_id}: {e}")
            return False

    def get_memories_without_embedding(self, limit: int = 100) -> list[dict]:
        """获取没有 embedding 的活跃记忆（用于 backfill）。"""
        rows = self._conn.execute(
            """SELECT id, category, text
               FROM memories
               WHERE archived = 0 AND embedding IS NULL
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_category(
        self,
        category: str,
        limit: int = 50,
        include_archived: bool = False,
    ) -> list[dict]:
        """Get memories by category, ordered by updated_at desc."""
        sql = """
            SELECT id, category, text, created_at, updated_at, access_count, archived
            FROM memories
            WHERE category = ?
        """
        params: list = [category]
        if not include_archived:
            sql += "  AND archived = 0\n"
        sql += "ORDER BY updated_at DESC\nLIMIT ?"
        params.append(limit)

        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_top_memories(self, limit: int = 80) -> list[dict]:
        """Get top memories for injection into AGENTS.md.

        Rules:
        - identity and preference: always included (up to 20 each)
        - entity, event, insight: ranked by hotness score, fill remaining slots

        Returns list of dicts sorted by category then hotness desc.
        """
        result: list[dict] = []
        remaining = limit

        # Phase 1: always-include categories
        for cat in ("identity", "preference", "correction"):
            cap = min(20, remaining)
            rows = self.get_by_category(cat, limit=cap)
            for r in rows:
                r["hotness"] = hotness_score(r["access_count"], r["updated_at"])
            result.extend(rows)
            remaining -= len(rows)

        if remaining <= 0:
            result.sort(key=lambda r: (r["category"], -r.get("hotness", 0)))
            return result

        # Phase 2: ranked categories — fetch candidates and sort by hotness
        candidates: list[dict] = []
        for cat in ("entity", "event", "insight"):
            # Fetch a generous pool to rank from
            rows = self.get_by_category(cat, limit=100)
            for r in rows:
                r["hotness"] = hotness_score(r["access_count"], r["updated_at"])
            candidates.extend(rows)

        # Sort by hotness descending and take top remaining
        candidates.sort(key=lambda r: r["hotness"], reverse=True)
        result.extend(candidates[:remaining])

        # Final sort: by category then hotness desc
        category_order = {"identity": 0, "preference": 1, "correction": 2, "entity": 3, "event": 4, "insight": 5}
        result.sort(key=lambda r: (category_order.get(r["category"], 9), -r.get("hotness", 0)))

        return result

    # ------------------------------------------------------------------
    # Update operations
    # ------------------------------------------------------------------

    def mark_accessed(self, ids: list[int]) -> None:
        """Increment access_count and update updated_at for given ids."""
        if not ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"""UPDATE memories
                SET access_count = access_count + 1,
                    updated_at = ?
                WHERE id IN ({placeholders})""",
            [now] + ids,
        )
        self._conn.commit()

    def archive_cold(self, threshold: float = 0.05, min_age_days: int = 7) -> int:
        """Archive memories with hotness below threshold and older than min_age_days.

        Returns count of archived memories.
        """
        # Fetch all active non-permanent memories
        rows = self._conn.execute(
            """SELECT id, category, access_count, updated_at
               FROM memories
               WHERE archived = 0
                 AND category NOT IN ('identity', 'preference', 'correction')"""
        ).fetchall()

        now = datetime.now(timezone.utc)
        to_archive: list[int] = []

        for r in rows:
            try:
                updated = datetime.fromisoformat(r["updated_at"])
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                age_days = (now - updated).total_seconds() / 86400.0
            except (ValueError, TypeError):
                age_days = 999.0

            if age_days < min_age_days:
                continue

            score = hotness_score(r["access_count"], r["updated_at"])
            if score < threshold:
                to_archive.append(r["id"])

        if to_archive:
            placeholders = ",".join("?" for _ in to_archive)
            now_iso = now.isoformat()
            self._conn.execute(
                f"""UPDATE memories
                    SET archived = 1, updated_at = ?
                    WHERE id IN ({placeholders})""",
                [now_iso] + to_archive,
            )
            self._conn.commit()
            logger.info(f"Archived {len(to_archive)} cold memories (threshold={threshold})")

        return len(to_archive)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return stats: total, by_category counts, archived count."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]

        archived = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 1"
        ).fetchone()[0]

        rows = self._conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories WHERE archived = 0 GROUP BY category"
        ).fetchall()
        by_category = {r["category"]: r["cnt"] for r in rows}

        return {
            "total": total,
            "archived": archived,
            "by_category": by_category,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to keep WAL file size in check."""
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("MemoryStore closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
