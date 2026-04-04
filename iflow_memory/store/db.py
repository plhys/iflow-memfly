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

# 密钥/凭证检测模式 — 匹配到的记忆在 add() 时自动脱敏
# 灵感来源：Claude Code secretScanner（gitleaks 正则），但我们不阻止写入，
# 而是把敏感值替换为 [REDACTED]，保留语义但去掉明文。
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key",    re.compile(r'(?i)(api[_\- ]?key|apikey)\s*[:=：]\s*\S{20,}')),
    ("token",      re.compile(r'(?i)(token|secret|password)\s*[:=：]\s*\S{16,}')),
    ("bearer",     re.compile(r'(?i)Bearer\s+(\S{20,})')),
    ("private_key", re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----')),
    ("hex_secret", re.compile(r'(?<![A-Za-z0-9])(?<!commit )(?<!hash )[0-9a-fA-F]{32,64}(?![A-Za-z0-9])')),
    ("app_id",     re.compile(r'cli_[a-f0-9]{16,}')),
    # GitHub PAT (classic & fine-grained)
    ("github_pat", re.compile(r'ghp_[A-Za-z0-9]{36,}')),
    ("github_oauth", re.compile(r'gho_[A-Za-z0-9]{36,}')),
    ("github_fine", re.compile(r'github_pat_[A-Za-z0-9]{22,}')),
    # OpenAI / Anthropic / common AI API keys
    ("openai_key", re.compile(r'sk-[A-Za-z0-9]{32,}')),
    ("anthropic_key", re.compile(r'sk-ant-[A-Za-z0-9\-]{32,}')),
    # AWS keys
    ("aws_key",    re.compile(r'AKIA[A-Z0-9]{16}')),
    # Cloudflare API tokens
    ("cf_token",   re.compile(r'[A-Za-z0-9_\-]{40}(?=.*cloudflare)', re.IGNORECASE)),
]


def _redact_secrets(text: str) -> tuple[str, bool]:
    """检测并脱敏文本中的密钥/凭证。

    不阻止写入，而是把敏感值替换为 [REDACTED]，保留语义但去掉明文。

    Returns:
        (脱敏后的文本, 是否发生了脱敏)
    """
    _REPLACEMENTS = {
        "api_key":     lambda m: m.group(1) + "=[REDACTED]",
        "token":       lambda m: m.group(1) + "=[REDACTED]",
        "bearer":      lambda _: "Bearer [REDACTED]",
        "private_key": lambda _: "[REDACTED PRIVATE KEY]",
        "hex_secret":  lambda _: "[REDACTED]",
        "app_id":      lambda _: "[REDACTED_APP_ID]",
        "github_pat":  lambda _: "ghp_[REDACTED]",
        "github_oauth": lambda _: "gho_[REDACTED]",
        "github_fine": lambda _: "github_pat_[REDACTED]",
        "openai_key":  lambda _: "sk-[REDACTED]",
        "anthropic_key": lambda _: "sk-ant-[REDACTED]",
        "aws_key":     lambda _: "AKIA[REDACTED]",
        "cf_token":    lambda _: "[REDACTED_CF_TOKEN]",
    }
    redacted = False
    result = text
    for name, pattern in _SECRET_PATTERNS:
        replacement = _REPLACEMENTS.get(name)
        if not replacement:
            continue
        new_result = pattern.sub(replacement, result)
        if new_result != result:
            result = new_result
            redacted = True
    return result, redacted


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
SCHEMA_VERSION = 8

# 半衰期配置（天）——按类别区分
_HALFLIFE_LONG = 90.0   # identity / correction / preference（长期事实）
_HALFLIFE_SHORT = 14.0  # entity / event / insight（短期事件）
_LONG_LIFE_CATEGORIES = frozenset({"identity", "correction", "preference"})


def _serialize_f32(vec: list[float]) -> bytes:
    """将 float 列表序列化为 little-endian float32 bytes（sqlite-vec 格式）。"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes) -> list[float]:
    """从 bytes 反序列化为 float 列表。"""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def hotness_score(access_count: int, updated_at: str,
                   category: str | None = None) -> float:
    """Calculate hotness score for memory ranking.

    Formula: sigmoid(log1p(access_count)) * exp(-0.693 * age_days / halflife)

    The sigmoid of log1p(access_count) gives a 0–1 popularity factor that
    grows quickly for the first few accesses then saturates.
    The exponential decay halves the score every *halflife* days since last
    update.  ``halflife`` depends on category:
      - identity / correction / preference → 90 days (long-term facts)
      - entity / event / insight (default) → 14 days (short-term events)
    """
    # Popularity factor
    popularity = 1.0 / (1.0 + math.exp(-math.log1p(access_count)))

    # Age decay — halflife varies by category
    halflife = (
        _HALFLIFE_LONG
        if category and category in _LONG_LIFE_CATEGORIES
        else _HALFLIFE_SHORT
    )

    try:
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0
    except (ValueError, TypeError):
        age_days = 30.0  # fallback: treat as old

    decay = math.exp(-0.693 * age_days / halflife)

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
        is_fresh = version < 1

        if version < 1:
            self._migrate_v1()
        if version < 2:
            self._migrate_v2()
        if version < 3:
            self._migrate_v3()
        if version < 4:
            self._migrate_v4()
        if version < 5:
            self._migrate_v5()
        if version < 6:
            self._migrate_v6()
        if version < 7:
            self._migrate_v7()
        if version < 8:
            self._migrate_v8()

        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

        if is_fresh:
            self._seed_memories()

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

    def _migrate_v5(self) -> None:
        """Migration v5: memory_links table for knowledge graph."""
        logger.info("Running migration v5: creating memory_links table")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL REFERENCES memories(id),
                target_id   INTEGER NOT NULL REFERENCES memories(id),
                strength    REAL    NOT NULL DEFAULT 0.0,
                link_type   TEXT    NOT NULL DEFAULT 'similar',
                created_at  TEXT    NOT NULL,
                UNIQUE(source_id, target_id)
            );

            CREATE INDEX IF NOT EXISTS idx_links_source
                ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target
                ON memory_links(target_id);
        """)
        self._conn.commit()

    def _migrate_v6(self) -> None:
        """Migration v6: scope column for memory partitioning.

        scope 字段实现记忆分区隔离：
        - "global": 所有渠道可见（默认）
        - "private": 仅创建者渠道可见
        这样同一个 daemon 可以服务多个身份（如妖妖酒和天灵灵），
        而不会互相污染记忆。
        """
        logger.info("Running migration v6: adding scope column")
        try:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def _migrate_v7(self) -> None:
        """Migration v7: source_file and source_line columns for provenance tracking.

        记录每条记忆的来源文件路径和行号，便于搜索结果定位原文。
        旧数据的 source_file/source_line 为 NULL，属正常情况。
        """
        logger.info("Running migration v7: adding source_file and source_line columns")
        try:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN source_file TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN source_line INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        self._conn.commit()

    def _migrate_v8(self) -> None:
        """Migration v8: needs_embed column for deferred embedding.

        当 embedding 生成失败时，记忆仍写入 SQLite 但标记 needs_embed=1，
        由 daemon 维护周期补算 embedding。
        """
        logger.info("Running migration v8: adding needs_embed column")
        try:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN needs_embed INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # 列已存在
        self._conn.commit()

    # ------------------------------------------------------------------
    # Seed memories (pre-installed lessons for new users)
    # ------------------------------------------------------------------

    _SEED_MEMORIES: list[tuple[str, str]] = [
        ("insight", "iFlow CLI 的 sub-agent（task tool）使用 write_file 写工作区外的文件会被拒绝，之后 shell 命令可能陷入死循环空转。操作工作区外的文件时，应在主对话中直接用 shell 命令写，不要走 sub-agent。"),
        ("insight", "iFlow CLI 上下文压缩（lightweightCompress）会物理删除 session 文件中的旧消息并重写文件。如果外部系统用消息计数追踪进度，文件重写后计数会错位，导致后续新消息永远读不到。"),
        ("insight", "iFlow CLI 的 session 文件不是只追加的——每次压缩或保存都会完整重写。依赖文件偏移量或行数做增量处理的方案都不可靠，应使用内容哈希做去重。"),
        ("insight", "iFlow CLI 的 COMPRESSION_RANGE=0.5，每次压缩移除约 50% 的消息。经过多轮压缩后上下文仍可能溢出，此时最近的对话会被丢弃。"),
        ("insight", "save_memory 工具保存的记忆需要等 daemon 下一轮处理才会注入到 AGENTS.md。如果保存后立即开新 session，新 session 可能还看不到刚保存的记忆。"),
        ("insight", "AGENTS.md 注入的内容有长度限制。如果记忆条目过多，injector 会按优先级截断。关键信息应保持简短，避免写入大段文本。"),
    ]

    def _seed_memories(self) -> None:
        """Write pre-installed lessons into a fresh database."""
        count = 0
        for category, text in self._SEED_MEMORIES:
            try:
                self.add(category, text, source_session="seed")
                count += 1
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to seed memory: {e}")
        if count:
            logger.info(f"Seeded {count} pre-installed memories")

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

    def get_memories_by_date(self, date_str: str, limit: int = 50) -> list[dict]:
        """获取指定日期创建的活跃记忆。"""
        rows = self._conn.execute(
            """SELECT category, text, created_at
               FROM memories
               WHERE archived = 0
                 AND created_at LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (f"{date_str}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_state_snapshot_by_date(self, date_str: str) -> dict | None:
        """获取指定日期最新的状态快照。"""
        rows = self._conn.execute(
            """SELECT goal, progress, decisions, next_steps, critical_context
               FROM state_snapshots
               WHERE date = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (date_str,),
        ).fetchall()
        if rows:
            return dict(rows[0])
        return None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, category: str, text: str, source_session: str = "",
            embedding: Optional[list[float]] = None,
            scope: str = "global",
            source_file: Optional[str] = None,
            source_line: Optional[int] = None) -> tuple[int, bool]:
        """Insert a new memory, or return existing id if duplicate.

        Raises ValueError if category is invalid or text is empty.
        Returns (row_id, is_new): row_id is the id of the new or existing
        memory; is_new is True only when a new row was actually inserted.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty")

        # 密钥兜底：即使 LLM 提取了含密钥的记忆，入库前也会脱敏
        text, was_redacted = _redact_secrets(text)
        if was_redacted:
            logger.info(f"[密钥过滤] 记忆已脱敏: {text[:60]}...")

        # Dedup: check for existing active memory with same text + category
        existing = self._conn.execute(
            "SELECT id FROM memories WHERE category = ? AND text = ? AND archived = 0",
            (category, text),
        ).fetchone()
        if existing:
            self.mark_accessed([existing[0]])
            logger.debug(f"Dedup: memory #{existing[0]} already exists, skipped")
            return existing[0], False

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
                return row["id"], False
            if _jaccard_similarity(norm_old, norm_new) >= _DEDUP_SIMILARITY_THRESHOLD:
                self.mark_accessed([row["id"]])
                logger.info(
                    f"Dedup (fuzzy): memory #{row['id']} similar, skipped. "
                    f"Existing: {row['text'][:50]}... | New: {text[:50]}..."
                )
                return row["id"], False

        now = datetime.now(timezone.utc).isoformat()
        embed_blob = _serialize_f32(embedding) if embedding else None
        needs_embed = 1 if (embed_blob is None and self._vec_enabled) else 0
        cur = self._conn.execute(
            """INSERT INTO memories (category, text, source_session, created_at, updated_at, embedding, scope, source_file, source_line, needs_embed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (category, text, source_session, now, now, embed_blob, scope, source_file, source_line, needs_embed),
        )
        row_id = cur.lastrowid

        # 同步写入向量索引（同一事务内）
        if embedding and self._vec_enabled:
            try:
                self._conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize_f32(embedding)),
                )
            except Exception as e:
                logger.warning(f"Failed to insert vec for memory #{row_id}: {e}")

        self._conn.commit()

        logger.debug(f"Added memory #{row_id} [{category}]: {text[:60]}")
        return row_id, True

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
        scope: str | None = None,
        date_from: str | None = None,
    ) -> list[dict]:
        """Full-text search across memories with LIKE fallback.

        First tries FTS5 MATCH (trigram). If no results, falls back to
        LIKE query which handles short keywords better.
        Optionally filter by category, scope, and/or date_from. Returns list of dicts with
        id, category, text, created_at, access_count, rank.
        Increments access_count for all returned results.
        """
        query = query.strip()
        if not query:
            return []

        # Phase 1: FTS5 MATCH — 先精确短语，无结果则拆词 OR
        def _build_fts_query(q: str) -> list[str]:
            """返回多个 FTS5 查询候选，优先级从高到低。"""
            escaped = q.replace('"', '""')
            candidates = [f'"{escaped}"']  # 精确短语
            # 按空格拆词，用 OR 扩大召回
            parts = q.split()
            if len(parts) > 1:
                or_terms = " OR ".join(f'"{p.replace(chr(34), chr(34)*2)}"' for p in parts if len(p) >= 2)
                if or_terms:
                    candidates.append(or_terms)
            return candidates

        rows = []
        for fts_query in _build_fts_query(query):
            sql = """
                SELECT m.id, m.category, m.text, m.created_at,
                       m.access_count, m.source_file, m.source_line, rank
                FROM memories_fts fts
                JOIN memories m ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.archived = 0
            """
            params: list = [fts_query]
            if category:
                sql += "  AND m.category = ?\n"
                params.append(category)
            if scope is not None:
                sql += "  AND m.scope = ?\n"
                params.append(scope)
            if date_from is not None:
                sql += "  AND m.created_at >= ?\n"
                params.append(date_from)
            sql += "ORDER BY rank\nLIMIT ?"
            params.append(limit)
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if rows:
                break  # 精确匹配有结果就用，否则降级到 OR

        # Phase 2: LIKE fallback when FTS5 returns nothing
        if not rows:
            sql = """
                SELECT id, category, text, created_at,
                       access_count, source_file, source_line, 0 as rank
                FROM memories
                WHERE archived = 0
                  AND text LIKE ?
            """
            params = [f"%{query}%"]
            if category:
                sql += "  AND category = ?\n"
                params.append(category)
            if scope is not None:
                sql += "  AND scope = ?\n"
                params.append(scope)
            if date_from is not None:
                sql += "  AND created_at >= ?\n"
                params.append(date_from)
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
        scope: str | None = None,
        date_from: str | None = None,
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
            return self.search(query, category=category, limit=limit, scope=scope, date_from=date_from)

        # CJK 自适应权重
        cjk_ratio = _cjk_ratio(query)
        if cjk_ratio > 0.3:
            w_vec, w_fts = 0.8, 0.2
        else:
            w_vec, w_fts = 0.5, 0.5

        k = 60  # RRF 常数

        # === FTS5 路 ===
        fts_results = self.search(query, category=category, limit=limit * 2, scope=scope, date_from=date_from)
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

            # 过滤已归档的，以及按 category/scope/date_from 筛选
            active = []
            for vr in vec_rows:
                filter_sql = "SELECT id FROM memories WHERE id = ? AND archived = 0"
                filter_params: list = [vr[0]]
                if category:
                    filter_sql += " AND category = ?"
                    filter_params.append(category)
                if scope is not None:
                    filter_sql += " AND scope = ?"
                    filter_params.append(scope)
                if date_from is not None:
                    filter_sql += " AND created_at >= ?"
                    filter_params.append(date_from)
                mem = self._conn.execute(filter_sql, filter_params).fetchone()
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
            f"""SELECT id, category, text, created_at, access_count,
                       source_file, source_line
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
        scope: str | None = None,
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
        if scope is not None:
            sql += "  AND scope = ?\n"
            params.append(scope)
        sql += "ORDER BY updated_at DESC\nLIMIT ?"
        params.append(limit)

        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_top_memories(self, limit: int = 50, scope: str | None = None) -> list[dict]:
        """Get top memories for injection into AGENTS.md.

        三阶段选择：
        1. 身份锚定 — identity/correction 按时间倒序（豁免 hotness），preference 按 hotness 排序
        2. 热度排序 — entity/event/insight 按 hotness 填充剩余名额
        3. 图谱扩展 — 对入选记忆做一跳关联，把高关联但未入选的记忆补进来

        每条记忆附带 freshness 标记（天数），供 injector 做新鲜度展示。

        Returns list of dicts sorted by category then hotness desc.
        """
        result: list[dict] = []
        seen_ids: set[int] = set()
        remaining = limit

        # Phase 1: 身份锚定
        # identity/correction 豁免 hotness（纠错和身份天然低频但极重要，按时间倒序取）
        # preference 按 hotness 排序（偏好有冷热之分）
        for cat in ("identity", "preference", "correction"):
            cap = min(8, remaining)
            if cat == "preference":
                # preference 按 hotness 排序，冷偏好沉底
                rows = self.get_by_category(cat, limit=50, scope=scope)
                for r in rows:
                    r["hotness"] = hotness_score(r["access_count"], r["updated_at"], category=cat)
                    r["age_days"] = self._calc_age_days(r["updated_at"])
                rows.sort(key=lambda r: r["hotness"], reverse=True)
                top_rows = rows[:cap]
            else:
                # identity/correction 按时间倒序（get_by_category 默认排序），不看 hotness
                top_rows = self.get_by_category(cat, limit=cap, scope=scope)
                for r in top_rows:
                    r["hotness"] = hotness_score(r["access_count"], r["updated_at"], category=cat)
                    r["age_days"] = self._calc_age_days(r["updated_at"])
            for r in top_rows:
                seen_ids.add(r["id"])
            result.extend(top_rows)
            remaining -= len(top_rows)

        if remaining <= 0:
            return self._sort_by_category(result)

        # Phase 2: 热度排序 — 从知识/事件/经验中选最活跃的
        candidates: list[dict] = []
        for cat in ("entity", "event", "insight"):
            rows = self.get_by_category(cat, limit=100, scope=scope)
            for r in rows:
                if r["id"] in seen_ids:
                    continue
                r["hotness"] = hotness_score(r["access_count"], r["updated_at"], category=cat)
                r["age_days"] = self._calc_age_days(r["updated_at"])
                candidates.append(r)

        candidates.sort(key=lambda r: r["hotness"], reverse=True)
        phase2_picks = candidates[:remaining]
        for r in phase2_picks:
            seen_ids.add(r["id"])
        result.extend(phase2_picks)
        remaining = limit - len(result)

        # Phase 3: 图谱扩展 — 对 Phase 2 入选的记忆做一跳关联
        # 把高关联但未入选的记忆补进来，让注入内容有"连贯性"
        if remaining > 0 and phase2_picks:
            graph_bonus = self._expand_via_graph(
                seed_ids=[r["id"] for r in phase2_picks],
                seen_ids=seen_ids,
                max_expand=min(remaining, 10),
                min_strength=0.35,
            )
            result.extend(graph_bonus)

        return self._sort_by_category(result)

    @staticmethod
    def _calc_age_days(updated_at: str) -> float:
        """计算记忆距今天数。"""
        try:
            updated = datetime.fromisoformat(updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 999.0

    @staticmethod
    def _sort_by_category(memories: list[dict]) -> list[dict]:
        """按类别顺序 + 热度降序排列。"""
        order = {"identity": 0, "preference": 1, "correction": 2,
                 "entity": 3, "event": 4, "insight": 5}
        memories.sort(key=lambda r: (order.get(r["category"], 9), -r.get("hotness", 0)))
        return memories

    def _expand_via_graph(
        self,
        seed_ids: list[int],
        seen_ids: set[int],
        max_expand: int = 10,
        min_strength: float = 0.35,
    ) -> list[dict]:
        """图谱一跳扩展：从种子记忆出发，找关联最强的未入选记忆。

        这是融合 A 的核心巧思——不是随机补记忆，而是沿着知识图谱的边
        找到跟已选记忆最相关的邻居，让注入内容形成"记忆簇"。
        """
        neighbor_scores: dict[int, float] = {}  # id -> 最高关联强度

        for seed_id in seed_ids:
            linked = self.get_linked_memories(
                seed_id, min_strength=min_strength, limit=5
            )
            for link in linked:
                lid = link["id"]
                if lid in seen_ids:
                    continue
                strength = link.get("strength", 0.0)
                if lid not in neighbor_scores or strength > neighbor_scores[lid]:
                    neighbor_scores[lid] = strength

        if not neighbor_scores:
            return []

        # 按关联强度排序，取 top
        sorted_neighbors = sorted(
            neighbor_scores.items(), key=lambda x: x[1], reverse=True
        )[:max_expand]

        bonus_ids = [n[0] for n in sorted_neighbors]
        if not bonus_ids:
            return []

        placeholders = ",".join("?" for _ in bonus_ids)
        rows = self._conn.execute(
            f"""SELECT id, category, text, created_at, updated_at, access_count, archived
                FROM memories
                WHERE id IN ({placeholders}) AND archived = 0""",
            bonus_ids,
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["hotness"] = hotness_score(d["access_count"], d["updated_at"], category=d.get("category"))
            d["age_days"] = self._calc_age_days(d["updated_at"])
            d["via_graph"] = True  # 标记为图谱扩展来的
            results.append(d)

        return results

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

    def get_needs_embed(self, limit: int = 50) -> list[dict]:
        """返回 needs_embed=1 的记忆列表（待补算 embedding）。"""
        rows = self._conn.execute(
            """SELECT id, category, text
               FROM memories
               WHERE needs_embed = 1 AND archived = 0
               ORDER BY id ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_embedding(self, memory_id: int, embedding: list[float]) -> None:
        """更新指定记忆的 embedding 并清除 needs_embed 标记。"""
        embed_blob = _serialize_f32(embedding)
        self._conn.execute(
            "UPDATE memories SET embedding = ?, needs_embed = 0 WHERE id = ?",
            (embed_blob, memory_id),
        )
        # 同步写入向量索引
        if self._vec_enabled:
            try:
                # 先尝试删除旧的向量（如果存在）
                self._conn.execute(
                    "DELETE FROM memories_vec WHERE rowid = ?", (memory_id,)
                )
                self._conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (memory_id, embed_blob),
                )
            except Exception as e:
                logger.warning(f"Failed to update vec for memory #{memory_id}: {e}")
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

            score = hotness_score(r["access_count"], r["updated_at"], category=r["category"])
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

    def archive_by_ids(self, ids: list[int]) -> int:
        """批量归档指定 ID 的记忆。

        Returns: 实际归档的数量。
        """
        if not ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"UPDATE memories SET archived = 1, updated_at = ? WHERE id IN ({placeholders}) AND archived = 0",
            [now] + list(ids),
        )
        self._conn.commit()
        count = cur.rowcount
        if count:
            logger.info(f"Archived {count} memories by ID")
        return count

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
    # Knowledge graph — memory linking
    # ------------------------------------------------------------------

    def create_links_for_memory(
        self,
        memory_id: int,
        embedding: Optional[list[float]] = None,
        similarity_threshold: float = 0.3,
        max_links: int = 5,
    ) -> int:
        """为一条新记忆创建与已有记忆的关联。

        策略：
        1. 如果有 embedding，用余弦相似度找最近邻
        2. 否则用关键词 Jaccard 相似度作为 fallback
        同一对记忆只建一条双向链接（UNIQUE 约束保证）。

        Returns: 创建的链接数。
        """
        # 获取新记忆的文本和分类
        row = self._conn.execute(
            "SELECT category, text FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            return 0

        new_text = row["text"]
        new_category = row["category"]
        now = datetime.now(timezone.utc).isoformat()
        created = 0

        if embedding and self._vec_enabled and self.embed_dim > 0:
            # 向量路径：用 sqlite-vec 找最近邻
            try:
                vec_rows = self._conn.execute(
                    """SELECT rowid, distance
                       FROM memories_vec
                       WHERE embedding MATCH ?
                       ORDER BY distance
                       LIMIT ?""",
                    (_serialize_f32(embedding), max_links + 5),
                ).fetchall()

                for vr in vec_rows:
                    target_id = vr[0]
                    if target_id == memory_id:
                        continue
                    # distance 是 L2 距离，转换为余弦相似度近似
                    # 对于归一化向量：cosine_sim ≈ 1 - distance²/2
                    distance = vr[1]
                    similarity = max(0.0, 1.0 - distance * distance / 2.0)
                    if similarity < similarity_threshold:
                        continue
                    # 检查目标未归档
                    active = self._conn.execute(
                        "SELECT id FROM memories WHERE id = ? AND archived = 0",
                        (target_id,),
                    ).fetchone()
                    if not active:
                        continue
                    if self._insert_link(memory_id, target_id, similarity, "similar", now):
                        created += 1
                    if created >= max_links:
                        break
            except Exception as e:
                logger.warning(f"[知识图谱] 向量关联失败，降级为关键词: {e}")
                created += self._create_links_by_keywords(
                    memory_id, new_text, new_category, similarity_threshold, max_links, now
                )
        else:
            # 关键词 fallback
            created += self._create_links_by_keywords(
                memory_id, new_text, new_category, similarity_threshold, max_links, now
            )

        if created > 0:
            self._conn.commit()
            logger.debug(f"[知识图谱] 记忆 #{memory_id} 建立 {created} 条关联")
        return created

    def _create_links_by_keywords(
        self,
        memory_id: int,
        text: str,
        category: str,
        threshold: float,
        max_links: int,
        now: str,
    ) -> int:
        """用 Jaccard bigram 相似度创建关联（无 embedding 时的 fallback）。"""
        norm_new = _normalize_text(text)
        # 取最近 200 条活跃记忆做候选
        candidates = self._conn.execute(
            """SELECT id, text FROM memories
               WHERE archived = 0 AND id != ?
               ORDER BY updated_at DESC
               LIMIT 200""",
            (memory_id,),
        ).fetchall()

        scored: list[tuple[int, float]] = []
        for row in candidates:
            norm_old = _normalize_text(row["text"])
            sim = _jaccard_similarity(norm_old, norm_new)
            if sim >= threshold:
                scored.append((row["id"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        created = 0
        for target_id, sim in scored[:max_links]:
            if self._insert_link(memory_id, target_id, sim, "similar", now):
                created += 1
        return created

    def _insert_link(
        self,
        source_id: int,
        target_id: int,
        strength: float,
        link_type: str,
        now: str,
    ) -> bool:
        """插入一条链接（双向），忽略重复。不 commit，由调用方统一提交。"""
        # 保证 source < target 以避免重复
        a, b = min(source_id, target_id), max(source_id, target_id)
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO memory_links
                   (source_id, target_id, strength, link_type, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (a, b, strength, link_type, now),
            )
            return True
        except Exception as e:
            logger.debug(f"[知识图谱] 链接插入跳过 ({a}->{b}): {e}")
            return False

    def get_linked_memories(
        self,
        memory_id: int,
        min_strength: float = 0.0,
        limit: int = 10,
    ) -> list[dict]:
        """获取与指定记忆关联的记忆列表。"""
        rows = self._conn.execute(
            """SELECT m.id, m.category, m.text, m.created_at, m.access_count,
                      ml.strength, ml.link_type
               FROM memory_links ml
               JOIN memories m ON m.id = CASE
                   WHEN ml.source_id = ? THEN ml.target_id
                   ELSE ml.source_id
               END
               WHERE (ml.source_id = ? OR ml.target_id = ?)
                 AND ml.strength >= ?
                 AND m.archived = 0
               ORDER BY ml.strength DESC
               LIMIT ?""",
            (memory_id, memory_id, memory_id, min_strength, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def expand_with_links(
        self,
        results: list[dict],
        max_expand: int = 3,
        min_strength: float = 0.3,
    ) -> list[dict]:
        """扩展搜索结果：为每条结果附加关联记忆。

        在每个 result dict 中添加 'linked' 字段。
        """
        for r in results:
            linked = self.get_linked_memories(
                r["id"], min_strength=min_strength, limit=max_expand
            )
            r["linked"] = linked
        return results

    def get_knowledge_graph(self, min_strength: float = 0.2, limit: int = 500) -> dict:
        """导出知识图谱数据（用于可视化）。

        Returns:
            {"nodes": [...], "links": [...]}
        """
        links = self._conn.execute(
            """SELECT ml.source_id, ml.target_id, ml.strength, ml.link_type
               FROM memory_links ml
               JOIN memories m1 ON m1.id = ml.source_id AND m1.archived = 0
               JOIN memories m2 ON m2.id = ml.target_id AND m2.archived = 0
               WHERE ml.strength >= ?
               ORDER BY ml.strength DESC
               LIMIT ?""",
            (min_strength, limit),
        ).fetchall()

        node_ids: set[int] = set()
        link_list: list[dict] = []
        for row in links:
            node_ids.add(row["source_id"])
            node_ids.add(row["target_id"])
            link_list.append({
                "source": row["source_id"],
                "target": row["target_id"],
                "strength": row["strength"],
                "type": row["link_type"],
            })

        nodes: list[dict] = []
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            rows = self._conn.execute(
                f"""SELECT id, category, text, access_count
                    FROM memories
                    WHERE id IN ({placeholders}) AND archived = 0""",
                list(node_ids),
            ).fetchall()
            for r in rows:
                nodes.append({
                    "id": r["id"],
                    "category": r["category"],
                    "text": r["text"][:80],
                    "access_count": r["access_count"],
                })

        return {"nodes": nodes, "links": link_list}

    def link_stats(self) -> dict:
        """知识图谱统计信息。"""
        total = self._conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
        avg_strength = self._conn.execute(
            "SELECT AVG(strength) FROM memory_links"
        ).fetchone()[0] or 0.0
        return {"total_links": total, "avg_strength": round(avg_strength, 3)}

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
