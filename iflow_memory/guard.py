"""记忆守护 — 容错、降级、异常恢复。

提供三个核心防护：
1. safe_db_write — DB 写操作保护（锁超时/磁盘满不崩溃）
2. error_boundary — MCP tool 异常隔离装饰器
3. daemon 异常恢复辅助
"""

import functools
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional, TypeVar, Callable

logger = logging.getLogger("iflow-memory")

T = TypeVar("T")


def safe_db_write(conn: sqlite3.Connection, sql: str, params: tuple = (),
                  commit: bool = True) -> Optional[sqlite3.Cursor]:
    """安全执行 DB 写操作。

    捕获 DB 锁超时、磁盘满等异常，返回 None 而不是崩溃。
    正常时返回 cursor。
    """
    try:
        cur = conn.execute(sql, params)
        if commit:
            conn.commit()
        return cur
    except sqlite3.OperationalError as e:
        err_msg = str(e).lower()
        if "database is locked" in err_msg:
            logger.error(f"[记忆守护] DB 锁超时，写入跳过: {sql[:80]}")
        elif "disk" in err_msg or "full" in err_msg:
            logger.error(f"[记忆守护] 磁盘空间不足，写入跳过: {sql[:80]}")
        else:
            logger.error(f"[记忆守护] DB 写入失败: {e}")
        return None
    except Exception as e:
        logger.error(f"[记忆守护] 未知 DB 错误: {e}")
        return None


def error_boundary(func: Callable) -> Callable:
    """MCP tool 异常隔离装饰器。

    被装饰的函数如果抛异常，不会传播到调用方，
    而是返回一个友好的错误信息。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"[记忆守护] {func.__name__} 异常已隔离: {e}")
            return {
                "content": [{"type": "text", "text": f"内部错误: {e}"}],
                "isError": True,
            }
    return wrapper


def check_disk_space(path: str | Path, min_mb: int = 50) -> bool:
    """检查磁盘剩余空间是否充足。"""
    try:
        stat = os.statvfs(str(path))
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < min_mb:
            logger.warning(f"[记忆守护] 磁盘空间不足: {free_mb:.0f}MB < {min_mb}MB")
            return False
        return True
    except Exception:
        return True  # 无法检测时不阻塞


def check_db_writable(db_path: str | Path) -> bool:
    """检查 DB 文件是否可写。"""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA quick_check")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[记忆守护] DB 不可写: {e}")
        return False


def daemon_health_check(config) -> dict:
    """Daemon 启动时的健康自检。

    Returns:
        {"db_ok": bool, "disk_ok": bool, "embed_ok": bool}
    """
    db_path = Path(config.memory_dir) / "memories.db"
    return {
        "db_ok": check_db_writable(db_path),
        "disk_ok": check_disk_space(config.memory_dir),
        "embed_ok": True,  # 由 daemon 初始化 embedder 后更新
    }
