"""
SQLite-based persistent checkpointer for conversation memory.

Provides durable conversation state that persists across application restarts.
Uses LangGraph's SqliteSaver for checkpoint management.
"""

import os
import sqlite3
from datetime import datetime, timedelta

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console

from src.utils.logger import custom_theme, get_logger

logger = get_logger()
console = Console(theme=custom_theme)


class PersistentCheckpointer:
    """
    SQLite-backed checkpointer for persistent conversation memory.

    Wraps LangGraph's SqliteSaver with additional session management capabilities
    including listing sessions, cleanup, and deletion.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize persistent checkpointer.

        Args:
            db_path: Path to SQLite database file. Defaults to CHECKPOINT_DB_PATH
                    environment variable or 'knowledges/checkpoints.db'.
        """
        if db_path is None:
            db_path = os.getenv("CHECKPOINT_DB_PATH", "knowledges/checkpoints.db")

        self.db_path = db_path

        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Create connection with WAL mode for better concurrency
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Initialize SqliteSaver
        self.saver = SqliteSaver(self.conn)

        logger.info(f"PersistentCheckpointer initialized at {db_path}")
        console.print(f"[info]Persistent memory enabled: {db_path}[/info]")

    def get_saver(self) -> SqliteSaver:
        """
        Get the underlying SqliteSaver instance.

        Returns:
            SqliteSaver instance for use with LangGraph agents.
        """
        return self.saver

    def list_sessions(self, limit: int = 10) -> list[tuple[str, str]]:
        """
        List recent conversation sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of (thread_id, last_active) tuples.
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT thread_id, MAX(created_at) as last_active
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_active DESC
                LIMIT ?
            """,
                (limit,),
            )
            sessions = cursor.fetchall()
            logger.info(f"Listed {len(sessions)} sessions")
            return sessions
        except sqlite3.OperationalError as e:
            # Table might not exist yet if no checkpoints saved
            logger.debug(f"Could not list sessions: {e}")
            return []
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Delete sessions older than this many days.

        Returns:
            Number of checkpoint entries deleted.
        """
        cutoff = datetime.now() - timedelta(days=days)
        try:
            cursor = self.conn.execute(
                """
                DELETE FROM checkpoints WHERE created_at < ?
            """,
                (cutoff.isoformat(),),
            )
            deleted = cursor.rowcount
            self.conn.commit()
            logger.info(f"Cleaned up {deleted} old checkpoint entries (older than {days} days)")
            return deleted
        except sqlite3.OperationalError as e:
            logger.debug(f"Could not cleanup sessions: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0

    def delete_session(self, thread_id: str) -> bool:
        """
        Delete a specific session.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            True if session was deleted, False otherwise.
        """
        try:
            cursor = self.conn.execute(
                """
                DELETE FROM checkpoints WHERE thread_id = ?
            """,
                (thread_id,),
            )
            self.conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted session: {thread_id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting session {thread_id}: {e}")
            return False

    def get_session_count(self) -> int:
        """
        Get total number of sessions.

        Returns:
            Number of unique sessions.
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT COUNT(DISTINCT thread_id) FROM checkpoints
            """
            )
            count = cursor.fetchone()[0]
            return count
        except sqlite3.OperationalError:
            return 0
        except Exception as e:
            logger.error(f"Error getting session count: {e}")
            return 0

    def close(self):
        """Close the database connection."""
        try:
            self.conn.close()
            logger.info("PersistentCheckpointer closed")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")


def get_checkpointer(use_persistent: bool = None):
    """
    Get appropriate checkpointer based on configuration (sync version).

    Args:
        use_persistent: Override for persistent memory setting.
                       If None, reads from USE_PERSISTENT_MEMORY env var.

    Returns:
        SqliteSaver for persistent memory, or MemorySaver for in-memory.
    """
    if use_persistent is None:
        use_persistent = os.getenv("USE_PERSISTENT_MEMORY", "true").lower() == "true"

    if use_persistent:
        try:
            checkpointer = PersistentCheckpointer()
            return checkpointer.get_saver()
        except Exception as e:
            logger.warning(f"SQLite checkpointer failed, falling back to memory: {e}")
            console.print(f"[warning]Persistent memory unavailable, using in-memory: {e}[/warning]")
            return MemorySaver()
    else:
        logger.info("Using in-memory checkpointer")
        return MemorySaver()


def get_async_checkpointer(use_persistent: bool = None) -> AsyncSqliteSaver | MemorySaver:
    """
    Get appropriate async checkpointer based on configuration.

    Use this for async agents (corrective, adaptive, multi_agent modes).
    IMPORTANT: Call this after setting the event loop with asyncio.set_event_loop()
    to ensure the aiosqlite connection uses the correct loop.

    Args:
        use_persistent: Override for persistent memory setting.
                       If None, reads from USE_PERSISTENT_MEMORY env var.

    Returns:
        AsyncSqliteSaver for persistent memory, or MemorySaver for in-memory.
    """
    import asyncio

    import aiosqlite

    if use_persistent is None:
        use_persistent = os.getenv("USE_PERSISTENT_MEMORY", "true").lower() == "true"

    if use_persistent:
        try:
            db_path = os.getenv("CHECKPOINT_DB_PATH", "knowledges/checkpoints.db")
            # Ensure directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            # Create AsyncSqliteSaver with proper async connection
            async def create_async_saver():
                conn = await aiosqlite.connect(db_path)
                await conn.execute("PRAGMA journal_mode=WAL")
                return AsyncSqliteSaver(conn)

            # Use the current event loop if available, otherwise create a new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                saver = loop.run_until_complete(create_async_saver())
            except RuntimeError:
                # No event loop or loop is closed - create new one
                saver = asyncio.run(create_async_saver())

            logger.info(f"AsyncSqliteSaver initialized at {db_path}")
            console.print(f"[info]Persistent memory enabled (async): {db_path}[/info]")
            return saver
        except Exception as e:
            logger.warning(f"Async SQLite checkpointer failed, falling back to memory: {e}")
            console.print(f"[warning]Persistent memory unavailable, using in-memory: {e}[/warning]")
            return MemorySaver()
    else:
        logger.info("Using in-memory checkpointer (async)")
        return MemorySaver()
