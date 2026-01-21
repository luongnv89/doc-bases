"""
Metrics tracking for RAG system performance monitoring.

Tracks:
- Query latency
- Retrieval counts
- Success/failure rates
- RAG mode usage
- Session activity
"""

import os
import sqlite3
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table

from src.utils.logger import custom_theme, get_logger

logger = get_logger()
console = Console(theme=custom_theme)


class MetricsTracker:
    """
    SQLite-based metrics tracker for RAG system monitoring.

    Provides persistent storage for query metrics and a dashboard
    for viewing performance statistics.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize metrics tracker.

        Args:
            db_path: Path to metrics database file. Defaults to METRICS_DB_PATH
                    environment variable or 'knowledges/metrics.db'.
        """
        if db_path is None:
            db_path = os.getenv("METRICS_DB_PATH", "knowledges/metrics.db")

        self.db_path = db_path

        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Create connection with WAL mode
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

        logger.info(f"MetricsTracker initialized at {db_path}")

    def _init_tables(self) -> None:
        """Create metrics tables if they don't exist."""
        self.conn.execute(
            """
CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    retrieval_count INTEGER DEFAULT 0,
    rag_mode TEXT NOT NULL,
    knowledge_base TEXT,
    session_id TEXT,
    success BOOLEAN DEFAULT TRUE,
    error TEXT
)
"""
        )

        # Create index for efficient time-based queries
        self.conn.execute(
            """
CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp
ON query_metrics(timestamp)
"""
        )

        self.conn.commit()

    def log_query(
        self,
        query: str,
        latency_ms: int,
        retrieval_count: int = 0,
        rag_mode: str = "basic",
        knowledge_base: str | None = None,
        session_id: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Log a query metric.

        Args:
            query: The user's query text.
            latency_ms: Query processing time in milliseconds.
            retrieval_count: Number of documents retrieved.
            rag_mode: RAG mode used (basic, corrective, adaptive, multi_agent).
            knowledge_base: Name of the knowledge base queried.
            session_id: Session identifier.
            success: Whether the query succeeded.
            error: Error message if query failed.
        """
        self.conn.execute(
            """
            INSERT INTO query_metrics
            (timestamp, query, latency_ms, retrieval_count, rag_mode, knowledge_base, session_id, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (datetime.now().isoformat(), query, latency_ms, retrieval_count, rag_mode, knowledge_base, session_id, success, error),
        )
        self.conn.commit()

        logger.debug(f"Logged query metric: {latency_ms}ms, mode={rag_mode}, success={success}")

    def get_stats(self, days: int = 7) -> dict:
        """
        Get aggregated statistics for the specified period.

        Args:
            days: Number of days to include in statistics.

        Returns:
            Dictionary with aggregated metrics.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Total queries
        total = self.conn.execute("SELECT COUNT(*) FROM query_metrics WHERE timestamp > ?", (cutoff,)).fetchone()[0]

        # Successful queries
        success = self.conn.execute("SELECT COUNT(*) FROM query_metrics WHERE timestamp > ? AND success = TRUE", (cutoff,)).fetchone()[0]

        # Average latency
        avg_latency = self.conn.execute("SELECT AVG(latency_ms) FROM query_metrics WHERE timestamp > ?", (cutoff,)).fetchone()[0] or 0

        # P50 latency
        p50_latency = self.conn.execute(
            """
            SELECT latency_ms FROM query_metrics
            WHERE timestamp > ?
            ORDER BY latency_ms
            LIMIT 1 OFFSET (SELECT COUNT(*) / 2 FROM query_metrics WHERE timestamp > ?)
        """,
            (cutoff, cutoff),
        ).fetchone()
        p50_latency = p50_latency[0] if p50_latency else 0

        # P95 latency
        p95_latency = self.conn.execute(
            """
            SELECT latency_ms FROM query_metrics
            WHERE timestamp > ?
            ORDER BY latency_ms
            LIMIT 1 OFFSET (SELECT COUNT(*) * 95 / 100 FROM query_metrics WHERE timestamp > ?)
        """,
            (cutoff, cutoff),
        ).fetchone()
        p95_latency = p95_latency[0] if p95_latency else 0

        # Queries by mode
        by_mode = self.conn.execute(
            """
            SELECT rag_mode, COUNT(*) FROM query_metrics
            WHERE timestamp > ? GROUP BY rag_mode
        """,
            (cutoff,),
        ).fetchall()

        # Queries by knowledge base
        by_kb = self.conn.execute(
            """
            SELECT knowledge_base, COUNT(*) FROM query_metrics
            WHERE timestamp > ? AND knowledge_base IS NOT NULL
            GROUP BY knowledge_base
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """,
            (cutoff,),
        ).fetchall()

        # Average retrieval count
        avg_retrieval = self.conn.execute("SELECT AVG(retrieval_count) FROM query_metrics WHERE timestamp > ?", (cutoff,)).fetchone()[0] or 0

        return {
            "total_queries": total,
            "successful_queries": success,
            "success_rate": (success / total * 100) if total > 0 else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "avg_retrieval_count": round(avg_retrieval, 2),
            "queries_by_mode": dict(by_mode),
            "queries_by_knowledge_base": dict(by_kb),
            "period_days": days,
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict]:
        """
        Get recent query errors.

        Args:
            limit: Maximum number of errors to return.

        Returns:
            List of error records.
        """
        cursor = self.conn.execute(
            """
            SELECT timestamp, query, error, rag_mode, knowledge_base
            FROM query_metrics
            WHERE success = FALSE
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        errors = []
        for row in cursor.fetchall():
            errors.append(
                {
                    "timestamp": row[0],
                    "query": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    "error": row[2],
                    "rag_mode": row[3],
                    "knowledge_base": row[4],
                }
            )

        return errors

    def display_dashboard(self, days: int = 7) -> None:
        """
        Display metrics dashboard in console.

        Args:
            days: Number of days to include in dashboard.
        """
        stats = self.get_stats(days)

        # Main stats table
        main_table = Table(title=f"[header]RAG Metrics Dashboard (Last {days} days)[/header]")
        main_table.add_column("Metric", style="info")
        main_table.add_column("Value", style="success")

        main_table.add_row("Total Queries", str(stats["total_queries"]))
        main_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
        main_table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
        main_table.add_row("P50 Latency", f"{stats['p50_latency_ms']}ms")
        main_table.add_row("P95 Latency", f"{stats['p95_latency_ms']}ms")
        main_table.add_row("Avg Docs Retrieved", f"{stats['avg_retrieval_count']:.1f}")

        console.print(main_table)

        # Mode breakdown table
        if stats["queries_by_mode"]:
            mode_table = Table(title="[header]Queries by RAG Mode[/header]")
            mode_table.add_column("Mode", style="info")
            mode_table.add_column("Count", style="success")

            for mode, count in stats["queries_by_mode"].items():
                mode_table.add_row(mode, str(count))

            console.print(mode_table)

        # Knowledge base breakdown
        if stats["queries_by_knowledge_base"]:
            kb_table = Table(title="[header]Top Knowledge Bases[/header]")
            kb_table.add_column("Knowledge Base", style="info")
            kb_table.add_column("Queries", style="success")

            for kb, count in stats["queries_by_knowledge_base"].items():
                kb_table.add_row(kb or "Unknown", str(count))

            console.print(kb_table)

        # Recent errors
        errors = self.get_recent_errors(5)
        if errors:
            error_table = Table(title="[header]Recent Errors[/header]")
            error_table.add_column("Time", style="info")
            error_table.add_column("Query", style="warning")
            error_table.add_column("Error", style="error")

            for err in errors:
                error_table.add_row(err["timestamp"][:19], err["query"][:30] + "...", err["error"][:40] if err["error"] else "Unknown")

            console.print(error_table)

    def cleanup_old_metrics(self, days: int = 90) -> int:
        """
        Delete metrics older than specified days.

        Args:
            days: Delete metrics older than this many days.

        Returns:
            Number of records deleted.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute("DELETE FROM query_metrics WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleaned up {deleted} old metric entries")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        try:
            self.conn.close()
            logger.info("MetricsTracker closed")
        except Exception as e:
            logger.error(f"Error closing metrics tracker: {e}")


# Global metrics tracker instance
_metrics_tracker: MetricsTracker | None = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Get or create the global metrics tracker instance.

    Returns:
        MetricsTracker singleton instance.
    """
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker
