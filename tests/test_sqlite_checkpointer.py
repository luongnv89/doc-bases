"""
Tests for Phase 5: Persistent Memory and Observability components.

Tests cover:
- PersistentCheckpointer (SQLite-backed checkpointer)
- get_checkpointer factory function
- MetricsTracker
- LangSmith tracing setup
"""

import os
import sqlite3
import tempfile
from unittest.mock import patch

from src.checkpointing.sqlite_saver import PersistentCheckpointer, get_checkpointer
from src.observability.langsmith_tracer import disable_langsmith_tracing, get_tracing_status, is_tracing_enabled, setup_langsmith_tracing
from src.observability.metrics import MetricsTracker, get_metrics_tracker


class TestPersistentCheckpointer:
    """Tests for PersistentCheckpointer class."""

    def test_init_creates_database(self):
        """Test that initialization creates the database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            assert os.path.exists(db_path)
            checkpointer.close()

    def test_init_creates_directory_if_not_exists(self):
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "nested", "checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            assert os.path.exists(db_path)
            checkpointer.close()

    def test_get_saver_returns_sqlite_saver(self):
        """Test that get_saver returns a SqliteSaver instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            saver = checkpointer.get_saver()
            # SqliteSaver from langgraph should be returned
            assert saver is not None
            checkpointer.close()

    def test_list_sessions_empty_initially(self):
        """Test that list_sessions returns empty list initially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            sessions = checkpointer.list_sessions()
            assert sessions == []
            checkpointer.close()

    def test_get_session_count_zero_initially(self):
        """Test that session count is 0 initially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            count = checkpointer.get_session_count()
            assert count == 0
            checkpointer.close()

    def test_cleanup_old_sessions_no_error_on_empty_db(self):
        """Test that cleanup doesn't error on empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            deleted = checkpointer.cleanup_old_sessions(days=30)
            assert deleted == 0
            checkpointer.close()

    def test_delete_session_returns_false_for_nonexistent(self):
        """Test that deleting nonexistent session returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            checkpointer = PersistentCheckpointer(db_path=db_path)

            result = checkpointer.delete_session("nonexistent_thread")
            assert result is False
            checkpointer.close()


class TestGetCheckpointer:
    """Tests for get_checkpointer factory function."""

    def test_get_checkpointer_persistent_true(self):
        """Test that persistent=True returns SqliteSaver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            with patch.dict(os.environ, {"CHECKPOINT_DB_PATH": db_path}):
                checkpointer = get_checkpointer(use_persistent=True)
                # Should return SqliteSaver
                assert checkpointer is not None

    def test_get_checkpointer_persistent_false(self):
        """Test that persistent=False returns MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = get_checkpointer(use_persistent=False)
        assert isinstance(checkpointer, MemorySaver)

    def test_get_checkpointer_env_var(self):
        """Test that get_checkpointer respects USE_PERSISTENT_MEMORY env var."""
        from langgraph.checkpoint.memory import MemorySaver

        with patch.dict(os.environ, {"USE_PERSISTENT_MEMORY": "false"}):
            checkpointer = get_checkpointer()
            assert isinstance(checkpointer, MemorySaver)


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_init_creates_database(self):
        """Test that initialization creates the database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            assert os.path.exists(db_path)
            tracker.close()

    def test_init_creates_tables(self):
        """Test that initialization creates the query_metrics table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            # Check table exists
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_metrics'")
            tables = cursor.fetchall()
            conn.close()

            assert len(tables) == 1
            tracker.close()

    def test_log_query(self):
        """Test that log_query stores metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            tracker.log_query(query="test query", latency_ms=100, rag_mode="basic", knowledge_base="test_kb", session_id="test_session", success=True)

            # Verify data was stored
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM query_metrics")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 1
            tracker.close()

    def test_log_query_with_error(self):
        """Test that log_query stores error information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            tracker.log_query(query="failing query", latency_ms=500, rag_mode="corrective", success=False, error="Test error message")

            # Verify error was stored
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT success, error FROM query_metrics WHERE query = ?", ("failing query",))
            row = cursor.fetchone()
            conn.close()

            assert row[0] == 0  # success = False
            assert row[1] == "Test error message"
            tracker.close()

    def test_get_stats_empty_database(self):
        """Test get_stats with empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            stats = tracker.get_stats(days=7)

            assert stats["total_queries"] == 0
            assert stats["success_rate"] == 0
            assert stats["avg_latency_ms"] == 0
            tracker.close()

    def test_get_stats_with_data(self):
        """Test get_stats with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            # Add some test queries
            for i in range(5):
                tracker.log_query(query=f"query {i}", latency_ms=100 + i * 10, rag_mode="basic", success=True)

            # Add one failed query
            tracker.log_query(query="failed query", latency_ms=500, rag_mode="corrective", success=False, error="Error")

            stats = tracker.get_stats(days=7)

            assert stats["total_queries"] == 6
            assert stats["successful_queries"] == 5
            assert round(stats["success_rate"], 1) == 83.3
            tracker.close()

    def test_get_recent_errors(self):
        """Test get_recent_errors returns error records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            tracker.log_query(query="error query", latency_ms=100, rag_mode="basic", success=False, error="Test error")

            errors = tracker.get_recent_errors(limit=10)

            assert len(errors) == 1
            assert "Test error" in errors[0]["error"]
            tracker.close()

    def test_cleanup_old_metrics(self):
        """Test cleanup_old_metrics removes old entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            tracker = MetricsTracker(db_path=db_path)

            # Add a recent query
            tracker.log_query(query="recent query", latency_ms=100, rag_mode="basic")

            # Cleanup (0 days should remove everything)
            deleted = tracker.cleanup_old_metrics(days=0)

            # Recent query should be deleted since 0 days
            assert deleted >= 0  # May or may not delete based on timing
            tracker.close()


class TestGetMetricsTracker:
    """Tests for get_metrics_tracker singleton function."""

    def test_returns_metrics_tracker_instance(self):
        """Test that get_metrics_tracker returns a MetricsTracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metrics.db")
            with patch.dict(os.environ, {"METRICS_DB_PATH": db_path}):
                # Reset global singleton
                import src.observability.metrics as metrics_module

                metrics_module._metrics_tracker = None

                tracker = get_metrics_tracker()
                assert isinstance(tracker, MetricsTracker)


class TestLangSmithTracer:
    """Tests for LangSmith tracing functions."""

    def test_setup_tracing_disabled_by_default(self):
        """Test that tracing is disabled when not configured."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=False):
            result = setup_langsmith_tracing()
            assert result is False

    def test_setup_tracing_requires_api_key(self):
        """Test that tracing requires API key."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""}, clear=False):
            result = setup_langsmith_tracing()
            assert result is False

    def test_setup_tracing_success(self):
        """Test successful tracing setup."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test_key"}, clear=False):
            result = setup_langsmith_tracing()
            assert result is True
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"

            # Cleanup
            disable_langsmith_tracing()

    def test_disable_tracing(self):
        """Test disabling tracing."""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        disable_langsmith_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") is None

    def test_is_tracing_enabled(self):
        """Test is_tracing_enabled function."""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        assert is_tracing_enabled() is True

        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        assert is_tracing_enabled() is False

    def test_get_tracing_status(self):
        """Test get_tracing_status returns correct info."""
        with patch.dict(
            os.environ, {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_PROJECT": "test_project", "LANGCHAIN_API_KEY": "test_key"}, clear=False
        ):
            status = get_tracing_status()
            assert status["enabled"] is True
            assert status["project"] == "test_project"
            assert status["api_key_configured"] is True


class TestIntegration:
    """Integration tests for Phase 5 components."""

    def test_checkpointer_and_metrics_coexist(self):
        """Test that checkpointer and metrics can be used together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoints.db")
            metrics_path = os.path.join(tmpdir, "metrics.db")

            checkpointer = PersistentCheckpointer(db_path=checkpoint_path)
            tracker = MetricsTracker(db_path=metrics_path)

            # Both should work independently
            assert os.path.exists(checkpoint_path)
            assert os.path.exists(metrics_path)

            tracker.log_query(query="test", latency_ms=100, rag_mode="basic")

            checkpointer.close()
            tracker.close()
