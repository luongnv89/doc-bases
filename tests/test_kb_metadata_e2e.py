"""End-to-end tests for KB file change detection feature.

These tests verify the metadata tracking and change detection functionality
without running the full RAG embedding process (which is slow and requires
API keys). The tests mock the setup_rag function to focus on the metadata
feature itself.
"""

import json
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli.main import app
from src.utils.kb_metadata import METADATA_FILENAME, KBMetadataManager, collect_file_info, detect_file_changes, format_change_report

runner = CliRunner()

# Test KB name and paths
TEST_KB_NAME = "e2e-test-kb"
KNOWLEDGES_DIR = Path("knowledges")


def cleanup_test_kb():
    """Clean up test knowledge base if it exists."""
    kb_path = KNOWLEDGES_DIR / TEST_KB_NAME
    if kb_path.exists():
        shutil.rmtree(kb_path)


def cleanup_test_docs(test_docs_dir: Path):
    """Clean up test documents directory."""
    if test_docs_dir.exists():
        shutil.rmtree(test_docs_dir)


def mock_setup_rag(*args, **kwargs):
    """Mock setup_rag to avoid expensive embedding operations."""
    kb_name = args[1] if len(args) > 1 else kwargs.get("knowledge_base_name", TEST_KB_NAME)
    # Create the KB directory structure as setup_rag would
    kb_path = KNOWLEDGES_DIR / kb_name
    kb_path.mkdir(parents=True, exist_ok=True)
    (kb_path / "vector_store").mkdir(exist_ok=True)
    return MagicMock()


class TestKBMetadataE2E:
    """End-to-end tests for KB metadata and change detection."""

    def setup_method(self):
        """Set up test fixtures."""
        cleanup_test_kb()
        self.test_docs_dir = Path("test_docs_e2e")
        cleanup_test_docs(self.test_docs_dir)

        # Create test documents directory
        self.test_docs_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after tests."""
        cleanup_test_kb()
        cleanup_test_docs(self.test_docs_dir)

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_kb_creation_saves_metadata(self, mock_rag):
        """Test that creating a KB saves metadata.json with file info."""
        # Create test documents
        (self.test_docs_dir / "readme.md").write_text("# Test Documentation\n\nThis is a test.")
        (self.test_docs_dir / "guide.txt").write_text("User guide content here.")

        # Create KB using CLI
        result = runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Verify KB was created successfully
        assert result.exit_code == 0, f"KB creation failed: {result.output}"
        assert "created successfully" in result.output

        # Verify metadata file exists
        metadata_path = KNOWLEDGES_DIR / TEST_KB_NAME / METADATA_FILENAME
        assert metadata_path.exists(), "Metadata file was not created"

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["kb_name"] == TEST_KB_NAME
        assert metadata["source_type"] == "folder"
        assert str(self.test_docs_dir.resolve()) in metadata["source_path"]
        assert len(metadata["indexed_files"]) == 2

        # Verify file info
        file_paths = [f["path"] for f in metadata["indexed_files"]]
        assert "readme.md" in file_paths
        assert "guide.txt" in file_paths

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_kb_info_shows_metadata(self, mock_rag):
        """Test that kb info command displays metadata."""
        # Create test document
        (self.test_docs_dir / "doc.md").write_text("# Document")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Run kb info
        result = runner.invoke(app, ["kb", "info", TEST_KB_NAME])

        assert result.exit_code == 0
        assert "Source Type" in result.output
        assert "folder" in result.output
        assert "Source Path" in result.output
        assert "Last Sync" in result.output
        assert "Indexed Files" in result.output

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_detect_no_changes(self, mock_rag):
        """Test that no changes are detected when files haven't changed."""
        # Create test document
        (self.test_docs_dir / "stable.md").write_text("# Stable Content")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert not report.has_changes
        assert len(report.added_files) == 0
        assert len(report.modified_files) == 0
        assert len(report.deleted_files) == 0

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_detect_added_file(self, mock_rag):
        """Test detection of newly added files."""
        # Create initial document
        (self.test_docs_dir / "original.md").write_text("# Original")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Add new file
        (self.test_docs_dir / "new_file.md").write_text("# New File Added")

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert "new_file.md" in report.added_files
        assert len(report.modified_files) == 0
        assert len(report.deleted_files) == 0

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_detect_modified_file(self, mock_rag):
        """Test detection of modified files."""
        # Create initial document
        test_file = self.test_docs_dir / "modifiable.md"
        test_file.write_text("# Original Content")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Wait a moment to ensure mtime difference
        time.sleep(1.1)

        # Modify file
        test_file.write_text("# Modified Content - Changed!")

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert "modifiable.md" in report.modified_files
        assert len(report.added_files) == 0
        assert len(report.deleted_files) == 0

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_detect_deleted_file(self, mock_rag):
        """Test detection of deleted files."""
        # Create initial documents
        (self.test_docs_dir / "keep.md").write_text("# Keep this")
        (self.test_docs_dir / "delete_me.md").write_text("# Delete this")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Delete one file
        (self.test_docs_dir / "delete_me.md").unlink()

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert "delete_me.md" in report.deleted_files
        assert len(report.added_files) == 0
        assert len(report.modified_files) == 0

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_detect_multiple_changes(self, mock_rag):
        """Test detection of multiple types of changes simultaneously."""
        # Create initial documents
        (self.test_docs_dir / "unchanged.md").write_text("# Unchanged")
        (self.test_docs_dir / "to_modify.md").write_text("# Will be modified")
        (self.test_docs_dir / "to_delete.md").write_text("# Will be deleted")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Wait for mtime difference
        time.sleep(1.1)

        # Make changes
        (self.test_docs_dir / "new_addition.md").write_text("# Newly added")
        (self.test_docs_dir / "to_modify.md").write_text("# Modified content now")
        (self.test_docs_dir / "to_delete.md").unlink()

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert "new_addition.md" in report.added_files
        assert "to_modify.md" in report.modified_files
        assert "to_delete.md" in report.deleted_files

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_nested_directory_tracking(self, mock_rag):
        """Test that files in nested directories are tracked."""
        # Create nested structure
        subdir = self.test_docs_dir / "subdir"
        subdir.mkdir()
        (self.test_docs_dir / "root.md").write_text("# Root level")
        (subdir / "nested.md").write_text("# Nested level")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Verify nested file is tracked
        manager = KBMetadataManager(TEST_KB_NAME)
        metadata = manager.load_metadata()
        assert metadata is not None, "Metadata should exist"

        file_paths = [f["path"] for f in metadata["indexed_files"]]

        assert "root.md" in file_paths
        # Check for nested file (path separator may vary)
        nested_found = any("nested.md" in p for p in file_paths)
        assert nested_found, f"Nested file not found in {file_paths}"

        # Add file to nested directory
        (subdir / "new_nested.md").write_text("# New nested file")

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert any("new_nested.md" in f for f in report.added_files)

    def test_old_kb_without_metadata(self):
        """Test behavior with KB that has no metadata (older KB)."""
        # Create KB directory manually without metadata
        kb_path = KNOWLEDGES_DIR / TEST_KB_NAME
        kb_path.mkdir(parents=True, exist_ok=True)

        # Detect changes - should report no_metadata error
        report = detect_file_changes(TEST_KB_NAME)

        assert not report.has_changes
        assert report.error == "no_metadata"

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_source_path_not_found(self, mock_rag):
        """Test behavior when source path no longer exists."""
        # Create test document
        (self.test_docs_dir / "temp.md").write_text("# Temporary")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Delete source directory
        shutil.rmtree(self.test_docs_dir)

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert report.error == "source_not_found"

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_single_file_kb(self, mock_rag):
        """Test KB created from a single file."""
        # Create single test file
        test_file = self.test_docs_dir / "single_doc.md"
        test_file.write_text("# Single Document KB")

        # Create KB from single file
        result = runner.invoke(
            app,
            ["kb", "add", "file", str(test_file), "--name", TEST_KB_NAME],
        )

        assert result.exit_code == 0, f"KB creation failed: {result.output}"

        # Verify metadata
        manager = KBMetadataManager(TEST_KB_NAME)
        metadata = manager.load_metadata()

        assert metadata["source_type"] == "file"
        assert len(metadata["indexed_files"]) == 1
        assert metadata["indexed_files"][0]["path"] == "single_doc.md"

        # Modify file
        time.sleep(1.1)
        test_file.write_text("# Modified Single Document")

        # Detect changes
        report = detect_file_changes(TEST_KB_NAME)

        assert report.has_changes
        assert "single_doc.md" in report.modified_files

    @patch("src.cli.commands.kb.setup_rag", side_effect=mock_setup_rag)
    def test_kb_overwrite_updates_metadata(self, mock_rag):
        """Test that overwriting a KB updates the metadata."""
        # Create initial document
        (self.test_docs_dir / "v1.md").write_text("# Version 1")

        # Create KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME],
        )

        # Check initial metadata
        manager = KBMetadataManager(TEST_KB_NAME)
        metadata1 = manager.load_metadata()
        assert metadata1 is not None
        initial_sync = metadata1["last_sync_at"]

        # Wait and add new file
        time.sleep(1.1)
        (self.test_docs_dir / "v2.md").write_text("# Version 2")

        # Overwrite KB
        runner.invoke(
            app,
            ["kb", "add", "folder", str(self.test_docs_dir), "--name", TEST_KB_NAME, "--overwrite"],
        )

        # Check updated metadata
        metadata2 = manager.load_metadata()
        assert metadata2 is not None

        assert metadata2["last_sync_at"] != initial_sync
        assert len(metadata2["indexed_files"]) == 2
        file_paths = [f["path"] for f in metadata2["indexed_files"]]
        assert "v1.md" in file_paths
        assert "v2.md" in file_paths


class TestChangeDetectionFormatting:
    """Tests for change report formatting."""

    def test_format_change_report_with_all_types(self):
        """Test formatting a report with all change types."""
        from src.utils.kb_metadata import ChangeReport

        report = ChangeReport(
            has_changes=True,
            added_files=["new1.md", "new2.md"],
            modified_files=["modified.md"],
            deleted_files=["old.md"],
            source_path="/path/to/docs",
            source_type="folder",
        )

        formatted = format_change_report(report, "test-kb")

        assert "Source files have changed" in formatted
        assert "Added" in formatted
        assert "new1.md" in formatted
        assert "Modified" in formatted
        assert "modified.md" in formatted
        assert "Deleted" in formatted
        assert "old.md" in formatted

    def test_format_truncates_long_lists(self):
        """Test that long file lists are truncated."""
        from src.utils.kb_metadata import ChangeReport

        report = ChangeReport(
            has_changes=True,
            added_files=[f"file{i}.md" for i in range(20)],
            modified_files=[],
            deleted_files=[],
            source_path="/path",
            source_type="folder",
        )

        formatted = format_change_report(report, "test-kb", max_files=5)

        assert "and 15 more" in formatted
        # First 5 should be shown
        assert "file0.md" in formatted
        assert "file4.md" in formatted


class TestMetadataDirectAPI:
    """Direct API tests for metadata functionality without CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path("test_direct_api")
        self.test_dir.mkdir(exist_ok=True)
        self.kb_name = "direct-api-test"
        cleanup_path = KNOWLEDGES_DIR / self.kb_name
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path)

    def teardown_method(self):
        """Clean up after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        cleanup_path = KNOWLEDGES_DIR / self.kb_name
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path)

    def test_collect_and_save_metadata(self):
        """Test collecting file info and saving metadata directly."""
        # Create test files
        (self.test_dir / "doc1.md").write_text("# Doc 1")
        (self.test_dir / "doc2.txt").write_text("Doc 2 content")
        subdir = self.test_dir / "sub"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested")

        # Create KB directory
        kb_path = KNOWLEDGES_DIR / self.kb_name
        kb_path.mkdir(parents=True, exist_ok=True)

        # Collect file info
        files = collect_file_info(str(self.test_dir))
        assert len(files) == 3

        # Save metadata
        manager = KBMetadataManager(self.kb_name)
        success = manager.save_metadata("folder", str(self.test_dir.resolve()), files)
        assert success

        # Load and verify
        metadata = manager.load_metadata()
        assert metadata is not None
        assert metadata["source_type"] == "folder"
        assert len(metadata["indexed_files"]) == 3

    def test_change_detection_workflow(self):
        """Test complete change detection workflow."""
        # Create initial files
        (self.test_dir / "initial.md").write_text("# Initial")
        (self.test_dir / "to_delete.md").write_text("# To Delete")

        # Create KB directory and metadata
        kb_path = KNOWLEDGES_DIR / self.kb_name
        kb_path.mkdir(parents=True, exist_ok=True)

        files = collect_file_info(str(self.test_dir))
        manager = KBMetadataManager(self.kb_name)
        manager.save_metadata("folder", str(self.test_dir.resolve()), files)

        # Verify no changes initially
        report = detect_file_changes(self.kb_name)
        assert not report.has_changes

        # Add a file
        (self.test_dir / "added.md").write_text("# Added")
        report = detect_file_changes(self.kb_name)
        assert report.has_changes
        assert "added.md" in report.added_files

        # Modify a file
        time.sleep(1.1)
        (self.test_dir / "initial.md").write_text("# Initial - modified")
        report = detect_file_changes(self.kb_name)
        assert report.has_changes
        assert "initial.md" in report.modified_files

        # Delete an originally indexed file
        (self.test_dir / "to_delete.md").unlink()
        report = detect_file_changes(self.kb_name)
        assert report.has_changes
        assert "to_delete.md" in report.deleted_files
