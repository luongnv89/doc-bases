"""Tests for the KB metadata module."""

from pathlib import Path

from src.utils.kb_metadata import (
    METADATA_VERSION,
    ChangeReport,
    FileInfo,
    KBMetadataManager,
    collect_file_info,
    collect_single_file_info,
    detect_file_changes,
    format_change_report,
    get_reindex_command,
    is_trackable_file,
)


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        """Test creating a FileInfo instance."""
        info = FileInfo(
            path="test.md",
            absolute_path="/path/to/test.md",
            mtime=1705834800.0,
            size=1234,
        )
        assert info.path == "test.md"
        assert info.absolute_path == "/path/to/test.md"
        assert info.mtime == 1705834800.0
        assert info.size == 1234


class TestChangeReport:
    """Tests for ChangeReport dataclass."""

    def test_change_report_no_changes(self):
        """Test creating a ChangeReport with no changes."""
        report = ChangeReport(
            has_changes=False,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path="/path/to/source",
            source_type="folder",
        )
        assert not report.has_changes
        assert report.error is None

    def test_change_report_with_changes(self):
        """Test creating a ChangeReport with changes."""
        report = ChangeReport(
            has_changes=True,
            added_files=["new.md"],
            modified_files=["modified.md"],
            deleted_files=["deleted.md"],
            source_path="/path/to/source",
            source_type="folder",
        )
        assert report.has_changes
        assert "new.md" in report.added_files
        assert "modified.md" in report.modified_files
        assert "deleted.md" in report.deleted_files


class TestIsTrackableFile:
    """Tests for is_trackable_file function."""

    def test_trackable_markdown(self):
        """Test that markdown files are trackable."""
        assert is_trackable_file(Path("docs/readme.md"))

    def test_trackable_python(self):
        """Test that Python files are trackable."""
        assert is_trackable_file(Path("src/main.py"))

    def test_trackable_pdf(self):
        """Test that PDF files are trackable."""
        assert is_trackable_file(Path("docs/manual.pdf"))

    def test_non_trackable_image(self):
        """Test that image files are not trackable."""
        assert not is_trackable_file(Path("images/logo.png"))

    def test_hidden_files(self):
        """Test that hidden files are not trackable."""
        assert not is_trackable_file(Path(".gitignore"))
        assert not is_trackable_file(Path(".hidden/file.md"))

    def test_pycache(self):
        """Test that __pycache__ files are not trackable."""
        assert not is_trackable_file(Path("__pycache__/module.pyc"))

    def test_node_modules(self):
        """Test that node_modules files are not trackable."""
        assert not is_trackable_file(Path("node_modules/pkg/index.js"))


class TestCollectFileInfo:
    """Tests for collect_file_info function."""

    def test_collect_from_folder(self, tmp_path):
        """Test collecting file info from a folder."""
        # Create test files
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "doc.txt").write_text("Document")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.md").write_text("Nested")

        files = collect_file_info(str(tmp_path))

        assert len(files) == 3
        paths = [f.path for f in files]
        assert "readme.md" in paths
        assert "doc.txt" in paths
        assert "subdir/nested.md" in paths or "subdir\\nested.md" in paths

    def test_collect_excludes_hidden(self, tmp_path):
        """Test that hidden files are excluded."""
        (tmp_path / ".hidden").write_text("hidden")
        (tmp_path / "visible.md").write_text("visible")

        files = collect_file_info(str(tmp_path))

        assert len(files) == 1
        assert files[0].path == "visible.md"

    def test_collect_nonexistent_folder(self):
        """Test collecting from nonexistent folder."""
        files = collect_file_info("/nonexistent/path")
        assert files == []


class TestCollectSingleFileInfo:
    """Tests for collect_single_file_info function."""

    def test_collect_single_file(self, tmp_path):
        """Test collecting info for a single file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("Test content")

        files = collect_single_file_info(str(test_file))

        assert len(files) == 1
        assert files[0].path == "test.md"
        assert files[0].size == len("Test content")

    def test_collect_nonexistent_file(self):
        """Test collecting info for nonexistent file."""
        files = collect_single_file_info("/nonexistent/file.md")
        assert files == []


class TestKBMetadataManager:
    """Tests for KBMetadataManager class."""

    def test_metadata_exists_false(self, tmp_path, monkeypatch):
        """Test metadata_exists returns False when no metadata."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        manager = KBMetadataManager("test-kb")
        assert not manager.metadata_exists()

    def test_save_and_load_metadata(self, tmp_path, monkeypatch):
        """Test saving and loading metadata."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        manager = KBMetadataManager("test-kb")
        files = [
            FileInfo(path="test.md", absolute_path="/path/test.md", mtime=1234.0, size=100),
        ]

        # Save
        result = manager.save_metadata("folder", "/path/to/source", files)
        assert result is True

        # Load
        metadata = manager.load_metadata()
        assert metadata is not None
        assert metadata["version"] == METADATA_VERSION
        assert metadata["kb_name"] == "test-kb"
        assert metadata["source_type"] == "folder"
        assert metadata["source_path"] == "/path/to/source"
        assert len(metadata["indexed_files"]) == 1

    def test_get_source_info(self, tmp_path, monkeypatch):
        """Test getting source info from metadata."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        manager = KBMetadataManager("test-kb")
        manager.save_metadata("folder", "/path/to/source", [])

        source_type, source_path = manager.get_source_info()
        assert source_type == "folder"
        assert source_path == "/path/to/source"


class TestDetectFileChanges:
    """Tests for detect_file_changes function."""

    def test_no_metadata(self, tmp_path, monkeypatch):
        """Test detection when no metadata exists."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        report = detect_file_changes("test-kb")

        assert not report.has_changes
        assert report.error == "no_metadata"

    def test_remote_source(self, tmp_path, monkeypatch):
        """Test detection for remote sources."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        manager = KBMetadataManager("test-kb")
        manager.save_metadata("repo", "https://github.com/user/repo", [])

        report = detect_file_changes("test-kb")

        assert not report.has_changes
        assert report.error == "remote_source"

    def test_source_not_found(self, tmp_path, monkeypatch):
        """Test detection when source path no longer exists."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))
        (tmp_path / "test-kb").mkdir()

        manager = KBMetadataManager("test-kb")
        manager.save_metadata("folder", "/nonexistent/path", [])

        report = detect_file_changes("test-kb")

        assert report.has_changes
        assert report.error == "source_not_found"

    def test_detect_added_file(self, tmp_path, monkeypatch):
        """Test detecting an added file."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))

        # Create KB directory and source directory
        (tmp_path / "test-kb").mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create initial file
        (source_dir / "original.md").write_text("Original")

        # Save metadata
        manager = KBMetadataManager("test-kb")
        files = collect_file_info(str(source_dir))
        manager.save_metadata("folder", str(source_dir), files)

        # Add new file
        (source_dir / "new.md").write_text("New file")

        # Detect changes
        report = detect_file_changes("test-kb")

        assert report.has_changes
        assert "new.md" in report.added_files

    def test_detect_deleted_file(self, tmp_path, monkeypatch):
        """Test detecting a deleted file."""
        monkeypatch.setattr("src.utils.kb_metadata.KNOWLEDGE_BASE_DIR", str(tmp_path))

        # Create KB directory and source directory
        (tmp_path / "test-kb").mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create initial files
        (source_dir / "keep.md").write_text("Keep")
        (source_dir / "delete.md").write_text("Delete")

        # Save metadata
        manager = KBMetadataManager("test-kb")
        files = collect_file_info(str(source_dir))
        manager.save_metadata("folder", str(source_dir), files)

        # Delete file
        (source_dir / "delete.md").unlink()

        # Detect changes
        report = detect_file_changes("test-kb")

        assert report.has_changes
        assert "delete.md" in report.deleted_files


class TestFormatChangeReport:
    """Tests for format_change_report function."""

    def test_format_no_metadata(self):
        """Test formatting when no metadata."""
        report = ChangeReport(
            has_changes=False,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path=None,
            source_type="unknown",
            error="no_metadata",
        )
        result = format_change_report(report, "test-kb")
        assert result == ""

    def test_format_with_changes(self):
        """Test formatting with changes."""
        report = ChangeReport(
            has_changes=True,
            added_files=["new.md"],
            modified_files=["modified.md"],
            deleted_files=["deleted.md"],
            source_path="/path",
            source_type="folder",
        )
        result = format_change_report(report, "test-kb")
        assert "Source files have changed" in result
        assert "Added" in result
        assert "new.md" in result

    def test_format_truncates_long_lists(self):
        """Test that long file lists are truncated."""
        report = ChangeReport(
            has_changes=True,
            added_files=[f"file{i}.md" for i in range(20)],
            modified_files=[],
            deleted_files=[],
            source_path="/path",
            source_type="folder",
        )
        result = format_change_report(report, "test-kb", max_files=5)
        assert "and 15 more" in result


class TestGetReindexCommand:
    """Tests for get_reindex_command function."""

    def test_folder_command(self):
        """Test command for folder source."""
        cmd = get_reindex_command("my-kb", "folder", "/path/to/docs")
        assert cmd == "docb kb add folder /path/to/docs --name my-kb --overwrite"

    def test_file_command(self):
        """Test command for file source."""
        cmd = get_reindex_command("my-kb", "file", "/path/to/doc.pdf")
        assert cmd == "docb kb add file /path/to/doc.pdf --name my-kb --overwrite"

    def test_repo_command(self):
        """Test command for repo source."""
        cmd = get_reindex_command("my-kb", "repo", "https://github.com/user/repo")
        assert cmd == "docb kb add repo https://github.com/user/repo --name my-kb --overwrite"
