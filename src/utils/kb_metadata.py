"""
Knowledge Base Metadata Management

This module provides functionality for tracking and detecting changes
in source files for knowledge bases. It stores metadata about indexed
files and can detect modifications since the last sync.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger()

KNOWLEDGE_BASE_DIR = "knowledges"
METADATA_FILENAME = "metadata.json"
METADATA_VERSION = "1.0"

# File extensions to track (follows DocumentLoader patterns)
TRACKABLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".sql",
    ".r",
    ".scala",
    ".swift",
    ".kt",
    ".kts",
}


@dataclass
class FileInfo:
    """Information about an indexed file."""

    path: str  # Relative path from source directory
    absolute_path: str
    mtime: float  # Modification time
    size: int  # File size in bytes


@dataclass
class ChangeReport:
    """Report of detected changes in source files."""

    has_changes: bool
    added_files: list[str]
    modified_files: list[str]
    deleted_files: list[str]
    source_path: str | None
    source_type: str
    error: str | None = None


class KBMetadataManager:
    """Manages metadata for a knowledge base."""

    def __init__(self, kb_name: str):
        self.kb_name = kb_name
        self.kb_path = Path(KNOWLEDGE_BASE_DIR) / kb_name
        self.metadata_path = self.kb_path / METADATA_FILENAME
        self._metadata: dict[str, Any] | None = None

    def metadata_exists(self) -> bool:
        """Check if metadata file exists."""
        return self.metadata_path.exists()

    def load_metadata(self) -> dict[str, Any] | None:
        """Load metadata from file."""
        if not self.metadata_exists():
            logger.debug(f"No metadata found for KB: {self.kb_name}")
            return None

        try:
            with open(self.metadata_path) as f:
                self._metadata = json.load(f)
                return self._metadata
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load metadata for KB {self.kb_name}: {e}")
            return None

    def save_metadata(
        self,
        source_type: str,
        source_path: str,
        indexed_files: list[FileInfo],
    ) -> bool:
        """Save metadata to file."""
        now = datetime.now().isoformat()

        metadata = {
            "version": METADATA_VERSION,
            "kb_name": self.kb_name,
            "source_type": source_type,
            "source_path": source_path,
            "created_at": now,
            "last_sync_at": now,
            "indexed_files": [asdict(f) for f in indexed_files],
        }

        try:
            # Ensure KB directory exists
            self.kb_path.mkdir(parents=True, exist_ok=True)

            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved metadata for KB {self.kb_name}: {len(indexed_files)} files")
            self._metadata = metadata
            return True
        except OSError as e:
            logger.error(f"Failed to save metadata for KB {self.kb_name}: {e}")
            return False

    def get_source_info(self) -> tuple[str | None, str | None]:
        """Get source type and path from metadata."""
        metadata = self._metadata or self.load_metadata()
        if not metadata:
            return None, None
        return metadata.get("source_type"), metadata.get("source_path")


def is_trackable_file(file_path: Path) -> bool:
    """Check if a file should be tracked for changes."""
    # Skip hidden files and directories
    if any(part.startswith(".") for part in file_path.parts):
        return False

    # Skip common non-document directories
    skip_dirs = {"__pycache__", "node_modules", ".git", ".svn", "venv", ".venv", "env"}
    if any(part in skip_dirs for part in file_path.parts):
        return False

    # Check extension
    return file_path.suffix.lower() in TRACKABLE_EXTENSIONS


def collect_file_info(folder_path: str) -> list[FileInfo]:
    """
    Walk a directory and collect file information.

    Args:
        folder_path: Path to the folder to scan

    Returns:
        List of FileInfo objects for trackable files
    """
    folder = Path(folder_path).resolve()
    files: list[FileInfo] = []

    if not folder.exists():
        logger.warning(f"Source folder does not exist: {folder_path}")
        return files

    if not folder.is_dir():
        logger.warning(f"Source path is not a directory: {folder_path}")
        return files

    try:
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue

            if not is_trackable_file(file_path):
                continue

            try:
                stat = file_path.stat()
                relative_path = str(file_path.relative_to(folder))

                files.append(
                    FileInfo(
                        path=relative_path,
                        absolute_path=str(file_path),
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                    )
                )
            except OSError as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
                continue

    except PermissionError as e:
        logger.warning(f"Permission denied accessing folder {folder_path}: {e}")

    return files


def collect_single_file_info(file_path: str) -> list[FileInfo]:
    """Collect file info for a single file."""
    path = Path(file_path).resolve()

    if not path.exists():
        logger.warning(f"Source file does not exist: {file_path}")
        return []

    try:
        stat = path.stat()
        return [
            FileInfo(
                path=path.name,
                absolute_path=str(path),
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        ]
    except OSError as e:
        logger.warning(f"Failed to stat file {file_path}: {e}")
        return []


def detect_file_changes(kb_name: str) -> ChangeReport:
    """
    Detect changes in source files since last sync.

    Args:
        kb_name: Name of the knowledge base

    Returns:
        ChangeReport with details of any changes
    """
    manager = KBMetadataManager(kb_name)
    metadata = manager.load_metadata()

    # No metadata - old KB without tracking
    if not metadata:
        return ChangeReport(
            has_changes=False,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path=None,
            source_type="unknown",
            error="no_metadata",
        )

    source_type = metadata.get("source_type", "")
    source_path = metadata.get("source_path", "")
    indexed_files = metadata.get("indexed_files", [])

    # For repo/website/url sources, limited change detection
    if source_type in ("repo", "website", "url"):
        return ChangeReport(
            has_changes=False,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path=source_path,
            source_type=source_type,
            error="remote_source",
        )

    # Check if source path exists
    source = Path(source_path)
    if not source.exists():
        return ChangeReport(
            has_changes=True,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path=source_path,
            source_type=source_type,
            error="source_not_found",
        )

    # Collect current files
    if source_type == "folder":
        current_files = collect_file_info(source_path)
    elif source_type == "file":
        current_files = collect_single_file_info(source_path)
    else:
        return ChangeReport(
            has_changes=False,
            added_files=[],
            modified_files=[],
            deleted_files=[],
            source_path=source_path,
            source_type=source_type,
            error="unknown_source_type",
        )

    # Build lookup maps
    indexed_map = {f["path"]: f for f in indexed_files}
    current_map = {f.path: f for f in current_files}

    # Detect changes
    added = []
    modified = []
    deleted = []

    # Check for added and modified files
    for path, info in current_map.items():
        if path not in indexed_map:
            added.append(path)
        else:
            indexed_info = indexed_map[path]
            # Check if mtime changed
            if abs(info.mtime - indexed_info.get("mtime", 0)) > 1:  # 1 second tolerance
                modified.append(path)

    # Check for deleted files
    for path in indexed_map:
        if path not in current_map:
            deleted.append(path)

    has_changes = bool(added or modified or deleted)

    return ChangeReport(
        has_changes=has_changes,
        added_files=added,
        modified_files=modified,
        deleted_files=deleted,
        source_path=source_path,
        source_type=source_type,
    )


def format_change_report(report: ChangeReport, kb_name: str, max_files: int = 10) -> str:
    """
    Format a change report for CLI display.

    Args:
        report: The ChangeReport to format
        kb_name: Name of the knowledge base
        max_files: Maximum number of files to show per category

    Returns:
        Formatted string for display
    """
    lines = []

    # Handle special cases
    if report.error == "no_metadata":
        return ""  # Will be handled separately

    if report.error == "source_not_found":
        lines.append(f"[yellow]Warning:[/yellow] Source path no longer exists: {report.source_path}")
        lines.append("You may want to re-index the knowledge base with a new source.")
        return "\n".join(lines)

    if report.error == "remote_source":
        return ""  # Remote sources don't support change detection

    if not report.has_changes:
        return ""

    lines.append("[yellow]Source files have changed since last sync:[/yellow]")
    lines.append("")

    def format_file_list(files: list[str], prefix: str, style: str) -> None:
        if not files:
            return
        lines.append(f"  {prefix} ({len(files)} file{'s' if len(files) != 1 else ''}):")
        for f in files[:max_files]:
            lines.append(f"    [{style}]• {f}[/{style}]")
        if len(files) > max_files:
            lines.append(f"    [dim]... and {len(files) - max_files} more[/dim]")

    format_file_list(report.added_files, "Added", "green")
    format_file_list(report.modified_files, "Modified", "yellow")
    format_file_list(report.deleted_files, "Deleted", "red")

    return "\n".join(lines)


def get_reindex_command(kb_name: str, source_type: str, source_path: str) -> str:
    """Generate the re-index command for a KB."""
    return f"docb kb add {source_type} {source_path} --name {kb_name} --overwrite"
