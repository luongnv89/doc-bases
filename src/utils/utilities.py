import os
import re
import subprocess

from src.utils.logger import get_logger

# Setup logging
logger = get_logger()


def get_version_from_git():
    """Retrieves the version number from the latest Git commit."""
    try:
        # Get the latest commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return f"Version: {commit_hash}"
    except Exception as e:
        logger.error(f"Failed to retrieve Git version: {e}")
        # Fallback to a hardcoded version or read from a VERSION file
        try:
            with open("VERSION") as f:
                return f"Version: {f.read().strip()}"
        except FileNotFoundError:
            return "Version: 0.1.0 (fallback)"


def generate_knowledge_base_name(source_type: int, input_str: str) -> str:
    """Generates a knowledge base name from user input."""
    if source_type == 1:  # Repo
        repo_name = input_str.split("/")[-1].replace(".git", "")
        return repo_name
    elif source_type == 2:  # Local File
        return os.path.splitext(os.path.basename(input_str))[0]
    elif source_type == 3:  # Local Folder
        return os.path.basename(input_str)
    elif source_type == 4:  # Website url
        return input_str.replace("https://", "").replace("http://", "").replace("/", "_")
    elif source_type == 5:  # Download file url
        return os.path.basename(input_str)
    else:
        return "default_knowledge_base"


def normalize_file_path(path: str) -> str:
    """
    Normalize a file path by unescaping shell escape sequences
    and expanding to absolute path.

    Args:
        path: The raw file path from user input

    Returns:
        str: Normalized absolute path
    """
    # Strip surrounding quotes
    path = path.strip("'\"")

    # Unescape common shell escape sequences
    # Handle escaped spaces, parentheses, brackets, etc.
    escape_sequences = [
        ("\\ ", " "),
        ("\\(", "("),
        ("\\)", ")"),
        ("\\[", "["),
        ("\\]", "]"),
        ("\\{", "{"),
        ("\\}", "}"),
        ("\\&", "&"),
        ("\\$", "$"),
        ("\\!", "!"),
        ("\\@", "@"),
        ("\\#", "#"),
        ("\\'", "'"),
        ('\\"', '"'),
    ]

    for escaped, unescaped in escape_sequences:
        path = path.replace(escaped, unescaped)

    # Expand ~ to user home directory
    path = os.path.expanduser(path)

    # Resolve to absolute path
    path = os.path.abspath(path)

    return path


def validate_file_paths(query: str) -> tuple[bool, list[str]]:
    """
    Detects file paths in a query and validates them for image files.

    Args:
        query: The user query string to validate

    Returns:
        tuple: (is_valid, list of invalid image file paths)
    """
    detected_files = set()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"}

    # Check quoted paths first (most reliable)
    quoted_pattern = r'["\']([^"\']*\.(?:png|jpg|jpeg|gif|bmp|webp|tiff|svg))["\']'
    matches = re.findall(quoted_pattern, query, re.IGNORECASE)
    detected_files.update(matches)

    # More robust approach: look for any image extension and extract the full path
    # This handles escaped spaces, special characters, and complex paths
    image_ext_pattern = r"(?:png|jpg|jpeg|gif|bmp|webp|tiff|svg)"

    # Find all image extensions in the query
    for match in re.finditer(image_ext_pattern, query, re.IGNORECASE):
        ext_start = match.start()
        ext_end = match.end()

        # Look backwards from the extension to find the path start
        path_start = -1
        for i in range(ext_start - 1, -1, -1):
            char = query[i]
            if char == "/":
                path_start = i
                break
            # Stop at quote or beginning of string
            if char in ["'", '"'] or i == 0:
                break

        if path_start != -1:
            # Extract the full path (handle escaped characters)
            path_end = ext_end
            # Look ahead to find where the path really ends
            for i in range(ext_end, len(query)):
                char = query[i]
                if char.isspace() or char in ["?", "!", ".", ",", ";", ":", "'", '"']:
                    path_end = i
                    break

            file_path = query[path_start:path_end]
            # Unescape common escape sequences for cleaner display
            file_path = file_path.replace("\\ ", " ").replace("\\(", "(").replace("\\)", ")")
            detected_files.add(file_path)

    # Filter for image files and remove duplicates
    invalid_image_files = []
    for file_path in detected_files:
        file_path = file_path.strip()
        _, ext = os.path.splitext(file_path.lower())
        if ext in image_extensions:
            invalid_image_files.append(file_path)

    is_valid = len(invalid_image_files) == 0
    return is_valid, invalid_image_files


def is_image_file(file_path: str) -> bool:
    """
    Checks if a file path points to an image file.

    Args:
        file_path: The file path to check

    Returns:
        bool: True if the file is an image, False otherwise
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions


def format_image_error_message(invalid_files: list[str]) -> str:
    """
    Formats an error message for detected image files.

    Args:
        invalid_files: List of invalid image file paths

    Returns:
        str: Formatted error message
    """
    if not invalid_files:
        return ""

    message = "❌ [error]Image files are not supported by this model.[/error]\n\n"

    if len(invalid_files) == 1:
        message += f"Detected image file: [warning]{invalid_files[0]}[/warning]\n\n"
    else:
        message += "Detected image files:\n"
        for file_path in invalid_files:
            message += f"  • [warning]{file_path}[/warning]\n"
        message += "\n"

    message += "💡 [info]Tip: Remove the image path(s) and ask your question using text only.[/info]"
    return message
