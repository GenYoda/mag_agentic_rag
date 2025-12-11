"""
File Utilities for Agentic Medical RAG System

Provides low-level file operations:
1. File hashing (MD5 for file bytes, SHA256 for content)
2. JSON read/write with error handling
3. Directory utilities
4. Path validation and normalization
5. Text normalization for consistent hashing

These are foundation tools used by:
- core/deduplication.py
- tools/kb_tools.py
- core/semantic_cache.py
- core/enhanced_memory.py
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# ============================================================================
# SECTION 1: File Hashing
# ============================================================================

def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of file contents (byte-level).
    
    Used for detecting if a PDF file has been modified (byte-for-byte comparison).
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("md5" or "sha256")
        
    Returns:
        str: Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm not supported
        
    Example:
        >>> file_hash = calculate_file_hash("medical_record.pdf")
        >>> # Returns: "5d41402abc4b2a76b9719d911017c592"
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if algorithm.lower() == "md5":
        hasher = hashlib.md5()
    elif algorithm.lower() == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'md5' or 'sha256'")
    
    # Read file in chunks to handle large files
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def calculate_content_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Calculate hash of normalized text content.
    
    Used for content-based deduplication (detects if PDF content is identical
    even if filename or metadata differs).
    
    Text is normalized before hashing:
    - Converted to lowercase
    - Whitespace normalized
    - Special characters removed
    
    Args:
        text: Text content to hash
        algorithm: Hash algorithm ("sha256" recommended for content)
        
    Returns:
        str: Hexadecimal hash string
        
    Example:
        >>> text1 = "Patient diagnosed with diabetes."
        >>> text2 = "PATIENT  DIAGNOSED   WITH DIABETES."
        >>> calculate_content_hash(text1) == calculate_content_hash(text2)
        True  # Same content despite formatting differences
    """
    # Normalize text for consistent hashing
    normalized = normalize_text_for_hashing(text)
    
    if algorithm.lower() == "sha256":
        hasher = hashlib.sha256()
    elif algorithm.lower() == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hasher.update(normalized.encode('utf-8'))
    return hasher.hexdigest()


def normalize_text_for_hashing(text: str) -> str:
    """
    Normalize text for consistent content hashing.
    
    Normalization steps:
    1. Convert to lowercase
    2. Remove ALL punctuation (for true content similarity)
    3. Remove extra whitespace (multiple spaces → single space)
    4. Strip leading/trailing whitespace
    
    Args:
        text: Raw text
        
    Returns:
        str: Normalized text
        
    Example:
        >>> normalize_text_for_hashing("  Patient  NAME:  John Doe.  ")
        'patient name john doe'
    """
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove ALL punctuation and special characters (keep only alphanumeric and spaces)
    # This ensures "diabetes." and "diabetes" are treated as identical
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    # Replace multiple whitespace with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


# ============================================================================
# SECTION 2: JSON I/O
# ============================================================================

def read_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely read JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Value to return if file doesn't exist or is invalid (default: None)
        
    Returns:
        Parsed JSON data, or default value if error
        
    Example:
        >>> data = read_json("pdf_tracker.json", default={})
        >>> # Returns {} if file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Error reading JSON from {file_path}: {e}")
        return default


def write_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    create_dirs: bool = True
) -> bool:
    """
    Safely write data to JSON file with error handling.
    
    Args:
        data: Data to serialize to JSON
        file_path: Path to output file
        indent: JSON indentation (default: 2)
        create_dirs: Create parent directories if they don't exist (default: True)
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> tracker = {"file.pdf": {"file_hash": "abc123"}}
        >>> write_json(tracker, "pdf_tracker.json")
        True
    """
    file_path = Path(file_path)
    
    try:
        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return True
        
    except (IOError, TypeError) as e:
        print(f"❌ Error writing JSON to {file_path}: {e}")
        return False


def update_json(
    file_path: Union[str, Path],
    updates: Dict[str, Any],
    create_if_missing: bool = True
) -> bool:
    """
    Update specific keys in a JSON file without overwriting the entire file.
    
    Args:
        file_path: Path to JSON file
        updates: Dictionary of key-value pairs to update
        create_if_missing: Create file if it doesn't exist (default: True)
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> update_json("pdf_tracker.json", {"new_file.pdf": {...}})
        True
    """
    file_path = Path(file_path)
    
    # Read existing data
    if file_path.exists():
        data = read_json(file_path, default={})
    elif create_if_missing:
        data = {}
    else:
        print(f"⚠️  File doesn't exist: {file_path}")
        return False
    
    # Update with new data
    if isinstance(data, dict):
        data.update(updates)
    else:
        print(f"❌ Existing data is not a dictionary: {file_path}")
        return False
    
    # Write back
    return write_json(data, file_path)


# ============================================================================
# SECTION 3: Directory Utilities
# ============================================================================

def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Path: Path object to the directory
        
    Example:
        >>> ensure_directory("data/knowledge_base")
        PosixPath('data/knowledge_base')
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_files(
    dir_path: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        dir_path: Directory to search
        pattern: Glob pattern (default: "*" for all files)
        recursive: Search subdirectories (default: False)
        
    Returns:
        List[Path]: List of matching file paths
        
    Example:
        >>> pdf_files = list_files("data/input", pattern="*.pdf")
        >>> # Returns: [Path('data/input/file1.pdf'), Path('data/input/file2.pdf')]
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        return []
    
    if recursive:
        return sorted([p for p in dir_path.rglob(pattern) if p.is_file()])
    else:
        return sorted([p for p in dir_path.glob(pattern) if p.is_file()])


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        dict: File metadata
            {
                'path': str,
                'name': str,
                'size_bytes': int,
                'size_mb': float,
                'modified': str (ISO format),
                'created': str (ISO format),
                'extension': str
            }
            
    Example:
        >>> info = get_file_info("medical_record.pdf")
        >>> print(info['size_mb'])
        2.5
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    
    return {
        'path': str(file_path.absolute()),
        'name': file_path.name,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'extension': file_path.suffix
    }


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        bool: True if file exists
    """
    return Path(file_path).exists()


def delete_file(file_path: Union[str, Path], ignore_errors: bool = True) -> bool:
    """
    Safely delete a file.
    
    Args:
        file_path: Path to file
        ignore_errors: Don't raise exception if file doesn't exist (default: True)
        
    Returns:
        bool: True if deleted, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        else:
            return ignore_errors
    except OSError as e:
        if not ignore_errors:
            raise
        print(f"⚠️  Error deleting file {file_path}: {e}")
        return False


# ============================================================================
# SECTION 4: Path Validation and Normalization
# ============================================================================

def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize path (resolve relative paths, remove redundant separators).
    
    Args:
        path: Path to normalize
        
    Returns:
        Path: Normalized absolute path
        
    Example:
        >>> normalize_path("./data/../data/input")
        PosixPath('/full/path/to/data/input')
    """
    return Path(path).resolve()


def is_safe_path(path: Union[str, Path], base_dir: Union[str, Path]) -> bool:
    """
    Check if path is within base directory (prevent directory traversal).
    
    Args:
        path: Path to check
        base_dir: Base directory path should be within
        
    Returns:
        bool: True if path is safe (within base_dir)
        
    Example:
        >>> is_safe_path("data/input/file.pdf", "data")
        True
        >>> is_safe_path("../etc/passwd", "data")
        False
    """
    try:
        path = normalize_path(path)
        base_dir = normalize_path(base_dir)
        return path.is_relative_to(base_dir)
    except (ValueError, OSError):
        return False


def get_relative_path(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Get relative path from base directory.
    
    Args:
        path: Full path
        base_dir: Base directory
        
    Returns:
        Path: Relative path
        
    Example:
        >>> get_relative_path("/home/user/data/input/file.pdf", "/home/user/data")
        PosixPath('input/file.pdf')
    """
    path = normalize_path(path)
    base_dir = normalize_path(base_dir)
    return path.relative_to(base_dir)


# ============================================================================
# SECTION 5: Batch Operations
# ============================================================================

def hash_directory(
    dir_path: Union[str, Path],
    pattern: str = "*.pdf",
    algorithm: str = "md5"
) -> Dict[str, str]:
    """
    Calculate hashes for all files in directory.
    
    Args:
        dir_path: Directory containing files
        pattern: File pattern to match (default: "*.pdf")
        algorithm: Hash algorithm (default: "md5")
        
    Returns:
        dict: {filename: hash_value}
        
    Example:
        >>> hashes = hash_directory("data/input", pattern="*.pdf")
        >>> # Returns: {"file1.pdf": "abc123", "file2.pdf": "def456"}
    """
    files = list_files(dir_path, pattern=pattern)
    
    hashes = {}
    for file_path in files:
        try:
            file_hash = calculate_file_hash(file_path, algorithm=algorithm)
            hashes[file_path.name] = file_hash
        except Exception as e:
            print(f"⚠️  Error hashing {file_path.name}: {e}")
    
    return hashes


def copy_file_metadata(src: Dict[str, Any], dst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to copy selected metadata fields between file info dicts.
    
    Args:
        src: Source metadata dict
        dst: Destination metadata dict (will be updated)
        
    Returns:
        dict: Updated destination dict
    """
    copy_fields = ['size_bytes', 'modified', 'created', 'extension']
    for field in copy_fields:
        if field in src:
            dst[field] = src[field]
    return dst


# ============================================================================
# SECTION 6: Size and Format Utilities
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        str: Formatted size (e.g., "2.5 MB")
        
    Example:
        >>> format_file_size(2621440)
        '2.50 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_directory_size(dir_path: Union[str, Path]) -> int:
    """
    Calculate total size of all files in directory (recursive).
    
    Args:
        dir_path: Directory path
        
    Returns:
        int: Total size in bytes
        
    Example:
        >>> size = get_directory_size("data/knowledge_base")
        >>> print(format_file_size(size))
        '125.50 MB'
    """
    total_size = 0
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        return 0
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size
