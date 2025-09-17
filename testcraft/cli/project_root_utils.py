"""
Project root derivation utilities for CLI.

Handles project root detection when --target-files are passed from outside the working directory.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def derive_project_root(
    project_path: Optional[Path] = None,
    target_files: Optional[list[Path]] = None
) -> Path:
    """
    Derive the actual project root path considering target files.

    When target files are provided, if any file is outside the current working directory,
    ascend from the file to find project markers (pyproject.toml, .git, pytest.ini).

    Args:
        project_path: Explicitly provided project path (takes priority)
        target_files: List of target files to generate tests for

    Returns:
        Path to the detected project root

    Raises:
        ValueError: If project root cannot be determined
    """
    # Priority 1: Explicitly provided project path
    if project_path:
        project_path = project_path.resolve()
        if project_path.exists():
            logger.debug(f"Using explicitly provided project path: {project_path}")
            return project_path
        else:
            raise ValueError(f"Provided project path does not exist: {project_path}")

    # Priority 2: Derive from target files if they exist outside current working directory
    cwd = Path.cwd().resolve()
    
    if target_files:
        for target_file in target_files:
            target_file = Path(target_file).resolve()
            
            # Check if target file is outside current working directory
            try:
                target_file.relative_to(cwd)
                # File is within cwd, continue to next file
                continue
            except ValueError:
                # File is outside cwd, attempt to find project root from file location
                logger.debug(f"Target file {target_file} is outside cwd, deriving project root")
                derived_root = _find_project_root_from_file(target_file)
                if derived_root:
                    logger.info(f"Derived project root from target file: {derived_root}")
                    return derived_root

    # Priority 3: Use current working directory and search upward
    logger.debug("Using current working directory as fallback")
    project_root = _find_project_root_from_file(cwd)
    if project_root:
        return project_root
    
    # Fallback: use current working directory
    logger.warning("Could not find project root markers, using current working directory")
    return cwd


def _find_project_root_from_file(file_path: Path) -> Optional[Path]:
    """
    Find project root by ascending from a file or directory.

    Looks for common project markers:
    - pyproject.toml (Python packaging)
    - .git (Git repository)
    - pytest.ini (pytest configuration)
    - setup.py (legacy Python packaging)
    - setup.cfg (Python packaging configuration)
    - pyproject.toml.lock (Poetry lock file indicates project root)

    Args:
        file_path: Starting path (file or directory)

    Returns:
        Path to project root if found, None otherwise
    """
    # If it's a file, start from its directory
    if file_path.is_file():
        search_path = file_path.parent
    else:
        search_path = file_path

    # Define project markers in order of preference
    project_markers = [
        "pyproject.toml",
        ".git",
        "pytest.ini",
        "setup.py", 
        "setup.cfg",
        "requirements.txt",  # Common in Python projects
        "Pipfile",          # Pipenv projects
        "poetry.lock",      # Poetry projects
        "pyproject.lock",   # Some poetry variations
    ]

    # Search upward from the current path
    current_path = search_path.resolve()
    
    while current_path != current_path.parent:  # Stop at filesystem root
        for marker in project_markers:
            marker_path = current_path / marker
            if marker_path.exists():
                logger.debug(f"Found project marker '{marker}' at {current_path}")
                return current_path
                
        # Move up one level
        current_path = current_path.parent
    
    logger.debug(f"No project markers found ascending from {search_path}")
    return None


def validate_project_path(project_path: Path) -> bool:
    """
    Validate that a path is a reasonable project root.

    Args:
        project_path: Path to validate

    Returns:
        True if path appears to be a valid project root
    """
    if not project_path.exists():
        return False
        
    if not project_path.is_dir():
        return False
    
    # Check for any Python files in the directory tree (basic sanity check)
    try:
        for py_file in project_path.rglob("*.py"):
            # Found at least one Python file, looks like a Python project
            return True
    except (PermissionError, OSError):
        # If we can't traverse the directory, assume it's valid
        pass
    
    # If no Python files found but directory exists, still consider valid
    # (maybe it's a new project or has Python files in subdirectories we can't access)
    return True
