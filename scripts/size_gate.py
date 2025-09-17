#!/usr/bin/env python3
"""
Size Gate - Enforce file size limits in CI

This script scans Python files in the repository and fails if any exceed
the maximum line count threshold (default: 1000 lines).

Excluded directories:
- .venv, test-env: Virtual environments
- build, htmlcov, testcraft.egg-info: Build artifacts and coverage reports
- __pycache__, .pytest_cache, .mypy_cache: Python/tool caches
- .git: Version control metadata

Rationale for 1000-line limit:
- Maintains code readability and comprehensibility
- Encourages proper separation of concerns
- Prevents monolithic modules that are hard to maintain
- Aligns with team coding standards for modular architecture

Environment variables:
- MAX_LINES: Override the default 1000-line threshold
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


def get_max_lines() -> int:
    """Get the maximum lines threshold from environment or default."""
    try:
        return int(os.environ.get('MAX_LINES', '1000'))
    except ValueError:
        print("Warning: Invalid MAX_LINES value, using default 1000")
        return 1000


def is_excluded_path(path: Path) -> bool:
    """Check if a path should be excluded from scanning."""
    excluded_dirs = {
        '.venv', 'test-env', 'build', 'htmlcov', 'testcraft.egg-info',
        '__pycache__', '.pytest_cache', '.mypy_cache', '.git', '_legacy'
    }
    
    # Check if any part of the path matches excluded directories
    for part in path.parts:
        if part in excluded_dirs:
            return True
    
    # Exclude legacy/backup files by naming pattern
    file_name = path.name.lower()
    if any(pattern in file_name for pattern in ['_legacy', '_backup', '_original']):
        return True
    
    return False


def count_lines_in_file(file_path: Path) -> int:
    """Count lines in a file, handling encoding issues gracefully."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except (OSError, IOError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0


def scan_python_files(root_dir: Path) -> List[Tuple[Path, int]]:
    """Scan for Python files and return those exceeding the line limit."""
    max_lines = get_max_lines()
    violations = []
    
    # Find all .py files recursively
    for py_file in root_dir.rglob('*.py'):
        # Skip excluded paths
        if is_excluded_path(py_file):
            continue
        
        # Count lines in the file
        line_count = count_lines_in_file(py_file)
        
        # Check if it exceeds the limit
        if line_count > max_lines:
            violations.append((py_file, line_count))
    
    return violations


def main() -> int:
    """Main entry point for the size gate."""
    # Get the repository root (assume script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    print(f"Scanning Python files in: {repo_root}")
    max_lines = get_max_lines()
    print(f"Maximum lines per file: {max_lines}")
    
    # Scan for violations
    violations = scan_python_files(repo_root)
    
    if not violations:
        print("✅ All Python files are within the size limit!")
        return 0
    
    # Report violations
    print(f"\n❌ Found {len(violations)} file(s) exceeding {max_lines} lines:")
    print()
    
    for file_path, line_count in sorted(violations, key=lambda x: x[1], reverse=True):
        # Show path relative to repo root for cleaner output
        relative_path = file_path.relative_to(repo_root)
        print(f"  {relative_path}: {line_count} lines")
    
    print()
    print("Please refactor these files to be under the size limit.")
    print("Consider breaking large files into smaller, focused modules.")
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
