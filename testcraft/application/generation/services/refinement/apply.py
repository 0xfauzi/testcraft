"""
Safe application of refined test content.

This module provides safe file writing operations with backup/rollback capabilities,
path validation, and atomic write operations for test refinement.
"""

import ast
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .....ports.writer_port import WriterPort

logger = logging.getLogger(__name__)


class SafeApplyService:
    """
    Service for safely applying refined content to test files.
    
    Provides backup/rollback capabilities, path validation, and atomic writes
    to ensure test files are never left in a corrupted state.
    """

    def __init__(
        self,
        writer_port: Optional[WriterPort] = None,
        validate_syntax: bool = True,
    ):
        """
        Initialize the safe apply service.

        Args:
            writer_port: Optional writer port for safe file operations
            validate_syntax: Whether to validate Python syntax before/after writes
        """
        self.writer_port = writer_port
        self.validate_syntax = validate_syntax

    def write_refined_content_safely(
        self, test_file: Path, refined_content: str
    ) -> dict[str, Any]:
        """
        Write refined content safely using WriterPort or fallback safety checks.
        
        Args:
            test_file: Path to test file to update
            refined_content: New content to write
            
        Returns:
            Dictionary with write result:
                - success: bool
                - error: str (if not successful)
                - backup_path: str (if backup created)
        """
        # Validate path safety (must be under tests/ directory)
        if not self._validate_test_path_safety(test_file):
            return {
                "success": False,
                "error": f"Path validation failed: {test_file} is not a valid test file path"
            }
        
        # Use WriterPort if available
        if self.writer_port:
            try:
                result = self.writer_port.write_test_file(
                    test_path=test_file,
                    test_content=refined_content,
                    overwrite=True
                )
                return {
                    "success": result.get("success", False),
                    "error": result.get("error") if not result.get("success", False) else None,
                    "backup_path": result.get("backup_path")
                }
            except Exception as e:
                return {"success": False, "error": f"WriterPort failed: {e}"}
        
        # Fallback to local safety checks
        return self._write_with_local_safety(test_file, refined_content)

    def _validate_test_path_safety(self, test_file: Path) -> bool:
        """
        Validate that the test file path is safe for refinement writes.
        
        Args:
            test_file: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            resolved_path = test_file.resolve()
            path_str = str(resolved_path)
            
            # Must be a Python file
            if not path_str.endswith('.py'):
                logger.warning("Refinement path validation failed: not a Python file: %s", path_str)
                return False
            
            # Must contain 'test' in the path (either 'tests/' directory or 'test_' filename)
            if not ('tests' in resolved_path.parts or test_file.name.startswith('test_')):
                logger.warning("Refinement path validation failed: not a test file path: %s", path_str)
                return False
                
            return True
            
        except Exception as e:
            logger.warning("Refinement path validation error for %s: %s", test_file, e)
            return False

    def _write_with_local_safety(
        self, test_file: Path, refined_content: str
    ) -> dict[str, Any]:
        """
        Write content with local safety checks (backup/rollback).
        
        Args:
            test_file: Path to test file to update
            refined_content: New content to write
            
        Returns:
            Dictionary with write result
        """
        backup_path = test_file.with_suffix(test_file.suffix + ".refine_backup")
        
        try:
            # Create backup
            if test_file.exists():
                backup_content = test_file.read_text(encoding="utf-8")
                backup_path.write_text(backup_content, encoding="utf-8")
            
            # Validate syntax before write if enabled
            if self.validate_syntax:
                try:
                    ast.parse(refined_content)
                except SyntaxError as e:
                    return {
                        "success": False,
                        "error": f"Content failed syntax validation before write: {e}"
                    }
            
            # Write refined content
            test_file.write_text(refined_content, encoding="utf-8")
            
            # Verify syntax after write
            if self.validate_syntax:
                try:
                    written_content = test_file.read_text(encoding="utf-8")
                    ast.parse(written_content)
                except Exception as e:
                    # Rollback on syntax failure
                    if backup_path.exists():
                        test_file.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                    return {
                        "success": False,
                        "error": f"Content failed syntax validation after write, rolled back: {e}"
                    }
            
            # Clean up backup on success
            if backup_path.exists():
                backup_path.unlink()
            
            return {"success": True}
            
        except Exception as e:
            # Attempt rollback on any failure
            if backup_path.exists() and test_file.exists():
                try:
                    test_file.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception as rollback_error:
                    logger.error("Failed to rollback after write error: %s", rollback_error)
            
            return {"success": False, "error": f"Write operation failed: {e}"}
        finally:
            # Clean up backup file in all cases
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception:
                    pass  # Best effort cleanup

    def prepare_test_environment(self, test_file: Path) -> dict[str, str]:
        """
        Prepare environment with reliable PYTHONPATH for test execution.
        
        Args:
            test_file: Path to test file being executed
            
        Returns:
            Environment dictionary with enhanced PYTHONPATH
        """
        env = os.environ.copy()
        
        # Build comprehensive PYTHONPATH for reliable imports
        python_paths = []
        
        # Add directory containing the test file
        test_dir = test_file.parent
        python_paths.append(str(test_dir))
        
        # Find project root and add it + src/ to PYTHONPATH
        try:
            # Look for project markers to find root
            project_root = test_dir
            while project_root != project_root.parent:
                # Check for common project markers
                markers = [
                    "pyproject.toml", "setup.py", "setup.cfg", 
                    ".git", "requirements.txt", "Pipfile", "uv.lock"
                ]
                
                if any((project_root / marker).exists() for marker in markers):
                    break
                project_root = project_root.parent
            
            # Add project root to PYTHONPATH if different from test_dir
            if project_root != test_dir:
                python_paths.append(str(project_root))
            
            # Add src/ directory if it exists
            src_path = project_root / "src"
            if src_path.exists() and src_path.is_dir():
                python_paths.append(str(src_path))
                
        except Exception as e:
            # Log but don't fail - fallback to just test_dir
            logger.debug("Could not detect project root for test PYTHONPATH: %s", e)
        
        # Combine with existing PYTHONPATH
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        
        return env

    def apply_refinement_safely(self, test_file: Path, refined_content: str) -> None:
        """
        Apply refined content to test file safely (legacy method).

        Args:
            test_file: Path to test file to update
            refined_content: New content to write

        Raises:
            Exception: If write operation fails
        """
        # Create backup
        backup_path = test_file.with_suffix(test_file.suffix + ".bak")
        backup_path.write_text(test_file.read_text(encoding="utf-8"), encoding="utf-8")

        try:
            # Write refined content
            test_file.write_text(refined_content, encoding="utf-8")
        except Exception:
            # Restore from backup on failure
            test_file.write_text(
                backup_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            raise
        finally:
            # Clean up backup
            if backup_path.exists():
                backup_path.unlink()
