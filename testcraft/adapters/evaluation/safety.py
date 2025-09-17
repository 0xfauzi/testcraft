"""
Safety validation functions for evaluation operations.

This module provides safety checks and validation utilities for evaluation
operations, including file path validation and content filtering.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SafetyValidator:
    """Handles safety validation for evaluation operations."""

    def __init__(self, project_root: Path, safety_enabled: bool = True):
        """
        Initialize safety validator.

        Args:
            project_root: Project root for safety validation
            safety_enabled: Whether to enforce safety policies
        """
        self.project_root = project_root
        self.safety_enabled = safety_enabled

    def should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during golden repo evaluation."""
        skip_patterns = [
            "__pycache__",
            ".git",
            "test_",
            "_test.py",
            "conftest.py",
            "__init__.py",
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def validate_file_path(self, file_path: Path) -> bool:
        """Validate that a file path is safe for operations."""
        if not self.safety_enabled:
            return True

        try:
            # Import safety policies from the existing module
            from ..io.safety import SafetyPolicies

            SafetyPolicies.validate_file_path(file_path, self.project_root)
            return True
        except Exception as e:
            logger.warning(f"File path validation failed for {file_path}: {e}")
            return False

    def validate_test_content(self, test_content: str) -> tuple[bool, list[str]]:
        """Validate test content for safety concerns."""
        warnings = []
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            "os.system(",
            "subprocess.call(",
            "eval(",
            "exec(",
            "import os",
            "from os import",
            "__import__",
            "open(",  # File operations should be carefully reviewed
        ]
        
        for pattern in dangerous_patterns:
            if pattern in test_content:
                warnings.append(f"Potentially unsafe operation detected: {pattern}")
        
        # Check for network operations
        network_patterns = [
            "requests.",
            "urllib.",
            "http.",
            "socket.",
            "telnet",
        ]
        
        for pattern in network_patterns:
            if pattern in test_content:
                warnings.append(f"Network operation detected: {pattern}")
        
        # Return True if no critical safety issues found
        # (warnings are informational, not blocking)
        return len(warnings) == 0 or not self.safety_enabled, warnings

    def sanitize_evaluation_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize evaluation metadata to remove sensitive information."""
        sanitized = metadata.copy()
        
        # Remove potentially sensitive keys
        sensitive_keys = [
            "api_key",
            "token",
            "password",
            "secret",
            "credential",
        ]
        
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
        
        return sanitized

    def validate_source_file_access(self, source_file: str) -> bool:
        """Validate that accessing a source file is safe."""
        if not self.safety_enabled:
            return True
            
        try:
            source_path = Path(source_file)
            
            # Check if file is within project boundaries
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            
            # Resolve to handle symlinks and relative paths
            resolved_path = source_path.resolve()
            resolved_project = self.project_root.resolve()
            
            # Check if the resolved path is within the project
            try:
                resolved_path.relative_to(resolved_project)
                return True
            except ValueError:
                logger.warning(f"Source file {source_file} is outside project root")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to validate source file access for {source_file}: {e}")
            return False
