"""
Tests for PytestRefiner service.

This module contains unit tests for the pytest refiner service.
"""

import pytest

from testcraft.application.generation.services.pytest_refiner import PytestRefiner


class TestPytestRefiner:
    """Test cases for PytestRefiner service."""

    def test_extract_import_path_from_failure_basic(self):
        """Test basic import path extraction from failure message."""
        failure_message = "ModuleNotFoundError: No module named 'testcraft.missing_module'"
        
        result = PytestRefiner.extract_import_path_from_failure(failure_message)
        
        # Should extract module path
        assert result == "testcraft.missing_module"

    def test_extract_import_path_from_failure_no_match(self):
        """Test import path extraction when no module error is found."""
        failure_message = "Some other error message"
        
        result = PytestRefiner.extract_import_path_from_failure(failure_message)
        
        # Should return empty string for non-import errors
        assert result == ""

    def test_extract_import_path_from_failure_various_formats(self):
        """Test import path extraction from various error message formats."""
        test_cases = [
            ("ImportError: cannot import name 'SomeClass' from 'mymodule'", ""),
            ("ModuleNotFoundError: No module named 'pkg.submodule'", "pkg.submodule"),
            ("", ""),
        ]
        
        for message, expected in test_cases:
            result = PytestRefiner.extract_import_path_from_failure(message)
            assert result == expected
