"""
Tests for writer adapter implementations.

This module tests both the append and AST merge writer adapters,
including safety policies, formatting, and error handling.
"""

import ast
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from testcraft.adapters.io.safety import SafetyError, SafetyPolicies
from testcraft.adapters.io.writer_append import WriterAppendAdapter, WriterAppendError
from testcraft.adapters.io.writer_ast_merge import (
    WriterASTMergeAdapter,
    WriterASTMergeError,
)


class TestSafetyPolicies:
    """Test safety policies and validation functions."""

    def test_validate_file_path_allows_test_directory(self):
        """Test that test directory paths are allowed."""
        # Should not raise
        SafetyPolicies.validate_file_path(Path("tests/test_example.py"))
        SafetyPolicies.validate_file_path(Path("test/test_example.py"))

    def test_validate_file_path_blocks_non_test_directory(self):
        """Test that non-test directory paths are blocked."""
        with pytest.raises(SafetyError, match="Writing only allowed"):
            SafetyPolicies.validate_file_path(Path("src/example.py"))

    def test_validate_file_path_blocks_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(SafetyError, match="Path traversal not allowed"):
            SafetyPolicies.validate_file_path(Path("tests/../secret.py"))

    def test_validate_file_path_blocks_hidden_files(self):
        """Test that hidden files are blocked."""
        with pytest.raises(SafetyError, match="Hidden files/directories not allowed"):
            SafetyPolicies.validate_file_path(Path("tests/.hidden_test.py"))

    def test_validate_file_size_allows_normal_content(self):
        """Test that normal-sized content is allowed."""
        content = "def test_example(): pass\n" * 100
        # Should not raise
        SafetyPolicies.validate_file_size(content)

    def test_validate_file_size_blocks_large_content(self):
        """Test that overly large content is blocked."""
        # Create content larger than 1MB
        large_content = "x" * (1024 * 1024 + 1)
        with pytest.raises(SafetyError, match="File size.*exceeds limit"):
            SafetyPolicies.validate_file_size(large_content)

    def test_validate_content_safety_allows_safe_code(self):
        """Test that safe code is allowed."""
        safe_content = """
import pytest
from mymodule import MyClass

def test_example():
    obj = MyClass()
    assert obj.method() == "expected"
"""
        # Should not raise
        SafetyPolicies.validate_content_safety(safe_content)

    def test_validate_content_safety_blocks_dangerous_patterns(self):
        """Test that dangerous patterns are blocked."""
        dangerous_patterns = [
            "exec('print(1)')",
            "eval('1+1')",
            "subprocess.call(['rm', '-rf', '/'])",
            "os.system('rm file')",
            "__import__('os')",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(SafetyError, match="dangerous pattern"):
                SafetyPolicies.validate_content_safety(pattern)

    def test_validate_python_syntax_allows_valid_python(self):
        """Test that valid Python syntax is allowed."""
        valid_code = """
def test_example():
    x = 1
    assert x == 1
"""
        # Should not raise
        SafetyPolicies.validate_python_syntax(valid_code)

    def test_validate_python_syntax_blocks_invalid_python(self):
        """Test that invalid Python syntax is blocked."""
        invalid_code = "def test_example(\n    pass"  # Missing closing parenthesis
        with pytest.raises(SafetyError, match="Invalid Python syntax"):
            SafetyPolicies.validate_python_syntax(invalid_code)

    def test_validate_test_file_name_allows_valid_names(self):
        """Test that valid test file names are allowed."""
        valid_names = [
            Path("tests/test_example.py"),
            Path("tests/my_test.py"),
            Path("test/test_complex_example.py"),
        ]

        for name in valid_names:
            # Should not raise
            SafetyPolicies.validate_test_file_name(name)

    def test_validate_test_file_name_blocks_invalid_names(self):
        """Test that invalid test file names are blocked."""
        invalid_names = [
            Path("tests/example.py"),  # No test prefix/suffix
            Path("tests/test_example.txt"),  # Wrong extension
            Path("tests/example_tests.py"),  # Wrong suffix format
        ]

        for name in invalid_names:
            with pytest.raises(SafetyError):
                SafetyPolicies.validate_test_file_name(name)


class TestWriterAppendAdapter:
    """Test the append writer adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_dir = self.temp_dir / "tests"
        self.test_dir.mkdir()
        self.adapter = WriterAppendAdapter(project_root=self.temp_dir)
        self.dry_run_adapter = WriterAppendAdapter(
            project_root=self.temp_dir, dry_run=True
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_new_file(self):
        """Test writing to a new file."""
        test_file = self.test_dir / "test_new.py"
        content = """
def test_example():
    assert True
"""

        with patch.object(
            self.adapter, "_format_content", return_value=content.strip()
        ):
            result = self.adapter.write_file(test_file, content)

        assert result["success"]
        assert result["file_path"] == str(test_file)
        assert not result["file_existed"]
        assert test_file.exists()

    def test_append_to_existing_file(self):
        """Test appending to an existing file."""
        test_file = self.test_dir / "test_existing.py"
        existing_content = "def test_first(): pass\n"
        new_content = "def test_second(): pass\n"

        # Create existing file
        test_file.write_text(existing_content)

        with patch.object(
            self.adapter, "_format_content", return_value=existing_content + new_content
        ):
            result = self.adapter.write_file(test_file, new_content)

        assert result["success"]
        assert result["file_existed"]
        assert result["backup_path"]  # Should create backup

    def test_dry_run_mode(self):
        """Test dry-run mode doesn't write files."""
        test_file = self.test_dir / "test_dry_run.py"
        content = "def test_example(): pass\n"

        result = self.dry_run_adapter.write_file(test_file, content)

        assert result["success"]
        assert result["dry_run"]
        assert "content_preview" in result
        assert not test_file.exists()  # File should not be created

    def test_write_test_file(self):
        """Test writing a test file with metadata extraction."""
        test_file = self.test_dir / "test_module.py"
        test_content = """
import pytest
from mymodule import MyClass

def test_example():
    assert True

def test_another():
    assert False
"""

        with patch.object(
            self.adapter, "_format_content", return_value=test_content.strip()
        ):
            result = self.adapter.write_test_file(
                test_file, test_content, source_file="mymodule.py"
            )

        assert result["success"]
        assert result["source_file"] == "mymodule.py"
        assert "test_example" in result["test_functions"]
        assert "test_another" in result["test_functions"]

    def test_backup_file(self):
        """Test file backup functionality."""
        test_file = self.test_dir / "test_backup.py"
        content = "def test_example(): pass\n"
        test_file.write_text(content)

        result = self.adapter.backup_file(test_file)

        assert result["success"]
        assert result["original_path"] == str(test_file)
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == content

    def test_ensure_directory(self):
        """Test directory creation."""
        new_dir = self.test_dir / "subdir"

        result = self.adapter.ensure_directory(new_dir)

        assert result["success"]
        assert result["created"]
        assert new_dir.exists()

    def test_validation_errors(self):
        """Test that validation errors are raised appropriately."""
        # Test path outside allowed directory
        with pytest.raises(WriterAppendError):
            self.adapter.write_file("src/bad.py", "content")

        # Test dangerous content
        with pytest.raises(WriterAppendError):
            self.adapter.write_file("tests/test_bad.py", "exec('evil code')")

    @patch("testcraft.adapters.io.writer_append.format_python_content")
    def test_format_content_with_black_and_isort(self, mock_format):
        """Test that content is formatted using the Python formatters module."""
        mock_format.return_value = "formatted content"

        content = "import os\nimport sys\ndef test():pass"
        formatted = self.adapter._format_content(content)

        # Should call the format_python_content function from python_formatters
        mock_format.assert_called_once_with(content, timeout=15, disable_ruff=False)
        assert formatted == "formatted content"

    @patch("subprocess.run")
    def test_format_content_fallback_on_error(self, mock_subprocess):
        """Test that formatting falls back to original content on error."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "black")

        content = "def test():\n    pass\n"
        formatted = self.adapter._format_content(content)

        # Should return original content when formatting fails
        assert formatted == content

    @patch("subprocess.run")
    def test_format_content_fallback_on_timeout(self, mock_subprocess):
        """Test that formatting falls back to original content on timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(["black"], 30)

        content = "def test():\n    pass\n"
        formatted = self.adapter._format_content(content)

        # Should return original content when formatting times out
        assert formatted == content


class TestWriterASTMergeAdapter:
    """Test the AST merge writer adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_dir = self.temp_dir / "tests"
        self.test_dir.mkdir()
        self.adapter = WriterASTMergeAdapter(project_root=self.temp_dir)
        self.dry_run_adapter = WriterASTMergeAdapter(
            project_root=self.temp_dir, dry_run=True
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_new_file(self):
        """Test writing to a new file."""
        test_file = self.test_dir / "test_new.py"
        content = """
import pytest

def test_example():
    assert True
"""

        with patch.object(
            self.adapter, "_format_content", return_value=content.strip()
        ):
            result = self.adapter.write_file(test_file, content)

        assert result["success"]
        assert result["file_path"] == str(test_file)
        assert not result["file_existed"]
        assert not result["merged"]
        assert test_file.exists()

    def test_merge_with_existing_file(self):
        """Test merging with an existing file."""
        test_file = self.test_dir / "test_merge.py"
        existing_content = """
import pytest

def test_first():
    assert True
"""
        new_content = """
import unittest

def test_second():
    assert False
"""

        # Create existing file
        test_file.write_text(existing_content)

        with patch.object(
            self.adapter, "_format_content", return_value="merged content"
        ):
            result = self.adapter.write_file(test_file, new_content)

        assert result["success"]
        assert result["file_existed"]
        assert result["merged"]
        assert result["backup_path"]

    def test_dry_run_with_diff(self):
        """Test dry-run mode generates diff."""
        test_file = self.test_dir / "test_diff.py"
        existing_content = "def test_old(): pass\n"
        new_content = "def test_new(): pass\n"

        # Create existing file
        test_file.write_text(existing_content)

        with patch.object(self.adapter, "_merge_content", return_value=new_content):
            with patch.object(
                self.adapter, "_format_content", return_value=new_content
            ):
                result = self.dry_run_adapter.write_file(test_file, new_content)

        assert result["success"]
        assert result["dry_run"]
        assert "diff" in result
        assert not test_file.read_text() == new_content  # Original file unchanged

    def test_merge_imports(self):
        """Test that imports are merged correctly."""
        existing_content = """
import os
import sys
from typing import List

def test_existing():
    pass
"""
        new_content = """
import pytest
import sys  # Duplicate, should not be added again
from typing import Dict

def test_new():
    pass
"""

        merged = self.adapter._merge_content(existing_content, new_content)
        tree = ast.parse(merged)

        # Extract import names
        import_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name)

        # Should have os, sys, and pytest (sys not duplicated)
        assert "os" in import_names
        assert "sys" in import_names
        assert "pytest" in import_names

    def test_merge_functions(self):
        """Test that functions are merged without duplicates."""
        existing_content = """
def test_existing():
    assert True

def test_shared():
    assert True
"""
        new_content = """
def test_new():
    assert False

def test_shared():  # Duplicate, should not be added again
    assert False
"""

        merged = self.adapter._merge_content(existing_content, new_content)
        tree = ast.parse(merged)

        # Extract function names
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)

        # Should have test_existing, test_shared (once), and test_new
        assert "test_existing" in function_names
        assert "test_new" in function_names
        assert function_names.count("test_shared") == 1  # No duplicates

    def test_merge_classes(self):
        """Test that classes are merged without duplicates."""
        existing_content = """
class TestExisting:
    def test_method(self):
        pass

class TestShared:
    def test_old(self):
        pass
"""
        new_content = """
class TestNew:
    def test_method(self):
        pass

class TestShared:  # Duplicate, should not be added again
    def test_new(self):
        pass
"""

        merged = self.adapter._merge_content(existing_content, new_content)
        tree = ast.parse(merged)

        # Extract class names
        class_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)

        # Should have TestExisting, TestShared (once), and TestNew
        assert "TestExisting" in class_names
        assert "TestNew" in class_names
        assert class_names.count("TestShared") == 1  # No duplicates

    def test_merge_fallback_on_syntax_error(self):
        """Test that merging falls back to concatenation on syntax errors."""
        existing_content = "def valid_function(): pass"
        invalid_content = "def invalid_function(\n    pass"  # Syntax error

        merged = self.adapter._merge_content(existing_content, invalid_content)

        # Should fall back to simple concatenation
        assert existing_content in merged
        assert invalid_content in merged

    def test_write_test_file_extracts_metadata(self):
        """Test that test file writing extracts proper metadata."""
        test_file = self.test_dir / "test_metadata.py"
        content = """
import pytest
from mymodule import MyClass

def test_function_one():
    assert True

def test_function_two():
    assert False

def helper_function():  # Should not be included in test functions
    pass
"""

        with patch.object(
            self.adapter, "_format_content", return_value=content.strip()
        ):
            result = self.adapter.write_test_file(
                test_file, content, source_file="mymodule.py"
            )

        assert result["success"]
        assert result["source_file"] == "mymodule.py"
        test_functions = result["test_functions"]
        assert "test_function_one" in test_functions
        assert "test_function_two" in test_functions
        assert (
            "helper_function" not in test_functions
        )  # Should not include non-test functions

    @patch("testcraft.adapters.io.writer_ast_merge.format_python_content")
    def test_format_content_integration(self, mock_format):
        """Test that the formatting pipeline is called correctly."""
        mock_format.return_value = "formatted content"

        content = "import os\ndef test():pass"
        formatted = self.adapter._format_content(content)

        # Should call the format_python_content function from python_formatters
        mock_format.assert_called_once_with(content, timeout=15, disable_ruff=False)
        assert formatted == "formatted content"

    @patch("subprocess.run")
    def test_format_content_timeout_fallback(self, mock_subprocess):
        """Test that formatting falls back to original content on timeout."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(["isort"], 30)

        content = "import os\ndef test(): pass"
        formatted = self.adapter._format_content(content)

        # Should return original content when formatting times out
        assert formatted == content

    def test_validation_integration(self):
        """Test that validation is properly integrated."""
        # Test unsafe path
        with pytest.raises(WriterASTMergeError):
            self.adapter.write_file("../outside.py", "content")

        # Test dangerous content
        with pytest.raises(WriterASTMergeError):
            self.adapter.write_file("tests/test_bad.py", "exec('malicious')")
