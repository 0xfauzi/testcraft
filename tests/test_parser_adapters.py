"""
Tests for parser adapter implementations.

This module contains comprehensive tests for the CodebaseParser and TestMapper
adapters, including parsing functionality, test mapping, and error handling.
"""

import tempfile
from pathlib import Path

import pytest

from testcraft.adapters.parsing.codebase_parser import CodebaseParser, ParseError
from testcraft.adapters.parsing.test_mapper import TestMapper, TestMappingError
from testcraft.domain.models import TestElement, TestElementType


@pytest.fixture
def sample_python_file():
    """Create a temporary Python file with sample code."""
    sample_code = '''"""Sample module for testing."""

import os
from typing import Dict, List

def public_function(x: int) -> int:
    """A public function for testing."""
    return x * 2

def _private_function(data: str) -> str:
    """A private function."""
    return data.lower()

class SampleClass:
    """A sample class for testing."""

    def __init__(self, value: int):
        """Initialize the sample class."""
        self.value = value

    def public_method(self) -> int:
        """A public method."""
        return self.value * 3

    def _private_method(self) -> bool:
        """A private method."""
        return self.value > 0

class AnotherClass:
    """Another sample class."""

    def calculate(self, x: int, y: int) -> int:
        """Calculate something."""
        return x + y
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        f.flush()
        yield Path(f.name)

    Path(f.name).unlink()  # Clean up


@pytest.fixture
def sample_test_file():
    """Create a temporary test file with sample tests."""
    test_code = '''"""Tests for the sample module."""

import pytest
from sample import SampleClass, public_function

def test_public_function():
    """Test the public function."""
    assert public_function(5) == 10

def test_private_function():
    """Test the private function."""
    from sample import _private_function
    assert _private_function("TEST") == "test"

class TestSampleClass:
    """Test class for SampleClass."""

    def test_init(self):
        """Test SampleClass initialization."""
        obj = SampleClass(10)
        assert obj.value == 10

    def test_public_method(self):
        """Test the public method."""
        obj = SampleClass(5)
        assert obj.public_method() == 15

def another_test():
    """Another test function."""
    pass
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f.flush()
        yield Path(f.name)

    Path(f.name).unlink()  # Clean up


class TestCodebaseParser:
    """Test cases for CodebaseParser."""

    def test_init(self):
        """Test CodebaseParser initialization."""
        parser = CodebaseParser()
        assert parser._cache == {}

    def test_parse_file_success(self, sample_python_file):
        """Test successful file parsing."""
        parser = CodebaseParser()
        result = parser.parse_file(sample_python_file)

        # Check result structure
        assert "ast" in result
        assert "elements" in result
        assert "imports" in result
        assert "language" in result
        assert "parse_errors" in result
        assert "source_content" in result

        # Check language detection
        assert result["language"] == "python"

        # Check elements were extracted
        elements = result["elements"]
        assert len(elements) >= 6  # 2 functions + 2 classes + 2+ methods

        # Check source content was extracted
        source_content = result["source_content"]
        assert len(source_content) >= 6

        # Check imports were extracted
        imports = result["imports"]
        assert len(imports) >= 2  # os and typing imports

        # Verify caching
        result2 = parser.parse_file(sample_python_file)
        assert result is result2  # Same object due to caching

    def test_parse_file_nonexistent(self):
        """Test parsing a non-existent file."""
        parser = CodebaseParser()
        with pytest.raises(ParseError, match="File does not exist"):
            parser.parse_file(Path("nonexistent.py"))

    def test_parse_file_syntax_error(self):
        """Test parsing a file with syntax errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def invalid_syntax(\n    pass")  # Missing closing parenthesis
            f.flush()

            parser = CodebaseParser()
            with pytest.raises(ParseError, match="Syntax error"):
                parser.parse_file(Path(f.name))

        Path(f.name).unlink()

    def test_extract_functions(self, sample_python_file):
        """Test function extraction."""
        parser = CodebaseParser()

        # Extract all functions
        functions = parser.extract_functions(sample_python_file, include_private=True)
        function_names = [f.name for f in functions]

        assert "public_function" in function_names
        assert "_private_function" in function_names
        assert len(functions) == 2

        # Extract only public functions
        public_functions = parser.extract_functions(
            sample_python_file, include_private=False
        )
        public_names = [f.name for f in public_functions]

        assert "public_function" in public_names
        assert "_private_function" not in public_names
        assert len(public_functions) == 1

    def test_extract_classes(self, sample_python_file):
        """Test class extraction."""
        parser = CodebaseParser()
        classes = parser.extract_classes(sample_python_file)
        class_names = [c.name for c in classes]

        assert "SampleClass" in class_names
        assert "AnotherClass" in class_names
        assert len(classes) == 2

        # Check class details
        sample_class = next(c for c in classes if c.name == "SampleClass")
        assert sample_class.type == TestElementType.CLASS
        assert sample_class.docstring == "A sample class for testing."

    def test_extract_methods(self, sample_python_file):
        """Test method extraction."""
        parser = CodebaseParser()

        # Extract all methods
        all_methods = parser.extract_methods(sample_python_file)
        method_names = [m.name for m in all_methods]

        assert "SampleClass.__init__" in method_names
        assert "SampleClass.public_method" in method_names
        assert "SampleClass._private_method" in method_names
        assert "AnotherClass.calculate" in method_names

        # Extract methods for specific class
        sample_methods = parser.extract_methods(
            sample_python_file, class_name="SampleClass"
        )
        sample_method_names = [m.name for m in sample_methods]

        assert "SampleClass.__init__" in sample_method_names
        assert "SampleClass.public_method" in sample_method_names
        assert "AnotherClass.calculate" not in sample_method_names

    def test_analyze_dependencies(self, sample_python_file):
        """Test dependency analysis."""
        parser = CodebaseParser()
        deps = parser.analyze_dependencies(sample_python_file)

        assert "imports" in deps
        assert "dependencies" in deps
        assert "internal_deps" in deps
        assert "circular_deps" in deps

        # Check that standard library modules are detected as external
        dependencies = deps["dependencies"]
        assert "os" in dependencies
        assert "typing" in dependencies


class TestTestMapper:
    """Test cases for TestMapper."""

    def test_init(self):
        """Test TestMapper initialization."""
        mapper = TestMapper()
        assert mapper._parser is not None
        assert len(mapper._test_patterns) > 0

    def test_map_tests_basic(self, sample_python_file, sample_test_file):
        """Test basic test mapping functionality."""
        # First parse the source file to get elements
        parser = CodebaseParser()
        source_result = parser.parse_file(sample_python_file)
        source_elements = source_result["elements"]

        # Map tests
        mapper = TestMapper()
        result = mapper.map_tests(source_elements, [str(sample_test_file)])

        # Check result structure
        assert "test_mapping" in result
        assert "missing_tests" in result
        assert "coverage_gaps" in result
        assert "test_suggestions" in result
        assert "total_source_elements" in result
        assert "total_test_elements" in result
        assert "coverage_percentage" in result

        # Check that some mappings were found
        test_mapping = result["test_mapping"]
        assert len(test_mapping) == len(source_elements)

        # Check coverage percentage calculation
        coverage = result["coverage_percentage"]
        assert 0 <= coverage <= 100

    def test_map_tests_no_existing_tests(self, sample_python_file):
        """Test mapping when no existing tests are provided."""
        parser = CodebaseParser()
        source_result = parser.parse_file(sample_python_file)
        source_elements = source_result["elements"]

        mapper = TestMapper()
        result = mapper.map_tests(source_elements, existing_tests=None)

        # All elements should be missing tests
        missing_tests = result["missing_tests"]
        assert len(missing_tests) == len(source_elements)

        # Coverage should be 0%
        assert result["coverage_percentage"] == 0.0

    def test_suggest_test_name_for_element(self):
        """Test test name suggestion for different element types."""
        mapper = TestMapper()

        # Test function
        func_element = TestElement(
            name="my_function",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="A test function",
        )
        assert mapper.suggest_test_name_for_element(func_element) == "test_my_function"

        # Test method
        method_element = TestElement(
            name="MyClass.my_method",
            type=TestElementType.METHOD,
            line_range=(10, 15),
            docstring="A test method",
        )
        assert mapper.suggest_test_name_for_element(method_element) == "test_my_method"

        # Test class
        class_element = TestElement(
            name="MyClass",
            type=TestElementType.CLASS,
            line_range=(1, 20),
            docstring="A test class",
        )
        assert mapper.suggest_test_name_for_element(class_element) == "TestMyClass"

    def test_find_test_file_for_source(self):
        """Test finding test files for source files."""
        mapper = TestMapper()

        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source file
            source_file = temp_path / "my_module.py"
            source_file.write_text("# Sample source")

            # Create test file
            test_dir = temp_path / "tests"
            test_dir.mkdir()
            test_file = test_dir / "test_my_module.py"
            test_file.write_text("# Sample test")

            # Test finding the test file
            found_test = mapper.find_test_file_for_source(
                source_file, test_directories=[test_dir]
            )

            assert found_test == test_file

    def test_find_test_file_not_found(self):
        """Test when no test file is found."""
        mapper = TestMapper()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_file = temp_path / "my_module.py"
            source_file.write_text("# Sample source")

            found_test = mapper.find_test_file_for_source(
                source_file, test_directories=[temp_path / "nonexistent"]
            )

            assert found_test is None

    def test_is_test_element(self):
        """Test identification of test elements."""
        mapper = TestMapper()

        # Test function with test prefix
        test_func = TestElement(
            name="test_something",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring=None,
        )
        assert mapper._is_test_element(test_func) is True

        # Regular function
        regular_func = TestElement(
            name="regular_function",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring=None,
        )
        assert mapper._is_test_element(regular_func) is False

        # Test class
        test_class = TestElement(
            name="TestSomething",
            type=TestElementType.CLASS,
            line_range=(1, 10),
            docstring=None,
        )
        assert mapper._is_test_element(test_class) is True

    def test_generate_test_template(self):
        """Test test template generation."""
        mapper = TestMapper()

        # Test function template
        func_element = TestElement(
            name="my_function",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="A test function",
        )
        template = mapper._generate_test_template(func_element)
        assert "def test_my_function():" in template
        assert "Test my_function." in template

        # Test class template
        class_element = TestElement(
            name="MyClass",
            type=TestElementType.CLASS,
            line_range=(1, 20),
            docstring="A test class",
        )
        template = mapper._generate_test_template(class_element)
        assert "class TestMyClass:" in template
        assert "Test MyClass initialization." in template

    def test_map_tests_error_handling(self):
        """Test error handling in test mapping."""
        mapper = TestMapper()

        # Test with invalid source elements (this should not raise an error)
        result = mapper.map_tests([], existing_tests=["nonexistent_file.py"])
        assert result["total_source_elements"] == 0

        # Test with None as source_elements should trigger an error
        with pytest.raises((TestMappingError, TypeError)):
            mapper.map_tests(None, existing_tests=[])


@pytest.mark.integration
class TestParserIntegration:
    """Integration tests for parser components."""

    def test_full_parsing_pipeline(self, sample_python_file, sample_test_file):
        """Test the full parsing and mapping pipeline."""
        # Parse source file
        parser = CodebaseParser()
        source_result = parser.parse_file(sample_python_file)

        # Verify source parsing worked
        assert len(source_result["elements"]) > 0
        assert len(source_result["source_content"]) > 0

        # Map tests
        mapper = TestMapper()
        mapping_result = mapper.map_tests(
            source_result["elements"], [str(sample_test_file)]
        )

        # Verify mapping worked
        assert mapping_result["total_source_elements"] > 0
        assert mapping_result["total_test_elements"] > 0
        assert 0 <= mapping_result["coverage_percentage"] <= 100

        # Verify some elements have source code
        source_content = source_result["source_content"]
        for _element_name, content in source_content.items():
            assert isinstance(content, str)
            assert len(content) > 0
            # Content should contain def or class keywords for Python code
            assert any(keyword in content for keyword in ["def ", "class "])

    def test_parser_cache_consistency(self, sample_python_file):
        """Test that parser caching maintains consistency."""
        parser = CodebaseParser()

        # Parse file twice
        result1 = parser.parse_file(sample_python_file)
        result2 = parser.parse_file(sample_python_file)

        # Results should be identical (cached)
        assert result1 is result2
        assert result1["elements"] == result2["elements"]
        assert result1["source_content"] == result2["source_content"]

        # Clear cache and parse again
        parser._cache.clear()
        result3 = parser.parse_file(sample_python_file)

        # Result should be different object but same content
        assert result3 is not result1
        assert len(result3["elements"]) == len(result1["elements"])
        assert len(result3["source_content"]) == len(result1["source_content"])
