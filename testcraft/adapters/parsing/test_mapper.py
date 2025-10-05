"""
Test mapper adapter implementation.

This module provides an adapter for mapping test functions to source code elements,
identifying test coverage gaps, and supporting pytest naming conventions.
"""

import re
from pathlib import Path
from typing import Any

from ...domain.models import TestElement, TestElementType
from .codebase_parser import CodebaseParser, ParseError


class TestMappingError(Exception):
    """Exception raised when test mapping fails."""

    pass


class TestMapper:
    """
    Adapter for mapping tests to source code elements.

    Implements test mapping functionality including identifying existing test
    coverage, mapping test functions to source elements, and suggesting
    missing test structures.
    """

    def __init__(self) -> None:
        """Initialize the test mapper."""
        self._parser = CodebaseParser()
        self._test_patterns = [
            # pytest naming conventions
            r"^test_(.+)$",  # test_function_name
            r"^(.+)_test$",  # function_name_test
            r"^Test([A-Z][a-zA-Z]*)$",  # TestClassName
        ]
        self._method_test_patterns = [
            r"^test_(.+)$",  # test_method_name
            r"^(.+)_test$",  # method_name_test
        ]

    def map_tests(
        self,
        source_elements: list[TestElement],
        existing_tests: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Map source elements to their corresponding test elements.

        Args:
            source_elements: List of TestElement objects from source code
            existing_tests: Optional list of existing test file paths
            **kwargs: Additional mapping parameters

        Returns:
            Dictionary containing:
                - 'test_mapping': Mapping of source elements to test elements
                - 'missing_tests': List of elements without corresponding tests
                - 'coverage_gaps': List of test coverage gaps identified
                - 'test_suggestions': Suggested test structure for missing tests

        Raises:
            TestMappingError: If test mapping fails
        """
        try:
            # Validate inputs
            if source_elements is None:
                raise TestMappingError("source_elements cannot be None")
            if not source_elements:
                return {
                    "test_mapping": {},
                    "missing_tests": [],
                    "coverage_gaps": [],
                    "test_suggestions": [],
                    "total_source_elements": 0,
                    "total_test_elements": 0,
                    "coverage_percentage": 100.0,
                }

            # Validate and parse existing test files if provided
            test_elements: list[TestElement] = []
            if existing_tests is not None:
                # Validate existing_tests is a list
                if not isinstance(existing_tests, list):
                    raise TestMappingError("existing_tests must be a list or None")

                # Filter out invalid entries
                valid_test_paths = []
                for test_path in existing_tests:
                    if isinstance(test_path, str) and test_path.strip():
                        valid_test_paths.append(test_path.strip())

                if valid_test_paths:
                    test_elements = self._parse_test_files(valid_test_paths)

            # Create mappings
            test_mapping = self._create_test_mapping(source_elements, test_elements)
            missing_tests = self._identify_missing_tests(source_elements, test_mapping)
            coverage_gaps = self._identify_coverage_gaps(source_elements, test_mapping)
            test_suggestions = self._generate_test_suggestions(missing_tests)

            return {
                "test_mapping": test_mapping,
                "missing_tests": missing_tests,
                "coverage_gaps": coverage_gaps,
                "test_suggestions": test_suggestions,
                "total_source_elements": len(source_elements),
                "total_test_elements": len(test_elements),
                "coverage_percentage": self._calculate_coverage_percentage(
                    source_elements, test_mapping
                ),
            }

        except Exception as e:
            if isinstance(e, TestMappingError):
                raise
            raise TestMappingError(f"Failed to map tests: {e}") from e

    def find_test_file_for_source(
        self, source_file_path: Path, test_directories: list[Path] | None = None
    ) -> Path | None:
        """
        Find the corresponding test file for a source file.

        Args:
            source_file_path: Path to the source file
            test_directories: Optional list of test directories to search

        Returns:
            Path to the test file if found, None otherwise
        """
        try:
            # Validate source_file_path
            if not source_file_path or not isinstance(source_file_path, Path):
                return None

            # Validate source file exists
            if not source_file_path.exists() or not source_file_path.is_file():
                return None

            if not test_directories:
                # Default test directories relative to source file
                try:
                    test_directories = [
                        source_file_path.parent / "tests",
                        source_file_path.parent.parent / "tests",
                        Path("tests"),
                    ]
                except (OSError, ValueError):
                    # If path operations fail, return None
                    return None
            else:
                # Validate test_directories is a list
                if not isinstance(test_directories, list):
                    return None

            try:
                source_name = source_file_path.stem
                if not source_name:
                    return None

                possible_test_names = [
                    f"test_{source_name}.py",
                    f"{source_name}_test.py",
                    f"test{source_name}.py",
                ]

                for test_dir in test_directories:
                    try:
                        if not test_dir or not isinstance(test_dir, Path):
                            continue

                        if not test_dir.exists() or not test_dir.is_dir():
                            continue

                        for test_name in possible_test_names:
                            if not test_name or not isinstance(test_name, str):
                                continue

                            test_file = test_dir / test_name
                            if test_file.exists() and test_file.is_file():
                                return test_file
                    except (OSError, ValueError, TypeError):
                        # Skip directories that cause path errors
                        continue

            except (AttributeError, TypeError, ValueError):
                # Handle cases where path operations fail
                return None

        except (TypeError, AttributeError, ValueError, OSError):
            # Handle any other validation errors
            return None

        return None

    def suggest_test_name_for_element(
        self, element: TestElement, style: str = "pytest"
    ) -> str:
        """
        Suggest a test function name for a source element.

        Args:
            element: TestElement to generate test name for
            style: Testing style ("pytest", "unittest")

        Returns:
            Suggested test function name
        """
        if not element or not hasattr(element, "name") or not element.name:
            return "test_element"

        if style == "pytest":
            if element.type == TestElementType.FUNCTION:
                return f"test_{element.name}"
            elif element.type == TestElementType.METHOD:
                # For methods like "ClassName.method_name"
                try:
                    parts = element.name.split(".", 1)
                    if len(parts) == 2:
                        _, method_name = parts
                        return f"test_{method_name}"
                    else:
                        # Fallback if no dot found
                        return f"test_{element.name}"
                except (AttributeError, ValueError):
                    # Fallback if split fails
                    return f"test_{element.name}"
            elif element.type == TestElementType.CLASS:
                return f"Test{element.name}"

        # Default fallback
        try:
            clean_name = element.name.replace(".", "_")
        except (AttributeError, TypeError):
            clean_name = str(element.name) if element.name else "element"
        return f"test_{clean_name}"

    def _parse_test_files(self, test_file_paths: list[str]) -> list[TestElement]:
        """Parse test files and extract test elements."""
        test_elements: list[TestElement] = []
        parsed_count = 0
        failed_count = 0

        if not test_file_paths:
            return test_elements

        for test_path_str in test_file_paths:
            # Validate input
            if not test_path_str or not isinstance(test_path_str, str):
                failed_count += 1
                continue

            try:
                test_path = Path(test_path_str)

                # Validate path exists and is readable
                if not test_path.exists():
                    failed_count += 1
                    continue

                if not test_path.is_file():
                    failed_count += 1
                    continue

                # Attempt to parse the file
                try:
                    parse_result = self._parser.parse_file(test_path)
                    elements = parse_result.get("elements", [])

                    # Filter for test functions/methods/classes
                    for element in elements:
                        if self._is_test_element(element):
                            test_elements.append(element)

                    parsed_count += 1

                except ParseError:
                    failed_count += 1
                    # Could log warning here if logging was available
                    continue
                except Exception:
                    # Catch any other parsing errors
                    failed_count += 1
                    continue

            except (OSError, ValueError, TypeError):
                # Handle path-related errors
                failed_count += 1
                continue

        return test_elements

    def _is_test_element(self, element: TestElement) -> bool:
        """Check if an element is a test element based on naming conventions."""
        name = element.name

        # Check function/method test patterns
        if element.type in [TestElementType.FUNCTION, TestElementType.METHOD]:
            for pattern in self._test_patterns:
                if re.match(pattern, name):
                    return True

        # Check class test patterns
        if element.type == TestElementType.CLASS:
            for pattern in self._test_patterns:
                if re.match(pattern, name):
                    return True

        return False

    def _create_test_mapping(
        self, source_elements: list[TestElement], test_elements: list[TestElement]
    ) -> dict[str, list[TestElement]]:
        """Create mapping from source elements to their test elements."""
        mapping = {}

        for source_element in source_elements:
            matching_tests = []

            for test_element in test_elements:
                if self._elements_match(source_element, test_element):
                    matching_tests.append(test_element)

            mapping[source_element.name] = matching_tests

        return mapping

    def _elements_match(
        self, source_element: TestElement, test_element: TestElement
    ) -> bool:
        """Check if a test element matches a source element."""
        source_name = source_element.name
        test_name = test_element.name

        # Handle class matching
        if source_element.type == TestElementType.CLASS:
            # Check for TestClassName pattern
            if test_name == f"Test{source_name}":
                return True

        # Handle function/method matching
        elif source_element.type in [TestElementType.FUNCTION, TestElementType.METHOD]:
            # Extract the actual function/method name (remove class prefix if method)
            if source_element.type == TestElementType.METHOD:
                try:
                    parts = source_name.split(".", 1)
                    if len(parts) != 2:
                        # If split doesn't produce exactly 2 parts, treat as simple name
                        actual_name = source_name
                    else:
                        _, method_name = parts
                        actual_name = method_name
                except (AttributeError, ValueError):
                    # Fallback if split fails
                    actual_name = source_name
            else:
                actual_name = source_name

            # Check various test naming patterns
            test_patterns = [
                f"test_{actual_name}",
                f"{actual_name}_test",
                f"test{actual_name}",
            ]

            for pattern in test_patterns:
                if test_name == pattern or test_name.endswith(f".{pattern}"):
                    return True

        return False

    def _identify_missing_tests(
        self,
        source_elements: list[TestElement],
        test_mapping: dict[str, list[TestElement]],
    ) -> list[TestElement]:
        """Identify source elements without corresponding tests."""
        missing_tests = []

        for source_element in source_elements:
            mapped_tests = test_mapping.get(source_element.name, [])
            if not mapped_tests:
                missing_tests.append(source_element)

        return missing_tests

    def _identify_coverage_gaps(
        self,
        source_elements: list[TestElement],
        test_mapping: dict[str, list[TestElement]],
    ) -> list[dict[str, Any]]:
        """Identify test coverage gaps."""
        coverage_gaps = []

        for source_element in source_elements:
            mapped_tests = test_mapping.get(source_element.name, [])

            gap_info: dict[str, Any] = {
                "element": source_element,
                "gap_type": None,
                "description": None,
            }

            if not mapped_tests:
                gap_info["gap_type"] = "missing_test"
                gap_info["description"] = (
                    f"No test found for {source_element.type} '{source_element.name}'"
                )
            elif len(mapped_tests) == 1:
                gap_info["gap_type"] = "single_test"
                gap_info["description"] = (
                    f"Only one test found for {source_element.type} '{source_element.name}'"
                )
            else:
                # Multiple tests found - this might be good coverage
                continue

            coverage_gaps.append(gap_info)

        return coverage_gaps

    def _generate_test_suggestions(
        self, missing_tests: list[TestElement]
    ) -> list[dict[str, Any]]:
        """Generate test structure suggestions for missing tests."""
        suggestions = []

        for element in missing_tests:
            suggestion = {
                "element": element,
                "suggested_test_name": self.suggest_test_name_for_element(element),
                "test_type": self._suggest_test_type(element),
                "priority": self._calculate_test_priority(element),
                "test_template": self._generate_test_template(element),
            }
            suggestions.append(suggestion)

        # Sort by priority (high priority first)
        suggestions.sort(
            key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(str(x["priority"]), 1),
            reverse=True,
        )

        return suggestions

    def _suggest_test_type(self, element: TestElement) -> str:
        """Suggest what type of test should be created."""
        if element.type == TestElementType.CLASS:
            return "unit_test_class"
        elif element.type == TestElementType.METHOD:
            if element.name.startswith("_"):
                return "private_method_test"
            return "method_test"
        elif element.type == TestElementType.FUNCTION:
            if element.name.startswith("_"):
                return "private_function_test"
            return "function_test"
        else:
            return "generic_test"

    def _calculate_test_priority(self, element: TestElement) -> str:
        """Calculate the priority for creating tests for an element."""
        # Public functions/methods get high priority
        try:
            name_parts = element.name.split(".")
            if name_parts and name_parts[-1] and not name_parts[-1].startswith("_"):
                return "high"
        except (AttributeError, TypeError, IndexError):
            # Fallback if name operations fail
            pass

        # Private methods get medium priority
        if element.type == TestElementType.METHOD:
            return "medium"

        # Other elements get low priority
        return "low"

    def _generate_test_template(self, element: TestElement) -> str:
        """Generate a basic test template for an element."""
        test_name = self.suggest_test_name_for_element(element)

        if element.type == TestElementType.CLASS:
            return f'''class {test_name}:
    """Test class for {element.name}."""

    def test_initialization(self):
        """Test {element.name} initialization."""
        # TODO: Implement test
        pass
'''
        else:
            return f'''def {test_name}():
    """Test {element.name}."""
    # TODO: Implement test
    pass
'''

    def _calculate_coverage_percentage(
        self,
        source_elements: list[TestElement],
        test_mapping: dict[str, list[TestElement]],
    ) -> float:
        """Calculate the percentage of source elements that have tests."""
        if not source_elements:
            return 100.0

        elements_with_tests = sum(
            1 for element in source_elements if test_mapping.get(element.name, [])
        )

        return (elements_with_tests / len(source_elements)) * 100.0
