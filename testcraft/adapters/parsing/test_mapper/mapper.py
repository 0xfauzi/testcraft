"""
Test mapper orchestrator implementation.

This module provides the main TestMapper class that orchestrates test mapping
functionality by coordinating analysis, scoring, and module path resolution.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from ....domain.models import TestElement, TestElementType
from ..codebase_parser import CodebaseParser, ParseError
from .analysis import TestFileAnalyzer
from .module_paths import ModulePathHelper
from .scoring import (
    calculate_mapping_scores,
    extract_covered_elements, 
    estimate_coverage_percentage,
    estimate_coverage_percentage_enhanced,
    calculate_test_priority,
    calculate_coverage_percentage,
    MIN_MAPPING_SCORE_THRESHOLD
)

logger = logging.getLogger(__name__)


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

    def __init__(self):
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
        
        # Initialize helper components
        self._analyzer = TestFileAnalyzer()
        self._module_path_helper = ModulePathHelper()

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
            # Parse existing test files if provided
            test_elements = []
            if existing_tests:
                test_elements = self._parse_test_files(existing_tests)

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
                "coverage_percentage": calculate_coverage_percentage(source_elements, test_mapping),
            }

        except Exception as e:
            if isinstance(e, TestMappingError):
                raise
            raise TestMappingError(f"Failed to map tests: {e}")

    def map_source_to_tests_usage_based(
        self,
        source_path: Path,
        test_files: list[Path],
        project_root: Optional[Path] = None
    ) -> dict[str, Any]:
        """
        Map source file to test files using usage-based AST analysis.
        
        This method analyzes test files for imports, function calls, attribute access,
        and mock patches to determine which test files actually exercise a source file.
        
        Args:
            source_path: Path to the source file to map
            test_files: List of potential test file paths
            project_root: Optional project root for module path derivation
            
        Returns:
            Dictionary containing:
                - 'test_files': List of test files that reference the source
                - 'elements_covered': Set of source elements that appear to be tested
                - 'mapping_scores': Dict of test file -> mapping score
                - 'coverage_percentage': Estimated coverage percentage
        """
        start_time = time.time()
        
        try:
            # Derive authoritative import paths for the source file
            source_module_paths = self._module_path_helper.derive_source_module_paths(
                source_path, project_root
            )
            
            if not source_module_paths:
                # Can't determine module path, fall back to filename-based mapping
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Usage-based mapping failed for {source_path}: could not derive module paths ({duration_ms:.1f}ms)")
                
                return {
                    'test_files': [],
                    'elements_covered': set(),
                    'mapping_scores': {},
                    'coverage_percentage': 0.0,
                    'fallback_reason': 'Could not derive module paths for source file',
                    'duration_ms': duration_ms,
                }
            
            # Analyze each test file for usage of the source module
            test_file_mappings = {}
            
            for test_file in test_files:
                if not test_file.exists():
                    continue
                    
                analysis = self._analyzer.analyze_test_file(test_file)
                if analysis is None:
                    continue
                
                # Calculate mapping score for this test file
                max_score = 0
                for module_path in source_module_paths:
                    score = analysis.get_mapping_score_for_module(module_path)
                    max_score = max(max_score, score)
                
                if max_score >= MIN_MAPPING_SCORE_THRESHOLD:
                    test_file_mappings[test_file] = analysis
            
            # Calculate final scores
            mapping_scores, total_score = calculate_mapping_scores(
                source_module_paths, test_file_mappings
            )
            
            # Extract elements that appear to be covered
            elements_covered = extract_covered_elements(
                source_path, source_module_paths, test_file_mappings
            )
            
            # Estimate coverage percentage
            coverage_percentage = estimate_coverage_percentage_enhanced(
                source_path, elements_covered, total_score
            )
            
            # Log success metrics
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Usage-based mapping for {source_path}: found {len(test_file_mappings)} test files, "
                f"total score {total_score}, {len(elements_covered)} elements covered ({duration_ms:.1f}ms)"
            )
            
            return {
                'test_files': list(test_file_mappings.keys()),
                'elements_covered': elements_covered,
                'mapping_scores': mapping_scores,
                'coverage_percentage': coverage_percentage,
                'source_module_paths': source_module_paths,
                'duration_ms': duration_ms,
                'analysis_stats': {
                    'test_files_analyzed': len(test_files),
                    'test_files_matched': len(test_file_mappings),
                    'total_mapping_score': total_score,
                    'elements_covered_count': len(elements_covered),
                }
            }
            
        except Exception as e:
            raise TestMappingError(f"Failed to perform usage-based mapping for {source_path}: {e}")

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
        if not test_directories:
            # Default test directories relative to source file
            test_directories = [
                source_file_path.parent / "tests",
                source_file_path.parent.parent / "tests",
                Path("tests"),
            ]

        source_name = source_file_path.stem
        possible_test_names = [
            f"test_{source_name}.py",
            f"{source_name}_test.py",
            f"test{source_name}.py",
        ]

        for test_dir in test_directories:
            if not test_dir.exists():
                continue

            for test_name in possible_test_names:
                test_file = test_dir / test_name
                if test_file.exists():
                    return test_file

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
        if style == "pytest":
            if element.type == TestElementType.FUNCTION:
                return f"test_{element.name}"
            elif element.type == TestElementType.METHOD:
                # For methods like "ClassName.method_name"
                class_name, method_name = element.name.split(".", 1)
                return f"test_{method_name}"
            elif element.type == TestElementType.CLASS:
                return f"Test{element.name}"

        # Default fallback
        clean_name = element.name.replace(".", "_")
        return f"test_{clean_name}"

    def _parse_test_files(self, test_file_paths: list[str]) -> list[TestElement]:
        """Parse test files and extract test elements."""
        test_elements = []

        for test_path_str in test_file_paths:
            test_path = Path(test_path_str)
            if not test_path.exists():
                continue

            try:
                parse_result = self._parser.parse_file(test_path)
                elements = parse_result["elements"]

                # Filter for test functions/methods/classes
                for element in elements:
                    if self._is_test_element(element):
                        test_elements.append(element)

            except ParseError:
                # Continue parsing other files even if one fails
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
                _, method_name = source_name.split(".", 1)
                actual_name = method_name
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

            gap_info = {
                "element": source_element,
                "gap_type": None,
                "description": None,
            }

            if not mapped_tests:
                gap_info["gap_type"] = "missing_test"
                gap_info[
                    "description"
                ] = f"No test found for {source_element.type} '{source_element.name}'"
            elif len(mapped_tests) == 1:
                gap_info["gap_type"] = "single_test"
                gap_info[
                    "description"
                ] = f"Only one test found for {source_element.type} '{source_element.name}'"
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
                "priority": calculate_test_priority(element.name, str(element.type)),
                "test_template": self._generate_test_template(element),
            }
            suggestions.append(suggestion)

        # Sort by priority (high priority first)
        suggestions.sort(
            key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]],
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
