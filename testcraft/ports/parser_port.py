"""
Parser Port interface definition.

This module defines the interface for parsing source code files and
mapping test elements, including AST analysis and code structure extraction.
"""

from pathlib import Path
from typing import Any

from typing_extensions import Protocol

from ..domain.models import TestElement


class ParserPort(Protocol):
    """
    Interface for parsing source code files and extracting test elements.

    This protocol defines the contract for parsing operations, including
    file parsing, AST analysis, and test element extraction.
    """

    def parse_file(
        self, file_path: Path, language: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Parse a source code file and extract structural information.

        Args:
            file_path: Path to the source file to parse
            language: Programming language (auto-detected if None)
            **kwargs: Additional parsing parameters

        Returns:
            Dictionary containing:
                - 'ast': Abstract syntax tree representation
                - 'elements': List of TestElement objects found
                - 'imports': List of import statements
                - 'language': Detected programming language
                - 'parse_errors': List of any parsing errors encountered

        Raises:
            ParserError: If file parsing fails
        """
        ...

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
            ParserError: If test mapping fails
        """
        ...

    def extract_functions(
        self, file_path: Path, include_private: bool = False, **kwargs: Any
    ) -> list[TestElement]:
        """
        Extract function definitions from a source file.

        Args:
            file_path: Path to the source file
            include_private: Whether to include private/protected functions
            **kwargs: Additional extraction parameters

        Returns:
            List of TestElement objects representing functions

        Raises:
            ParserError: If function extraction fails
        """
        ...

    def extract_classes(
        self, file_path: Path, include_abstract: bool = True, **kwargs: Any
    ) -> list[TestElement]:
        """
        Extract class definitions from a source file.

        Args:
            file_path: Path to the source file
            include_abstract: Whether to include abstract classes
            **kwargs: Additional extraction parameters

        Returns:
            List of TestElement objects representing classes

        Raises:
            ParserError: If class extraction fails
        """
        ...

    def extract_methods(
        self, file_path: Path, class_name: str | None = None, **kwargs: Any
    ) -> list[TestElement]:
        """
        Extract method definitions from a source file or specific class.

        Args:
            file_path: Path to the source file
            class_name: Optional specific class to extract methods from
            **kwargs: Additional extraction parameters

        Returns:
            List of TestElement objects representing methods

        Raises:
            ParserError: If method extraction fails
        """
        ...

    def analyze_dependencies(self, file_path: Path, **kwargs: Any) -> dict[str, Any]:
        """
        Analyze dependencies and imports in a source file.

        Args:
            file_path: Path to the source file
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing:
                - 'imports': List of imported modules/functions
                - 'dependencies': List of external dependencies
                - 'internal_deps': List of internal module dependencies
                - 'circular_deps': List of circular dependencies found

        Raises:
            ParserError: If dependency analysis fails
        """
        ...
