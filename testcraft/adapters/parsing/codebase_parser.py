"""
Codebase parser adapter implementation.

This module provides an adapter for parsing Python source code files,
extracting both structural metadata and source code content for test generation.
"""

import ast
from pathlib import Path
from typing import Any

from ...domain.models import TestElement, TestElementType


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass


class CodebaseParser:
    """
    Adapter for parsing Python source code files.

    Implements the ParserPort interface for extracting structural information
    and source code content from Python files using AST analysis.
    """

    def __init__(self):
        """Initialize the codebase parser."""
        self._cache: dict[str, dict[str, Any]] = {}

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
                - 'source_content': Dict mapping element IDs to source code

        Raises:
            ParseError: If file parsing fails
        """
        try:
            # Check cache first (avoid calling absolute() to prevent cwd issues)
            cache_key = str(file_path)
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Read file content
            if not file_path.exists():
                raise ParseError(f"File does not exist: {file_path}")

            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            # Detect language (currently only Python supported)
            detected_language = self._detect_language(file_path, language)
            if detected_language != "python":
                raise ParseError(f"Unsupported language: {detected_language}")

            # Parse AST
            try:
                tree = ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                raise ParseError(f"Syntax error in {file_path}: {e}")

            # Extract elements and their source code
            elements, source_content = self._extract_elements(
                tree, source_code, file_path
            )

            # Extract imports
            imports = self._extract_imports(tree)

            result = {
                "ast": tree,
                "elements": elements,
                "imports": imports,
                "language": detected_language,
                "parse_errors": [],
                "source_content": source_content,
                "file_path": str(file_path),
                "source_lines": source_code.splitlines(),
            }

            # Cache the result
            self._cache[cache_key] = result
            return result

        except Exception as e:
            if isinstance(e, ParseError):
                raise
            raise ParseError(f"Failed to parse {file_path}: {e}")

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
            ParseError: If function extraction fails
        """
        parse_result = self.parse_file(file_path)
        elements = parse_result["elements"]

        functions = [elem for elem in elements if elem.type == TestElementType.FUNCTION]

        if not include_private:
            functions = [func for func in functions if not func.name.startswith("_")]

        return functions

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
            ParseError: If class extraction fails
        """
        parse_result = self.parse_file(file_path)
        elements = parse_result["elements"]

        classes = [elem for elem in elements if elem.type == TestElementType.CLASS]

        # TODO: Add logic to filter abstract classes if needed
        # This would require analyzing the AST to check for ABC inheritance

        return classes

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
            ParseError: If method extraction fails
        """
        parse_result = self.parse_file(file_path)
        elements = parse_result["elements"]

        methods = [elem for elem in elements if elem.type == TestElementType.METHOD]

        if class_name:
            # Filter methods by class name
            # The method names are stored as "ClassName.method_name"
            methods = [
                method for method in methods if method.name.startswith(f"{class_name}.")
            ]

        return methods

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
            ParseError: If dependency analysis fails
        """
        parse_result = self.parse_file(file_path)
        imports = parse_result["imports"]

        # Categorize imports
        external_deps = []
        internal_deps = []

        for import_info in imports:
            module_name = import_info["module"]
            if self._is_external_dependency(module_name):
                external_deps.append(module_name)
            else:
                internal_deps.append(module_name)

        return {
            "imports": imports,
            "dependencies": external_deps,
            "internal_deps": internal_deps,
            "circular_deps": [],  # TODO: Implement circular dependency detection
        }

    def _detect_language(self, file_path: Path, language: str | None) -> str:
        """Detect the programming language of a file."""
        if language:
            return language.lower()

        suffix = file_path.suffix.lower()
        if suffix == ".py":
            return "python"

        # Default to python for now
        return "python"

    def _extract_elements(
        self, tree: ast.AST, source_code: str, file_path: Path
    ) -> tuple[list[TestElement], dict[str, str]]:
        """
        Extract TestElement objects and their source code from the AST.

        Returns:
            Tuple of (elements, source_content_mapping)
        """
        elements = []
        source_content = {}
        source_lines = source_code.splitlines()

        # Extract top-level elements only (not nested)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                element, source = self._extract_function_element(
                    node, source_lines, parent_class=None
                )
                elements.append(element)
                source_content[element.name] = source

            elif isinstance(node, ast.ClassDef):
                element, source = self._extract_class_element(node, source_lines)
                elements.append(element)
                source_content[element.name] = source

                # Extract methods within the class
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        method_element, method_source = self._extract_function_element(
                            class_node, source_lines, parent_class=node.name
                        )
                        elements.append(method_element)
                        source_content[method_element.name] = method_source

        return elements, source_content

    def _extract_function_element(
        self,
        node: ast.FunctionDef,
        source_lines: list[str],
        parent_class: str | None = None,
    ) -> tuple[TestElement, str]:
        """Extract a TestElement for a function/method and its source code."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get docstring
        docstring = ast.get_docstring(node)

        # Determine element type and name
        if parent_class:
            element_type = TestElementType.METHOD
            element_name = f"{parent_class}.{node.name}"
        else:
            element_type = TestElementType.FUNCTION
            element_name = node.name

        # Extract source code
        source_code = "\n".join(source_lines[start_line - 1 : end_line])

        element = TestElement(
            name=element_name,
            type=element_type,
            line_range=(start_line, end_line),
            docstring=docstring,
        )

        return element, source_code

    def _extract_class_element(
        self, node: ast.ClassDef, source_lines: list[str]
    ) -> tuple[TestElement, str]:
        """Extract a TestElement for a class and its source code."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get docstring
        docstring = ast.get_docstring(node)

        # Extract source code
        source_code = "\n".join(source_lines[start_line - 1 : end_line])

        element = TestElement(
            name=node.name,
            type=TestElementType.CLASS,
            line_range=(start_line, end_line),
            docstring=docstring,
        )

        return element, source_code

    def _extract_imports(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract import information from the AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": node.module or "",
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )

        return imports

    def _is_external_dependency(self, module_name: str) -> bool:
        """Determine if a module is an external dependency."""
        # Simple heuristic: if it's a standard library or common third-party module
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "typing",
            "pathlib",
            "collections",
            "dataclasses",
            "enum",
            "functools",
            "itertools",
            "datetime",
        }

        common_third_party = {
            "pytest",
            "numpy",
            "pandas",
            "requests",
            "flask",
            "django",
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "click",
        }

        base_module = module_name.split(".")[0]
        return base_module in stdlib_modules or base_module in common_third_party
