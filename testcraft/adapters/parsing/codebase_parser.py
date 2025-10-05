"""
Codebase parser adapter implementation.

This module provides an adapter for parsing Python source code files,
extracting both structural metadata and source code content for test generation.
"""

import ast
import importlib.util
import sys
import threading
from collections import OrderedDict
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

    def __init__(self, max_cache_size: int = 100) -> None:
        """Initialize the codebase parser.

        Args:
            max_cache_size: Maximum number of files to cache (default: 100)
        """
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = max_cache_size

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
            ParseError: If file parsing fails completely
        """
        cache_key = str(file_path)
        parse_errors: list[dict[str, Any]] = []

        # Check cache first (thread-safe read)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            # Read file content
            if not file_path.exists():
                parse_errors.append(
                    {
                        "type": "file_error",
                        "message": f"File does not exist: {file_path}",
                        "line": None,
                        "column": None,
                    }
                )
                raise ParseError(f"File does not exist: {file_path}")

            try:
                with open(file_path, encoding="utf-8") as f:
                    source_code = f.read()
            except OSError as e:
                parse_errors.append(
                    {
                        "type": "io_error",
                        "message": f"Failed to read file: {e}",
                        "line": None,
                        "column": None,
                    }
                )
                raise ParseError(f"Failed to read {file_path}: {e}") from e

            # Detect language (currently only Python supported)
            detected_language = self._detect_language(file_path, language)
            if detected_language != "python":
                parse_errors.append(
                    {
                        "type": "language_error",
                        "message": f"Unsupported language: {detected_language}",
                        "line": None,
                        "column": None,
                    }
                )
                raise ParseError(f"Unsupported language: {detected_language}")

            # Parse AST with error collection
            tree = None
            try:
                tree = ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                parse_errors.append(
                    {
                        "type": "syntax_error",
                        "message": f"Syntax error: {e}",
                        "line": e.lineno,
                        "column": e.offset,
                    }
                )
                # Try to parse partial content for better error recovery
                tree = self._parse_partial_ast(source_code, file_path, parse_errors)

            # Extract elements and their source code (continue even with parse errors)
            elements, source_content = self._extract_elements_safe(
                tree, source_code, file_path, parse_errors
            )

            # Extract imports (continue even with parse errors)
            imports = self._extract_imports_safe(tree, parse_errors) if tree else []

            result = {
                "ast": tree,
                "elements": elements,
                "imports": imports,
                "language": detected_language,
                "parse_errors": parse_errors,
                "source_content": source_content,
                "file_path": str(file_path),
                "source_lines": source_code.splitlines(),
            }

            # Cache the result (thread-safe write with eviction)
            with self._cache_lock:
                # Evict oldest entries if cache is full
                while len(self._cache) >= self._max_cache_size:
                    self._cache.popitem(last=False)
                self._cache[cache_key] = result

            return result

        except Exception as e:
            if isinstance(e, ParseError):
                raise
            parse_errors.append(
                {
                    "type": "parsing_error",
                    "message": f"Unexpected error: {e}",
                    "line": None,
                    "column": None,
                }
            )
            raise ParseError(f"Failed to parse {file_path}: {e}") from e

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
            if self._is_external_dependency(module_name, file_path):
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
        self, tree: ast.Module, source_code: str, file_path: Path
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

    def _extract_imports(self, tree: ast.Module) -> list[dict[str, Any]]:
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

    def _parse_partial_ast(
        self, source_code: str, file_path: Path, parse_errors: list[dict[str, Any]]
    ) -> ast.Module | None:
        """Attempt to parse partial AST when syntax errors are present."""
        try:
            # Try parsing line by line to find where the error occurs
            lines = source_code.splitlines()
            valid_lines: list[str] = []

            for i, line in enumerate(lines, 1):
                try:
                    # Try parsing just this line in context
                    test_code = "\n".join(valid_lines + [line])
                    ast.parse(test_code, filename=str(file_path))
                    valid_lines.append(line)
                except SyntaxError:
                    # If this line causes syntax error, try to fix common issues
                    fixed_line = self._fix_common_syntax_issues(line)
                    if fixed_line != line:
                        try:
                            test_code = "\n".join(valid_lines + [fixed_line])
                            ast.parse(test_code, filename=str(file_path))
                            valid_lines.append(fixed_line)
                        except SyntaxError:
                            # Line is unparseable, stop here
                            parse_errors.append(
                                {
                                    "type": "unparseable_line",
                                    "message": f"Cannot parse line {i}: {line.strip()}",
                                    "line": i,
                                    "column": None,
                                }
                            )
                            break
                    else:
                        parse_errors.append(
                            {
                                "type": "unparseable_line",
                                "message": f"Cannot parse line {i}: {line.strip()}",
                                "line": i,
                                "column": None,
                            }
                        )
                        break

            if valid_lines:
                partial_code = "\n".join(valid_lines)
                try:
                    return ast.parse(partial_code, filename=str(file_path))
                except SyntaxError:
                    pass

        except Exception:
            pass

        return None

    def _fix_common_syntax_issues(self, line: str) -> str:
        """Attempt to fix common syntax issues in a line."""
        # Fix missing colons after function definitions, class definitions, etc.
        line = line.rstrip()
        if line and not line.endswith(":") and not line.endswith(","):
            # Check if this looks like a function/class definition
            if any(
                keyword in line
                for keyword in [
                    "def ",
                    "class ",
                    "if ",
                    "elif ",
                    "else",
                    "for ",
                    "while ",
                    "try",
                    "except ",
                    "finally",
                ]
            ):
                return line + ":"
        return line

    def _extract_elements_safe(
        self,
        tree: ast.Module | None,
        source_code: str,
        file_path: Path,
        parse_errors: list[dict[str, Any]],
    ) -> tuple[list[TestElement], dict[str, str]]:
        """Extract elements safely, handling parse errors gracefully."""
        elements: list[TestElement] = []
        source_content: dict[str, str] = {}

        if not tree:
            parse_errors.append(
                {
                    "type": "no_ast",
                    "message": "No AST available for element extraction",
                    "line": None,
                    "column": None,
                }
            )
            return elements, source_content

        try:
            elements, source_content = self._extract_elements(
                tree, source_code, file_path
            )
        except Exception as e:
            parse_errors.append(
                {
                    "type": "element_extraction_error",
                    "message": f"Failed to extract elements: {e}",
                    "line": None,
                    "column": None,
                }
            )

        return elements, source_content

    def _extract_imports_safe(
        self, tree: ast.Module | None, parse_errors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract imports safely, handling parse errors gracefully."""
        if not tree:
            parse_errors.append(
                {
                    "type": "no_ast",
                    "message": "No AST available for import extraction",
                    "line": None,
                    "column": None,
                }
            )
            return []

        try:
            return self._extract_imports(tree)
        except Exception as e:
            parse_errors.append(
                {
                    "type": "import_extraction_error",
                    "message": f"Failed to extract imports: {e}",
                    "line": None,
                    "column": None,
                }
            )
            return []

    def _is_external_dependency(self, module_name: str, file_path: Path) -> bool:
        """Determine if a module is an external dependency."""
        base_module = module_name.split(".")[0]

        # Check if it's in sys.modules (indicates it's loaded/available)
        if base_module in sys.modules:
            return True

        # Try to find the module spec
        try:
            spec = importlib.util.find_spec(base_module)
            if spec is not None:
                # Check if it's a standard library module by checking origin
                if spec.origin and "site-packages" not in spec.origin:
                    return True
        except (ImportError, AttributeError):
            pass

        # Check project root for requirements or pyproject.toml
        try:
            project_root = self._find_project_root(file_path.parent)
            if self._is_in_requirements(base_module, project_root):
                return True
        except Exception:
            pass

        # Fallback to heuristics
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
            "re",
            "math",
            "random",
            "uuid",
            "hashlib",
            "base64",
            "urllib",
            "http",
            "socket",
            "ssl",
            "subprocess",
            "shutil",
            "tempfile",
            "glob",
            "fnmatch",
            "pickle",
            "sqlite3",
            "csv",
            "configparser",
            "argparse",
            "logging",
            "multiprocessing",
            "concurrent",
            "asyncio",
            "contextlib",
            "operator",
            "copy",
            "weakref",
            "gc",
            "inspect",
            "traceback",
            "warnings",
            "abc",
            "io",
            "string",
            "textwrap",
            "codecs",
            "locale",
            "struct",
            "array",
            "binascii",
            "calendar",
            "time",
            "sched",
            "queue",
            "threading",
            "zipfile",
            "tarfile",
            "gzip",
            "bz2",
            "lzma",
            "zipimport",
            "imp",
            "importlib",
            "pkgutil",
            "modulefinder",
            "runpy",
            "compileall",
            "py_compile",
            "dis",
            "pydoc",
            "doctest",
            "unittest",
            "test",
            "pdb",
            "profile",
            "cProfile",
            "pstats",
            "timeit",
            "trace",
            "cgitb",
            "webbrowser",
            "wsgiref",
            "html",
            "xml",
            "email",
            "cgi",
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
            "rich",
            "typer",
            "httpx",
            "aiohttp",
            "uvicorn",
            "gunicorn",
            "celery",
            "redis",
            "jinja2",
            "markupsafe",
            "werkzeug",
            "itsdangerous",
            "blinker",
            "alembic",
            "marshmallow",
            "apispec",
            "connexion",
            "tornado",
            "sanic",
            "bottle",
            "cherrypy",
            "web",
            "pyramid",
            "zope",
            "twisted",
            "gevent",
            "grequests",
            "tqdm",
            "colorama",
            "termcolor",
            "blessed",
            "curses",
            "pygame",
            "kivy",
            "tkinter",
            "wx",
            "pyqt",
            "pyside",
            "pillow",
            "opencv",
            "scikit",
            "image",
            "scipy",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "altair",
            "folium",
            "geopandas",
            "shapely",
            "fiona",
            "rasterio",
            "pyproj",
            "cartopy",
            "networkx",
            "igraph",
            "graph",
            "toolz",
            "cytoolz",
            "more",
            "itertools",
            "boltons",
            "python",
            "dateutil",
            "pytz",
            "babel",
            "mako",
            "chameleon",
            "lxml",
            "beautifulsoup4",
            "html5lib",
            "bleach",
            "markdown",
            "mistune",
            "docutils",
            "sphinx",
            "mkdocs",
            "pelican",
            "hugo",
            "jekyll",
            "git",
            "dulwich",
            "gitpython",
            "paramiko",
            "scp",
            "fabric",
            "invoke",
            "pexpect",
            "sh",
            "plumbum",
            "fire",
            "docopt",
            "argh",
            "cement",
            "cliff",
            "cmd2",
            "prompt",
            "toolkit",
            "questionary",
            "inquirer",
            "pyinquirer",
            "cookiecutter",
            "copier",
            "time",
        }

        return base_module in stdlib_modules or base_module in common_third_party

    def _find_project_root(self, start_path: Path) -> Path:
        """Find the project root by looking for pyproject.toml or requirements.txt."""
        current = start_path
        for _ in range(10):  # Prevent infinite loops
            if (current / "pyproject.toml").exists() or (
                current / "requirements.txt"
            ).exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return start_path

    def _is_in_requirements(self, module_name: str, project_root: Path) -> bool:
        """Check if a module is listed in requirements files."""
        # Check pyproject.toml dependencies
        pyproject_file = project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomllib

                with open(pyproject_file, "rb") as f:
                    data = tomllib.load(f)
                    dependencies = data.get("project", {}).get("dependencies", [])
                    for dep in dependencies:
                        dep_name = (
                            dep.split()[0]
                            .split(">=")[0]
                            .split("==")[0]
                            .split("~=")[0]
                            .split("<")[0]
                        )
                        if dep_name.lower() == module_name.lower():
                            return True
            except Exception:
                pass

        # Check requirements.txt
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip().split("#")[0]  # Remove comments
                        if line:
                            dep_name = (
                                line.split()[0]
                                .split(">=")[0]
                                .split("==")[0]
                                .split("~=")[0]
                                .split("<")[0]
                            )
                            if dep_name.lower() == module_name.lower():
                                return True
            except Exception:
                pass

        return False
