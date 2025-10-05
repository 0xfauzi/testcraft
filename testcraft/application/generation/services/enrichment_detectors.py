"""
Context enrichment detectors service.

Contains detection methods for various context enrichment features including
environment variables, client boundaries, pytest fixtures, and side effects.
"""

from __future__ import annotations

import ast
import logging
import re
import signal
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None

logger = logging.getLogger(__name__)


# Constants for performance optimization
MAX_FILE_SIZE = 1024 * 1024  # 1MB limit
SCAN_TIMEOUT = 30  # seconds
SUPPORTED_ENCODINGS = ["utf-8", "utf-16", "latin-1"]


@contextmanager
def timeout_context(seconds: int) -> Generator[None, None, None]:
    """Context manager for timeout handling."""

    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def detect_file_encoding(file_path: Path) -> str:
    """
    Detect file encoding using chardet if available, fallback to trial-and-error.

    Args:
        file_path: Path to the file to detect encoding for

    Returns:
        Detected encoding string
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return "utf-8"

    # Try chardet first if available
    if HAS_CHARDET:
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(1024)  # Read first 1KB for detection
                result = chardet.detect(raw_data)
                if (
                    result
                    and result.get("encoding")
                    and result.get("confidence", 0) > 0.7
                ):
                    encoding = result["encoding"]
                    if encoding and encoding.lower() in SUPPORTED_ENCODINGS:
                        return encoding
        except Exception:
            pass

    # Fallback to trial-and-error
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with open(file_path, encoding=encoding) as f:
                f.read(100)  # Try to read first 100 chars
                return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Last resort fallback
    return "utf-8"


def safe_read_file(file_path: Path, max_size: int = MAX_FILE_SIZE) -> str | None:
    """
    Safely read a file with encoding detection and size limits.

    Args:
        file_path: Path to the file to read
        max_size: Maximum file size in bytes

    Returns:
        File contents as string, or None if file cannot be read
    """
    try:
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size:
            logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
            return None

        # Detect encoding and read
        encoding = detect_file_encoding(file_path)
        with open(file_path, encoding=encoding) as f:
            return f.read()

    except (OSError, UnicodeDecodeError, UnicodeError, PermissionError) as e:
        logger.debug(f"Could not read file {file_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error reading file {file_path}: {e}")
        return None


def extract_string_literals_from_ast(node: ast.AST) -> set[str]:
    """Extract all string literals from an AST node."""
    strings = set()

    class StringExtractor(ast.NodeVisitor):
        def visit_Str(self, node: ast.Str) -> None:
            strings.add(node.s)

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str):
                strings.add(node.value)

    extractor = StringExtractor()
    extractor.visit(node)
    return strings


class EnrichmentDetectors:
    """
    Service for detecting various context enrichment patterns.

    Provides static methods for detecting environment variable usage,
    database/HTTP client boundaries, pytest fixtures, and side effects
    that may need mocking in tests.
    """

    @staticmethod
    def detect_env_config_usage(
        source_text: str, ast_tree: ast.AST | None, max_vars: int = 20
    ) -> dict[str, list[str]]:
        """
        Detect environment variable and configuration usage patterns using AST and regex.

        Uses AST parsing for function calls and regex for simple patterns to avoid
        ReDoS vulnerabilities while maintaining detection accuracy.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree for parsing
            max_vars: Maximum number of variables to return

        Returns:
            Dictionary with env_vars and config_keys lists
        """
        if not isinstance(source_text, str):
            raise ValueError("source_text must be a string")
        if not isinstance(max_vars, int) or max_vars < 1:
            raise ValueError("max_vars must be a positive integer")

        env_vars: set[str] = set()
        config_keys: set[str] = set()
        errors: list[str] = []

        try:
            # Parse AST for function calls
            if ast_tree:
                env_vars.update(
                    EnrichmentDetectors._extract_env_vars_from_ast(ast_tree)
                )
                config_keys.update(
                    EnrichmentDetectors._extract_config_keys_from_ast(ast_tree)
                )
        except (SyntaxError, ValueError) as e:
            errors.append(f"AST parsing failed: {e}")
        except Exception as e:
            errors.append(f"Unexpected AST error: {e}")

        # Fallback to regex for simple patterns (less vulnerable patterns only)
        try:
            # Environment variable patterns - using word boundaries to prevent ReDoS
            simple_env_patterns = [
                r"\bos\.environ\[[\'\"]([A-Z0-9_]+)[\'\"]\]",
                r"\benviron\[[\'\"]([A-Z0-9_]+)[\'\"]\]",
                r"([A-Z0-9_]+)\s*=\s*os\.environ(?:\.get\([^)]*\))?",
            ]

            for pattern in simple_env_patterns:
                try:
                    matches = re.findall(pattern, source_text, re.IGNORECASE)
                    env_vars.update(matches)
                except re.error as e:
                    errors.append(f"Regex error in env pattern: {e}")

            # Configuration patterns
            simple_config_patterns = [
                r"\bconfig\.get\(\s*[\'\"]([A-Za-z0-9_]+)[\'\"]",
                r"\bsettings\.([A-Z0-9_]+)",
                r"\.env\[[\'\"]([A-Za-z0-9_]+)[\'\"]\]",
                r"\bload_dotenv\(\)",
                r"\bConfig\(\)\.([A-Za-z0-9_]+)",
            ]

            for pattern in simple_config_patterns:
                try:
                    matches = re.findall(pattern, source_text)
                    if matches:
                        if pattern.endswith("()"):  # Special case for dotenv
                            config_keys.add("dotenv_usage")
                        else:
                            config_keys.update(matches)
                except re.error as e:
                    errors.append(f"Regex error in config pattern: {e}")

        except Exception as e:
            errors.append(f"Error in regex detection: {e}")

        # Log errors if any occurred
        if errors:
            logger.warning(f"Errors in env/config detection: {'; '.join(errors)}")

        return {
            "env_vars": sorted(env_vars)[:max_vars],
            "config_keys": sorted(config_keys)[:max_vars],
        }

    @staticmethod
    def _extract_env_vars_from_ast(node: ast.AST) -> set[str]:
        """Extract environment variable names from AST using safe parsing."""
        env_vars = set()

        class EnvVarExtractor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                # Handle os.environ.get('VAR'), os.getenv('VAR')
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "os"
                        and node.func.value.attr == "environ"
                        and node.func.attr == "get"
                    ):
                        # Extract string argument
                        if node.args and isinstance(
                            node.args[0], ast.Str | ast.Constant
                        ):
                            var_name = (
                                node.args[0].value
                                if hasattr(node.args[0], "value")
                                else node.args[0].s
                            )
                            if isinstance(var_name, str):
                                env_vars.add(var_name)

                    elif (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                        and node.func.attr == "getenv"
                    ):
                        # Extract string argument
                        if node.args and isinstance(
                            node.args[0], ast.Str | ast.Constant
                        ):
                            var_name = (
                                node.args[0].value
                                if hasattr(node.args[0], "value")
                                else node.args[0].s
                            )
                            if isinstance(var_name, str):
                                env_vars.add(var_name)

                self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                # Handle os.environ['VAR'] or environ['VAR']
                if isinstance(node.value, ast.Attribute):
                    if (
                        isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "os"
                        and node.value.attr == "environ"
                    ):
                        # Extract string slice
                        if isinstance(node.slice, ast.Str | ast.Constant):
                            var_name = (
                                node.slice.value
                                if hasattr(node.slice, "value")
                                else node.slice.s
                            )
                            if isinstance(var_name, str):
                                env_vars.add(var_name)

                elif isinstance(node.value, ast.Name) and node.value.id == "environ":
                    # Extract string slice
                    if isinstance(node.slice, ast.Str | ast.Constant):
                        var_name = (
                            node.slice.value
                            if hasattr(node.slice, "value")
                            else node.slice.s
                        )
                        if isinstance(var_name, str):
                            env_vars.add(var_name)

                self.generic_visit(node)

        extractor = EnvVarExtractor()
        extractor.visit(node)
        return env_vars

    @staticmethod
    def _extract_config_keys_from_ast(node: ast.AST) -> set[str]:
        """Extract configuration keys from AST using safe parsing."""
        config_keys = set()

        class ConfigExtractor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                # Handle config.get('key'), settings.KEY, Config().KEY
                if isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "get"
                        and len(node.args) >= 1
                        and isinstance(node.args[0], ast.Str | ast.Constant)
                    ):
                        # Check if it's config.get() or similar
                        if isinstance(
                            node.func.value, ast.Name
                        ) and node.func.value.id in ("config", "settings"):
                            key = (
                                node.args[0].value
                                if hasattr(node.args[0], "value")
                                else node.args[0].s
                            )
                            if isinstance(key, str):
                                config_keys.add(key)

                    # Handle Config().KEY
                    elif (
                        isinstance(node.func.value, ast.Call)
                        and isinstance(node.func.value.func, ast.Name)
                        and node.func.value.func.id == "Config"
                    ):
                        if isinstance(node.func, ast.Attribute):
                            config_keys.add(node.func.attr)

                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                # Handle settings.KEY, config.KEY patterns
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in ("config", "settings")
                    and isinstance(node.attr, str)
                ):
                    config_keys.add(node.attr)

                self.generic_visit(node)

        extractor = ConfigExtractor()
        extractor.visit(node)
        return config_keys

    @staticmethod
    def detect_client_boundaries(
        source_text: str, ast_tree: ast.AST | None
    ) -> dict[str, list[str]]:
        """
        Detect database and HTTP client boundary patterns using AST and regex.

        Uses AST parsing for imports and function calls, regex for pattern matching
        to avoid ReDoS vulnerabilities.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree for parsing

        Returns:
            Dictionary with database_clients and http_clients lists
        """
        if not isinstance(source_text, str):
            raise ValueError("source_text must be a string")

        db_clients: set[str] = set()
        http_clients: set[str] = set()
        errors: list[str] = []

        try:
            # Parse AST for imports and function calls
            if ast_tree:
                db_clients.update(
                    EnrichmentDetectors._extract_db_clients_from_ast(ast_tree)
                )
                http_clients.update(
                    EnrichmentDetectors._extract_http_clients_from_ast(ast_tree)
                )
        except (SyntaxError, ValueError) as e:
            errors.append(f"AST parsing failed: {e}")
        except Exception as e:
            errors.append(f"Unexpected AST error: {e}")

        # Fallback to regex for simple patterns (safe patterns only)
        try:
            # Database patterns - using word boundaries to prevent ReDoS
            db_patterns = [
                r"\bsqlite3\.connect\b",
                r"\bpsycopg2\.connect\b",
                r"\basyncpg\.connect\b",
                r"\bpymysql\.connect\b",
                r"\bcreate_engine\b",
                r"\bdjango\.db\b",
                r"\bredis\.Redis\b",
                r"\bpymongo\.",
                r"\bmotor\.",
            ]

            for pattern in db_patterns:
                try:
                    if re.search(pattern, source_text, re.IGNORECASE):
                        # Map patterns to client families
                        if "sqlite3" in pattern:
                            db_clients.add("sqlite3")
                        elif "psycopg2" in pattern:
                            db_clients.add("psycopg2")
                        elif "asyncpg" in pattern:
                            db_clients.add("asyncpg")
                        elif "pymysql" in pattern:
                            db_clients.add("pymysql")
                        elif "create_engine" in pattern:
                            db_clients.add("sqlalchemy")
                        elif "django.db" in pattern:
                            db_clients.add("django")
                        elif "redis.Redis" in pattern:
                            db_clients.add("redis")
                        elif "pymongo" in pattern or "motor" in pattern:
                            db_clients.add("mongodb")
                except re.error as e:
                    errors.append(f"Regex error in db pattern: {e}")

            # HTTP client patterns
            http_patterns = [
                r"\brequests\.",
                r"\bhttpx\.",
                r"\baiohttp\.",
                r"\burllib\.request\b",
                r"\bpycurl\.",
            ]

            for pattern in http_patterns:
                try:
                    if re.search(pattern, source_text, re.IGNORECASE):
                        # Map patterns to client families
                        if "requests" in pattern:
                            http_clients.add("requests")
                        elif "httpx" in pattern:
                            http_clients.add("httpx")
                        elif "aiohttp" in pattern:
                            http_clients.add("aiohttp")
                        elif "urllib" in pattern:
                            http_clients.add("urllib")
                        elif "pycurl" in pattern:
                            http_clients.add("pycurl")
                except re.error as e:
                    errors.append(f"Regex error in http pattern: {e}")

        except Exception as e:
            errors.append(f"Error in regex detection: {e}")

        # Log errors if any occurred
        if errors:
            logger.warning(f"Errors in client boundary detection: {'; '.join(errors)}")

        return {
            "database_clients": sorted(db_clients)[:10],
            "http_clients": sorted(http_clients)[:10],
        }

    @staticmethod
    def _extract_db_clients_from_ast(node: ast.AST) -> set[str]:
        """Extract database client usage from AST."""
        db_clients = set()

        class DBClientExtractor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name in ("sqlite3", "psycopg2", "asyncpg", "pymysql"):
                        db_clients.add(alias.name)
                    elif alias.name == "sqlalchemy":
                        db_clients.add("sqlalchemy")
                    elif alias.name == "redis":
                        db_clients.add("redis")

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module in ("django.db", "pymongo", "motor"):
                    if "django.db" in node.module:
                        db_clients.add("django")
                    elif "pymongo" in node.module or "motor" in node.module:
                        db_clients.add("mongodb")

            def visit_Call(self, node: ast.Call) -> None:
                # Detect create_engine calls (SQLAlchemy)
                if isinstance(node.func, ast.Name) and node.func.id == "create_engine":
                    db_clients.add("sqlalchemy")

                self.generic_visit(node)

        extractor = DBClientExtractor()
        extractor.visit(node)
        return db_clients

    @staticmethod
    def _extract_http_clients_from_ast(node: ast.AST) -> set[str]:
        """Extract HTTP client usage from AST."""
        http_clients = set()

        class HTTPClientExtractor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name in (
                        "requests",
                        "httpx",
                        "aiohttp",
                        "urllib",
                        "pycurl",
                    ):
                        http_clients.add(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module and any(
                    client in node.module
                    for client in ["requests", "httpx", "aiohttp", "urllib", "pycurl"]
                ):
                    if "requests" in node.module:
                        http_clients.add("requests")
                    elif "httpx" in node.module:
                        http_clients.add("httpx")
                    elif "aiohttp" in node.module:
                        http_clients.add("aiohttp")
                    elif "urllib" in node.module:
                        http_clients.add("urllib")
                    elif "pycurl" in node.module:
                        http_clients.add("pycurl")

        extractor = HTTPClientExtractor()
        extractor.visit(node)
        return http_clients

    @staticmethod
    def discover_comprehensive_fixtures(
        project_root: Path, max_fixtures: int = 15
    ) -> dict[str, Any]:
        """
        Discover comprehensive pytest fixture information with performance optimizations.

        Args:
            project_root: Root path of the project to scan
            max_fixtures: Maximum number of fixtures per category

        Returns:
            Dictionary with builtin, custom, and third_party fixture lists
        """
        if not isinstance(project_root, Path):
            raise ValueError("project_root must be a Path object")
        if not project_root.exists() or not project_root.is_dir():
            raise ValueError("project_root must be an existing directory")
        if not isinstance(max_fixtures, int) or max_fixtures < 1:
            raise ValueError("max_fixtures must be a positive integer")

        builtin_fixtures = [
            "tmp_path",
            "tmp_path_factory",
            "tmpdir",
            "tmpdir_factory",
            "request",
            "config",
            "record_property",
            "record_testsuite_property",
            "capfd",
            "capfdbinary",
            "caplog",
            "capsys",
            "capsysbinary",
            "monkeypatch",
            "pytestconfig",
            "cache",
        ]

        custom_fixtures: dict[str, str] = {}  # name -> scope
        third_party_fixtures: set[str] = set()
        errors: list[str] = []
        files_processed = 0

        try:
            # Use timeout context for the entire operation
            with timeout_context(SCAN_TIMEOUT):
                # Scan test files for custom fixtures with early termination
                patterns = ["**/conftest.py", "**/test_*.py", "**/*_test.py"]

                for pattern in patterns:
                    if len(custom_fixtures) >= max_fixtures:
                        break

                    try:
                        matching_files = list(project_root.rglob(pattern))
                        for test_file in matching_files:
                            if len(custom_fixtures) >= max_fixtures:
                                break

                            # Check file size before processing
                            if test_file.stat().st_size > MAX_FILE_SIZE:
                                continue

                            files_processed += 1

                            # Use safe file reading with encoding detection
                            content = safe_read_file(test_file)
                            if content is None:
                                continue

                            try:
                                # Parse AST for better fixture detection
                                try:
                                    tree = ast.parse(content)
                                    custom_fixtures.update(
                                        EnrichmentDetectors._extract_fixtures_from_ast(
                                            tree
                                        )
                                    )
                                except (SyntaxError, ValueError):
                                    # Fallback to regex if AST parsing fails
                                    fixture_matches = re.findall(
                                        r"@pytest\.fixture(?:\((.*?)\))?\s*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                                        content,
                                        re.DOTALL,
                                    )

                                    for args, name in fixture_matches:
                                        # Extract scope if present
                                        scope = "function"  # default
                                        if args and "scope=" in args:
                                            scope_match = re.search(
                                                r'scope=[\'"](\w+)[\'"]', args
                                            )
                                            if scope_match:
                                                scope = scope_match.group(1)
                                        custom_fixtures[name] = scope

                                # Detect third-party fixtures by imports (AST-based)
                                try:
                                    tree = ast.parse(content)
                                    third_party_fixtures.update(
                                        EnrichmentDetectors._extract_third_party_fixtures_from_ast(
                                            tree
                                        )
                                    )
                                except (SyntaxError, ValueError):
                                    # Fallback to simple string matching
                                    if (
                                        "pytest-django" in content
                                        or "django_db" in content
                                    ):
                                        third_party_fixtures.update(
                                            [
                                                "db",
                                                "django_db",
                                                "client",
                                                "admin_client",
                                            ]
                                        )
                                    if (
                                        "pytest-asyncio" in content
                                        or "event_loop" in content
                                    ):
                                        third_party_fixtures.add("event_loop")
                                    if "pytest-httpx" in content:
                                        third_party_fixtures.add("httpx_mock")

                            except Exception as e:
                                errors.append(f"Error processing {test_file}: {e}")
                                continue

                    except Exception as e:
                        errors.append(f"Error scanning pattern {pattern}: {e}")
                        continue

        except TimeoutError as e:
            logger.warning(f"Fixture discovery timed out: {e}")
            errors.append("Scan timeout reached")
        except Exception as e:
            errors.append(f"Error discovering fixtures: {e}")

        # Log performance and error summary
        if errors:
            logger.warning(
                f"Fixture discovery completed with {len(errors)} errors. "
                f"Processed {files_processed} files."
            )

        # Apply limits to results
        max_per_category = max_fixtures // 3
        return {
            "builtin": builtin_fixtures[:max_per_category],
            "custom": dict(list(custom_fixtures.items())[:max_per_category]),
            "third_party": sorted(third_party_fixtures)[:max_per_category],
            "_errors": errors,  # Include errors for debugging
            "_files_processed": files_processed,
        }

    @staticmethod
    def _extract_fixtures_from_ast(node: ast.AST) -> dict[str, str]:
        """Extract pytest fixture definitions from AST."""
        fixtures = {}

        class FixtureExtractor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Look for @pytest.fixture decorators
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "pytest"
                        and decorator.func.attr == "fixture"
                    ):
                        # Extract scope from decorator arguments
                        scope = "function"  # default
                        if decorator.args:
                            for arg in decorator.args:
                                if isinstance(arg, ast.Str) and arg.s == "scope":
                                    # Look for scope in keywords
                                    pass
                        # Check keyword arguments
                        for keyword in decorator.keywords:
                            if keyword.arg == "scope" and isinstance(
                                keyword.value, ast.Str
                            ):
                                scope = keyword.value.s
                        fixtures[node.name] = scope

                self.generic_visit(node)

        extractor = FixtureExtractor()
        extractor.visit(node)
        return fixtures

    @staticmethod
    def _extract_third_party_fixtures_from_ast(node: ast.AST) -> set[str]:
        """Extract third-party fixture usage from AST."""
        third_party_fixtures = set()

        class ThirdPartyExtractor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name in (
                        "pytest_django",
                        "pytest_asyncio",
                        "pytest_httpx",
                    ):
                        if "django" in alias.name:
                            third_party_fixtures.update(
                                ["db", "django_db", "client", "admin_client"]
                            )
                        elif "asyncio" in alias.name:
                            third_party_fixtures.add("event_loop")
                        elif "httpx" in alias.name:
                            third_party_fixtures.add("httpx_mock")

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module in ("pytest_django", "pytest_asyncio", "pytest_httpx"):
                    if "django" in node.module:
                        third_party_fixtures.update(
                            ["db", "django_db", "client", "admin_client"]
                        )
                    elif "asyncio" in node.module:
                        third_party_fixtures.add("event_loop")
                    elif "httpx" in node.module:
                        third_party_fixtures.add("httpx_mock")

        extractor = ThirdPartyExtractor()
        extractor.visit(node)
        return third_party_fixtures

    @staticmethod
    def detect_side_effect_boundaries(
        source_text: str, ast_tree: ast.AST | None
    ) -> dict[str, list[str]]:
        """
        Detect side-effect boundaries that may need mocking using AST and regex.

        Uses AST parsing for imports and function calls, regex for pattern matching
        to identify potential side effects that require mocking in tests.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree for parsing

        Returns:
            Dictionary mapping side-effect categories to detected patterns
        """
        if not isinstance(source_text, str):
            raise ValueError("source_text must be a string")

        side_effects: dict[str, set[str]] = {
            "filesystem": set(),
            "network": set(),
            "time": set(),
            "process": set(),
            "random": set(),
            "system": set(),
        }
        errors: list[str] = []

        try:
            # Parse AST for imports and function calls
            if ast_tree:
                ast_results = EnrichmentDetectors._extract_side_effects_from_ast(
                    ast_tree
                )
                for category, effects in ast_results.items():
                    side_effects[category].update(effects)

        except (SyntaxError, ValueError) as e:
            errors.append(f"AST parsing failed: {e}")
        except Exception as e:
            errors.append(f"Unexpected AST error: {e}")

        # Fallback to regex for simple patterns (safe patterns only)
        try:
            # Pattern families for side-effects - using word boundaries to prevent ReDoS
            patterns = {
                "filesystem": [
                    r"\bopen\s*\(",
                    r"\bpathlib\.Path\b",
                    r"\bos\.path\.",
                    r"\bshutil\.",
                    r"\bglob\.",
                    r"\btempfile\.",
                ],
                "network": [
                    r"\brequests\.",
                    r"\bhttpx\.",
                    r"\burllib\.",
                    r"\bsocket\.",
                    r"\baiohttp\.",
                ],
                "time": [
                    r"\btime\.(sleep|time)\b",
                    r"\bdatetime\.(now|utcnow)\b",
                    r"\btimezone\.",
                    r"\bsleep\(",
                ],
                "process": [
                    r"\bsubprocess\.",
                    r"\bos\.system\b",
                    r"\bpopen\(",
                    r"\bos\.fork\b",
                    r"\bmultiprocessing\.",
                ],
                "random": [
                    r"\brandom\.",
                    r"\buuid\.",
                    r"\bsecrets\.",
                    r"\bnumpy\.random\b",
                    r"\bchoice\(",
                ],
                "system": [
                    r"\bos\.environ\b",
                    r"\bsys\.",
                    r"\bplatform\.",
                    r"\bos\.getcwd\b",
                    r"\bos\.chdir\b",
                ],
            }

            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    try:
                        matches = re.findall(pattern, source_text, re.IGNORECASE)
                        if matches:
                            # Keep only first few matches to avoid noise
                            side_effects[category].update(matches[:5])
                    except re.error as e:
                        errors.append(f"Regex error in {category} pattern: {e}")

        except Exception as e:
            errors.append(f"Error in regex detection: {e}")

        # Log errors if any occurred
        if errors:
            logger.warning(f"Errors in side-effect detection: {'; '.join(errors)}")

        # Convert sets to sorted lists and limit size
        return {
            category: sorted(effects)[:5]
            for category, effects in side_effects.items()
            if effects
        }

    @staticmethod
    def _extract_side_effects_from_ast(node: ast.AST) -> dict[str, set[str]]:
        """Extract side effect patterns from AST."""
        side_effects: dict[str, set[str]] = {
            "filesystem": set(),
            "network": set(),
            "time": set(),
            "process": set(),
            "random": set(),
            "system": set(),
        }

        class SideEffectExtractor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in ("os", "sys", "platform"):
                        side_effects["system"].add(module)
                    elif module in ("time", "datetime"):
                        side_effects["time"].add(module)
                    elif module in ("random", "uuid", "secrets"):
                        side_effects["random"].add(module)
                    elif module in ("subprocess", "multiprocessing"):
                        side_effects["process"].add(module)
                    elif module in ("socket",):
                        side_effects["network"].add(module)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    module = node.module.split(".")[0]
                    if module in (
                        "os",
                        "sys",
                        "platform",
                        "pathlib",
                        "shutil",
                        "glob",
                        "tempfile",
                    ):
                        if module == "os":
                            side_effects["system"].add("os")
                        elif module in ("pathlib", "shutil", "glob", "tempfile"):
                            side_effects["filesystem"].add(module)
                    elif module in ("time", "datetime", "timezone"):
                        side_effects["time"].add(module)
                    elif module in ("random", "uuid", "secrets"):
                        side_effects["random"].add(module)
                    elif module in ("subprocess", "multiprocessing"):
                        side_effects["process"].add(module)
                    elif module in ("requests", "httpx", "urllib", "socket", "aiohttp"):
                        side_effects["network"].add(module)

            def visit_Call(self, node: ast.Call) -> None:
                # Detect specific function calls that indicate side effects
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("open", "sleep"):
                        if node.func.id == "open":
                            side_effects["filesystem"].add("open")
                        elif node.func.id == "sleep":
                            side_effects["time"].add("sleep")

                elif isinstance(node.func, ast.Attribute):
                    # Handle os.system, os.getcwd, etc.
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == "os":
                            if node.func.attr in ("system", "getcwd", "chdir"):
                                side_effects["system"].add(f"os.{node.func.attr}")
                            elif node.func.attr == "path":
                                side_effects["filesystem"].add("os.path")
                        elif node.func.value.id in ("time", "datetime"):
                            side_effects["time"].add(
                                f"{node.func.value.id}.{node.func.attr}"
                            )
                        elif node.func.value.id == "random":
                            side_effects["random"].add(f"random.{node.func.attr}")

                self.generic_visit(node)

        extractor = SideEffectExtractor()
        extractor.visit(node)
        return side_effects
