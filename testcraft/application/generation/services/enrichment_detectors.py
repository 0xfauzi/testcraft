"""
Context enrichment detectors service.

Contains detection methods for various context enrichment features including
environment variables, client boundaries, pytest fixtures, and side effects.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        Detect environment variable and configuration usage patterns.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree (unused in current implementation)
            max_vars: Maximum number of variables to return

        Returns:
            Dictionary with env_vars and config_keys lists
        """
        env_vars: set[str] = set()
        config_keys: set[str] = set()

        try:
            # Environment variable patterns
            env_patterns = [
                r"os\.environ\[[\'\"]([A-Z0-9_]+)[\'\"]\]",
                r"os\.environ\.get\([\'\"]([A-Z0-9_]+)[\'\"]",
                r"os\.getenv\([\'\"]([A-Z0-9_]+)[\'\"]",
                r"getenv\([\'\"]([A-Z0-9_]+)[\'\"]",
                r"environ\[[\'\"]([A-Z0-9_]+)[\'\"]\]",
                r"([A-Z0-9_]+)\s*=.*os\.environ",
            ]

            for pattern in env_patterns:
                matches = re.findall(pattern, source_text)
                env_vars.update(matches)

            # Configuration patterns
            config_patterns = [
                r"config\.get\(\s*[\'\"]([A-Za-z0-9_]+)[\'\"]",
                r"settings\.([A-Z0-9_]+)",
                r"\.env\[[\'\"]([A-Za-z0-9_]+)[\'\"]\]",
                r"load_dotenv\(\)",  # Indicates dotenv usage
                r"Config\(\)\.([A-Za-z0-9_]+)",
            ]

            for pattern in config_patterns:
                matches = re.findall(pattern, source_text)
                if matches:
                    if pattern.endswith("()"):  # Special case for dotenv
                        config_keys.add("dotenv_usage")
                    else:
                        config_keys.update(matches)

        except Exception as e:
            logger.debug("Error detecting env/config usage: %s", e)

        return {
            "env_vars": sorted(env_vars)[:max_vars],
            "config_keys": sorted(config_keys)[:max_vars],
        }

    @staticmethod
    def detect_client_boundaries(
        source_text: str, ast_tree: ast.AST | None
    ) -> dict[str, list[str]]:
        """
        Detect database and HTTP client boundary patterns.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree (unused in current implementation)

        Returns:
            Dictionary with database_clients and http_clients lists
        """
        db_clients: set[str] = set()
        http_clients: set[str] = set()

        try:
            # Database patterns
            db_patterns = {
                "sqlite3": [r"sqlite3\.connect", r"sqlite3\.Connection"],
                "psycopg2": [r"psycopg2\.connect", r"psycopg2\.pool"],
                "asyncpg": [r"asyncpg\.connect", r"asyncpg\.create_pool"],
                "pymysql": [r"pymysql\.connect", r"pymysql\.Connection"],
                "sqlalchemy": [r"create_engine", r"Session\(\)", r"sessionmaker"],
                "django": [r"django\.db", r"models\.(Model|Manager)"],
                "redis": [r"redis\.Redis", r"redis\.StrictRedis"],
                "mongodb": [r"pymongo\.", r"motor\."],
            }

            for family, patterns in db_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, source_text, re.IGNORECASE):
                        db_clients.add(family)
                        break

            # HTTP client patterns
            http_patterns = {
                "requests": [r"requests\.", r"Session\(\)"],
                "httpx": [r"httpx\.", r"AsyncClient", r"Client\(\)"],
                "aiohttp": [r"aiohttp\.", r"ClientSession"],
                "urllib": [r"urllib\.request", r"urlopen"],
                "pycurl": [r"pycurl\.", r"curl"],
            }

            for family, patterns in http_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, source_text, re.IGNORECASE):
                        http_clients.add(family)
                        break

        except Exception as e:
            logger.debug("Error detecting client boundaries: %s", e)

        return {
            "database_clients": sorted(db_clients)[:10],
            "http_clients": sorted(http_clients)[:10],
        }

    @staticmethod
    def discover_comprehensive_fixtures(
        project_root: Path, max_fixtures: int = 15
    ) -> dict[str, Any]:
        """
        Discover comprehensive pytest fixture information.

        Args:
            project_root: Root path of the project to scan
            max_fixtures: Maximum number of fixtures per category

        Returns:
            Dictionary with builtin, custom, and third_party fixture lists
        """
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

        try:
            # Scan test files for custom fixtures
            for pattern in ["**/conftest.py", "**/test_*.py", "**/*_test.py"]:
                for test_file in project_root.rglob(pattern):
                    try:
                        content = test_file.read_text(encoding="utf-8")

                        # Find @pytest.fixture decorators
                        fixture_matches = re.findall(
                            r"@pytest\.fixture(?:\((.*?)\))?\s*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                            content,
                            re.DOTALL,
                        )

                        for args, name in fixture_matches:
                            # Extract scope if present
                            scope = "function"  # default
                            if args and "scope=" in args:
                                scope_match = re.search(r'scope=[\'"](\w+)[\'"]', args)
                                if scope_match:
                                    scope = scope_match.group(1)
                            custom_fixtures[name] = scope

                        # Detect third-party fixtures by imports
                        if "pytest-django" in content or "django_db" in content:
                            third_party_fixtures.update(
                                ["db", "django_db", "client", "admin_client"]
                            )
                        if "pytest-asyncio" in content or "event_loop" in content:
                            third_party_fixtures.add("event_loop")
                        if "pytest-httpx" in content:
                            third_party_fixtures.add("httpx_mock")

                    except Exception:
                        continue

        except Exception as e:
            logger.debug("Error discovering fixtures: %s", e)

        return {
            "builtin": builtin_fixtures[: max_fixtures // 3],
            "custom": dict(list(custom_fixtures.items())[: max_fixtures // 3]),
            "third_party": sorted(third_party_fixtures)[: max_fixtures // 3],
        }

    @staticmethod
    def detect_side_effect_boundaries(
        source_text: str, ast_tree: ast.AST | None
    ) -> dict[str, list[str]]:
        """
        Detect side-effect boundaries that may need mocking.

        Args:
            source_text: Source code text to analyze
            ast_tree: Optional AST tree (unused in current implementation)

        Returns:
            Dictionary mapping side-effect categories to detected patterns
        """
        side_effects: dict[str, set[str]] = {
            "filesystem": set(),
            "network": set(),
            "time": set(),
            "process": set(),
            "random": set(),
            "system": set(),
        }

        try:
            # Pattern families for side-effects
            patterns = {
                "filesystem": [
                    r"\bopen\s*\(",
                    r"pathlib\.Path",
                    r"os\.path\.",
                    r"shutil\.",
                    r"glob\.",
                    r"tempfile\.",
                ],
                "network": [
                    r"requests\.",
                    r"httpx\.",
                    r"urllib\.",
                    r"socket\.",
                    r"aiohttp\.",
                ],
                "time": [
                    r"time\.(sleep|time)",
                    r"datetime\.(now|utcnow)",
                    r"timezone\.",
                    r"sleep\(",
                ],
                "process": [
                    r"subprocess\.",
                    r"os\.system",
                    r"popen\(",
                    r"os\.fork",
                    r"multiprocessing\.",
                ],
                "random": [
                    r"random\.",
                    r"uuid\.",
                    r"secrets\.",
                    r"numpy\.random",
                    r"choice\(",
                ],
                "system": [
                    r"os\.environ",
                    r"sys\.",
                    r"platform\.",
                    r"os\.getcwd",
                    r"os\.chdir",
                ],
            }

            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    matches = re.findall(f"({pattern}[a-zA-Z_]*)", source_text)
                    if matches:
                        # Keep only first few matches to avoid noise
                        side_effects[category].update(
                            match[0] if isinstance(match, tuple) else match
                            for match in matches[:5]
                        )

        except Exception as e:
            logger.debug("Error detecting side-effect boundaries: %s", e)

        # Convert sets to sorted lists and limit size
        return {
            category: sorted(effects)[:5]
            for category, effects in side_effects.items()
            if effects
        }
