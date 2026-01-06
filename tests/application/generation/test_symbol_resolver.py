from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT_DIR)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import textwrap

import pytest

pytest.importorskip("pydantic")

from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.application.generation.services.symbol_resolver import SymbolResolver


class _StubImportResolver:
    """Minimal import resolver for tests that exposes sys.path roots."""

    def __init__(self, sys_path_roots: list[Path]) -> None:
        self._roots = [str(root) for root in sys_path_roots]

    def resolve(self, file_path: Path):  # pragma: no cover - simple stub
        return {
            "target_import": "",
            "sys_path_roots": self._roots,
            "needs_bootstrap": False,
            "bootstrap_conftest": "",
        }


def _write_module(base: Path, relative: str, content: str) -> Path:
    module_path = base / relative
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return module_path


def test_resolves_symbol_imported_from_sibling_module(tmp_path: Path) -> None:
    project_root = tmp_path / "weather_project"
    package_root = project_root / "weather_collector"
    package_root.mkdir(parents=True)

    # Ensure package structure is valid for imports
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    _write_module(
        package_root,
        "api_client.py",
        """
        class WeatherAPIClient:
            def fetch(self) -> None:
                pass
        """,
    )

    _write_module(
        package_root,
        "scheduler.py",
        """
        from .api_client import WeatherAPIClient


        class WeatherScheduler:
            def schedule(self) -> None:
                return WeatherAPIClient()
        """,
    )

    parser = CodebaseParser()
    resolver = SymbolResolver(
        parser_port=parser,
        import_resolver=_StubImportResolver([project_root]),
    )

    result = resolver.resolve_single_symbol(
        "weather_collector.scheduler.WeatherAPIClient",
        project_root=project_root,
    )

    assert result is not None
    assert result.kind == "class"
    assert result.signature.startswith("class WeatherAPIClient")


def test_runtime_fallback_handles_dynamic_symbol(tmp_path: Path) -> None:
    project_root = tmp_path / "dynamic_project"
    package_root = project_root / "weather_collector"
    package_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")

    _write_module(
        package_root,
        "dynamic_helpers.py",
        """
        make_client = lambda: None  # noqa: E731 - intentionally dynamic assignment
        """,
    )

    _write_module(
        package_root,
        "scheduler.py",
        """
        from .dynamic_helpers import make_client


        def get_client():
            return make_client()
        """,
    )

    parser = CodebaseParser()
    resolver = SymbolResolver(
        parser_port=parser,
        import_resolver=_StubImportResolver([project_root]),
        allow_runtime_imports=True,
    )

    result = resolver.resolve_single_symbol(
        "weather_collector.scheduler.make_client",
        project_root=project_root,
    )

    assert result is not None
    assert result.kind == "func"
    assert result.doc is None or isinstance(result.doc, str)
    assert "def" in result.signature


def test_resolver_skips_known_external_modules(tmp_path: Path) -> None:
    project_root = tmp_path / "sample_project"
    project_root.mkdir(parents=True)

    parser = CodebaseParser()
    resolver = SymbolResolver(
        parser_port=parser,
        import_resolver=_StubImportResolver([project_root]),
    )

    resolver.update_environment_modules(external_modules={"external_lib"})

    result = resolver.resolve_single_symbol("external_lib.module.Helper")

    assert result is None


def test_resolver_skips_logger_singleton(tmp_path: Path) -> None:
    project_root = tmp_path / "dummy"
    project_root.mkdir(parents=True)

    module_path = project_root / "module.py"
    module_path.write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n",
        encoding="utf-8",
    )

    parser = CodebaseParser()
    resolver = SymbolResolver(
        parser_port=parser,
        import_resolver=_StubImportResolver([project_root]),
    )

    result = resolver.resolve_single_symbol("module.logger", project_root=project_root)

    assert result is None
