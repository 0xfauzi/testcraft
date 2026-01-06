from __future__ import annotations

from pathlib import Path

from testcraft.application.generation.services.import_resolver import ImportResolver

from .import_resolver.namespace_fixtures import create_namespace_project


def _resolved_sys_roots(result: dict[str, object]) -> set[Path]:
    return {Path(root).resolve() for root in result["sys_path_roots"]}


def test_resolve_handles_tests_namespace_layout(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)
    resolver = ImportResolver()

    result = resolver.resolve(project["test_file"])

    assert (
        result["target_import"]
        == "import tests.src.weather_collector.test_config as _under_test"
    )

    sys_roots = _resolved_sys_roots(result)
    assert project["src_source_root"].resolve() in sys_roots
    assert project["tests_source_root"].resolve() in sys_roots

    assert result["needs_bootstrap"] is True
    assert "sys.path" in result["bootstrap_conftest"]


def test_resolve_preserves_source_imports(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)
    resolver = ImportResolver()

    result = resolver.resolve(project["src_root"] / "config.py")

    assert result["target_import"] == "import weather_collector.config as _under_test"

    sys_roots = _resolved_sys_roots(result)
    assert project["src_source_root"].resolve() in sys_roots
    assert project["tests_source_root"].resolve() in sys_roots

    assert result["needs_bootstrap"] is True
    assert "sys.path" in result["bootstrap_conftest"]
