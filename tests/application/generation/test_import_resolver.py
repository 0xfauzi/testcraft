from __future__ import annotations

from pathlib import Path

from testcraft.application.generation.services.import_resolver import ImportResolver

from .services.import_resolver.namespace_fixtures import create_namespace_project


def test_import_resolver_handles_tests_src_namespace(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)

    resolver = ImportResolver()
    import_map = resolver.resolve(project["test_file"])

    assert import_map["target_import"] == (
        f"import {project['tests_module_import']} as _under_test"
    )
    assert import_map["needs_bootstrap"] is False
    assert any(
        str(project["tests_src_root"].resolve()) in root
        or root.endswith("tests/src/weather_collector")
        for root in import_map["sys_path_roots"]
    )


def test_import_resolver_sets_expected_sys_path_roots(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)

    resolver = ImportResolver()
    test_import_map = resolver.resolve(project["test_file"])
    source_import_map = resolver.resolve(project["src_root"] / "config.py")

    assert any(
        root.endswith("tests/src/weather_collector")
        for root in test_import_map["sys_path_roots"]
    )
    assert any(root.endswith("src") for root in source_import_map["sys_path_roots"])


def test_import_resolver_builds_expected_source_import(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)

    resolver = ImportResolver()
    source_import_map = resolver.resolve(project["src_root"] / "config.py")

    assert source_import_map["target_import"] == (
        f"import {project['source_module_import']} as _under_test"
    )
    assert any(
        str(project["src_source_root"].resolve()) in root or root.endswith("src")
        for root in source_import_map["sys_path_roots"]
    )
