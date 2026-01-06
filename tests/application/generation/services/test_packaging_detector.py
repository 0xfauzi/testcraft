from __future__ import annotations

from pathlib import Path

from testcraft.application.generation.services.packaging_detector import (
    PackagingDetector,
)

from .import_resolver.namespace_fixtures import create_namespace_project


def test_detect_packaging_includes_tests_namespace_root(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)

    packaging_info = PackagingDetector.detect_packaging(project["project_root"])

    resolved_roots = {root.resolve() for root in packaging_info.source_roots}

    assert project["tests_source_root"].resolve() in resolved_roots
    assert (
        packaging_info.get_canonical_import(project["test_file"])
        == project["tests_module_import"]
    )
    assert "tests." not in packaging_info.disallowed_import_prefixes
    assert packaging_info.is_import_allowed(project["tests_module_import"])


def test_detect_packaging_returns_expected_source_module(tmp_path: Path) -> None:
    project = create_namespace_project(tmp_path)

    packaging_info = PackagingDetector.detect_packaging(project["project_root"])

    resolved_roots = {root.resolve() for root in packaging_info.source_roots}

    assert project["src_source_root"].resolve() in resolved_roots
    canonical_import = packaging_info.get_canonical_import(
        project["src_root"] / "config.py"
    )
    assert canonical_import == project["source_module_import"]
