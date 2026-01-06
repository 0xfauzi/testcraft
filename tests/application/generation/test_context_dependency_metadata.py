from __future__ import annotations

from textwrap import dedent

from testcraft.application.generation.services.context_pack import ContextPackBuilder


def test_collect_dependency_metadata_from_manifests(tmp_path) -> None:
    pyproject_content = dedent(
        """
        [project]
        dependencies = [
          "requests>=2.0",
          "rich==13.0.0",
        ]

        [project.optional-dependencies]
        dev = [
          "pytest>=7.0",
        ]

        [dependency-groups]
        docs = ["mkdocs>=1.6.0"]
        """
    ).strip()
    (tmp_path / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    requirements_content = dedent(
        """
        numpy==1.25.0
        # comment
        sqlalchemy-utils==0.41.0
        uvicorn>=0.23
        -e git+https://example.com/ignored.git#egg=ignored
        """
    ).strip()
    (tmp_path / "requirements.txt").write_text(requirements_content, encoding="utf-8")

    builder = ContextPackBuilder()
    metadata = builder._collect_dependency_metadata(tmp_path)

    expected_distributions = {
        "requests": ">=2.0",
        "rich": "==13.0.0",
        "pytest": ">=7.0",
        "mkdocs": ">=1.6.0",
        "numpy": "==1.25.0",
        "sqlalchemy-utils": "==0.41.0",
        "uvicorn": ">=0.23",
    }

    assert metadata.distributions == expected_distributions

    expected_modules = {
        "requests",
        "rich",
        "pytest",
        "mkdocs",
        "numpy",
        "sqlalchemy-utils",
        "uvicorn",
    }
    # External modules include normalized (underscore) variants as well.
    assert expected_modules.issubset(set(metadata.external_modules))

    cached = builder._collect_dependency_metadata(tmp_path)
    assert cached is metadata


def test_collect_dependency_metadata_without_manifests(tmp_path) -> None:
    builder = ContextPackBuilder()
    metadata = builder._collect_dependency_metadata(tmp_path)

    assert metadata.distributions == {}
    assert metadata.external_modules == []
