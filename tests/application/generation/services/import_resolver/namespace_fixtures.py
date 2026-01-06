from __future__ import annotations

import textwrap
from pathlib import Path


def create_namespace_project(
    tmp_path: Path, project_name: str = "weather_project"
) -> dict[str, Path]:
    """Create a sample project that mirrors the weather-collector namespace layout.

    Returns a mapping with handy references to key paths used by tests.
    """

    project_root = tmp_path / project_name
    src_root = project_root / "src" / "weather_collector"
    tests_src_root = project_root / "tests" / "src" / "weather_collector"

    src_root.mkdir(parents=True, exist_ok=True)
    tests_src_root.mkdir(parents=True, exist_ok=True)

    (project_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "weather-collector"

            [tool.pytest.ini_options]
            pythonpath = ["src", "tests/src"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    # Establish primary package
    (src_root / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "config.py").write_text(
        textwrap.dedent(
            """
            class Settings:
                app_name = "Weather Collector"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    test_file = tests_src_root / "test_config.py"
    test_file.write_text(
        textwrap.dedent(
            """
            from weather_collector.config import Settings


            def test_settings_defaults() -> None:
                settings = Settings()
                assert settings.app_name == "Weather Collector"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    test_module_import = ".".join(
        test_file.relative_to(project_root).with_suffix("").parts
    )
    source_module_import = "weather_collector.config"

    return {
        "project_root": project_root,
        "src_root": src_root,
        "src_source_root": src_root.parent,
        "tests_src_root": tests_src_root,
        "tests_source_root": tests_src_root.parent,
        "test_file": test_file,
        "tests_module_import": test_module_import,
        "source_module_import": source_module_import,
    }
