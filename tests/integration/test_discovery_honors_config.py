from pathlib import Path

from testcraft.adapters.io.file_discovery import FileDiscoveryService
from testcraft.config.models import TestPatternConfig


def test_discovery_includes_src_pkg_with_minimal_excludes(tmp_path: Path):
    project = tmp_path
    src = project / "src" / "pkg"
    src.mkdir(parents=True)

    file_a = src / "a.py"
    file_a.write_text("def foo():\n    return 1\n", encoding="utf-8")

    cfg = TestPatternConfig(
        exclude_dirs=[
            "venv",
            ".venv",
            "env",
            "node_modules",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "htmlcov",
            "dist",
            "build",
            ".tox",
            "site-packages",
        ]
    )

    discovery = FileDiscoveryService(config=cfg)
    files = discovery.discover_source_files(
        project, file_patterns=["*.py"], include_test_files=False
    )

    assert str(file_a.resolve()) in files


def test_discovery_skips_virtualenv_trees(tmp_path: Path):
    project = tmp_path

    src_dir = project / "src" / "weather_collector"
    src_dir.mkdir(parents=True)
    app_file = src_dir / "app.py"
    app_file.write_text("def run():\n    return True\n", encoding="utf-8")

    venv_bin = project / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_file = venv_bin / "activate_this.py"
    venv_file.write_text("print('venv')\n", encoding="utf-8")

    config = TestPatternConfig()
    discovery = FileDiscoveryService(config=config)

    files = discovery.discover_source_files(
        project, file_patterns=["*.py"], include_test_files=False
    )

    assert str(app_file.resolve()) in files
    assert str(venv_file.resolve()) not in files

    filtered = discovery.filter_existing_files([venv_file], project)
    assert filtered == []
