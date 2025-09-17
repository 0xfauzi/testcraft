from __future__ import annotations

import tomllib
from pathlib import Path


def get_pytest_settings(source_path: Path) -> list[str]:
    pytest_settings: list[str] = []
    try:
        project_root = source_path.parent
        while project_root.parent != project_root:
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                ini_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
                if isinstance(ini_opts, dict):
                    for k, v in ini_opts.items():
                        pytest_settings.append(f"{k}={v}")
            except Exception:
                pass
    except Exception:
        pass
    return pytest_settings


def get_pytest_settings_context(config: dict, source_path: Path) -> list[str]:
    items: list[str] = []
    try:
        context_cats = config.get("context_categories", {})
        if not context_cats.get("pytest_settings", True):
            return items
        settings = get_pytest_settings(source_path)
        if settings:
            items.append(f"# pytest settings: {settings[:5]}")
    except Exception:
        pass
    return items



