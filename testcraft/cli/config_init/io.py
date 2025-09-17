"""I/O utilities for configuration file management."""

import json
from pathlib import Path
from typing import Any

from ...config.models import TestCraftConfig
from ...ports.ui_port import UIPort


class ConfigIO:
    """Handles configuration file I/O operations."""

    def __init__(self, ui: UIPort):
        """Initialize with UI adapter."""
        self.ui = ui

    def determine_output_path(
        self, format_type: str, output_path: Path | None = None
    ) -> Path:
        """Determine the appropriate output file path."""
        if output_path:
            return output_path

        extensions = {
            "toml": ".testcraft.toml",
            "yaml": ".testcraft.yml", 
            "json": ".testcraft.json",
        }
        return Path(extensions.get(format_type, ".testcraft.toml"))

    def check_file_exists(self, config_file: Path) -> bool:
        """Check if config file exists and handle overwrite confirmation."""
        if not config_file.exists():
            return False

        return self.ui.get_user_confirmation(
            f"Configuration file {config_file} already exists. Overwrite?",
            default=False,
        )

    def write_config_file(
        self, config_file: Path, content: str
    ) -> None:
        """Write configuration content to file."""
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(content)

    def load_existing_config(self, config_file: Path) -> dict[str, Any]:
        """Load existing configuration file by format."""
        suffix = config_file.suffix.lower()
        config_dict: dict[str, Any] = {}

        try:
            if suffix == ".toml":
                import tomllib  # py311+
                with open(config_file, "rb") as f:
                    config_dict = tomllib.load(f) or {}
            elif suffix in (".yml", ".yaml"):
                import yaml
                with open(config_file, encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f) or {}
            elif suffix == ".json":
                with open(config_file, encoding="utf-8") as f:
                    config_dict = json.load(f) or {}
            else:
                # Fallback: treat as TOML
                import tomllib
                with open(config_file, "rb") as f:
                    config_dict = tomllib.load(f) or {}
        except Exception:
            # If load fails, start from an empty dict
            config_dict = {}

        return config_dict

    def write_merged_config(
        self, config_file: Path, merged_config: dict[str, Any]
    ) -> None:
        """Write merged configuration back to file in original format."""
        suffix = config_file.suffix.lower()

        try:
            if suffix == ".toml":
                try:
                    import tomli_w
                    with open(config_file, "wb") as f:
                        tomli_w.dump(merged_config, f)
                except ImportError:
                    # Fallback: simple TOML emitter (minimal, may drop comments)
                    content = self._dict_to_simple_toml(merged_config)
                    with open(config_file, "w", encoding="utf-8") as f:
                        f.write(content)
            elif suffix in (".yml", ".yaml"):
                import yaml
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged_config, f, sort_keys=False)
            elif suffix == ".json":
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(merged_config, f, indent=2)
            else:
                # Default to TOML fallback
                content = self._dict_to_simple_toml(merged_config)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(content)

        except Exception as e:
            raise Exception(f"Failed to write configuration updates: {e}")

    def deep_merge(self, base: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge updates into config."""
        result = base.copy()
        for k, v in inc.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = self.deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _dict_to_simple_toml(self, data: dict[str, Any], prefix: str = "") -> str:
        """Convert dict to simple TOML format (fallback)."""
        lines = []

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                lines.append(f"\n[{full_key}]")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        lines.append(f'{sub_key} = "{sub_value}"')
                    else:
                        lines.append(f"{sub_key} = {sub_value}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, list):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f"{key} = {value}")

        return "\n".join(lines)
