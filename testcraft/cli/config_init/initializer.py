"""Configuration initializer that orchestrates user flow and writes files."""

import logging
from pathlib import Path

from ...config.models import TestCraftConfig
from ...ports.ui_port import UIPort
from .generators import JSONGenerator, TOMLGenerator, YAMLGenerator
from .io import ConfigIO
from .preferences import PreferencesCollector

logger = logging.getLogger(__name__)


class ConfigInitializationError(Exception):
    """Raised when configuration initialization fails."""

    pass


class ConfigInitializer:
    """Initialize configuration files with guided setup."""

    def __init__(self, ui: UIPort):
        """Initialize with UI adapter."""
        self.ui = ui
        self.io = ConfigIO(ui)
        self.preferences = PreferencesCollector(ui)
        self.generators = {
            "toml": TOMLGenerator(),
            "yaml": YAMLGenerator(),
            "json": JSONGenerator(),
        }

    def create_config_file(
        self,
        format_type: str = "toml",
        minimal: bool = False,
        output_path: Path | None = None,
    ) -> Path:
        """
        Create a configuration file in the specified format.

        Args:
            format_type: Format to use ('toml', 'yaml', 'json')
            minimal: Whether to create minimal configuration
            output_path: Custom output path

        Returns:
            Path to created configuration file
        """
        try:
            # Determine output file path
            config_file = self.io.determine_output_path(format_type, output_path)

            # Check if file already exists
            if config_file.exists():
                if not self.io.check_file_exists(config_file):
                    self.ui.display_info(
                        "Configuration initialization cancelled", "Cancelled"
                    )
                    return config_file

            # Generate configuration content
            if minimal:
                content = self._generate_minimal_config(format_type)
            else:
                content = self._generate_comprehensive_config(format_type)

            # Write configuration file
            self.io.write_config_file(config_file, content)

            logger.info(f"Configuration file created: {config_file}")
            return config_file

        except Exception as e:
            raise ConfigInitializationError(f"Failed to create configuration file: {e}")

    def run_guided_setup(self, config_file: Path) -> None:
        """Run guided configuration setup."""
        try:
            self.ui.display_info("Starting guided configuration setup", "Guided Setup")

            # Collect user preferences
            preferences = self.preferences.collect_user_preferences()

            # If TOML, regenerate full commented config with chosen prefs activated (preserve comments)
            suffix = config_file.suffix.lower()
            if suffix == ".toml":
                overrides = self.preferences.preferences_to_overrides(preferences)
                generator = self.generators["toml"]
                content = generator.generate_comprehensive_config(
                    comment_unused=True, overrides=overrides
                )
                self.io.write_config_file(config_file, content)
                self.ui.display_success(
                    f"Configuration updated with your preferences: {config_file}",
                    "Setup Complete",
                )
            else:
                # Fallback to dictionary merge for non-TOML formats
                self._update_config_with_preferences(config_file, preferences)

        except Exception as e:
            self.ui.display_error(f"Guided setup failed: {e}", "Setup Error")

    def _generate_minimal_config(self, format_type: str) -> str:
        """Generate minimal configuration content dynamically from current models."""
        generator = self.generators.get(format_type)
        if not generator:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")

        minimal_config = generator.generate_minimal_config()
        return generator.format_config_content(minimal_config)

    def _generate_comprehensive_config(self, format_type: str) -> str:
        """Generate comprehensive configuration with comments from current models."""
        generator = self.generators.get(format_type)
        if not generator:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")

        return generator.generate_comprehensive_config()

    def _update_config_with_preferences(
        self, config_file: Path, preferences: dict[str, any]
    ) -> None:
        """Update configuration file with user preferences (persist selections)."""
        try:
            # Load existing config
            config_dict = self.io.load_existing_config(config_file)

            # Map preferences → config updates
            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))

            updates = {}

            # LLM provider and streaming
            if preferences.get("llm_provider"):
                updates.setdefault("llm", {})["default_provider"] = preferences["llm_provider"]
            if "enable_streaming" in preferences:
                updates.setdefault("llm", {})["enable_streaming"] = bool(preferences["enable_streaming"])

            # Generation: test framework and coverage threshold (percent → 0..1)
            if preferences.get("test_framework"):
                updates.setdefault("generation", {})["test_framework"] = preferences["test_framework"]
            if preferences.get("min_coverage") is not None:
                try:
                    pct = float(preferences["min_coverage"])  # 0..100
                    updates.setdefault("generation", {})["coverage_threshold"] = clamp01(pct / 100.0)
                except Exception:
                    pass

            # Refinement enable and annotations
            if "enable_refinement" in preferences:
                gen = updates.setdefault("generation", {})
                gen["enable_refinement"] = bool(preferences["enable_refinement"])
                refine = gen.setdefault("refine", {})
                refine["enable"] = bool(preferences["enable_refinement"])

            if preferences.get("annotate_failed_tests") is not None:
                updates.setdefault("generation", {}).setdefault("refine", {})[
                    "annotate_failed_tests"
                ] = bool(preferences["annotate_failed_tests"])

            if preferences.get("annotation_style"):
                updates.setdefault("generation", {}).setdefault("refine", {})[
                    "annotation_style"
                ] = preferences["annotation_style"]

            if preferences.get("annotation_placement"):
                updates.setdefault("generation", {}).setdefault("refine", {})[
                    "annotation_placement"
                ] = preferences["annotation_placement"]

            # Deep-merge updates into config
            merged = self.io.deep_merge(config_dict, updates)

            # Write back in original format
            self.io.write_merged_config(config_file, merged)

            self.ui.display_success(
                f"Applied guided setup preferences to {config_file}",
                "Setup Complete",
            )

        except Exception as e:
            self.ui.display_error(
                f"Failed to write configuration updates: {e}",
                "Setup Error",
            )
            # Still show preferences for manual application if needed
            self.ui.print_divider("Your Preferences")
            for key, value in preferences.items():
                self.ui.console.print(f"[highlight]{key}:[/] {value}")
