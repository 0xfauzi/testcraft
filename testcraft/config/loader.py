"""Configuration loader for TestCraft."""

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore[assignment]

from .models import TestCraftConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigLoader:
    """Configuration loader that merges YAML files, environment variables, and CLI arguments."""

    DEFAULT_CONFIG_FILES = [
        ".testcraft.toml",  # TOML files (preferred)
        "testcraft.toml",
        ".testgen.toml",  # Legacy support (TOML)
    ]

    ENV_PREFIX = "TESTCRAFT_"

    def __init__(self, config_file: str | Path | None = None) -> None:
        """Initialize the configuration loader.

        Args:
            config_file: Path to configuration file. If None, will search for default files.
        """
        self.config_file = Path(config_file) if config_file else None
        self._config_cache: TestCraftConfig | None = None

    def load_config(
        self,
        env_overrides: dict[str, Any] | None = None,
        cli_overrides: dict[str, Any] | None = None,
        reload: bool = False,
    ) -> TestCraftConfig:
        """Load configuration from all sources.

        Args:
            env_overrides: Environment variable overrides
            cli_overrides: CLI argument overrides
            reload: Force reload even if cached

        Returns:
            Validated TestCraft configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config_cache is not None and not reload:
            return self._config_cache

        try:
            # Start with default configuration
            config_dict: dict[str, Any] = {}

            # 1. Load from configuration file (TOML or YAML)
            file_config = self._load_config_file()
            if file_config:
                config_dict = self._deep_merge(config_dict, file_config)
                logger.debug(
                    f"Loaded configuration from {self._get_config_file_path()}"
                )

            # 2. Apply environment variable overrides
            env_config = env_overrides or self._load_env_config()
            if env_config:
                config_dict = self._deep_merge(config_dict, env_config)
                logger.debug("Applied environment variable overrides")

            # 3. Apply CLI overrides (highest priority)
            if cli_overrides:
                config_dict = self._deep_merge(config_dict, cli_overrides)
                logger.debug("Applied CLI argument overrides")

            # 4. Validate and create Pydantic model
            self._config_cache = TestCraftConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")

            return self._config_cache

        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _load_config_file(self) -> dict[str, Any] | None:
        """Load configuration from TOML or YAML file."""
        config_file = self._get_config_file_path()

        if not config_file or not config_file.exists():
            logger.debug("No configuration file found, using defaults")
            return None

        try:
            # Determine file type by extension
            if config_file.suffix.lower() == ".toml":
                return self._load_toml_file(config_file)
            elif config_file.suffix.lower() in (".yml", ".yaml"):
                # Deprecation notice for YAML when explicitly specified
                logger.warning(
                    "YAML config is deprecated. Prefer TOML (.testcraft.toml)."
                )
                return self._load_yaml_file(config_file)
            else:
                logger.warning(f"Unknown configuration file type: {config_file}")
                return None

        except OSError as e:
            error_msg = f"Failed to read {config_file}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _load_toml_file(self, config_file: Path) -> dict[str, Any] | None:
        """Load configuration from TOML file."""
        try:
            with open(config_file, "rb") as f:
                content = tomllib.load(f)

            # Handle empty content
            if not content:
                logger.warning(f"Configuration file {config_file} is empty")
                return None

            return content

        except tomllib.TOMLDecodeError as e:
            error_msg = f"Invalid TOML in {config_file}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _load_yaml_file(self, config_file: Path) -> dict[str, Any] | None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                content = yaml.safe_load(f)

            # Handle empty or None content
            if not content:
                logger.warning(f"Configuration file {config_file} is empty")
                return None

            return content

        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in {config_file}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _load_env_config(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        env_config: dict[str, Any] = {}

        valid_root_keys = set(TestCraftConfig.model_fields.keys())

        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.ENV_PREFIX) :].lower()

                # Convert environment variable name to nested dict structure
                # e.g., TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE -> coverage.minimum_line_coverage
                nested_keys = config_key.replace("__", ".").split(".")

                # Skip helper/testing-only environment variables outside the config schema
                root_key = nested_keys[0]
                if root_key not in valid_root_keys:
                    logger.debug(
                        "Ignoring environment override %s (no config field for '%s')",
                        key,
                        root_key,
                    )
                    continue

                # Parse value (try to convert to appropriate type)
                parsed_value = self._parse_env_value(value)

                # Set nested value in config dict
                self._set_nested_value(env_config, nested_keys, parsed_value)

        return env_config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle numeric values
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # Handle list values (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # Return as string
        return value

    def _set_nested_value(self, config: dict[str, Any], keys: list, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _get_config_file_path(self) -> Path | None:
        """Get the path to the configuration file."""
        if self.config_file:
            return self.config_file

        # Search for default configuration files
        for filename in self.DEFAULT_CONFIG_FILES:
            path = Path(filename)
            if path.exists():
                return path

        return None

    def _deep_merge(
        self, base: dict[str, Any], updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Deeply merge updates into base dictionary."""
        result = base.copy()

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def create_sample_config(self, filepath: str | Path | None = None) -> Path:
        """Create a comprehensive sample configuration file.

        Args:
            filepath: Path for the config file. Defaults to .testcraft.toml

        Returns:
            Path to the created configuration file
        """
        if filepath is None:
            filepath = Path(".testcraft.toml")
        else:
            filepath = Path(filepath)

        # Generate comprehensive config with all options and explanations
        config_content = """# TestCraft Configuration (TOML)
# Adjust and uncomment values as needed for your project.

[test_patterns]
test_patterns = ["test_*.py", "*_test.py", "tests/**/test_*.py"]
exclude = ["migrations/*", "*/deprecated/*", "__pycache__/*", "*.pyc"]
exclude_dirs = []

[style]
framework = "pytest"
assertion_style = "pytest"
mock_library = "unittest.mock"

[coverage]
minimum_line_coverage = 80.0
minimum_branch_coverage = 70.0
regenerate_if_below = 60.0
pytest_args = []
junit_xml = true

[coverage.runner]
mode = "python-module"
python = ""
pytest_path = "pytest"
custom_cmd = []
cwd = ""
args = []

[coverage.env]
propagate = true
append_pythonpath = []

[coverage.env.extra]
# "KEY" = "value"

[generation]
include_docstrings = true
generate_fixtures = true
parametrize_similar_tests = true
max_test_methods_per_class = 20
always_analyze_new_files = false
immediate_refinement = true
enable_refinement = true
batch_size = 5
disable_ruff_format = false

[generation.test_runner]
enable = false
args = []
cwd = ""
junit_xml = true

[generation.merge]
strategy = "append"
dry_run = false
formatter = "none"

[generation.refine]
enable = false
max_retries = 2
backoff_base_sec = 1.0
backoff_max_sec = 8.0
stop_on_no_change = true
max_total_minutes = 5.0
strategy = "auto"

[environment]
auto_detect = true
preferred_manager = "auto"
respect_virtual_env = true
dependency_validation = true

[llm]
default_provider = "openai"  # "openai", "anthropic", "azure-openai", or "bedrock"
temperature = 0.1
max_retries = 3
enable_streaming = false
openai_model = "gpt-4.1"
openai_timeout = 60.0
openai_max_tokens = 12000
openai_base_url = ""
anthropic_model = "claude-sonnet-4"
anthropic_timeout = 60.0
anthropic_max_tokens = 100000
azure_openai_deployment = "gpt-4.1"
azure_openai_api_version = "2024-02-15-preview"
azure_openai_timeout = 60.0
bedrock_model_id = "anthropic.claude-3-7-sonnet-v1:0"
bedrock_timeout = 60.0

[manual_fix]
enable = true
on_fail = false
auto_accept = false
output_dir = ".testcraft/manual_fixes"

[planning]
enabled = true
auto_accept = false
max_retries = 2

[telemetry]
enabled = false
backend = "opentelemetry"
service_name = "testcraft"
environment = "development"
capture_llm_calls = true
collect_metrics = true

[telemetry.global_attributes]
# "feature_flag" = "beta"

[telemetry.backends.opentelemetry]
endpoint = ""
headers = {}
insecure = false
timeout = 10
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(config_content)

        logger.info(f"Sample YAML configuration created at {filepath}")
        return filepath

    def create_sample_toml_config(
        self, filepath: str | Path | None = None, minimal: bool = False
    ) -> Path:
        """Create a sample TOML configuration file.

        Args:
            filepath: Path for the config file. Defaults to .testcraft.toml
            minimal: If True, creates a minimal config; otherwise comprehensive

        Returns:
            Path to the created configuration file
        """
        if tomli_w is None:
            raise ConfigurationError("tomli-w library is required to create TOML files")

        if filepath is None:
            filepath = Path(".testcraft.toml")
        else:
            filepath = Path(filepath)

        if minimal:
            config_content = self._get_minimal_toml_config()
        else:
            config_content = self._get_comprehensive_toml_config()

        with open(filepath, "wb") as f:
            tomli_w.dump(config_content, f)

        logger.info(f"Sample TOML configuration created at {filepath}")
        return filepath

    def validate_config(self, config_dict: dict[str, Any]) -> None:
        """Validate a configuration dictionary without creating the full model.

        Args:
            config_dict: Configuration dictionary to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            TestCraftConfig(**config_dict)
        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            raise ConfigurationError(error_msg) from e

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of the current configuration."""
        if self._config_cache is None:
            self.load_config()

        return {
            "config_file": (
                str(self._get_config_file_path())
                if self._get_config_file_path()
                else "None"
            ),
            "coverage_thresholds": {
                "minimum_line": self._config_cache.coverage.minimum_line_coverage,
                "minimum_branch": self._config_cache.coverage.minimum_branch_coverage,
                "regenerate_below": self._config_cache.coverage.regenerate_if_below,
            },
            "test_framework": self._config_cache.style.framework,
            "environment_manager": self._config_cache.environment.preferred_manager,
            "cost_limits": {
                "daily": self._config_cache.cost_management.cost_thresholds.daily_limit,
                "per_request": self._config_cache.cost_management.cost_thresholds.per_request_limit,
            },
            "quality_analysis": self._config_cache.quality.enable_quality_analysis,
            "mutation_testing": self._config_cache.quality.enable_mutation_testing,
        }

    def _get_minimal_toml_config(self) -> dict[str, Any]:
        """Get minimal TOML configuration for quickstart."""
        return {
            # Essential settings for getting started
            "style": {"framework": "pytest", "assertion_style": "pytest"},
            "coverage": {"minimum_line_coverage": 80.0, "junit_xml": True},
            "generation": {"include_docstrings": True, "generate_fixtures": True},
            "evaluation": {
                "enabled": False,
                "acceptance_checks": True,
                "llm_judge_enabled": False,
            },
            "llm": {
                "default_provider": "openai",
                "openai_model": "gpt-4.1",
                "temperature": 0.1,
            },
        }

    def _get_comprehensive_toml_config(self) -> dict[str, Any]:
        """Get comprehensive TOML configuration with all options."""
        # Create a default TestCraftConfig instance and convert to dict
        default_config = TestCraftConfig()
        config_dict = default_config.model_dump(
            exclude_none=True
        )  # Exclude None values for TOML

        # Further filter to ensure no None values remain in nested structures
        return self._filter_none_values(config_dict)

    def _filter_none_values(self, obj: Any) -> Any:
        """Recursively filter out None values from configuration objects."""
        if isinstance(obj, dict):
            return {
                k: self._filter_none_values(v) for k, v in obj.items() if v is not None
            }
        elif isinstance(obj, list):
            return [self._filter_none_values(item) for item in obj if item is not None]
        else:
            return obj


# Convenience function for quick configuration loading
def load_config(
    config_file: str | Path | None = None,
    env_overrides: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> TestCraftConfig:
    """Load TestCraft configuration from all sources.

    Args:
        config_file: Path to configuration file
        env_overrides: Environment variable overrides
        cli_overrides: CLI argument overrides

    Returns:
        Validated TestCraft configuration
    """
    loader = ConfigLoader(config_file)
    return loader.load_config(env_overrides, cli_overrides)
