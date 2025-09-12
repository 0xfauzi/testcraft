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
    tomli_w = None

from .models import TestCraftConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigLoader:
    """Configuration loader that merges YAML files, environment variables, and CLI arguments."""

    DEFAULT_CONFIG_FILES = [
        ".testcraft.toml",  # TOML files (preferred)
        ".testcraft.yml",
        ".testcraft.yaml",
        "testcraft.toml",
        "testcraft.yml",
        "testcraft.yaml",
        ".testgen.toml",  # Legacy support (TOML)
        ".testgen.yml",  # Legacy support (YAML)
        ".testgen.yaml",  # Legacy support (YAML)
    ]

    ENV_PREFIX = "TESTCRAFT_"

    def __init__(self, config_file: str | Path | None = None):
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
            config_dict = {}

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
        env_config = {}

        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.ENV_PREFIX) :].lower()

                # Convert environment variable name to nested dict structure
                # e.g., TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE -> coverage.minimum_line_coverage
                nested_keys = config_key.replace("__", ".").split(".")

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
            filepath: Path for the config file. Defaults to .testcraft.yml

        Returns:
            Path to the created configuration file
        """
        if filepath is None:
            filepath = Path(".testcraft.yml")
        else:
            filepath = Path(filepath)

        # Generate comprehensive config with all options and explanations
        config_content = """# TestCraft Configuration
# This file contains all available configuration options with detailed explanations.
# Uncomment and modify the sections you want to customize.

# =============================================================================
# TEST DISCOVERY PATTERNS
# =============================================================================

test_patterns:
  # Patterns for finding test files (supports glob patterns)
  test_patterns:
    - 'test_*.py'
    - '*_test.py'
    - 'tests/**/test_*.py'

  # Files and patterns to exclude from test generation
  exclude:
    - 'migrations/*'
    - '*/deprecated/*'
    - '__pycache__/*'
    - '*.pyc'

  # Additional directories to exclude (common ones are already included by default)
  exclude_dirs: []
    # - 'vendor'
    # - 'third_party'

# =============================================================================
# TEST GENERATION STYLE
# =============================================================================

style:
  framework: 'pytest'              # Options: 'pytest', 'unittest'
  assertion_style: 'pytest'       # Options: 'pytest', 'unittest', 'auto'
  mock_library: 'unittest.mock'   # Options: 'unittest.mock', 'pytest-mock', 'auto'

# =============================================================================
# COVERAGE ANALYSIS & THRESHOLDS
# =============================================================================

coverage:
  # Coverage thresholds
  minimum_line_coverage: 80.0      # Minimum line coverage percentage
  minimum_branch_coverage: 70.0    # Minimum branch coverage percentage
  regenerate_if_below: 60.0        # Regenerate tests if coverage drops below this

  # Additional pytest arguments for coverage runs
  pytest_args: []                  # e.g., ['-v', '--tb=short']
  junit_xml: true                  # Enable JUnit XML for all coverage runs

  # Test runner configuration
  runner:
    mode: 'python-module'          # Options: 'python-module', 'pytest-path', 'custom'
    python: null                   # Python executable (null = current sys.executable)
    pytest_path: 'pytest'         # Path to pytest when mode is 'pytest-path'
    custom_cmd: []                 # Custom command when mode is 'custom'
    cwd: null                      # Working directory (null = project root)
    args: []                       # Runner-specific args before pytest_args

  # Environment configuration for test runs
  env:
    propagate: true                # Inherit current environment variables
    extra: {}                      # Additional environment variables
    append_pythonpath: []          # Paths to append to PYTHONPATH

# =============================================================================
# TEST GENERATION BEHAVIOR
# =============================================================================

generation:
  # Test content and structure options
  include_docstrings: true         # Include docstrings in test methods (true, false, "minimal")
  generate_fixtures: true          # Generate pytest fixtures for common setup
  parametrize_similar_tests: true  # Use @pytest.mark.parametrize for similar tests
  max_test_methods_per_class: 20   # Maximum test methods per class (0 for unlimited)
  always_analyze_new_files: false  # Always analyze new files even if they have tests

  # Post-generation test runner (runs pytest after generating tests)
  test_runner:
    enable: false                  # Enable post-generation test execution
    args: []                       # Extra pytest args, e.g., ['-q', '-x']
    cwd: null                      # Working directory (null = project root)
    junit_xml: true                # Generate JUnit XML for failure parsing

  # Test merging strategy
  merge:
    strategy: 'append'             # Options: 'append', 'ast-merge'
    dry_run: false                 # Preview changes without applying
    formatter: 'none'              # Code formatter to apply after merge

  # Test refinement loop (AI-powered test fixing)
  refine:
    enable: false                  # Enable AI-powered test refinement
    max_retries: 2                 # Maximum refinement attempts
    backoff_base_sec: 1.0          # Base delay between refinement attempts
    backoff_max_sec: 8.0           # Maximum delay between attempts
    stop_on_no_change: true        # Stop if LLM returns no changes
    max_total_minutes: 5.0         # Maximum total time for refinement
    strategy: 'auto'               # Refinement strategy: 'auto', 'comprehensive', 'balanced',
                                   # 'dependency_focused', 'logic_focused', 'setup_focused'

# =============================================================================
# ENVIRONMENT DETECTION & MANAGEMENT
# =============================================================================

environment:
  # Environment detection settings
  auto_detect: true                # Auto-detect current environment manager
  preferred_manager: 'auto'        # 'poetry' | 'pipenv' | 'conda' | 'uv' | 'venv' | 'auto'
  respect_virtual_env: true        # Always use current virtual env
  dependency_validation: true      # Validate deps before running tests

  # Environment-specific overrides
  overrides:
    poetry:
      use_poetry_run: true         # Use `poetry run pytest` instead of direct python
      respect_poetry_venv: true
    pipenv:
      use_pipenv_run: true         # Use `pipenv run pytest`
    conda:
      activate_environment: true   # Ensure conda environment is active
    uv:
      use_uv_run: false           # Use direct python instead of `uv run`

# =============================================================================
# COST MANAGEMENT & OPTIMIZATION
# =============================================================================

cost_management:
  # File size limits for cost control
  max_file_size_kb: 50             # Skip files larger than this (KB)
  max_context_size_chars: 100000   # Limit total context size
  max_files_per_request: 15        # Override batch size for large files
  use_cheaper_model_threshold_kb: 10 # Use cheaper model for files < this size
  enable_content_compression: true  # Remove comments/whitespace in prompts

  # Cost thresholds and limits
  cost_thresholds:
    daily_limit: 50.0              # Maximum daily cost in USD
    per_request_limit: 2.0         # Maximum cost per request in USD
    warning_threshold: 1.0         # Warn when request exceeds this cost

  # Additional optimizations
  skip_trivial_files: true         # Skip files with < 5 functions/classes
  token_usage_logging: true        # Log token usage for cost tracking

# =============================================================================
# TEST QUALITY ANALYSIS
# =============================================================================

quality:
  # Quality analysis settings
  enable_quality_analysis: true    # Enable quality analysis by default
  enable_mutation_testing: true    # Enable mutation testing by default
  minimum_quality_score: 75.0      # Minimum acceptable quality score (%)
  minimum_mutation_score: 80.0     # Minimum acceptable mutation score (%)
  max_mutants_per_file: 50         # Maximum mutants per file for performance
  mutation_timeout: 30             # Timeout in seconds for mutation testing
  display_detailed_results: true   # Show detailed quality analysis results
  enable_pattern_analysis: true    # Enable failure pattern analysis for smart refinement

  # Modern Python Mutators
  modern_mutators:
    enable_type_hints: true        # Enable type hint mutations
    enable_async_await: true       # Enable async/await mutations
    enable_dataclass: true         # Enable dataclass mutations
    type_hints_severity: 'medium'  # Severity: 'low', 'medium', 'high'
    async_severity: 'high'         # Async mutations often critical
    dataclass_severity: 'medium'   # Dataclass mutations typically medium severity

# (Removed sections: security, prompt_engineering, context)

# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDES
# =============================================================================
#
# You can override any configuration value using environment variables with the prefix TESTCRAFT_
# Use double underscores (__) to separate nested keys.
#
# Examples:
#   TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE=85
#   TESTCRAFT_GENERATION__TEST_RUNNER__ENABLE=true
#   TESTCRAFT_COST_MANAGEMENT__DAILY_LIMIT=25.0
#
# =============================================================================
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
