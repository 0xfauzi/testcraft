"""Configuration initialization with support for multiple formats."""

import json
import logging
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config.models import TestCraftConfig
from ..ports.ui_port import UIPort

logger = logging.getLogger(__name__)


class ConfigInitializationError(Exception):
    """Raised when configuration initialization fails."""

    pass


class ConfigInitializer:
    """Initialize configuration files with guided setup."""

    def __init__(self, ui: UIPort):
        """Initialize with UI adapter."""
        self.ui = ui

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
            if output_path:
                config_file = output_path
            else:
                extensions = {
                    "toml": ".testcraft.toml",
                    "yaml": ".testcraft.yml",
                    "json": ".testcraft.json",
                }
                config_file = Path(extensions.get(format_type, ".testcraft.toml"))

            # Check if file already exists
            if config_file.exists():
                if not self.ui.get_user_confirmation(
                    f"Configuration file {config_file} already exists. Overwrite?",
                    default=False,
                ):
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
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Configuration file created: {config_file}")
            return config_file

        except Exception as e:
            raise ConfigInitializationError(f"Failed to create configuration file: {e}")

    def run_guided_setup(self, config_file: Path) -> None:
        """Run guided configuration setup."""
        try:
            self.ui.display_info("Starting guided configuration setup", "Guided Setup")

            # Collect user preferences
            preferences = self._collect_user_preferences()

            # Update configuration file with preferences
            self._update_config_with_preferences(config_file, preferences)

            self.ui.display_success(
                f"Configuration updated with your preferences: {config_file}",
                "Setup Complete",
            )

        except Exception as e:
            self.ui.display_error(f"Guided setup failed: {e}", "Setup Error")

    def _generate_minimal_config(self, format_type: str) -> str:
        """Generate minimal configuration content dynamically from current models."""
        # Create default config instance
        config = TestCraftConfig()

        # Extract minimal settings using model data
        minimal_config = {
            "llm": {
                "default_provider": config.llm.default_provider,
                "openai_model": config.llm.openai_model,
                "anthropic_model": config.llm.anthropic_model,
                "temperature": config.llm.temperature,
            },
            "style": {
                "framework": config.style.framework,
                "assertion_style": config.style.assertion_style,
            },
            "coverage": {
                "minimum_line_coverage": config.coverage.minimum_line_coverage,
                "minimum_branch_coverage": config.coverage.minimum_branch_coverage,
            },
            # Include key refinement settings in minimal config
            "generation": {
                "refine": {
                    "enable": config.generation.refine.enable,
                    "annotate_failed_tests": config.generation.refine.annotate_failed_tests,
                }
            },
        }

        return self._format_config_content(minimal_config, format_type)

    def _generate_comprehensive_config(self, format_type: str) -> str:
        """Generate comprehensive configuration with comments from current models."""
        if format_type == "toml":
            return self._generate_dynamic_toml_config()
        elif format_type == "yaml":
            return self._generate_dynamic_yaml_config()
        elif format_type == "json":
            return self._generate_dynamic_json_config()
        else:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")

    def _generate_toml_config(self) -> str:
        """DEPRECATED: Generate comprehensive TOML configuration with all available options.

        This method uses a hardcoded template and is kept for backward compatibility.
        Use _generate_dynamic_toml_config() instead for current model-based generation.
        """
        return """# TestCraft Configuration (TOML)
# Complete configuration with all available options and detailed comments
# Generated by TestCraft Configuration Wizard

# =============================================================================
# LLM PROVIDER CONFIGURATION
# Configure AI models for test generation, analysis, and refinement
# =============================================================================

[llm]
# Default LLM provider to use for all operations
# Options: 'openai', 'anthropic', 'azure-openai', 'bedrock'
default_provider = "openai"

# OpenAI Configuration (GPT-4.1 default)
# API Key: Set OPENAI_API_KEY environment variable
openai_model = "gpt-4.1"              # Model for test generation/analysis
openai_base_url = ""                  # Custom API base URL (optional)
openai_max_tokens = 12000             # Maximum tokens in response (auto-calculated)
openai_timeout = 60.0                 # Request timeout in seconds (5.0-600.0)

# Anthropic Claude Configuration (Claude Sonnet 4 default)
# API Key: Set ANTHROPIC_API_KEY environment variable  
anthropic_model = "claude-sonnet-4"   # Model with extended thinking capabilities
anthropic_max_tokens = 100000         # Maximum tokens for Claude (up to 200k context)
anthropic_timeout = 60.0              # Request timeout in seconds

# Azure OpenAI Configuration (deployment name)
# API Key: Set AZURE_OPENAI_API_KEY environment variable
# Endpoint: Set AZURE_OPENAI_ENDPOINT environment variable
azure_openai_deployment = "claude-sonnet-4"   # Azure deployment name
azure_openai_api_version = "2024-02-15-preview"  # API version
azure_openai_timeout = 60.0           # Request timeout

# AWS Bedrock Configuration (Claude models via Bedrock)
# Credentials: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
aws_region = ""                       # AWS region (e.g., "us-east-1")
bedrock_model_id = "anthropic.claude-3-7-sonnet-v1:0"  # Bedrock model ID
bedrock_timeout = 60.0                # Request timeout

# General LLM Settings
temperature = 0.1                     # Response randomness (0.0-2.0, lower = deterministic)
                                      # NOTE: Reasoning models (o4-mini, o3, o4) ignore this and use default (1.0)
max_retries = 3                       # Maximum retry attempts (0-10)
enable_streaming = false              # Enable streaming responses (where supported)

# =============================================================================
# TEST FILE PATTERNS
# Configure which files to include/exclude from test generation
# =============================================================================

[test_patterns]
# Glob patterns for finding test files
test_patterns = [
    "test_*.py",
    "*_test.py", 
    "tests/**/test_*.py"
]

# Files and patterns to exclude from test generation
exclude = [
    "migrations/*",
    "*/deprecated/*",
    "__pycache__/*",
    "*.pyc"
]

# Directories to exclude from scanning (virtual envs, build dirs, etc.)
exclude_dirs = [
    # Virtual environments
    "venv", "env", ".env", ".venv", "virtualenv",
    # Build directories  
    "build", "dist", "*.egg-info", "*.dist-info",
    # Cache directories
    "__pycache__", ".pytest_cache", ".coverage", ".cache",
    # IDE directories
    ".vscode", ".idea", ".vs",
    # Version control
    ".git", ".hg", ".svn",
    # Test generation artifacts
    ".artifacts"
]

# =============================================================================
# TEST GENERATION STYLE
# Configure the style and framework for generated tests
# =============================================================================

[style]
framework = "pytest"              # Testing framework: 'pytest', 'unittest'
assertion_style = "pytest"        # Assertion style: 'pytest', 'unittest', 'auto'
mock_library = "unittest.mock"    # Mock library: 'unittest.mock', 'pytest-mock', 'auto'

# =============================================================================
# COVERAGE ANALYSIS
# Configure test coverage requirements and analysis
# =============================================================================

[coverage]
minimum_line_coverage = 80.0         # Minimum line coverage percentage (0.0-100.0)
minimum_branch_coverage = 70.0       # Minimum branch coverage percentage (0.0-100.0)
regenerate_if_below = 60.0           # Regenerate tests if coverage drops below this
junit_xml = true                     # Enable JUnit XML for all coverage runs
pytest_args = []                     # Extra arguments for pytest coverage runs

# Test runner configuration for coverage analysis
[coverage.runner]
mode = "python-module"               # Execution mode: 'python-module', 'pytest-path', 'custom'
python = ""                          # Python executable (empty = current sys.executable)
pytest_path = "pytest"              # Path to pytest when mode is 'pytest-path'
custom_cmd = []                      # Custom command when mode is 'custom'
cwd = ""                            # Working directory (empty = project root)
args = []                           # Runner-specific args before pytest_args

# Environment configuration for test execution
[coverage.env]
propagate = true                     # Inherit current environment variables
extra = {}                          # Additional environment variables
append_pythonpath = []              # Paths to append to PYTHONPATH

# =============================================================================
# TEST GENERATION BEHAVIOR
# Configure how tests are generated and processed
# =============================================================================

[generation]
include_docstrings = true            # Include docstrings in test methods (true/false/"minimal")
generate_fixtures = true             # Generate pytest fixtures for common setup
parametrize_similar_tests = true     # Use @pytest.mark.parametrize for similar tests
max_test_methods_per_class = 20      # Maximum test methods per class (0 = unlimited)
always_analyze_new_files = false    # Always analyze new files even if they have tests

# Post-generation test execution
[generation.test_runner]
enable = false                       # Enable post-generation test execution
args = []                           # Extra pytest arguments
cwd = ""                            # Working directory (empty = project root)
junit_xml = true                    # Enable JUnit XML for reliable failure parsing

# Test merging strategies
[generation.merge]
strategy = "append"                  # Merging strategy: 'append', 'ast-merge'
dry_run = false                     # Preview changes without applying
formatter = "none"                  # Code formatter after merge: 'black', 'isort', 'none'

# AI-powered test refinement
[generation.refine]
enable = false                      # Enable AI-powered test refinement
max_retries = 2                     # Maximum refinement attempts (0-10)
backoff_base_sec = 1.0             # Base delay between attempts (0.1+)
backoff_max_sec = 8.0              # Maximum delay between attempts (1.0+)
stop_on_no_change = true           # Stop if LLM returns no changes
max_total_minutes = 5.0            # Maximum total time for refinement (0.1+)

# Failed refinement annotation configuration
annotate_failed_tests = true       # Annotate test files with fix instructions when refinement fails
annotation_placement = "top"       # Where to place annotations: "top" or "bottom"
annotation_include_failure_excerpt = true  # Include trimmed failure output in annotations
annotation_max_failure_chars = 600 # Maximum characters of failure output to include (100-2000)
annotation_style = "docstring"     # Annotation style: "docstring" (triple quotes) or "hash" (comment lines)
include_llm_fix_instructions = true # Include LLM fix instructions in failure annotations

# Refinement strategy: Coming in future release
# Currently all refinement uses generic approach

# =============================================================================
# COST MANAGEMENT
# Control AI usage costs and optimize resource consumption
# =============================================================================

[cost_management]
max_file_size_kb = 50                # Skip files larger than this (KB)
max_context_size_chars = 100000      # Limit total context size (characters)
max_files_per_request = 15           # Override batch size for large files
use_cheaper_model_threshold_kb = 10  # Use cheaper model for files < this size
enable_content_compression = true    # Remove comments/whitespace in prompts
skip_trivial_files = true           # Skip files with < 5 functions/classes
token_usage_logging = true          # Log token usage for cost tracking

# Cost thresholds and limits
[cost_management.cost_thresholds]
daily_limit = 50.0                  # Maximum daily cost in USD
per_request_limit = 2.0             # Maximum cost per request in USD
warning_threshold = 1.0             # Warn when request exceeds this cost (USD)

# =============================================================================
# QUALITY ANALYSIS
# Configure test quality metrics and mutation testing
# =============================================================================

[quality]
enable_quality_analysis = true      # Enable quality analysis by default
enable_mutation_testing = true      # Enable mutation testing by default
minimum_quality_score = 75.0        # Minimum acceptable quality score (0.0-100.0)
minimum_mutation_score = 80.0       # Minimum acceptable mutation score (0.0-100.0)
max_mutants_per_file = 50           # Maximum mutants per file for performance
mutation_timeout = 30               # Timeout in seconds for mutation testing
display_detailed_results = true     # Show detailed quality analysis results
enable_pattern_analysis = true      # Enable failure pattern analysis for refinement

# Modern Python mutator configurations
[quality.modern_mutators]
enable_type_hints = true            # Enable type hint mutations
enable_async_await = true           # Enable async/await mutations  
enable_dataclass = true             # Enable dataclass mutations
# Severity levels: 'low', 'medium', 'high'
type_hints_severity = "medium"      # Type hint mutation severity
async_severity = "high"             # Async mutation severity
dataclass_severity = "medium"       # Dataclass mutation severity

# =============================================================================
# ENVIRONMENT MANAGEMENT
# Configure virtual environment detection and integration
# =============================================================================

[environment]
auto_detect = true                  # Auto-detect current environment manager
# Preferred manager: 'poetry', 'pipenv', 'conda', 'uv', 'venv', 'auto'
preferred_manager = "auto"
respect_virtual_env = true          # Always use current virtual environment
dependency_validation = true       # Validate dependencies before running tests

# Poetry-specific settings
[environment.overrides.poetry]
use_poetry_run = true              # Use 'poetry run' for commands
respect_poetry_venv = true         # Respect Poetry's virtual environment

# Pipenv-specific settings  
[environment.overrides.pipenv]
use_pipenv_run = true              # Use 'pipenv run' for commands

# Conda-specific settings
[environment.overrides.conda] 
activate_environment = true        # Activate conda environment

# UV-specific settings (fast Python package installer)
[environment.overrides.uv]
use_uv_run = false                 # Use 'uv run' for commands

# (Removed sections: prompt_engineering, context, security)

# =============================================================================
# TELEMETRY & OBSERVABILITY (Optional)
# Configure telemetry collection for monitoring and analytics
# =============================================================================

[telemetry]
enabled = false                     # Enable telemetry collection
# Backend options: 'opentelemetry', 'datadog', 'jaeger', 'noop'
backend = "opentelemetry"
service_name = "testcraft"          # Service name for telemetry
service_version = ""                # Service version (auto-detected if empty)
environment = "development"         # Environment: 'development', 'staging', 'production'

# Tracing configuration
trace_sampling_rate = 1.0          # Trace sampling rate (0.0-1.0)
capture_llm_calls = true           # Trace LLM API calls
capture_coverage_runs = true       # Trace coverage analysis operations
capture_file_operations = true     # Trace file read/write operations
capture_test_generation = true     # Trace test generation processes

# Metrics configuration
collect_metrics = true             # Enable metrics collection
metrics_interval_seconds = 30      # Metrics collection interval
track_token_usage = true           # Track LLM token usage metrics
track_coverage_delta = true        # Track coverage improvement metrics
track_test_pass_rate = true        # Track test success/failure rates

# Privacy and anonymization
anonymize_file_paths = true        # Hash file paths in telemetry data
anonymize_code_content = true      # Exclude actual code content from telemetry
opt_out_data_collection = false    # Completely disable data collection

# Global attributes to attach to all telemetry
global_attributes = {}

# Backend-specific configurations
[telemetry.backends.opentelemetry]
endpoint = ""                      # OTLP endpoint (auto-detect if empty)
headers = {}                       # Additional headers for OTLP exporter
insecure = false                   # Use insecure gRPC connection
timeout = 10                       # Timeout for exports in seconds

[telemetry.backends.datadog]
api_key = ""                       # Datadog API key (or DD_API_KEY env var)
site = "datadoghq.com"            # Datadog site
service = "testcraft"             # Service name
env = "development"               # Environment
version = ""                      # Service version

[telemetry.backends.jaeger]
endpoint = "http://localhost:14268/api/traces"  # Jaeger endpoint
agent_host_name = "localhost"     # Jaeger agent hostname  
agent_port = 6831                 # Jaeger agent port

# =============================================================================
# EVALUATION HARNESS (Advanced)
# Configure test evaluation and regression testing
# =============================================================================

[evaluation]
enabled = false                    # Enable evaluation harness functionality
golden_repos_path = ""            # Path to golden repositories for regression testing

# Acceptance checks configuration
acceptance_checks = true           # Enable automated acceptance checks (syntax, imports, pytest)

# LLM-as-judge configuration
llm_judge_enabled = true          # Enable LLM-as-judge evaluation
# Evaluation dimensions for LLM-as-judge
rubric_dimensions = ["correctness", "coverage", "clarity", "safety"]

# A/B testing and statistical analysis
statistical_testing = true        # Enable statistical significance testing
confidence_level = 0.95           # Statistical confidence level (0.5-0.99)

# Human review configuration
human_review_enabled = false      # Enable human-in-the-loop review

# Artifact and state management
artifacts_path = ".testcraft/evaluation_artifacts"  # Path for evaluation artifacts
state_file = ".testcraft_evaluation_state.json"     # File for evaluation state

# Evaluation timeouts and limits
evaluation_timeout_seconds = 300  # Timeout for individual evaluations (10-3600)
batch_size = 10                   # Batch size for A/B testing (1-100)

# Prompt version for evaluation
prompt_version = ""               # Specific prompt version (empty = latest)

# =============================================================================
# API CREDENTIALS
# =============================================================================
# API credentials are loaded from environment variables for security.
# Set these environment variables in your system or .env file:
#
# OpenAI: OPENAI_API_KEY
# Anthropic: ANTHROPIC_API_KEY  
# Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
# AWS Bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
#
# Custom endpoints (optional):
# AZURE_OPENAI_ENDPOINT, OLLAMA_BASE_URL
"""

    def _generate_yaml_config(self) -> str:
        """DEPRECATED: Generate comprehensive YAML configuration (fallback).

        Use _generate_dynamic_yaml_config() instead for current model-based generation.
        """
        # Use existing YAML from config loader if available
        from ..config.loader import ConfigLoader

        loader = ConfigLoader()
        return loader.create_sample_config().read_text()

    def _generate_json_config(self) -> str:
        """DEPRECATED: Generate comprehensive JSON configuration.

        Use _generate_dynamic_json_config() instead for current model-based generation.
        """
        config = TestCraftConfig()
        config_dict = config.model_dump()
        return json.dumps(config_dict, indent=2)

    def _generate_dynamic_toml_config(self) -> str:
        """Generate comprehensive TOML configuration dynamically from current models."""
        config = TestCraftConfig()

        toml_lines = []
        toml_lines.append("# TestCraft Configuration (TOML)")
        toml_lines.append(
            "# Complete configuration with all available options and detailed comments"
        )
        toml_lines.append("# Generated dynamically from current TestCraft models")
        toml_lines.append("")

        # Generate sections dynamically from the model
        self._add_model_section_to_toml(
            config, toml_lines, "", "TestCraft Configuration"
        )

        # Add credential information at the end
        toml_lines.extend(
            [
                "",
                "# =============================================================================",
                "# API CREDENTIALS",
                "# =============================================================================",
                "# API credentials are loaded from environment variables for security.",
                "# Set these environment variables in your system or .env file:",
                "#",
                "# OpenAI: OPENAI_API_KEY",
                "# Anthropic: ANTHROPIC_API_KEY",
                "# Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",
                "# AWS Bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION",
                "#",
                "# Custom endpoints (optional):",
                "# AZURE_OPENAI_ENDPOINT, OLLAMA_BASE_URL",
            ]
        )

        return "\n".join(toml_lines)

    def _generate_dynamic_yaml_config(self) -> str:
        """Generate comprehensive YAML configuration dynamically from current models."""
        config = TestCraftConfig()
        config_dict = config.model_dump()

        try:
            import yaml

            # Add header comments
            header = """# TestCraft Configuration (YAML)
# Complete configuration with all available options and detailed comments
# Generated dynamically from current TestCraft models

"""
            yaml_content = yaml.dump(
                config_dict, default_flow_style=False, indent=2, sort_keys=False
            )
            return header + yaml_content

        except ImportError:
            # Fallback if PyYAML not available
            return self._dict_to_simple_yaml(config_dict)

    def _generate_dynamic_json_config(self) -> str:
        """Generate comprehensive JSON configuration dynamically from current models."""
        config = TestCraftConfig()
        config_dict = config.model_dump()

        # JSON doesn't support comments, so we add them in a special field
        config_dict["_info"] = {
            "description": "TestCraft Configuration (JSON)",
            "note": "Generated dynamically from current TestCraft models",
            "credentials": "Set API keys in environment variables (see documentation)",
        }

        return json.dumps(config_dict, indent=2, sort_keys=False)

    def _add_model_section_to_toml(
        self,
        model: BaseModel,
        toml_lines: list[str],
        section_prefix: str,
        section_title: str,
    ) -> None:
        """Add a model section to TOML lines with proper formatting and comments."""
        model_fields = model.model_fields
        model_data = model.model_dump()

        # Add section header
        if section_prefix:
            toml_lines.extend(
                [
                    "",
                    "# " + "=" * 77,
                    f"# {section_title.upper()}",
                    "# " + "=" * 77,
                    "",
                    f"[{section_prefix}]",
                ]
            )

        # Process each field
        for field_name, field_info in model_fields.items():
            field_value = model_data.get(field_name)
            field_description = (
                field_info.description or f"Configuration for {field_name}"
            )

            # Check if this is a nested BaseModel
            annotation = field_info.annotation
            origin = typing.get_origin(annotation)

            # Handle Optional/Union types
            if origin is typing.Union:
                args = typing.get_args(annotation)
                # Check if it's Optional (Union with None)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            # Check if it's a BaseModel subclass
            is_nested_model = (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and isinstance(field_value, dict)
            )

            if is_nested_model:
                nested_section = (
                    f"{section_prefix}.{field_name}" if section_prefix else field_name
                )
                nested_model = getattr(model, field_name)
                self._add_model_section_to_toml(
                    nested_model, toml_lines, nested_section, field_description
                )
                continue

            # Handle regular fields
            toml_lines.append(f"# {field_description}")

            # Format value based on type
            if isinstance(field_value, str):
                # Escape quotes in string values
                escaped_value = field_value.replace('"', '\\"')
                toml_lines.append(f'{field_name} = "{escaped_value}"')
            elif isinstance(field_value, bool):
                toml_lines.append(f"{field_name} = {str(field_value).lower()}")
            elif isinstance(field_value, (int, float)):
                toml_lines.append(f"{field_name} = {field_value}")
            elif isinstance(field_value, list):
                if not field_value:  # Empty list
                    toml_lines.append(f"{field_name} = []")
                elif all(isinstance(item, str) for item in field_value):
                    # Escape quotes in string items
                    escaped_items = [item.replace('"', '\\"') for item in field_value]
                    formatted_list = (
                        "[" + ", ".join(f'"{item}"' for item in escaped_items) + "]"
                    )
                    toml_lines.append(f"{field_name} = {formatted_list}")
                else:
                    formatted_list = (
                        "[" + ", ".join(str(item) for item in field_value) + "]"
                    )
                    toml_lines.append(f"{field_name} = {formatted_list}")
            elif isinstance(field_value, dict):
                if field_value:  # Non-empty dict
                    # Format dict properly for TOML
                    dict_items = []
                    for k, v in field_value.items():
                        if isinstance(v, str):
                            escaped_v = v.replace('"', '\\"')
                            dict_items.append(f'"{k}" = "{escaped_v}"')
                        else:
                            dict_items.append(f'"{k}" = {v}')
                    if dict_items:
                        toml_lines.append(
                            f"{field_name} = {{ {', '.join(dict_items)} }}"
                        )
                    else:
                        toml_lines.append(f"{field_name} = {{}}")
                else:  # Empty dict
                    toml_lines.append(f"{field_name} = {{}}")
            elif field_value is None:
                toml_lines.append(f"# {field_name} = null  # Optional field")
            else:
                toml_lines.append(f"{field_name} = {field_value}")

            toml_lines.append("")

    def _dict_to_simple_yaml(self, data: dict[str, Any], indent: int = 0) -> str:
        """Convert dict to simple YAML format (fallback when PyYAML not available)."""
        lines = []
        indent_str = "  " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(self._dict_to_simple_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}:")
                for item in value:
                    if isinstance(item, str):
                        lines.append(f'{indent_str}  - "{item}"')
                    else:
                        lines.append(f"{indent_str}  - {item}")
            elif isinstance(value, str):
                lines.append(f'{indent_str}{key}: "{value}"')
            else:
                lines.append(f"{indent_str}{key}: {value}")

        return "\n".join(lines)

    def _format_config_content(
        self, config_dict: dict[str, Any], format_type: str
    ) -> str:
        """Format configuration dictionary to specified format."""
        if format_type == "toml":
            try:
                import tomli_w

                return tomli_w.dumps(config_dict)
            except ImportError:
                # Fallback to manual TOML generation
                return self._dict_to_simple_toml(config_dict)
        elif format_type == "yaml":
            try:
                import yaml

                return yaml.dump(config_dict, default_flow_style=False, indent=2)
            except ImportError:
                raise ConfigInitializationError("PyYAML not available for YAML format")
        elif format_type == "json":
            return json.dumps(config_dict, indent=2)
        else:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")

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

    def _collect_user_preferences(self) -> dict[str, Any]:
        """Collect user preferences through interactive prompts."""
        preferences = {}

        self.ui.print_divider("LLM Provider Setup")

        # LLM Provider selection
        providers = ["openai", "anthropic", "azure-openai", "bedrock"]
        provider = self.ui.get_user_input(
            "Select LLM provider", input_type="choice", choices=providers
        )
        preferences["llm_provider"] = provider

        # Test framework preference
        self.ui.print_divider("Test Framework Setup")

        frameworks = ["pytest", "unittest"]
        framework = self.ui.get_user_input(
            "Select test framework", input_type="choice", choices=frameworks
        )
        preferences["test_framework"] = framework

        # Coverage thresholds
        self.ui.print_divider("Coverage Configuration")

        min_coverage = self.ui.get_user_input(
            "Minimum line coverage percentage (0-100)", input_type="number", default=80
        )
        preferences["min_coverage"] = min_coverage

        # Enable advanced features
        self.ui.print_divider("Advanced Features")

        enable_refinement = self.ui.get_user_confirmation(
            "Enable AI-powered test refinement?", default=False
        )
        preferences["enable_refinement"] = enable_refinement

        enable_streaming = self.ui.get_user_confirmation(
            "Enable streaming LLM responses?", default=False
        )
        preferences["enable_streaming"] = enable_streaming

        # Failed refinement annotation preferences
        if enable_refinement:
            self.ui.print_divider("Refinement Annotation Setup")

            annotate_failed_tests = self.ui.get_user_confirmation(
                "Annotate test files with fix instructions when refinement fails?",
                default=True,
            )
            preferences["annotate_failed_tests"] = annotate_failed_tests

            if annotate_failed_tests:
                annotation_styles = ["docstring", "hash"]
                annotation_style = self.ui.get_user_input(
                    "Annotation style", input_type="choice", choices=annotation_styles
                )
                preferences["annotation_style"] = annotation_style

                annotation_placements = ["top", "bottom"]
                annotation_placement = self.ui.get_user_input(
                    "Where to place annotations",
                    input_type="choice",
                    choices=annotation_placements,
                )
                preferences["annotation_placement"] = annotation_placement

        return preferences

    def _update_config_with_preferences(
        self, config_file: Path, preferences: dict[str, Any]
    ) -> None:
        """Update configuration file with user preferences."""
        # This is a simplified implementation
        # In a real implementation, you'd parse the existing config and update specific values

        self.ui.display_info(
            "Configuration preferences have been noted. "
            f"Please manually edit {config_file} to apply specific settings.",
            "Manual Configuration Required",
        )

        # Display preferences for user reference
        self.ui.print_divider("Your Preferences")
        for key, value in preferences.items():
            self.ui.console.print(f"[highlight]{key}:[/] {value}")
