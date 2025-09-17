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

            # If TOML, regenerate full commented config with chosen prefs activated (preserve comments)
            suffix = config_file.suffix.lower()
            if suffix == ".toml":
                overrides = self._preferences_to_overrides(preferences)
                content = self._generate_dynamic_toml_config(comment_unused=True, overrides=overrides)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(content)
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
            "generation": {
                "test_framework": config.generation.test_framework,
                "batch_size": config.generation.batch_size,
                "coverage_threshold": config.generation.coverage_threshold,
                "enable_refinement": config.generation.enable_refinement,
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
azure_openai_deployment = "gpt-4.1"   # Azure deployment name (must be OpenAI model)
azure_openai_api_version = "2024-02-15-preview"  # API version
azure_openai_timeout = 60.0           # Request timeout

# AWS Bedrock Configuration (Claude models via Bedrock)
# Credentials: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
aws_region = ""                       # AWS region (e.g., "us-east-1")
bedrock_model_id = "anthropic.claude-sonnet-4-20250514-v1:0"  # Official Claude Sonnet 4 on Bedrock
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

# (Deprecated sections removed: generation.test_runner, generation.merge)

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

    def _generate_dynamic_toml_config(self, comment_unused: bool = True, overrides: dict[str, Any] | None = None) -> str:
        """Generate comprehensive TOML configuration dynamically from current models.

        Args:
            comment_unused: When True, emit every option commented out by default.
            overrides: A dot-keyed mapping of config keys to values to activate (uncommented).
        """
        config = TestCraftConfig()
        
        toml_lines = []
        toml_lines.append("# TestCraft Configuration (TOML)")
        toml_lines.append("# Complete configuration with all available options and detailed comments")
        toml_lines.append("# Generated dynamically from current TestCraft models")
        toml_lines.append("#")
        toml_lines.append("# Configuration Schema:")
        toml_lines.append("# - [test_patterns] - File discovery patterns and test discovery settings")
        toml_lines.append("# - [llm] - Large Language Model configuration")
        toml_lines.append("# - [generation] - Test generation behavior (includes budgets, refinement)")
        toml_lines.append("# - [cost_management] - Cost thresholds and optimization")
        toml_lines.append("# - [telemetry] - Observability and telemetry configuration")
        toml_lines.append("# - [evaluation] - Test evaluation harness (optional)")
        toml_lines.append("# - [environment] - Environment detection and management")
        toml_lines.append("# - [logging] - Logging behavior configuration")
        toml_lines.append("# - [ui] - User interface behavior")
        toml_lines.append("# - [planning] - Test planning configuration")
        toml_lines.append("#")
        toml_lines.append("# NOTE: Deprecated sections (style, coverage, quality) are no longer generated")
        toml_lines.append("# Use generation.test_framework instead of style.framework")
        toml_lines.append("")
        
        # Normalize overrides to dot-key map
        overrides = overrides or {}
        flat_overrides = self._flatten_overrides_dict(overrides)
        
        # Generate sections dynamically from the model
        self._add_model_section_to_toml(
            config,
            toml_lines,
            section_prefix="",
            section_title="TestCraft Configuration",
            comment_unused=comment_unused,
            overrides=flat_overrides,
        )
        
        # Add credential information at the end
        toml_lines.extend([
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
        ])
        
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
            yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)
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
            "credentials": "Set API keys in environment variables (see documentation)"
        }
        
        return json.dumps(config_dict, indent=2, sort_keys=False)
    
    def _add_model_section_to_toml(self, model: BaseModel, toml_lines: list[str], section_prefix: str, section_title: str, comment_unused: bool = True, overrides: dict[str, Any] | None = None) -> None:
        """Add a model section to TOML lines with proper formatting and comments.

        If comment_unused is True, each option is emitted as a commented line unless
        a corresponding override is provided in overrides (dot-key form), in which case
        the option is emitted as an active (uncommented) line with the override value.
        """
        model_fields = model.model_fields
        model_data = model.model_dump()
        overrides = overrides or {}
        
        # Add section header
        if section_prefix:
            toml_lines.extend([
                "",
                "# " + "=" * 77,
                f"# {section_title.upper()}",
                "# " + "=" * 77,
                "",
                f"[{section_prefix}]"
            ])
        
        # Deprecated sections to skip in TOML generation
        deprecated_sections = {'style', 'coverage', 'quality', 'context_enrichment'}
        
        # Partition fields into non-nested and nested to avoid redeclaring parent tables
        non_nested_fields: list[tuple[str, Any, Any, str]] = []
        nested_fields: list[tuple[str, Any, Any, str]] = []

        for field_name, field_info in model_fields.items():
            field_value = model_data.get(field_name)
            field_description = field_info.description or f"Configuration for {field_name}"

            # Skip deprecated sections entirely in TOML generation
            if field_name in deprecated_sections or 'DEPRECATED' in field_description:
                continue

            # Check if this is a nested BaseModel
            annotation = field_info.annotation
            origin = typing.get_origin(annotation)

            if origin is typing.Union:
                args = typing.get_args(annotation)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            is_nested_model = (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and isinstance(field_value, dict)
            )

            if is_nested_model:
                nested_fields.append((field_name, field_info, field_value, field_description))
            else:
                non_nested_fields.append((field_name, field_info, field_value, field_description))

        # If this is the context categories table, add a general consequences note
        if section_prefix.endswith("generation.context_categories"):
            toml_lines.append("# Enabling a category includes that context in prompts; disabling reduces prompt size")
            toml_lines.append("# Consequences: Turning categories off lowers cost/latency but may omit useful signals.")
            toml_lines.append("")

        # First, write all non-nested fields under the current section
        for field_name, _field_info, field_value, field_description in non_nested_fields:
            # Special handling for prompt budgets and section caps to provide richer comments/structure
            is_prompt_budgets_table = section_prefix.endswith("generation.prompt_budgets")

            # Add field description comment
            toml_lines.append(f"# {field_description}")

            # Add consequences notes for key prompt budget fields
            if is_prompt_budgets_table and field_name == "per_item_chars":
                toml_lines.append("# Consequences: Higher allows richer items but may reduce the number of items that fit.")
                toml_lines.append("# Lower keeps items terse and increases diversity but may truncate useful detail.")
            if is_prompt_budgets_table and field_name == "total_chars":
                toml_lines.append("# Consequences: Higher overall budget improves recall; increases cost and risk of longer prompts.")
                toml_lines.append("# Lower forces tighter selection; safer and cheaper but may omit helpful context.")

            # Render section_caps as a dedicated nested table with per-key explanations
            if (
                is_prompt_budgets_table
                and field_name == "section_caps"
                and isinstance(field_value, dict)
            ):
                self._add_section_caps_table_to_toml(toml_lines, section_prefix, field_value)
                toml_lines.append("")
                continue

            # Determine if this key is overridden (active) or should remain commented
            full_key = f"{section_prefix}.{field_name}" if section_prefix else field_name
            has_override = full_key in overrides
            value_to_write = overrides.get(full_key, field_value)
            comment_prefix = "" if has_override or not comment_unused else "# "

            # Default rendering logic
            if isinstance(value_to_write, str):
                escaped_value = value_to_write.replace('"', '\\"')
                toml_lines.append(f'{comment_prefix}{field_name} = "{escaped_value}"')
            elif isinstance(value_to_write, bool):
                toml_lines.append(f'{comment_prefix}{field_name} = {str(value_to_write).lower()}')
            elif isinstance(value_to_write, (int, float)):
                toml_lines.append(f'{comment_prefix}{field_name} = {value_to_write}')
            elif isinstance(value_to_write, list):
                if not value_to_write:
                    toml_lines.append(f'{comment_prefix}{field_name} = []')
                elif all(isinstance(item, str) for item in value_to_write):
                    escaped_items = [item.replace('"', '\\"') for item in value_to_write]
                    formatted_list = '[' + ', '.join(f'"{item}"' for item in escaped_items) + ']'
                    toml_lines.append(f'{comment_prefix}{field_name} = {formatted_list}')
                else:
                    formatted_list = '[' + ', '.join(str(item) for item in value_to_write) + ']'
                    toml_lines.append(f'{comment_prefix}{field_name} = {formatted_list}')
            elif isinstance(value_to_write, dict):
                if value_to_write:
                    dict_items = []
                    for k, v in value_to_write.items():
                        if v is None:
                            continue
                        elif isinstance(v, str):
                            escaped_v = v.replace('"', '\\"')
                            dict_items.append(f'"{k}" = "{escaped_v}"')
                        elif isinstance(v, bool):
                            dict_items.append(f'"{k}" = {str(v).lower()}')
                        else:
                            dict_items.append(f'"{k}" = {v}')
                    if dict_items:
                        toml_lines.append(f'{comment_prefix}{field_name} = {{ {", ".join(dict_items)} }}')
                    else:
                        toml_lines.append(f'{comment_prefix}{field_name} = {{}}')
                else:
                    toml_lines.append(f'{comment_prefix}{field_name} = {{}}')
            elif value_to_write is None:
                # Always comment out optional/None fields
                toml_lines.append(f'# {field_name} = null  # Optional field')
            else:
                toml_lines.append(f'{comment_prefix}{field_name} = {value_to_write}')

            toml_lines.append("")

        # Then, write nested tables
        for field_name, _field_info, _field_value, field_description in nested_fields:
            nested_section = f"{section_prefix}.{field_name}" if section_prefix else field_name
            nested_model = getattr(model, field_name)
            self._add_model_section_to_toml(
                nested_model,
                toml_lines,
                nested_section,
                field_description,
                comment_unused=comment_unused,
                overrides=overrides,
            )

    def _add_section_caps_table_to_toml(self, toml_lines: list[str], section_prefix: str, caps: dict[str, Any], comment_unused: bool = True, overrides: dict[str, Any] | None = None) -> None:
        """Emit a dedicated [..section_caps] table with per-key explanations and consequences.

        Respects comment_unused/overrides similar to other fields.
        """
        overrides = overrides or {}
        # Header and guidance
        toml_lines.append("# Per-section item limits used when assembling LLM context.")
        toml_lines.append("# Increasing a cap shifts budget toward that section; total_chars still applies.")
        toml_lines.append(f"[{section_prefix}.section_caps]")

        # Stable order for readability
        ordered_keys = [
            "snippets",
            "neighbors",
            "test_exemplars",
            "contracts",
            "deps_config_fixtures",
            "coverage_hints",
            "callgraph",
            "error_paths",
            "usage_examples",
            "pytest_settings",
            "side_effects",
            "path_constraints",
        ]

        explanations = {
            "snippets": "Short, symbol-aware code/doc snippets. Higher = more variety; too high can dilute signal.",
            "neighbors": "Import-graph neighbor files (header + slice). Higher adds periphery; increases prompt size.",
            "test_exemplars": "Summaries mined from existing tests. Higher = more examples; may crowd other context.",
            "contracts": "API contract summaries (signature/params/returns/raises/invariants). Higher clarifies spec; costs budget.",
            "deps_config_fixtures": "Env/config keys, DB/HTTP clients, pytest fixtures. Usually 1–2 is enough.",
            "coverage_hints": "Per-file coverage hints (if wired). Keep small; harmless when empty.",
            "callgraph": "Call-graph/import edges + nearby files. Higher adds topology; can add noise.",
            "error_paths": "Exception types from docs/AST. Higher emphasizes error handling; reduces room elsewhere.",
            "usage_examples": "High-quality usage snippets. Higher helps synthesis; too high may overfit to examples.",
            "pytest_settings": "Key pytest ini options. 1 is typically sufficient.",
            "side_effects": "Detected side-effect boundaries (fs/env/network). Higher reveals risks; may add noise.",
            "path_constraints": "Branch/condition summaries. Higher improves logic coverage; can crowd prompt.",
        }

        for key in ordered_keys:
            if key in explanations:
                toml_lines.append(f"# {explanations[key]}")
            value = caps.get(key)
            if value is None:
                continue
            full_key = f"{section_prefix}.section_caps.{key}"
            has_override = full_key in overrides
            value_to_write = overrides.get(full_key, value)
            comment_prefix = "" if has_override or not comment_unused else "# "
            toml_lines.append(f"{comment_prefix}{key} = {value_to_write}")
    
    def _flatten_overrides_dict(self, data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a nested dict into dot-key mapping for overrides."""
        flat: dict[str, Any] = {}
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_overrides_dict(v, key))
            else:
                flat[key] = v
        return flat
    
    def _preferences_to_overrides(self, preferences: dict[str, Any]) -> dict[str, Any]:
        """Map guided setup preferences to concrete config overrides (dot-keyed)."""
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))
        overrides: dict[str, Any] = {}
        # LLM provider and streaming
        if preferences.get("llm_provider"):
            overrides.setdefault("llm", {})["default_provider"] = preferences["llm_provider"]
        if "enable_streaming" in preferences:
            overrides.setdefault("llm", {})["enable_streaming"] = bool(preferences["enable_streaming"])
        # Generation: framework and coverage threshold
        if preferences.get("test_framework"):
            overrides.setdefault("generation", {})["test_framework"] = preferences["test_framework"]
        if preferences.get("min_coverage") is not None:
            try:
                pct = float(preferences["min_coverage"])  # 0..100
                overrides.setdefault("generation", {})["coverage_threshold"] = clamp01(pct / 100.0)
            except Exception:
                pass
        # Refinement enable and annotations
        if "enable_refinement" in preferences:
            gen = overrides.setdefault("generation", {})
            gen["enable_refinement"] = bool(preferences["enable_refinement"])
            refine = gen.setdefault("refine", {})
            refine["enable"] = bool(preferences["enable_refinement"])
        if preferences.get("annotate_failed_tests") is not None:
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotate_failed_tests"
            ] = bool(preferences["annotate_failed_tests"])
        if preferences.get("annotation_style"):
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotation_style"
            ] = preferences["annotation_style"]
        if preferences.get("annotation_placement"):
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotation_placement"
            ] = preferences["annotation_placement"]
        return overrides
    
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
                        lines.append(f"{indent_str}  - \"{item}\"")
                    else:
                        lines.append(f"{indent_str}  - {item}")
            elif isinstance(value, str):
                lines.append(f"{indent_str}{key}: \"{value}\"")
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
                "Annotate test files with fix instructions when refinement fails?", default=True
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
                    "Where to place annotations", input_type="choice", choices=annotation_placements
                )
                preferences["annotation_placement"] = annotation_placement

        return preferences

    def _update_config_with_preferences(
        self, config_file: Path, preferences: dict[str, Any]
    ) -> None:
        """Update configuration file with user preferences (persist selections)."""
        # Load existing config by format
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

        # Map preferences → config updates
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))

        updates: dict[str, Any] = {}

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
        def deep_merge(base: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
            result = base.copy()
            for k, v in inc.items():
                if (
                    k in result
                    and isinstance(result[k], dict)
                    and isinstance(v, dict)
                ):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        merged = deep_merge(config_dict, updates)

        # Write back in original format
        try:
            if suffix == ".toml":
                try:
                    import tomli_w
                    with open(config_file, "wb") as f:
                        tomli_w.dump(merged, f)
                except Exception:
                    # Fallback: simple TOML emitter (minimal, may drop comments)
                    content = self._dict_to_simple_toml(merged)
                    with open(config_file, "w", encoding="utf-8") as f:
                        f.write(content)
            elif suffix in (".yml", ".yaml"):
                import yaml
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, sort_keys=False)
            elif suffix == ".json":
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)
            else:
                # Default to TOML fallback
                content = self._dict_to_simple_toml(merged)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(content)

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
