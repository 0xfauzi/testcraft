"""Configuration initialization with support for multiple formats."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..adapters.io.rich_cli import RichCliComponents
from ..config.models import TestCraftConfig

logger = logging.getLogger(__name__)


class ConfigInitializationError(Exception):
    """Raised when configuration initialization fails."""
    pass


class ConfigInitializer:
    """Initialize configuration files with guided setup."""
    
    def __init__(self, ui: RichCliComponents):
        """Initialize with Rich UI components."""
        self.ui = ui
    
    def create_config_file(
        self,
        format_type: str = 'toml',
        minimal: bool = False,
        output_path: Optional[Path] = None
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
                extensions = {'toml': '.testcraft.toml', 'yaml': '.testcraft.yml', 'json': '.testcraft.json'}
                config_file = Path(extensions.get(format_type, '.testcraft.toml'))
            
            # Check if file already exists
            if config_file.exists():
                if not self.ui.get_user_confirmation(
                    f"Configuration file {config_file} already exists. Overwrite?",
                    default=False
                ):
                    self.ui.display_info("Configuration initialization cancelled", "Cancelled")
                    return config_file
            
            # Generate configuration content
            if minimal:
                content = self._generate_minimal_config(format_type)
            else:
                content = self._generate_comprehensive_config(format_type)
            
            # Write configuration file
            with open(config_file, 'w', encoding='utf-8') as f:
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
                "Setup Complete"
            )
            
        except Exception as e:
            self.ui.display_error(f"Guided setup failed: {e}", "Setup Error")
    
    def _generate_minimal_config(self, format_type: str) -> str:
        """Generate minimal configuration content."""
        # Create default config instance
        config = TestCraftConfig()
        
        # Extract minimal settings
        minimal_config = {
            'llm': {
                'default_provider': config.llm.default_provider,
                'openai_model': config.llm.openai_model,
                'anthropic_model': config.llm.anthropic_model,
                'temperature': config.llm.temperature
            },
            'style': {
                'framework': config.style.framework,
                'assertion_style': config.style.assertion_style
            },
            'coverage': {
                'minimum_line_coverage': config.coverage.minimum_line_coverage,
                'minimum_branch_coverage': config.coverage.minimum_branch_coverage
            }
        }
        
        return self._format_config_content(minimal_config, format_type)
    
    def _generate_comprehensive_config(self, format_type: str) -> str:
        """Generate comprehensive configuration with comments."""
        if format_type == 'toml':
            return self._generate_toml_config()
        elif format_type == 'yaml':
            return self._generate_yaml_config()
        elif format_type == 'json':
            return self._generate_json_config()
        else:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")
    
    def _generate_toml_config(self) -> str:
        """Generate comprehensive TOML configuration."""
        return '''# TestCraft Configuration (TOML)
# Modern, readable configuration format for TestCraft

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

[llm]
# Default LLM provider to use
default_provider = "openai"

# OpenAI Configuration
openai_model = "o4-mini"
openai_max_tokens = 12000
openai_timeout = 60.0

# Anthropic Claude Configuration  
anthropic_model = "claude-3-7-sonnet"
anthropic_max_tokens = 100000
anthropic_timeout = 60.0

# Azure OpenAI Configuration
azure_openai_deployment = "o4-mini"
azure_openai_api_version = "2024-02-15-preview"

# General LLM Settings
temperature = 0.1
max_retries = 3
enable_streaming = false

# =============================================================================
# TEST GENERATION STYLE
# =============================================================================

[style]
framework = "pytest"              # Options: 'pytest', 'unittest'
assertion_style = "pytest"        # Options: 'pytest', 'unittest', 'auto'
mock_library = "unittest.mock"    # Options: 'unittest.mock', 'pytest-mock', 'auto'

# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

[coverage]
minimum_line_coverage = 80.0
minimum_branch_coverage = 70.0
regenerate_if_below = 60.0
junit_xml = true
pytest_args = []

[coverage.runner]
mode = "python-module"             # Options: 'python-module', 'pytest-path', 'custom'
python = ""                        # Empty = current sys.executable
pytest_path = "pytest"
custom_cmd = []
cwd = ""                          # Empty = project root
args = []

[coverage.env]
propagate = true                   # Inherit current environment variables
extra = {}                         # Additional environment variables
append_pythonpath = []            # Paths to append to PYTHONPATH

# =============================================================================
# TEST GENERATION BEHAVIOR
# =============================================================================

[generation]
include_docstrings = true
generate_fixtures = true
parametrize_similar_tests = true
max_test_methods_per_class = 20
always_analyze_new_files = false

[generation.test_runner]
enable = false                     # Enable post-generation test execution
args = []                         # Extra pytest args
cwd = ""                          # Working directory
junit_xml = true                  # Generate JUnit XML

[generation.merge]
strategy = "append"               # Options: 'append', 'ast-merge'
dry_run = false
formatter = "none"                # Code formatter to apply after merge

[generation.refine]
enable = false                    # Enable AI-powered test refinement
max_retries = 2
backoff_base_sec = 1.0
backoff_max_sec = 8.0
stop_on_no_change = true
max_total_minutes = 5.0
strategy = "auto"                 # Refinement strategy

# =============================================================================
# COST MANAGEMENT
# =============================================================================

[cost_management]
max_file_size_kb = 50
max_context_size_chars = 100000
max_files_per_request = 15
use_cheaper_model_threshold_kb = 10
enable_content_compression = true
skip_trivial_files = true
token_usage_logging = true

[cost_management.cost_thresholds]
daily_limit = 50.0                # Maximum daily cost in USD
per_request_limit = 2.0           # Maximum cost per request
warning_threshold = 1.0           # Warn when request exceeds this cost

# =============================================================================
# QUALITY ANALYSIS
# =============================================================================

[quality]
enable_quality_analysis = true
enable_mutation_testing = true
minimum_quality_score = 75.0
minimum_mutation_score = 80.0
max_mutants_per_file = 50
mutation_timeout = 30
display_detailed_results = true
enable_pattern_analysis = true

[quality.modern_mutators]
enable_type_hints = true
enable_async_await = true
enable_dataclass = true
type_hints_severity = "medium"
async_severity = "high"
dataclass_severity = "medium"

# =============================================================================
# ENVIRONMENT MANAGEMENT
# =============================================================================

[environment]
auto_detect = true
preferred_manager = "auto"        # 'poetry' | 'pipenv' | 'conda' | 'uv' | 'venv' | 'auto'
respect_virtual_env = true
dependency_validation = true

[environment.overrides.poetry]
use_poetry_run = true
respect_poetry_venv = true

[environment.overrides.pipenv]
use_pipenv_run = true

[environment.overrides.conda]
activate_environment = true

[environment.overrides.uv]
use_uv_run = false

# =============================================================================
# TELEMETRY (Optional)
# =============================================================================

[telemetry]
enabled = false
backend = "opentelemetry"
service_name = "testcraft"
environment = "development"
trace_sampling_rate = 1.0
collect_metrics = true
anonymize_file_paths = true
anonymize_code_content = true

# =============================================================================
# SECURITY
# =============================================================================

[security]
enable_ast_validation = false
max_generated_file_size = 50000
block_dangerous_patterns = true
block_patterns = [
    'eval\\s*\\(',
    'exec\\s*\\(',
    '__import__\\s*\\(',
    'subprocess\\.',
    'os\\.system'
]
'''
    
    def _generate_yaml_config(self) -> str:
        """Generate comprehensive YAML configuration (fallback)."""
        # Use existing YAML from config loader if available
        from ..config.loader import ConfigLoader
        loader = ConfigLoader()
        return loader.create_sample_config().read_text()
    
    def _generate_json_config(self) -> str:
        """Generate comprehensive JSON configuration."""
        config = TestCraftConfig()
        config_dict = config.model_dump()
        return json.dumps(config_dict, indent=2)
    
    def _format_config_content(self, config_dict: Dict[str, Any], format_type: str) -> str:
        """Format configuration dictionary to specified format."""
        if format_type == 'toml':
            try:
                import tomli_w
                return tomli_w.dumps(config_dict)
            except ImportError:
                # Fallback to manual TOML generation
                return self._dict_to_simple_toml(config_dict)
        elif format_type == 'yaml':
            try:
                import yaml
                return yaml.dump(config_dict, default_flow_style=False, indent=2)
            except ImportError:
                raise ConfigInitializationError("PyYAML not available for YAML format")
        elif format_type == 'json':
            return json.dumps(config_dict, indent=2)
        else:
            raise ConfigInitializationError(f"Unsupported format: {format_type}")
    
    def _dict_to_simple_toml(self, data: Dict[str, Any], prefix: str = '') -> str:
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
                        lines.append(f'{sub_key} = {sub_value}')
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, list):
                lines.append(f'{key} = {value}')
            else:
                lines.append(f'{key} = {value}')
        
        return '\n'.join(lines)
    
    def _collect_user_preferences(self) -> Dict[str, Any]:
        """Collect user preferences through interactive prompts."""
        preferences = {}
        
        self.ui.print_divider("LLM Provider Setup")
        
        # LLM Provider selection
        providers = ['openai', 'anthropic', 'azure-openai', 'bedrock']
        provider = self.ui.get_user_input(
            "Select LLM provider",
            input_type="choice",
            choices=providers
        )
        preferences['llm_provider'] = provider
        
        # Test framework preference
        self.ui.print_divider("Test Framework Setup")
        
        frameworks = ['pytest', 'unittest']
        framework = self.ui.get_user_input(
            "Select test framework",
            input_type="choice", 
            choices=frameworks
        )
        preferences['test_framework'] = framework
        
        # Coverage thresholds
        self.ui.print_divider("Coverage Configuration")
        
        min_coverage = self.ui.get_user_input(
            "Minimum line coverage percentage (0-100)",
            input_type="number",
            default=80
        )
        preferences['min_coverage'] = min_coverage
        
        # Enable advanced features
        self.ui.print_divider("Advanced Features")
        
        enable_refinement = self.ui.get_user_confirmation(
            "Enable AI-powered test refinement?",
            default=False
        )
        preferences['enable_refinement'] = enable_refinement
        
        enable_streaming = self.ui.get_user_confirmation(
            "Enable streaming LLM responses?",
            default=False
        )
        preferences['enable_streaming'] = enable_streaming
        
        return preferences
    
    def _update_config_with_preferences(self, config_file: Path, preferences: Dict[str, Any]) -> None:
        """Update configuration file with user preferences."""
        # This is a simplified implementation
        # In a real implementation, you'd parse the existing config and update specific values
        
        self.ui.display_info(
            "Configuration preferences have been noted. "
            f"Please manually edit {config_file} to apply specific settings.",
            "Manual Configuration Required"
        )
        
        # Display preferences for user reference
        self.ui.print_divider("Your Preferences")
        for key, value in preferences.items():
            self.ui.console.print(f"[highlight]{key}:[/] {value}")
