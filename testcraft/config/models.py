"""Pydantic models for TestCraft configuration."""

from typing import Dict, Any, List, Optional, Literal, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TestPatternConfig(BaseModel):
    """Configuration for test file discovery patterns."""
    
    test_patterns: List[str] = Field(
        default=[
            'test_*.py',
            '*_test.py',
            'tests/**/test_*.py'
        ],
        description="Patterns for finding test files (supports glob patterns)"
    )
    
    exclude: List[str] = Field(
        default=[
            'migrations/*',
            '*/deprecated/*',
            '__pycache__/*',
            '*.pyc'
        ],
        description="Files and patterns to exclude from test generation"
    )
    
    exclude_dirs: List[str] = Field(
        default=[
            # Virtual environments
            'venv', 'env', '.env', '.venv', 'virtualenv',
            'ENV', 'env.bak', 'venv.bak',
            # Poetry virtual environments
            '.venv-*', 'poetry-*',
            # Conda environments
            'conda-meta', 'envs',
            # Pipenv environments
            '.venv-*',
            # Python build directories
            'build', 'dist', '*.egg-info', '*.dist-info',
            'pip-wheel-metadata', 'pip-build-env',
            # Cache directories
            '__pycache__', '.pytest_cache', '.coverage',
            '.cache', '.mypy_cache', '.ruff_cache',
            '.tox', '.nox', '.hypothesis',
            # IDE and editor directories
            '.vscode', '.idea', '.vs', '.atom',
            '.sublime-project', '.sublime-workspace',
            # Version control
            '.git', '.hg', '.svn', '.bzr',
            # Package managers
            'node_modules', 'bower_components',
            # Documentation build
            'docs/_build', '_build', 'site',
            # Testing and CI
            '.pytest_cache', '.coverage', 'htmlcov',
            '.stestr', '.testrepository',
            # Test generation artifacts - IMPORTANT: Exclude from LLM context
            '.artifacts',
            # Python-specific directories
            'site-packages', 'lib', 'lib64', 'include',
            'bin', 'Scripts', 'share', 'pyvenv.cfg',
            # OS-specific
            '.DS_Store', 'Thumbs.db',
            # Temporary directories
            'tmp', 'temp', '.tmp', '.temp',
            # Legacy Python
            'lib2to3', 'test', 'tests',
            # Jupyter
            '.ipynb_checkpoints',
            # Docker
            '.dockerignore', 'docker-compose.override.yml'
        ],
        description="Directories to exclude from scanning"
    )


class TestStyleConfig(BaseModel):
    """Configuration for test generation style."""
    
    framework: Literal['pytest', 'unittest'] = Field(
        default='pytest',
        description="Testing framework to use"
    )
    
    assertion_style: Literal['pytest', 'unittest', 'auto'] = Field(
        default='pytest',
        description="Assertion style to use in generated tests"
    )
    
    mock_library: Literal['unittest.mock', 'pytest-mock', 'auto'] = Field(
        default='unittest.mock',
        description="Mock library to use in generated tests"
    )


class TestRunnerConfig(BaseModel):
    """Configuration for test runner execution."""
    
    mode: Literal['python-module', 'pytest-path', 'custom'] = Field(
        default='python-module',
        description="Test runner execution mode"
    )
    
    python: Optional[str] = Field(
        default=None,
        description="Python executable path (None = current sys.executable)"
    )
    
    pytest_path: str = Field(
        default='pytest',
        description="Path to pytest when mode is 'pytest-path'"
    )
    
    custom_cmd: List[str] = Field(
        default_factory=list,
        description="Custom command when mode is 'custom'"
    )
    
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory (None = project root)"
    )
    
    args: List[str] = Field(
        default_factory=list,
        description="Runner-specific args before pytest_args"
    )


class TestEnvironmentConfig(BaseModel):
    """Configuration for test execution environment."""
    
    propagate: bool = Field(
        default=True,
        description="Inherit current environment variables"
    )
    
    extra: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables"
    )
    
    append_pythonpath: List[str] = Field(
        default_factory=list,
        description="Paths to append to PYTHONPATH"
    )


class CoverageConfig(BaseModel):
    """Configuration for test coverage analysis."""
    
    minimum_line_coverage: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum line coverage percentage"
    )
    
    minimum_branch_coverage: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum branch coverage percentage"
    )
    
    regenerate_if_below: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Regenerate tests if coverage drops below this"
    )
    
    pytest_args: List[str] = Field(
        default_factory=list,
        description="Extra arguments appended to pytest command"
    )
    
    junit_xml: bool = Field(
        default=True,
        description="Enable JUnit XML for all coverage runs"
    )
    
    runner: TestRunnerConfig = Field(
        default_factory=TestRunnerConfig,
        description="Test runner configuration"
    )
    
    env: TestEnvironmentConfig = Field(
        default_factory=TestEnvironmentConfig,
        description="Environment configuration for test runs"
    )


class MergeConfig(BaseModel):
    """Configuration for test merging strategies."""
    
    strategy: Literal['append', 'ast-merge'] = Field(
        default='append',
        description="Test merging strategy"
    )
    
    dry_run: bool = Field(
        default=False,
        description="Preview changes without applying"
    )
    
    formatter: str = Field(
        default='none',
        description="Code formatter to apply after merge"
    )


class PostGenerationTestRunnerConfig(BaseModel):
    """Configuration for post-generation test execution."""
    
    enable: bool = Field(
        default=False,
        description="Enable post-generation test execution"
    )
    
    args: List[str] = Field(
        default_factory=list,
        description="Extra pytest args"
    )
    
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory (None = project root)"
    )
    
    junit_xml: bool = Field(
        default=True,
        description="Enable JUnit XML for reliable failure parsing"
    )


class RefineConfig(BaseModel):
    """Configuration for AI-powered test refinement."""
    
    enable: bool = Field(
        default=False,
        description="Enable AI-powered test refinement"
    )
    
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum refinement attempts"
    )
    
    backoff_base_sec: float = Field(
        default=1.0,
        ge=0.1,
        description="Base delay between refinement attempts"
    )
    
    backoff_max_sec: float = Field(
        default=8.0,
        ge=1.0,
        description="Maximum delay between attempts"
    )
    
    stop_on_no_change: bool = Field(
        default=True,
        description="Stop if LLM returns no changes"
    )
    
    max_total_minutes: float = Field(
        default=5.0,
        ge=0.1,
        description="Maximum total time for refinement"
    )
    
    strategy: Literal[
        'auto', 'comprehensive', 'balanced', 
        'dependency_focused', 'logic_focused', 'setup_focused'
    ] = Field(
        default='auto',
        description="Refinement strategy"
    )


class TestGenerationConfig(BaseModel):
    """Configuration for test generation behavior."""
    
    include_docstrings: Union[bool, Literal['minimal']] = Field(
        default=True,
        description="Include docstrings in test methods"
    )
    
    generate_fixtures: bool = Field(
        default=True,
        description="Generate pytest fixtures for common setup"
    )
    
    parametrize_similar_tests: bool = Field(
        default=True,
        description="Use @pytest.mark.parametrize for similar tests"
    )
    
    max_test_methods_per_class: int = Field(
        default=20,
        ge=0,
        description="Maximum test methods per class (0 for unlimited)"
    )
    
    always_analyze_new_files: bool = Field(
        default=False,
        description="Always analyze new files even if they have tests"
    )
    
    test_runner: PostGenerationTestRunnerConfig = Field(
        default_factory=PostGenerationTestRunnerConfig,
        description="Post-generation test runner configuration"
    )
    
    merge: MergeConfig = Field(
        default_factory=MergeConfig,
        description="Test merging configuration"
    )
    
    refine: RefineConfig = Field(
        default_factory=RefineConfig,
        description="Test refinement configuration"
    )


class EnvironmentOverrideConfig(BaseModel):
    """Configuration for environment-specific overrides."""
    
    poetry: Dict[str, Any] = Field(
        default={
            'use_poetry_run': True,
            'respect_poetry_venv': True
        },
        description="Poetry-specific settings"
    )
    
    pipenv: Dict[str, Any] = Field(
        default={
            'use_pipenv_run': True
        },
        description="Pipenv-specific settings"
    )
    
    conda: Dict[str, Any] = Field(
        default={
            'activate_environment': True
        },
        description="Conda-specific settings"
    )
    
    uv: Dict[str, Any] = Field(
        default={
            'use_uv_run': False
        },
        description="UV-specific settings"
    )


class EnvironmentConfig(BaseModel):
    """Configuration for environment detection and management."""
    
    auto_detect: bool = Field(
        default=True,
        description="Auto-detect current environment manager"
    )
    
    preferred_manager: Literal['poetry', 'pipenv', 'conda', 'uv', 'venv', 'auto'] = Field(
        default='auto',
        description="Preferred environment manager"
    )
    
    respect_virtual_env: bool = Field(
        default=True,
        description="Always use current virtual env"
    )
    
    dependency_validation: bool = Field(
        default=True,
        description="Validate deps before running tests"
    )
    
    overrides: EnvironmentOverrideConfig = Field(
        default_factory=EnvironmentOverrideConfig,
        description="Environment-specific overrides"
    )


class CostThresholdConfig(BaseModel):
    """Configuration for cost thresholds and limits."""
    
    daily_limit: float = Field(
        default=50.0,
        ge=0.0,
        description="Maximum daily cost in USD"
    )
    
    per_request_limit: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum cost per request in USD"
    )
    
    warning_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Warn when request exceeds this cost"
    )


class CostConfig(BaseModel):
    """Configuration for cost management and optimization."""
    
    max_file_size_kb: int = Field(
        default=50,
        ge=1,
        description="Skip files larger than this (KB)"
    )
    
    max_context_size_chars: int = Field(
        default=100000,
        ge=1000,
        description="Limit total context size"
    )
    
    max_files_per_request: int = Field(
        default=15,
        ge=1,
        description="Override batch size for large files"
    )
    
    use_cheaper_model_threshold_kb: int = Field(
        default=10,
        ge=1,
        description="Use cheaper model for files < this size"
    )
    
    enable_content_compression: bool = Field(
        default=True,
        description="Remove comments/whitespace in prompts"
    )
    
    cost_thresholds: CostThresholdConfig = Field(
        default_factory=CostThresholdConfig,
        description="Cost thresholds and limits"
    )
    
    skip_trivial_files: bool = Field(
        default=True,
        description="Skip files with < 5 functions/classes"
    )
    
    token_usage_logging: bool = Field(
        default=True,
        description="Log token usage for cost tracking"
    )


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    
    enable_ast_validation: bool = Field(
        default=False,
        description="Use AST validation (slower but more secure)"
    )
    
    max_generated_file_size: int = Field(
        default=50000,
        ge=1000,
        description="Maximum size for generated test files (bytes)"
    )
    
    block_dangerous_patterns: bool = Field(
        default=True,
        description="Block potentially dangerous code patterns"
    )
    
    block_patterns: List[str] = Field(
        default_factory=lambda: [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
        ],
        description="Patterns to block in generated code"
    )


class ModernMutatorConfig(BaseModel):
    """Configuration for modern Python mutators."""
    
    enable_type_hints: bool = Field(
        default=True,
        description="Enable type hint mutations"
    )
    
    enable_async_await: bool = Field(
        default=True,
        description="Enable async/await mutations"
    )
    
    enable_dataclass: bool = Field(
        default=True,
        description="Enable dataclass mutations"
    )
    
    type_hints_severity: Literal['low', 'medium', 'high'] = Field(
        default='medium',
        description="Severity for type hint mutations"
    )
    
    async_severity: Literal['low', 'medium', 'high'] = Field(
        default='high',
        description="Async mutations severity"
    )
    
    dataclass_severity: Literal['low', 'medium', 'high'] = Field(
        default='medium',
        description="Dataclass mutations severity"
    )


class QualityConfig(BaseModel):
    """Configuration for test quality analysis."""
    
    enable_quality_analysis: bool = Field(
        default=True,
        description="Enable quality analysis by default"
    )
    
    enable_mutation_testing: bool = Field(
        default=True,
        description="Enable mutation testing by default"
    )
    
    minimum_quality_score: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable quality score (%)"
    )
    
    minimum_mutation_score: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable mutation score (%)"
    )
    
    max_mutants_per_file: int = Field(
        default=50,
        ge=1,
        description="Maximum mutants per file for performance"
    )
    
    mutation_timeout: int = Field(
        default=30,
        ge=1,
        description="Timeout in seconds for mutation testing"
    )
    
    display_detailed_results: bool = Field(
        default=True,
        description="Show detailed quality analysis results"
    )
    
    enable_pattern_analysis: bool = Field(
        default=True,
        description="Enable failure pattern analysis for smart refinement"
    )
    
    modern_mutators: ModernMutatorConfig = Field(
        default_factory=ModernMutatorConfig,
        description="Modern Python mutators configuration"
    )


class PromptEngineeringConfig(BaseModel):
    """Configuration for prompt engineering settings."""
    
    use_2025_guidelines: bool = Field(
        default=True,
        description="Use improved prompts following latest best practices"
    )
    
    encourage_step_by_step: bool = Field(
        default=True,
        description="Include step-by-step reasoning prompts (legacy)"
    )
    
    use_positive_negative_examples: bool = Field(
        default=True,
        description="Include positive/negative examples in prompts"
    )
    
    minimize_xml_structure: bool = Field(
        default=True,
        description="Reduce excessive XML tags in prompts"
    )
    
    decisive_recommendations: bool = Field(
        default=True,
        description="Encourage single, strong recommendations"
    )
    
    preserve_uncertainty: bool = Field(
        default=False,
        description="Whether to include hedging language"
    )
    
    use_enhanced_reasoning: bool = Field(
        default=True,
        description="Use advanced Chain-of-Thought reasoning"
    )
    
    enable_self_debugging: bool = Field(
        default=True,
        description="Enable self-debugging and review checkpoints"
    )
    
    use_enhanced_examples: bool = Field(
        default=True,
        description="Use detailed examples with reasoning"
    )
    
    enable_failure_strategies: bool = Field(
        default=True,
        description="Use failure-specific debugging strategies"
    )
    
    confidence_based_adaptation: bool = Field(
        default=True,
        description="Adapt prompts based on confidence levels"
    )
    
    track_reasoning_quality: bool = Field(
        default=True,
        description="Monitor and track reasoning quality"
    )


class ContextConfig(BaseModel):
    """Configuration for context retrieval and processing."""
    
    retrieval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context retrieval settings"
    )
    
    hybrid_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights for hybrid search"
    )
    
    rerank_model: Optional[str] = Field(
        default=None,
        description="Model to use for reranking"
    )
    
    hyde: bool = Field(
        default=False,
        description="Enable HyDE (Hypothetical Document Embeddings)"
    )


class TestCraftConfig(BaseModel):
    """Main configuration model for TestCraft."""
    
    # Test discovery and patterns
    test_patterns: TestPatternConfig = Field(
        default_factory=TestPatternConfig,
        description="Test file discovery patterns"
    )
    
    # Test generation style
    style: TestStyleConfig = Field(
        default_factory=TestStyleConfig,
        description="Test generation style configuration"
    )
    
    # Coverage analysis
    coverage: CoverageConfig = Field(
        default_factory=CoverageConfig,
        description="Test coverage configuration"
    )
    
    # Test generation behavior
    generation: TestGenerationConfig = Field(
        default_factory=TestGenerationConfig,
        description="Test generation behavior configuration"
    )
    
    # Environment management
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Environment detection and management"
    )
    
    # Cost management
    cost_management: CostConfig = Field(
        default_factory=CostConfig,
        description="Cost management and optimization"
    )
    
    # Security settings
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    
    # Quality analysis
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Test quality analysis configuration"
    )
    
    # Prompt engineering
    prompt_engineering: PromptEngineeringConfig = Field(
        default_factory=PromptEngineeringConfig,
        description="Prompt engineering configuration"
    )
    
    # Context processing
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context retrieval and processing configuration"
    )
    
    @field_validator('coverage')
    @classmethod
    def validate_coverage_thresholds(cls, v):
        """Ensure coverage thresholds are logically consistent."""
        if v.regenerate_if_below > v.minimum_line_coverage:
            raise ValueError(
                "regenerate_if_below cannot be higher than minimum_line_coverage"
            )
        if v.minimum_branch_coverage > v.minimum_line_coverage:
            raise ValueError(
                "minimum_branch_coverage cannot be higher than minimum_line_coverage"
            )
        return v
    
    @field_validator('cost_management')
    @classmethod
    def validate_cost_thresholds(cls, v):
        """Ensure cost thresholds are logically consistent."""
        thresholds = v.cost_thresholds
        if thresholds.warning_threshold > thresholds.per_request_limit:
            raise ValueError(
                "warning_threshold cannot be higher than per_request_limit"
            )
        return v
    
    @field_validator('quality')
    @classmethod
    def validate_quality_scores(cls, v):
        """Ensure quality scores are within valid ranges."""
        if v.minimum_quality_score > 100 or v.minimum_quality_score < 0:
            raise ValueError("minimum_quality_score must be between 0 and 100")
        if v.minimum_mutation_score > 100 or v.minimum_mutation_score < 0:
            raise ValueError("minimum_mutation_score must be between 0 and 100")
        return v
    
    def get_nested_value(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'coverage.minimum_line_coverage')."""
        keys = key.split('.')
        value = self.model_dump()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update_from_dict(self, updates: Dict[str, Any]) -> 'TestCraftConfig':
        """Update configuration with values from a dictionary."""
        current_dict = self.model_dump()
        updated_dict = self._deep_merge(current_dict, updates)
        return TestCraftConfig(**updated_dict)
    
    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """Deeply merge updates into base dictionary."""
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        use_enum_values=True
    )
