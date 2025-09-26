"""Pydantic models for TestCraft configuration."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestPatternConfig(BaseModel):
    """Configuration for test file discovery patterns."""

    test_patterns: list[str] = Field(
        default=["test_*.py", "*_test.py", "tests/**/test_*.py"],
        description="Patterns for finding test files (supports glob patterns)",
    )

    exclude: list[str] = Field(
        default=["migrations/*", "*/deprecated/*", "__pycache__/*", "*.pyc"],
        description="Files and patterns to exclude from test generation",
    )

    exclude_dirs: list[str] = Field(
        default=[
            # Virtual environments
            "venv",
            "env",
            ".env",
            ".venv",
            "virtualenv",
            "ENV",
            "env.bak",
            "venv.bak",
            # Poetry virtual environments
            ".venv-*",
            "poetry-*",
            # Conda environments
            "conda-meta",
            "envs",
            # Pipenv environments
            ".venv-*",
            # UV environments (modern package manager)
            ".uv-cache",
            # Python build directories
            "build",
            "dist",
            "*.egg-info",
            "*.dist-info",
            "pip-wheel-metadata",
            "pip-build-env",
            # Cache directories
            "__pycache__",
            ".pytest_cache",
            ".coverage",
            ".cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".nox",
            ".hypothesis",
            # IDE and editor directories
            ".vscode",
            ".idea",
            ".vs",
            ".atom",
            ".sublime-project",
            ".sublime-workspace",
            # Version control
            ".git",
            ".hg",
            ".svn",
            ".bzr",
            # Package managers
            "node_modules",
            "bower_components",
            # Documentation build
            "docs/_build",
            "_build",
            "site",
            # Testing and CI
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".stestr",
            ".testrepository",
            # Test generation artifacts - IMPORTANT: Exclude from LLM context
            ".artifacts",
            # Python-specific directories that are commonly in virtual environments
            "site-packages",
            "lib",
            "lib64",
            "include",
            "bin",
            "Scripts",
            "share",
            "pyvenv.cfg",
            "Lib",  # Windows Python virtual env
            "local",  # Unix virtual env symlink directory
            # Third-party package directories (common in virtual environments)
            "pkg_resources",
            "pip",
            "setuptools",
            "wheel",
            "distutils-precedence.pth",
            # OS-specific
            ".DS_Store",
            "Thumbs.db",
            # Temporary directories
            "tmp",
            "temp",
            ".tmp",
            ".temp",
            # Legacy Python
            "lib2to3",
            # Common test directories (only exclude if at root level via logic)
            "test",
            "tests",
            # Jupyter
            ".ipynb_checkpoints",
            # Docker
            ".dockerignore",
            "docker-compose.override.yml",
            # Package management files and directories
            ".pipfile",
            ".poetry",
            ".pdm-build",
            # Additional virtual environment indicators
            "pyvenv.cfg",
            "conda-meta",
            # Common CI/CD directories
            ".github",
            ".gitlab",
            ".circleci",
            ".travis",
            # Documentation directories that might contain large files
            "docs",
            "_static",
            "_templates",
        ],
        description="Directories to exclude from scanning",
    )


class TestStyleConfig(BaseModel):
    """Configuration for test generation style."""

    framework: Literal["pytest", "unittest"] = Field(
        default="pytest", description="Testing framework to use"
    )

    assertion_style: Literal["pytest", "unittest", "auto"] = Field(
        default="pytest", description="Assertion style to use in generated tests"
    )

    mock_library: Literal["unittest.mock", "pytest-mock", "auto"] = Field(
        default="unittest.mock", description="Mock library to use in generated tests"
    )


class TestRunnerConfig(BaseModel):
    """Configuration for test runner execution."""

    mode: Literal["python-module", "pytest-path", "custom"] = Field(
        default="python-module", description="Test runner execution mode"
    )

    python: str | None = Field(
        default=None,
        description="Python executable path (None = current sys.executable)",
    )

    pytest_path: str = Field(
        default="pytest", description="Path to pytest when mode is 'pytest-path'"
    )

    custom_cmd: list[str] = Field(
        default_factory=list, description="Custom command when mode is 'custom'"
    )

    cwd: str | None = Field(
        default=None, description="Working directory (None = project root)"
    )

    args: list[str] = Field(
        default_factory=list, description="Runner-specific args before pytest_args"
    )


class TestEnvironmentConfig(BaseModel):
    """Configuration for test execution environment."""

    propagate: bool = Field(
        default=True, description="Inherit current environment variables"
    )

    extra: dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )

    append_pythonpath: list[str] = Field(
        default_factory=list, description="Paths to append to PYTHONPATH"
    )


class CoverageConfig(BaseModel):
    """Configuration for test coverage analysis."""

    minimum_line_coverage: float = Field(
        default=80.0, ge=0.0, le=100.0, description="Minimum line coverage percentage"
    )

    minimum_branch_coverage: float = Field(
        default=70.0, ge=0.0, le=100.0, description="Minimum branch coverage percentage"
    )

    regenerate_if_below: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Regenerate tests if coverage drops below this",
    )

    pytest_args: list[str] = Field(
        default_factory=list, description="Extra arguments appended to pytest command"
    )

    junit_xml: bool = Field(
        default=True, description="Enable JUnit XML for all coverage runs"
    )

    runner: TestRunnerConfig = Field(
        default_factory=TestRunnerConfig, description="Test runner configuration"
    )

    env: TestEnvironmentConfig = Field(
        default_factory=TestEnvironmentConfig,
        description="Environment configuration for test runs",
    )


class MergeConfig(BaseModel):
    """Configuration for test merging strategies."""

    strategy: Literal["append", "ast-merge"] = Field(
        default="append", description="Test merging strategy"
    )

    dry_run: bool = Field(default=False, description="Preview changes without applying")

    formatter: str = Field(
        default="none", description="Code formatter to apply after merge"
    )


class PostGenerationTestRunnerConfig(BaseModel):
    """Configuration for post-generation test execution."""

    enable: bool = Field(
        default=False, description="Enable post-generation test execution"
    )

    args: list[str] = Field(default_factory=list, description="Extra pytest args")

    cwd: str | None = Field(
        default=None, description="Working directory (None = project root)"
    )

    junit_xml: bool = Field(
        default=True, description="Enable JUnit XML for reliable failure parsing"
    )


class RefineConfig(BaseModel):
    """Configuration for AI-powered test refinement."""

    enable: bool = Field(default=False, description="Enable AI-powered test refinement")

    max_retries: int = Field(
        default=2, ge=0, le=10, description="Maximum refinement attempts"
    )

    backoff_base_sec: float = Field(
        default=1.0, ge=0.1, description="Base delay between refinement attempts"
    )

    backoff_max_sec: float = Field(
        default=8.0, ge=1.0, description="Maximum delay between attempts"
    )

    stop_on_no_change: bool = Field(
        default=True, description="Stop if LLM returns no changes"
    )

    max_total_minutes: float = Field(
        default=5.0, ge=0.1, description="Maximum total time for refinement"
    )

    # Immediate refinement configuration
    immediate_refinement: bool = Field(
        default=True, description="Enable immediate write-and-refine per file"
    )

    max_refine_workers: int = Field(
        default=2, ge=1, le=8, description="Limit concurrent pytest/refine workers"
    )

    keep_failed_writes: bool = Field(
        default=False,
        description="Keep test files that fail to write or have syntax errors",
    )

    refine_on_first_failure_only: bool = Field(
        default=True,
        description="Stop refinement at first pytest failure within a file",
    )

    refinement_backoff_sec: float = Field(
        default=0.2, ge=0.0, le=5.0, description="Backoff between refinement iterations"
    )

    # Strict refinement policies (new)
    strict_assertion_preservation: bool = Field(
        default=True,
        description="Prevent refinement from weakening test assertions to pass (detects production bugs)",
    )

    fail_on_xfail_markers: bool = Field(
        default=True,
        description="Treat tests marked with xfail as refinement failures (prevents masking bugs)",
    )

    allow_xfail_on_suspected_bugs: bool = Field(
        default=False,
        description="Allow adding xfail markers when production bugs are suspected (teams can opt-in)",
    )

    report_suspected_prod_bugs: bool = Field(
        default=True,
        description="Generate detailed reports when refinement suspects production bugs",
    )

    # Refinement guardrails configuration
    refinement_guardrails: dict[str, Any] = Field(
        default_factory=lambda: {
            "reject_empty": True,
            "reject_literal_none": True,
            "reject_identical": True,
            "validate_syntax": True,
            "format_on_refine": True,
        },
        description="Safety guardrails for refinement operations",
    )

    pytest_args_for_refinement: list[str] = Field(
        default=["-vv", "--tb=short", "-x"],
        description="Pytest arguments specifically for refinement runs",
    )

    # Content validation and equivalence checking
    allow_ast_equivalence_check: bool = Field(
        default=True,
        description="Enable AST-based semantic equivalence checking to detect meaningful vs cosmetic changes",
    )

    treat_cosmetic_as_no_change: bool = Field(
        default=True,
        description="Treat cosmetic-only changes (whitespace, formatting) as no change to avoid unnecessary iterations",
    )

    max_diff_hunks: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of diff hunks to include in logs and reports for readability",
    )

    # Failed test annotation configuration
    annotate_failed_tests: bool = Field(
        default=True,
        description="Annotate test files with fix instructions when refinement fails",
    )

    annotation_placement: Literal["top", "bottom"] = Field(
        default="top", description="Where to place failure annotations in test files"
    )

    annotation_include_failure_excerpt: bool = Field(
        default=True, description="Include trimmed failure output in annotations"
    )

    annotation_max_failure_chars: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Maximum characters of failure output to include in annotations",
    )

    annotation_style: Literal["docstring", "hash"] = Field(
        default="docstring",
        description="Style of annotation: docstring (triple quotes) or hash (comment lines)",
    )

    include_llm_fix_instructions: bool = Field(
        default=True, description="Include LLM fix instructions in failure annotations"
    )

    # Import path resolution and targeting
    prefer_runtime_import_paths: bool = Field(
        default=True,
        description="Prefer import paths extracted from error traces over source tree aliases for better mocking",
    )

    # Timeout and hang prevention
    enable_timeout_detection: bool = Field(
        default=True,
        description="Enable timeout detection and classification for hanging tests",
    )

    timeout_threshold_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Threshold for classifying test executions as hanging/timing out",
    )

    # Schema validation and repair
    enable_schema_repair: bool = Field(
        default=True,
        description="Enable single-shot LLM schema repair for malformed refinement outputs",
    )

    schema_repair_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for schema repair prompts (lower = more deterministic)",
    )

    # Preflight analysis
    enable_preflight_analysis: bool = Field(
        default=True,
        description="Enable preflight canonicalization analysis to provide proactive suggestions to LLM",
    )


class OrchestratorConfig(BaseModel):
    """Configuration for LLM orchestrator behavior."""

    enable_manual_fix: bool = Field(
        default=True, description="Enable MANUAL FIX stage for suspected product bugs"
    )

    max_plan_retries: int = Field(
        default=2, ge=0, le=5, description="Maximum retries for PLAN stage"
    )

    max_refine_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retries for REFINE stage"
    )

    symbol_resolution_timeout: int = Field(
        default=30, ge=5, description="Timeout for symbol resolution in seconds"
    )

    enable_context_repacking: bool = Field(
        default=True, description="Enable context repacking when symbols are resolved"
    )

    enforce_canonical_imports: bool = Field(
        default=True, description="Enforce use of canonical imports in generated tests"
    )


class TestGenerationConfig(BaseModel):
    """Configuration for test generation behavior."""

    include_docstrings: bool | Literal["minimal"] = Field(
        default=True, description="Include docstrings in test methods"
    )

    generate_fixtures: bool = Field(
        default=True, description="Generate pytest fixtures for common setup"
    )

    parametrize_similar_tests: bool = Field(
        default=True, description="Use @pytest.mark.parametrize for similar tests"
    )

    max_test_methods_per_class: int = Field(
        default=20, ge=0, description="Maximum test methods per class (0 for unlimited)"
    )

    always_analyze_new_files: bool = Field(
        default=False, description="Always analyze new files even if they have tests"
    )

    test_runner: PostGenerationTestRunnerConfig = Field(
        default_factory=PostGenerationTestRunnerConfig,
        description="Post-generation test runner configuration",
    )

    merge: MergeConfig = Field(
        default_factory=MergeConfig, description="Test merging configuration"
    )

    # Quality gates
    enable_quality_gates: bool = Field(
        default=True, description="Enable quality gates validation before writing tests"
    )

    refine: RefineConfig = Field(
        default_factory=RefineConfig, description="Test refinement configuration"
    )

    orchestrator: OrchestratorConfig = Field(
        default_factory=OrchestratorConfig, description="LLM orchestrator configuration"
    )


class ContextEnrichmentConfig(BaseModel):
    """Configuration for context enrichment during test generation."""

    enable_env_detection: bool = Field(
        default=True, description="Enable environment variable usage detection"
    )

    enable_db_boundary_detection: bool = Field(
        default=True, description="Enable database client boundary detection"
    )

    enable_http_boundary_detection: bool = Field(
        default=True, description="Enable HTTP client boundary detection"
    )

    enable_comprehensive_fixtures: bool = Field(
        default=True, description="Enable comprehensive pytest fixture discovery"
    )

    enable_side_effect_detection: bool = Field(
        default=True, description="Enable side-effect boundary detection"
    )

    max_env_vars: int = Field(
        default=20, ge=1, le=100, description="Maximum environment variables to include"
    )

    max_fixtures: int = Field(
        default=15, ge=1, le=50, description="Maximum fixtures to include"
    )


class EnvironmentOverrideConfig(BaseModel):
    """Configuration for environment-specific overrides."""

    poetry: dict[str, Any] = Field(
        default={"use_poetry_run": True, "respect_poetry_venv": True},
        description="Poetry-specific settings",
    )

    pipenv: dict[str, Any] = Field(
        default={"use_pipenv_run": True}, description="Pipenv-specific settings"
    )

    conda: dict[str, Any] = Field(
        default={"activate_environment": True}, description="Conda-specific settings"
    )

    uv: dict[str, Any] = Field(
        default={"use_uv_run": False}, description="UV-specific settings"
    )


class EnvironmentConfig(BaseModel):
    """Configuration for environment detection and management."""

    auto_detect: bool = Field(
        default=True, description="Auto-detect current environment manager"
    )

    preferred_manager: Literal["poetry", "pipenv", "conda", "uv", "venv", "auto"] = (
        Field(default="auto", description="Preferred environment manager")
    )

    respect_virtual_env: bool = Field(
        default=True, description="Always use current virtual env"
    )

    dependency_validation: bool = Field(
        default=True, description="Validate deps before running tests"
    )

    overrides: EnvironmentOverrideConfig = Field(
        default_factory=EnvironmentOverrideConfig,
        description="Environment-specific overrides",
    )


class CostThresholdConfig(BaseModel):
    """Configuration for cost thresholds and limits."""

    daily_limit: float = Field(
        default=50.0, ge=0.0, description="Maximum daily cost in USD"
    )

    per_request_limit: float = Field(
        default=2.0, ge=0.0, description="Maximum cost per request in USD"
    )

    warning_threshold: float = Field(
        default=1.0, ge=0.0, description="Warn when request exceeds this cost"
    )


class CostConfig(BaseModel):
    """Configuration for cost management and optimization."""

    max_file_size_kb: int = Field(
        default=50, ge=1, description="Skip files larger than this (KB)"
    )

    max_context_size_chars: int = Field(
        default=100000, ge=1000, description="Limit total context size"
    )

    max_files_per_request: int = Field(
        default=15, ge=1, description="Override batch size for large files"
    )

    use_cheaper_model_threshold_kb: int = Field(
        default=10, ge=1, description="Use cheaper model for files < this size"
    )

    enable_content_compression: bool = Field(
        default=True, description="Remove comments/whitespace in prompts"
    )

    cost_thresholds: CostThresholdConfig = Field(
        default_factory=CostThresholdConfig, description="Cost thresholds and limits"
    )

    skip_trivial_files: bool = Field(
        default=True, description="Skip files with < 5 functions/classes"
    )

    token_usage_logging: bool = Field(
        default=True, description="Log token usage for cost tracking"
    )


class DeterminismConfig(BaseModel):
    """Configuration for determinism testing in quality gates."""

    seed: int = Field(default=42, description="Random seed for reproducible test runs")

    freeze_time: bool = Field(
        default=False, description="Freeze time during determinism testing"
    )

    tz: str = Field(default="UTC", description="Timezone for determinism testing")


class QualityConfig(BaseModel):
    """Configuration for test quality analysis."""

    enable_quality_analysis: bool = Field(
        default=True, description="Enable quality analysis by default"
    )

    enable_mutation_testing: bool = Field(
        default=True, description="Enable mutation testing by default"
    )

    minimum_quality_score: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable quality score (%)",
    )

    minimum_mutation_score: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable mutation score (%)",
    )

    max_mutants_per_file: int = Field(
        default=50, ge=1, description="Maximum mutants per file for performance"
    )

    # Quality gates configuration
    enable_import_gate: bool = Field(
        default=True, description="Enforce canonical import as first import"
    )

    enable_bootstrap_gate: bool = Field(
        default=True, description="Ensure bootstrap requirements are met"
    )

    enable_compile_gate: bool = Field(
        default=True, description="Ensure pytest can import generated tests"
    )

    enable_determinism_gate: bool = Field(
        default=True, description="Run pytest twice with same seed and compare"
    )

    enable_coverage_gate: bool = Field(
        default=True, description="Measure coverage delta and validate improvement"
    )

    enable_mutation_gate: bool = Field(
        default=False, description="Enable mutation sampling gate"
    )

    # Gate-specific settings
    determinism_timeout_seconds: int = Field(
        default=60, ge=10, description="Timeout for determinism testing"
    )

    compile_timeout_seconds: int = Field(
        default=30, ge=5, description="Timeout for compile testing"
    )

    import_timeout_seconds: int = Field(
        default=30, ge=5, description="Timeout for import validation"
    )

    mutation_timeout: int = Field(
        default=30, ge=1, description="Timeout in seconds for mutation testing"
    )

    display_detailed_results: bool = Field(
        default=True, description="Show detailed quality analysis results"
    )

    enable_pattern_analysis: bool = Field(
        default=True, description="Enable failure pattern analysis for smart refinement"
    )


"""
Note: security, prompt_engineering, and context settings were removed
because no runtime logic consumes them yet. Reintroduce only when wired.
"""


class TelemetryBackendConfig(BaseModel):
    """Configuration for specific telemetry backends."""

    opentelemetry: dict[str, Any] = Field(
        default_factory=lambda: {
            "endpoint": None,  # Auto-detect or use OTEL_EXPORTER_OTLP_ENDPOINT
            "headers": {},  # Additional headers for OTLP exporter
            "insecure": False,  # Use insecure gRPC connection
            "timeout": 10,  # Timeout for exports in seconds
        },
        description="OpenTelemetry-specific configuration",
    )

    datadog: dict[str, Any] = Field(
        default_factory=lambda: {
            "api_key": None,  # DD_API_KEY env var if None
            "site": "datadoghq.com",  # Datadog site
            "service": "testcraft",  # Service name
            "env": "development",  # Environment
            "version": None,  # Service version
        },
        description="Datadog-specific configuration",
    )

    jaeger: dict[str, Any] = Field(
        default_factory=lambda: {
            "endpoint": "http://localhost:14268/api/traces",
            "agent_host_name": "localhost",
            "agent_port": 6831,
        },
        description="Jaeger-specific configuration",
    )


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider settings."""

    # OpenAI Configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    openai_model: str = Field(
        default="gpt-4.1", description="OpenAI model to use for test generation"
    )
    openai_base_url: str | None = Field(
        default=None, description="Custom OpenAI API base URL (optional)"
    )
    openai_max_tokens: int = Field(
        default=12000,
        ge=100,
        le=16384,
        description="Maximum tokens for OpenAI requests",
    )
    openai_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for OpenAI requests (seconds)",
    )

    # Anthropic Claude Configuration
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4",
        description="Anthropic model to use for test generation",
    )
    anthropic_max_tokens: int = Field(
        default=100000,
        ge=100,
        le=128000,
        description="Maximum tokens for Anthropic requests",
    )
    anthropic_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Anthropic requests (seconds)",
    )

    # Azure OpenAI Configuration
    azure_openai_api_key: str | None = Field(
        default=None,
        description="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY environment variable)",
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT environment variable)",
    )
    azure_openai_deployment: str = Field(
        default="gpt-4.1", description="Azure OpenAI deployment name"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    azure_openai_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Azure OpenAI requests (seconds)",
    )

    # AWS Bedrock Configuration
    aws_region: str | None = Field(
        default=None,
        description="AWS region for Bedrock (or set AWS_REGION environment variable)",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID (or set AWS_ACCESS_KEY_ID environment variable)",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key (or set AWS_SECRET_ACCESS_KEY environment variable)",
    )
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-7-sonnet-v1:0", description="AWS Bedrock model ID"
    )
    bedrock_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Bedrock requests (seconds)",
    )

    # General LLM Settings
    default_provider: Literal["openai", "anthropic", "azure-openai", "bedrock"] = Field(
        default="openai", description="Default LLM provider to use"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retries for LLM requests"
    )
    enable_streaming: bool = Field(
        default=False, description="Enable streaming responses where supported"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (lower = more deterministic)",
    )

    # Beta/extended feature gating
    class BetaFeaturesConfig(BaseModel):
        anthropic_enable_extended_thinking: bool = False
        anthropic_enable_extended_output: bool = False
        openai_enable_extended_output: bool = False

    beta: BetaFeaturesConfig = Field(
        default_factory=BetaFeaturesConfig,
        description="Feature gates for beta/extended capabilities",
    )

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "azure_openai_api_key",
        "aws_secret_access_key",
    )
    @classmethod
    def validate_api_keys_not_empty(cls, v: Any) -> Any:
        """Ensure API keys are not empty strings."""
        if v is not None and v.strip() == "":
            return None
        return v

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_azure_endpoint(cls, v: Any) -> Any:
        """Validate Azure endpoint URL format."""
        if v is not None and v.strip() and not v.startswith(("http://", "https://")):
            raise ValueError(
                "Azure OpenAI endpoint must be a valid URL starting with http:// or https://"
            )
        return v


class EvaluationConfig(BaseModel):
    """Configuration for test evaluation harness."""

    enabled: bool = Field(
        default=False, description="Enable evaluation harness functionality"
    )

    golden_repos_path: str | None = Field(
        default=None, description="Path to golden repositories for regression testing"
    )

    # Acceptance checks configuration
    acceptance_checks: bool = Field(
        default=True,
        description="Enable automated acceptance checks (syntax, imports, pytest)",
    )

    # LLM-as-judge configuration
    llm_judge_enabled: bool = Field(
        default=True, description="Enable LLM-as-judge evaluation"
    )

    rubric_dimensions: list[str] = Field(
        default=["correctness", "coverage", "clarity", "safety"],
        description="Evaluation dimensions for LLM-as-judge",
    )

    # A/B testing and statistical analysis
    statistical_testing: bool = Field(
        default=True,
        description="Enable statistical significance testing for A/B comparisons",
    )

    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Statistical confidence level for A/B testing",
    )

    # Human review configuration
    human_review_enabled: bool = Field(
        default=False, description="Enable human-in-the-loop review"
    )

    # Artifact and state management
    artifacts_path: str = Field(
        default=".testcraft/evaluation_artifacts",
        description="Path for storing evaluation artifacts",
    )

    state_file: str = Field(
        default=".testcraft_evaluation_state.json",
        description="File for storing evaluation state",
    )

    # Evaluation timeouts and limits
    evaluation_timeout_seconds: int = Field(
        default=300, ge=10, le=3600, description="Timeout for individual evaluations"
    )

    batch_size: int = Field(
        default=10, ge=1, le=100, description="Batch size for A/B testing"
    )

    # Prompt registry configuration for evaluation
    prompt_version: str | None = Field(
        default=None,
        description="Specific prompt version for LLM-as-judge (None = latest)",
    )


class TelemetryConfig(BaseModel):
    """Configuration for telemetry and observability."""

    enabled: bool = Field(default=False, description="Enable telemetry collection")

    backend: Literal["opentelemetry", "datadog", "jaeger", "noop"] = Field(
        default="opentelemetry", description="Telemetry backend to use"
    )

    service_name: str = Field(
        default="testcraft", description="Service name for telemetry"
    )

    service_version: str | None = Field(
        default=None, description="Service version (auto-detected if None)"
    )

    environment: str = Field(
        default="development",
        description="Environment name (development, staging, production)",
    )

    # Tracing configuration
    trace_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate (0.0 to 1.0)"
    )

    capture_llm_calls: bool = Field(default=True, description="Trace LLM API calls")

    capture_coverage_runs: bool = Field(
        default=True, description="Trace coverage analysis operations"
    )

    capture_file_operations: bool = Field(
        default=True, description="Trace file read/write operations"
    )

    capture_test_generation: bool = Field(
        default=True, description="Trace test generation processes"
    )

    # Metrics configuration
    collect_metrics: bool = Field(default=True, description="Enable metrics collection")

    metrics_interval_seconds: int = Field(
        default=30, ge=1, description="Metrics collection interval"
    )

    track_token_usage: bool = Field(
        default=True, description="Track LLM token usage metrics"
    )

    track_coverage_delta: bool = Field(
        default=True, description="Track coverage improvement metrics"
    )

    track_test_pass_rate: bool = Field(
        default=True, description="Track test success/failure rates"
    )

    # Privacy and anonymization
    anonymize_file_paths: bool = Field(
        default=True, description="Hash file paths in telemetry data"
    )

    anonymize_code_content: bool = Field(
        default=True, description="Exclude actual code content from telemetry"
    )

    opt_out_data_collection: bool = Field(
        default=False,
        description="Completely disable data collection (overrides enabled)",
    )

    # Resource attributes
    global_attributes: dict[str, Any] = Field(
        default_factory=dict, description="Global attributes to attach to all telemetry"
    )

    # Backend-specific configurations
    backends: TelemetryBackendConfig = Field(
        default_factory=TelemetryBackendConfig,
        description="Backend-specific configuration",
    )

    @field_validator("trace_sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: Any) -> Any:
        """Ensure sampling rate is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("trace_sampling_rate must be between 0.0 and 1.0")
        return v


class TestCraftConfig(BaseModel):
    """Main configuration model for TestCraft."""

    # Test discovery and patterns
    test_patterns: TestPatternConfig = Field(
        default_factory=TestPatternConfig, description="Test file discovery patterns"
    )

    # Test generation style
    style: TestStyleConfig = Field(
        default_factory=TestStyleConfig,
        description="Test generation style configuration",
    )

    # Coverage analysis
    coverage: CoverageConfig = Field(
        default_factory=CoverageConfig, description="Test coverage configuration"
    )

    # Test generation behavior
    generation: TestGenerationConfig = Field(
        default_factory=TestGenerationConfig,
        description="Test generation behavior configuration",
    )

    # Context enrichment configuration
    context_enrichment: ContextEnrichmentConfig = Field(
        default_factory=ContextEnrichmentConfig,
        description="Context enrichment configuration",
    )

    # Environment management
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Environment detection and management",
    )

    # Cost management
    cost_management: CostConfig = Field(
        default_factory=CostConfig, description="Cost management and optimization"
    )

    # Removed unused sections: security, prompt_engineering, context

    # Quality analysis
    quality: QualityConfig = Field(
        default_factory=QualityConfig, description="Test quality analysis configuration"
    )

    # Telemetry and observability
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Telemetry and observability configuration",
    )

    # Evaluation harness configuration
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Test evaluation harness configuration",
    )

    # LLM providers and AI configuration
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="Large Language Model provider configuration",
    )

    @field_validator("coverage")
    @classmethod
    def validate_coverage_thresholds(cls, v: Any) -> Any:
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

    @field_validator("cost_management")
    @classmethod
    def validate_cost_thresholds(cls, v: Any) -> Any:
        """Ensure cost thresholds are logically consistent."""
        thresholds = v.cost_thresholds
        if thresholds.warning_threshold > thresholds.per_request_limit:
            raise ValueError(
                "warning_threshold cannot be higher than per_request_limit"
            )
        return v

    @field_validator("quality")
    @classmethod
    def validate_quality_scores(cls, v: Any) -> Any:
        """Ensure quality scores are within valid ranges."""
        if v.minimum_quality_score > 100 or v.minimum_quality_score < 0:
            raise ValueError("minimum_quality_score must be between 0 and 100")
        if v.minimum_mutation_score > 100 or v.minimum_mutation_score < 0:
            raise ValueError("minimum_mutation_score must be between 0 and 100")
        return v

    def get_nested_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'coverage.minimum_line_coverage')."""
        keys = key.split(".")
        value = self.model_dump()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update_from_dict(self, updates: dict[str, Any]) -> "TestCraftConfig":
        """Update configuration with values from a dictionary."""
        current_dict = self.model_dump()
        updated_dict = self._deep_merge(current_dict, updates)
        return TestCraftConfig(**updated_dict)

    def _deep_merge(self, base: dict, updates: dict) -> dict:
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

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )
