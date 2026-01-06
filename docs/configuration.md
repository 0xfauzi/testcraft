# TestCraft Configuration Reference

This document provides a comprehensive reference for all TestCraft configuration options. TestCraft uses TOML format for configuration files (`.testcraft.toml`).

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Test Patterns](#test-patterns)
3. [Test Style Configuration](#test-style-configuration)
4. [Coverage Analysis](#coverage-analysis)
5. [Test Generation](#test-generation)
6. [Planning Workflow](#planning-workflow)
7. [Evaluation Harness](#evaluation-harness)
8. [Environment Management](#environment-management)
9. [Cost Management](#cost-management)
10. [Quality Analysis](#quality-analysis)
11. [Enhanced Import System](#enhanced-import-system)
12. [Telemetry & Observability](#telemetry--observability)
13. [LLM Providers](#llm-providers)
14. [Environment Variable Overrides](#environment-variable-overrides)

## Configuration File Structure

TestCraft uses **TOML format only** for configuration files. The `testcraft init-config` command generates `.testcraft.toml`.

TestCraft looks for configuration files in the following order:

1. `.testcraft.toml` (project-specific, TOML only)
2. `pyproject.toml` (in `[tool.testcraft]` section, TOML only)
3. Global configuration in `~/.config/testcraft/config.toml`

**Note**: YAML and JSON formats are deprecated. Use TOML.

### Example Minimal Configuration

```toml
[style]
framework = "pytest"
assertion_style = "pytest"

[coverage]
minimum_line_coverage = 80.0

[llm]
default_provider = "openai"
openai_model = "gpt-4.1"
```

### Example Comprehensive Configuration

See [.testcraft-comprehensive.toml](.testcraft-comprehensive.toml) for a complete example with all options.

## Test Patterns

Configure how TestCraft discovers and generates test files.

```toml
[test_patterns]
# Patterns for finding test files (supports glob patterns)
test_patterns = ["test_*.py", "*_test.py", "tests/**/test_*.py"]

# Files and patterns to exclude from test generation
exclude = ["migrations/*", "*/deprecated/*", "__pycache__/*", "*.pyc"]

# Additional directories to exclude
exclude_dirs = [
    "venv", "env", ".env", ".venv", "virtualenv",  # Virtual environments
    "build", "dist", "*.egg-info",                 # Build directories
    "__pycache__", ".pytest_cache", ".coverage",   # Cache directories
    ".vscode", ".idea",                            # IDE directories
    ".git",                                        # Version control
    ".testcraft"                                   # TestCraft artifacts
]
```

### Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `test_patterns` | `List[str]` | `["test_*.py", "*_test.py"]` | Glob patterns for test files |
| `exclude` | `List[str]` | `[]` | File patterns to exclude |
| `exclude_dirs` | `List[str]` | Common exclusions | Directory patterns to exclude |

## Test Style Configuration

Configure the style and framework for generated tests.

```toml
[style]
framework = "pytest"              # Options: "pytest", "unittest"
assertion_style = "pytest"       # Options: "pytest", "unittest", "auto"
mock_library = "unittest.mock"   # Options: "unittest.mock", "pytest-mock", "auto"
```

### Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `framework` | `str` | `"pytest"` | Test framework to use |
| `assertion_style` | `str` | `"pytest"` | Assertion style for tests |
| `mock_library` | `str` | `"unittest.mock"` | Mocking library preference |

## Coverage Analysis

Configure coverage thresholds, measurement, and test execution.

```toml
[coverage]
# Coverage thresholds
minimum_line_coverage = 80.0      # Minimum line coverage percentage
minimum_branch_coverage = 70.0    # Minimum branch coverage percentage
regenerate_if_below = 60.0         # Regenerate tests if coverage drops below this

# Additional pytest arguments for coverage runs
pytest_args = ["-v", "--tb=short"]
junit_xml = true                   # Enable JUnit XML for all coverage runs

# Coverage measurement options (NEW)
include = []                       # Explicit source directories for coverage measurement
omit = [                          # Patterns to omit from coverage
    "*/tests/*",
    "*/test_*.py",
    "*_test.py",
    "*/.venv/*",
    "*/venv/*",
    "*/site-packages/*",
    "*/.tox/*",
]
data_dir = ".artifacts/coverage"   # Directory for coverage data files and reports
aggregate = true                   # Combine coverage data across multiple runs

# Test runner configuration
[coverage.runner]
mode = "python-module"             # Options: "python-module", "pytest-path", "custom"
python = ""                        # Python executable (empty = current sys.executable)
pytest_path = "pytest"            # Path to pytest when mode is "pytest-path"
custom_cmd = []                    # Custom command when mode is "custom"
cwd = ""                          # Working directory (empty = project root)
args = []                         # Runner-specific args before pytest_args

# Environment configuration for test runs
[coverage.env]
propagate = true                   # Inherit current environment variables
extra = {}                         # Additional environment variables
append_pythonpath = []             # Paths to append to PYTHONPATH
```

### Coverage Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `minimum_line_coverage` | `float` | `80.0` | Minimum acceptable line coverage |
| `minimum_branch_coverage` | `float` | `70.0` | Minimum acceptable branch coverage |
| `regenerate_if_below` | `float` | `60.0` | Threshold for test regeneration |
| `pytest_args` | `List[str]` | `[]` | Additional pytest arguments |
| `junit_xml` | `bool` | `true` | Generate JUnit XML output (informational in current pipeline) |
| `include` | `List[str]` | `[]` | Explicit source directories for measurement |
| `omit` | `List[str]` | Common exclusions | Patterns to omit from coverage |
| `data_dir` | `str` | `".artifacts/coverage"` | Directory for coverage artifacts |
| `aggregate` | `bool` | `true` | Combine coverage across multiple runs |

### Coverage CLI Usage

TestCraft provides a `coverage` command for measuring and reporting code coverage:

```bash
# Basic usage - measure coverage for entire project
testcraft coverage .

# Specify source files and tests explicitly
testcraft coverage . \
    --source-files src/module.py \
    --test-files tests/test_module.py

# Generate XML report (useful for CI/CD)
testcraft coverage . --format xml --output-dir .artifacts/coverage

# Multiple formats at once
testcraft coverage . --format xml --format json --format detailed

# Filter coverage with include/omit patterns
testcraft coverage . \
    --include src/ \
    --omit "*/migrations/*" \
    --omit "*/tests/*"

# Combine results from multiple runs (aggregation)
# Run tests covering different modules, all stored in same output-dir
testcraft coverage . --test-files tests/test_auth.py --output-dir .coverage-data
testcraft coverage . --test-files tests/test_api.py --output-dir .coverage-data
# View combined results
testcraft coverage . --format xml --output-dir .coverage-data
```

**Available Formats:**
- `detailed` - Detailed per-file coverage report (default)
- `summary` - Concise summary of overall coverage
- `json` - JSON format for programmatic processing
- `xml` - Cobertura XML format (for CI/CD integration)

Note: HTML output is intentionally not exposed in the CLI. Prefer XML for CI/CD systems and JSON/text for local inspection.

## Test Generation

Configure test generation behavior and post-processing.

```toml
[generation]
# Test content and structure options
include_docstrings = true          # Include docstrings in test methods
generate_fixtures = true           # Generate pytest fixtures for common setup
parametrize_similar_tests = true   # Use @pytest.mark.parametrize for similar tests
max_test_methods_per_class = 20    # Maximum test methods per class (0 for unlimited)
always_analyze_new_files = false   # Always analyze new files even if they have tests

# Post-generation test runner
[generation.test_runner]
enable = false                     # Enable post-generation test execution
args = ["-q", "-x"]               # Extra pytest args
cwd = ""                          # Working directory (empty = project root)
junit_xml = true                   # Generate JUnit XML for failure parsing

# Test merging strategy
[generation.merge]
strategy = "append"                # Options: "append", "ast-merge"
dry_run = false                    # Preview changes without applying
formatter = "black"                # Code formatter to apply after merge (informational; current writer manages formatting)

# Test refinement loop (AI-powered test fixing)
[generation.refine]
enable = false                     # Enable AI-powered test refinement
max_retries = 2                    # Maximum refinement attempts
backoff_base_sec = 1.0             # Base delay between refinement attempts
backoff_max_sec = 8.0              # Maximum delay between attempts
stop_on_no_change = true           # Stop if LLM returns no changes
max_total_minutes = 5.0            # Maximum total time for refinement

# Content validation and equivalence checking (NEW)
allow_ast_equivalence_check = true           # Enable AST-based semantic equivalence checking
treat_cosmetic_as_no_change = true           # Treat cosmetic changes as no change
max_diff_hunks = 3                           # Max diff hunks in logs/reports

# Import path resolution and targeting (NEW)
prefer_runtime_import_paths = true           # Prefer runtime import paths from error traces

# Timeout and hang prevention (NEW)
enable_timeout_detection = true              # Enable timeout detection and classification
timeout_threshold_seconds = 30.0             # Threshold for hanging test classification

# Schema validation and repair (NEW)
enable_schema_repair = true                  # Enable LLM schema repair for malformed outputs
schema_repair_temperature = 0.0              # Temperature for schema repair (0.0 = deterministic)

# Preflight analysis (NEW)
enable_preflight_analysis = true             # Enable preflight canonicalization analysis

# LLM Orchestrator configuration
[generation.orchestrator]
enable_manual_fix = true                     # Enable MANUAL FIX stage for suspected product bugs
max_plan_retries = 2                         # Maximum retries for PLAN stage
max_refine_retries = 3                       # Maximum retries for REFINE stage

# Manual Fix Guidance (NEW - Task 33)
[manual_fix]
enable = true                                # Enable manual-fix guidance features
on_fail = false                              # Auto-trigger after refinement failure (opt-in)
auto_accept = false                          # Auto-accept recommendations without prompt
output_dir = ".testcraft/manual_fixes"       # Directory for Markdown artifacts
model = ""                                   # Override model ID (optional, uses default if empty)
max_tokens = 8000                            # Max tokens for manual-fix guidance
temperature = 0.1                            # LLM temperature for manual-fix prompts
prompt_version = ""                          # Prompt version override (optional)
```

Note:
- The `generation.test_runner.*` block is currently not invoked by the generation pipeline; tests are executed by the refinement subsystem and coverage use case.
- The `generation.merge.*` options are informational at present; the writer performs AST merge/formatting internally.

### Manual Fix Guidance (NEW - Task 33)

TestCraft includes a Manual Fix Guidance system that triggers when test refinement exhausts all attempts and suspects a production bug. When enabled, it generates:

1. **Deliberately Failing Test**: A pytest module that will pass only after the code bug is fixed
2. **BUG NOTE**: Detailed Markdown documentation with root cause analysis, reproduction steps, and fix suggestions

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | `bool` | `true` | Enable manual-fix guidance features |
| `on_fail` | `bool` | `false` | Auto-trigger after refinement failure (opt-in for safety) |
| `auto_accept` | `bool` | `false` | Auto-accept recommendations without user prompt |
| `output_dir` | `str` | `".testcraft/manual_fixes"` | Directory for Markdown artifacts |
| `model` | `str?` | `null` | Override LLM model (uses default if not set) |
| `max_tokens` | `int` | `8000` | Maximum tokens for guidance generation |
| `temperature` | `float` | `0.1` | LLM temperature (0.0-2.0) |
| `prompt_version` | `str?` | `null` | Prompt version override (optional) |

**CLI Flags:**

```bash
# Enable manual-fix trigger on refinement failure
testcraft generate . --manual-fix-on-fail

# Auto-accept manual-fix recommendations
testcraft generate . --manual-fix-on-fail --auto-accept-fixes

# Standalone manual-fix command
testcraft manual-fix . --target-file src/module.py --target-object func \
  --trace-excerpt "AssertionError: ..." --auto-accept-fixes
```

**Artifact Structure:**

Artifacts are persisted as Markdown files in `<output_dir>/<target>_<hash>.md` with:
- Metadata header (target, hash, timestamp, canonical import, model info)
- Failing test code block (Python)
- BUG NOTE section (Markdown)

Deduplication is performed by content hash to avoid redundant artifacts.

### LLM Refinement Reliability Features (NEW)

TestCraft now includes advanced features to improve the reliability and effectiveness of AI-powered test refinement:

#### Content Equivalence Detection

The system now uses **layered content validation** to accurately detect when LLM changes are meaningful:

- **String Identity**: Basic string comparison (fastest)
- **Normalization**: Whitespace and line-ending normalization
- **Cosmetic Detection**: Identifies formatting-only changes (indentation, spacing)
- **AST Equivalence**: Semantic comparison using Python AST parsing

This prevents unnecessary refinement iterations when the LLM makes cosmetic or semantically equivalent changes.

#### Import Path Targeting

TestCraft now extracts **runtime import paths** from pytest failure traces to provide more accurate context to the LLM:

```python
# Instead of source tree aliases:
from myproject.modules.scheduler import JobScheduler  # Less reliable

# LLM now sees runtime paths from traces:
from myproject_package.scheduler import JobScheduler  # More reliable for mocking
```

#### Schema Validation and Repair

LLM responses now undergo strict JSON schema validation with automatic repair:

- **Required Fields**: `refined_content`, `changes_made`, `reason`
- **Optional Fields**: `suspected_prod_bug` (auto-added as `null`)
- **Single-Shot Repair**: If validation fails, a focused repair prompt attempts to fix the response
- **Type Coercion**: Automatic string-to-boolean conversion where appropriate

#### Preflight Analysis

Before sending code to the LLM, TestCraft performs **canonicalization analysis** to detect common Python issues:

- Incorrect dunder method casing (`__Init__` → `__init__`)
- Keyword casing issues (`False` vs `false`)
- Import statement formatting problems

These suggestions are included in the LLM prompt to prevent common errors.

#### Timeout and Hang Detection

TestCraft now detects and classifies timeouts in test execution:

- **Execution Time Analysis**: Warns when tests approach timeout thresholds
- **Pattern Detection**: Identifies `time.sleep()`, `input()`, and other blocking operations
- **Hang Classification**: Categorizes timeouts as immediate hangs vs. slow execution
- **Actionable Suggestions**: Provides specific guidance for stubbing problematic operations

### Generation Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_docstrings` | `bool` | `true` | Include docstrings in generated tests |
| `generate_fixtures` | `bool` | `true` | Create pytest fixtures for setup |
| `parametrize_similar_tests` | `bool` | `true` | Use parametrized tests where appropriate |
| `max_test_methods_per_class` | `int` | `20` | Limit test methods per class |

### Symbol Resolution Options

Symbol resolution drives the PLAN/REFINE retry loop when the LLM references
helpers that are missing from the current context. The following settings live
under `generation`:

- `enable_symbol_resolution` (`bool`, default `true`): toggles the
  missing-symbol retry loop entirely.
- `symbol_resolution.allow_runtime_imports` (`bool`, default `false`): when
  enabled, TestCraft may import the target module in a sandboxed subprocess to
  resolve dynamically assigned helpers that static parsing cannot find. Leave
  disabled for projects with import side effects.
- `symbol_resolution.runtime_timeout_sec` (`float`, default `10.0`): execution
  timeout for the optional runtime import fallback.

### Refinement Options Reference (NEW)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `allow_ast_equivalence_check` | `bool` | `true` | Enable AST-based semantic equivalence checking |
| `treat_cosmetic_as_no_change` | `bool` | `true` | Treat formatting-only changes as no change |
| `max_diff_hunks` | `int` | `3` | Maximum diff hunks to include in logs |
| `prefer_runtime_import_paths` | `bool` | `true` | Extract import paths from error traces |
| `enable_timeout_detection` | `bool` | `true` | Enable timeout detection and classification |
| `timeout_threshold_seconds` | `float` | `30.0` | Threshold for classifying hangs |
| `enable_schema_repair` | `bool` | `true` | Enable LLM schema validation and repair |
| `schema_repair_temperature` | `float` | `0.0` | Temperature for repair prompts |
| `enable_preflight_analysis` | `bool` | `true` | Enable canonicalization preflight checks |
| `iteration_timeout_sec` | `float` | `30.0` | Per-iteration timeout for refinement loops |

### Modular Configuration System

TestCraft uses a **centralized configuration system** that handles complex merging of defaults, user overrides, and nested settings. Configuration is managed through the `GenerationConfig` class in `testcraft/application/generation/config.py`.

#### Configuration Merging

The system supports deep merging for nested settings like `context_categories` and `prompt_budgets`:

```toml
[generation]
batch_size = 10
enable_context = true

[generation.context_categories]
snippets = true
neighbors = true
contracts = false

[generation.prompt_budgets]
per_item_chars = 2000
total_chars = 15000

[generation.prompt_budgets.section_caps]
snippets = 8
neighbors = 3
```

#### Context Enrichment Mapping

The legacy `context_enrichment` configuration is automatically mapped to the new `context_categories` system:

```toml
[generation.context_enrichment]
enable_env_detection = true
enable_db_boundary_detection = true
enable_side_effect_detection = false
```

This automatically enables corresponding context categories and preserves backward compatibility.

#### Configuration Validation

The system validates configuration values and provides sensible defaults for invalid settings:
- `batch_size` must be ≥ 1 (default: 5)
- `coverage_threshold` must be 0.0-1.0 (default: 0.8)
- `max_refinement_iterations` must be ≥ 1 (default: 3)
- Character limits must be reasonable values

## Planning Workflow

Configure the interactive planning workflow for test generation.

```toml
[planning]
enabled = true                    # Enable planning stage before test generation
auto_accept = false               # Auto-accept plans without user interaction
prompt_template = ""              # Custom prompt template for planning (uses default if empty)
max_retries = 2                   # Maximum retries for planning stage
```

### Planning Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable planning stage before generation |
| `auto_accept` | `bool` | `false` | Auto-accept plans without interactive review |
| `prompt_template` | `str?` | `null` | Custom prompt template (optional) |
| `max_retries` | `int` | `2` | Maximum retries for planning stage |

**CLI Flags:**

```bash
# Generate with planning workflow
testcraft generate . --plan-first

# Auto-accept plan without review
testcraft generate . --plan-first --auto-accept-plan
```

## Evaluation Harness

Configure the comprehensive evaluation system for test quality assessment.

```toml
[evaluation]
enabled = true                     # Enable evaluation harness functionality
golden_repos_path = "golden_repos" # Path to golden repositories for regression testing

# Acceptance checks configuration
acceptance_checks = true           # Enable automated acceptance checks (syntax, imports, pytest)

# LLM-as-judge configuration
llm_judge_enabled = true           # Enable LLM-as-judge evaluation
rubric_dimensions = ["correctness", "coverage", "clarity", "safety"]  # Evaluation dimensions

# A/B testing and statistical analysis
statistical_testing = true        # Enable statistical significance testing
confidence_level = 0.95           # Statistical confidence level (0.5-0.99)

# Human review configuration
human_review_enabled = false      # Enable human-in-the-loop review

# Artifact and state management
artifacts_path = ".testcraft/evaluation_artifacts"  # Path for storing evaluation artifacts
state_file = ".testcraft_evaluation_state.json"     # File for storing evaluation state

# Evaluation timeouts and limits
evaluation_timeout_seconds = 300  # Timeout for individual evaluations (10-3600)
batch_size = 10                   # Batch size for A/B testing (1-100)

# Prompt registry configuration for evaluation
prompt_version = ""               # Specific prompt version for LLM-as-judge (empty = latest)
```

### Evaluation Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable evaluation harness |
| `acceptance_checks` | `bool` | `true` | Run automated acceptance checks |
| `llm_judge_enabled` | `bool` | `false` | Enable LLM-based quality assessment |
| `rubric_dimensions` | `List[str]` | See above | Evaluation criteria |
| `statistical_testing` | `bool` | `true` | Enable statistical analysis |
| `confidence_level` | `float` | `0.95` | Statistical confidence level |
| `evaluation_timeout_seconds` | `int` | `300` | Timeout for evaluations |

### Rubric Dimensions

Available rubric dimensions for LLM-as-judge evaluation:

- **correctness**: Test logic accuracy and validity
- **coverage**: Completeness of test coverage
- **clarity**: Code readability and maintainability
- **safety**: Error handling and edge case coverage
- **maintainability**: Long-term code quality
- **performance**: Test execution efficiency
- **integration**: Integration test quality

## Environment Management

Configure environment detection and dependency management.

```toml
[environment]
# Environment detection settings
auto_detect = true                 # Auto-detect current environment manager
preferred_manager = "auto"         # Options: "poetry", "pipenv", "conda", "uv", "venv", "auto"
respect_virtual_env = true         # Always use current virtual env
dependency_validation = true       # Validate deps before running tests

# Environment-specific overrides
[environment.overrides.poetry]
use_poetry_run = true              # Use `poetry run pytest` instead of direct python
respect_poetry_venv = true

[environment.overrides.pipenv]
use_pipenv_run = true              # Use `pipenv run pytest`

[environment.overrides.conda]
activate_environment = true       # Ensure conda environment is active

[environment.overrides.uv]
use_uv_run = false                # Use direct python instead of `uv run`
```

### Environment Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_detect` | `bool` | `true` | Automatically detect environment manager |
| `preferred_manager` | `str` | `"auto"` | Preferred environment manager |
| `respect_virtual_env` | `bool` | `true` | Use current virtual environment |
| `dependency_validation` | `bool` | `true` | Validate dependencies before tests |

## Cost Management

Configure cost control and optimization settings.

```toml
[cost_management]
# File size limits for cost control
max_file_size_kb = 50              # Skip files larger than this (KB)
max_context_size_chars = 100000    # Limit total context size
max_files_per_request = 15         # Override batch size for large files
use_cheaper_model_threshold_kb = 10 # Use cheaper model for files < this size
enable_content_compression = true   # Remove comments/whitespace in prompts

# Additional optimizations
skip_trivial_files = true          # Skip files with < 5 functions/classes
token_usage_logging = true         # Log token usage for cost tracking

# Cost thresholds and limits
[cost_management.cost_thresholds]
daily_limit = 50.0                 # Maximum daily cost in USD
per_request_limit = 2.0            # Maximum cost per request in USD
warning_threshold = 1.0            # Warn when request exceeds this cost
```

### Cost Management Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_file_size_kb` | `int` | `50` | Maximum file size to process |
| `max_context_size_chars` | `int` | `100000` | Maximum context size |
| `daily_limit` | `float` | `50.0` | Daily spending limit (USD) |
| `per_request_limit` | `float` | `2.0` | Per-request spending limit (USD) |

## Quality Analysis

Configure mutation testing and quality analysis.

```toml
[quality]
# Quality analysis settings
enable_quality_analysis = true     # Enable quality analysis by default
enable_mutation_testing = true     # Enable mutation testing by default
minimum_quality_score = 75.0       # Minimum acceptable quality score (%)
minimum_mutation_score = 80.0      # Minimum acceptable mutation score (%)
max_mutants_per_file = 50          # Maximum mutants per file for performance
mutation_timeout = 30              # Timeout in seconds for mutation testing
display_detailed_results = true    # Show detailed quality analysis results
enable_pattern_analysis = true     # Enable failure pattern analysis for smart refinement

# Modern Python Mutators
[quality.modern_mutators]
enable_type_hints = true           # Enable type hint mutations
enable_async_await = true          # Enable async/await mutations
enable_dataclass = true            # Enable dataclass mutations
type_hints_severity = "medium"     # Severity: "low", "medium", "high"
async_severity = "high"            # Async mutations often critical
dataclass_severity = "medium"      # Dataclass mutations typically medium severity
```

### Quality Analysis Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_quality_analysis` | `bool` | `true` | Enable quality analysis |
| `enable_mutation_testing` | `bool` | `true` | Enable mutation testing |
| `minimum_quality_score` | `float` | `75.0` | Minimum quality threshold |
| `minimum_mutation_score` | `float` | `80.0` | Minimum mutation score |

## Telemetry & Observability

Configure telemetry collection and observability.

```toml
[telemetry]
enabled = false                    # Enable telemetry collection
backend = "opentelemetry"          # Options: "opentelemetry", "datadog", "jaeger", "noop"
service_name = "testcraft"         # Service name for telemetry
service_version = ""               # Service version (empty = auto-detected)
environment = "development"        # Environment name

# Tracing configuration
trace_sampling_rate = 1.0          # Trace sampling rate (0.0 to 1.0)
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
endpoint = ""                      # Auto-detect or use OTEL_EXPORTER_OTLP_ENDPOINT
headers = {}                       # Additional headers for OTLP exporter
insecure = false                   # Use insecure gRPC connection
timeout = 10                       # Timeout for exports in seconds
```

### Telemetry Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable telemetry collection |
| `backend` | `str` | `"opentelemetry"` | Telemetry backend to use |
| `trace_sampling_rate` | `float` | `1.0` | Trace sampling rate |
| `anonymize_file_paths` | `bool` | `true` | Anonymize file paths |

## LLM Providers

Configure LLM providers and model settings.

```toml
[llm]
# General LLM Settings
default_provider = "openai"       # Options: "openai", "anthropic", "azure-openai", "bedrock"
max_retries = 3                   # Maximum number of retries for LLM requests
enable_streaming = false          # Enable streaming responses where supported
temperature = 0.1                 # Temperature for LLM responses (lower = more deterministic)

# OpenAI Configuration
openai_api_key = ""               # OpenAI API key (or set OPENAI_API_KEY environment variable)
openai_model = "gpt-4.1"          # OpenAI model to use for test generation
openai_base_url = ""              # Custom OpenAI API base URL (optional)
openai_max_tokens = 12000         # Maximum tokens for OpenAI requests
openai_timeout = 60.0             # Timeout for OpenAI requests (seconds)

# Anthropic Claude Configuration
anthropic_api_key = ""            # Anthropic API key (or set ANTHROPIC_API_KEY environment variable)
anthropic_model = "claude-sonnet-4" # Anthropic model to use for test generation
anthropic_max_tokens = 100000     # Maximum tokens for Anthropic requests
anthropic_timeout = 60.0          # Timeout for Anthropic requests (seconds)

# Azure OpenAI Configuration
azure_openai_api_key = ""         # Azure OpenAI API key (or set AZURE_OPENAI_API_KEY environment variable)
azure_openai_endpoint = ""        # Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT environment variable)
azure_openai_deployment = "gpt-4.1" # Azure OpenAI deployment name
azure_openai_api_version = "2024-02-15-preview" # Azure OpenAI API version
azure_openai_timeout = 60.0       # Timeout for Azure OpenAI requests (seconds)

# AWS Bedrock Configuration
aws_region = ""                   # AWS region for Bedrock (or set AWS_REGION environment variable)
aws_access_key_id = ""            # AWS access key ID (or set AWS_ACCESS_KEY_ID environment variable)
aws_secret_access_key = ""        # AWS secret access key (or set AWS_SECRET_ACCESS_KEY environment variable)
bedrock_model_id = "anthropic.claude-3-7-sonnet-v1:0" # AWS Bedrock model ID
bedrock_timeout = 60.0            # Timeout for Bedrock requests (seconds)
```

### LLM Provider Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_provider` | `str` | `"openai"` | Default LLM provider |
| `temperature` | `float` | `0.1` | Response randomness (0.0-1.0) |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `enable_streaming` | `bool` | `false` | Enable streaming responses |

### Supported Models

#### OpenAI Models
- `gpt-4.1` (default, recommended)
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`
- `o4-mini`

#### Anthropic Models
- `claude-sonnet-4` (default, recommended)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

#### Azure OpenAI Models
- Use Azure deployment names configured in your Azure OpenAI service
- Default deployment: `gpt-4.1`

#### AWS Bedrock Models
- `anthropic.claude-3-7-sonnet-v1:0` (default)
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`

### Beta/Extended Feature Gates

Add feature gates under the LLM section to control beta features:

```toml
[llm.beta]
anthropic_enable_extended_thinking = false
anthropic_enable_extended_output = false
openai_enable_extended_output = false
```

These flags gate extended provider capabilities (e.g., configurable thinking budgets) even when a model supports them. Defaults are disabled.

## Environment Variable Overrides

Override any configuration value using environment variables with the prefix `TESTCRAFT_`. Use double underscores (`__`) to separate nested keys.

### Examples

```bash
# Coverage settings
export TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE=85
export TESTCRAFT_COVERAGE__JUNIT_XML=true

# Generation settings
export TESTCRAFT_GENERATION__INCLUDE_DOCSTRINGS=false
export TESTCRAFT_GENERATION__TEST_RUNNER__ENABLE=true

# Cost management
export TESTCRAFT_COST_MANAGEMENT__DAILY_LIMIT=25.0
export TESTCRAFT_COST_MANAGEMENT__MAX_FILE_SIZE_KB=75

# Evaluation settings
export TESTCRAFT_EVALUATION__ENABLED=true
export TESTCRAFT_EVALUATION__LLM_JUDGE_ENABLED=true
export TESTCRAFT_EVALUATION__CONFIDENCE_LEVEL=0.99

# LLM settings
export TESTCRAFT_LLM__DEFAULT_PROVIDER=anthropic
export TESTCRAFT_LLM__TEMPERATURE=0.2
```

### Environment Variable Precedence

Configuration sources are evaluated in this order (later sources override earlier ones):

1. Default values
2. Global config file (`~/.config/testcraft/config.toml`)
3. Project config file (`.testcraft.toml` or `pyproject.toml`)
4. Environment variables (`TESTCRAFT_*`)
5. Command-line arguments

### API Keys

API keys should always be set via environment variables for security:

```bash
# Required based on your chosen provider
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION="us-east-1"
```

## Configuration Validation

TestCraft validates configuration files on startup. Common validation errors:

### Invalid Values
```toml
[coverage]
minimum_line_coverage = 150.0  # ❌ Error: Must be between 0.0 and 100.0
```

### Missing Dependencies
```toml
[evaluation]
enabled = true
llm_judge_enabled = true  # ❌ Error: Requires LLM provider configuration
```

### Invalid Types
```toml
[generation]
include_docstrings = "yes"  # ❌ Error: Must be boolean (true/false)
```

## Best Practices

### 1. Environment-Specific Configs

Use different configurations for different environments:

```bash
# Development
cp .testcraft.dev.toml .testcraft.toml

# Production
cp .testcraft.prod.toml .testcraft.toml
```

### 2. Security

- Never commit API keys to version control
- Use environment variables for sensitive data
- Enable security patterns blocking for generated code

### 3. Performance

- Set appropriate file size limits for cost control
- Use cheaper models for simple files
- Enable content compression for large contexts

### 4. Quality

- Set realistic coverage thresholds based on your codebase
- Enable mutation testing for critical code
- Use evaluation harness for continuous improvement

### 5. Monitoring

- Enable telemetry in production environments
- Track token usage and costs
- Monitor test generation quality trends

---

For practical usage examples, see the [Advanced Usage Guide](advanced-usage.md) and [Architecture Guide](architecture.md).

## Enhanced Import System

TestCraft includes an enhanced import system that provides recursive directory trees and authoritative module path derivation to ensure generated tests have correct import statements.

### Features

#### 1. Recursive Directory Tree
- Provides comprehensive project structure context to the LLM
- Configurable depth and entry limits for performance
- Optional filtering to Python files only

#### 2. Authoritative Module Path Derivation
- Automatically derives correct dotted import paths (e.g., `src.mypackage.module`)
- Handles `src/` layouts, namespace packages, and complex project structures
- Validates import paths by attempting actual imports
- Provides fallback candidates when primary path fails

#### 3. Enhanced Usage Examples
- Prioritizes module-qualified import patterns in context examples
- Searches for usage examples using derived module paths
- Provides better context for LLM test generation

#### 4. Reliable Test Execution Environment
- Automatically configures `PYTHONPATH` to include project root and `src/`
- Ensures generated tests can import modules correctly during pytest execution
- Applies to both coverage measurement and test refinement

### Configuration

Directory tree and usage example features are configured via `context_enrichment`:

```toml
[context_enrichment]
enable_usage_examples = true  # Use enhanced module-qualified usage examples
enable_env_detection = true
enable_db_boundary_detection = true
enable_http_boundary_detection = true
enable_comprehensive_fixtures = true
enable_side_effect_detection = true
```

**Note**: Directory tree depth and entry limits are internal defaults and not user-configurable.

### How It Works

1. **During Context Assembly**:
   - Builds recursive directory tree with safety limits
   - Provides comprehensive project structure to LLM

2. **During Test Generation**:
   - Derives authoritative module path for the target file
   - Injects module path and import suggestions into LLM context
   - Enhances usage examples with module-qualified imports

3. **During Test Execution**:
   - Configures `PYTHONPATH` to include project root and `src/`
   - Ensures imports work correctly in both coverage and refinement

### Benefits

- **Eliminates Import Guessing**: LLM gets exact import paths instead of guessing
- **Higher Success Rate**: More generated tests work correctly on first attempt
- **Better Project Understanding**: LLM sees complete project structure
- **Consistent Import Patterns**: All tests follow the same import conventions
- **Reduced Manual Fixes**: Less time spent fixing import statements

### Telemetry

The system tracks module path derivation success rates. Telemetry backend is selected by `telemetry.backend` when `telemetry.enabled=true`. If disabled or an unknown backend is set, TestCraft falls back to a no-op telemetry adapter gracefully.

- `module_path_derived_total`: Total attempts
- `module_path_derived_success`: Successful validations
- `module_path_status_*`: Breakdown by validation status
- `module_path_fallback_used`: When fallback paths are used
- `module_path_has_src`: Whether `src/` is in the path
- `module_path_depth`: Depth of derived module paths

### Troubleshooting

#### Common Issues

1. **Module not found errors**: Check that `PYTHONPATH` includes project root
2. **Wrong import paths**: Verify project structure follows Python conventions
3. **Performance issues**: Reduce `max_depth` or `max_entries_per_dir`
4. **Validation failures**: Check logs for specific import validation errors

#### Debug Mode

Enable debug logging to see module path derivation details:

```bash
export TESTCRAFT_LOG_LEVEL=DEBUG
testcraft generate --verbose
```
