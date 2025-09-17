# TestCraft Configuration Reference

This document provides a comprehensive reference for all TestCraft configuration options. TestCraft uses TOML format for configuration files (`.testcraft.toml`).

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Schema Overview](#schema-overview)
3. [Test Patterns](#test-patterns)
4. [LLM Configuration](#llm-configuration)
5. [Generation Configuration](#generation-configuration)
6. [Cost Management](#cost-management)
7. [Telemetry & Observability](#telemetry--observability)
8. [Environment Management](#environment-management)
9. [Evaluation Harness](#evaluation-harness)
10. [Migration Guide](#migration-guide)
11. [Environment Variables](#environment-variables)

## Configuration File Structure

TestCraft looks for configuration files in the following order:

1. `.testcraft.toml` (project-specific)
2. `pyproject.toml` (in `[tool.testcraft]` section)
3. Global configuration in `~/.config/testcraft/config.toml`

## Schema Overview

TestCraft's configuration is organized into the following main sections:

- `[test_patterns]` - File discovery patterns and test discovery settings
- `[llm]` - Large Language Model configuration
- `[generation]` - Test generation behavior (includes budgets, refinement)
- `[cost_management]` - Cost thresholds and optimization
- `[telemetry]` - Observability and telemetry configuration
- `[evaluation]` - Test evaluation harness (optional)
- `[environment]` - Environment detection and management
- `[logging]` - Logging behavior configuration
- `[ui]` - User interface behavior

### What Drives Runtime Logic

| TOML Configuration | Runtime Impact |
|-------------------|----------------|
| `generation.batch_size` | Number of files processed concurrently |
| `generation.immediate_refinement` | Enable immediate write-and-refine per file |
| `generation.enable_refinement` | Toggle AI-powered test refinement |
| `generation.test_framework` | Framework used in generated tests |
| `generation.prompt_budgets.*` | Context size limits for LLM prompts |
| `generation.context_categories.*` | Which context types to include |
| `generation.refine.*` | Refinement behavior and guardrails |
| `llm.enable_streaming` | Use streaming LLM responses |
| `cost_management.cost_thresholds.*` | Cost limits and warnings |
| `telemetry.enabled` | Enable observability features |

## Test Patterns

Configure file discovery and test detection behavior:

```toml
[test_patterns]
# Patterns for finding test files (supports glob patterns)
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

# Directories to exclude from scanning (extensive default list provided)
exclude_dirs = [
    "venv", "env", ".env", ".venv",
    "node_modules", ".git", "__pycache__"
]

[test_patterns.test_discovery]
# Test discovery mode: 'auto', 'pytest-collector', 'ast', 'globs'
mode = "auto"
# Enable classification of support files (conftest.py, fixture-only files)
classify_support_files = true
# Timeout in seconds for pytest collection
collector_timeout_sec = 15
# Cache TTL in seconds for discovery results
cache_ttl_sec = 600
```

## LLM Configuration

TestCraft uses a **unified LLM adapter system** that allows you to seamlessly switch between providers without changing your code. All supported providers use the same interface and return consistent response schemas.

### Model Catalog System

TestCraft maintains a centralized **model catalog** (`testcraft/config/model_catalog.toml`) that serves as the single source of truth for:

- **Model limits**: Context windows, output limits, thinking tokens
- **Pricing information**: Per-million token costs for accurate cost tracking
- **Feature flags**: Vision, tool use, structured outputs, reasoning capabilities
- **Beta features**: Optional extended context/output limits with explicit opt-in
- **Provenance**: Links to official vendor documentation and verification dates

This catalog-driven approach ensures:
- ✅ **Accurate limits**: Token budgets never exceed documented vendor defaults
- ✅ **Consistent pricing**: Cost calculations sourced from official documentation
- ✅ **Feature gating**: Beta features require explicit configuration
- ✅ **Up-to-date metadata**: Regular verification against vendor documentation

### Supported Providers

TestCraft supports four major LLM providers with full interchangeability:

- **OpenAI** (`openai`) - GPT-4 and o-series models with reasoning capabilities
- **Anthropic Claude** (`anthropic`) - Claude 3.5 Sonnet with extended thinking support
- **Azure OpenAI** (`azure-openai`) - Enterprise OpenAI with custom deployments
- **AWS Bedrock** (`bedrock`) - Claude models via AWS infrastructure

All model metadata is automatically loaded from the model catalog, ensuring consistent behavior across providers.

### Basic Configuration

```toml
[llm]
# Default LLM provider to use (interchangeable)
default_provider = "openai"
# Enable streaming responses where supported
enable_streaming = false
# Temperature for LLM responses (lower = more deterministic)
temperature = 0.1
# Maximum number of retries for LLM requests
max_retries = 3

# OpenAI Configuration
openai_model = "gpt-4"
openai_max_tokens = 12000
openai_timeout = 60.0

# Anthropic Claude Configuration  
anthropic_model = "claude-3-5-sonnet-20241022"
anthropic_max_tokens = 100000
anthropic_timeout = 60.0

# Azure OpenAI Configuration
azure_openai_deployment = "gpt-4"
azure_openai_api_version = "2024-02-15-preview"
azure_openai_max_tokens = 12000
azure_openai_timeout = 60.0

# AWS Bedrock Configuration
bedrock_model_id = "anthropic.claude-3-5-sonnet-20241022-v1:0"
aws_region = "us-east-1"
bedrock_timeout = 60.0

# Note: API keys are set via environment variables for security
# OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_OPENAI_API_KEY, 
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
```

### Beta Features Configuration

TestCraft supports optional beta features that extend model capabilities beyond documented defaults. **These features are disabled by default** to ensure compliance with vendor specifications:

```toml
[llm.beta]
# Enable extended output limits beyond documented defaults
enable_extended_output = false
# Enable extended context windows for supported models  
enable_extended_context = false

# Provider-specific beta feature settings
[llm.beta.anthropic]
# Enable Anthropic beta headers for extended features
enable_extended_output = false
enable_extended_context = false

[llm.beta.openai] 
# Enable OpenAI beta features (when available)
enable_extended_output = false
enable_extended_context = false
```

**Important**: Beta features may incur additional costs and are subject to vendor rate limits. Only enable these features if you explicitly need them and have verified the associated costs.

### Model Catalog Management

You can view and verify your model catalog using the built-in CLI commands:

```bash
# View all available models with limits and pricing
testcraft models show

# Verify model catalog against current usage
testcraft models verify  

# View recent changes to the catalog
testcraft models diff --since=2025-01-01

# Filter by provider
testcraft models show --provider=openai
```

For more details, see the [Model Management Guide](models.md).

### Unified Router Behavior

The LLM router (`LLMRouter`) provides a consistent interface across all providers:

- **Interchangeable providers**: Change `default_provider` to switch providers instantly
- **Consistent schemas**: All operations return the same response structure
- **Unified token budgeting**: Per-request token limits calculated automatically
- **Preserved specialties**: Provider-specific features (thinking modes, reasoning) work seamlessly
- **Centralized prompts**: All providers use the same `PromptRegistry` for consistent behavior

### Provider Capabilities & Differences

While all providers are interchangeable, each has unique strengths and capabilities:

#### OpenAI Features
- **o-series reasoning models** (o1-preview, o1-mini): Advanced reasoning with internal monologue
- **Responses API**: Special API for o-series models with different parameter handling
- **Temperature handling**: o-series models ignore custom temperature settings
- **Token budgeting**: Uses `max_completion_tokens` vs `max_tokens` based on model type

#### Anthropic Claude Features
- **Extended thinking mode**: Supports explicit thinking tokens for reasoning
- **Large context windows**: Up to 200K tokens for comprehensive code analysis
- **Thinking budgets**: Automatically applies thinking token limits when supported
- **Advanced code reasoning**: Particularly strong at code analysis and test generation

#### Azure OpenAI Features
- **Enterprise deployment**: Custom model deployments with controlled access
- **Metadata mapping**: Deployment names mapped to unified model identifiers
- **Regional availability**: Deploy models in specific geographic regions
- **Enhanced security**: Enterprise-grade security and compliance features

#### AWS Bedrock Features
- **Multi-region support**: Deploy across AWS regions for optimal performance
- **Claude via AWS**: Access Anthropic models through AWS infrastructure
- **Thinking guidance**: Embeds thinking instructions in prompts when thinking tokens requested
- **Unified billing**: Consolidated costs through AWS billing

### Capability Detection

TestCraft automatically detects and exposes provider capabilities:

```python
from testcraft.adapters.llm import get_capabilities

# Get capabilities for current provider
capabilities = llm_router.get_capabilities()
print(f"Supports thinking: {capabilities['supports_thinking_mode']}")
print(f"Max context: {capabilities['max_context_tokens']}")
print(f"Reasoning model: {capabilities['is_reasoning_model']}")
```

**Available capability flags:**
- `supports_thinking_mode`: Provider supports explicit thinking tokens
- `is_reasoning_model`: Model has built-in reasoning capabilities (e.g., o1-preview)
- `has_reasoning_capabilities`: Model can perform complex reasoning tasks
- `max_context_tokens`: Maximum input context size
- `max_output_tokens`: Maximum output token limit
- `max_thinking_tokens`: Maximum thinking tokens (if supported)

## Generation Configuration

This is the core configuration section that controls test generation behavior:

### Basic Generation Settings

```toml
[generation]
# Number of files to process in parallel
batch_size = 5
# Coverage threshold for reporting (0.0 to 1.0)
coverage_threshold = 0.8
# Testing framework to use
test_framework = "pytest"
# Enable AI-powered test refinement
enable_refinement = true
# Maximum refinement attempts per test file
max_refinement_iterations = 3
# Enable immediate write-and-refine per file
immediate_refinement = true
# Limit concurrent pytest/refine workers
max_refine_workers = 2
# Keep test files that fail to write or have syntax errors
keep_failed_writes = false
# Backoff between refinement iterations (seconds)
refinement_backoff_sec = 0.2
# Disable Ruff formatting if it causes issues
disable_ruff_format = false
```

### Prompt Budgets

Control context size limits for LLM prompts:

```toml
[generation.prompt_budgets]
# Character limit per context item
per_item_chars = 1500
# Consequences: Higher allows richer items but may reduce the number of items that fit.
# Lower keeps items terse and increases diversity but may truncate useful detail.
# Total character limit for context
total_chars = 10000
# Consequences: Higher overall budget improves recall; increases cost and prompt length.
# Lower forces tighter selection; safer and cheaper but may omit helpful context.

[generation.prompt_budgets.section_caps]
# Maximum items per context section (per-category item caps). Increasing a cap shifts
# budget toward that section; total_chars still applies.
#
# Section meanings and consequences
# - snippets: Short, symbol-aware code/doc snippets. Higher = more variety; too high can dilute signal.
# - neighbors: Import-graph neighbor files. Higher adds periphery; increases prompt size.
# - test_exemplars: Summaries mined from existing tests. Higher = more examples; may crowd other context.
# - contracts: API contract summaries (signature/params/returns/raises/invariants). Higher clarifies spec; costs budget.
# - deps_config_fixtures: Env/config keys, DB/HTTP clients, pytest fixtures. Usually 1–2 is enough.
# - coverage_hints: Per-file coverage hints (if wired). Keep small; harmless when empty.
# - callgraph: Call-graph/import edges + nearby files. Higher adds topology; can add noise.
# - error_paths: Exception types from docs/AST. Higher emphasizes error handling; reduces room elsewhere.
# - usage_examples: High-quality usage snippets. Higher helps synthesis; too high may overfit to examples.
# - pytest_settings: Key pytest ini options. 1 is typically sufficient.
# - side_effects: Detected side-effect boundaries (fs/env/network). Higher reveals risks; may add noise.
# - path_constraints: Branch/condition summaries. Higher improves logic coverage; can crowd prompt.
snippets = 10
neighbors = 5
test_exemplars = 5
contracts = 8
deps_config_fixtures = 2
coverage_hints = 3
callgraph = 3
error_paths = 3
usage_examples = 5
pytest_settings = 1
side_effects = 1
path_constraints = 3
```

### Context Budgets

Configure resource-intensive operations:

```toml
[generation.context_budgets.directory_tree]
# Maximum directory depth to traverse
max_depth = 4
# Maximum files/directories per directory
max_entries_per_dir = 200
# Only include .py files and directories
include_py_only = true
```

### Context Categories

Control which context types to include:

```toml
[generation.context_categories]
# Include code snippets from source files
snippets = true
# Include neighboring/related files
neighbors = true
# Include test examples from existing tests
test_exemplars = true
# Include interface contracts and type annotations
contracts = true
# Include dependencies, config, and fixtures
deps_config_fixtures = true
# Include coverage analysis hints
coverage_hints = true
# Include call graph analysis
callgraph = true
# Include error paths and exception handling
error_paths = true
# Include usage examples
usage_examples = true
# Include pytest configuration and settings
pytest_settings = true
# Include side effects detection
side_effects = true
# Include path constraints and validation
path_constraints = true
```

### Context Enrichment

Configure context enrichment features:

```toml
[generation.context_enrichment]
# Enable environment variable usage detection
enable_env_detection = true
# Enable database client boundary detection
enable_db_boundary_detection = true
# Enable HTTP client boundary detection
enable_http_boundary_detection = true
# Enable comprehensive pytest fixture discovery
enable_comprehensive_fixtures = true
# Enable side-effect boundary detection
enable_side_effect_detection = true
# Maximum environment variables to include
max_env_vars = 20
# Maximum fixtures to include
max_fixtures = 15
```

### Refinement Configuration

Control AI-powered test refinement behavior:

```toml
[generation.refine]
# Enable AI-powered test refinement
enable = true
# Maximum refinement attempts
max_retries = 2
# Base delay between refinement attempts
backoff_base_sec = 1.0
# Maximum delay between attempts
backoff_max_sec = 8.0
# Stop if LLM returns no changes
stop_on_no_change = true
# Maximum total time for refinement (minutes)
max_total_minutes = 5.0

# Strict refinement policies
strict_assertion_preservation = true
fail_on_xfail_markers = true
allow_xfail_on_suspected_bugs = false
report_suspected_prod_bugs = true

# Failed test annotation configuration
annotate_failed_tests = true
annotation_placement = "top"
annotation_include_failure_excerpt = true
annotation_max_failure_chars = 600
annotation_style = "docstring"
include_llm_fix_instructions = true

# Timeout detection
enable_timeout_detection = true
timeout_threshold_seconds = 30.0
```

## Cost Management

Configure cost thresholds and optimization settings:

```toml
[cost_management]
# Skip files larger than this (KB)
max_file_size_kb = 50
# Limit total context size
max_context_size_chars = 100000
# Override batch size for large files
max_files_per_request = 15
# Use cheaper model for files < this size
use_cheaper_model_threshold_kb = 10
# Remove comments/whitespace in prompts
enable_content_compression = true
# Skip files with < 5 functions/classes
skip_trivial_files = true
# Log token usage for cost tracking
token_usage_logging = true

[cost_management.cost_thresholds]
# Maximum daily cost in USD
daily_limit = 50.0
# Maximum cost per request in USD
per_request_limit = 2.0
# Warn when request exceeds this cost
warning_threshold = 1.0
```

## Telemetry & Observability

Configure observability and telemetry collection:

```toml
[telemetry]
# Enable telemetry collection
enabled = false
# Telemetry backend to use: 'opentelemetry', 'datadog', 'jaeger', 'noop'
backend = "opentelemetry"
# Service name for telemetry
service_name = "testcraft"
# Environment name (development, staging, production)
environment = "development"

# Tracing configuration
trace_sampling_rate = 1.0
capture_llm_calls = true
capture_coverage_runs = true
capture_file_operations = true
capture_test_generation = true

# Metrics configuration
collect_metrics = true
metrics_interval_seconds = 30
track_token_usage = true
track_coverage_delta = true
track_test_pass_rate = true

# Privacy and anonymization
anonymize_file_paths = true
anonymize_code_content = true
opt_out_data_collection = false
```

## Environment Management

Configure environment detection and management:

```toml
[environment]
# Auto-detect current environment manager
auto_detect = true
# Preferred environment manager
preferred_manager = "auto"  # or "poetry", "pipenv", "conda", "uv", "venv"
# Always use current virtual env
respect_virtual_env = true
# Validate deps before running tests
dependency_validation = true
```

## Evaluation Harness

Configure test evaluation features (optional):

```toml
[evaluation]
# Enable evaluation harness functionality
enabled = false
# Path to golden repositories for regression testing
golden_repos_path = "/path/to/golden/repos"
# Enable automated acceptance checks
acceptance_checks = true
# Enable LLM-as-judge evaluation
llm_judge_enabled = true
# Evaluation dimensions for LLM-as-judge
rubric_dimensions = ["correctness", "coverage", "clarity", "safety"]
```

## Migration Guide

### From Previous Versions

**Deprecated Sections Removed:**
- `[style]` → Use `generation.test_framework`
- `[coverage]` → No longer used (coverage logic built-in)
- `[quality]` → No runtime implementation
- `[context_enrichment]` at root → Moved to `generation.context_enrichment`

**Key Changes:**
- All generation settings now under `[generation]`
- Prompt budgets and context categories explicitly configurable
- Refinement behavior fully configurable under `generation.refine`
- Cost management moved to `[cost_management]` (was `[cost]`)

**Migration Example:**
```toml
# OLD (deprecated)
[style]
framework = "pytest"

[coverage]
minimum_line_coverage = 80.0

# NEW (current)
[generation]
test_framework = "pytest"
coverage_threshold = 0.8
```

## Environment Variables

API credentials and sensitive settings are configured via environment variables:

```bash
# LLM Provider API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"

# Custom Endpoints (optional)
export AZURE_OPENAI_ENDPOINT="https://your-azure-endpoint.openai.azure.com"
export OLLAMA_BASE_URL="http://localhost:11434/api"

# Telemetry (if using OpenTelemetry)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# TestCraft Behavior (optional)
export TESTCRAFT_UI="minimal"  # or "classic"
```

## Example Complete Configuration

```toml
# TestCraft Configuration
# This example shows commonly used settings

[test_patterns]
test_patterns = ["test_*.py", "*_test.py", "tests/**/test_*.py"]
exclude = ["migrations/*", "__pycache__/*"]

[llm]
# Use Claude for advanced reasoning and large context
default_provider = "anthropic"
anthropic_model = "claude-3-5-sonnet-20241022"
enable_streaming = false
temperature = 0.1

# Alternative: Use OpenAI for o-series reasoning models
# default_provider = "openai"
# openai_model = "o1-preview"

# Alternative: Use Azure OpenAI for enterprise
# default_provider = "azure-openai"
# azure_openai_deployment = "gpt-4-deployment"

[generation]
batch_size = 3
test_framework = "pytest"
enable_refinement = true
immediate_refinement = true
max_refinement_iterations = 2
coverage_threshold = 0.85

[generation.prompt_budgets]
total_chars = 8000
per_item_chars = 1200

[generation.context_categories]
snippets = true
test_exemplars = true
contracts = true
deps_config_fixtures = true

[generation.refine]
enable = true
max_retries = 2
strict_assertion_preservation = true
annotate_failed_tests = true

[cost_management]
max_file_size_kb = 40
daily_limit = 25.0
per_request_limit = 1.5

[telemetry]
enabled = false

[logging]
max_debug_chars = 1500

[ui]
default_style = "classic"
```

---

For more information, see the [TestCraft documentation](https://github.com/your-repo/testcraft/docs) or run `testcraft config --help`.