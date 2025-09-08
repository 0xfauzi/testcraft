# TestCraft

AI-powered test generation and evaluation platform for Python projects.

## Overview

TestCraft is an intelligent test generation system that analyzes Python codebases and automatically generates comprehensive test suites using Large Language Models (LLMs). Built with Clean Architecture principles, it provides a robust, extensible platform for automated test creation, evaluation, and continuous improvement.

## 🚀 Key Features

### Test Generation
- **Intelligent Test Generation**: Uses advanced LLMs to generate contextually appropriate tests
- **Multiple LLM Support**: Compatible with OpenAI GPT, Anthropic Claude, Azure OpenAI, and AWS Bedrock
- **Coverage Analysis**: Integrates with pytest and coverage tools for comprehensive analysis
- **Smart Refinement**: AI-powered test fixing with iterative improvement

### Evaluation Harness (New!)
- **Automated Acceptance Checks**: Syntax validation, import checking, pytest execution, coverage measurement
- **LLM-as-Judge Evaluation**: Rubric-driven test quality assessment with rationale generation
- **A/B Testing Pipeline**: Statistical significance testing for prompt optimization
- **Bias Detection**: Comprehensive bias analysis and mitigation recommendations
- **Golden Repository Testing**: Regression detection against known-good repositories

### Architecture & Extensibility
- **Clean Architecture**: Modular design with clear separation of concerns
- **Extensible Design**: Plugin-based architecture for custom adapters
- **Rich CLI Interface**: Beautiful command-line interface with progress tracking
- **TOML Configuration**: Comprehensive configuration system with environment detection

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- uv (recommended) or pip

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd testcraft
```

2. Install dependencies:
```bash
uv sync
```

3. Install in development mode:
```bash
uv pip install -e .
```

4. Configure your LLM provider (choose one):
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"  
export AZURE_OPENAI_API_KEY="your-api-key"
export AWS_ACCESS_KEY_ID="your-access-key"
```

## 🚀 Quick Start

### Basic Test Generation

Generate comprehensive tests for your Python modules:

```bash
# Generate tests for a specific module
testcraft generate src/mymodule.py

# Generate tests with coverage analysis
testcraft coverage src/mymodule.py

# Generate tests with custom configuration
testcraft generate src/mymodule.py --config .testcraft.toml
```

### Configuration

Create a `.testcraft.toml` file in your project root:

```toml
[style]
framework = "pytest"
assertion_style = "pytest"

[coverage]
minimum_line_coverage = 80.0
junit_xml = true

[generation]
include_docstrings = true
generate_fixtures = true

[evaluation]
enabled = true
acceptance_checks = true
llm_judge_enabled = true

[llm]
default_provider = "openai"
openai_model = "o4-mini"
temperature = 0.1
```

### Advanced: Evaluation and A/B Testing

TestCraft includes a powerful evaluation harness for testing and improving your test generation:

```bash
# Run A/B testing on different prompt variants
testcraft evaluation ab-test ab_config.json --statistical-testing

# Analyze statistical significance of evaluation results
testcraft evaluation statistical-analysis results.json

# Detect evaluation bias patterns
testcraft evaluation bias-detection --time-window 30

# Run comprehensive evaluation campaigns
testcraft evaluation campaign campaign_config.json --verbose
```

### Common Commands

```bash
# Show version and status
testcraft version

# Initialize configuration interactively  
testcraft config init

# Get help for any command
testcraft --help
testcraft evaluation --help
```

## 🏗️ Architecture

TestCraft follows **Clean Architecture** principles with clear separation of concerns:

### Project Structure

```
testcraft/
├── domain/          # Core business logic and entities
├── application/     # Use cases and application services  
├── ports/           # Interface definitions (contracts)
├── adapters/        # External integrations
│   ├── llm/         # LLM provider integrations
│   ├── io/          # File system, UI, and artifact storage
│   ├── coverage/    # Test coverage analysis
│   ├── evaluation/  # Evaluation harness and A/B testing
│   ├── context/     # Code context and retrieval
│   ├── parsing/     # Code parsing and analysis  
│   ├── refine/      # Test refinement and improvement
│   └── telemetry/   # Observability and metrics
├── config/          # Configuration management
├── prompts/         # Prompt registry and templates
└── cli/             # Command-line interface
```

### Key Components

- **Evaluation Harness** (`evaluation/harness.py`): Comprehensive test evaluation system
- **LLM Router** (`adapters/llm/router.py`): Multi-provider LLM integration  
- **Configuration System** (`config/`): TOML-based configuration with validation
- **Artifact Store** (`adapters/io/artifact_store.py`): Evaluation result storage
- **Rich UI** (`adapters/io/ui_rich.py`): Beautiful terminal interfaces

## 🛠️ Development

### Development Tools

- **ruff**: Linting and code formatting
- **mypy**: Type checking
- **pytest**: Testing framework with coverage
- **rich**: Beautiful terminal UI components

### Setting Up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd testcraft
uv sync

# Install pre-commit hooks (optional)
pre-commit install

# Run all development tools
uv run pytest --cov=testcraft    # Tests with coverage
uv run ruff check .             # Linting
uv run ruff format .            # Code formatting  
uv run mypy testcraft/          # Type checking
```

### Running the Evaluation Harness

```bash
# Test evaluation harness directly
python evaluation/harness.py

# Run evaluation tests
uv run pytest tests/test_evaluation_integration.py -v

# Test CLI evaluation commands
testcraft evaluation --help
```

### Testing Guidelines

1. Write tests for new features in `tests/`
2. Include integration tests for evaluation components
3. Use pytest fixtures for common test setup
4. Maintain test coverage above 80%
5. Test CLI commands with various configurations

## 📚 Documentation

- **README.md**: This file - project overview and quick start
- **Advanced Usage**: See `docs/advanced-usage.md` (coming soon)
- **Configuration Reference**: See `docs/configuration.md` (coming soon) 
- **Architecture Guide**: See `docs/architecture.md` (coming soon)
- **Contributing**: See `CONTRIBUTING.md` (coming soon)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** and create a feature branch
2. **Follow the architecture**: Use clean architecture patterns
3. **Add tests**: Include unit and integration tests
4. **Update documentation**: Keep docs in sync with changes
5. **Run quality checks**: Ensure linting, formatting, and type checking pass
6. **Test evaluation features**: Verify evaluation harness works correctly
7. **Submit a pull request** with a clear description of changes

### Contribution Guidelines

- Follow existing code style and architecture patterns
- Add type hints to all public functions
- Include docstrings for new modules and classes
- Update configuration examples if adding new config options
- Test CLI commands and provide usage examples
- Consider evaluation impact for test generation changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔗 Additional Resources

- [Example Configurations](.testcraft-comprehensive.toml)
- [Evaluation Harness Documentation](evaluation/harness.py)  
- [CLI Command Reference](testcraft/cli/main.py)
- [Architecture Patterns](testcraft/ports/)

For detailed usage examples, advanced configuration, and best practices, see the upcoming comprehensive documentation.
