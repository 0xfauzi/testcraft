# Contributing to TestCraft

We welcome contributions to TestCraft! This document provides guidelines for contributing to the project, from reporting issues to submitting pull requests.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Style and Standards](#code-style-and-standards)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Issue Guidelines](#issue-guidelines)
8. [Documentation Standards](#documentation-standards)
9. [Evaluation System Contributions](#evaluation-system-contributions)
10. [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- uv (recommended) or pip
- Git
- Basic understanding of Clean Architecture principles

### Fork and Clone

1. Fork the TestCraft repository
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/testcraft.git
cd testcraft
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/original/testcraft.git
```

## Development Environment

### Initial Setup

```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Environment Configuration

Create a `.env` file for local development:

```bash
# API keys for testing (optional)
OPENAI_API_KEY=your-test-key
ANTHROPIC_API_KEY=your-test-key

# Development settings
TESTCRAFT_LOG_LEVEL=DEBUG
TESTCRAFT_EVALUATION__ENABLED=true
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=testcraft --cov-report=html

# Run specific test categories
uv run pytest tests/test_domain/          # Unit tests
uv run pytest tests/test_integration/    # Integration tests
uv run pytest tests/test_evaluation/     # Evaluation tests

# Run CLI tests
uv run pytest tests/test_cli/ -v
```

### Development Tools

```bash
# Linting and formatting
uv run ruff check .                   # Check linting
uv run ruff format .                  # Format code

# Type checking
uv run mypy testcraft/

# Run all quality checks
make check                            # If Makefile exists
```

## Code Style and Standards

### Python Style Guide

- Follow PEP 8 with these modifications:
  - Line length: 88 characters (Black default)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures

### Code Formatting

We use **ruff** for linting and formatting:

```bash
# Format code before committing
uv run ruff format .

# Check and fix linting issues
uv run ruff check . --fix
```

### Type Hints

All public functions and methods must include type hints:

```python
# ✅ Good
def generate_test(source_file: Path, config: GenerationConfig) -> GeneratedTest:
    """Generate test for source file."""
    pass

# ❌ Bad
def generate_test(source_file, config):
    """Generate test for source file."""
    pass
```

### Documentation

All public classes and functions must have docstrings:

```python
class TestGenerator:
    """Generates comprehensive test suites for Python code.

    This class uses Large Language Models to analyze Python source code
    and generate appropriate test cases following best practices.

    Attributes:
        llm_adapter: Adapter for LLM communication
        config: Generation configuration
    """

    def generate_tests(
        self,
        source_files: List[Path],
        config: GenerationConfig
    ) -> List[GeneratedTest]:
        """Generate tests for the provided source files.

        Args:
            source_files: List of Python files to generate tests for
            config: Configuration options for test generation

        Returns:
            List of generated test objects with metadata

        Raises:
            GenerationError: If test generation fails
            ConfigurationError: If configuration is invalid
        """
        pass
```

## Architecture Guidelines

TestCraft follows Clean Architecture principles. When contributing:

### Layer Responsibilities

1. **Domain Layer** (`testcraft/domain/`):
   - Contains core business entities and rules
   - No dependencies on external frameworks
   - Pure Python, highly testable

2. **Application Layer** (`testcraft/application/`):
   - Contains use cases and business workflows
   - Orchestrates domain objects and port interactions
   - Depends only on domain and ports

3. **Ports** (`testcraft/ports/`):
   - Define interface contracts
   - Use Python Protocol for type checking
   - Keep interfaces focused and cohesive

4. **Adapters** (`testcraft/adapters/`):
   - Implement ports for external systems
   - Handle framework-specific details
   - Can depend on external libraries

### Dependency Rules

- **Inner layers never depend on outer layers**
- Use dependency injection through constructors
- Depend on interfaces (ports), not implementations

### Example: Adding a New Adapter

1. **Define the Port** (if needed):
```python
# testcraft/ports/new_port.py
class NewServicePort(Protocol):
    """Port for new service integration."""

    async def perform_operation(self, data: Any) -> Result:
        """Perform the required operation."""
        ...
```

2. **Implement the Adapter**:
```python
# testcraft/adapters/new_service/main_adapter.py
class NewServiceAdapter(NewServicePort):
    """Implementation of new service integration."""

    def __init__(self, config: NewServiceConfig):
        self.config = config
        self.client = NewServiceClient(config)

    async def perform_operation(self, data: Any) -> Result:
        """Perform operation using external service."""
        try:
            response = await self.client.call_api(data)
            return Result.success(response)
        except ExternalServiceError as e:
            return Result.failure(str(e))
```

3. **Register in Dependency Container**:
```python
# testcraft/cli/dependency_injection.py
def get_new_service_adapter(self) -> NewServicePort:
    """Get new service adapter instance."""
    if 'new_service_adapter' not in self._instances:
        self._instances['new_service_adapter'] = NewServiceAdapter(
            config=self.config.new_service
        )
    return self._instances['new_service_adapter']
```

## Testing Requirements

### Test Coverage

- Maintain minimum 80% test coverage
- Domain logic must have 95%+ coverage
- All public APIs must be tested

### Test Types

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test adapter integrations and use cases
3. **CLI Tests**: Test command-line interface functionality
4. **Evaluation Tests**: Test evaluation harness functionality

### Test Organization

```
tests/
├── unit/                    # Unit tests
│   ├── test_domain/        # Domain model tests
│   ├── test_application/   # Use case tests
│   └── test_adapters/      # Adapter unit tests
├── integration/            # Integration tests
│   ├── test_llm_integration/
│   ├── test_evaluation_integration/
│   └── test_coverage_integration/
├── cli/                    # CLI tests
│   └── test_commands/
└── fixtures/               # Test fixtures and data
    ├── sample_code/
    ├── test_configs/
    └── evaluation_data/
```

### Test Patterns

```python
# Unit test example
class TestGenerationRules:
    """Test domain business rules."""

    def test_confidence_calculation_high_quality(self):
        """Test confidence calculation for high-quality metrics."""
        metrics = {
            'syntax_valid': True,
            'imports_work': True,
            'tests_pass': True,
            'coverage_improvement': 90.0
        }

        confidence = TestGenerationRules.calculate_confidence(metrics)

        assert confidence >= 0.9

# Integration test example
class TestLLMIntegration:
    """Integration tests for LLM adapters."""

    @pytest.mark.integration
    async def test_openai_test_generation(self, openai_adapter, sample_source):
        """Test OpenAI adapter generates valid tests."""
        result = await openai_adapter.generate_tests(
            source_files=[sample_source],
            analysis=CodeAnalysis(),
            config=LLMConfig()
        )

        assert len(result) > 0
        assert result[0].content.startswith("def test_")
```

### Mock Guidelines

- Use `unittest.mock` for mocking external dependencies
- Mock at the adapter boundary, not within domain logic
- Provide realistic mock responses

```python
# Good mocking practice
@pytest.fixture
def mock_llm_adapter():
    """Mock LLM adapter for testing."""
    adapter = Mock(spec=LLMPort)
    adapter.generate_tests.return_value = [
        GeneratedTest(
            content="def test_example(): assert True",
            file_path=Path("test_example.py"),
            source_file=Path("example.py"),
            confidence_score=0.9
        )
    ]
    return adapter
```

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

2. **Create feature branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** following the guidelines above

4. **Run all quality checks**:
```bash
# Run tests
uv run pytest

# Check formatting and linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy testcraft/

# Test CLI functionality
testcraft --help
testcraft evaluation --help
```

5. **Update documentation** if needed

6. **Commit with clear messages**:
```bash
git commit -m "feat: add statistical analysis to evaluation harness

- Implement statistical significance testing
- Add confidence interval calculations
- Update evaluation port interface
- Add comprehensive tests and documentation"
```

### Pull Request Template

Use this template for pull requests:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing functionality)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Evaluation system enhancement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] CLI tests added/updated
- [ ] All tests pass locally

## Documentation
- [ ] Code documentation updated
- [ ] README updated (if needed)
- [ ] Configuration docs updated (if needed)
- [ ] Architecture docs updated (if needed)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Changes are backwards compatible (or breaking changes documented)
- [ ] Evaluation harness impact considered
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Evaluation Impact**: Consider impact on evaluation system
4. **Documentation**: Ensure documentation is complete
5. **Testing**: Verify adequate test coverage

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Run command...
2. With configuration...
3. Expected vs actual behavior...

## Environment
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.11.5]
- TestCraft version: [e.g., 1.2.3]
- LLM provider: [e.g., OpenAI GPT-4]

## Configuration
```toml
# Relevant configuration sections
```

## Additional Context
Screenshots, logs, or other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the requested feature.

## Use Case
Describe the problem this feature would solve.

## Proposed Solution
Your proposed approach to implementing this feature.

## Alternative Solutions
Other approaches considered.

## Impact on Evaluation System
How this feature might affect the evaluation harness.
```

## Documentation Standards

### Documentation Types

1. **Code Documentation**: Inline docstrings and comments
2. **User Documentation**: README, usage guides, tutorials
3. **Developer Documentation**: Architecture, contributing, API reference
4. **Configuration Documentation**: Complete configuration reference

### Documentation Updates

When making changes, update relevant documentation:

- **README.md**: For user-facing features
- **docs/advanced-usage.md**: For advanced features and evaluation
- **docs/configuration.md**: For new configuration options
- **docs/architecture.md**: For architectural changes
- **Docstrings**: For code changes

### Writing Style

- Use clear, concise language
- Include practical examples
- Provide both basic and advanced usage patterns
- Link to related documentation sections

## Evaluation System Contributions

The evaluation harness is a critical component. Special considerations:

### Adding Evaluation Metrics

1. **Define the metric** in domain models
2. **Implement in evaluation adapter**
3. **Add configuration options**
4. **Include comprehensive tests**
5. **Document the metric thoroughly**

### Evaluation Performance

- Consider computational cost of new evaluation methods
- Implement batch processing where appropriate
- Add timeout and resource limits
- Test with large datasets

### Statistical Methods

- Ensure statistical validity
- Document assumptions and limitations
- Provide interpretation guidance
- Include confidence measures

### Example: Adding a New Evaluation Metric

```python
# 1. Domain model
@dataclass
class BusinessLogicCoverage:
    """Domain model for business logic coverage metric."""
    score: float
    covered_rules: List[str]
    total_rules: List[str]

# 2. Port extension
class EvaluationPort(Protocol):
    def calculate_business_logic_coverage(
        self,
        test_content: str,
        source_code: str
    ) -> BusinessLogicCoverage:
        """Calculate business logic coverage."""
        ...

# 3. Adapter implementation
class TestcraftEvaluationAdapter(EvaluationPort):
    def calculate_business_logic_coverage(
        self,
        test_content: str,
        source_code: str
    ) -> BusinessLogicCoverage:
        """Implementation of business logic coverage calculation."""
        # Analysis logic here
        pass

# 4. Configuration
[evaluation.metrics]
business_logic_coverage = true
business_logic_threshold = 0.8

# 5. Tests
class TestBusinessLogicCoverage:
    def test_calculates_coverage_correctly(self):
        # Test implementation
        pass
```

## Release Process

### Version Numbers

TestCraft uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including evaluation tests
4. **Update documentation** for any new features
5. **Create release notes** highlighting evaluation improvements
6. **Tag release** with version number

### Evaluation System Releases

Special attention for evaluation system changes:

- **Backwards Compatibility**: Ensure existing evaluation configs work
- **Performance Impact**: Document performance changes
- **Statistical Validity**: Verify statistical methods are correct
- **Documentation**: Update evaluation documentation thoroughly

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn the codebase
- Focus on technical merit in discussions

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check docs first for usage questions

### Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Major contributions highlighted
- **Documentation**: Expert contributors may be listed as reviewers

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues and PRs
3. Ask in GitHub Discussions
4. Create an issue for clarification

Thank you for contributing to TestCraft! Your contributions help make AI-powered test generation better for everyone.
