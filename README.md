# TestCraft

AI-powered test generation tool for Python projects.

## Overview

TestCraft is an intelligent test generation system that analyzes Python codebases and automatically generates comprehensive test suites using Large Language Models (LLMs). Built with Clean Architecture principles, it provides a robust, extensible platform for automated test creation.

## Features

- **Intelligent Test Generation**: Uses LLMs to generate contextually appropriate tests
- **Clean Architecture**: Modular design with clear separation of concerns
- **Multiple LLM Support**: Compatible with various LLM providers
- **Coverage Analysis**: Integrates with pytest and coverage tools
- **Extensible Design**: Plugin-based architecture for custom adapters
- **CLI Interface**: Easy-to-use command-line tool

## Installation

### Prerequisites

- Python 3.11 or higher
- uv (recommended) or pip

### Setup

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

## Usage

```bash
# Show version
testcraft version

# Generate tests for a Python module
testcraft generate <module-path>

# Analyze coverage
testcraft coverage <module-path>

# Get help
testcraft --help
```

## Development

### Project Structure

```
testcraft/
├── domain/          # Core business logic and entities
├── application/     # Use cases and application services
├── adapters/        # External integrations (LLM, file system, etc.)
├── ports/           # Interface definitions
└── cli/             # Command-line interface
```

### Development Tools

- **ruff**: Linting and code formatting
- **black**: Code formatting
- **mypy**: Type checking
- **pytest**: Testing framework

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=testcraft

# Run linting
uv run ruff check .

# Format code
uv run black .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
