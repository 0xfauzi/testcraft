"""Test discovery and pattern configuration models."""

from typing import Literal

from pydantic import BaseModel, Field


class TestDiscoveryConfig(BaseModel):
    """Configuration for hybrid test discovery system."""

    mode: Literal["auto", "pytest-collector", "ast", "globs"] = Field(
        default="auto",
        description="Test discovery mode: 'auto' uses pytest collector with fallbacks, 'pytest-collector' uses only pytest collection, 'ast' uses AST-based classification, 'globs' uses legacy glob patterns"
    )
    
    classify_support_files: bool = Field(
        default=True,
        description="Enable classification of support files (conftest.py, fixture-only files)"
    )
    
    enable_coverage_probe: bool = Field(
        default=False,
        description="Enable optional coverage quick probe as last resort for test detection"
    )
    
    collector_timeout_sec: int = Field(
        default=15,
        ge=1,
        le=300,
        description="Timeout in seconds for pytest collection"
    )
    
    probe_timeout_sec: int = Field(
        default=20,
        ge=1,
        le=300,
        description="Timeout in seconds for coverage probe operations"
    )
    
    cache_ttl_sec: int = Field(
        default=600,
        ge=0,
        description="Cache TTL in seconds for discovery results (0 disables cache)"
    )
    
    max_ast_files: int = Field(
        default=5000,
        ge=1,
        description="Maximum number of files to parse with AST for safety in huge repos"
    )
    
    mapper_min_score: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum mapping score to consider a test file covers a source file"
    )


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

    # Test discovery configuration
    test_discovery: TestDiscoveryConfig = Field(
        default_factory=TestDiscoveryConfig,
        description="Hybrid test discovery configuration"
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
