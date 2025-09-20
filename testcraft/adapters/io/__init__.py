"""
IO adapters for file operations.

This module provides adapters for various file I/O operations including
test file writing with different strategies and safety policies.
"""

from . import async_runner, python_formatters, subprocess_safe
from .artifact_store import (
    ArtifactMetadata,
    ArtifactStoreAdapter,
    ArtifactStoreError,
    ArtifactType,
    CleanupPolicy,
    store_coverage_report,
    store_generated_test,
    store_llm_response,
)
from .file_discovery import FileDiscoveryError, FileDiscoveryService
from .reporter_json import JsonReportAdapter, ReportError
from .rich_cli import (
    TESTCRAFT_THEME,
    RichCliComponents,
    create_default_cli,
    print_coverage_summary,
    print_project_overview,
    print_test_results_summary,
)
from .safety import SafetyPolicies
from .state_json import StateJsonAdapter, StateJsonError
from .ui_rich import RichUIAdapter, UIError
from .writer_append import WriterAppendAdapter
from .writer_ast_merge import WriterASTMergeAdapter

__all__ = [
    "WriterAppendAdapter",
    "WriterASTMergeAdapter",
    "StateJsonAdapter",
    "StateJsonError",
    "JsonReportAdapter",
    "ReportError",
    "ArtifactStoreAdapter",
    "ArtifactStoreError",
    "ArtifactType",
    "ArtifactMetadata",
    "CleanupPolicy",
    "store_coverage_report",
    "store_generated_test",
    "store_llm_response",
    "RichUIAdapter",
    "UIError",
    "RichCliComponents",
    "TESTCRAFT_THEME",
    "create_default_cli",
    "print_coverage_summary",
    "print_test_results_summary",
    "print_project_overview",
    "SafetyPolicies",
    "FileDiscoveryService",
    "FileDiscoveryError",
    "async_runner",
    "subprocess_safe",
    "python_formatters",
]
