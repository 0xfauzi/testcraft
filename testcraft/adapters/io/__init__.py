"""
IO adapters for file operations.

This module provides adapters for various file I/O operations including
test file writing with different strategies and safety policies.
"""

from .writer_append import WriterAppendAdapter
from .writer_ast_merge import WriterASTMergeAdapter
from .state_json import StateJsonAdapter, StateJsonError
from .reporter_json import JsonReportAdapter, ReportError
from .artifact_store import (
    ArtifactStoreAdapter,
    ArtifactStoreError,
    ArtifactType,
    ArtifactMetadata,
    CleanupPolicy,
    store_coverage_report,
    store_generated_test,
    store_llm_response,
)
from .ui_rich import RichUIAdapter, UIError
from .rich_cli import (
    RichCliComponents,
    TESTCRAFT_THEME,
    create_default_cli,
    print_coverage_summary,
    print_test_results_summary,
    print_project_overview,
)
from .safety import SafetyPolicies
from .file_discovery import FileDiscoveryService, FileDiscoveryError
from . import async_runner
from . import subprocess_safe
from . import python_formatters

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
    "python_formatters"
]
