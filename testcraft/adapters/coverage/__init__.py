"""Coverage adapters for measuring code coverage."""

from .coverage_py_adapter import CoveragePyAdapter, NoOpCoverageAdapter

__all__ = ["CoveragePyAdapter", "NoOpCoverageAdapter"]
