"""
Backward compatibility shim for test_mapper module.

This shim re-exports the TestMapper class and related components from the 
new modular test_mapper package to maintain API compatibility.
"""

# Re-export main classes from the new package structure
from .test_mapper import (
    TestMapper,
    TestMappingError,
    ImportInfo,
    UsageInfo,
    TestFileAnalysis,
    TestFileAnalyzer,
    ModulePathHelper,
    calculate_mapping_scores,
    extract_covered_elements,
    estimate_coverage_percentage,
    estimate_coverage_percentage_enhanced,
    calculate_test_priority,
    calculate_coverage_percentage,
    MIN_MAPPING_SCORE_THRESHOLD,
)

# Make all previously exported items available
__all__ = [
    "TestMapper",
    "TestMappingError",
    "ImportInfo",
    "UsageInfo",
    "TestFileAnalysis",
    "TestFileAnalyzer",
    "ModulePathHelper",
    "calculate_mapping_scores",
    "extract_covered_elements",
    "estimate_coverage_percentage",
    "estimate_coverage_percentage_enhanced",
    "calculate_test_priority", 
    "calculate_coverage_percentage",
    "MIN_MAPPING_SCORE_THRESHOLD",
]
