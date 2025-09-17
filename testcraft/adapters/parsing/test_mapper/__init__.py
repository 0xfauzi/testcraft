"""
Test mapper package for mapping tests to source code elements.

This package provides modular test mapping functionality split across:
- analysis: AST-based analysis of test files and dataclasses
- scoring: Scoring functions and threshold constants  
- module_paths: Module path derivation utilities
- mapper: Main TestMapper orchestrator class

The package maintains backward compatibility by re-exporting the main classes.
"""

from .analysis import ImportInfo, UsageInfo, TestFileAnalysis, TestFileAnalyzer
from .mapper import TestMapper, TestMappingError
from .module_paths import ModulePathHelper
from .scoring import (
    calculate_mapping_scores,
    extract_covered_elements,
    estimate_coverage_percentage,
    estimate_coverage_percentage_enhanced,
    calculate_test_priority,
    calculate_coverage_percentage,
    MIN_MAPPING_SCORE_THRESHOLD,
)

__all__ = [
    # Main classes
    "TestMapper",
    "TestMappingError",
    
    # Analysis components
    "ImportInfo",
    "UsageInfo", 
    "TestFileAnalysis",
    "TestFileAnalyzer",
    
    # Utility classes
    "ModulePathHelper",
    
    # Scoring functions
    "calculate_mapping_scores",
    "extract_covered_elements",
    "estimate_coverage_percentage", 
    "estimate_coverage_percentage_enhanced",
    "calculate_test_priority",
    "calculate_coverage_percentage",
    
    # Constants
    "MIN_MAPPING_SCORE_THRESHOLD",
]
