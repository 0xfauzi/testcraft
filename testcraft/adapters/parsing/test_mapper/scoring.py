"""
Scoring functions and threshold constants for test mapping.

This module provides scoring logic for determining the strength of mappings
between source files and test files, with configurable thresholds.
"""

from pathlib import Path
from typing import Dict, Set

# Configurable thresholds
MIN_MAPPING_SCORE_THRESHOLD = 2  # Minimum score for considering a test file mapped to source
ELEMENT_COVERAGE_WEIGHT = 10     # Base percentage per covered element
MAX_BASE_COVERAGE = 80           # Maximum coverage from element count alone
SCORE_BOOST_WEIGHT = 5           # Percentage boost per mapping score point
MAX_SCORE_BOOST = 20             # Maximum boost from score


def calculate_mapping_scores(
    source_module_paths: list[str],
    test_file_mappings: dict,
    threshold: int = MIN_MAPPING_SCORE_THRESHOLD
) -> tuple[dict, int]:
    """
    Calculate mapping scores for test files against source module paths.
    
    Args:
        source_module_paths: List of possible module paths for the source
        test_file_mappings: Dict of test file -> analysis mappings
        threshold: Minimum score threshold for inclusion
        
    Returns:
        Tuple of (mapping_scores dict, total_score)
    """
    mapping_scores = {}
    total_score = 0
    
    for test_file, analysis in test_file_mappings.items():
        max_score = 0
        for module_path in source_module_paths:
            score = analysis.get_mapping_score_for_module(module_path)
            max_score = max(max_score, score)
        
        if max_score >= threshold:
            mapping_scores[test_file] = max_score
            total_score += max_score
    
    return mapping_scores, total_score


def extract_covered_elements(
    source_path: Path,
    source_module_paths: list[str],
    test_file_mappings: dict
) -> set[str]:
    """Extract source elements that appear to be covered by tests."""
    covered_elements = set()
    
    # This is a simplified implementation - could be enhanced with more
    # sophisticated analysis of which specific functions/classes are used
    for test_file, analysis in test_file_mappings.items():
        for call in analysis.usage.function_calls:
            covered_elements.add(call)
        for attr in analysis.usage.attribute_accesses:
            covered_elements.add(attr)
        for patch in analysis.usage.mock_patches:
            covered_elements.add(patch)
    
    return covered_elements


def estimate_coverage_percentage(
    source_path: Path,
    elements_covered: set[str],
    total_score: int
) -> float:
    """Estimate coverage percentage based on mapping signals."""
    if total_score == 0:
        return 0.0
    
    # Simple heuristic - could be improved with more analysis
    # Higher scores and more covered elements suggest better coverage
    base_percentage = min(100.0, total_score * 10.0)
    element_bonus = min(20.0, len(elements_covered) * 2.0)
    
    return min(100.0, base_percentage + element_bonus)


def estimate_coverage_percentage_enhanced(
    source_path: Path, 
    elements_covered: set[str], 
    total_score: int
) -> float:
    """
    Enhanced coverage estimation based on elements covered and mapping score.
    
    This version uses the configurable constants for more predictable results.
    """
    if not elements_covered:
        return 0.0
    
    # Simple heuristic: base coverage on number of elements + score boost
    base_coverage = min(len(elements_covered) * ELEMENT_COVERAGE_WEIGHT, MAX_BASE_COVERAGE)
    score_boost = min(total_score * SCORE_BOOST_WEIGHT, MAX_SCORE_BOOST)
    
    return min(base_coverage + score_boost, 100.0)


def calculate_test_priority(element_name: str, element_type: str) -> str:
    """Calculate the priority for creating tests for an element."""
    # Public functions/methods get high priority
    if not element_name.split(".")[-1].startswith("_"):
        return "high"

    # Private methods get medium priority
    if element_type.upper() == "METHOD" or element_type == "method":
        return "medium"

    # Other elements get low priority
    return "low"


def calculate_coverage_percentage(
    source_elements: list,
    test_mapping: dict
) -> float:
    """Calculate the percentage of source elements that have tests."""
    if not source_elements:
        return 100.0

    elements_with_tests = sum(
        1 for element in source_elements if test_mapping.get(element.name, [])
    )

    return (elements_with_tests / len(source_elements)) * 100.0
