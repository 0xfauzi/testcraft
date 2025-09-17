"""
Refinement package for pytest execution and test refinement.

This package contains modular components for:
- Running pytest and classifying results
- Parsing and formatting failure output  
- Annotating failed tests with guidance
- Orchestrating LLM-based refinement
- Writing and formatting test files
- Guardrails and validation for refined content
- Safe application of changes with backup/rollback
- Manual suggestions and preflight analysis

The main facade is PytestRefiner which delegates to these modules.
"""

from .apply import SafeApplyService
from .guardrails import RefinementGuardrails
from .manual_suggestions import ManualSuggestionsService
from .refiner import PytestRefiner

__all__ = [
    "PytestRefiner",
    "RefinementGuardrails", 
    "SafeApplyService",
    "ManualSuggestionsService",
]
