"""
Guardrails and validation for test refinement operations.

This module provides validation logic for refined test content, including
AST equivalence checks, empty/None/identical content detection, and syntax validation.
"""

import ast
import difflib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class RefinementGuardrails:
    """
    Validation and guardrails for refined test content.
    
    Provides comprehensive validation including syntax checks, content comparison,
    AST equivalence detection, and cosmetic-only change detection.
    """

    def __init__(
        self,
        reject_empty: bool = True,
        reject_literal_none: bool = True,
        reject_identical: bool = True,
        validate_syntax: bool = True,
        treat_cosmetic_as_no_change: bool = True,
        allow_ast_equivalence_check: bool = True,
    ):
        """
        Initialize guardrails with validation settings.

        Args:
            reject_empty: Reject empty or whitespace-only content
            reject_literal_none: Reject literal "None"/"null" strings
            reject_identical: Reject content identical to input
            validate_syntax: Validate Python syntax
            treat_cosmetic_as_no_change: Treat cosmetic changes as no-change
            allow_ast_equivalence_check: Enable AST equivalence checking
        """
        self.reject_empty = reject_empty
        self.reject_literal_none = reject_literal_none
        self.reject_identical = reject_identical
        self.validate_syntax = validate_syntax
        self.treat_cosmetic_as_no_change = treat_cosmetic_as_no_change
        self.allow_ast_equivalence_check = allow_ast_equivalence_check

    def validate_refined_content(
        self, refined_content: str | None, current_content: str
    ) -> dict[str, Any]:
        """
        Validate refined content with layered checks and detailed statuses.
        
        Args:
            refined_content: Content returned by LLM
            current_content: Current test file content for comparison
            
        Returns:
            Dictionary with validation result:
                - is_valid: bool
                - reason: str (if not valid)
                - status: str (detailed status)
                - diff_snippet: str (unified diff for logging)
        """
        # Check for None/non-string content - this is invalid output, not "no change"
        if refined_content is None:
            return {
                "is_valid": False, 
                "reason": "LLM returned None content",
                "status": "llm_invalid_output",
                "diff_snippet": "N/A - None content"
            }
        
        if not isinstance(refined_content, str):
            return {
                "is_valid": False, 
                "reason": f"LLM returned non-string content: {type(refined_content)}",
                "status": "llm_invalid_output",
                "diff_snippet": f"N/A - {type(refined_content)} content"
            }
        
        # Check for empty or whitespace-only content - this is invalid output
        if self.reject_empty and not refined_content.strip():
            return {
                "is_valid": False, 
                "reason": "LLM returned empty or whitespace-only content",
                "status": "llm_invalid_output",
                "diff_snippet": "N/A - empty content"
            }
        
        # Check for literal "None", "null" strings (case-insensitive) - this is invalid output
        if self.reject_literal_none:
            content_lower = refined_content.strip().lower()
            if content_lower in ("none", "null"):
                return {
                    "is_valid": False, 
                    "reason": f"LLM returned literal '{refined_content.strip()}' content",
                    "status": "llm_invalid_output",
                    "diff_snippet": f"N/A - literal '{refined_content.strip()}'"
                }
        
        # Layered content comparison checks
        diff_snippet = self._compute_diff_snippet(current_content, refined_content)
        
        # Layer 1: Normalize newlines and trailing spaces
        normalized_current = self._normalize_content(current_content)
        normalized_refined = self._normalize_content(refined_content)
        
        if normalized_current == normalized_refined:
            if self.reject_identical:
                return {
                    "is_valid": False,
                    "reason": "LLM returned identical content to input (normalized)",
                    "status": "content_identical",
                    "diff_snippet": diff_snippet
                }
        
        # Layer 2: Check if only whitespace/formatting differs
        if self._is_cosmetic_only_change(current_content, refined_content):
            if self.reject_identical and self.treat_cosmetic_as_no_change:
                return {
                    "is_valid": False,
                    "reason": "LLM returned content with only cosmetic formatting changes",
                    "status": "content_cosmetic_noop",
                    "diff_snippet": diff_snippet
                }
        
        # Layer 3: Optional AST comparison for Python tests
        if self.allow_ast_equivalence_check and self._is_ast_equivalent(current_content, refined_content):
            if self.reject_identical:
                return {
                    "is_valid": False,
                    "reason": "LLM returned semantically identical Python code (AST equivalent)",
                    "status": "content_semantically_identical",
                    "diff_snippet": diff_snippet
                }
        
        # Validate Python syntax if enabled - this is a syntax error
        if self.validate_syntax:
            try:
                ast.parse(refined_content)
            except SyntaxError as e:
                return {
                    "is_valid": False, 
                    "reason": f"LLM returned invalid Python syntax: {e}",
                    "status": "syntax_error",
                    "diff_snippet": diff_snippet
                }
            except Exception as e:
                return {
                    "is_valid": False, 
                    "reason": f"LLM returned unparseable Python: {e}",
                    "status": "syntax_error",
                    "diff_snippet": diff_snippet
                }
        
        return {
            "is_valid": True, 
            "status": "valid", 
            "diff_snippet": diff_snippet
        }

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for comparison by handling newlines and trailing spaces.
        
        Args:
            content: Raw content string
            
        Returns:
            Normalized content string
        """
        # Normalize line endings to \n
        normalized = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Strip trailing spaces from each line but preserve structure
        lines = normalized.split('\n')
        normalized_lines = [line.rstrip() for line in lines]
        
        # Remove trailing empty lines
        while normalized_lines and not normalized_lines[-1]:
            normalized_lines.pop()
        
        return '\n'.join(normalized_lines)
    
    def _is_cosmetic_only_change(self, original: str, refined: str) -> bool:
        """
        Check if changes are only cosmetic (formatting, whitespace).
        
        This uses simple heuristics to detect if the changes could be made by
        formatters like Black or tools like ruff.
        
        Args:
            original: Original content
            refined: Refined content
            
        Returns:
            True if changes appear to be cosmetic only
        """
        # Remove all whitespace and compare
        original_clean = re.sub(r'\s+', ' ', original.strip())
        refined_clean = re.sub(r'\s+', ' ', refined.strip())
        
        if original_clean == refined_clean:
            return True
        
        # Check if only indentation changes
        original_lines = original.strip().split('\n')
        refined_lines = refined.strip().split('\n')
        
        if len(original_lines) != len(refined_lines):
            return False
        
        # Compare lines after stripping leading/trailing whitespace
        for orig_line, ref_line in zip(original_lines, refined_lines):
            if orig_line.strip() != ref_line.strip():
                return False
        
        # If we get here, only whitespace differs
        return True
    
    def _is_ast_equivalent(self, original: str, refined: str) -> bool:
        """
        Check if two Python code strings are AST-equivalent.
        
        Args:
            original: Original Python code
            refined: Refined Python code
            
        Returns:
            True if AST structures are equivalent, False otherwise
        """
        try:
            # Parse both into ASTs
            original_ast = ast.parse(original)
            refined_ast = ast.parse(refined)
            
            # Convert ASTs to comparable form (dump removes location info)
            original_dump = ast.dump(original_ast)
            refined_dump = ast.dump(refined_ast)
            
            return original_dump == refined_dump
            
        except (SyntaxError, ValueError, TypeError) as e:
            # If either fails to parse, they're not equivalent
            logger.debug(f"AST comparison failed: {e}")
            return False
    
    def _compute_diff_snippet(self, original: str, refined: str, max_hunks: int = 3) -> str:
        """
        Compute a short unified diff snippet for logging.
        
        Args:
            original: Original content
            refined: Refined content  
            max_hunks: Maximum number of diff hunks to include
            
        Returns:
            Short unified diff string
        """
        try:
            original_lines = original.splitlines(keepends=True)
            refined_lines = refined.splitlines(keepends=True)
            
            diff_lines = list(difflib.unified_diff(
                original_lines,
                refined_lines,
                fromfile='original',
                tofile='refined',
                n=10  # Context lines
            ))
            
            if not diff_lines:
                return "No differences"
            
            # Limit to first max_hunks hunks
            limited_diff = []
            hunk_count = 0
            
            for line in diff_lines:
                limited_diff.append(line)
                if line.startswith('@@'):
                    hunk_count += 1
                    if hunk_count > max_hunks:
                        limited_diff.append("... (diff truncated)\n")
                        break
            
            return ''.join(limited_diff)
            
        except Exception as e:
            return f"Diff computation failed: {e}"
