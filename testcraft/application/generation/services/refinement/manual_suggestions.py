"""
Manual fix suggestions and preflight analysis for test refinement.

This module provides functionality for generating manual fix suggestions,
preflight canonicalization checks, and import path analysis for test files.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Optional

from .....application.generation.services.refinement.refiner import PytestRefiner

logger = logging.getLogger(__name__)


class ManualSuggestionsService:
    """
    Service for generating manual fix suggestions and preflight analysis.
    
    Provides preflight canonicalization suggestions, import path analysis,
    and manual fix recommendations when automated refinement fails.
    """

    def __init__(self):
        """Initialize the manual suggestions service."""
        pass

    def get_preflight_suggestions(self, current_content: str) -> str:
        """
        Get preflight canonicalization suggestions without auto-editing.
        
        Args:
            current_content: Current test file content
            
        Returns:
            String with suggestions or empty string if none
        """
        suggestions = []
        
        if not current_content:
            return ""
        
        # Check for common dunder/keyword/import issues
        lines = current_content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for missing underscores in dunders
            dunder_issues = [
                ("_init_", "__init__"),
                ("_enter_", "__enter__"),
                ("_exit_", "__exit__"),
                ("_name_", "__name__"),
                ("_main_", "__main__"),
            ]
            
            for wrong, correct in dunder_issues:
                if wrong in line and correct not in line:
                    suggestions.append(f"Line {i}: Replace '{wrong}' with '{correct}'")
            
            # Check for incorrect casing of Python keywords
            case_issues = [
                ("none", "None"),
                ("true", "True"), 
                ("false", "False"),
            ]
            
            for wrong, correct in case_issues:
                if f" {wrong}" in line.lower() or f"={wrong}" in line.lower():
                    if wrong.lower() in line.lower() and correct not in line:
                        suggestions.append(f"Line {i}: Use '{correct}' instead of '{wrong}' (case sensitive)")
            
            # Check for common import case mistakes
            if line_stripped.startswith("import ") or line_stripped.startswith("from "):
                import_issues = [
                    ("table", "Table"),  # rich.table.Table
                    ("console", "Console"),  # rich.console.Console
                ]
                
                for wrong, correct in import_issues:
                    if wrong in line and correct not in line:
                        suggestions.append(f"Line {i}: Check import casing - may need '{correct}' instead of '{wrong}'")
        
        if suggestions:
            return "Found potential issues:\n" + "\n".join(f"- {s}" for s in suggestions[:5])  # Limit to top 5
        
        return "No obvious canonicalization issues detected"

    def format_source_context(self, source_context: dict[str, Any] | None) -> str:
        """
        Format source context cleanly to avoid duplication.
        
        Args:
            source_context: Source context dict from payload
            
        Returns:
            Formatted source context string
        """
        if not source_context:
            return "No source context available"
        
        if isinstance(source_context, dict):
            # Handle structured source context
            parts = []
            
            if "related_source_files" in source_context:
                files = source_context["related_source_files"][:2]  # Limit to 2
                for file_info in files:
                    if isinstance(file_info, dict):
                        path = file_info.get("path", "Unknown")
                        content = file_info.get("content", "")[:1000]  # Truncate
                        parts.append(f"File: {path}\n{content}")
            
            if parts:
                return "\n---\n".join(parts)
        
        # Fallback to string representation
        context_str = str(source_context)[:2000]  # Truncate long context
        return context_str if context_str.strip() else "Empty source context"

    def select_active_import_path(self, failure_output: str, current_content: str) -> str:
        """
        Choose the best active import path for mocking/patching targets.
        Strategy:
        1) Use PytestRefiner.extract_import_path_from_failure if plausible
        2) Otherwise, fall back to AST-derived modules from the test content
        
        Args:
            failure_output: Pytest failure output
            current_content: Current test file content
            
        Returns:
            Best active import path or empty string if none found
        """
        # Primary from failure output
        primary = PytestRefiner.extract_import_path_from_failure(failure_output)
        if primary and self._is_plausible_module_path(primary):
            return primary
        
        # AST-only fallback from test content
        ast_candidates = self._derive_import_modules_from_test_ast(current_content)
        for cand in ast_candidates:
            if self._is_plausible_module_path(cand):
                return cand
        
        # Nothing plausible found
        return ""

    def generate_manual_fix_suggestions_via_chat(
        self, 
        llm_adapter: Any,
        system_prompt: str, 
        user_prompt: str
    ) -> dict[str, Any]:
        """
        Best-effort manual suggestions using the LLM adapter's chat completion.

        This avoids schema-specific normalization by directly parsing returned JSON.
        
        Args:
            llm_adapter: LLM adapter instance
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            Dictionary with manual suggestions and metadata
        """
        # Attempt to use a generic path available in OpenAIAdapter
        try:
            # Most adapters will route through a chat completion under the hood via refine_content;
            # we re-use that path by calling a private helper if available, else fall back to
            # refine_content with a JSON-only user prompt and then parse.
            if hasattr(llm_adapter, "_chat_completion"):
                # type: ignore[attr-defined]
                resp = llm_adapter._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)  # noqa: E1101
                content = resp.get("content", "")
            else:
                # Last resort: call refine_content with empty original_content
                resp_dict = llm_adapter.refine_content(
                    original_content="",
                    refinement_instructions=user_prompt,
                    system_prompt=system_prompt,
                )
                # Try to get raw content if available, else synthesize JSON from changes_made
                content = resp_dict.get("raw_content") or resp_dict.get("changes_made") or "{}"
        except Exception:
            content = "{}"

        try:
            import json
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        
        return {
            "manual_suggestions": content if isinstance(content, str) else "",
        }

    def _is_plausible_module_path(self, module_path: str) -> bool:
        """
        Basic plausibility checks for a Python module path without using regex.
        - Must contain at least one dot (package.module)
        - Each segment must start with a letter or underscore and be alphanumeric/underscore
        - Must not end with common non-Python extensions (toml, md, txt, json, yaml, yml, ini, cfg, lock)
        - Must not contain spaces
        """
        if not isinstance(module_path, str):
            return False
        module_path = module_path.strip()
        if not module_path or " " in module_path:
            return False
        # Disallow obvious non-Python filenames
        disallowed_suffixes = (".toml", ".md", ".txt", ".json", ".yaml", ".yml", ".ini", ".cfg", ".lock")
        for suf in disallowed_suffixes:
            if module_path.lower().endswith(suf):
                return False
        # Require dotted path (package.module)
        if "." not in module_path:
            return False
        # Validate segments
        for segment in module_path.split('.'):
            if not segment:
                return False
            first = segment[0]
            if not (first.isalpha() or first == "_"):
                return False
            for ch in segment[1:]:
                if not (ch.isalnum() or ch == "_"):
                    return False
        return True

    def _derive_import_modules_from_test_ast(self, current_content: str) -> list[str]:
        """
        Use AST to derive likely application module import paths from the test file content.
        Preference:
        - from X import Y -> use X
        - import X.Y as Z -> use X.Y
        - Filter out common test/stdlib/third-party utility modules when possible
        """
        candidates: list[str] = []
        try:
            tree = ast.parse(current_content)
        except Exception:
            return candidates

        def add_candidate(mod: str) -> None:
            if not isinstance(mod, str):
                return
            mod = mod.strip()
            if not mod:
                return
            if mod in candidates:
                return
            # Light filtering of obvious non-targets
            top = mod.split('.')[0]
            filtered_tops = {
                "pytest", "unittest", "json", "re", "os", "sys", "pathlib", "typing",
                "datetime", "time", "collections", "itertools", "functools", "math",
                "rich", "logging", "schedule",
            }
            if top in filtered_tops:
                return
            candidates.append(mod)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if getattr(node, 'level', 0) == 0 and isinstance(getattr(node, 'module', None), str):
                    add_candidate(node.module)
            elif isinstance(node, ast.Import):
                for alias in getattr(node, 'names', []) or []:
                    name = getattr(alias, 'name', None)
                    if isinstance(name, str) and "." in name:
                        add_candidate(name)

        # Prefer dotted modules and plausible module paths
        plausible = [m for m in candidates if self._is_plausible_module_path(m)]
        if plausible:
            return plausible
        return candidates
