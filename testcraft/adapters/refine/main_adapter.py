"""
Main adapter for test refinement operations.

This module implements the RefinePort interface, providing functionality
for refining tests based on pytest failures and other quality issues.
"""

import ast
import logging
import time
from pathlib import Path
from typing import Any

from ...adapters.io.subprocess_safe import run_subprocess_simple
from ...application.generation.services.pytest_refiner import PytestRefiner
from ...config.models import RefineConfig
from ...domain.models import RefineOutcome
from ...ports.llm_port import LLMPort
from ...ports.telemetry_port import TelemetryPort
from ...ports.writer_port import WriterPort

logger = logging.getLogger(__name__)


class RefineAdapter:
    """
    Adapter for refining tests based on failures and quality issues.

    This adapter implements the RefinePort interface and uses LLM integration
    to intelligently fix pytest failures and improve test quality.
    """

    def __init__(
        self,
        llm: LLMPort,
        config: RefineConfig | None = None,
        writer_port: WriterPort | None = None,
        telemetry_port: TelemetryPort | None = None,
    ):
        """
        Initialize the refine adapter.

        Args:
            llm: LLM adapter for generating refinements
            config: Refinement configuration with guardrails settings
            writer_port: Optional writer port for safe file operations
            telemetry_port: Optional telemetry port for observability
        """
        self.llm = llm
        self.config = config or RefineConfig()
        self.writer_port = writer_port
        self.telemetry_port = telemetry_port

        # Extract guardrails config with defaults
        guardrails = self.config.refinement_guardrails
        self.reject_empty = guardrails.get("reject_empty", True)
        self.reject_literal_none = guardrails.get("reject_literal_none", True)
        self.reject_identical = guardrails.get("reject_identical", True)
        self.validate_syntax = guardrails.get("validate_syntax", True)
        self.format_on_refine = guardrails.get("format_on_refine", True)

    def refine_from_failures(
        self,
        test_file: str | Path,
        failure_output: str,
        source_context: dict[str, Any] | None = None,
        max_iterations: int = 3,
        timeout_seconds: int = 300,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Refine a test file based on pytest failure output.

        This method iteratively refines test files by:
        1. Analyzing pytest failure output
        2. Sending failure + current test code to LLM
        3. Getting refined test code back
        4. Safely applying changes
        5. Re-running pytest to verify fixes
        6. Repeating until success or max iterations reached

        Args:
            test_file: Path to the test file that failed
            failure_output: Raw pytest failure output (stdout/stderr)
            source_context: Optional source code context for fixing
            max_iterations: Maximum number of refinement attempts
            timeout_seconds: Maximum total time to spend on refinement
            **kwargs: Additional refinement parameters

        Returns:
            Dictionary containing:
                - 'success': Whether refinement was successful
                - 'refined_content': Updated test file content if successful
                - 'iterations_used': Number of refinement iterations performed
                - 'final_status': Final pytest run status
                - 'error': Error message if refinement failed
        """
        test_path = Path(test_file)
        if not test_path.exists():
            return {
                "success": False,
                "error": f"Test file not found: {test_path}",
                "iterations_used": 0,
                "final_status": "file_not_found",
                "fix_instructions": "",
                "active_import_path": "",
                "preflight_suggestions": "",
                "llm_confidence": None,
                "improvement_areas": [],
                "iteration": 0,
            }

        start_time = time.time()
        iteration = 0
        previous_content = None

        for iteration in range(1, max_iterations + 1):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                return {
                    "success": False,
                    "error": f"Timeout after {timeout_seconds} seconds",
                    "iterations_used": iteration - 1,
                    "final_status": "timeout",
                    "fix_instructions": "",
                    "active_import_path": "",
                    "preflight_suggestions": "",
                    "llm_confidence": None,
                    "improvement_areas": [],
                    "iteration": iteration,
                }

            # Read current test content
            try:
                current_content = test_path.read_text(encoding="utf-8")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read test file: {e}",
                    "iterations_used": iteration - 1,
                    "final_status": "read_error",
                    "fix_instructions": "",
                    "active_import_path": "",
                    "preflight_suggestions": "",
                    "llm_confidence": None,
                    "improvement_areas": [],
                    "iteration": iteration,
                }

            # Check for no-change condition
            if previous_content is not None and current_content == previous_content:
                # For no-change before LLM, compute context manually
                active_import_path = self._select_active_import_path(
                    failure_output=failure_output,
                    current_content=current_content,
                )
                preflight_suggestions = self._get_preflight_suggestions(current_content)

                return {
                    "success": False,
                    "error": "No changes made in refinement iteration",
                    "iterations_used": iteration - 1,
                    "final_status": "no_change",
                    "fix_instructions": preflight_suggestions,
                    "active_import_path": active_import_path or "",
                    "preflight_suggestions": preflight_suggestions,
                    "llm_confidence": None,
                    "improvement_areas": [],
                    "iteration": iteration,
                }

            # Prepare refinement payload
            payload = self._build_refinement_payload(
                test_file=test_path,
                current_content=current_content,
                failure_output=failure_output,
                source_context=source_context,
                iteration=iteration,
                **kwargs,
            )

            # Get refined content from LLM with validation
            try:
                refined_content, validation_result = self._get_llm_refinement_validated(
                    payload, current_content
                )

                if not validation_result["is_valid"]:
                    if self.telemetry_port:
                        with self.telemetry_port.create_child_span(
                            "refine_validation_failed"
                        ) as span:
                            span.set_attribute(
                                "validation_reason", validation_result["reason"]
                            )
                            span.set_attribute(
                                "validation_status",
                                validation_result.get("status", "unknown"),
                            )
                            span.set_attribute(
                                "refined_content_length",
                                len(refined_content) if refined_content else 0,
                            )
                            if validation_result.get("suspected_prod_bug"):
                                span.set_attribute("suspected_prod_bug", True)

                    logger.debug(
                        "LLM refinement validation failed for %s: %s (status: %s)\nDiff snippet:\n%s",
                        test_path,
                        validation_result["reason"],
                        validation_result.get("status", "unknown"),
                        validation_result.get("diff_snippet", "No diff available"),
                    )

                    # Check if production bug was suspected
                    if validation_result.get("suspected_prod_bug"):
                        logger.info(
                            "Production bug suspected for %s: %s",
                            test_path,
                            validation_result["suspected_prod_bug"],
                        )

                        if self.config.report_suspected_prod_bugs:
                            return {
                                "success": False,
                                "error": "Suspected production bug detected",
                                "iterations_used": iteration,
                                "final_status": "prod_bug_suspected",
                                "suspected_prod_bug": validation_result[
                                    "suspected_prod_bug"
                                ],
                                "fix_instructions": validation_result.get(
                                    "changes_made", ""
                                ),
                                "active_import_path": validation_result.get(
                                    "active_import_path", ""
                                ),
                                "preflight_suggestions": validation_result.get(
                                    "preflight_suggestions", ""
                                ),
                                "llm_confidence": validation_result.get(
                                    "llm_confidence"
                                ),
                                "improvement_areas": validation_result.get(
                                    "improvement_areas", []
                                ),
                                "iteration": iteration,
                            }

                    # Map validation status to final_status appropriately
                    validation_status = validation_result.get("status", "unknown")
                    if validation_status in (
                        "content_identical",
                        "content_cosmetic_noop",
                        "content_semantically_identical",
                    ):
                        # These are all true "no change" scenarios
                        final_status = "llm_no_change"
                    elif validation_status == "llm_invalid_output":
                        # This is invalid output (None, empty, etc.) - should retry
                        final_status = "llm_invalid_output"
                    elif validation_status == "syntax_error":
                        # Syntax error in refined content
                        final_status = "syntax_error"
                    else:
                        # Unknown validation failure
                        final_status = "validation_failed"

                    return {
                        "success": False,
                        "error": validation_result["reason"],
                        "iterations_used": iteration,
                        "final_status": final_status,
                        "fix_instructions": validation_result.get("changes_made")
                        or self._get_preflight_suggestions(current_content),
                        "active_import_path": validation_result.get(
                            "active_import_path", ""
                        ),
                        "preflight_suggestions": validation_result.get(
                            "preflight_suggestions", ""
                        ),
                        "llm_confidence": validation_result.get("llm_confidence"),
                        "improvement_areas": validation_result.get(
                            "improvement_areas", []
                        ),
                        "iteration": iteration,
                    }

            except Exception as e:
                if self.telemetry_port:
                    with self.telemetry_port.create_child_span(
                        "refine_llm_error"
                    ) as span:
                        span.set_attribute("error", str(e))

                # For LLM errors, we don't have validation_result, so compute context manually
                active_import_path = self._select_active_import_path(
                    failure_output=payload.get("pytest_failure_output", ""),
                    current_content=current_content,
                )
                preflight_suggestions = self._get_preflight_suggestions(current_content)

                return {
                    "success": False,
                    "error": f"LLM refinement failed: {e}",
                    "iterations_used": iteration,
                    "final_status": "llm_error",
                    "fix_instructions": preflight_suggestions,
                    "active_import_path": active_import_path or "",
                    "preflight_suggestions": preflight_suggestions,
                    "llm_confidence": None,
                    "improvement_areas": [],
                    "iteration": iteration,
                }

            # Apply changes safely via WriterPort or fallback
            try:
                write_result = self._write_refined_content_safely(
                    test_path, refined_content
                )
                if not write_result.get("success", False):
                    if self.telemetry_port:
                        with self.telemetry_port.create_child_span(
                            "refine_write_failed"
                        ) as span:
                            span.set_attribute(
                                "write_error", write_result.get("error", "Unknown")
                            )

                    return {
                        "success": False,
                        "error": f"Failed to apply refinement: {write_result.get('error', 'Unknown')}",
                        "iterations_used": iteration,
                        "final_status": "apply_error",
                        "fix_instructions": validation_result.get("changes_made")
                        or self._get_preflight_suggestions(current_content),
                        "active_import_path": validation_result.get(
                            "active_import_path", ""
                        ),
                        "preflight_suggestions": validation_result.get(
                            "preflight_suggestions", ""
                        ),
                        "llm_confidence": validation_result.get("llm_confidence"),
                        "improvement_areas": validation_result.get(
                            "improvement_areas", []
                        ),
                        "iteration": iteration,
                    }

                previous_content = current_content

                if self.telemetry_port:
                    with self.telemetry_port.create_child_span(
                        "refine_write_success"
                    ) as span:
                        span.set_attribute("write_succeeded", True)
                        span.set_attribute(
                            "refined_content_length", len(refined_content)
                        )

            except Exception as e:
                if self.telemetry_port:
                    with self.telemetry_port.create_child_span(
                        "refine_write_exception"
                    ) as span:
                        span.set_attribute("exception", str(e))

                # For write exceptions, we may still have validation_result from earlier
                return {
                    "success": False,
                    "error": f"Failed to apply refinement: {e}",
                    "iterations_used": iteration,
                    "final_status": "apply_error",
                    "fix_instructions": validation_result.get("changes_made", "")
                    if "validation_result" in locals()
                    else self._get_preflight_suggestions(current_content),
                    "active_import_path": validation_result.get(
                        "active_import_path", ""
                    )
                    if "validation_result" in locals()
                    else "",
                    "preflight_suggestions": validation_result.get(
                        "preflight_suggestions", ""
                    )
                    if "validation_result" in locals()
                    else self._get_preflight_suggestions(current_content),
                    "llm_confidence": validation_result.get("llm_confidence")
                    if "validation_result" in locals()
                    else None,
                    "improvement_areas": validation_result.get("improvement_areas", [])
                    if "validation_result" in locals()
                    else [],
                    "iteration": iteration,
                }

            # Re-run pytest to verify fixes
            pytest_result = self._run_pytest_verification(test_path)

            if pytest_result["success"]:
                return {
                    "success": True,
                    "refined_content": refined_content,
                    "iterations_used": iteration,
                    "final_status": "success",
                }
            else:
                # Update failure output for next iteration
                failure_output = pytest_result.get("output", failure_output)

        # Max iterations reached - provide final context for annotation
        final_active_import_path = self._select_active_import_path(
            failure_output=failure_output,
            current_content=previous_content if previous_content else "",
        )
        final_preflight = self._get_preflight_suggestions(
            previous_content if previous_content else ""
        )

        return {
            "success": False,
            "error": f"Max iterations ({max_iterations}) reached without success",
            "iterations_used": max_iterations,
            "final_status": "max_iterations",
            "fix_instructions": final_preflight,
            "active_import_path": final_active_import_path or "",
            "preflight_suggestions": final_preflight,
            "llm_confidence": None,
            "improvement_areas": [],
            "iteration": max_iterations,
        }

    def _build_refinement_payload(
        self,
        test_file: Path,
        current_content: str,
        failure_output: str,
        source_context: dict[str, Any] | None = None,
        iteration: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Build the payload for LLM refinement request.

        Args:
            test_file: Path to the test file
            current_content: Current content of the test file
            failure_output: Pytest failure output
            source_context: Optional source code context
            iteration: Current iteration number
            **kwargs: Additional context

        Returns:
            Dictionary payload for LLM request
        """
        payload = {
            "task": "refine_failing_test",
            "test_file_path": str(test_file),
            "current_test_content": current_content,
            "pytest_failure_output": failure_output,
            "iteration": iteration,
            "instructions": [
                "Analyze the pytest failure output and current test content",
                "Identify the root cause of the test failure",
                "Generate fixed test code that addresses the specific failure",
                "Ensure the fix is minimal and focused on the actual issue",
                "Preserve existing test structure and style where possible",
                "Return only the corrected test file content",
            ],
        }

        # Add source context if available
        if source_context:
            payload["source_context"] = source_context

        # Add any additional context from kwargs
        payload.update({k: v for k, v in kwargs.items() if k not in payload})

        return payload

    def _payload_to_instructions(self, payload: dict[str, Any]) -> str:
        """
        Convert refinement payload to instructions for LLM.
        
        Args:
            payload: Refinement request payload
            
        Returns:
            Formatted instructions string for LLM
        """
        instructions = []
        
        # Add basic task information
        test_file_path = payload.get("test_file_path", "unknown")
        iteration = payload.get("iteration", 1)
        
        instructions.append(f"Test File: {test_file_path}")
        instructions.append(f"Iteration: {iteration}")
        instructions.append("")
        
        # Add current test content
        current_content = payload.get("current_test_content", "")
        if current_content:
            instructions.append("Current Test Content:")
            instructions.append("```python")
            instructions.append(current_content)
            instructions.append("```")
            instructions.append("")
        
        # Add failure output
        failure_output = payload.get("pytest_failure_output", "")
        if failure_output:
            instructions.append("Pytest Failure Output:")
            instructions.append("```")
            instructions.append(failure_output)
            instructions.append("```")
            instructions.append("")
        
        # Add source context if available
        source_context = payload.get("source_context")
        if source_context:
            instructions.append("Source Code Context:")
            for key, value in source_context.items():
                instructions.append(f"- {key}: {value}")
            instructions.append("")
        
        # Add strict preservation rules if enabled
        if hasattr(self.config, 'strict_assertion_preservation') and self.config.strict_assertion_preservation:
            instructions.append("STRICT SEMANTIC PRESERVATION MODE ACTIVE")
            instructions.append("- DO NOT weaken assertions")
            instructions.append("- DO NOT change expected values to match buggy")
            instructions.append("- If test failure indicates production bug, report as suspected_prod_bug")
            instructions.append("")
        
        # Add task instructions
        task_instructions = payload.get("instructions", [])
        if task_instructions:
            instructions.append("Task Instructions:")
            for instruction in task_instructions:
                instructions.append(f"- {instruction}")
        
        return "\n".join(instructions)

    def _get_preflight_suggestions(self, current_content: str) -> str:
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
        lines = current_content.split("\n")

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
                        suggestions.append(
                            f"Line {i}: Use '{correct}' instead of '{wrong}' (case sensitive)"
                        )

            # Check for common import case mistakes
            if line_stripped.startswith("import ") or line_stripped.startswith("from "):
                import_issues = [
                    ("table", "Table"),  # rich.table.Table
                    ("console", "Console"),  # rich.console.Console
                ]

                for wrong, correct in import_issues:
                    if wrong in line and correct not in line:
                        suggestions.append(
                            f"Line {i}: Check import casing - may need '{correct}' instead of '{wrong}'"
                        )

        if suggestions:
            return "Found potential issues:\n" + "\n".join(
                f"- {s}" for s in suggestions[:5]
            )  # Limit to top 5

        return "No obvious canonicalization issues detected"

    def _format_source_context(self, source_context: dict[str, Any] | None) -> str:
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

    def _get_llm_refinement_validated(
        self, payload: dict[str, Any], current_content: str
    ) -> tuple[str, dict[str, Any]]:
        """
        Get refined test content from LLM with comprehensive validation.

        Args:
            payload: Refinement request payload
            current_content: Current test file content for comparison

        Returns:
            Tuple of (refined_content, validation_result) where validation_result contains:
                - is_valid: bool
                - reason: str (if not valid)
                - status: str (llm_invalid_output, llm_no_change, etc.)
                - suspected_prod_bug: dict (if production bug detected)

        Raises:
            Exception: If LLM request fails
        """
        # Extract active import path from failure output
        failure_output = payload.get("pytest_failure_output", "")
        active_import_path = self._select_active_import_path(
            failure_output=failure_output,
            current_content=current_content,
        )

        # Build preflight suggestions
        preflight_suggestions = self._get_preflight_suggestions(current_content)

        # Prepare context for the prompt registry template
        prompt_context = {
            "code_content": current_content,
            "failure_output": failure_output,
            "active_import_path": active_import_path or "Not detected",
            "preflight_suggestions": preflight_suggestions,
            "source_context": self._format_source_context(
                payload.get("source_context")
            ),
            "version": "v1",
        }

        # Log refinement request for debugging
        logger.debug(
            "Refinement request for %s (iteration %d):\n"
            "- Current content length: %d chars\n"
            "- Failure output length: %d chars\n"
            "- Active import path: %s\n"
            "- Preflight suggestions: %s",
            payload.get("test_file_path", "unknown"),
            payload.get("iteration", 1),
            len(current_content),
            len(failure_output),
            active_import_path or "None",
            "Yes" if preflight_suggestions.strip() else "None",
        )

        # Use prompt registry for system and user prompts
        from ...prompts.registry import PromptRegistry

        prompt_registry = PromptRegistry()

        system_prompt = prompt_registry.get_system_prompt("llm_content_refinement")
        user_prompt = prompt_registry.get_user_prompt(
            "llm_content_refinement", **prompt_context
        )

        # Make LLM request passing system and user prompts separately
        response = self.llm.refine_content(
            original_content=current_content,
            refinement_instructions=user_prompt,
            system_prompt=system_prompt,
        )

        # Log raw LLM response for debugging
        logger.debug(
            "LLM refinement raw response for %s:\n"
            "- Response type: %s\n"
            "- Response keys: %s\n"
            "- Response preview: %.500s...",
            payload.get("test_file_path", "unknown"),
            type(response).__name__,
            list(response.keys()) if isinstance(response, dict) else "N/A",
            str(response)[:500] if response else "None",
        )

        # Extract test content from response
        refined_content = self._extract_refined_content(response)

        # Extract LLM metadata from response
        changes_made = None
        improvement_areas = None
        confidence = None
        suspected_bug = None

        if isinstance(response, dict):
            changes_made = response.get("changes_made")
            improvement_areas = response.get("improvement_areas")
            confidence = response.get("confidence")
            suspected_bug = response.get("suspected_prod_bug")

            if suspected_bug and suspected_bug.strip().lower() not in (
                "null",
                "none",
                "",
            ):
                logger.info(
                    "LLM detected suspected production bug for %s: %s",
                    payload.get("test_file_path", "unknown"),
                    suspected_bug,
                )

        # Log extracted content for debugging
        logger.debug(
            "Extracted refined content for %s:\n"
            "- Content type: %s\n"
            "- Content length: %d chars\n"
            "- Content preview: %.200s...\n"
            "- Content is None: %s\n"
            "- Content is 'None'/'null': %s\n"
            "- Changes made: %s\n"
            "- Confidence: %s",
            payload.get("test_file_path", "unknown"),
            type(refined_content).__name__,
            len(refined_content) if refined_content else 0,
            refined_content[:200] if refined_content else "None/Empty",
            refined_content is None,
            refined_content.strip().lower() in ("none", "null")
            if refined_content
            else False,
            "Yes" if changes_made else "None",
            confidence if confidence is not None else "None",
        )

        # Validate the refined content
        validation_result = self._validate_refined_content(
            refined_content, current_content
        )

        # Add LLM metadata to validation result
        if changes_made:
            validation_result["changes_made"] = changes_made
        if improvement_areas:
            validation_result["improvement_areas"] = improvement_areas
        if confidence is not None:
            validation_result["llm_confidence"] = confidence
        if suspected_bug:
            validation_result["suspected_prod_bug"] = suspected_bug

        # Add context information to validation result
        validation_result["active_import_path"] = active_import_path or ""
        validation_result["preflight_suggestions"] = preflight_suggestions

        # Log validation result for debugging
        logger.debug(
            "Validation result for %s:\n"
            "- Is valid: %s\n"
            "- Reason: %s\n"
            "- Status: %s\n"
            "- Suspected bug: %s",
            payload.get("test_file_path", "unknown"),
            validation_result.get("is_valid"),
            validation_result.get("reason", "N/A"),
            validation_result.get("status", "N/A"),
            validation_result.get("suspected_prod_bug", "None"),
        )

        return refined_content, validation_result

    def _is_plausible_module_path(self, module_path: str) -> bool:
        """
        Basic plausibility checks for a Python module path without using regex.
        - Must contain at least one dot (package.module)
        - Each segment must start with a letter or underscore and be alphanumeric/underscore
        - Must not end with common non-Python extensions (toml, md, txt, json, yaml, yml, ini, cfg, lock)
        - Must not contain spaces
        """
        module_path = module_path.strip()
        if not module_path or " " in module_path:
            return False
        # Disallow obvious non-Python filenames
        disallowed_suffixes = (
            ".toml",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".lock",
        )
        for suf in disallowed_suffixes:
            if module_path.lower().endswith(suf):
                return False
        # Require dotted path (package.module)
        if "." not in module_path:
            return False
        # Validate segments
        for segment in module_path.split("."):
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
            mod = mod.strip()
            if not mod:
                return
            if mod in candidates:
                return
            # Light filtering of obvious non-targets
            top = mod.split(".")[0]
            filtered_tops = {
                "pytest",
                "unittest",
                "json",
                "re",
                "os",
                "sys",
                "pathlib",
                "typing",
                "datetime",
                "time",
                "collections",
                "itertools",
                "functools",
                "math",
                "rich",
                "logging",
                "schedule",
            }
            if top in filtered_tops:
                return
            candidates.append(mod)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = getattr(node, "module", None)
                if getattr(node, "level", 0) == 0 and isinstance(module_name, str):
                    add_candidate(module_name)
            elif isinstance(node, ast.Import):
                for alias in getattr(node, "names", []) or []:
                    name = getattr(alias, "name", None)
                    if name and isinstance(name, str) and "." in name:
                        add_candidate(name)

        # Prefer dotted modules and plausible module paths
        plausible = [m for m in candidates if self._is_plausible_module_path(m)]
        if plausible:
            return plausible
        return candidates

    def _select_active_import_path(
        self, failure_output: str, current_content: str
    ) -> str:
        """
        Choose the best active import path for mocking/patching targets.
        Strategy:
        1) Use PytestRefiner.extract_import_path_from_failure if plausible
        2) Otherwise, fall back to AST-derived modules from the test content
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

    def _extract_refined_content(self, llm_response: dict[str, Any]) -> str:
        """
        Extract refined test content from LLM refine_content response.

        Args:
            llm_response: LLM response dictionary from refine_content

        Returns:
            Extracted refined content
        """
        # The refine_content method returns a dict with 'refined_content' key
        if isinstance(llm_response, dict) and "refined_content" in llm_response:
            return str(llm_response["refined_content"]).strip()

        # Fallback: treat as string and extract code if needed
        response_str = str(llm_response)
        return self._extract_test_content(response_str)

    def _extract_test_content(self, llm_response: str) -> str:
        """
        Extract test content from LLM response.

        Args:
            llm_response: Raw LLM response

        Returns:
            Extracted test content
        """
        # Try to extract code from markdown code fences
        if "```" in llm_response:
            lang_tag = "```python"
            if lang_tag in llm_response:
                start = llm_response.find(lang_tag) + len(lang_tag)
            else:
                start = llm_response.find("```") + len("```")
            end = llm_response.find("```", start)
            if start >= 0 and end > start:
                return llm_response[start:end].strip()

        # If the response looks like JSON, try to extract refined_content
        try:
            import json

            brace_start = llm_response.find("{")
            brace_end = llm_response.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                obj = json.loads(llm_response[brace_start : brace_end + 1])
                if isinstance(obj, dict) and "refined_content" in obj:
                    return str(obj["refined_content"]).strip()
        except Exception:
            pass

        # Fallback: trimmed response
        return llm_response.strip()

    def _validate_refined_content(
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
                "diff_snippet": "N/A - None content",
            }

        # This check is needed for runtime safety even if MyPy thinks it's unreachable
        if not isinstance(refined_content, str):
            return {
                "is_valid": False,
                "reason": f"LLM returned non-string content: {type(refined_content)}",
                "status": "llm_invalid_output",
                "diff_snippet": f"N/A - {type(refined_content)} content",
            }

        # Check for empty or whitespace-only content - this is invalid output
        if self.reject_empty and not refined_content.strip():
            return {
                "is_valid": False,
                "reason": "LLM returned empty or whitespace-only content",
                "status": "llm_invalid_output",
                "diff_snippet": "N/A - empty content",
            }

        # Check for literal "None", "null" strings (case-insensitive) - this is invalid output
        if self.reject_literal_none:
            content_lower = refined_content.strip().lower()
            if content_lower in ("none", "null"):
                return {
                    "is_valid": False,
                    "reason": f"LLM returned literal '{refined_content.strip()}' content",
                    "status": "llm_invalid_output",
                    "diff_snippet": f"N/A - literal '{refined_content.strip()}'",
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
                    "diff_snippet": diff_snippet,
                }

        # Layer 2: Check if only whitespace/formatting differs
        if self._is_cosmetic_only_change(current_content, refined_content):
            # Get config for treating cosmetic as no-change
            treat_cosmetic_as_no_change = getattr(
                self.config, "treat_cosmetic_as_no_change", True
            )

            if self.reject_identical and treat_cosmetic_as_no_change:
                return {
                    "is_valid": False,
                    "reason": "LLM returned content with only cosmetic formatting changes",
                    "status": "content_cosmetic_noop",
                    "diff_snippet": diff_snippet,
                }

        # Layer 3: Optional AST comparison for Python tests
        allow_ast_check = getattr(self.config, "allow_ast_equivalence_check", True)
        if allow_ast_check and self._is_ast_equivalent(
            current_content, refined_content
        ):
            if self.reject_identical:
                return {
                    "is_valid": False,
                    "reason": "LLM returned semantically identical Python code (AST equivalent)",
                    "status": "content_semantically_identical",
                    "diff_snippet": diff_snippet,
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
                    "diff_snippet": diff_snippet,
                }
            except Exception as e:
                return {
                    "is_valid": False,
                    "reason": f"LLM returned unparseable Python: {e}",
                    "status": "syntax_error",
                    "diff_snippet": diff_snippet,
                }

        return {"is_valid": True, "status": "valid", "diff_snippet": diff_snippet}

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for comparison by handling newlines and trailing spaces.

        Args:
            content: Raw content string

        Returns:
            Normalized content string
        """
        # Normalize line endings to \n
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing spaces from each line but preserve structure
        lines = normalized.split("\n")
        normalized_lines = [line.rstrip() for line in lines]

        # Remove trailing empty lines
        while normalized_lines and not normalized_lines[-1]:
            normalized_lines.pop()

        return "\n".join(normalized_lines)

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
        import re

        # Remove all whitespace except in strings
        original_clean = re.sub(r"\s+", " ", original.strip())
        refined_clean = re.sub(r"\s+", " ", refined.strip())

        if original_clean == refined_clean:
            return True

        # Check if only indentation changes
        original_lines = original.strip().split("\n")
        refined_lines = refined.strip().split("\n")

        if len(original_lines) != len(refined_lines):
            return False

        # Compare lines after stripping leading/trailing whitespace
        for orig_line, ref_line in zip(original_lines, refined_lines, strict=False):
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

    def _compute_diff_snippet(
        self, original: str, refined: str, max_hunks: int = 3
    ) -> str:
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
            import difflib

            original_lines = original.splitlines(keepends=True)
            refined_lines = refined.splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    original_lines,
                    refined_lines,
                    fromfile="original",
                    tofile="refined",
                    n=10,  # Context lines
                )
            )

            if not diff_lines:
                return "No differences"

            # Limit to first max_hunks hunks
            limited_diff = []
            hunk_count = 0

            for line in diff_lines:
                limited_diff.append(line)
                if line.startswith("@@"):
                    hunk_count += 1
                    if hunk_count > max_hunks:
                        limited_diff.append("... (diff truncated)\n")
                        break

            return "".join(limited_diff)

        except Exception as e:
            return f"Diff computation failed: {e}"

    def _write_refined_content_safely(
        self, test_file: Path, refined_content: str
    ) -> dict[str, Any]:
        """
        Write refined content safely using WriterPort or fallback safety checks.

        Args:
            test_file: Path to test file to update
            refined_content: New content to write

        Returns:
            Dictionary with write result:
                - success: bool
                - error: str (if not successful)
                - backup_path: str (if backup created)
        """
        # Validate path safety (must be under tests/ directory)
        if not self._validate_test_path_safety(test_file):
            return {
                "success": False,
                "error": f"Path validation failed: {test_file} is not a valid test file path",
            }

        # Use WriterPort if available
        if self.writer_port:
            try:
                result = self.writer_port.write_test_file(
                    test_path=test_file, test_content=refined_content, overwrite=True
                )
                return {
                    "success": result.get("success", False),
                    "error": result.get("error")
                    if not result.get("success", False)
                    else None,
                    "backup_path": result.get("backup_path"),
                }
            except Exception as e:
                return {"success": False, "error": f"WriterPort failed: {e}"}

        # Fallback to local safety checks
        return self._write_with_local_safety(test_file, refined_content)

    def _validate_test_path_safety(self, test_file: Path) -> bool:
        """
        Validate that the test file path is safe for refinement writes.

        Args:
            test_file: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            resolved_path = test_file.resolve()
            path_str = str(resolved_path)

            # Must be a Python file
            if not path_str.endswith(".py"):
                logger.warning(
                    "Refinement path validation failed: not a Python file: %s", path_str
                )
                return False

            # Must contain 'test' in the path (either 'tests/' directory or 'test_' filename)
            if not (
                "tests" in resolved_path.parts or test_file.name.startswith("test_")
            ):
                logger.warning(
                    "Refinement path validation failed: not a test file path: %s",
                    path_str,
                )
                return False

            return True

        except Exception as e:
            logger.warning("Refinement path validation error for %s: %s", test_file, e)
            return False

    def _write_with_local_safety(
        self, test_file: Path, refined_content: str
    ) -> dict[str, Any]:
        """
        Write content with local safety checks (backup/rollback).

        Args:
            test_file: Path to test file to update
            refined_content: New content to write

        Returns:
            Dictionary with write result
        """
        backup_path = test_file.with_suffix(test_file.suffix + ".refine_backup")

        try:
            # Create backup
            if test_file.exists():
                backup_content = test_file.read_text(encoding="utf-8")
                backup_path.write_text(backup_content, encoding="utf-8")

            # Validate syntax before write if enabled
            if self.validate_syntax:
                try:
                    ast.parse(refined_content)
                except SyntaxError as e:
                    return {
                        "success": False,
                        "error": f"Content failed syntax validation before write: {e}",
                    }

            # Write refined content
            test_file.write_text(refined_content, encoding="utf-8")

            # Verify syntax after write
            if self.validate_syntax:
                try:
                    written_content = test_file.read_text(encoding="utf-8")
                    ast.parse(written_content)
                except Exception as e:
                    # Rollback on syntax failure
                    if backup_path.exists():
                        test_file.write_text(
                            backup_path.read_text(encoding="utf-8"), encoding="utf-8"
                        )
                    return {
                        "success": False,
                        "error": f"Content failed syntax validation after write, rolled back: {e}",
                    }

            # Clean up backup on success
            if backup_path.exists():
                backup_path.unlink()

            return {"success": True}

        except Exception as e:
            # Attempt rollback on any failure
            if backup_path.exists() and test_file.exists():
                try:
                    test_file.write_text(
                        backup_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                except Exception as rollback_error:
                    logger.error(
                        "Failed to rollback after write error: %s", rollback_error
                    )

            return {"success": False, "error": f"Write operation failed: {e}"}
        finally:
            # Clean up backup file in all cases
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception:
                    pass  # Best effort cleanup

    def _apply_refinement_safely(self, test_file: Path, refined_content: str) -> None:
        """
        Apply refined content to test file safely.

        Args:
            test_file: Path to test file to update
            refined_content: New content to write

        Raises:
            Exception: If write operation fails
        """
        # Create backup
        backup_path = test_file.with_suffix(test_file.suffix + ".bak")
        backup_path.write_text(test_file.read_text(encoding="utf-8"), encoding="utf-8")

        try:
            # Write refined content
            test_file.write_text(refined_content, encoding="utf-8")
        except Exception:
            # Restore from backup on failure
            test_file.write_text(
                backup_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            raise
        finally:
            # Clean up backup
            if backup_path.exists():
                backup_path.unlink()

    def _run_pytest_verification(self, test_file: Path) -> dict[str, Any]:
        """
        Run pytest on the refined test file to verify fixes with reliable import paths.

        Args:
            test_file: Path to test file to run

        Returns:
            Dictionary with success status and output
        """
        try:
            # Prepare environment with comprehensive PYTHONPATH
            env = self._prepare_test_environment(test_file)

            stdout, stderr, return_code = run_subprocess_simple(
                ["python", "-m", "pytest", str(test_file), "-v"],
                timeout=60,
                raise_on_error=False,
                env=env,  # Pass enhanced environment
            )

            output = ""
            if stdout:
                output += stdout
            if stderr:
                output += stderr

            return {
                "success": return_code == 0,
                "output": output,
                "return_code": return_code,
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Pytest execution failed: {e}",
                "return_code": -1,
            }

    def _prepare_test_environment(self, test_file: Path) -> dict[str, str]:
        """Prepare environment with reliable PYTHONPATH for test execution."""
        import os

        env = os.environ.copy()

        # Build comprehensive PYTHONPATH for reliable imports
        python_paths = []

        # Add directory containing the test file
        test_dir = test_file.parent
        python_paths.append(str(test_dir))

        # Find project root and add it + src/ to PYTHONPATH
        try:
            # Look for project markers to find root
            project_root = test_dir
            while project_root != project_root.parent:
                # Check for common project markers
                markers = [
                    "pyproject.toml",
                    "setup.py",
                    "setup.cfg",
                    ".git",
                    "requirements.txt",
                    "Pipfile",
                    "uv.lock",
                ]

                if any((project_root / marker).exists() for marker in markers):
                    break
                project_root = project_root.parent

            # Add project root to PYTHONPATH if different from test_dir
            if project_root != test_dir:
                python_paths.append(str(project_root))

            # Add src/ directory if it exists
            src_path = project_root / "src"
            if src_path.exists() and src_path.is_dir():
                python_paths.append(str(src_path))

        except Exception as e:
            # Log but don't fail - fallback to just test_dir
            logger.debug("Could not detect project root for test PYTHONPATH: %s", e)

        # Combine with existing PYTHONPATH
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        return env

    # Implement other required methods from RefinePort
    def refine(
        self,
        test_files: list[str | Path],
        source_files: list[str | Path] | None = None,
        refinement_goals: list[str] | None = None,
        **kwargs: Any,
    ) -> RefineOutcome:
        """Basic refine method - delegates to more specific methods."""
        # This is a placeholder implementation
        # In a full implementation, this would coordinate multiple refinement operations
        return RefineOutcome(
            updated_files=[str(f) for f in test_files],
            rationale="General refinement not yet implemented",
            plan="Use refine_from_failures for pytest-based refinement",
        )

    def analyze_test_quality(
        self,
        test_file: str | Path,
        source_file: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze test quality - placeholder implementation."""
        return {
            "quality_score": 0.5,
            "coverage_score": 0.5,
            "maintainability_score": 0.5,
            "issues": ["Not implemented yet"],
            "recommendations": ["Use refine_from_failures for specific improvements"],
        }

    def suggest_improvements(
        self,
        test_file: str | Path,
        improvement_type: str = "comprehensive",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Suggest improvements - placeholder implementation."""
        return {
            "suggestions": ["Not implemented yet"],
            "priority": ["low"],
            "estimated_effort": ["unknown"],
            "expected_benefit": ["unknown"],
        }

    def optimize_test_structure(
        self,
        test_file: str | Path,
        optimization_goals: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize test structure - placeholder implementation."""
        return {
            "optimized_structure": "Not implemented yet",
            "changes_needed": ["Not implemented yet"],
            "benefits": ["Not implemented yet"],
            "migration_plan": "Not implemented yet",
        }

    def enhance_test_coverage(
        self,
        test_file: str | Path,
        source_file: str | Path,
        coverage_gaps: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Enhance test coverage - placeholder implementation."""
        return {
            "new_tests": ["Not implemented yet"],
            "coverage_improvement": 0.0,
            "test_additions": ["Not implemented yet"],
            "coverage_analysis": "Not implemented yet",
        }
