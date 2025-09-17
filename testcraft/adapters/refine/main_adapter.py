"""
Main adapter for test refinement operations.

This module implements the RefinePort interface, providing functionality
for refining tests based on pytest failures and other quality issues.
"""

import logging
import time
from pathlib import Path
from typing import Any

from ...adapters.io.subprocess_safe import run_subprocess_simple
from ...application.generation.services.refinement import (
    ManualSuggestionsService,
    RefinementGuardrails,
    SafeApplyService,
)
from ...application.generation.services.refinement.refiner import PytestRefiner
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
    
    This is a thin orchestrator that delegates to specialized services:
    - RefinementGuardrails: Content validation and guardrails
    - SafeApplyService: Safe file writing with backup/rollback
    - ManualSuggestionsService: Manual fix suggestions and preflight analysis
    - PytestRefiner: Core refinement logic and LLM integration
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
        
        # Initialize specialized services
        guardrails_config = self.config.refinement_guardrails
        self.guardrails = RefinementGuardrails(
            reject_empty=guardrails_config.get("reject_empty", True),
            reject_literal_none=guardrails_config.get("reject_literal_none", True),
            reject_identical=guardrails_config.get("reject_identical", True),
            validate_syntax=guardrails_config.get("validate_syntax", True),
            treat_cosmetic_as_no_change=getattr(self.config, 'treat_cosmetic_as_no_change', True),
            allow_ast_equivalence_check=getattr(self.config, 'allow_ast_equivalence_check', True),
        )
        
        self.apply_service = SafeApplyService(
            writer_port=writer_port,
            validate_syntax=guardrails_config.get("validate_syntax", True),
        )
        
        self.suggestions_service = ManualSuggestionsService()

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
            return self._create_error_result(
                "file_not_found", f"Test file not found: {test_path}", 0
            )

        start_time = time.time()
        iteration = 0
        previous_content = None

        for iteration in range(1, max_iterations + 1):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                return self._create_error_result(
                    "timeout", f"Timeout after {timeout_seconds} seconds", iteration - 1
                )

            # Read current test content
            try:
                current_content = test_path.read_text(encoding="utf-8")
            except Exception as e:
                return self._create_error_result(
                    "read_error", f"Failed to read test file: {e}", iteration - 1
                )

            # Check for no-change condition
            if previous_content is not None and current_content == previous_content:
                return self._create_no_change_result(failure_output, current_content, iteration - 1)

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
                    test_path, current_content, failure_output, source_context, iteration, **kwargs
                )
                
                if not validation_result["is_valid"]:
                    return self._handle_validation_failure(validation_result, iteration)
                
            except Exception as e:
                return self._handle_llm_error(e, failure_output, current_content, iteration)

            # Apply changes safely
            try:
                write_result = self.apply_service.write_refined_content_safely(test_path, refined_content)
                if not write_result.get("success", False):
                    return self._handle_write_failure(write_result, validation_result, iteration)
                
                previous_content = current_content
                self._log_write_success(refined_content)
                        
            except Exception as e:
                return self._handle_write_exception(e, validation_result, current_content, iteration)

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

        # Max iterations reached
        return self._create_max_iterations_result(
            max_iterations, failure_output, previous_content
        )

    def manual_fix_suggestions(
        self,
        test_file: str | Path,
        failure_output: str,
        source_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Ask the LLM for targeted manual fix suggestions and a root-cause explanation.

        Returns a dict with keys: manual_suggestions, root_cause, active_import_path,
        preflight_suggestions, llm_confidence, improvement_areas.
        """
        test_path = Path(test_file)
        try:
            current_content = test_path.read_text(encoding="utf-8")
        except Exception:
            current_content = ""

        # Use suggestions service for analysis
        active_import_path = self.suggestions_service.select_active_import_path(
            failure_output, current_content
        )
        preflight_suggestions = self.suggestions_service.get_preflight_suggestions(current_content)
        formatted_source_context = self.suggestions_service.format_source_context(source_context)

        # Build prompts using the registry
        from ...prompts.registry import PromptRegistry
        prompt_registry = PromptRegistry()
        system_prompt = prompt_registry.get_system_prompt(
            "llm_manual_fix_suggestions"
        )
        user_prompt = prompt_registry.get_user_prompt(
            "llm_manual_fix_suggestions",
            code_content=current_content,
            failure_output=failure_output,
            active_import_path=active_import_path or "Not detected",
            preflight_suggestions=preflight_suggestions,
            source_context=formatted_source_context,
        )

        # Prefer calling a specialized method if the LLM adapter implements it
        llm_has_method = hasattr(self.llm, "manual_fix_suggestions")
        try:
            if llm_has_method:
                # type: ignore[attr-defined]
                result = self.llm.manual_fix_suggestions(  # noqa: E1101
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            else:
                # Fallback: use suggestions service
                result = self.suggestions_service.generate_manual_fix_suggestions_via_chat(
                    self.llm, system_prompt, user_prompt
                )
        except Exception as e:
            logger.warning("Manual fix suggestions failed: %s", e)
            result = {}

        # Ensure a consistent shape
        manual_suggestions = str(result.get("manual_suggestions", "")).strip()
        root_cause = str(result.get("root_cause", "")).strip()
        llm_confidence = result.get("llm_confidence")
        improvement_areas = result.get("improvement_areas", [])

        payload = {
            "manual_suggestions": manual_suggestions or preflight_suggestions,
            "root_cause": root_cause or "Root cause not identified by model",
            "active_import_path": active_import_path or "",
            "preflight_suggestions": preflight_suggestions,
            "llm_confidence": llm_confidence,
            "improvement_areas": improvement_areas or [],
        }

        return payload

    # Manual suggestions method now delegated to suggestions service

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

    
    # Preflight suggestions and context formatting now delegated to suggestions service

    def _get_llm_refinement_validated(
        self,
        test_path: Path,
        current_content: str,
        failure_output: str,
        source_context: dict[str, Any] | None,
        iteration: int,
        **kwargs: Any,
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
        # Build refinement payload
        payload = self._build_refinement_payload(
            test_path, current_content, failure_output, source_context, iteration, **kwargs
        )
        
        # Get context using suggestions service
        active_import_path = self.suggestions_service.select_active_import_path(
            failure_output, current_content
        )
        
        # Build preflight suggestions
        preflight_suggestions = self.suggestions_service.get_preflight_suggestions(current_content)
        
        # Prepare context for the prompt registry template
        prompt_context = {
            "code_content": current_content,
            "failure_output": failure_output,
            "active_import_path": active_import_path or "Not detected",
            "preflight_suggestions": preflight_suggestions,
            "source_context": self.suggestions_service.format_source_context(payload.get("source_context")),
            "version": "v1"
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
            "Yes" if preflight_suggestions.strip() else "None"
        )

        # Use prompt registry for system and user prompts
        from ...prompts.registry import PromptRegistry
        prompt_registry = PromptRegistry()
        
        system_prompt = prompt_registry.get_system_prompt("llm_content_refinement")
        user_prompt = prompt_registry.get_user_prompt(
            "llm_content_refinement",
            **prompt_context
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
            str(response)[:500] if response else "None"
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
            
            if suspected_bug and suspected_bug.strip().lower() not in ("null", "none", ""):
                logger.info(
                    "LLM detected suspected production bug for %s: %s",
                    payload.get("test_file_path", "unknown"),
                    suspected_bug
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
            refined_content.strip().lower() in ("none", "null") if refined_content else False,
            "Yes" if changes_made else "None",
            confidence if confidence is not None else "None"
        )
        
        # Validate the refined content
        validation_result = self._validate_refined_content(refined_content, current_content)
        
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
            validation_result.get("suspected_prod_bug", "None")
        )
        
        return refined_content, validation_result

    # Import path analysis methods now delegated to suggestions service


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
            return llm_response["refined_content"].strip()

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
        Validate refined content using the guardrails service.
        
        Args:
            refined_content: Content returned by LLM
            current_content: Current test file content for comparison
            
        Returns:
            Dictionary with validation result from guardrails service
        """
        return self.guardrails.validate_refined_content(refined_content, current_content)

    # Validation methods now delegated to guardrails service

    # Write methods now delegated to apply service

    def _run_pytest_verification(self, test_file: Path) -> dict[str, Any]:
        """
        Run pytest on the refined test file to verify fixes with reliable import paths.

        Args:
            test_file: Path to test file to run

        Returns:
            Dictionary with success status and output
        """
        try:
            # Use apply service for environment preparation
            env = self.apply_service.prepare_test_environment(test_file)
            
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

    # Environment preparation now delegated to apply service

    # Helper methods for result creation and error handling
    def _create_error_result(self, status: str, error: str, iterations: int) -> dict[str, Any]:
        """Create a standardized error result."""
        return {
            "success": False,
            "error": error,
            "iterations_used": iterations,
            "final_status": status,
            "fix_instructions": "",
            "active_import_path": "",
            "preflight_suggestions": "",
            "llm_confidence": None,
            "improvement_areas": [],
            "iteration": iterations,
        }

    def _create_no_change_result(self, failure_output: str, current_content: str, iterations: int) -> dict[str, Any]:
        """Create result for no-change condition."""
        active_import_path = self.suggestions_service.select_active_import_path(failure_output, current_content)
        preflight_suggestions = self.suggestions_service.get_preflight_suggestions(current_content)
        
        return {
            "success": False,
            "error": "No changes made in refinement iteration",
            "iterations_used": iterations,
            "final_status": "no_change",
            "fix_instructions": preflight_suggestions,
            "active_import_path": active_import_path or "",
            "preflight_suggestions": preflight_suggestions,
            "llm_confidence": None,
            "improvement_areas": [],
            "iteration": iterations + 1,
        }

    def _create_max_iterations_result(self, max_iterations: int, failure_output: str, previous_content: str | None) -> dict[str, Any]:
        """Create result for max iterations reached."""
        content = previous_content or ""
        active_import_path = self.suggestions_service.select_active_import_path(failure_output, content)
        preflight_suggestions = self.suggestions_service.get_preflight_suggestions(content)
        
        return {
            "success": False,
            "error": f"Max iterations ({max_iterations}) reached without success",
            "iterations_used": max_iterations,
            "final_status": "max_iterations",
            "fix_instructions": preflight_suggestions,
            "active_import_path": active_import_path or "",
            "preflight_suggestions": preflight_suggestions,
            "llm_confidence": None,
            "improvement_areas": [],
            "iteration": max_iterations,
        }

    def _handle_validation_failure(self, validation_result: dict[str, Any], iteration: int) -> dict[str, Any]:
        """Handle validation failure with telemetry."""
        if self.telemetry_port:
            with self.telemetry_port.create_child_span("refine_validation_failed") as span:
                span.set_attribute("validation_reason", validation_result["reason"])
                span.set_attribute("validation_status", validation_result.get("status", "unknown"))
                if validation_result.get("suspected_prod_bug"):
                    span.set_attribute("suspected_prod_bug", True)
        
        logger.debug(
            "LLM refinement validation failed: %s (status: %s)\nDiff snippet:\n%s", 
            validation_result["reason"], 
            validation_result.get("status", "unknown"),
            validation_result.get("diff_snippet", "No diff available")
        )
        
        # Handle suspected production bug
        if validation_result.get("suspected_prod_bug") and self.config.report_suspected_prod_bugs:
            logger.info("Production bug suspected: %s", validation_result["suspected_prod_bug"])
            return self._create_prod_bug_result(validation_result, iteration)
        
        # Map validation status to final_status
        status_mapping = {
            "content_identical": "llm_no_change",
            "content_cosmetic_noop": "llm_no_change", 
            "content_semantically_identical": "llm_no_change",
            "llm_invalid_output": "llm_invalid_output",
            "syntax_error": "syntax_error",
        }
        final_status = status_mapping.get(validation_result.get("status", "unknown"), "validation_failed")
        
        return {
            "success": False,
            "error": validation_result["reason"],
            "iterations_used": iteration,
            "final_status": final_status,
            "fix_instructions": validation_result.get("changes_made") or validation_result.get("preflight_suggestions", ""),
            "active_import_path": validation_result.get("active_import_path", ""),
            "preflight_suggestions": validation_result.get("preflight_suggestions", ""),
            "llm_confidence": validation_result.get("llm_confidence"),
            "improvement_areas": validation_result.get("improvement_areas", []),
            "iteration": iteration,
        }

    def _create_prod_bug_result(self, validation_result: dict[str, Any], iteration: int) -> dict[str, Any]:
        """Create result for suspected production bug."""
        return {
            "success": False,
            "error": "Suspected production bug detected",
            "iterations_used": iteration,
            "final_status": "prod_bug_suspected",
            "suspected_prod_bug": validation_result["suspected_prod_bug"],
            "fix_instructions": validation_result.get("changes_made", ""),
            "active_import_path": validation_result.get("active_import_path", ""),
            "preflight_suggestions": validation_result.get("preflight_suggestions", ""),
            "llm_confidence": validation_result.get("llm_confidence"),
            "improvement_areas": validation_result.get("improvement_areas", []),
            "iteration": iteration,
        }

    def _handle_llm_error(self, error: Exception, failure_output: str, current_content: str, iteration: int) -> dict[str, Any]:
        """Handle LLM error with telemetry."""
        if self.telemetry_port:
            with self.telemetry_port.create_child_span("refine_llm_error") as span:
                span.set_attribute("error", str(error))
        
        active_import_path = self.suggestions_service.select_active_import_path(failure_output, current_content)
        preflight_suggestions = self.suggestions_service.get_preflight_suggestions(current_content)
        
        return {
            "success": False,
            "error": f"LLM refinement failed: {error}",
            "iterations_used": iteration,
            "final_status": "llm_error",
            "fix_instructions": preflight_suggestions,
            "active_import_path": active_import_path or "",
            "preflight_suggestions": preflight_suggestions,
            "llm_confidence": None,
            "improvement_areas": [],
            "iteration": iteration,
        }

    def _handle_write_failure(self, write_result: dict[str, Any], validation_result: dict[str, Any], iteration: int) -> dict[str, Any]:
        """Handle write failure with telemetry."""
        if self.telemetry_port:
            with self.telemetry_port.create_child_span("refine_write_failed") as span:
                span.set_attribute("write_error", write_result.get("error", "Unknown"))
        
        return {
            "success": False,
            "error": f"Failed to apply refinement: {write_result.get('error', 'Unknown')}",
            "iterations_used": iteration,
            "final_status": "apply_error",
            "fix_instructions": validation_result.get("changes_made") or validation_result.get("preflight_suggestions", ""),
            "active_import_path": validation_result.get("active_import_path", ""),
            "preflight_suggestions": validation_result.get("preflight_suggestions", ""),
            "llm_confidence": validation_result.get("llm_confidence"),
            "improvement_areas": validation_result.get("improvement_areas", []),
            "iteration": iteration,
        }

    def _handle_write_exception(self, error: Exception, validation_result: dict[str, Any], current_content: str, iteration: int) -> dict[str, Any]:
        """Handle write exception with telemetry."""
        if self.telemetry_port:
            with self.telemetry_port.create_child_span("refine_write_exception") as span:
                span.set_attribute("exception", str(error))
        
        # Fallback context if validation_result not available
        fallback_suggestions = self.suggestions_service.get_preflight_suggestions(current_content)
        
        return {
            "success": False,
            "error": f"Failed to apply refinement: {error}",
            "iterations_used": iteration,
            "final_status": "apply_error",
            "fix_instructions": validation_result.get("changes_made", "") if validation_result else fallback_suggestions,
            "active_import_path": validation_result.get("active_import_path", "") if validation_result else "",
            "preflight_suggestions": validation_result.get("preflight_suggestions", "") if validation_result else fallback_suggestions,
            "llm_confidence": validation_result.get("llm_confidence") if validation_result else None,
            "improvement_areas": validation_result.get("improvement_areas", []) if validation_result else [],
            "iteration": iteration,
        }

    def _log_write_success(self, refined_content: str) -> None:
        """Log successful write with telemetry."""
        if self.telemetry_port:
            with self.telemetry_port.create_child_span("refine_write_success") as span:
                span.set_attribute("write_succeeded", True)
                span.set_attribute("refined_content_length", len(refined_content))

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
