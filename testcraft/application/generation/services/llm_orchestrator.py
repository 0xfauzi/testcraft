"""
LLM Orchestrator for PLAN/GENERATE/REFINE loops with symbol resolution.

This service implements the LLM orchestrator as specified in the context assembly
specification, including the missing_symbols resolution loop that fetches precise
definitions on demand during PLAN and REFINE stages.
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ....config.models import OrchestratorConfig
from ....domain.models import ContextPack, Conventions, ResolvedDef
from ....ports.llm_port import LLMPort
from ....ports.parser_port import ParserPort
from ....prompts.registry import PromptRegistry
from .context_assembler import ContextAssembler
from .context_pack import ContextPackBuilder
from .symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Base exception for LLM orchestrator errors."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        retry_count: int | None = None,
        cause: Exception | None = None,
        previous_errors: list[OrchestratorError] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.retry_count = retry_count
        self.cause = cause
        self.previous_errors = previous_errors or []

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.stage:
            parts.append(f"Stage: {self.stage}")
        if self.retry_count is not None:
            parts.append(f"Retry count: {self.retry_count}")
        if self.previous_errors:
            parts.append(f"Previous errors: {len(self.previous_errors)}")
        if self.cause:
            parts.append(f"Cause: {self.cause}")
        return " | ".join(parts)

    def add_previous_error(self, error: OrchestratorError) -> None:
        """Add a previous error to the error chain."""
        self.previous_errors.append(error)

    @property
    def all_errors(self) -> list[OrchestratorError]:
        """Get all errors in the chain, including this one."""
        return self.previous_errors + [self]


class PlanStageFailedException(OrchestratorError):
    """Exception raised when PLAN stage fails."""

    def __init__(
        self,
        message: str,
        retry_count: int | None = None,
        cause: Exception | None = None,
        previous_errors: list[OrchestratorError] | None = None,
    ):
        super().__init__(
            message,
            stage="PLAN",
            retry_count=retry_count,
            cause=cause,
            previous_errors=previous_errors,
        )


class GenerateStageFailedException(OrchestratorError):
    """Exception raised when GENERATE stage fails."""

    def __init__(
        self,
        message: str,
        retry_count: int | None = None,
        cause: Exception | None = None,
        previous_errors: list[OrchestratorError] | None = None,
    ):
        super().__init__(
            message,
            stage="GENERATE",
            retry_count=retry_count,
            cause=cause,
            previous_errors=previous_errors,
        )


class RefineStageFailedException(OrchestratorError):
    """Exception raised when REFINE stage fails."""

    def __init__(
        self,
        message: str,
        retry_count: int | None = None,
        cause: Exception | None = None,
        previous_errors: list[OrchestratorError] | None = None,
    ):
        super().__init__(
            message,
            stage="REFINE",
            retry_count=retry_count,
            cause=cause,
            previous_errors=previous_errors,
        )


class SymbolResolutionFailedException(OrchestratorError):
    """Exception raised when symbol resolution fails."""

    def __init__(
        self,
        message: str,
        missing_symbols: list[str] | None = None,
        retry_count: int | None = None,
        cause: Exception | None = None,
        previous_errors: list[OrchestratorError] | None = None,
    ):
        super().__init__(
            message,
            stage="SYMBOL_RESOLUTION",
            retry_count=retry_count,
            cause=cause,
            previous_errors=previous_errors,
        )
        self.missing_symbols = missing_symbols or []


class LLMOrchestrator:
    """
    LLM orchestrator implementing the 4-stage test generation pipeline with symbol resolution.

    Handles the complete test generation workflow including:
    - PLAN stage with missing_symbols resolution
    - GENERATE stage for test creation
    - REFINE stage for test repair with symbol resolution
    - MANUAL FIX stage for real product bugs (failing test + bug report)
    - Context re-packing and retry logic
    """

    def __init__(
        self,
        llm_port: LLMPort,
        parser_port: ParserPort,
        context_assembler: ContextAssembler,
        context_pack_builder: ContextPackBuilder | None = None,
        symbol_resolver: SymbolResolver | None = None,
        prompt_registry: PromptRegistry | None = None,
        config: OrchestratorConfig | None = None,
        max_plan_retries: int | None = None,
        max_refine_retries: int | None = None,
    ) -> None:
        """
        Initialize the LLM orchestrator.

        Args:
            llm_port: LLM port for generating and refining tests
            parser_port: Parser port for extracting symbol definitions
            context_assembler: Service for assembling context
            context_pack_builder: Builder for context packs (optional)
            symbol_resolver: Service for resolving missing symbols (optional)
            prompt_registry: Prompt registry for stage-specific prompts (optional)
            config: Orchestrator configuration (optional)
            max_plan_retries: Maximum retries for PLAN stage (overrides config if provided)
            max_refine_retries: Maximum retries for REFINE stage (overrides config if provided)
        """
        self._llm = llm_port
        self._parser = parser_port
        self._context_assembler = context_assembler
        self._context_pack_builder = context_pack_builder
        self._symbol_resolver = symbol_resolver or SymbolResolver(parser_port)
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._config = config or OrchestratorConfig()
        self._stdlib_modules = self._compute_stdlib_modules()
        self._stdlib_roots_lower = {
            module.split(".")[0].lower() for module in self._stdlib_modules if module
        }

        # Use config values or override with direct parameters
        self._max_plan_retries = (
            max_plan_retries
            if max_plan_retries is not None
            else self._config.max_plan_retries
        )
        self._max_refine_retries = (
            max_refine_retries
            if max_refine_retries is not None
            else self._config.max_refine_retries
        )

    def plan_and_generate(
        self,
        context_pack: ContextPack,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full PLAN/GENERATE workflow.

        Args:
            context_pack: Context pack for test generation
            project_root: Project root directory

        Returns:
            Dictionary containing generated tests and metadata
        """
        logger.info(
            "Starting PLAN/GENERATE workflow for %s", context_pack.target.object
        )

        # PLAN stage with symbol resolution
        try:
            plan = self._plan_stage(context_pack, project_root)
        except PlanStageFailedException as e:
            raise ValueError(f"PLAN stage failed: {e}") from e

        # GENERATE stage
        try:
            generated_code = self._generate_stage(context_pack, plan)
        except GenerateStageFailedException as e:
            raise ValueError(f"GENERATE stage failed: {e}") from e

        return {
            "plan": plan,
            "generated_code": generated_code,
            "context_pack": context_pack,
        }

    def plan_generate_refine(
        self,
        context_pack: ContextPack,
        project_root: Path | None = None,
        existing_code: str | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full 3-stage workflow: PLAN/GENERATE/REFINE.

        Args:
            context_pack: Context pack for test generation
            project_root: Project root directory
            existing_code: Existing test code to refine (if any)
            feedback: Execution feedback for refinement

        Returns:
            Dictionary containing plan, generated/refined code, and metadata
        """
        logger.info(
            "Starting PLAN/GENERATE/REFINE workflow for %s", context_pack.target.object
        )

        # PLAN stage with symbol resolution
        try:
            plan = self._plan_stage(context_pack, project_root)
        except PlanStageFailedException as e:
            raise ValueError(f"PLAN stage failed: {e}") from e

        # GENERATE stage (or refine if existing code provided)
        if existing_code and feedback:
            # REFINE stage
            try:
                refined_code = self._refine_stage(
                    context_pack, existing_code, feedback, project_root
                )
            except RefineStageFailedException as e:
                raise ValueError(f"REFINE stage failed: {e}") from e

            return {
                "plan": plan,
                "refined_code": refined_code,
                "context_pack": context_pack,
                "stage": "refine",
            }
        else:
            # GENERATE stage
            try:
                generated_code = self._generate_stage(context_pack, plan)
            except GenerateStageFailedException as e:
                raise ValueError(f"GENERATE stage failed: {e}") from e

            return {
                "plan": plan,
                "generated_code": generated_code,
                "context_pack": context_pack,
                "stage": "generate",
            }

    def manual_fix_stage(
        self,
        context_pack: ContextPack,
        feedback: dict[str, Any],
        project_root: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Execute the MANUAL FIX stage for suspected product bugs.

        Args:
            context_pack: Context pack for the failing code
            feedback: Execution feedback indicating the bug
            project_root: Project root directory

        Returns:
            Dictionary containing failing test and bug report, or None if not applicable
        """
        if not self._config.enable_manual_fix:
            logger.info(
                "MANUAL FIX stage disabled by configuration for %s",
                context_pack.target.object,
            )
            return None

        logger.info("Starting MANUAL FIX stage for %s", context_pack.target.object)

        try:
            # Get prompts from registry
            system_prompt = self._prompt_registry.get_system_prompt(
                "orchestrator_manual_fix"
            )

            # Prepare context for user prompt
            context = {
                "import_map": {
                    "target_import": self._resolve_target_import(context_pack),
                },
                "focal": {
                    "source": context_pack.focal.source,
                },
                "gwt_snippets": {
                    "then": context_pack.property_context.gwt_snippets.then,
                },
                "feedback": {
                    "trace_excerpt": feedback.get("trace_excerpt", ""),
                    "notes": feedback.get("notes", ""),
                },
                "conventions": self._format_conventions(context_pack.conventions),
            }

            # Get user prompt from registry
            user_prompt = self._prompt_registry.get_user_prompt(
                "orchestrator_manual_fix",
                additional_context=context,
                version=self._prompt_registry.version,
            )

            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Call LLM for manual fix guidance
            response = self._llm.generate_tests(code_content=full_prompt)
            response_text = self._extract_response_text(response)

            # Parse the response - it should contain both test and bug report
            # For now, return the response as-is
            # In a full implementation, we'd parse it into structured format
            return {
                "manual_fix_response": response_text,
                "context_pack": context_pack,
                "stage": "manual_fix",
            }

        except Exception as e:
            logger.exception("Error in MANUAL FIX stage: %s", e)
            return None

    def plan_stage(
        self, context_pack: ContextPack, project_root: Path | None = None
    ) -> dict[str, Any]:
        """
        Execute the PLAN stage with symbol resolution.

        Args:
            context_pack: Context pack for planning
            project_root: Project root directory

        Returns:
            Plan dictionary if successful

        Raises:
            ValueError: If PLAN stage fails
        """
        try:
            return self._plan_stage(context_pack, project_root)
        except PlanStageFailedException as e:
            raise ValueError(f"PLAN stage failed: {e}") from e

    def refine_stage(
        self,
        context_pack: ContextPack,
        existing_code: str,
        feedback: dict[str, Any],
        project_root: Path | None = None,
    ) -> str:
        """
        Execute the REFINE stage with symbol resolution.

        Args:
            context_pack: Context pack for refinement
            existing_code: Current test code to refine
            feedback: Execution feedback (traceback, coverage, etc.)
            project_root: Project root directory

        Returns:
            Refined code if successful

        Raises:
            ValueError: If REFINE stage fails
        """
        try:
            return self._refine_stage(
                context_pack, existing_code, feedback, project_root
            )
        except RefineStageFailedException as e:
            raise ValueError(f"REFINE stage failed: {e}") from e

    def _validate_plan_response(self, plan: Any, *, retry_count: int) -> dict[str, Any]:
        if not isinstance(plan, dict):
            logger.error(
                "PLAN response payload must be a JSON object, got %s",
                type(plan).__name__,
            )
            raise PlanStageFailedException(
                "PLAN response payload must be a JSON object",
                retry_count=retry_count,
            )

        error_value = plan.get("error")
        if isinstance(error_value, str):
            if error_value.strip():
                logger.error("PLAN response indicated error: %s", error_value)
                raise PlanStageFailedException(
                    f"PLAN stage refused to proceed: {error_value}",
                    retry_count=retry_count,
                )
        elif error_value is not None:
            logger.error(
                "PLAN response 'error' must be a string, got %s",
                type(error_value).__name__,
            )
            raise PlanStageFailedException(
                "PLAN response failed schema validation: 'error' must be a string",
                retry_count=retry_count,
            )

        plan_entries = plan.get("plan")
        if not isinstance(plan_entries, list):
            logger.error(
                "PLAN response 'plan' field must be a list, got %s",
                type(plan_entries).__name__,
            )
            raise PlanStageFailedException(
                "PLAN response failed schema validation: 'plan' field must be a list",
                retry_count=retry_count,
            )

        if plan_entries and not all(isinstance(entry, dict) for entry in plan_entries):
            logger.error("PLAN response 'plan' entries must be JSON objects")
            raise PlanStageFailedException(
                "PLAN response failed schema validation: plan entries must be objects",
                retry_count=retry_count,
            )

        if "missing_symbols" in plan:
            missing_symbols = plan["missing_symbols"]
            if not isinstance(missing_symbols, list) or not all(
                isinstance(symbol, str) for symbol in missing_symbols
            ):
                logger.error(
                    "PLAN response 'missing_symbols' must be a list of strings, got %r",
                    missing_symbols,
                )
                raise PlanStageFailedException(
                    "PLAN response failed schema validation: 'missing_symbols' must be a list of strings",
                    retry_count=retry_count,
                )

        if "metadata" in plan and not isinstance(plan["metadata"], dict):
            logger.error(
                "PLAN response 'metadata' must be a JSON object, got %s",
                type(plan["metadata"]).__name__,
            )
            raise PlanStageFailedException(
                "PLAN response failed schema validation: 'metadata' must be an object",
                retry_count=retry_count,
            )

        return plan

    def _validate_refine_response(
        self, response_data: Any, *, retry_count: int
    ) -> dict[str, Any]:
        if not isinstance(response_data, dict):
            logger.error(
                "REFINE response payload must be a JSON object, got %s",
                type(response_data).__name__,
            )
            raise RefineStageFailedException(
                "REFINE response payload must be a JSON object",
                retry_count=retry_count,
            )

        if "missing_symbols" in response_data:
            missing_symbols = response_data["missing_symbols"]
            if not isinstance(missing_symbols, list) or not all(
                isinstance(symbol, str) for symbol in missing_symbols
            ):
                logger.error(
                    "REFINE response 'missing_symbols' must be a list of strings, got %r",
                    missing_symbols,
                )
                raise RefineStageFailedException(
                    "REFINE response failed schema validation: 'missing_symbols' must be a list of strings",
                    retry_count=retry_count,
                )

        if "error" in response_data:
            error_value = response_data["error"]
            if isinstance(error_value, str):
                if error_value.strip():
                    logger.error("REFINE response indicated error: %s", error_value)
                    raise RefineStageFailedException(
                        f"REFINE stage refused to proceed: {error_value}",
                        retry_count=retry_count,
                    )
            elif error_value is not None:
                logger.error(
                    "REFINE response 'error' must be a string, got %s",
                    type(error_value).__name__,
                )
                raise RefineStageFailedException(
                    "REFINE response failed schema validation: 'error' must be a string",
                    retry_count=retry_count,
                )

        for key in ("refined_code", "tests"):
            value = response_data.get(key)
            if value is not None and not isinstance(value, str):
                logger.error(
                    "REFINE response '%s' must be a string, got %s",
                    key,
                    type(value).__name__,
                )
                raise RefineStageFailedException(
                    f"REFINE response failed schema validation: '{key}' must be a string",
                    retry_count=retry_count,
                )

        return response_data

    def _plan_stage(
        self, context_pack: ContextPack, project_root: Path | None = None
    ) -> dict[str, Any]:
        """
        Execute PLAN stage with symbol resolution loop.

        Args:
            context_pack: Context pack for planning
            project_root: Project root directory

        Returns:
            Plan dictionary if successful

        Raises:
            PlanStageFailedException: If PLAN stage fails after all retries
        """
        # Validate retry count parameters
        if self._max_plan_retries < 0:
            raise ValueError(
                f"Invalid max_plan_retries: {self._max_plan_retries}. Must be >= 0"
            )

        if context_pack.focal.is_placeholder:
            reason = context_pack.focal.placeholder_reason or "unspecified"
            raise PlanStageFailedException(
                f"Focal source is a placeholder: {reason}",
                retry_count=0,
            )

        current_context = context_pack
        retry_count = 0
        previous_errors: list[PlanStageFailedException] = []
        unresolved_symbols_union: set[str] = set()

        while retry_count < self._max_plan_retries:
            try:
                logger.info("Executing PLAN stage (attempt %d)", retry_count + 1)

                # Create PLAN prompts (system + user)
                system_prompt, user_prompt = self._build_plan_prompts(current_context)

                # Call LLM for planning
                response = self._llm.generate_tests(
                    code_content="",
                    custom_system_prompt=system_prompt,
                    custom_user_prompt=user_prompt,
                    return_raw=True,
                )
                logger.debug("PLAN stage response payload: %r", response)
                if isinstance(response, dict):
                    response_text = response.get("raw") or response.get("tests") or ""
                else:
                    response_text = str(response)

                logger.debug("PLAN stage raw response: %r", response_text)

                # Parse response as JSON
                try:
                    plan = json.loads(response_text)
                    plan = self._validate_plan_response(plan, retry_count=retry_count)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse PLAN response as JSON: %s", e)
                    raise PlanStageFailedException(
                        "PLAN response contained malformed JSON",
                        retry_count=retry_count,
                        cause=e,
                    ) from e

                # Check for missing symbols
                missing_symbols = plan.get("missing_symbols", [])
                if not missing_symbols:
                    # No missing symbols - return the plan
                    return plan

                dependency_metadata = current_context.dependency_metadata
                internal_roots: set[str] = set()

                target_module_path = current_context.target.module_path or ""
                if target_module_path:
                    internal_roots.add(target_module_path.split(".")[0].lower())

                target_import_stmt = (
                    current_context.import_map.target_import
                    if current_context.import_map
                    else ""
                )
                if target_import_stmt:
                    try:
                        import_target = target_import_stmt.replace(
                            "import", "", 1
                        ).strip()
                        import_target = import_target.split("as")[0].strip()
                        if import_target:
                            internal_roots.add(import_target.split(".")[0].lower())
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.debug(
                            "Unable to parse target import %s: %s",
                            target_import_stmt,
                            exc,
                        )

                for resolved_def in current_context.resolved_defs or []:
                    if resolved_def and resolved_def.name:
                        internal_roots.add(resolved_def.name.split(".")[0].lower())

                external_roots = {
                    (module or "").split(".")[0].lower()
                    for module in dependency_metadata.external_modules
                    if module
                }
                stdlib_roots = self._stdlib_roots_lower

                filtered_missing_symbols: list[str] = []
                skipped_symbols: list[str] = []

                for symbol in missing_symbols:
                    root = (symbol or "").split(".")[0].lower()
                    if (
                        root
                        and root not in internal_roots
                        and (root in external_roots or root in stdlib_roots)
                    ):
                        skipped_symbols.append(symbol)
                        continue
                    filtered_missing_symbols.append(symbol)

                if skipped_symbols:
                    logger.debug(
                        "Skipping missing_symbols treated as external/stdlib: %s",
                        skipped_symbols,
                    )

                if not filtered_missing_symbols:
                    plan["missing_symbols"] = []
                    return plan

                logger.info(
                    "Found %d missing symbols in PLAN response",
                    len(filtered_missing_symbols),
                )

                plan["missing_symbols"] = filtered_missing_symbols

                self._symbol_resolver.update_environment_modules(
                    external_modules=set(dependency_metadata.external_modules),
                    stdlib_modules=set(self._stdlib_modules),
                )

                # Resolve missing symbols
                resolved_defs = self._symbol_resolver.resolve_symbols(
                    filtered_missing_symbols, project_root
                )

                if not resolved_defs:
                    unresolved_symbols_union.update(filtered_missing_symbols)
                    logger.warning(
                        "Could not resolve missing symbols on attempt %d",
                        retry_count + 1,
                    )
                    if retry_count + 1 >= self._max_plan_retries:
                        self._report_unresolved_symbols(
                            unresolved_symbols_union, project_root
                        )
                        plan["missing_symbols"] = []
                        metadata = plan.setdefault("metadata", {})
                        metadata.setdefault("symbol_resolution", {})[
                            "unresolved_symbols"
                        ] = sorted(unresolved_symbols_union)
                        return plan
                    # Continue retrying in case the next LLM call doesn't have missing symbols
                    retry_count += 1
                    continue

                # Add resolved definitions to context pack
                updated_resolved_defs = (
                    list(current_context.resolved_defs) + resolved_defs
                )
                current_context = self._update_context_pack_resolved_defs(
                    current_context, updated_resolved_defs
                )

                unresolved_symbols_union.clear()
                retry_count += 1
                logger.info(
                    "Retrying PLAN stage with resolved symbols (attempt %d)",
                    retry_count + 1,
                )

            except PlanStageFailedException:
                # Re-raise our own exceptions to collect them
                raise
            except Exception as e:
                logger.exception("Error in PLAN stage: %s", e)
                error = PlanStageFailedException(
                    "Unexpected error in PLAN stage", retry_count=retry_count, cause=e
                )
                previous_errors.append(error)
                retry_count += 1

        logger.warning("PLAN stage exceeded maximum retries")
        final_error = PlanStageFailedException(
            f"PLAN stage exceeded maximum retries ({self._max_plan_retries})",
            retry_count=retry_count,
        )
        # Add all previous errors to the final error
        for error in previous_errors:
            final_error.add_previous_error(error)
        raise final_error

    def _generate_stage(self, context_pack: ContextPack, plan: dict[str, Any]) -> str:
        """
        Execute GENERATE stage.

        Args:
            context_pack: Context pack for generation
            plan: Approved plan from PLAN stage

        Returns:
            Generated test code if successful

        Raises:
            GenerateStageFailedException: If generation fails
        """
        try:
            logger.info("Executing GENERATE stage")

            # Create GENERATE prompt
            generate_prompt = self._create_generate_prompt(context_pack, plan)

            # Call LLM for generation
            response = self._llm.generate_tests(code_content=generate_prompt)
            response_text = self._extract_response_text(response)

            # Extract code from response
            generated_code = self._extract_code_from_response(response_text)
            if not generated_code:
                logger.warning("No code found in GENERATE response")
                raise GenerateStageFailedException("No code found in GENERATE response")

            return generated_code

        except GenerateStageFailedException:
            raise  # Re-raise our own exception
        except Exception as e:
            logger.exception("Error in GENERATE stage: %s", e)
            raise GenerateStageFailedException(
                "Unexpected error in GENERATE stage", cause=e
            ) from e

    def _refine_stage(
        self,
        context_pack: ContextPack,
        existing_code: str,
        feedback: dict[str, Any],
        project_root: Path | None = None,
    ) -> str:
        """
        Execute REFINE stage with symbol resolution loop.

        Args:
            context_pack: Context pack for refinement
            existing_code: Current test code to refine
            feedback: Execution feedback
            project_root: Project root directory

        Returns:
            Refined code if successful

        Raises:
            RefineStageFailedException: If REFINE stage fails after all retries
        """
        # Validate retry count parameters
        if self._max_refine_retries < 0:
            raise ValueError(
                f"Invalid max_refine_retries: {self._max_refine_retries}. Must be >= 0"
            )

        current_context = context_pack
        retry_count = 0
        previous_errors: list[RefineStageFailedException] = []

        while retry_count < self._max_refine_retries:
            try:
                logger.info("Executing REFINE stage (attempt %d)", retry_count + 1)

                # Create REFINE prompt
                refine_prompt = self._create_refine_prompt(
                    current_context, existing_code, feedback
                )

                # Call LLM for refinement
                response = self._llm.generate_tests(code_content=refine_prompt)

                # Parse response as dict/JSON or extract code
                if isinstance(response, dict):
                    response_data = self._validate_refine_response(
                        response, retry_count=retry_count
                    )
                else:
                    response_text = self._extract_response_text(response)
                    if response_text.strip().startswith("{"):
                        try:
                            response_data = json.loads(response_text)
                            response_data = self._validate_refine_response(
                                response_data, retry_count=retry_count
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                "Failed to parse REFINE response as JSON: %s", e
                            )
                            raise RefineStageFailedException(
                                "REFINE response contained malformed JSON",
                                retry_count=retry_count,
                                cause=e,
                            ) from e
                    else:
                        response_data = None

                if response_data is not None:
                    try:
                        if "missing_symbols" in response_data:
                            # Handle missing symbols
                            missing_symbols = response_data["missing_symbols"]
                            if missing_symbols:
                                logger.info(
                                    "Found %d missing symbols in REFINE response",
                                    len(missing_symbols),
                                )

                                # Resolve missing symbols
                                resolved_defs = self._symbol_resolver.resolve_symbols(
                                    missing_symbols, project_root
                                )

                                if resolved_defs:
                                    # Add resolved definitions to context pack
                                    updated_resolved_defs = (
                                        list(current_context.resolved_defs)
                                        + resolved_defs
                                    )
                                    current_context = (
                                        self._update_context_pack_resolved_defs(
                                            current_context, updated_resolved_defs
                                        )
                                    )

                                    retry_count += 1
                                    continue  # Retry with resolved symbols

                        # If no symbols to resolve, extract code from response
                        refined_payload = response_data.get("refined_code")
                        if not (
                            isinstance(refined_payload, str) and refined_payload.strip()
                        ):
                            tests_payload = response_data.get("tests")
                            if isinstance(tests_payload, str) and tests_payload.strip():
                                refined_payload = tests_payload

                        if not (
                            isinstance(refined_payload, str) and refined_payload.strip()
                        ):
                            logger.error(
                                "REFINE response did not include usable code payload"
                            )
                            raise RefineStageFailedException(
                                "REFINE response missing 'refined_code' or 'tests' content",
                                retry_count=retry_count,
                            )

                        refined_code = refined_payload
                    except RefineStageFailedException:
                        raise
                    except Exception as exc:
                        refined_payload = response_data.get(
                            "refined_code"
                        ) or response_data.get("tests")
                        if not (
                            isinstance(refined_payload, str)
                            and refined_payload
                            and refined_payload.strip()
                        ):
                            logger.error(
                                "REFINE response did not include usable code payload"
                            )
                            raise RefineStageFailedException(
                                "REFINE response missing 'refined_code' or 'tests' content",
                                retry_count=retry_count,
                            ) from exc
                        refined_code = refined_payload
                else:
                    # Treat as code text
                    refined_code = self._extract_response_text(response)

                # Extract code from response
                refined_code = self._extract_code_from_response(refined_code)
                if not refined_code:
                    logger.warning("No code found in REFINE response")
                    error = RefineStageFailedException(
                        "No code found in REFINE response", retry_count=retry_count
                    )
                    previous_errors.append(error)
                    retry_count += 1
                    continue

                return refined_code

            except RefineStageFailedException:
                # Re-raise our own exceptions to collect them
                raise
            except Exception as e:
                logger.exception("Error in REFINE stage: %s", e)
                error = RefineStageFailedException(
                    "Unexpected error in REFINE stage", retry_count=retry_count, cause=e
                )
                previous_errors.append(error)
                retry_count += 1

        logger.warning("REFINE stage exceeded maximum retries")
        final_error = RefineStageFailedException(
            f"REFINE stage exceeded maximum retries ({self._max_refine_retries})",
            retry_count=retry_count,
        )
        # Add all previous errors to the final error
        for error in previous_errors:
            final_error.add_previous_error(error)
        raise final_error

    def _build_plan_prompts(self, context_pack: ContextPack) -> tuple[str, str]:
        """Create PLAN stage system and user prompts using the registry."""
        system_prompt = self._prompt_registry.get_system_prompt("orchestrator_plan")

        context = {
            "target": {
                "module_file": str(context_pack.target.module_file),
                "object": context_pack.target.object,
            },
            "import_map": {
                "target_import": self._resolve_target_import(context_pack),
            },
            "focal": {
                "source": context_pack.focal.source,
                "signature": context_pack.focal.signature,
                "docstring": context_pack.focal.docstring or "",
                "is_placeholder": context_pack.focal.is_placeholder,
                "placeholder_reason": context_pack.focal.placeholder_reason or "",
            },
            "resolved_defs_compact": self._format_resolved_defs(
                context_pack.resolved_defs
            ),
            "gwt_snippets": context_pack.property_context.gwt_snippets.__dict__,
            "conventions": self._format_conventions(context_pack.conventions),
            "dependency_metadata": self._format_dependency_metadata(
                context_pack.dependency_metadata
            ),
        }

        user_prompt = self._prompt_registry.get_user_prompt(
            "orchestrator_plan",
            additional_context=context,
            version=self._prompt_registry.version,
        )

        return system_prompt, user_prompt

    def _report_unresolved_symbols(
        self, symbols: set[str], project_root: Path | None
    ) -> None:
        """Log aggregated unresolved symbols with actionable guidance."""

        if not symbols:
            return

        sorted_symbols = sorted(symbols)
        logger.warning(
            "Symbol resolution could not resolve %d symbol(s): %s",
            len(sorted_symbols),
            ", ".join(sorted_symbols),
        )

        hints = [
            "Ensure the listed modules live under configured source roots (e.g., src/ or tests/src).",
            "Add __init__.py files for packages or include namespace directories in your TestCraft configuration.",
        ]

        if project_root is not None:
            try:
                hints.append(
                    f"Project root used during resolution: {project_root.resolve()}"
                )
            except Exception:  # pragma: no cover - defensive logging
                hints.append(
                    "Project root used during resolution could not be resolved."
                )

        for hint in hints:
            logger.info("Symbol resolution hint: %s", hint)

    def _create_generate_prompt(
        self, context_pack: ContextPack, plan: dict[str, Any]
    ) -> str:
        """Create GENERATE stage prompt using PromptRegistry."""
        # Get system prompt from registry
        system_prompt = self._prompt_registry.get_system_prompt("orchestrator_generate")

        # Prepare context for user prompt
        context = {
            "import_map": {
                "target_import": self._resolve_target_import(context_pack),
            },
            "focal": {
                "source": context_pack.focal.source,
                "is_placeholder": context_pack.focal.is_placeholder,
                "placeholder_reason": context_pack.focal.placeholder_reason or "",
            },
            "resolved_defs_compact": self._format_resolved_defs(
                context_pack.resolved_defs
            ),
            "property_context_compact": self._format_property_context(
                context_pack.property_context
            ),
            "conventions": self._format_conventions(context_pack.conventions),
            "approved_plan_json": json.dumps(plan.get("plan", []), indent=2),
            "dependency_metadata": self._format_dependency_metadata(
                context_pack.dependency_metadata
            ),
        }

        # Get user prompt from registry
        user_prompt = self._prompt_registry.get_user_prompt(
            "orchestrator_generate",
            additional_context=context,
            version=self._prompt_registry.version,
        )

        return f"{system_prompt}\n\n{user_prompt}"

    def _create_refine_prompt(
        self, context_pack: ContextPack, existing_code: str, feedback: dict[str, Any]
    ) -> str:
        """Create REFINE stage prompt using PromptRegistry."""
        # Get system prompt from registry
        system_prompt = self._prompt_registry.get_system_prompt("orchestrator_refine")

        # Prepare context for user prompt
        context = {
            "import_map": {
                "target_import": self._resolve_target_import(context_pack),
            },
            "focal": {
                "source": context_pack.focal.source,
                "is_placeholder": context_pack.focal.is_placeholder,
                "placeholder_reason": context_pack.focal.placeholder_reason or "",
            },
            "current_tests": self._extract_failing_parts(existing_code, feedback),
            "feedback": {
                "result": feedback.get("result", "unknown"),
                "trace_excerpt": feedback.get("trace_excerpt", ""),
                "coverage_gaps": feedback.get("coverage", {}),
                "notes": feedback.get("notes", ""),
            },
        }

        # Get user prompt from registry
        user_prompt = self._prompt_registry.get_user_prompt(
            "orchestrator_refine",
            additional_context=context,
            version=self._prompt_registry.version,
        )

        return f"{system_prompt}\n\n{user_prompt}"

    def _update_context_pack_resolved_defs(
        self, context_pack: ContextPack, new_resolved_defs: list[ResolvedDef]
    ) -> ContextPack:
        """Update context pack with new resolved definitions."""
        # Since ContextPack is immutable, we need to create a new one
        # This is a simplified version - in reality we'd need to handle all fields
        return ContextPack(
            target=context_pack.target,
            import_map=context_pack.import_map,
            focal=context_pack.focal,
            resolved_defs=new_resolved_defs,
            property_context=context_pack.property_context,
            conventions=context_pack.conventions,
            budget=context_pack.budget,
            context=context_pack.context,
            dependency_metadata=context_pack.dependency_metadata,
        )

    def _resolve_target_import(self, context_pack: ContextPack) -> str:
        """Safely resolve the canonical import for prompt templating."""
        import_map = context_pack.import_map
        if import_map is None:
            return "[import unavailable]"
        target_import = getattr(import_map, "target_import", None)
        if target_import:
            return str(target_import)
        return "[import unavailable]"

    def _format_resolved_defs(self, resolved_defs: list[ResolvedDef]) -> str:
        """Format resolved definitions for prompt."""
        if not resolved_defs:
            return "None"

        formatted = []
        for defn in resolved_defs:
            formatted.append(f"- {defn.name} ({defn.kind}): {defn.signature}")
            if defn.doc:
                formatted.append(f"  {defn.doc}")

        return "\n".join(formatted)

    def _format_gwt_snippets(self, snippets: list[str]) -> str:
        """Format GWT snippets for prompt."""
        if not snippets:
            return "None"

        return "\n".join(f"- {snippet}" for snippet in snippets)

    def _format_conventions(self, conventions: Conventions) -> str:
        """Format conventions for prompt."""
        return f"""
- Test style: {conventions.test_style}
- Allowed libs: {", ".join(conventions.allowed_libs)}
- Determinism: seed={conventions.determinism.seed}, tz={conventions.determinism.tz}, freeze_time={conventions.determinism.freeze_time}
- IO policy: network={conventions.io_policy.network}, fs={conventions.io_policy.fs}
""".strip()

    def _format_property_context(self, property_context) -> str:
        """Format property context for prompt."""
        context_parts = []

        if property_context.ranked_methods:
            context_parts.append("Ranked methods:")
            for method in property_context.ranked_methods:
                context_parts.append(
                    f"  - {method.qualname} ({method.level}, {method.relation})"
                )

        if (
            property_context.gwt_snippets.given
            or property_context.gwt_snippets.when
            or property_context.gwt_snippets.then
        ):
            context_parts.append("GWT patterns:")
            if property_context.gwt_snippets.given:
                context_parts.append(
                    f"  GIVEN: {'; '.join(property_context.gwt_snippets.given)}"
                )
            if property_context.gwt_snippets.when:
                context_parts.append(
                    f"  WHEN: {'; '.join(property_context.gwt_snippets.when)}"
                )
            if property_context.gwt_snippets.then:
                context_parts.append(
                    f"  THEN: {'; '.join(property_context.gwt_snippets.then)}"
                )

        return "\n".join(context_parts) if context_parts else "None"

    def _format_dependency_metadata(self, metadata) -> dict[str, str]:
        """Format dependency metadata for inclusion in prompts."""

        if metadata is None:
            return {
                "external_modules": "",
                "distributions": "",
            }

        limit = 200

        external_modules = ", ".join(metadata.external_modules[:limit])
        distribution_items = list(metadata.distributions.items())[:limit]
        distributions = ", ".join(
            f"{name}{version}" if version else str(name)
            for name, version in distribution_items
        )

        return {
            "external_modules": external_modules,
            "distributions": distributions,
        }

    @staticmethod
    def _compute_stdlib_modules() -> set[str]:
        """Collect standard library module names for resolver filtering."""
        modules: set[str] = set(sys.builtin_module_names)
        stdlib_names = getattr(sys, "stdlib_module_names", None)
        if stdlib_names:
            try:
                modules.update(stdlib_names)
            except Exception:
                logger.debug("Unable to extend stdlib module names", exc_info=True)
        return {module for module in modules if module}

    def _extract_failing_parts(
        self, existing_code: str, feedback: dict[str, Any]
    ) -> str:
        """Extract failing parts from existing code and feedback."""
        # This is a simplified implementation
        # In reality, you'd use AST to extract specific failing lines
        return existing_code  # For now, return the full code

    def _extract_response_text(self, response) -> str:
        """Extract text from LLM response."""
        # Handle different response formats
        if isinstance(response, dict):
            # Common structured keys returned by adapters
            for key in (
                "tests",
                "plan",
                "content",
                "generated_code",
                "refined_code",
                "manual_fix_response",
            ):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            # Fallback to JSON string for logging/debugging
            try:
                return json.dumps(response)
            except TypeError:
                return str(response)
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _extract_code_from_response(self, response_text: str) -> str | None:
        """Extract code from LLM response."""
        # Look for code blocks
        import re

        # Try to find code between triple backticks
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        match = re.search(code_block_pattern, response_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no fenced block, see if the full response looks like Python code
        stripped = response_text.strip()
        if stripped:
            try:
                ast.parse(stripped)
                logger.debug(
                    "GENERATE response lacked fenced code block; using full response as code."
                )
                return stripped
            except SyntaxError:
                logger.debug(
                    "Full response is not valid Python; falling back to heuristic extraction."
                )

        # If no code blocks found, try to find the last Python code section
        lines = response_text.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                in_code = True
                code_lines.append(line)
            elif in_code:
                if line.strip() and not line.startswith(" "):
                    # Line doesn't start with space, might be end of code
                    break
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return None
