"""OpenAI adapter facade that orchestrates client, logging, and token policy."""

from __future__ import annotations

import logging
import time
from typing import Any

import openai

from ....config.credentials import CredentialManager
from ....ports.cost_port import CostPort
from ....ports.llm_error import LLMError
from ....ports.llm_port import LLMPort
from ....prompts.registry import PromptRegistry
from ..base import BaseLLMAdapter
from ..common import parse_json_response, with_retries
from ..pricing import calculate_cost
from .client import OpenAIClient, OpenAIError
from .logging_artifacts import OpenAILoggingArtifacts
from .token_policy import OpenAITokenPolicy

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter, LLMPort):
    """
    Production OpenAI adapter using the latest v1.106.1 SDK.

    Features:
    - Secure credential management via environment variables
    - Proper error handling and retries with exponential backoff
    - Support for latest OpenAI models (GPT-4o, GPT-4o-mini, o1-mini reasoning model)
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        timeout: float = 180.0,
        max_tokens: int | None = None,  # Will be calculated automatically
        temperature: float = 0.1,
        max_retries: int = 3,
        base_url: str | None = None,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI adapter.

        Args:
            model: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini", "o1-mini")
            timeout: Request timeout in seconds (default: 180s for test generation)
            max_tokens: Maximum tokens in response (auto-calculated if None)
            temperature: Response randomness (0.0-2.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            base_url: Custom API base URL (optional)
            credential_manager: Custom credential manager (optional)
            prompt_registry: Custom prompt registry (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional OpenAI client parameters
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize base adapter (prompts, creds, cost, token calculator)
        BaseLLMAdapter.__init__(
            self,
            provider="openai",
            model=model,
            prompt_registry=prompt_registry or PromptRegistry(),
            cost_port=cost_port,
            credential_manager=credential_manager,
        )

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Initialize components
        self._openai_client = OpenAIClient(
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
            credential_manager=self.credential_manager,
            **kwargs,
        )
        self.logging_artifacts = OpenAILoggingArtifacts()
        self.token_policy = OpenAITokenPolicy(model)
        
        # Expose the underlying client for backward compatibility with tests
        self._client = self._openai_client.client
        # Also expose as .client for test mocking
        self.client = self._openai_client

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate test cases for the provided code content."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        complexity_level = self.token_policy.estimate_complexity(code_content)
        budgets = self._calc_budgets(
            use_case="test_generation",
            input_text=code_content + (context or ""),
            complexity=complexity_level,
        )
        # Allow per-request override for max_tokens, ignore thinking_tokens (not applicable for OpenAI)
        max_tokens = int(kwargs.pop("max_tokens", budgets["max_tokens"]))
        # Remove thinking_tokens if provided to avoid duplicate kwargs downstream
        kwargs.pop("thinking_tokens", None)

        # Extract generation preferences if provided
        gen_prefs = {
            "include_docstrings": kwargs.pop("include_docstrings", True),
            "generate_fixtures": kwargs.pop("generate_fixtures", True),
            "parametrize_similar_tests": kwargs.pop("parametrize_similar_tests", True),
            "max_test_methods_per_class": kwargs.pop("max_test_methods_per_class", 20),
        }

        # Get prompts from registry via _prompts helper
        # Separate the context into repository context and enhanced context
        repository_context, enhanced_context = self._parse_context_sections(context or "")
        
        system_prompt, user_prompt = self._prompts(
            "llm_test_generation",
            code_content=code_content,
            repository_context=repository_context,
            enhanced_context=enhanced_context,
            generation_preferences=self._format_preferences(gen_prefs),
            test_framework=test_framework,
        )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                thinking_tokens=None,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={"reasoning": parsed.data.get("reasoning", "")},
                    raw_provider_fields={"finish_reason": result.get("finish_reason")},
                )
                return {
                    "tests": parsed.data.get("tests", content),
                    "coverage_focus": parsed.data.get("coverage_focus", []),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": metadata,
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"parse_error": parsed.error},
                    raw_provider_fields={"raw_content": content, "finish_reason": result.get("finish_reason")},
                )
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"OpenAI test generation failed: {e}")
            raise LLMError(
                message=f"Test generation failed: {e}",
                provider="openai",
                operation="generate_tests",
                model=self.model,
            ) from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        complexity_level = self.token_policy.estimate_complexity(code_content)
        budgets = self._calc_budgets(
            use_case="code_analysis",
            input_text=code_content,
            complexity=complexity_level,
        )
        # Allow per-request override for max_tokens; ignore thinking_tokens for OpenAI
        max_tokens = int(kwargs.pop("max_tokens", budgets["max_tokens"]))
        kwargs.pop("thinking_tokens", None)

        # Get prompts from registry via _prompts helper
        system_prompt, user_prompt = self._prompts(
            "llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                thinking_tokens=None,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={
                        "analysis_type": analysis_type,
                        "summary": parsed.data.get("analysis_summary", ""),
                    },
                )
                return {
                    "testability_score": parsed.data.get("testability_score", 5.0),
                    "complexity_metrics": parsed.data.get("complexity_metrics", {}),
                    "recommendations": parsed.data.get("recommendations", []),
                    "potential_issues": parsed.data.get("potential_issues", []),
                    "metadata": metadata,
                }
            else:
                # Fallback if JSON parsing fails
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={
                        "analysis_type": analysis_type,
                        "raw_content": content,
                        "parse_error": parsed.error,
                    },
                )
                return {
                    "testability_score": 5.0,
                    "complexity_metrics": {},
                    "recommendations": [],
                    "potential_issues": [],
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"OpenAI code analysis failed: {e}")
            raise LLMError(
                message=f"Code analysis failed: {e}",
                provider="openai",
                operation="analyze_code",
                model=self.model,
            ) from e

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        complexity_level = self.token_policy.estimate_complexity(original_content)
        budgets = self._calc_budgets(
            use_case="refinement",
            input_text=original_content + refinement_instructions,
            complexity=complexity_level,
        )
        # Allow per-request override for max_tokens; ignore thinking_tokens for OpenAI
        max_tokens = int(kwargs.pop("max_tokens", budgets["max_tokens"]))
        kwargs.pop("thinking_tokens", None)

        # Use pre-rendered prompts from the caller (if provided). Otherwise fall back to registry.
        if system_prompt is None:
            system_prompt, _ = self._prompts("llm_content_refinement")
        user_prompt = refinement_instructions

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                thinking_tokens=None,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse and normalize refinement response using base helper
            refined_data, parsed_info = self._parse_and_normalize_refinement(content)

            if refined_data:
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                )
                return {
                    "refined_content": refined_data["refined_content"],
                    "changes_made": refined_data["changes_made"],
                    "confidence": refined_data["confidence"],
                    "improvement_areas": refined_data["improvement_areas"],
                    "suspected_prod_bug": refined_data.get("suspected_prod_bug"),
                    "metadata": metadata,
                }
            else:
                # Schema validation failed
                error_msg = parsed_info.get("error", "Unknown parsing error")
                logger.error(f"OpenAI schema validation failed: {error_msg}")
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                    extra={"schema_error": error_msg},
                )
                return {
                    "refined_content": original_content,  # Safe fallback
                    "changes_made": f"Schema validation failed: {error_msg}",
                    "confidence": 0.0,
                    "improvement_areas": ["schema_error"],
                    "suspected_prod_bug": None,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"OpenAI content refinement failed: {e}")
            raise LLMError(
                message=f"Content refinement failed: {e}",
                provider="openai",
                operation="refine_content",
                model=self.model,
            ) from e

    def _parse_context_sections(self, context: str) -> tuple[str, str]:
        """Parse merged context into repository context and enhanced context sections."""
        if not context:
            return "", ""
        
        # Look for enhanced context markers that indicate module/planning info
        enhanced_markers = [
            "# Module Import Information",
            "# DETAILED_TEST_PLANS", 
            "Module Path:",
            "Import Hint:",
            "## Plan "
        ]
        
        # Try to split context at enhanced sections
        for marker in enhanced_markers:
            if marker in context:
                parts = context.split(marker, 1)
                repository_context = parts[0].strip()
                enhanced_context = (marker + parts[1]).strip() if len(parts) > 1 else ""
                return repository_context, enhanced_context
        
        # If no enhanced markers found, treat entire context as repository context
        return context, ""
    
    def _format_preferences(self, preferences: dict) -> str:
        """Format generation preferences for the template."""
        if not preferences:
            return "Default preferences"
        
        formatted = []
        for key, value in preferences.items():
            formatted.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a test plan for the provided code content."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        complexity_level = self.token_policy.estimate_complexity(code_content)
        budgets = self._calc_budgets(
            use_case="test_generation",  # Use test_generation for test planning budgets
            input_text=code_content + (context or ""),
            complexity=complexity_level,
        )
        max_tokens = budgets["max_tokens"]
        thinking_tokens = budgets["thinking_tokens"]

        # Add simple telemetry tracking for planning operations
        planning_start_time = time.time()

        # Get prompts from registry via _prompts helper
        additional_context = {"context": context} if context else {}
        system_prompt, user_prompt = self._prompts(
            "llm_test_planning_v1",
            code_content=code_content,
            additional_context=additional_context,
        )

        # If caller provided multiple elements, request per-element plans in one response
        elements = kwargs.pop("elements", None)
        if isinstance(elements, list) and elements:
            try:
                import json as _json
                elements_json = _json.dumps(elements, ensure_ascii=False)
            except Exception:
                elements_json = str(elements)
            user_prompt += (
                "\n\n"
                "You are planning tests for multiple elements in this file.\n"
                "Elements (name, type, optional line_range):\n" + elements_json + "\n\n"
                "Return valid JSON including the standard fields (plan_summary, detailed_plan, confidence, scenarios, mocks, fixtures, data_matrix, edge_cases, error_paths, dependencies, notes) "
                "AND an additional field 'element_plans' as an array, where each item has: {name, type, plan_summary, detailed_plan, confidence?, scenarios?}."
            )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                thinking_tokens=thinking_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                )
                response = {
                    "plan_summary": parsed.data.get("plan_summary", ""),
                    "detailed_plan": parsed.data.get("detailed_plan", ""),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "scenarios": parsed.data.get("scenarios", []),
                    "mocks": parsed.data.get("mocks", ""),
                    "fixtures": parsed.data.get("fixtures", ""),
                    "data_matrix": parsed.data.get("data_matrix", []),
                    "edge_cases": parsed.data.get("edge_cases", []),
                    "error_paths": parsed.data.get("error_paths", []),
                    "dependencies": parsed.data.get("dependencies", []),
                    "notes": parsed.data.get("notes", ""),
                    "metadata": metadata,
                }
                # Preserve per-element plans if provided
                if isinstance(parsed.data.get("element_plans"), list):
                    response["element_plans"] = parsed.data.get("element_plans")

                # Log planning operation completion
                planning_duration = time.time() - planning_start_time
                logger.debug(f"Test plan generation completed in {planning_duration:.2f}s with confidence {response['confidence']}")

                return response
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                metadata = self._unify_metadata(
                    provider="openai",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"parse_error": parsed.error},
                )
                response = {
                    "plan_summary": "Test planning completed (JSON parse failed)",
                    "detailed_plan": content,
                    "confidence": 0.3,
                    "notes": f"JSON parsing failed: {parsed.error}",
                    "metadata": metadata,
                }

                return response

        except Exception as e:
            planning_duration = time.time() - planning_start_time
            logger.error(f"OpenAI test plan generation failed after {planning_duration:.2f}s: {e}")
            raise LLMError(
                message=f"Test plan generation failed: {e}",
                provider="openai",
                operation="generate_test_plan",
                model=self.model,
            ) from e

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        thinking_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a chat completion request to OpenAI."""
        # Use provided max_tokens or fallback to instance default, then enforce catalog limits
        requested_tokens = max_tokens if max_tokens is not None else self.max_tokens
        tokens_to_use = self.token_policy.enforce_catalog_limits(requested_tokens, "chat_completion")

        # Branch: Use Responses API for o-series reasoning models
        if self.token_policy.is_o_series_reasoning_model():
            try:
                # Build a safe combined input. Responses API supports structured inputs,
                # but combining here keeps compatibility across SDK versions.
                combined_input = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

                # Verbose request log
                self.logging_artifacts.debug_log_request(
                    "responses.create",
                    {
                        "model": self.model,
                        "input": combined_input,
                        "max_output_tokens": tokens_to_use,
                        **{k: v for k, v in kwargs.items()},
                    },
                )

                response = self._client.responses.create(
                    model=self.model,
                    input=combined_input,
                    max_output_tokens=tokens_to_use,
                    **kwargs,
                )

                # Extract content and usage
                content = ""
                try:
                    content = getattr(response, "output_text", "") or ""
                except Exception:
                    content = ""
                if not content:
                    # Fallback: try to extract text from response.output blocks
                    try:
                        output = getattr(response, "output", None)
                        if output and isinstance(output, list):
                            # Find first text block
                            for block in output:
                                # block may contain {"content": [{"type":"output_text", "text":"..."}]} or similar
                                items = None
                                if isinstance(block, dict):
                                    items = block.get("content") or block.get("items")
                                if items and isinstance(items, list):
                                    for item in items:
                                        text = None
                                        if isinstance(item, dict):
                                            text = item.get("text") or item.get("value")
                                        if text:
                                            content = str(text)
                                            break
                                if content:
                                    break
                    except Exception:
                        pass

                usage_info = {}
                try:
                    if getattr(response, "usage", None):
                        usage = response.usage
                        # usage may expose input_tokens/output_tokens/total_tokens
                        prompt_tokens = getattr(usage, "input_tokens", None)
                        completion_tokens = getattr(usage, "output_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                        usage_info = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                            if total_tokens is not None
                            else ((prompt_tokens or 0) + (completion_tokens or 0)),
                        }
                except Exception:
                    usage_info = {}

                # Track costs if cost port is available
                try:
                    if getattr(response, "usage", None):
                        usage = response.usage
                        tokens_total = getattr(usage, "total_tokens", None) or (
                            getattr(usage, "input_tokens", 0)
                            + getattr(usage, "output_tokens", 0)
                        )
                        cost = self._calculate_api_cost(
                            usage, getattr(response, "model", self.model)
                        )
                        self._track_cost(
                            operation="responses",
                            tokens_used=tokens_total,
                            cost=cost,
                            api_calls=1,
                            model=getattr(response, "model", self.model),
                            prompt_tokens=getattr(usage, "input_tokens", None),
                            completion_tokens=getattr(usage, "output_tokens", None),
                        )
                except Exception as e:
                    logger.debug(f"Cost tracking skipped due to usage format: {e}")

                # Verbose response log
                self.logging_artifacts.debug_log_response(
                    "responses.create",
                    content=content,
                    usage=usage_info,
                    raw_obj=response,
                )

                return {
                    "content": content,
                    "usage": usage_info,
                    "model": getattr(response, "model", self.model),
                    "finish_reason": None,
                }

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise OpenAIError(f"OpenAI API error: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error in Responses API call: {e}")
                raise OpenAIError(f"Responses API call failed: {e}") from e

        # Default path: Chat Completions for non o-series models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prepare request parameters using token policy
        request_params = self.token_policy.prepare_request_params(
            max_tokens=tokens_to_use,
            temperature=self.temperature,
            **kwargs,
        )

        # Enforce JSON output when supported (best-effort)
        request_params["response_format"] = {"type": "json_object"}

        try:
            # Verbose request log
            self.logging_artifacts.debug_log_request(
                "chat.completions.create",
                {
                    "model": self.model,
                    "messages": messages,
                    **request_params,
                },
            )

            try:
                response = self._openai_client.chat_completion(
                    messages=messages,
                    **request_params,
                )
            except TypeError as te:
                err_msg = str(te)
                # Fallback: remove response_format if unsupported
                if "response_format" in err_msg and "unexpected keyword" in err_msg:
                    req2 = dict(request_params)
                    req2.pop("response_format", None)
                    response = self._openai_client.chat_completion(messages=messages, **req2)
                # Fallback: older SDKs may not support max_completion_tokens
                elif (
                    "max_completion_tokens" in err_msg
                    and "unexpected keyword" in err_msg
                ):
                    req2 = dict(request_params)
                    tokens_param = req2.pop("max_completion_tokens", None)
                    if tokens_param is not None:
                        req2["max_tokens"] = tokens_param
                    response = self._openai_client.chat_completion(messages=messages, **req2)
                else:
                    raise

            # Extract content and usage information
            content = ""
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.content:
                    content = message.content

            usage_info = {}
            if response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Track costs if cost port is available
            if response.usage:
                try:
                    cost = self._calculate_api_cost(response.usage, response.model)
                    self._track_cost(
                        operation="chat_completion",
                        tokens_used=response.usage.total_tokens,
                        cost=cost,
                        api_calls=1,
                        model=response.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                    )
                except Exception as e:
                    logger.warning(f"Cost tracking failed: {e}")

            # Verbose response log
            self.logging_artifacts.debug_log_response(
                "chat.completions.create",
                content=content,
                usage=usage_info,
                raw_obj=response,
            )

            return {
                "content": content,
                "usage": usage_info,
                "model": response.model,
                "finish_reason": (
                    response.choices[0].finish_reason if response.choices else None
                ),
            }

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIError(f"OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise OpenAIError(f"Chat completion failed: {e}") from e

    def _calculate_api_cost(self, usage: Any, model: str) -> float:
        """Calculate the API cost based on token usage and model pricing.

        Args:
            usage: OpenAI usage object with token counts
            model: Model name used for the request

        Returns:
            Calculated cost in USD
        """
        return calculate_cost(usage, model, provider="openai")

    def _supports_reasoning_features(self) -> bool:
        """Check if the model supports reasoning features."""
        from ....config.model_catalog_loader import resolve_model
        catalog_entry = resolve_model("openai", self.model)
        if catalog_entry and catalog_entry.flags:
            return bool(catalog_entry.flags.reasoning_capable)
        return False

    def _requires_completion_tokens_param(self) -> bool:
        """Check if the model requires max_completion_tokens parameter."""
        return self._supports_reasoning_features()

    def _supports_temperature_adjustment(self) -> bool:
        """Check if the model supports temperature adjustment."""
        # Reasoning models (o1 series) don't support temperature
        return not self._supports_reasoning_features()

    def _should_persist_artifacts(self) -> bool:
        """Check if artifacts should be persisted for debugging."""
        # Check if verbose logging is enabled
        import logging
        return logger.isEnabledFor(logging.DEBUG)

    def _is_o_series_reasoning_model(self) -> bool:
        """Check if this is an o1-series reasoning model."""
        return self.model.startswith('o1-') or self.model.startswith('o1')

    def _debug_log_request(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Debug log the request payload."""
        self.logging_artifacts.debug_log_request(endpoint, payload=payload)

    def _debug_log_response(self, endpoint: str, content: str = "", usage: dict[str, Any] | None = None, raw_obj: Any = None) -> None:
        """Debug log the response."""
        self.logging_artifacts.debug_log_response(endpoint, content=content, usage=usage, raw_obj=raw_obj)

    def _artifact_request(self, endpoint: str, payload: dict[str, Any], pretty_json: str = "") -> None:
        """Store request artifacts for debugging."""
        super()._artifact_request(endpoint, payload)

    def _artifact_response(self, endpoint: str, content: str, usage: dict[str, Any], raw_obj: Any) -> None:
        """Store response artifacts for debugging."""
        response_dict = {"content": content, "usage": usage, "raw": str(raw_obj) if raw_obj else None}
        super()._artifact_response(endpoint, response_dict)
