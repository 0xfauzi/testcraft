"""Real Anthropic Claude adapter implementation using the latest v0.66.0 SDK."""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic import Anthropic
from anthropic.types import Message

from ...config.credentials import CredentialError, CredentialManager
from ...config.model_catalog_loader import resolve_model
from ...ports.cost_port import CostPort
from ...ports.llm_error import LLMError
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
from .token_calculator import TokenCalculator
from .base import BaseLLMAdapter
from .pricing import calculate_cost

logger = logging.getLogger(__name__)


class ClaudeError(Exception):
    """Claude adapter specific errors."""

    pass


class ClaudeAdapter(BaseLLMAdapter, LLMPort):
    """
    Production Anthropic Claude adapter using the latest v0.66.0 SDK.

    Features:
    - Secure credential management via environment variables
    - Proper error handling and retries with exponential backoff
    - Support for latest Claude models (3.5 Sonnet, 3.7 Sonnet, Sonnet 4, Opus 4)
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    - System prompt support for better control
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        timeout: float = 180.0,
        max_tokens: int | None = None,  # Will be calculated automatically
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        # Beta features configuration
        enable_extended_context: bool = False,
        enable_extended_output: bool = False,
        **kwargs: Any,
    ):
        """Initialize Claude adapter.

        Args:
            model: Claude model name (e.g., "claude-3-5-sonnet", "claude-3-7-sonnet", "claude-sonnet-4")
            timeout: Request timeout in seconds (default: 180s for test generation)
            max_tokens: Maximum tokens in response (auto-calculated if None)
            temperature: Response randomness (0.0-1.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            prompt_registry: Custom prompt registry (optional)
            cost_port: Optional cost tracking port (optional)
            enable_extended_context: Enable extended context beyond documented limits (default: False)
            enable_extended_output: Enable extended output tokens beyond documented limits (default: False)
            **kwargs: Additional Anthropic client parameters
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Beta features configuration
        self.enable_extended_context = enable_extended_context
        self.enable_extended_output = enable_extended_output

        # Initialize base adapter (prompts, creds, cost, token calculator)
        BaseLLMAdapter.__init__(
            self,
            provider="anthropic",
            model=model,
            prompt_registry=prompt_registry,
            cost_port=cost_port,
            credential_manager=credential_manager,
        )
        
        # Override token calculator with beta feature flags
        self.token_calculator = TokenCalculator(
            provider="anthropic", 
            model=model,
            enable_extended_context=self.enable_extended_context,
            enable_extended_output=self.enable_extended_output
        )

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Initialize Anthropic client
        self._client: Anthropic | None = None
        self._initialize_client(**kwargs)
        
        # Log beta feature configuration for structured monitoring
        if self.enable_extended_context or self.enable_extended_output:
            enabled_features = []
            if self.enable_extended_context:
                enabled_features.append("extended_context")
            if self.enable_extended_output:
                enabled_features.append("extended_output")
            logger.info(
                f"ClaudeAdapter initialized with beta features: {', '.join(enabled_features)} "
                f"for model {self.model}"
            )
        else:
            logger.debug(f"ClaudeAdapter initialized with standard limits for model {self.model}")

    def _supports_thinking_tokens(self) -> bool:
        """Check if the current model supports thinking tokens based on catalog."""
        catalog_entry = resolve_model("anthropic", self.model)
        if catalog_entry and catalog_entry.limits:
            return bool(catalog_entry.limits.max_thinking and catalog_entry.limits.max_thinking > 0)
        return False

    def _enforce_catalog_limits(self, requested_tokens: int, use_case: str = "general") -> int:
        """Enforce catalog-defined token limits for the model.
        
        Args:
            requested_tokens: Number of tokens requested
            use_case: Use case context for logging
            
        Returns:
            Token count clamped to catalog limits
        """
        catalog_entry = resolve_model("anthropic", self.model)
        if catalog_entry and catalog_entry.limits:
            # Use extended output limit if beta features are enabled, otherwise use default
            if self.enable_extended_output:
                # For extended output, we might allow higher limits but still need an upper bound
                # This would be configured in catalog or could be a multiple of default
                max_allowed = catalog_entry.limits.default_max_output * 2  # Example multiplier
                logger.debug(f"Extended output enabled, using {max_allowed} token limit")
            else:
                max_allowed = catalog_entry.limits.default_max_output
            
            if requested_tokens > max_allowed:
                logger.warning(
                    f"Requested {requested_tokens} tokens for {use_case} but model '{self.model}' "
                    f"limit is {max_allowed} tokens, clamping to catalog limit"
                )
                return max_allowed
        else:
            logger.debug(f"No catalog limits found for model '{self.model}', using requested value")
        
        return requested_tokens

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize the Anthropic client with credentials."""
        try:
            credentials = self.credential_manager.get_provider_credentials("anthropic")

            client_kwargs = {
                "api_key": credentials["api_key"],
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                **kwargs,
            }

            self._client = Anthropic(**client_kwargs)

            logger.info(f"Anthropic client initialized with model: {self.model}")

        except CredentialError as e:
            logger.warning(
                f"Anthropic credentials not available, using stub client: {e}"
            )

            class _StubMessages:
                def create(self, **_kwargs):
                    class _Usage:
                        input_tokens = 0
                        output_tokens = 0

                    class _Block:
                        def __init__(self, text: str) -> None:
                            self.text = text

                    class _Resp:
                        def __init__(self) -> None:
                            self.content = [
                                _Block(
                                    '{"testability_score": 5.0, "complexity_metrics": {}, "recommendations": [], "potential_issues": [], "analysis_summary": "stub"}'
                                )
                            ]
                            self.usage = _Usage()
                            self.model = "stub-model"
                            self.stop_reason = "end_turn"

                    return _Resp()

            class _StubClient:
                def __init__(self) -> None:
                    self.messages = _StubMessages()

            self._client = _StubClient()  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"Anthropic client init failed, using stub client: {e}")

            class _StubMessages:
                def create(self, **_kwargs):
                    class _Usage:
                        input_tokens = 0
                        output_tokens = 0

                    class _Block:
                        def __init__(self, text: str) -> None:
                            self.text = text

                    class _Resp:
                        def __init__(self) -> None:
                            self.content = [
                                _Block(
                                    '{"testability_score": 5.0, "complexity_metrics": {}, "recommendations": [], "potential_issues": [], "analysis_summary": "stub"}'
                                )
                            ]
                            self.usage = _Usage()
                            self.model = "stub-model"
                            self.stop_reason = "end_turn"

                    return _Resp()

            class _StubClient:
                def __init__(self) -> None:
                    self.messages = _StubMessages()

            self._client = _StubClient()  # type: ignore[assignment]

    @property
    def client(self) -> Anthropic:
        """Get the Anthropic client, initializing if needed."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate test cases for the provided code content."""

        additional_context = {"context": context} if context else {}
        system_prompt, user_prompt = self._prompts(
            "llm_test_generation",
            code_content=code_content,
            additional_context=additional_context,
            test_framework=test_framework,
        )

        # Per-request budgets
        budgets = self._calc_budgets(
            use_case="test_generation",
            input_text=code_content + (context or ""),
        )

        # Embed thinking guidance if supported by the model and thinking tokens are allocated
        sys_prompt = system_prompt
        if budgets.get("thinking_tokens") and self._supports_thinking_tokens():
            sys_prompt = (
                f"{system_prompt}\n\nGuidance: Use careful internal reasoning up to {int(budgets['thinking_tokens'])} tokens to plan your answer, "
                "but do not include internal reasoning in the final output."
            )
        elif budgets.get("thinking_tokens") and not self._supports_thinking_tokens():
            logger.debug(f"Thinking tokens requested but not supported by model '{self.model}' for test generation")

        def call() -> dict[str, Any]:
            return self._create_message(
                system=sys_prompt, user_message=user_prompt, max_tokens=int(budgets["max_tokens"]), **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="anthropic",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={"reasoning": parsed.data.get("reasoning", "")},
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
                    provider="anthropic",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"parse_error": parsed.error, "raw_content": content},
                )
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Claude test generation failed: {e}")
            raise LLMError(
                message=f"Test generation failed: {e}",
                provider="anthropic",
                operation="generate_tests",
                model=self.model,
            ) from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        system_prompt, user_prompt = self._prompts(
            "llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        # Per-request budgets with override support
        budgets = self._calc_budgets(
            use_case="code_analysis",
            input_text=code_content,
        )
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )
        effective_thinking_tokens = (
            kwargs.pop("thinking_tokens") if "thinking_tokens" in kwargs else budgets.get("thinking_tokens")
        )

        # Embed thinking guidance if supported by the model and thinking tokens are allocated
        sys_prompt = system_prompt
        if effective_thinking_tokens and self._supports_thinking_tokens():
            sys_prompt = (
                f"{system_prompt}\n\nGuidance: Use careful internal reasoning up to {int(effective_thinking_tokens)} tokens to plan your answer, "
                "but do not include internal reasoning in the final output."
            )
        elif effective_thinking_tokens and not self._supports_thinking_tokens():
            logger.debug(f"Thinking tokens requested but not supported by model '{self.model}' for code analysis")

        def call() -> dict[str, Any]:
            return self._create_message(
                system=sys_prompt,
                user_message=user_prompt,
                max_tokens=effective_max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="anthropic",
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
                    provider="anthropic",
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
            logger.error(f"Claude code analysis failed: {e}")
            raise LLMError(
                message=f"Code analysis failed: {e}",
                provider="anthropic",
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
        if system_prompt is None:
            system_prompt, _ = self._prompts("llm_content_refinement")
        user_prompt = refinement_instructions

        # Per-request budgets
        budgets = self._calc_budgets(
            use_case="refinement",
            input_text=original_content + refinement_instructions,
        )

        # Embed thinking guidance if supported by the model and thinking tokens are allocated
        sys_prompt = system_prompt
        if budgets.get("thinking_tokens") and self._supports_thinking_tokens():
            sys_prompt = (
                f"{system_prompt}\n\nGuidance: Use careful internal reasoning up to {int(budgets['thinking_tokens'])} tokens to plan your answer, "
                "but do not include internal reasoning in the final output."
            )
        elif budgets.get("thinking_tokens") and not self._supports_thinking_tokens():
            logger.debug(f"Thinking tokens requested but not supported by model '{self.model}' for content refinement")

        def call() -> dict[str, Any]:
            return self._create_message(
                system=sys_prompt, user_message=user_prompt, max_tokens=int(budgets["max_tokens"]), **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Use base helper for initial parsing and normalization
            normalized_data, parsed_info = self._parse_and_normalize_refinement(content)
            
            if normalized_data is None:
                # Attempt repair using Claude-specific logic if base parsing fails
                parsed = parse_json_response(content)
                if parsed.success and parsed.data:
                    # Check if refined_content is specifically invalid for Claude
                    if "refined_content" not in parsed.data or self._is_invalid_refined_content(parsed.data.get("refined_content")):
                        from .common import create_repair_prompt
                        
                        repair_prompt = create_repair_prompt(
                            parsed_info.get("error", "Invalid schema"),
                            ["refined_content", "changes_made", "confidence", "improvement_areas"]
                        )
                        
                        try:
                            repair_result = self._create_message(
                                system=system_prompt,
                                user_message=f"{user_prompt}\n\n{repair_prompt}",
                                temperature=0.0,  # Deterministic repair
                                **kwargs
                            )
                            
                            repair_content = repair_result.get("content", "")
                            normalized_data, repaired_parsed_info = self._parse_and_normalize_refinement(repair_content)
                            if normalized_data:
                                logger.info("Claude schema repair successful.")
                                parsed_info = repaired_parsed_info
                            else:
                                logger.error(f"Claude repair failed: {repaired_parsed_info.get('error')}")
                            
                        except Exception as repair_e:
                            logger.error(f"Claude repair attempt failed: {repair_e}")
            
            if normalized_data:
                # Success case
                metadata = self._unify_metadata(
                    provider="anthropic",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                )
                return {
                    "refined_content": normalized_data["refined_content"],
                    "changes_made": normalized_data["changes_made"],
                    "confidence": normalized_data["confidence"],
                    "improvement_areas": normalized_data["improvement_areas"],
                    "suspected_prod_bug": normalized_data.get("suspected_prod_bug"),
                    "metadata": metadata,
                }
            else:
                # Schema validation failed even after repair
                logger.error(f"Claude schema validation failed: {parsed_info.get('error')}")
                metadata = self._unify_metadata(
                    provider="anthropic",
                    model=self.model,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                    extra={"schema_error": parsed_info.get("error")},
                )
                return {
                    "refined_content": original_content,  # Safe fallback
                    "changes_made": f"Schema validation failed: {parsed_info.get('error')}",
                    "confidence": 0.0,
                    "improvement_areas": ["schema_error"],
                    "suspected_prod_bug": None,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Claude content refinement failed: {e}")
            raise LLMError(
                message=f"Content refinement failed: {e}",
                provider="anthropic",
                operation="refine_content",
                model=self.model,
            ) from e
    
    def _is_invalid_refined_content(self, content: Any) -> bool:
        """Check if refined_content is invalid (None, empty, or literal 'None'/'null')."""
        if content is None:
            return True
        if not isinstance(content, str):
            return True
        if not content.strip():
            return True
        if content.strip().lower() in ("none", "null"):
            return True
        return False
    
    def _get_beta_headers(self) -> dict[str, str]:
        """Get beta headers for the current model if beta features are enabled."""
        beta_headers = {}
        
        # Only proceed if at least one beta feature is enabled
        if not (self.enable_extended_context or self.enable_extended_output):
            return beta_headers
        
        # Load beta headers from model catalog
        try:
            catalog_entry = resolve_model(self.provider, self.model)
            if catalog_entry and catalog_entry.beta and catalog_entry.beta.headers:
                beta_headers = catalog_entry.beta.headers.copy()
                
                # Log when beta features are enabled and headers are applied
                if beta_headers:
                    features_enabled = []
                    if self.enable_extended_context:
                        features_enabled.append("extended_context")
                    if self.enable_extended_output:
                        features_enabled.append("extended_output")
                    
                    logger.info(
                        f"Beta features enabled for {self.model}: {', '.join(features_enabled)}. "
                        f"Applied beta headers: {list(beta_headers.keys())}"
                    )
                else:
                    logger.debug(f"Beta features enabled for {self.model} but no beta headers defined in catalog")
                    
        except Exception as e:
            logger.warning(f"Failed to load beta headers from catalog for {self.model}: {e}")
        
        return beta_headers

    def _create_message(
        self, system: str, user_message: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a message using Claude's messages API."""

        # Use provided max_tokens or instance default, then enforce catalog limits
        requested_max_tokens = kwargs.get("max_tokens", self.max_tokens)
        enforced_max_tokens = self._enforce_catalog_limits(requested_max_tokens, "message_creation")

        request_kwargs = {
            "model": self.model,
            "max_tokens": enforced_max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
            **{k: v for k, v in kwargs.items() if k != "max_tokens"},  # Remove max_tokens from kwargs to avoid duplicate
        }
        
        # Add beta headers if beta features are enabled
        beta_headers = self._get_beta_headers()
        if beta_headers:
            # Anthropic client accepts extra_headers parameter
            extra_headers = kwargs.get("extra_headers", {})
            extra_headers.update(beta_headers)
            request_kwargs["extra_headers"] = extra_headers

        try:
            response: Message = self.client.messages.create(**request_kwargs)

            # Extract content from response
            content = ""
            if response.content and len(response.content) > 0:
                # Claude responses come as a list of content blocks
                content_blocks = []
                for block in response.content:
                    if hasattr(block, "text"):
                        content_blocks.append(block.text)
                content = "\n".join(content_blocks)

            # Extract usage information
            usage_info = {}
            if response.usage:
                usage_info = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }

            # Track costs using base helper if cost port is available
            if self.cost_port and response.usage:
                cost = self._calculate_api_cost(response.usage, response.model)
                self._track_cost(
                    operation="message_creation",
                    tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                    cost=cost,
                    api_calls=1,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            return {
                "content": content,
                "usage": usage_info,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "stop_sequence": getattr(response, "stop_sequence", None),
            }

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            status_code = getattr(e, "status_code", None)
            raise LLMError(
                message=f"Anthropic API error: {e}",
                provider="anthropic",
                operation="message_creation",
                model=self.model,
                status_code=status_code,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in message creation: {e}")
            raise LLMError(
                message=f"Message creation failed: {e}",
                provider="anthropic",
                operation="message_creation",
                model=self.model,
            ) from e

    def _calculate_api_cost(self, usage: Any, model: str) -> float:
        """Calculate the API cost based on token usage and model pricing.

        Args:
            usage: Anthropic usage object with token counts
            model: Model name used for the request

        Returns:
            Calculated cost in USD
        """
        return calculate_cost(usage, model, provider="anthropic")

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a test plan for the provided code content."""
        import time
        
        # Add simple telemetry tracking for planning operations
        planning_start_time = time.time()

        additional_context = {"context": context} if context else {}
        system_prompt, user_prompt = self._prompts(
            "llm_test_planning_v1",
            code_content=code_content,
            additional_context=additional_context,
        )

        # Per-request budgets using base helper
        budgets = self._calc_budgets(
            use_case="code_analysis",  # Use code_analysis as closest match for test planning
            input_text=code_content + (context or ""),
        )

        # Embed thinking guidance if supported by the model and thinking tokens are allocated
        sys_prompt = system_prompt
        if budgets.get("thinking_tokens") and self._supports_thinking_tokens():
            sys_prompt = (
                f"{system_prompt}\n\nGuidance: Use careful internal reasoning up to {int(budgets['thinking_tokens'])} tokens to plan your answer, "
                "but do not include internal reasoning in the final output."
            )
        elif budgets.get("thinking_tokens") and not self._supports_thinking_tokens():
            logger.debug(f"Thinking tokens requested but not supported by model '{self.model}' for test planning")

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
            return self._create_message(
                system=sys_prompt,
                user_message=user_prompt,
                max_tokens=int(budgets["max_tokens"]),
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="anthropic",
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
                logger.debug(f"Claude test plan generation completed in {planning_duration:.2f}s with confidence {response['confidence']}")
                
                return response
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                metadata = self._unify_metadata(
                    provider="anthropic",
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
            logger.error(f"Claude test plan generation failed after {planning_duration:.2f}s: {e}")
            raise LLMError(
                message=f"Test plan generation failed: {e}",
                provider="anthropic",
                operation="generate_test_plan",
                model=self.model,
            ) from e
