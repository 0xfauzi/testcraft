"""Real Azure OpenAI adapter implementation using the latest v1.2.0 SDK."""

from __future__ import annotations

import logging
from typing import Any

import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

from ...config.credentials import CredentialError, CredentialManager
from ...config.model_catalog_loader import resolve_model
from ...ports.cost_port import CostPort
from ...ports.llm_error import LLMError
from ...prompts.registry import PromptRegistry
from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries
from .base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class AzureOpenAIError(Exception):
    """Azure OpenAI adapter specific errors."""

    pass


class AzureOpenAIAdapter(BaseLLMAdapter, LLMPort):
    """
    Production Azure OpenAI adapter using the latest v1.2.0 SDK.

    Features:
    - Secure credential management via environment variables
    - Azure AD authentication support
    - Proper error handling and retries with exponential backoff
    - Support for Azure OpenAI deployments
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    """

    def __init__(
        self,
        deployment: str = "gpt-4o",
        api_version: str = "2024-02-15-preview",
        timeout: float = 180.0,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        **kwargs: Any,
    ):
        """Initialize Azure OpenAI adapter.

        Args:
            deployment: Azure OpenAI deployment name
            api_version: Azure OpenAI API version
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0-2.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional Azure OpenAI client parameters
        """
        # Azure-specific attributes
        self.deployment = deployment
        self.api_version = api_version
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize base adapter (provides token_calculator and helpers)
        # Use canonical OpenAI model for token calculations to ensure correct limits
        canonical_model = self._map_deployment_to_canonical_model_static(self.deployment)
        BaseLLMAdapter.__init__(
            self,
            provider="openai",  # Use "openai" provider for catalog lookup
            model=canonical_model,  # Use canonical model for proper limits
            prompt_registry=prompt_registry,
            credential_manager=credential_manager,
            cost_port=cost_port,
        )

        # Initialize Azure OpenAI client
        self._client: AzureOpenAI | None = None
        self._initialize_client(**kwargs)

    @staticmethod
    def _map_deployment_to_canonical_model_static(deployment: str) -> str:
        """Static version of deployment mapping for use during initialization."""
        # Common Azure deployment name patterns to canonical model mapping
        deployment_lower = deployment.lower()
        
        # GPT-4o variants
        if "gpt-4o-mini" in deployment_lower or "4o-mini" in deployment_lower:
            return "gpt-4o-mini"
        elif "gpt-4o" in deployment_lower or "4o" in deployment_lower:
            return "gpt-4o"
        
        # o1-series reasoning models
        elif "o1-mini" in deployment_lower:
            return "o1-mini"
        elif "o1-preview" in deployment_lower or "o1" in deployment_lower:
            return "o1-mini"  # Fallback to o1-mini for now
            
        # GPT-4 variants  
        elif "gpt-4-turbo" in deployment_lower or "gpt4-turbo" in deployment_lower:
            return "gpt-4o"  # Map older turbo to gpt-4o
        elif "gpt-4" in deployment_lower or "gpt4" in deployment_lower:
            return "gpt-4o"  # Map gpt-4 variants to gpt-4o
            
        # Default fallback
        else:
            return "gpt-4o"

    def _map_deployment_to_canonical_model(self) -> str:
        """Map Azure deployment name to canonical OpenAI model ID.
        
        This ensures that Azure deployments use the same limits and capabilities
        as their underlying OpenAI models for parameter compliance.
        
        Returns:
            Canonical OpenAI model ID for catalog lookup
        """
        canonical = self._map_deployment_to_canonical_model_static(self.deployment)
        if canonical == "gpt-4o" and self.deployment.lower() != "gpt-4o":
            logger.warning(f"Unknown Azure deployment '{self.deployment}', using gpt-4o defaults")
        return canonical

    def _supports_reasoning_features(self) -> bool:
        """Check if the underlying model supports reasoning features."""
        canonical_model = self._map_deployment_to_canonical_model()
        catalog_entry = resolve_model("openai", canonical_model)
        if catalog_entry and catalog_entry.flags:
            return bool(catalog_entry.flags.reasoning_capable)
        return False

    def _requires_completion_tokens_param(self) -> bool:
        """Check if the deployment requires max_completion_tokens parameter."""
        return self._supports_reasoning_features()

    def _enforce_catalog_limits(self, requested_tokens: int, use_case: str = "general") -> int:
        """Enforce catalog-defined token limits based on the canonical model.
        
        Args:
            requested_tokens: Number of tokens requested
            use_case: Use case context for logging
            
        Returns:
            Token count clamped to catalog limits
        """
        canonical_model = self._map_deployment_to_canonical_model()
        catalog_entry = resolve_model("openai", canonical_model)  # Use OpenAI provider for canonical lookup
        
        if catalog_entry and catalog_entry.limits:
            max_allowed = catalog_entry.limits.default_max_output
            if requested_tokens > max_allowed:
                logger.warning(
                    f"Requested {requested_tokens} tokens for {use_case} but deployment '{self.deployment}' "
                    f"(canonical model '{canonical_model}') limit is {max_allowed} tokens, clamping to catalog limit"
                )
                return max_allowed
        else:
            logger.debug(f"No catalog limits found for deployment '{self.deployment}' (canonical: {canonical_model}), using requested value")
        
        return requested_tokens

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize the Azure OpenAI client with credentials."""
        try:
            credentials = self.credential_manager.get_provider_credentials(
                "azure-openai"
            )

            client_kwargs = {
                "api_key": credentials["api_key"],
                "azure_endpoint": credentials["azure_endpoint"],
                "api_version": self.api_version,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                **kwargs,
            }

            self._client = AzureOpenAI(**client_kwargs)

            logger.info(
                f"Azure OpenAI client initialized with deployment: {self.deployment}"
            )

        except CredentialError as e:
            logger.warning(
                f"Azure OpenAI credentials not available, using stub client: {e}"
            )

            class _StubChatCompletions:
                def create(self, **_kwargs):
                    class _Choice:
                        def __init__(self, text: str) -> None:
                            class _Msg:
                                def __init__(self, content: str) -> None:
                                    self.content = content

                            self.message = _Msg(text)
                            self.finish_reason = "stop"

                    class _Resp:
                        def __init__(self) -> None:
                            self.choices = [
                                _Choice(
                                    '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                                )
                            ]
                            self.usage = None
                            self.model = "stub-deployment"

                    return _Resp()

            class _StubChat:
                def __init__(self) -> None:
                    self.completions = _StubChatCompletions()

            class _StubClient:
                def __init__(self) -> None:
                    self.chat = _StubChat()

            self._client = _StubClient()  # type: ignore[assignment]

        except Exception as e:
            logger.warning(f"Azure client init failed, using stub client: {e}")

            class _StubChatCompletions:
                def create(self, **_kwargs):
                    class _Choice:
                        def __init__(self, text: str) -> None:
                            class _Msg:
                                def __init__(self, content: str) -> None:
                                    self.content = content

                            self.message = _Msg(text)
                            self.finish_reason = "stop"

                    class _Resp:
                        def __init__(self) -> None:
                            self.choices = [
                                _Choice(
                                    '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                                )
                            ]
                            self.usage = None
                            self.model = "stub-deployment"

                    return _Resp()

            class _StubChat:
                def __init__(self) -> None:
                    self.completions = _StubChatCompletions()

            class _StubClient:
                def __init__(self) -> None:
                    self.chat = _StubChat()

            self._client = _StubClient()  # type: ignore[assignment]

    def _calculate_api_cost(self, usage_info: dict[str, Any], deployment: str) -> float:
        """
        Calculate the cost of an API call based on token usage and deployment.

        Args:
            usage_info: Dictionary containing prompt_tokens, completion_tokens, total_tokens
            deployment: The Azure OpenAI deployment name

        Returns:
            The calculated cost in USD
        """
        # MIGRATION: Use catalog-driven pricing per Task 31.10
        # Hardcoded pricing tiers removed - now uses model catalog
        from .pricing import calculate_cost
        
        try:
            # Try to use catalog-driven pricing
            total_cost = calculate_cost(
                deployment, "openai",  # Azure uses OpenAI models
                usage_info.get("prompt_tokens", 0),
                usage_info.get("completion_tokens", 0)
            )
        except Exception as e:
            logger.warning(f"Could not calculate cost from catalog for Azure deployment '{deployment}': {e}")
            # Conservative fallback pricing (per 1000 tokens)
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            
            # Use conservative estimates when catalog is unavailable
            prompt_cost = (prompt_tokens / 1000) * 0.01  # $10 per million
            completion_cost = (completion_tokens / 1000) * 0.03  # $30 per million
            total_cost = prompt_cost + completion_cost

        return total_cost

    @property
    def client(self) -> AzureOpenAI:
        """Get the Azure OpenAI client, initializing if needed."""
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

        # Prompts from registry
        additional_context = {"context": context} if context else {}
        system_prompt, user_prompt = self._prompts(
            "llm_test_generation",
            code_content=code_content,
            additional_context=additional_context,
            test_framework=test_framework,
        )

        budgets = self._calc_budgets(
            use_case="test_generation",
            input_text=code_content + (context or ""),
        )
        # Allow per-request override for max_tokens
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={
                        "api_version": self.api_version,
                        "reasoning": parsed.data.get("reasoning", ""),
                    },
                    raw_provider_fields={"deployment": self.deployment},
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
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"api_version": self.api_version, "parse_error": parsed.error},
                    raw_provider_fields={"deployment": self.deployment, "raw_content": content},
                )
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Azure OpenAI test generation failed: {e}")
            raise LLMError(
                message=f"Test generation failed: {e}",
                provider="azure-openai",
                operation="generate_tests",
                model=self.deployment,
            ) from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Prompts from registry
        system_prompt, user_prompt = self._prompts(
            "llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        budgets = self._calc_budgets(
            use_case="code_analysis",
            input_text=code_content,
        )
        # Allow per-request override for max_tokens
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={
                        "api_version": self.api_version,
                        "analysis_type": analysis_type,
                        "summary": parsed.data.get("analysis_summary", ""),
                    },
                    raw_provider_fields={"deployment": self.deployment},
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
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={
                        "api_version": self.api_version,
                        "analysis_type": analysis_type,
                        "raw_content": content,
                        "parse_error": parsed.error,
                    },
                    raw_provider_fields={"deployment": self.deployment},
                )
                return {
                    "testability_score": 5.0,
                    "complexity_metrics": {},
                    "recommendations": [],
                    "potential_issues": [],
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Azure OpenAI code analysis failed: {e}")
            raise LLMError(
                message=f"Code analysis failed: {e}",
                provider="azure-openai",
                operation="analyze_code",
                model=self.deployment,
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

        # Use pre-rendered instructions built by the caller (RefineAdapter)
        if system_prompt is None:
            system_prompt, _ = self._prompts("llm_content_refinement")
        user_prompt = refinement_instructions

        budgets = self._calc_budgets(
            use_case="refinement",
            input_text=original_content + refinement_instructions,
        )
        # Allow per-request override for max_tokens
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=effective_max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse and normalize refinement response using base helper
            refined_data, parsed_info = self._parse_and_normalize_refinement(content)

            if refined_data:
                metadata = self._unify_metadata(
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                    extra={"api_version": self.api_version},
                    raw_provider_fields={"deployment": self.deployment},
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
                logger.error(f"Azure OpenAI schema validation failed: {error_msg}")
                metadata = self._unify_metadata(
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info=parsed_info,
                    extra={"api_version": self.api_version, "schema_error": error_msg},
                    raw_provider_fields={"deployment": self.deployment},
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
            logger.error(f"Azure OpenAI content refinement failed: {e}")
            raise LLMError(
                message=f"Content refinement failed: {e}",
                provider="azure-openai",
                operation="refine_content",
                model=self.deployment,
            ) from e

    def _chat_completion(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Make a chat completion request to Azure OpenAI."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_kwargs = {
            "model": self.deployment,  # Use deployment name as model for Azure
            "messages": messages,
            **kwargs,
        }

        # Use proper parameter name based on underlying model capabilities and enforce catalog limits
        requested_tokens = max_tokens if max_tokens is not None else self.max_tokens
        tokens_to_use = self._enforce_catalog_limits(requested_tokens, "chat_completion")
        
        if self._requires_completion_tokens_param():
            request_kwargs["max_completion_tokens"] = tokens_to_use
            # Don't set temperature for reasoning models (they use fixed temperature)
            if not self._supports_reasoning_features():
                request_kwargs["temperature"] = self.temperature
            else:
                logger.debug(f"Skipping temperature for reasoning deployment {self.deployment} (uses fixed temperature)")
        else:
            request_kwargs["max_tokens"] = tokens_to_use
            request_kwargs["temperature"] = self.temperature

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                **request_kwargs
            )

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

                # Track costs using unified Base helper
                if usage_info:
                    try:
                        cost = self._calculate_api_cost(usage_info, self.deployment)
                        self._track_cost(
                            operation="chat_completion",
                            tokens_used=usage_info["total_tokens"],
                            cost=cost,
                            deployment=self.deployment,
                            api_version=self.api_version,
                            prompt_tokens=usage_info["prompt_tokens"],
                            completion_tokens=usage_info["completion_tokens"],
                        )
                    except Exception as e:
                        logger.debug(f"Cost tracking failed: {e}")

            return {
                "content": content,
                "usage": usage_info,
                "deployment": response.model,  # Azure returns deployment name
                "finish_reason": (
                    response.choices[0].finish_reason if response.choices else None
                ),
            }

        except openai.APIError as e:
            logger.error(f"Azure OpenAI API error: {e}")
            status_code = getattr(e, "status_code", None)
            raise LLMError(
                message=f"Azure OpenAI API error: {e}",
                provider="azure-openai",
                operation="chat_completion",
                model=self.deployment,
                status_code=status_code,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise LLMError(
                message=f"Chat completion failed: {e}",
                provider="azure-openai",
                operation="chat_completion",
                model=self.deployment,
            ) from e

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

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_planning", input_length=input_length
        )

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
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={"api_version": self.api_version},
                    raw_provider_fields={"deployment": self.deployment},
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
                logger.debug(f"Azure test plan generation completed in {planning_duration:.2f}s with confidence {response['confidence']}")
                
                return response
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                metadata = self._unify_metadata(
                    provider="azure-openai",
                    model=self.deployment,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"api_version": self.api_version, "parse_error": parsed.error},
                    raw_provider_fields={"deployment": self.deployment},
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
            logger.error(f"Azure test plan generation failed after {planning_duration:.2f}s: {e}")
            raise LLMError(
                message=f"Test plan generation failed: {e}",
                provider="azure-openai",
                operation="generate_test_plan",
                model=self.deployment,
            ) from e
