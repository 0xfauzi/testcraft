"""Real AWS Bedrock adapter implementation using LangChain ChatBedrock."""

from __future__ import annotations

import logging
from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from ...config.credentials import CredentialError, CredentialManager
from ...config.model_catalog_loader import resolve_model
from ...ports.cost_port import CostPort
from ...ports.llm_error import LLMError
from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries
from .base import BaseLLMAdapter
from ...prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class BedrockError(Exception):
    """Bedrock adapter specific errors."""

    pass


class BedrockAdapter(BaseLLMAdapter, LLMPort):
    """
    Production AWS Bedrock adapter using LangChain ChatBedrock.

    Features:
    - Secure credential management via AWS credentials
    - Proper error handling and retries with exponential backoff
    - Support for Anthropic Claude models on Bedrock
    - LangChain ChatBedrock integration for consistency
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        **kwargs: Any,
    ):
        """Initialize Bedrock adapter.

        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
            region_name: AWS region name (defaults to credentials or environment)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0-1.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional ChatBedrock parameters
        """
        # Bedrock-specific attributes
        self.model_id = model_id
        self.region_name = region_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize base adapter (provides token_calculator and helpers)
        BaseLLMAdapter.__init__(
            self,
            provider="bedrock",
            model=self.model_id,
            prompt_registry=prompt_registry,
            credential_manager=credential_manager,
            cost_port=cost_port,
        )

        # Initialize ChatBedrock client
        self._client: ChatBedrock | None = None
        self._initialize_client(**kwargs)

    def _supports_thinking_tokens(self) -> bool:
        """Check if the model supports thinking tokens."""
        catalog_entry = resolve_model("bedrock", self.model_id)
        if catalog_entry and catalog_entry.limits:
            return bool(catalog_entry.limits.max_thinking and catalog_entry.limits.max_thinking > 0)
        return False

    def _get_max_tokens_for_model(self) -> int:
        """Get the maximum tokens allowed for this model from the catalog."""
        catalog_entry = resolve_model("bedrock", self.model_id)
        if catalog_entry and catalog_entry.limits:
            return catalog_entry.limits.default_max_output
        return self.max_tokens

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize the ChatBedrock client with credentials."""
        try:
            credentials = self.credential_manager.get_provider_credentials("bedrock")

            client_kwargs = {
                "model_id": self.model_id,
                "model_kwargs": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                "region_name": self.region_name or credentials["region_name"],
                "aws_access_key_id": credentials["aws_access_key_id"],
                "aws_secret_access_key": credentials["aws_secret_access_key"],
                **kwargs,
            }

            self._client = ChatBedrock(**client_kwargs)

            logger.info(f"ChatBedrock client initialized with model: {self.model_id}")

        except CredentialError as e:
            # Use a stub client in test environments where credentials are absent
            logger.warning(f"Bedrock credentials not available, using stub client: {e}")

            class _StubClient:
                def invoke(self, messages, **_):
                    class _Resp:
                        content = '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                        response_metadata = {}

                    return _Resp()

            self._client = _StubClient()  # type: ignore[assignment]

        except Exception as e:
            # Fallback to stub client on unexpected init failure
            logger.warning(f"ChatBedrock init failed, using stub client: {e}")

            class _StubClient:
                def invoke(self, messages, **_):
                    class _Resp:
                        content = '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                        response_metadata = {}

                    return _Resp()

            self._client = _StubClient()  # type: ignore[assignment]

    @property
    def client(self) -> ChatBedrock:
        """Get the ChatBedrock client, initializing if needed."""
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
        system_message, user_content = self._prompts(
            "llm_test_generation",
            code_content=code_content,
            additional_context=additional_context,
            test_framework=test_framework,
        )

        budgets = self._calc_budgets(
            use_case="test_generation",
            input_text=code_content + (context or ""),
        )
        # Support per-request overrides
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )
        effective_thinking_tokens = (
            kwargs.pop("thinking_tokens") if "thinking_tokens" in kwargs else budgets.get("thinking_tokens")
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=effective_max_tokens,
                thinking_tokens=effective_thinking_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={"reasoning": parsed.data.get("reasoning", "")},
                    raw_provider_fields={"model_id": self.model_id, "response_metadata": result.get("response_metadata", {})},
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
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"parse_error": parsed.error},
                    raw_provider_fields={"model_id": self.model_id, "response_metadata": result.get("response_metadata", {}), "raw_content": content},
                )
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock test generation failed: {e}")
            raise LLMError(
                message=f"Test generation failed: {e}",
                provider="bedrock",
                operation="generate_tests",
                model=self.model_id,
            ) from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Prompts from registry
        system_message, user_content = self._prompts(
            "llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        budgets = self._calc_budgets(
            use_case="code_analysis",
            input_text=code_content,
        )
        # Support per-request overrides
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )
        effective_thinking_tokens = (
            kwargs.pop("thinking_tokens") if "thinking_tokens" in kwargs else budgets.get("thinking_tokens")
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=effective_max_tokens,
                thinking_tokens=effective_thinking_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                metadata = self._unify_metadata(
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    extra={
                        "analysis_type": analysis_type,
                        "summary": parsed.data.get("analysis_summary", ""),
                    },
                    raw_provider_fields={"model_id": self.model_id},
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
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={
                        "analysis_type": analysis_type,
                        "raw_content": content,
                        "parse_error": parsed.error,
                    },
                    raw_provider_fields={"model_id": self.model_id},
                )
                return {
                    "testability_score": 5.0,
                    "complexity_metrics": {},
                    "recommendations": [],
                    "potential_issues": [],
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock code analysis failed: {e}")
            raise LLMError(
                message=f"Code analysis failed: {e}",
                provider="bedrock",
                operation="analyze_code",
                model=self.model_id,
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

        # Use provided system prompt or default from registry
        if system_prompt is None:
            system_message, _ = self._prompts("llm_content_refinement")
        else:
            system_message = system_prompt
        user_content = refinement_instructions

        budgets = self._calc_budgets(
            use_case="refinement",
            input_text=refinement_instructions,
        )
        # Support per-request overrides
        effective_max_tokens = int(
            (kwargs.pop("max_tokens") if "max_tokens" in kwargs else budgets["max_tokens"])  # type: ignore[arg-type]
        )
        effective_thinking_tokens = (
            kwargs.pop("thinking_tokens") if "thinking_tokens" in kwargs else budgets.get("thinking_tokens")
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=effective_max_tokens,
                thinking_tokens=effective_thinking_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                # Use common schema validation and repair
                from .common import normalize_refinement_response, create_repair_prompt
                
                validation_result = normalize_refinement_response(parsed.data)
                
                if not validation_result.is_valid:
                    logger.warning(
                        "Bedrock returned invalid schema: %s. Attempting repair...",
                        validation_result.error
                    )
                    
                    # Attempt single-shot repair with minimal prompt
                    repair_prompt = create_repair_prompt(
                        validation_result.error,
                        ["refined_content", "changes_made", "confidence", "improvement_areas"]
                    )
                    
                    try:
                        repair_result = self._invoke_chat(
                            system_message=system_message,
                            user_content=f"{user_content}\n\n{repair_prompt}",
                            temperature=0.0,  # Deterministic repair
                            **kwargs
                        )
                        
                        repair_content = repair_result.get("content", "")
                        repair_parsed = parse_json_response(repair_content)
                        
                        if repair_parsed.success and repair_parsed.data:
                            repair_validation = normalize_refinement_response(repair_parsed.data)
                            if repair_validation.is_valid:
                                logger.info("Bedrock schema repair successful.")
                                validation_result = repair_validation
                            else:
                                logger.error(f"Bedrock repair failed: {repair_validation.error}")
                        
                    except Exception as repair_e:
                        logger.error(f"Bedrock repair attempt failed: {repair_e}")
                
                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data
                    metadata = self._unify_metadata(
                        provider="bedrock",
                        model=self.model_id,
                        usage=result.get("usage"),
                        parsed_info={
                            "parsed": True,
                            "repaired": validation_result.repaired,
                            "repair_type": validation_result.repair_type,
                        },
                        raw_provider_fields={"model_id": self.model_id},
                    )
                    return {
                        "refined_content": response_data["refined_content"],
                        "changes_made": response_data["changes_made"],
                        "confidence": response_data["confidence"],
                        "improvement_areas": response_data["improvement_areas"],
                        "suspected_prod_bug": response_data.get("suspected_prod_bug"),
                        "metadata": metadata,
                    }
                else:
                    # Schema validation failed even after repair
                    logger.error(f"Bedrock schema validation failed: {validation_result.error}")
                    metadata = self._unify_metadata(
                        provider="bedrock",
                        model=self.model_id,
                        usage=result.get("usage"),
                        parsed_info={"parsed": False},
                        extra={"schema_error": validation_result.error},
                        raw_provider_fields={"model_id": self.model_id},
                    )
                    return {
                        "refined_content": original_content,  # Safe fallback
                        "changes_made": f"Schema validation failed: {validation_result.error}",
                        "confidence": 0.0,
                        "improvement_areas": ["schema_error"],
                        "suspected_prod_bug": None,
                        "metadata": metadata,
                    }
            else:
                # Fallback if JSON parsing fails
                metadata = self._unify_metadata(
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"raw_content": content, "parse_error": parsed.error},
                    raw_provider_fields={"model_id": self.model_id},
                )
                return {
                    "refined_content": content or original_content,
                    "changes_made": "Refinement applied (JSON parse failed)",
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock content refinement failed: {e}")
            raise LLMError(
                message=f"Content refinement failed: {e}",
                provider="bedrock",
                operation="refine_content",
                model=self.model_id,
            ) from e

    def _invoke_chat(
        self, system_message: str, user_content: str, *, max_tokens: int | None = None, thinking_tokens: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Invoke ChatBedrock with system and user messages."""

        # Optional thinking guidance embedded in the system message (no chain-of-thought leakage)
        # Only add thinking guidance if the model actually supports thinking tokens
        system_msg = system_message
        if thinking_tokens and self._supports_thinking_tokens():
            system_msg += (
                f"\n\nGuidance: Use careful internal reasoning up to {int(thinking_tokens)} tokens as needed to plan your answer, "
                "but do not include your internal reasoning in the final output."
            )
        elif thinking_tokens and not self._supports_thinking_tokens():
            # Log when thinking tokens are requested but not supported
            logger.debug(f"Thinking tokens requested but not supported by {self.model_id}")

        # Create LangChain messages
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_content),
        ]

        # Per-request override for max_tokens if provided, but enforce catalog limits
        if max_tokens is not None:
            # Ensure we don't exceed the model's documented limits
            catalog_max = self._get_max_tokens_for_model()
            effective_max_tokens = min(int(max_tokens), catalog_max)
            if effective_max_tokens < max_tokens:
                logger.warning(f"Requested {max_tokens} tokens but model {self.model_id} limit is {catalog_max}, using {effective_max_tokens}")
            kwargs = {**kwargs, "max_tokens": effective_max_tokens}

        try:
            # Invoke the ChatBedrock client
            response = self.client.invoke(messages, **kwargs)

            # Extract content from LangChain AIMessage response
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract usage information if available
            usage_info = {}
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata
                # Try to extract token usage from various possible locations
                if "usage" in metadata:
                    usage_data = metadata["usage"]
                    usage_info = {
                        "input_tokens": usage_data.get("input_tokens", 0),
                        "output_tokens": usage_data.get("output_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                    }
                elif "token_usage" in metadata:
                    usage_data = metadata["token_usage"]
                    usage_info = {
                        "input_tokens": usage_data.get("prompt_tokens", 0),
                        "output_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                    }

            return {
                "content": content,
                "usage": usage_info,
                "model_id": self.model_id,
                "response_metadata": getattr(response, "response_metadata", {}),
            }

        except Exception as e:
            logger.error(f"ChatBedrock invocation error: {e}")
            raise LLMError(
                message=f"ChatBedrock invocation failed: {e}",
                provider="bedrock",
                operation="invoke",
                model=self.model_id,
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
            return self._invoke_chat(
                system_message=system_prompt,
                user_content=user_prompt,
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
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": True},
                    raw_provider_fields={"model_id": self.model_id},
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
                logger.debug(f"Bedrock test plan generation completed in {planning_duration:.2f}s with confidence {response['confidence']}")
                
                return response
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                metadata = self._unify_metadata(
                    provider="bedrock",
                    model=self.model_id,
                    usage=result.get("usage"),
                    parsed_info={"parsed": False},
                    extra={"parse_error": parsed.error},
                    raw_provider_fields={"model_id": self.model_id},
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
            logger.error(f"Bedrock test plan generation failed after {planning_duration:.2f}s: {e}")
            raise LLMError(
                message=f"Test plan generation failed: {e}",
                provider="bedrock",
                operation="generate_test_plan",
                model=self.model_id,
            ) from e
