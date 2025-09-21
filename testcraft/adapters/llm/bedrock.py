"""Real AWS Bedrock adapter implementation using LangChain ChatBedrock."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
from .token_calculator import TokenCalculator
from .pricing import calculate_cost as pricing_calculate_cost

logger = logging.getLogger(__name__)


class BedrockError(Exception):
    """Bedrock adapter specific errors."""

    pass


class BedrockAdapter(LLMPort):
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
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region_name: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        beta: dict[str, Any] | None = None,
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
            prompt_registry: Custom prompt registry (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional ChatBedrock parameters
        """
        self.model_id = model_id
        self.region_name = region_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize credential manager
        self.credential_manager = credential_manager or CredentialManager()

        # Initialize prompt registry
        self.prompt_registry = prompt_registry or PromptRegistry()

        # Initialize cost tracking
        self.cost_port = cost_port
        self.beta = beta or {}

        # Initialize token calculator - map model_id to normalized model name
        model_name = self._map_model_id_to_model(model_id)
        self.token_calculator = TokenCalculator(provider="bedrock", model=model_name)

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Initialize ChatBedrock client
        self._client: ChatBedrock | None = None
        self._initialize_client(**kwargs)

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

    def _map_model_id_to_model(self, model_id: str) -> str:
        """Map Bedrock model ID to normalized model identifier.

        Args:
            model_id: Bedrock model ID

        Returns:
            Normalized model identifier for TokenCalculator
        """
        model_lower = model_id.lower()

        # Map common Bedrock model IDs to standard model names
        if "claude-3-7-sonnet" in model_lower:
            return "anthropic.claude-3-7-sonnet-v1:0"
        elif "claude-sonnet-4" in model_lower:
            return "anthropic.claude-sonnet-4-v1:0"
        elif "claude-opus-4" in model_lower:
            return "anthropic.claude-opus-4-v1:0"
        elif "claude-3-sonnet" in model_lower:
            return "anthropic.claude-3-sonnet-20240229-v1:0"
        elif "claude-3-haiku" in model_lower:
            return "anthropic.claude-3-haiku-20240307-v1:0"
        else:
            # Return the model_id as-is if no mapping found
            return model_id

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

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_generation", input_length=input_length
        )

        # Get prompts from registry
        system_message = self.prompt_registry.get_system_prompt(
            prompt_type="llm_test_generation", test_framework=test_framework
        )

        additional_context = {"context": context} if context else {}
        user_content = self.prompt_registry.get_user_prompt(
            prompt_type="llm_test_generation",
            code_content=code_content,
            additional_context=additional_context,
            test_framework=test_framework,
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={"reasoning": parsed.data.get("reasoning", "")},
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={"parse_error": parsed.error},
                )

                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock test generation failed: {e}")
            raise BedrockError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(code_content)
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="code_analysis", input_length=input_length
        )

        # Get prompts from registry
        system_message = self.prompt_registry.get_system_prompt(
            prompt_type="llm_code_analysis", analysis_type=analysis_type
        )

        user_content = self.prompt_registry.get_user_prompt(
            prompt_type="llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
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
            logger.error(f"Bedrock code analysis failed: {e}")
            raise BedrockError(f"Code analysis failed: {e}") from e

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            original_content + refinement_instructions
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="refinement", input_length=input_length
        )

        # Use pre-rendered instructions built by the caller (RefineAdapter) or fallback to registry
        if system_prompt is None:
            system_message = self.prompt_registry.get_system_prompt(
                prompt_type="llm_content_refinement"
            )
        else:
            system_message = system_prompt

        user_content = refinement_instructions

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                # Use common schema validation and repair
                from .common import create_repair_prompt, normalize_refinement_response

                validation_result = normalize_refinement_response(parsed.data)

                if not validation_result.is_valid:
                    logger.warning(
                        "Bedrock returned invalid schema: %s. Attempting repair...",
                        validation_result.error,
                    )

                    # Attempt single-shot repair with minimal prompt
                    repair_prompt = create_repair_prompt(
                        validation_result.error,
                        [
                            "refined_content",
                            "changes_made",
                            "confidence",
                            "improvement_areas",
                        ],
                    )

                    try:
                        repair_result = self._invoke_chat(
                            system_message=system_message,
                            user_content=f"{user_content}\n\n{repair_prompt}",
                            temperature=0.0,  # Deterministic repair
                            **kwargs,
                        )

                        repair_content = repair_result.get("content", "")
                        repair_parsed = parse_json_response(repair_content)

                        if repair_parsed.success and repair_parsed.data:
                            repair_validation = normalize_refinement_response(
                                repair_parsed.data
                            )
                            if repair_validation.is_valid:
                                logger.info("Bedrock schema repair successful.")
                                validation_result = repair_validation
                            else:
                                logger.error(
                                    f"Bedrock repair failed: {repair_validation.error}"
                                )

                    except Exception as repair_e:
                        logger.error(f"Bedrock repair attempt failed: {repair_e}")

                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="bedrock",
                        model_identifier=self.model_id,
                        usage_data=result.get("usage"),
                        parsed=True,
                        extras={
                            "repaired": validation_result.repaired,
                            "repair_type": validation_result.repair_type,
                        },
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
                    logger.error(
                        f"Bedrock schema validation failed: {validation_result.error}"
                    )

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="bedrock",
                        model_identifier=self.model_id,
                        usage_data=result.get("usage"),
                        parsed=False,
                        extras={"schema_error": validation_result.error},
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
                        "raw_content": content,
                        "parse_error": parsed.error,
                    },
                )

                return {
                    "refined_content": content or original_content,
                    "changes_made": "Refinement applied (JSON parse failed)",
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock content refinement failed: {e}")
            raise BedrockError(f"Content refinement failed: {e}") from e

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a comprehensive test plan for the provided code content."""

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_generation", input_length=input_length
        )

        # Get prompts from registry
        system_message = self.prompt_registry.get_system_prompt(
            prompt_type="llm_test_plan_generation"
        )

        additional_context = {"context": context} if context else {}
        user_content = self.prompt_registry.get_user_prompt(
            prompt_type="llm_test_plan_generation",
            code_content=code_content,
            additional_context=additional_context,
        )

        def call() -> dict[str, Any]:
            return self._invoke_chat(
                system_message=system_message,
                user_content=user_content,
                max_tokens=max_tokens,
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={"reasoning": parsed.data.get("reasoning", "")},
                )

                return {
                    "test_plan": parsed.data.get("test_plan", ""),
                    "test_coverage_areas": parsed.data.get("test_coverage_areas", []),
                    "test_priorities": parsed.data.get("test_priorities", []),
                    "estimated_complexity": parsed.data.get(
                        "estimated_complexity", "moderate"
                    ),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": metadata,
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="bedrock",
                    model_identifier=self.model_id,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={"parse_error": parsed.error},
                )

                return {
                    "test_plan": content or "Could not generate test plan",
                    "test_coverage_areas": [
                        "functions",
                        "edge_cases",
                        "error_handling",
                    ],
                    "test_priorities": ["high", "medium", "low"],
                    "estimated_complexity": "moderate",
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Bedrock test plan generation failed: {e}")
            raise BedrockError(f"Test plan generation failed: {e}") from e

    def _estimate_complexity(
        self, code_content: str
    ) -> Literal["simple", "moderate", "complex"]:
        """Estimate code complexity for token calculation.

        Args:
            code_content: Code to analyze

        Returns:
            Complexity level estimate
        """
        lines = code_content.split("\n")
        line_count = len([line for line in lines if line.strip()])

        # Count potential complexity indicators
        complexity_indicators = [
            "class ",
            "def ",
            "async ",
            "await ",
            "try:",
            "except:",
            "finally:",
            "with ",
            "for ",
            "while ",
            "if ",
            "elif ",
            "lambda",
            "yield",
            "import ",
            "from ",
            "@",
            "raise ",
            "assert ",
        ]

        indicator_count = sum(
            1
            for line in lines
            for indicator in complexity_indicators
            if indicator in line
        )

        # Simple heuristic based on lines and complexity indicators
        if line_count < 50 and indicator_count < 10:
            return "simple"
        elif line_count < 200 and indicator_count < 30:
            return "moderate"
        else:
            return "complex"

    def _invoke_chat(
        self,
        system_message: str,
        user_content: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke ChatBedrock with system and user messages."""

        # Use provided max_tokens or fallback to instance default
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        # Clamp to catalog cap
        try:
            cap = self.token_calculator.limits.max_output
            tokens_to_use = min(int(tokens_to_use or cap), int(cap))
        except Exception:
            pass

        # Create LangChain messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_content),
        ]

        # Update model kwargs with the calculated max_tokens
        updated_kwargs = kwargs.copy()
        if "model_kwargs" not in updated_kwargs:
            updated_kwargs["model_kwargs"] = {}

        # Override max_tokens in model_kwargs
        updated_kwargs["model_kwargs"]["max_tokens"] = tokens_to_use

        try:
            # Invoke the ChatBedrock client
            response = self.client.invoke(messages, **updated_kwargs)

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

            # Track costs if possible
            try:
                if self.cost_port and usage_info:
                    cost = pricing_calculate_cost(usage_info, "bedrock", self.model_id)
                    tokens_total = usage_info.get("total_tokens") or (
                        usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0)
                    )
                    self.cost_port.track_usage(
                        service="bedrock",
                        operation="invoke",
                        cost_data={
                            "cost": cost,
                            "tokens_used": tokens_total,
                            "api_calls": 1,
                            "model": self.model_id,
                        },
                    )
            except Exception:
                pass

            return {
                "content": content,
                "usage": usage_info,
                "model_id": self.model_id,
                "response_metadata": getattr(response, "response_metadata", {}),
            }

        except Exception as e:
            logger.error(f"ChatBedrock invocation error: {e}")
            raise BedrockError(f"ChatBedrock invocation failed: {e}") from e
