"""Real Anthropic Claude adapter implementation using the latest v0.66.0 SDK."""

from __future__ import annotations

import logging
from typing import Any, Literal

import anthropic
from anthropic import Anthropic
from anthropic.types import Message

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
from .pricing import calculate_cost as pricing_calculate_cost
from .token_calculator import TokenCalculator

logger = logging.getLogger(__name__)


class ClaudeError(Exception):
    """Claude adapter specific errors."""

    pass


class ClaudeAdapter(LLMPort):
    """
    Production Anthropic Claude adapter using the latest v0.66.0 SDK.

    Features:
    - Secure credential management via environment variables
    - Proper error handling and retries with exponential backoff
    - Support for latest Claude models (3.7 Sonnet, Sonnet 4, Opus 4)
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    - System prompt support for better control
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4",
        timeout: float = 180.0,
        max_tokens: int | None = None,  # Will be calculated automatically
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        beta: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Claude adapter.

        Args:
            model: Claude model name (e.g., "claude-3-7-sonnet", "claude-sonnet-4", "claude-opus-4")
            timeout: Request timeout in seconds (default: 180s for test generation)
            max_tokens: Maximum tokens in response (auto-calculated if None)
            temperature: Response randomness (0.0-1.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            prompt_registry: Custom prompt registry (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional Anthropic client parameters
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize credential manager
        self.credential_manager = credential_manager or CredentialManager()

        # Initialize prompt registry
        self.prompt_registry = prompt_registry or PromptRegistry()

        # Initialize cost tracking
        self.cost_port = cost_port
        self.beta = beta or {}

        # Initialize token calculator
        self.token_calculator = TokenCalculator(provider="anthropic", model=model)

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Initialize Anthropic client
        self._client: Anthropic | None = None
        self._initialize_client(**kwargs)

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
            logger.error(f"Anthropic credentials not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Anthropic client initialization failed: {e}")
            raise

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

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_generation", input_length=input_length
        )

        # Calculate thinking tokens for Claude's configurable thinking mode
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(code_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="test_generation", complexity_level=complexity_level
            )

        # Get prompts from registry
        system_prompt = self.prompt_registry.get_system_prompt(
            prompt_type="llm_test_generation", test_framework=test_framework
        )

        additional_context = {"context": context} if context else {}
        user_prompt = self.prompt_registry.get_user_prompt(
            prompt_type="llm_test_generation",
            code_content=code_content,
            additional_context=additional_context,
            test_framework=test_framework,
        )

        def call() -> dict[str, Any]:
            return self._create_message(
                system=system_prompt,
                user_message=user_prompt,
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
                # Use normalized metadata
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="anthropic",
                    model_identifier=self.model,
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
                    provider="anthropic",
                    model_identifier=self.model,
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
            logger.error(f"Claude test generation failed: {e}")
            raise ClaudeError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(code_content)
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="code_analysis", input_length=input_length
        )

        # Calculate thinking tokens for Claude's configurable thinking mode
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(code_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="code_analysis", complexity_level=complexity_level
            )

        # Get prompts from registry
        system_prompt = self.prompt_registry.get_system_prompt(
            prompt_type="llm_code_analysis", analysis_type=analysis_type
        )

        user_prompt = self.prompt_registry.get_user_prompt(
            prompt_type="llm_code_analysis",
            code_content=code_content,
            analysis_type=analysis_type,
        )

        def call() -> dict[str, Any]:
            return self._create_message(
                system=system_prompt,
                user_message=user_prompt,
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="anthropic",
                    model_identifier=self.model,
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
                    provider="anthropic",
                    model_identifier=self.model,
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
            logger.error(f"Claude code analysis failed: {e}")
            raise ClaudeError(f"Code analysis failed: {e}") from e

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
        input_length = self.token_calculator.estimate_input_tokens(
            original_content + refinement_instructions
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="refinement", input_length=input_length
        )

        # Calculate thinking tokens for Claude's configurable thinking mode
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(original_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="refinement", complexity_level=complexity_level
            )

        # Use pre-rendered instructions built by the caller (RefineAdapter) or fallback to registry
        if system_prompt is None:
            system_prompt = self.prompt_registry.get_system_prompt(
                prompt_type="llm_content_refinement"
            )

        user_prompt = refinement_instructions

        def call() -> dict[str, Any]:
            return self._create_message(
                system=system_prompt,
                user_message=user_prompt,
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
                # Use common schema validation and repair
                from .common import create_repair_prompt, normalize_refinement_response

                validation_result = normalize_refinement_response(parsed.data)

                if not validation_result.is_valid:
                    logger.warning(
                        "Claude returned invalid schema: %s. Attempting repair...",
                        validation_result.error,
                    )

                    # Attempt single-shot repair with minimal prompt
                    if (
                        "refined_content" not in parsed.data
                        or self._is_invalid_refined_content(
                            parsed.data.get("refined_content")
                        )
                    ):
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
                            repair_result = self._create_message(
                                system=system_prompt,
                                user_message=f"{user_prompt}\n\n{repair_prompt}",
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
                                    logger.info("Claude schema repair successful.")
                                    validation_result = repair_validation
                                else:
                                    logger.error(
                                        f"Claude repair failed: {repair_validation.error}"
                                    )

                        except Exception as repair_e:
                            logger.error(f"Claude repair attempt failed: {repair_e}")

                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="anthropic",
                        model_identifier=self.model,
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
                        f"Claude schema validation failed: {validation_result.error}"
                    )

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="anthropic",
                        model_identifier=self.model,
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
                    provider="anthropic",
                    model_identifier=self.model,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={"raw_content": content, "parse_error": parsed.error},
                )

                return {
                    "refined_content": content or original_content,
                    "changes_made": "Refinement applied (JSON parse failed)",
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            logger.error(f"Claude content refinement failed: {e}")
            raise ClaudeError(f"Content refinement failed: {e}") from e

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a comprehensive test plan for the provided code content."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_generation", input_length=input_length
        )

        # Calculate thinking tokens for Claude's configurable thinking mode
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(code_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="test_generation", complexity_level=complexity_level
            )

        # Get prompts from registry
        system_prompt = self.prompt_registry.get_system_prompt(
            prompt_type="llm_test_plan_generation"
        )

        additional_context = {"context": context} if context else {}
        user_prompt = self.prompt_registry.get_user_prompt(
            prompt_type="llm_test_plan_generation",
            code_content=code_content,
            additional_context=additional_context,
        )

        def call() -> dict[str, Any]:
            return self._create_message(
                system=system_prompt,
                user_message=user_prompt,
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="anthropic",
                    model_identifier=self.model,
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
                    provider="anthropic",
                    model_identifier=self.model,
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
            logger.error(f"Claude test plan generation failed: {e}")
            raise ClaudeError(f"Test plan generation failed: {e}") from e

    def _estimate_complexity(
        self, code_content: str
    ) -> Literal["simple", "moderate", "complex"]:
        """Estimate code complexity for thinking token calculation.

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

    def _create_message(
        self,
        system: str,
        user_message: str,
        max_tokens: int | None = None,
        thinking_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a message using Claude's messages API."""

        # Use provided max_tokens or fallback to instance default
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        # Clamp to catalog cap
        try:
            cap = self.token_calculator.limits.max_output
            tokens_to_use = min(int(tokens_to_use or cap), int(cap))
        except (AttributeError, ValueError, TypeError):
            # Fallback to safe default if token calculator fails
            tokens_to_use = 4096

        request_kwargs = {
            "model": self.model,
            "max_tokens": tokens_to_use,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
            **kwargs,
        }

        # Add thinking tokens if supported and provided
        if (
            thinking_tokens is not None
            and self.token_calculator.supports_thinking_mode()
            and bool(self.beta.get("anthropic_enable_extended_thinking", False))
        ):
            # Use the correct parameter structure as per Claude API documentation
            if "thinking" not in kwargs:
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_tokens,
                }

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

            # Track costs if cost port is available
            if self.cost_port and response.usage:
                try:
                    # Calculate cost based on token usage and model
                    cost = pricing_calculate_cost(
                        response.usage, "anthropic", response.model
                    )

                    cost_data = {
                        "cost": cost,
                        "tokens_used": response.usage.input_tokens
                        + response.usage.output_tokens,
                        "api_calls": 1,
                        "model": response.model,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }

                    self.cost_port.track_usage(
                        service="anthropic",
                        operation="message_creation",
                        cost_data=cost_data,
                    )
                except Exception as e:
                    # Don't fail the request if cost tracking fails
                    logger.warning(f"Cost tracking failed: {e}")

            return {
                "content": content,
                "usage": usage_info,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "stop_sequence": getattr(response, "stop_sequence", None),
            }

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise ClaudeError(f"Anthropic API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in message creation: {e}")
            raise ClaudeError(f"Message creation failed: {e}") from e

    def _calculate_api_cost(self, usage: Any, model: str) -> float:
        # Backward-compat method: delegate to centralized pricing
        return pricing_calculate_cost(usage, "anthropic", model)
