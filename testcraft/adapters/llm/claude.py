"""Real Anthropic Claude adapter implementation using the latest v0.66.0 SDK."""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic import Anthropic
from anthropic.types import Message

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
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
                system=system_prompt, user_message=user_prompt, **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                return {
                    "tests": parsed.data.get("tests", content),
                    "coverage_focus": parsed.data.get("coverage_focus", []),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": {
                        "model": self.model,
                        "parsed": True,
                        "reasoning": parsed.data.get("reasoning", ""),
                        **result.get("usage", {}),
                    },
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {parsed.error}")
                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": {
                        "model": self.model,
                        "parsed": False,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
                }

        except Exception as e:
            logger.error(f"Claude test generation failed: {e}")
            raise ClaudeError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

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
                system=system_prompt, user_message=user_prompt, **kwargs
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                return {
                    "testability_score": parsed.data.get("testability_score", 5.0),
                    "complexity_metrics": parsed.data.get("complexity_metrics", {}),
                    "recommendations": parsed.data.get("recommendations", []),
                    "potential_issues": parsed.data.get("potential_issues", []),
                    "metadata": {
                        "model": self.model,
                        "analysis_type": analysis_type,
                        "parsed": True,
                        "summary": parsed.data.get("analysis_summary", ""),
                        **result.get("usage", {}),
                    },
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    "testability_score": 5.0,
                    "complexity_metrics": {},
                    "recommendations": [],
                    "potential_issues": [],
                    "metadata": {
                        "model": self.model,
                        "analysis_type": analysis_type,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
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

        # Use pre-rendered instructions built by the caller (RefineAdapter) or fallback to registry
        if system_prompt is None:
            system_prompt = self.prompt_registry.get_system_prompt(
                prompt_type="llm_content_refinement"
            )

        user_prompt = refinement_instructions

        def call() -> dict[str, Any]:
            return self._create_message(
                system=system_prompt, user_message=user_prompt, **kwargs
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
                        "Claude returned invalid schema: %s. Attempting repair...",
                        validation_result.error
                    )
                    
                    # Attempt single-shot repair with minimal prompt
                    if "refined_content" not in parsed.data or self._is_invalid_refined_content(parsed.data.get("refined_content")):
                        repair_prompt = create_repair_prompt(
                            validation_result.error,
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
                            repair_parsed = parse_json_response(repair_content)
                            
                            if repair_parsed.success and repair_parsed.data:
                                repair_validation = normalize_refinement_response(repair_parsed.data)
                                if repair_validation.is_valid:
                                    logger.info("Claude schema repair successful.")
                                    validation_result = repair_validation
                                else:
                                    logger.error(f"Claude repair failed: {repair_validation.error}")
                            
                        except Exception as repair_e:
                            logger.error(f"Claude repair attempt failed: {repair_e}")
                
                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data
                    return {
                        "refined_content": response_data["refined_content"],
                        "changes_made": response_data["changes_made"],
                        "confidence": response_data["confidence"],
                        "improvement_areas": response_data["improvement_areas"],
                        "suspected_prod_bug": response_data.get("suspected_prod_bug"),
                        "metadata": {
                            "model": self.model,
                            "parsed": True,
                            "repaired": validation_result.repaired,
                            "repair_type": validation_result.repair_type,
                            **result.get("usage", {}),
                        },
                    }
                else:
                    # Schema validation failed even after repair
                    logger.error(f"Claude schema validation failed: {validation_result.error}")
                    return {
                        "refined_content": original_content,  # Safe fallback
                        "changes_made": f"Schema validation failed: {validation_result.error}",
                        "confidence": 0.0,
                        "improvement_areas": ["schema_error"],
                        "suspected_prod_bug": None,
                        "metadata": {
                            "model": self.model,
                            "parsed": False,
                            "schema_error": validation_result.error,
                            **result.get("usage", {}),
                        },
                    }
            else:
                # Fallback if JSON parsing fails
                return {
                    "refined_content": content or original_content,
                    "changes_made": "Refinement applied (JSON parse failed)",
                    "confidence": 0.3,
                    "metadata": {
                        "model": self.model,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
                }

        except Exception as e:
            logger.error(f"Claude content refinement failed: {e}")
            raise ClaudeError(f"Content refinement failed: {e}") from e
    
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
        self, system: str, user_message: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a message using Claude's messages API."""

        request_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
            **kwargs,
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
                    cost = self._calculate_api_cost(response.usage, response.model)

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
        """Calculate the API cost based on token usage and model pricing.

        Args:
            usage: Anthropic usage object with token counts
            model: Model name used for the request

        Returns:
            Calculated cost in USD
        """
        # Anthropic pricing (as of 2024) - costs per 1,000 tokens
        # Prices may change, so this should ideally be configurable
        model_pricing = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-7-sonnet": {"input": 0.003, "output": 0.015},  # Alias for 3.5
            # Add more models as needed
        }

        # Default pricing if model not found (use haiku rates as conservative default)
        pricing = model_pricing.get(model, {"input": 0.00025, "output": 0.00125})

        # Calculate cost based on token usage
        input_cost = (usage.input_tokens / 1000) * pricing["input"]
        output_cost = (usage.output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        logger.debug(
            f"Cost calculation for {model}: input={input_cost:.6f}, output={output_cost:.6f}, total={total_cost:.6f}"
        )

        return total_cost
