"""Real OpenAI adapter implementation using the latest v1.106.1 SDK."""

from __future__ import annotations

import logging
from typing import Any, Literal

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
from .token_calculator import TokenCalculator

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """OpenAI adapter specific errors."""

    pass


class OpenAIAdapter(LLMPort):
    """
    Production OpenAI adapter using the latest v1.106.1 SDK.

    Features:
    - Secure credential management via environment variables
    - Proper error handling and retries with exponential backoff
    - Support for latest OpenAI models (GPT-5, GPT-4.1, o4-mini reasoning model)
    - Configurable timeouts and token limits
    - Structured JSON response parsing
    """

    def __init__(
        self,
        model: str = "o4-mini",
        timeout: float = 60.0,
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
            model: OpenAI model name (e.g., "o4-mini", "gpt-5", "gpt-4.1")
            timeout: Request timeout in seconds
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

        # Initialize credential manager
        self.credential_manager = credential_manager or CredentialManager()

        # Initialize prompt registry
        self.prompt_registry = prompt_registry or PromptRegistry()
        
        # Initialize cost tracking
        self.cost_port = cost_port

        # Initialize token calculator
        self.token_calculator = TokenCalculator(provider="openai", model=model)

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Initialize OpenAI client
        self._client: OpenAI | None = None
        self._initialize_client(base_url, **kwargs)

    def _initialize_client(self, base_url: str | None = None, **kwargs: Any) -> None:
        """Initialize the OpenAI client with credentials."""
        try:
            credentials = self.credential_manager.get_provider_credentials("openai")

            client_kwargs = {
                "api_key": credentials["api_key"],
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                **kwargs,
            }

            # Use custom base URL if provided
            if base_url:
                client_kwargs["base_url"] = base_url
            elif credentials.get("base_url"):
                client_kwargs["base_url"] = credentials["base_url"]

            self._client = OpenAI(**client_kwargs)

            logger.info(f"OpenAI client initialized with model: {self.model}")

        except CredentialError as e:
            raise OpenAIError(f"Failed to initialize OpenAI client: {e}") from e
        except Exception as e:
            raise OpenAIError(
                f"Unexpected error initializing OpenAI client: {e}"
            ) from e

    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client, initializing if needed."""
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

        # Calculate thinking tokens only for models that support configurable thinking
        # (OpenAI reasoning models like o4-mini handle reasoning internally)
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
            logger.error(f"OpenAI test generation failed: {e}")
            raise OpenAIError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(code_content)
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="code_analysis", input_length=input_length
        )

        # Calculate thinking tokens only for models that support configurable thinking
        # (OpenAI reasoning models like o4-mini handle reasoning internally)
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
            logger.error(f"OpenAI code analysis failed: {e}")
            raise OpenAIError(f"Code analysis failed: {e}") from e

    def refine_content(
        self, original_content: str, refinement_instructions: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""

        # Calculate optimal max_tokens and thinking_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            original_content + refinement_instructions
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="refinement", input_length=input_length
        )

        # Calculate thinking tokens only for models that support configurable thinking
        # (OpenAI reasoning models like o4-mini handle reasoning internally)
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(original_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="refinement", complexity_level=complexity_level
            )

        # Get prompts from registry
        system_prompt = self.prompt_registry.get_system_prompt(
            prompt_type="llm_content_refinement"
        )

        user_prompt = self.prompt_registry.get_user_prompt(
            prompt_type="llm_content_refinement",
            code_content=original_content,
            refinement_instructions=refinement_instructions,
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
                return {
                    "refined_content": parsed.data.get(
                        "refined_content", original_content
                    ),
                    "changes_made": parsed.data.get("changes_made", "No changes made"),
                    "confidence": parsed.data.get("confidence", 0.5),
                    "metadata": {
                        "model": self.model,
                        "parsed": True,
                        "improvement_areas": parsed.data.get("improvement_areas", []),
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
            logger.error(f"OpenAI content refinement failed: {e}")
            raise OpenAIError(f"Content refinement failed: {e}") from e

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

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        thinking_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a chat completion request to OpenAI."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use provided max_tokens or fallback to instance default
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": tokens_to_use,
            "temperature": self.temperature,
            **kwargs,
        }

        # Add thinking tokens only for models that support configurable thinking (not OpenAI)
        # OpenAI reasoning models (like o4-mini) handle reasoning internally without configurable budgets
        if (
            thinking_tokens is not None
            and self.token_calculator.supports_thinking_mode()
        ):
            # Note: This would be for future non-OpenAI models that support configurable thinking tokens
            # Currently no OpenAI models use this pattern - they use built-in reasoning instead
            logger.debug(
                f"Using thinking tokens: {thinking_tokens} for model {self.model}"
            )
        elif self.token_calculator.is_reasoning_model():
            # Log that we're using a reasoning model with built-in capabilities
            logger.debug(f"Using reasoning model with built-in reasoning: {self.model}")

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

            # Track costs if cost port is available
            if self.cost_port and response.usage:
                try:
                    # Calculate cost based on token usage and model
                    cost = self._calculate_api_cost(response.usage, response.model)
                    
                    cost_data = {
                        "cost": cost,
                        "tokens_used": response.usage.total_tokens,
                        "api_calls": 1,
                        "model": response.model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                    
                    self.cost_port.track_usage(
                        service="openai",
                        operation="chat_completion", 
                        cost_data=cost_data
                    )
                except Exception as e:
                    # Don't fail the request if cost tracking fails
                    logger.warning(f"Cost tracking failed: {e}")

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
        # OpenAI pricing (as of 2024) - costs per 1,000 tokens
        # Prices may change, so this should ideally be configurable
        model_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "o4-mini": {"input": 0.00015, "output": 0.0006},  # Same as gpt-4o-mini
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            # Add more models as needed
        }
        
        # Default pricing if model not found (use gpt-4o-mini rates)
        pricing = model_pricing.get(model, {"input": 0.00015, "output": 0.0006})
        
        # Calculate cost based on token usage
        prompt_cost = (usage.prompt_tokens / 1000) * pricing["input"]
        completion_cost = (usage.completion_tokens / 1000) * pricing["output"]
        total_cost = prompt_cost + completion_cost
        
        logger.debug(f"Cost calculation for {model}: prompt={prompt_cost:.6f}, completion={completion_cost:.6f}, total={total_cost:.6f}")
        
        return total_cost
