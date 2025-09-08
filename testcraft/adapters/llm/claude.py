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
        model: str = "claude-3-7-sonnet",
        timeout: float = 60.0,
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
            timeout: Request timeout in seconds
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
            raise ClaudeError(f"Failed to initialize Anthropic client: {e}") from e
        except Exception as e:
            raise ClaudeError(
                f"Unexpected error initializing Anthropic client: {e}"
            ) from e

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

        system_prompt = f"""You are an expert Python test generator specializing in {test_framework} tests. Your task is to generate comprehensive, production-ready test cases for the provided Python code.

Requirements:
- Use {test_framework} testing framework
- Generate tests covering normal usage, edge cases, and error conditions
- Include appropriate fixtures, mocks, and test data as needed
- Focus on achieving high code coverage and testing all logical paths
- Write clean, readable, and maintainable test code
- Include descriptive test method names and docstrings where helpful

Please return your response as valid JSON in this exact format:
{{
  "tests": "# Your complete test code here",
  "coverage_focus": ["list", "of", "specific", "areas", "to", "test"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of your test strategy and approach"
}}

Generate thorough, professional-quality test code."""

        user_prompt = f"Code to generate tests for:\n\n```python\n{code_content}\n```"

        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"

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

        system_prompt = f"""You are an expert Python code analyst. Perform a {analysis_type} analysis of the provided code to assess its quality, testability, and potential issues.

Your analysis should cover:
- Testability assessment (score from 0-10, where 10 is most testable)
- Code complexity metrics (cyclomatic complexity, nesting depth, function count, etc.)
- Specific recommendations for improving testability and code quality
- Identification of potential issues, code smells, or anti-patterns
- Assessment of dependencies, coupling, and overall architecture

Please return your analysis as valid JSON in this exact format:
{{
  "testability_score": 8.5,
  "complexity_metrics": {{
    "cyclomatic_complexity": 5,
    "nesting_depth": 3,
    "function_count": 10,
    "lines_of_code": 150
  }},
  "recommendations": ["specific suggestion 1", "specific suggestion 2"],
  "potential_issues": ["identified issue 1", "identified issue 2"],
  "analysis_summary": "Brief overall summary of findings and key insights"
}}

Provide actionable, specific recommendations."""

        user_prompt = f"Code to analyze:\n\n```python\n{code_content}\n```"

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
        self, original_content: str, refinement_instructions: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""

        system_prompt = """You are an expert Python developer with deep knowledge of testing best practices, code quality, and software engineering principles. Your task is to refine the provided content according to the specific instructions given.

Focus on:
- Improving code quality, readability, and maintainability
- Fixing any bugs, issues, or potential problems
- Enhancing test coverage and test quality
- Following Python best practices and modern conventions
- Maintaining or improving functionality while enhancing structure
- Ensuring code is production-ready and robust

Please return your refined content as valid JSON in this exact format:
{{
  "refined_content": "# Your improved content here",
  "changes_made": "Detailed summary of all changes and improvements applied",
  "confidence": 0.9,
  "improvement_areas": ["area1", "area2", "area3"]
}}

Provide clear explanations of your improvements."""

        user_prompt = f"""Original content to refine:
```python
{original_content}
```

Specific refinement instructions:
{refinement_instructions}

Please improve the content according to these instructions while maintaining functionality."""

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
            logger.error(f"Claude content refinement failed: {e}")
            raise ClaudeError(f"Content refinement failed: {e}") from e

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
                        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                        "api_calls": 1,
                        "model": response.model,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                    
                    self.cost_port.track_usage(
                        service="anthropic",
                        operation="message_creation", 
                        cost_data=cost_data
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
        
        logger.debug(f"Cost calculation for {model}: input={input_cost:.6f}, output={output_cost:.6f}, total={total_cost:.6f}")
        
        return total_cost
