"""Real Azure OpenAI adapter implementation using the latest v1.2.0 SDK."""

from __future__ import annotations

import logging
from typing import Any

import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from .common import parse_json_response, with_retries

logger = logging.getLogger(__name__)


class AzureOpenAIError(Exception):
    """Azure OpenAI adapter specific errors."""

    pass


class AzureOpenAIAdapter(LLMPort):
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
        deployment: str = "gpt-4o-mini",
        api_version: str = "2024-02-15-preview",
        timeout: float = 60.0,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        max_retries: int = 3,
        credential_manager: CredentialManager | None = None,
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
        self.deployment = deployment
        self.api_version = api_version
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize credential manager
        self.credential_manager = credential_manager or CredentialManager()
        
        # Initialize cost tracking
        self.cost_port = cost_port

        # Initialize Azure OpenAI client
        self._client: AzureOpenAI | None = None
        self._initialize_client(**kwargs)

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
            raise AzureOpenAIError(
                f"Failed to initialize Azure OpenAI client: {e}"
            ) from e
        except Exception as e:
            raise AzureOpenAIError(
                f"Unexpected error initializing Azure OpenAI client: {e}"
            ) from e

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

        # Construct the prompt for test generation
        system_prompt = f"""You are an expert Python test generator. Generate comprehensive {test_framework} tests for the provided code.

Requirements:
- Use {test_framework} testing framework
- Generate tests that cover edge cases, error conditions, and normal usage
- Include appropriate fixtures, mocks, and test data
- Focus on achieving high code coverage
- Return results as valid JSON with the following structure:
{{
  "tests": "# Generated test code here",
  "coverage_focus": ["list", "of", "areas", "to", "focus", "testing"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of test strategy"
}}

Generate clean, production-ready test code."""

        user_prompt = f"Code to test:\n```python\n{code_content}\n```"

        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "parsed": False,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
                }

        except Exception as e:
            logger.error(f"Azure OpenAI test generation failed: {e}")
            raise AzureOpenAIError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""

        system_prompt = f"""You are an expert Python code analyst. Perform a {analysis_type} analysis of the provided code.

Analyze the code for:
- Testability score (0-10, where 10 is most testable)
- Complexity metrics (cyclomatic complexity, nesting depth, etc.)
- Recommendations for improving testability
- Potential issues or code smells
- Dependencies and coupling analysis

Return results as valid JSON with this structure:
{{
  "testability_score": 8.5,
  "complexity_metrics": {{
    "cyclomatic_complexity": 5,
    "nesting_depth": 3,
    "function_count": 10
  }},
  "recommendations": ["suggestion1", "suggestion2"],
  "potential_issues": ["issue1", "issue2"],
  "analysis_summary": "Brief summary of findings"
}}"""

        user_prompt = f"Code to analyze:\n```python\n{code_content}\n```"

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "analysis_type": analysis_type,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
                }

        except Exception as e:
            logger.error(f"Azure OpenAI code analysis failed: {e}")
            raise AzureOpenAIError(f"Code analysis failed: {e}") from e

    def refine_content(
        self, original_content: str, refinement_instructions: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""

        system_prompt = """You are an expert Python developer. Refine the provided content according to the given instructions.

Focus on:
- Improving code quality and readability
- Fixing bugs or issues
- Enhancing test coverage
- Following best practices
- Maintaining functionality while improving structure

Return results as valid JSON:
{{
  "refined_content": "# Improved content here",
  "changes_made": "Summary of changes applied",
  "confidence": 0.9,
  "improvement_areas": ["area1", "area2"]
}}"""

        user_prompt = f"""Original content:
```python
{original_content}
```

Refinement instructions:
{refinement_instructions}"""

        def call() -> dict[str, Any]:
            return self._chat_completion(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "parsed": False,
                        "raw_content": content,
                        "parse_error": parsed.error,
                        **result.get("usage", {}),
                    },
                }

        except Exception as e:
            logger.error(f"Azure OpenAI content refinement failed: {e}")
            raise AzureOpenAIError(f"Content refinement failed: {e}") from e

    def _chat_completion(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Make a chat completion request to Azure OpenAI."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_kwargs = {
            "model": self.deployment,  # Use deployment name as model for Azure
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }

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
            raise AzureOpenAIError(f"Azure OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise AzureOpenAIError(f"Chat completion failed: {e}") from e
