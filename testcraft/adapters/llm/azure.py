"""Real Azure OpenAI adapter implementation using the latest v1.2.0 SDK."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

from ...config.credentials import CredentialError, CredentialManager
from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .common import parse_json_response, with_retries
from .pricing import calculate_cost as pricing_calculate_cost
from .token_calculator import TokenCalculator

logger = logging.getLogger(__name__)


class AzureOpenAIError(Exception):
    """Azure OpenAI adapter specific errors."""

    pass


class AzureOpenAIClientProtocol(Protocol):
    """Protocol defining the interface for Azure OpenAI clients."""

    @property
    def chat(self) -> Any:
        """Get the chat completions interface."""
        ...

    def chat_completions_create(self, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion."""
        ...


class AzureOpenAIStubClient:
    """Stub client that implements the same interface as AzureOpenAI for testing."""

    def __init__(self) -> None:
        self.chat = AzureOpenAIStubChat()

    def chat_completions_create(self, **kwargs: Any) -> ChatCompletion:
        """Create a stub chat completion response."""
        return self.chat.completions.create(**kwargs)


class AzureOpenAIStubChat:
    """Stub chat interface."""

    def __init__(self) -> None:
        self.completions = AzureOpenAIStubCompletions()


class AzureOpenAIStubCompletions:
    """Stub completions interface."""

    def create(self, **kwargs: Any) -> Any:
        """Create a stub completion response."""

        class _Choice:
            def __init__(self, text: str) -> None:
                class _Message:
                    def __init__(self, content: str) -> None:
                        self.content = content

                self.message = _Message(text)
                self.finish_reason = "stop"

        class _Response:
            def __init__(self) -> None:
                self.choices = [
                    _Choice(
                        '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                    )
                ]
                self.usage = None
                self.model = "stub-deployment"

        return _Response()


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
    - Resource management with context manager support
    - Circuit breaker pattern for API failures

    Usage as context manager (recommended):
        with AzureOpenAIAdapter() as adapter:
            result = adapter.generate_tests(code_content)
    """

    def __init__(
        self,
        deployment: str = "gpt-4.1",
        api_version: str = "2024-02-15-preview",
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
        """Initialize Azure OpenAI adapter.

        Args:
            deployment: Azure OpenAI deployment name
            api_version: Azure OpenAI API version
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response (auto-calculated if None)
            temperature: Response randomness (0.0-2.0, lower = more deterministic)
            max_retries: Maximum retry attempts
            credential_manager: Custom credential manager (optional)
            prompt_registry: Custom prompt registry (optional)
            cost_port: Optional cost tracking port (optional)
            **kwargs: Additional Azure OpenAI client parameters
        """
        # Input validation
        self._validate_inputs(
            deployment, api_version, timeout, temperature, max_retries
        )

        self.deployment = deployment
        self.api_version = api_version
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize managers
        self.credential_manager = credential_manager or CredentialManager()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.cost_port = cost_port
        self.beta = beta or {}

        # Initialize token calculator - map deployment to normalized model name
        model_name = self._map_deployment_to_model(deployment)
        self.token_calculator = TokenCalculator(
            provider="azure-openai", model=model_name
        )

        # Set max_tokens (use provided value or calculate automatically)
        self.max_tokens = max_tokens or self.token_calculator.calculate_max_tokens(
            "test_generation"
        )

        # Circuit breaker for API failures
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._circuit_breaker_timeout = 300  # 5 minutes

        # Initialize client state
        self._client: AzureOpenAIClientProtocol | None = None
        self._is_stub_mode = False
        self._cost_tracking_enabled = cost_port is not None
        self._closed = False

        # Initialize Azure OpenAI client
        self._initialize_client(**kwargs)

    def _validate_inputs(
        self,
        deployment: str,
        api_version: str,
        timeout: float,
        temperature: float,
        max_retries: int,
    ) -> None:
        """Validate input parameters."""
        if not deployment or not isinstance(deployment, str):
            raise ValueError("deployment must be a non-empty string")

        if not api_version or not isinstance(api_version, str):
            raise ValueError("api_version must be a non-empty string")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def is_available(self) -> bool:
        """Check if real Azure OpenAI is available (not in stub mode)."""
        return not self._is_stub_mode and self._client is not None

    def _create_stub_client(self) -> AzureOpenAIClientProtocol:
        """Create and return a stub client."""
        logger.warning(
            "Azure OpenAI credentials not available, using stub client. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables "
            "to use real Azure OpenAI."
        )
        return AzureOpenAIStubClient()

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
            self._is_stub_mode = False

            logger.info(
                f"Azure OpenAI client initialized with deployment: {self.deployment}"
            )

        except CredentialError:
            self._client = self._create_stub_client()
            self._is_stub_mode = True

        except Exception as e:
            logger.warning(f"Azure client init failed, using stub client: {e}")
            self._client = self._create_stub_client()
            self._is_stub_mode = True

    def _calculate_api_cost(self, usage_info: dict[str, Any], deployment: str) -> float:
        # Delegate to centralized pricing with normalized model for azure-openai
        model = self._map_deployment_to_model(deployment)
        return pricing_calculate_cost(usage_info, "azure-openai", model)

    def _map_deployment_to_model(self, deployment: str) -> str:
        """Map Azure deployment name to normalized model identifier.

        Args:
            deployment: Azure deployment name

        Returns:
            Normalized model identifier for TokenCalculator
        """
        deployment_lower = deployment.lower()

        # Map common Azure deployments to standard model names
        if "gpt-5" in deployment_lower:
            return "gpt-5"
        elif "gpt-4.1" in deployment_lower:
            return "gpt-4.1"
        elif "o4-mini" in deployment_lower:
            return "o4-mini"
        elif "gpt-4o-mini" in deployment_lower:
            return "gpt-4o-mini"
        elif "gpt-4o" in deployment_lower:
            return "gpt-4o"
        elif "gpt-4" in deployment_lower:
            return "gpt-4"
        elif "gpt-35-turbo" in deployment_lower or "gpt-3.5-turbo" in deployment_lower:
            return "gpt-3.5-turbo"
        else:
            # Default fallback - use the deployment name as-is
            return deployment

    @property
    def client(self) -> AzureOpenAIClientProtocol:
        """Get the Azure OpenAI client, initializing if needed."""
        if self._client is None:
            self._initialize_client()
        if self._client is None:
            raise AzureOpenAIError("Failed to initialize client")
        return self._client

    def __enter__(self) -> AzureOpenAIAdapter:
        """Enter context manager."""
        if self._closed:
            raise AzureOpenAIError("Cannot reuse closed AzureOpenAIAdapter")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        self.close()

    def close(self) -> None:
        """Close the adapter and cleanup resources."""
        if self._closed:
            return

        self._closed = True

        # Close the real client if it exists
        if self._client is not None and not self._is_stub_mode:
            try:
                # Azure OpenAI client doesn't have an explicit close method
                # but we can help garbage collection
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Azure OpenAI client: {e}")

        self._client = None
        logger.info("Azure OpenAI adapter closed")

    def _track_usage_safe(
        self, service: str, operation: str, cost_data: dict[str, Any], **kwargs: Any
    ) -> None:
        """Safely track usage with structured error handling."""
        if not self._cost_tracking_enabled or self._closed:
            return

        try:
            if self.cost_port:
                self.cost_port.track_usage(
                    service=service, operation=operation, cost_data=cost_data, **kwargs
                )
        except Exception as e:
            logger.warning(f"Cost tracking failed: {e}")
            # Don't re-raise - cost tracking shouldn't break the main flow

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should prevent API calls."""
        if self._consecutive_failures >= self._max_consecutive_failures:
            raise AzureOpenAIError(
                f"Circuit breaker open: {self._consecutive_failures} consecutive failures. "
                f"Will retry after {self._circuit_breaker_timeout} seconds."
            )

    def _record_success(self) -> None:
        """Record a successful API call."""
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record a failed API call."""
        self._consecutive_failures += 1
        logger.warning(f"Consecutive failures: {self._consecutive_failures}")

    def _validate_request_inputs(self, **kwargs: Any) -> None:
        """Validate common request inputs."""
        if not kwargs.get("messages"):
            raise ValueError("messages parameter is required")

        if kwargs.get("max_tokens", 0) < 1:
            raise ValueError("max_tokens must be positive")

        if not 0.0 <= kwargs.get("temperature", self.temperature) <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate test cases for the provided code content."""
        if self._closed:
            raise AzureOpenAIError("Cannot use closed AzureOpenAIAdapter")

        if not code_content or not isinstance(code_content, str):
            raise ValueError("code_content must be a non-empty string")

        if context is not None and not isinstance(context, str):
            raise ValueError("context must be a string or None")

        if not test_framework or not isinstance(test_framework, str):
            raise ValueError("test_framework must be a non-empty string")

        # Check circuit breaker
        self._check_circuit_breaker()

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(
            code_content + (context or "")
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="test_generation", input_length=input_length
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
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")
            self._record_success()  # Record successful API call

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                # Use normalized metadata
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "reasoning": parsed.data.get("reasoning", ""),
                    },
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
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "parse_error": parsed.error,
                    },
                )

                return {
                    "tests": content,
                    "coverage_focus": ["functions", "edge_cases", "error_handling"],
                    "confidence": 0.3,
                    "metadata": metadata,
                }

        except Exception as e:
            self._record_failure()  # Record failed API call
            logger.error(f"Azure OpenAI test generation failed: {e}")
            raise AzureOpenAIError(f"Test generation failed: {e}") from e

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code for testability, complexity, and potential issues."""
        if self._closed:
            raise AzureOpenAIError("Cannot use closed AzureOpenAIAdapter")

        if not code_content or not isinstance(code_content, str):
            raise ValueError("code_content must be a non-empty string")

        if not analysis_type or not isinstance(analysis_type, str):
            raise ValueError("analysis_type must be a non-empty string")

        # Check circuit breaker
        self._check_circuit_breaker()

        # Calculate optimal max_tokens for this specific request
        input_length = self.token_calculator.estimate_input_tokens(code_content)
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case="code_analysis", input_length=input_length
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
                **kwargs,
            )

        try:
            result = with_retries(call, retries=self.max_retries)
            content = result.get("content", "")
            self._record_success()  # Record successful API call

            # Parse JSON response
            parsed = parse_json_response(content)

            if parsed.success and parsed.data:
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
            self._record_failure()  # Record failed API call
            logger.error(f"Azure OpenAI code analysis failed: {e}")
            raise AzureOpenAIError(f"Code analysis failed: {e}") from e

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
            system_prompt = self.prompt_registry.get_system_prompt(
                prompt_type="llm_content_refinement"
            )
        user_prompt = refinement_instructions

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
                # Use common schema validation and repair
                from .common import create_repair_prompt, normalize_refinement_response

                validation_result = normalize_refinement_response(parsed.data)

                if not validation_result.is_valid:
                    logger.warning(
                        "Azure OpenAI returned invalid schema: %s. Attempting repair...",
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
                        repair_result = self._chat_completion(
                            system_prompt=system_prompt,
                            user_prompt=f"{user_prompt}\n\n{repair_prompt}",
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
                                logger.info("Azure OpenAI schema repair successful.")
                                validation_result = repair_validation
                            else:
                                logger.error(
                                    f"Azure OpenAI repair failed: {repair_validation.error}"
                                )

                    except Exception as repair_e:
                        logger.error(f"Azure OpenAI repair attempt failed: {repair_e}")

                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="azure-openai",
                        model_identifier=self.deployment,
                        usage_data=result.get("usage"),
                        parsed=True,
                        extras={
                            "deployment": self.deployment,
                            "api_version": self.api_version,
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
                        f"Azure OpenAI schema validation failed: {validation_result.error}"
                    )

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="azure-openai",
                        model_identifier=self.deployment,
                        usage_data=result.get("usage"),
                        parsed=False,
                        extras={
                            "deployment": self.deployment,
                            "api_version": self.api_version,
                            "schema_error": validation_result.error,
                        },
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
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
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
            logger.error(f"Azure OpenAI content refinement failed: {e}")
            raise AzureOpenAIError(f"Content refinement failed: {e}") from e

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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=True,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "reasoning": parsed.data.get("reasoning", ""),
                    },
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
                    provider="azure-openai",
                    model_identifier=self.deployment,
                    usage_data=result.get("usage"),
                    parsed=False,
                    extras={
                        "deployment": self.deployment,
                        "api_version": self.api_version,
                        "parse_error": parsed.error,
                    },
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
            logger.error(f"Azure OpenAI test plan generation failed: {e}")
            raise AzureOpenAIError(f"Test plan generation failed: {e}") from e

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

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a chat completion request to Azure OpenAI."""
        if self._closed:
            raise AzureOpenAIError("Cannot use closed AzureOpenAIAdapter")

        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a non-empty string")

        if not user_prompt or not isinstance(user_prompt, str):
            raise ValueError("user_prompt must be a non-empty string")

        # Validate request inputs
        request_kwargs = {
            "model": self.deployment,  # Use deployment name as model for Azure
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        self._validate_request_inputs(**request_kwargs)

        # Use provided max_tokens or fallback to instance default
        tokens_to_use = request_kwargs["max_tokens"]
        # Clamp to catalog cap
        try:
            cap = self.token_calculator.limits.max_output
            tokens_to_use = min(int(tokens_to_use or cap), int(cap))
            request_kwargs["max_tokens"] = tokens_to_use
        except Exception:
            pass

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

                # Track costs if cost_port is available
                if usage_info:
                    cost = self._calculate_api_cost(usage_info, self.deployment)
                    self._track_usage_safe(
                        service="azure-openai",
                        operation="chat_completion",
                        cost_data={
                            "cost": cost,
                            "tokens_used": usage_info["total_tokens"],
                            "api_calls": 1,
                        },
                        deployment=self.deployment,
                        api_version=self.api_version,
                        prompt_tokens=usage_info["prompt_tokens"],
                        completion_tokens=usage_info["completion_tokens"],
                    )

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
