"""Real OpenAI adapter implementation using the latest v1.106.1 SDK."""

from __future__ import annotations

import json
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
        model: str = "gpt-4.1",
        timeout: float = 180.0,
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
            timeout: Request timeout in seconds (default: 180s for test generation)
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

    # -------------------------------------------------------------
    # Debug logging helpers (verbose mode)
    # -------------------------------------------------------------
    def _debug_log_request(self, endpoint: str, payload: dict[str, Any]) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            # Avoid logging API keys/headers (we do not place them here anyway)
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(payload)
        logger.debug(
            "\n===== LLM REQUEST (%s) =====\n%s\n===== END REQUEST =====",
            endpoint,
            pretty,
        )

    def _debug_log_response(
        self,
        endpoint: str,
        *,
        content: str | None = None,
        usage: dict[str, Any] | None = None,
        raw_obj: Any | None = None,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        # Raw content (as-is)
        if content is not None:
            logger.debug(
                "\n----- LLM RAW CONTENT (%s) -----\n%s\n----- END RAW CONTENT -----",
                endpoint,
                content,
            )
        # Usage
        if usage:
            try:
                usage_pretty = json.dumps(usage, indent=2, ensure_ascii=False)
            except Exception:
                usage_pretty = str(usage)
            logger.debug(
                "\n----- LLM USAGE (%s) -----\n%s\n----- END USAGE -----",
                endpoint,
                usage_pretty,
            )
        # Full raw SDK response if available
        if raw_obj is not None:
            try:
                if hasattr(raw_obj, "model_dump"):
                    raw_pretty = json.dumps(
                        raw_obj.model_dump(), indent=2, ensure_ascii=False
                    )  # type: ignore[attr-defined]
                elif hasattr(raw_obj, "dict"):
                    raw_pretty = json.dumps(
                        raw_obj.dict(), indent=2, ensure_ascii=False
                    )  # type: ignore[call-arg]
                else:
                    raw_pretty = str(raw_obj)
            except Exception:
                raw_pretty = str(raw_obj)
            logger.debug(
                "\n===== LLM RAW RESPONSE (%s) =====\n%s\n===== END RESPONSE =====",
                endpoint,
                raw_pretty,
            )

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
            # In test environments without credentials, fall back to a stub client
            logger.warning(f"OpenAI credentials not available, using stub client: {e}")

            class _StubChatCompletions:
                def create(self, **_kwargs):
                    class _Choice:
                        def __init__(self, text: str) -> None:
                            class _Msg:
                                def __init__(self, content: str) -> None:
                                    self.content = content

                            self.message = _Msg(text)
                            self.finish_reason = "stop"

                    class _Usage:
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0

                    class _Resp:
                        def __init__(self) -> None:
                            self.choices = [
                                _Choice(
                                    '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                                )
                            ]
                            self.usage = _Usage()
                            self.model = "stub-model"

                    return _Resp()

            class _StubResponses:
                def create(self, **_kwargs):
                    class _Usage:
                        input_tokens = 0
                        output_tokens = 0
                        total_tokens = 0

                    class _Resp:
                        output_text = '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                        usage = _Usage()
                        model = "stub-model"

                    return _Resp()

            class _StubClient:
                def __init__(self) -> None:
                    self.chat = type(
                        "_Chat", (), {"completions": _StubChatCompletions()}
                    )()
                    self.responses = _StubResponses()

            self._client = _StubClient()  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"OpenAI client init failed, using stub client: {e}")

            class _StubChatCompletions:
                def create(self, **_kwargs):
                    class _Choice:
                        def __init__(self, text: str) -> None:
                            class _Msg:
                                def __init__(self, content: str) -> None:
                                    self.content = content

                            self.message = _Msg(text)
                            self.finish_reason = "stop"

                    class _Usage:
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0

                    class _Resp:
                        def __init__(self) -> None:
                            self.choices = [
                                _Choice(
                                    '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                                )
                            ]
                            self.usage = _Usage()
                            self.model = "stub-model"

                    return _Resp()

            class _StubResponses:
                def create(self, **_kwargs):
                    class _Usage:
                        input_tokens = 0
                        output_tokens = 0
                        total_tokens = 0

                    class _Resp:
                        output_text = '{"tests": "# stub", "coverage_focus": [], "confidence": 0.0}'
                        usage = _Usage()
                        model = "stub-model"

                    return _Resp()

            class _StubClient:
                def __init__(self) -> None:
                    self.chat = type(
                        "_Chat", (), {"completions": _StubChatCompletions()}
                    )()
                    self.responses = _StubResponses()

            self._client = _StubClient()  # type: ignore[assignment]

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
                # Use normalized metadata
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="openai",
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
                    provider="openai",
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
                from .common import normalize_metadata

                metadata = normalize_metadata(
                    provider="openai",
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
                    provider="openai",
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
            logger.error(f"OpenAI code analysis failed: {e}")
            raise OpenAIError(f"Code analysis failed: {e}") from e

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

        # Calculate thinking tokens only for models that support configurable thinking
        # (OpenAI reasoning models like o4-mini handle reasoning internally)
        thinking_tokens = None
        if self.token_calculator.supports_thinking_mode():
            complexity_level = self._estimate_complexity(original_content)
            thinking_tokens = self.token_calculator.calculate_thinking_tokens(
                use_case="refinement", complexity_level=complexity_level
            )

        # Use pre-rendered prompts from the caller (if provided). Otherwise fall back to registry.
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
                        "OpenAI returned invalid schema: %s. Attempting repair...",
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

                        def repair_call() -> dict[str, Any]:
                            return self._chat_completion(
                                system_prompt=system_prompt,
                                user_prompt=f"{user_prompt}\n\n{repair_prompt}",
                                max_tokens=max_tokens,
                                thinking_tokens=thinking_tokens,
                                temperature=0.0,  # Deterministic repair
                                **kwargs,
                            )

                        try:
                            repair_result = with_retries(repair_call, retries=1)
                            repair_content = repair_result.get("content", "")
                            repair_parsed = parse_json_response(repair_content)

                            if repair_parsed.success and repair_parsed.data:
                                repair_validation = normalize_refinement_response(
                                    repair_parsed.data
                                )
                                if repair_validation.is_valid:
                                    logger.info("OpenAI schema repair successful.")
                                    validation_result = repair_validation
                                else:
                                    logger.error(
                                        f"OpenAI repair failed: {repair_validation.error}"
                                    )

                        except Exception as repair_e:
                            logger.error(f"OpenAI repair attempt failed: {repair_e}")

                # Return consistent response structure
                if validation_result.is_valid and validation_result.data:
                    response_data = validation_result.data

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="openai",
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
                        f"OpenAI schema validation failed: {validation_result.error}"
                    )

                    from .common import normalize_metadata

                    metadata = normalize_metadata(
                        provider="openai",
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
                    provider="openai",
                    model_identifier=self.model,
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
            logger.error(f"OpenAI content refinement failed: {e}")
            raise OpenAIError(f"Content refinement failed: {e}") from e

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
                    provider="openai",
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
                    provider="openai",
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
            logger.error(f"OpenAI test plan generation failed: {e}")
            raise OpenAIError(f"Test plan generation failed: {e}") from e

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

    def _requires_completion_tokens_param(self) -> bool:
        """Check if the model requires max_completion_tokens instead of max_tokens.

        OpenAI's newer models (o4-mini, o3, o4) require max_completion_tokens
        instead of the legacy max_tokens parameter.

        Returns:
            True if model requires max_completion_tokens, False for max_tokens
        """
        # Models that require max_completion_tokens instead of max_tokens
        completion_token_models = [
            "o4-mini",
            "o3",
            "o4",
            # Add other models as OpenAI updates their API requirements
        ]

        return any(model_name in self.model for model_name in completion_token_models)

    def _is_o_series_reasoning_model(self) -> bool:
        """Detect OpenAI o-series reasoning models that should use Responses API.

        Returns:
            True if the model is an o-series (o4-mini, o3, o4), else False.
        """
        o_series = ["o4-mini", "o3", "o4"]
        return any(name in self.model for name in o_series)

    def _supports_temperature_adjustment(self) -> bool:
        """Check if the model supports custom temperature values.

        OpenAI's reasoning models (o4-mini, o3, o4) only support the default
        temperature of 1.0 and don't allow custom temperature values.

        Returns:
            True if model supports temperature adjustment, False if only default
        """
        # Reasoning models that only support default temperature (1.0)
        reasoning_models = [
            "o4-mini",
            "o3",
            "o4",
            # Add other reasoning models as they're released
        ]

        return not any(model_name in self.model for model_name in reasoning_models)

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
        # Use provided max_tokens or fallback to instance default
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens

        # Branch: Use Responses API for o-series reasoning models
        if self._is_o_series_reasoning_model():
            try:
                # Build a safe combined input. Responses API supports structured inputs,
                # but combining here keeps compatibility across SDK versions.
                combined_input = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

                # Prepare request kwargs for Responses API
                responses_kwargs = {
                    "model": self.model,
                    "input": combined_input,
                    "max_output_tokens": tokens_to_use,
                    **kwargs,
                }

                # Temperature is fixed for o-series; don't set it.

                # Call Responses API with fallback for SDKs that don't support
                # certain parameter names
                # Verbose request log
                self._debug_log_request(
                    "responses.create",
                    {
                        "model": self.model,
                        "input": combined_input,
                        "max_output_tokens": responses_kwargs.get("max_output_tokens"),
                        **{
                            k: v
                            for k, v in responses_kwargs.items()
                            if k not in {"model", "input", "max_output_tokens"}
                        },
                    },
                )

                try:
                    response = self.client.responses.create(**responses_kwargs)  # type: ignore[attr-defined]
                except TypeError as te:
                    err_msg = str(te)
                    # Fallback: older SDKs may expect 'max_tokens' instead of 'max_output_tokens'
                    if "max_output_tokens" in err_msg:
                        try:
                            alt_kwargs = dict(responses_kwargs)
                            alt_kwargs.pop("max_output_tokens", None)
                            alt_kwargs["max_tokens"] = tokens_to_use
                            # Verbose request log for fallback
                            self._debug_log_request(
                                "responses.create (fallback)",
                                {
                                    "model": self.model,
                                    "input": combined_input,
                                    "max_tokens": tokens_to_use,
                                    **{
                                        k: v
                                        for k, v in alt_kwargs.items()
                                        if k not in {"model", "input", "max_tokens"}
                                    },
                                },
                            )
                            response = self.client.responses.create(**alt_kwargs)  # type: ignore[attr-defined]
                        except Exception as e2:
                            raise OpenAIError(
                                f"Responses API call failed (fallback): {e2}"
                            ) from e2
                    else:
                        raise OpenAIError(f"Responses API call failed: {te}") from te

                # Extract content and usage
                content = ""
                try:
                    content = getattr(response, "output_text", "") or ""
                except Exception:
                    content = ""
                if not content:
                    # Fallback: try to extract text from response.output blocks
                    try:
                        output = getattr(response, "output", None)
                        if output and isinstance(output, list):
                            # Find first text block
                            for block in output:
                                # block may contain {"content": [{"type":"output_text", "text":"..."}]} or similar
                                items = None
                                if isinstance(block, dict):
                                    items = block.get("content") or block.get("items")
                                if items and isinstance(items, list):
                                    for item in items:
                                        text = None
                                        if isinstance(item, dict):
                                            text = item.get("text") or item.get("value")
                                        if text:
                                            content = str(text)
                                            break
                                if content:
                                    break
                    except Exception:
                        pass

                usage_info = {}
                try:
                    if getattr(response, "usage", None):
                        usage = response.usage
                        # usage may expose input_tokens/output_tokens/total_tokens
                        prompt_tokens = getattr(usage, "input_tokens", None)
                        completion_tokens = getattr(usage, "output_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                        usage_info = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                            if total_tokens is not None
                            else ((prompt_tokens or 0) + (completion_tokens or 0)),
                        }
                except Exception:
                    usage_info = {}

                # Track costs if cost port is available
                try:
                    if self.cost_port and getattr(response, "usage", None):
                        usage = response.usage
                        # Only proceed if usage exposes token fields we can read
                        if any(
                            hasattr(usage, attr)
                            for attr in (
                                "input_tokens",
                                "output_tokens",
                                "total_tokens",
                            )
                        ) or isinstance(usage, dict):
                            cost = self._calculate_api_cost(
                                usage, getattr(response, "model", self.model)
                            )
                            tokens_total = getattr(usage, "total_tokens", None) or (
                                getattr(usage, "input_tokens", 0)
                                + getattr(usage, "output_tokens", 0)
                            )
                            cost_data = {
                                "cost": cost,
                                "tokens_used": tokens_total,
                                "api_calls": 1,
                                "model": getattr(response, "model", self.model),
                                "prompt_tokens": getattr(usage, "input_tokens", None),
                                "completion_tokens": getattr(
                                    usage, "output_tokens", None
                                ),
                            }
                            self.cost_port.track_usage(
                                service="openai",
                                operation="responses",
                                cost_data=cost_data,
                            )
                except Exception as e:
                    logger.debug(f"Cost tracking skipped due to usage format: {e}")

                # Verbose response log
                self._debug_log_response(
                    "responses.create",
                    content=content,
                    usage=usage_info,
                    raw_obj=response,
                )

                return {
                    "content": content,
                    "usage": usage_info,
                    "model": getattr(response, "model", self.model),
                    "finish_reason": None,
                }

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise OpenAIError(f"OpenAI API error: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error in Responses API call: {e}")
                raise OpenAIError(f"Responses API call failed: {e}") from e

        # Default path: Chat Completions for non o-series models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        # Enforce JSON output when supported (best-effort)
        request_kwargs["response_format"] = {"type": "json_object"}

        # Add temperature only if the model supports custom temperature values
        if self._supports_temperature_adjustment():
            request_kwargs["temperature"] = self.temperature
        else:
            logger.debug(
                f"Skipping temperature parameter for reasoning model {self.model} (uses default temperature)"
            )

        # Use max_completion_tokens for newer OpenAI models that require it
        if self._requires_completion_tokens_param():
            request_kwargs["max_completion_tokens"] = tokens_to_use
        else:
            request_kwargs["max_tokens"] = tokens_to_use

        # Thinking tokens are not applicable for OpenAI models (built-in reasoning)

        try:
            # Verbose request log
            self._debug_log_request(
                "chat.completions.create",
                {
                    "model": self.model,
                    "messages": messages,
                    **{
                        k: v
                        for k, v in request_kwargs.items()
                        if k not in {"model", "messages"}
                    },
                },
            )
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    **request_kwargs
                )
            except TypeError as te:
                err_msg = str(te)
                # Fallback: remove response_format if unsupported
                if "response_format" in err_msg and "unexpected keyword" in err_msg:
                    req2 = dict(request_kwargs)
                    req2.pop("response_format", None)
                    response = self.client.chat.completions.create(**req2)
                # Fallback: older SDKs may not support max_completion_tokens
                elif (
                    "max_completion_tokens" in err_msg
                    and "unexpected keyword" in err_msg
                ):
                    req2 = dict(request_kwargs)
                    tokens_param = req2.pop("max_completion_tokens", None)
                    if tokens_param is not None:
                        req2["max_tokens"] = tokens_param
                    response = self.client.chat.completions.create(**req2)
                else:
                    raise

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
                        cost_data=cost_data,
                    )
                except Exception as e:
                    logger.warning(f"Cost tracking failed: {e}")

            # Verbose response log
            self._debug_log_response(
                "chat.completions.create",
                content=content,
                usage=usage_info,
                raw_obj=response,
            )

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

        # Extract token counts across SDK variants
        try:
            # Support object-style usage with prompt/completion or input/output tokens
            prompt_tokens = (
                getattr(usage, "prompt_tokens", None)
                if hasattr(usage, "prompt_tokens")
                else None
            )
            if prompt_tokens is None:
                prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = (
                getattr(usage, "completion_tokens", None)
                if hasattr(usage, "completion_tokens")
                else None
            )
            if completion_tokens is None:
                completion_tokens = getattr(usage, "output_tokens", None)
        except Exception:
            prompt_tokens = None
            completion_tokens = None

        # Dict-style fallback
        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        if completion_tokens is None and isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens") or usage.get(
                "output_tokens"
            )

        # Final safety defaults
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)

        # Calculate cost based on token usage
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        total_cost = prompt_cost + completion_cost

        logger.debug(
            f"Cost calculation for {model}: prompt={prompt_cost:.6f}, completion={completion_cost:.6f}, total={total_cost:.6f}"
        )

        return total_cost
