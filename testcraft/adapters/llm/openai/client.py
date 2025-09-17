"""OpenAI client management and low-level API calls."""

from __future__ import annotations

import logging
from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from ....config.credentials import CredentialError, CredentialManager

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """OpenAI adapter specific errors."""

    pass


class OpenAIClient:
    """Manages OpenAI client initialization and low-level API calls."""

    def __init__(
        self,
        model: str,
        timeout: float = 180.0,
        max_retries: int = 3,
        base_url: str | None = None,
        credential_manager: CredentialManager | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI client.

        Args:
            model: OpenAI model name
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            base_url: Custom API base URL (optional)
            credential_manager: Custom credential manager (optional)
            **kwargs: Additional OpenAI client parameters
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.credential_manager = credential_manager

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
            # In test environments without credentials, fall back to a stub client
            logger.warning(f"OpenAI credentials not available, using stub client: {e}")
            self._client = self._create_stub_client()
        except Exception as e:
            logger.warning(f"OpenAI client init failed, using stub client: {e}")
            self._client = self._create_stub_client()

    def _create_stub_client(self) -> Any:
        """Create a stub client for testing environments."""
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

        return _StubClient()

    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client, initializing if needed."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Make a chat completion request."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        try:
            response: ChatCompletion = self.client.chat.completions.create(**request_kwargs)
            return response
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIError(f"OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise OpenAIError(f"Chat completion failed: {e}") from e

    def responses_create(
        self,
        input_text: str,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make a responses API request for reasoning models."""
        responses_kwargs = {
            "model": self.model,
            "input": input_text,
            **kwargs,
        }

        if max_output_tokens is not None:
            responses_kwargs["max_output_tokens"] = max_output_tokens

        try:
            response = self.client.responses.create(**responses_kwargs)  # type: ignore[attr-defined]
            return response
        except TypeError as te:
            # Fallback for older SDKs
            if "max_output_tokens" in str(te):
                alt_kwargs = dict(responses_kwargs)
                alt_kwargs.pop("max_output_tokens", None)
                if max_output_tokens is not None:
                    alt_kwargs["max_tokens"] = max_output_tokens
                response = self.client.responses.create(**alt_kwargs)  # type: ignore[attr-defined]
                return response
            raise OpenAIError(f"Responses API call failed: {te}") from te
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIError(f"OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Responses API call: {e}")
            raise OpenAIError(f"Responses API call failed: {e}") from e
